"""Fail-to-pass validation of PB-SWE-Live candidates (runs on the Spark).

For each candidate from harvest_swe_live.py:
1. Cut the merged PR into a gold patch (source) and a test patch (tests) from a local mirror.
2. Build (once per repo) a Docker image with the repo's dependencies installed editable at
   /workspace, the same mount point ContainerSandbox uses, so task workspaces mounted there are live.
3. In a network-less container run the PR's test files at base+tests and at base+tests+gold.
   FAIL_TO_PASS = tests that pass only with the gold patch; PASS_TO_PASS = tests passing in both.
4. Keep a task only if FAIL_TO_PASS is non-empty and no test that passed before breaks after.

Usage (Spark):
    python scripts/bench/validate_swe_live.py --candidates data/swe_live/candidates.jsonl \
        --out data/swe_live/validated.jsonl --workers 4
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

CACHE = Path.home() / ".cache" / "swe_live"
SKIP_REPOS = {"pandas-dev/pandas", "scikit-learn/scikit-learn", "matplotlib/matplotlib"}  # compiled builds
IGNORED_PREFIXES = ("docs/", "doc/", "changelog.d/", "changes/", "news/", "newsfragments/", ".github/")
IGNORED_NAMES = re.compile(r"^(changes|changelog|history|news|release-notes)(\.\w+)?$", re.I)
DEP_FILES = re.compile(r"(^|/)(pyproject\.toml|setup\.py|setup\.cfg|requirements[^/]*\.txt|tox\.ini|noxfile\.py)$")
TEST_LINE = re.compile(r"^(PASSED|FAILED|ERROR|XPASS|XFAIL|SKIPPED)\s+(\S+::\S+)")
PY = "3.12"

DEPS_SCRIPT = r"""
import configparser, pathlib, re, sys, tomllib
root = pathlib.Path('.')
reqs, extras = [], []
try:
    data = tomllib.loads((root / 'pyproject.toml').read_text())
except Exception:
    data = {}
groups = data.get('dependency-groups', {})
def resolve(name, seen=()):
    out = []
    for item in groups.get(name, []):
        if isinstance(item, dict) and 'include-group' in item and item['include-group'] not in seen:
            out += resolve(item['include-group'], seen + (name,))
        elif isinstance(item, str):
            out.append(item)
    return out
for g in groups:
    if re.search(r'test|dev', g, re.I):
        reqs += resolve(g)
opt = data.get('project', {}).get('optional-dependencies', {})
extras = [k for k in opt if re.search(r'test|dev', k, re.I)]
for f in list(root.glob('requirements*/*test*.txt')) + list(root.glob('*test*requirements*.txt')) +          list(root.glob('requirements*test*.txt')) + list(root.glob('requirements/dev*.txt')) + list(root.glob('tests/requirements*.txt')):
    for line in f.read_text().splitlines():
        line = line.split('#')[0].strip()
        if line and not line.startswith('-'):
            reqs.append(line)
if (root / 'tox.ini').exists():
    cp = configparser.ConfigParser(interpolation=None)
    try:
        cp.read(root / 'tox.ini')
        for sec in cp.sections():
            if sec.startswith('testenv') and cp.has_option(sec, 'deps'):
                for line in cp.get(sec, 'deps').splitlines():
                    line = line.split('#')[0].strip()
                    if line and not line.startswith(('-', '{', '.')) and ':' not in line.split('[')[0]:
                        reqs.append(line)
    except Exception:
        pass
pathlib.Path('/tmp/extra_reqs.txt').write_text(chr(10).join(dict.fromkeys(reqs)))
pathlib.Path('/tmp/extras.txt').write_text(','.join(extras))
"""

DOCKERFILE = """FROM python:{py}-slim
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential && rm -rf /var/lib/apt/lists/*
ENV SETUPTOOLS_SCM_PRETEND_VERSION=99.0.0 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY repo /workspace
COPY deps.py /tmp/deps.py
WORKDIR /workspace
RUN pip install --no-cache-dir -q -U pip setuptools wheel && pip install --no-cache-dir -q pytest pytest-xdist pytest-timeout hypothesis typing_extensions  && python /tmp/deps.py && cat /tmp/extra_reqs.txt  && (E=$(cat /tmp/extras.txt); if [ -n "$E" ]; then pip install --no-cache-dir -q -e ".[$E]" || pip install --no-cache-dir -q -e .; else pip install --no-cache-dir -q -e .; fi)  && (pip install --no-cache-dir -q -r /tmp/extra_reqs.txt || while read -r r; do [ -n "$r" ] && pip install --no-cache-dir -q "$r" || true; done < /tmp/extra_reqs.txt)
"""

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def repo_lock(repo: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(repo, threading.Lock())


def sh(cmd: list[str], cwd: Path | None = None, inp: str | None = None, timeout: float = 1800) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, input=inp, capture_output=True, text=True, timeout=timeout,
                          encoding="utf-8", errors="replace")


def slug(repo: str) -> str:
    return repo.replace("/", "__").lower()


def mirror(repo: str) -> Path:
    path = CACHE / "mirrors" / f"{slug(repo)}.git"
    if not path.is_dir():
        path.parent.mkdir(parents=True, exist_ok=True)
        # Plain bare clone: branches and tags (--mirror's refs/pull/* would be huge; blobless clones failed
        # to serve blobs to `git diff` here).
        r = sh(["git", "clone", "--bare", f"https://github.com/{repo}.git", str(path)],
               timeout=3600)
        if r.returncode:
            raise RuntimeError(f"mirror failed: {r.stderr[-400:]}")
    return path


def is_test_path(path: str) -> bool:
    p = path.lower()
    name = p.rsplit("/", 1)[-1]
    return (p.startswith(("tests/", "test/", "testing/")) or "/tests/" in p or "/test/" in p
            or "/testing/" in p or name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py")


def split_patch(mirror_dir: Path, base: str, merge: str) -> tuple[str, str, list[str], list[str]]:
    names = sh(["git", "diff", "--name-only", base, merge], cwd=mirror_dir).stdout.split()
    src, tests = [], []
    for n in names:
        if is_test_path(n):
            tests.append(n)
        elif n.startswith(IGNORED_PREFIXES) or IGNORED_NAMES.match(n.rsplit("/", 1)[-1]):
            continue
        else:
            src.append(n)
    gold = sh(["git", "diff", base, merge, "--", *src], cwd=mirror_dir).stdout if src else ""
    test = sh(["git", "diff", base, merge, "--", *tests], cwd=mirror_dir).stdout if tests else ""
    return gold, test, src, tests


def checkout(mirror_dir: Path, commit: str, dest: Path) -> None:
    r = sh(["git", "clone", "-q", "--shared", str(mirror_dir), str(dest)])
    if r.returncode:
        raise RuntimeError(r.stderr[-400:])
    r = sh(["git", "checkout", "-q", "-f", commit], cwd=dest)
    if r.returncode:
        raise RuntimeError(r.stderr[-400:])


def ensure_image(repo: str, base: str) -> str:
    # One environment per repo: task code is mounted over the editable install at /workspace, so only the
    # dependency set comes from this base commit; validation proves the gold patch passes in it.
    tag = f"pbswe-{slug(repo).replace('__', '-')}:env"
    with repo_lock(repo):
        if sh(["docker", "image", "inspect", tag]).returncode == 0:
            return tag
        with tempfile.TemporaryDirectory() as tmp:
            ctx = Path(tmp)
            checkout(mirror(repo), base, ctx / "repo")
            shutil.rmtree(ctx / "repo" / ".git", ignore_errors=True)
            (ctx / "Dockerfile").write_text(DOCKERFILE.format(py=PY), encoding="utf-8")
            (ctx / "deps.py").write_text(DEPS_SCRIPT, encoding="utf-8")
            r = sh(["docker", "build", "-q", "-t", tag, str(ctx)], timeout=3600)
            if r.returncode:
                raise RuntimeError(f"image build failed: {r.stderr[-800:]}")
    return tag


def run_tests(image: str, ws: Path, test_files: list[str], timeout: float = 900) -> tuple[dict[str, str], str]:
    cmd = ("python -m pytest -p no:cacheprovider -rA -q --no-header -o addopts='' --timeout 300 "
           + " ".join(test_files))
    docker = ["docker", "run", "--rm", "--network", "none", "--cap-drop", "ALL",
              "--security-opt", "no-new-privileges:true", "--memory", "6g", "--cpus", "4",
              "-v", f"{ws}:/workspace:rw", "-w", "/workspace", "-e", "PYTHONDONTWRITEBYTECODE=1",
              image, "bash", "-lc", cmd]
    try:
        r = sh(docker, timeout=timeout)
        out = r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return {}, "TIMEOUT"
    status: dict[str, str] = {}
    for line in out.splitlines():
        m = TEST_LINE.match(line.strip())
        if m:
            status[m.group(2)] = m.group(1)
    return status, out[-3000:]


def apply(ws: Path, patch: str) -> bool:
    if not patch.strip():
        return True
    return sh(["git", "apply", "--whitespace=nowarn", "-"], cwd=ws, inp=patch).returncode == 0


def validate(cand: dict) -> dict:
    repo, base, merge = cand["repo"], cand["base_commit"], cand["merge_commit"]
    rec = {"instance_id": f"{slug(repo)}-{cand['pr_number']}", "repo": repo, "pr_number": cand["pr_number"]}
    try:
        mdir = mirror(repo)
        if sh(["git", "cat-file", "-e", f"{merge}^{{commit}}"], cwd=mdir).returncode:
            sh(["git", "fetch", "-q", "origin", merge], cwd=mdir, timeout=3600)
        gold, test, src, tests = split_patch(mdir, base, merge)
        py_tests = [t for t in tests if re.search(r"(^|/)(test_[^/]*|[^/]*_test)\.py$", t)]
        if not gold or not py_tests:
            return {**rec, "status": "skip", "reason": "no gold or no python tests"}
        if any(DEP_FILES.search(s) for s in src):
            return {**rec, "status": "skip", "reason": "dependency files changed"}
        image = ensure_image(repo, base)
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "ws"
            checkout(mdir, base, ws)
            if not apply(ws, test):
                return {**rec, "status": "skip", "reason": "test patch does not apply"}
            existing = [t for t in py_tests if (ws / t).is_file()]
            before, before_tail = run_tests(image, ws, existing)
            if not apply(ws, gold):
                return {**rec, "status": "skip", "reason": "gold patch does not apply"}
            after, after_tail = run_tests(image, ws, existing)
        passed_after = {t for t, s in after.items() if s == "PASSED"}
        passed_before = {t for t, s in before.items() if s == "PASSED"}
        f2p = sorted(passed_after - passed_before)
        p2p = sorted(passed_after & passed_before)
        broke = sorted(passed_before - passed_after)
        if not after:
            return {**rec, "status": "fail", "reason": "no test results after gold", "log": after_tail}
        if not f2p or broke:
            return {**rec, "status": "fail", "reason": f"f2p={len(f2p)} broke={len(broke)}",
                    "log_before": before_tail[-800:]}
        issue = cand["issues"][0]
        statement = f"{issue['title']}\n\n{issue['body']}".strip()
        return {**rec, "status": "ok", "image": image, "base_commit": base, "merge_commit": merge,
                "pr_url": cand["pr_url"], "issue_url": issue["url"], "merged_at": cand["merged_at"],
                "problem_statement": statement, "gold_patch": gold, "test_patch": test,
                "test_files": existing, "FAIL_TO_PASS": f2p, "PASS_TO_PASS": p2p,
                "src_files": src}
    except Exception as exc:  # recorded, never silently dropped
        return {**rec, "status": "error", "reason": str(exc)[-800:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/swe_live/candidates.jsonl")
    ap.add_argument("--out", default="data/swe_live/validated.jsonl")
    ap.add_argument("--log", default="data/swe_live/validation_log.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--per-repo", type=int, default=12, help="max candidates tried per repo")
    ap.add_argument("--min-issue-chars", type=int, default=120)
    args = ap.parse_args()

    cands, per = [], {}
    for line in Path(args.candidates).read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        body = (c["issues"][0].get("body") or "")
        if c["repo"] in SKIP_REPOS or len(body) < args.min_issue_chars:
            continue
        if per.get(c["repo"], 0) >= args.per_repo:
            continue
        per[c["repo"]] = per.get(c["repo"], 0) + 1
        cands.append(c)
    print(f"validating {len(cands)} candidates from {len(per)} repos", flush=True)

    # Mirror sequentially first so parallel workers never race on a clone.
    for repo in per:
        try:
            mirror(repo)
        except Exception as exc:
            print(f"mirror {repo}: {exc}", flush=True)

    done = set()
    log_path = Path(args.log)
    if log_path.is_file():
        done = {json.loads(l)["instance_id"] for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [c for c in cands if f"{slug(c['repo'])}-{c['pr_number']}" not in done]
    with cf.ThreadPoolExecutor(args.workers) as pool, log_path.open("a", encoding="utf-8") as log_fh:
        for res in pool.map(validate, todo):
            log_fh.write(json.dumps(res, ensure_ascii=False) + "\n")
            log_fh.flush()
            print(res["instance_id"], res["status"], res.get("reason", ""), flush=True)

    ok = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    ok = [r for r in ok if r["status"] == "ok"]
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ok), encoding="utf-8")
    print(f"validated tasks: {len(ok)} -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
