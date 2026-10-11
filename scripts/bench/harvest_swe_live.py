"""Harvest PB-SWE-Live candidate tasks from recently merged bug-fix PRs.

A candidate is a merged PR that closes at least one issue and changes both
source files and test files. Only metadata is collected here; patches are cut
and fail-to-pass validated on the Spark by validate_swe_live.py.

The problem statement is the linked issue text only (never the PR body, which
usually describes the fix).

Usage:
    python scripts/bench/harvest_swe_live.py --since 2026-08-15 --out data/swe_live/candidates.jsonl
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_REPOS = [
    "pallets/click", "pallets/flask", "pallets/jinja", "pallets/werkzeug", "pallets/itsdangerous",
    "psf/requests", "psf/black", "encode/httpx", "encode/starlette", "urllib3/urllib3",
    "python-attrs/attrs", "python-attrs/cattrs", "marshmallow-code/marshmallow",
    "more-itertools/more-itertools", "pypa/packaging", "pypa/pip", "pypa/setuptools",
    "pytest-dev/pytest", "pytest-dev/pluggy", "pylint-dev/astroid", "pylint-dev/pylint",
    "PyCQA/isort", "PyCQA/flake8", "Textualize/rich", "Textualize/textual", "fastapi/typer",
    "sqlfluff/sqlfluff", "networkx/networkx", "sphinx-doc/sphinx", "sympy/sympy",
    "dateutil/dateutil", "arrow-py/arrow", "python-jsonschema/jsonschema", "tox-dev/tox",
    "pydantic/pydantic", "python/mypy", "astral-sh/ruff", "scrapy/scrapy", "Delgan/loguru",
    "jd/tenacity", "hynek/structlog", "aio-libs/yarl", "aio-libs/multidict", "msiemens/tinydb",
    "tomlkit/tomlkit" , "python-poetry/poetry-core", "jpadilla/pyjwt", "agronholm/anyio",
    "graphql-python/graphene", "joke2k/faker", "pygments/pygments", "Python-Markdown/markdown",
    "lepture/mistune", "mkdocs/mkdocs", "beetbox/confuse", "conan-io/conan", "iterative/dvc",
    "pandas-dev/pandas", "matplotlib/matplotlib", "scikit-learn/scikit-learn", "xarray-contrib/xarray",
    "pydata/xarray", "psf/cachecontrol", "boto/botocore", "getmoto/moto", "huggingface/datasets",
]

QUERY = """
query($q: String!, $cursor: String) {
  search(query: $q, type: ISSUE, first: 25, after: $cursor) {
    pageInfo { hasNextPage endCursor }
    nodes {
      ... on PullRequest {
        number title url mergedAt baseRefName
        mergeCommit { oid parents(first: 2) { nodes { oid } } }
        additions deletions changedFiles
        files(first: 60) { nodes { path additions deletions } }
        closingIssuesReferences(first: 3) { nodes { number title body url createdAt } }
      }
    }
  }
}
"""


def is_test_path(path: str) -> bool:
    p = path.lower()
    name = p.rsplit("/", 1)[-1]
    return (
        p.startswith("tests/") or p.startswith("test/") or "/tests/" in p or "/test/" in p
        or "/testing/" in p or name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"
    )


def gh_graphql(query: str, **variables: str | None) -> dict:
    cmd = ["gh", "api", "graphql", "-f", f"query={query}"]
    for key, value in variables.items():
        if value is not None:
            cmd += ["-f", f"{key}={value}"]
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip())
    return json.loads(out.stdout)


def harvest_repo(repo: str, since: str, max_pages: int) -> list[dict]:
    q = f"repo:{repo} is:pr is:merged merged:>={since}"
    found: list[dict] = []
    cursor = None
    for _ in range(max_pages):
        data = gh_graphql(QUERY, q=q, cursor=cursor)["data"]["search"]
        for pr in data["nodes"]:
            if not pr or not pr.get("mergeCommit"):
                continue
            issues = [i for i in pr["closingIssuesReferences"]["nodes"] if (i.get("body") or "").strip()]
            if not issues:
                continue
            files = pr["files"]["nodes"]
            src = [f["path"] for f in files if f["path"].endswith(".py") and not is_test_path(f["path"])]
            tests = [f["path"] for f in files if f["path"].endswith(".py") and is_test_path(f["path"])]
            non_py = [f["path"] for f in files if not f["path"].endswith(".py")]
            src_lines = sum(f["additions"] + f["deletions"] for f in files if f["path"] in src)
            if not src or not tests or len(src) > 4 or src_lines > 300:
                continue
            parents = pr["mergeCommit"]["parents"]["nodes"]
            found.append({
                "repo": repo,
                "pr_number": pr["number"],
                "pr_url": pr["url"],
                "pr_title": pr["title"],
                "merged_at": pr["mergedAt"],
                "merge_commit": pr["mergeCommit"]["oid"],
                "base_commit": parents[0]["oid"],
                "merge_parents": len(parents),
                "src_files": src,
                "test_files": tests,
                "other_files": non_py,
                "src_lines_changed": src_lines,
                "issues": [
                    {"number": i["number"], "title": i["title"], "body": i["body"], "url": i["url"],
                     "created_at": i["createdAt"]}
                    for i in issues
                ],
            })
        if not data["pageInfo"]["hasNextPage"]:
            break
        cursor = data["pageInfo"]["endCursor"]
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-08-15")
    ap.add_argument("--repos", nargs="*", default=DEFAULT_REPOS)
    ap.add_argument("--max-pages", type=int, default=8)
    ap.add_argument("--out", default="data/swe_live/candidates.jsonl")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with out.open("w", encoding="utf-8") as fh:
        for repo in args.repos:
            try:
                rows = harvest_repo(repo, args.since, args.max_pages)
            except RuntimeError as exc:
                print(f"{repo}: ERROR {exc}", file=sys.stderr)
                continue
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += len(rows)
            print(f"{repo}: {len(rows)} candidates", file=sys.stderr)
    print(f"total candidates: {total} -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
