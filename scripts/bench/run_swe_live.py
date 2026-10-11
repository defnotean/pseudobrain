"""Run a model on PB-SWE-Live with a minimal mini-SWE-agent-style harness and grade it.

Harness (deliberately close to mini-SWE-agent, which Qwen used for its published numbers):
- one bash command per model turn, executed in a network-less container at the task's base commit;
- the workspace is a fresh git repo built from `git archive <base>` (no future history to peek at);
- the agent ends by running `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`;
- grading applies the PR's own tests to the agent's diff in a fresh container; resolved means every
  FAIL_TO_PASS and PASS_TO_PASS test passes.

Works against any OpenAI-compatible endpoint (vLLM, the PseudoBrain server).

Usage (Spark):
    python scripts/bench/run_swe_live.py --tasks data/swe_live/validated.jsonl --base-url http://127.0.0.1:8000/v1 \
        --model qwen3.8-27b --name q38_27b_stock --workers 8
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import json
import random
import re
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from validate_swe_live import CACHE, TEST_LINE, mirror, sh, slug  # same directory

SUBMIT = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
SYSTEM = f"""You are a helpful assistant that can interact with a computer shell to solve programming tasks.

Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).
Include a THOUGHT section before your command explaining your reasoning.

Format:
THOUGHT: your reasoning

```bash
your_command_here
```

Rules:
- Each command runs in a new subshell in /workspace, so `cd` does not persist; prefix commands with `cd dir &&` if needed.
- There is no network access. Do not try to install packages.
- Edit files with sed, python, or `cat <<'EOF' > file` heredocs. Commands must be non-interactive.
- When you are done, run exactly: `echo {SUBMIT}`. After that you cannot continue.
"""

INSTANCE = """<pr_description>
Consider the following issue reported in the repository at /workspace:

{problem}
</pr_description>

<instructions>
Make changes to non-test files in /workspace so that the issue is resolved. The fix should be general and
consistent with the codebase. Recommended workflow: explore the relevant code, write a script that
reproduces the issue, fix the source, verify the fix with your script and edge cases, then submit.
Do not modify existing tests.
</instructions>"""

# Run as the workspace owner: root with --cap-drop ALL cannot write files owned by the host user.
USER_ARGS = ["--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp"] if hasattr(os, "getuid") else []

BASH_BLOCK = re.compile(r"```bash\s*\n(.*?)\n```", re.S)


def chat(base_url: str, model: str, messages: list[dict], args) -> tuple[dict, dict]:
    body = {"model": model, "messages": messages, "max_tokens": args.max_tokens,
            "temperature": args.temperature, "top_p": args.top_p}
    if args.top_k:
        body["top_k"] = args.top_k
    if args.chat_template_kwargs:
        body["chat_template_kwargs"] = json.loads(args.chat_template_kwargs)
    if args.reasoning_effort:
        body["reasoning_effort"] = args.reasoning_effort
    req = urllib.request.Request(f"{base_url}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=args.request_timeout) as r:
                data = json.loads(r.read())
            return data["choices"][0]["message"], data.get("usage", {})
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            if exc.code == 400:  # context overflow or bad request: not retryable
                raise RuntimeError(f"HTTP 400: {detail}") from exc
            time.sleep(5 * (attempt + 1))
        except Exception:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("model request failed after retries")


def make_workspace(task: dict, dest: Path) -> None:
    mdir = mirror(task["repo"])
    dest.mkdir(parents=True)
    archive = subprocess.run(["git", "archive", task["base_commit"]], cwd=mdir, capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive.stdout, check=True)
    for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                ["git", "-c", "user.name=bench", "-c", "user.email=bench@local", "commit", "-q", "-m", "base"]):
        subprocess.run(cmd, cwd=dest, check=True, capture_output=True)


def docker_run(image: str, ws: Path) -> str:
    r = sh(["docker", "run", "-d", "--network", "none", "--cap-drop", "ALL", "--security-opt",
            "no-new-privileges:true", "--memory", "6g", "--cpus", "4", "--pids-limit", "512", *USER_ARGS,
            "-v", f"{ws}:/workspace:rw", "-w", "/workspace", "-e", "PYTHONDONTWRITEBYTECODE=1",
            image, "sleep", "infinity"])
    if r.returncode:
        raise RuntimeError(r.stderr)
    return r.stdout.strip()


def docker_exec(cid: str, command: str, timeout: float) -> str:
    try:
        r = subprocess.run(["docker", "exec", "-w", "/workspace", cid, "bash", "-lc", command],
                           capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
        out = (r.stdout + r.stderr)
        return f"<returncode>{r.returncode}</returncode>\n<output>\n{out}</output>"
    except subprocess.TimeoutExpired:
        return f"<returncode>124</returncode>\n<output>\nCommand timed out after {timeout:.0f}s.</output>"


def truncate(obs: str, limit: int) -> str:
    if len(obs) <= limit:
        return obs
    half = limit // 2
    return obs[:half] + f"\n... [{len(obs) - limit} characters elided] ...\n" + obs[-half:]


def grade(task: dict, diff: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp) / "g"
        make_workspace(task, ws)
        if diff.strip():
            r = sh(["git", "apply", "--whitespace=nowarn", "-"], cwd=ws, inp=diff)
            if r.returncode:
                return {"resolved": False, "reason": "agent diff does not apply", "detail": r.stderr[-500:]}
        # The PR's tests win over any agent edits to the same files.
        touched = re.findall(r"^diff --git a/(\S+) b/", task["test_patch"], re.M)
        for path in touched:
            sh(["git", "checkout", "-q", "HEAD", "--", path], cwd=ws)
        r = sh(["git", "apply", "--whitespace=nowarn", "-"], cwd=ws, inp=task["test_patch"])
        if r.returncode:
            return {"resolved": False, "reason": "test patch does not apply", "detail": r.stderr[-500:]}
        cmd = ("python -m pytest -p no:cacheprovider -rA -q --no-header -o addopts='' --timeout 300 "
               + " ".join(task["test_files"]))
        r = sh(["docker", "run", "--rm", "--network", "none", "--cap-drop", "ALL", "--memory", "6g", "--cpus", "4", *USER_ARGS,
                "-v", f"{ws}:/workspace:rw", "-w", "/workspace", "-e", "PYTHONDONTWRITEBYTECODE=1",
                task["image"], "bash", "-lc", cmd], timeout=1200)
    status = {}
    for line in (r.stdout + r.stderr).splitlines():
        m = TEST_LINE.match(line.strip())
        if m:
            status[m.group(2)] = m.group(1)
    f2p_ok = [t for t in task["FAIL_TO_PASS"] if status.get(t) == "PASSED"]
    p2p_ok = [t for t in task["PASS_TO_PASS"] if status.get(t) == "PASSED"]
    resolved = len(f2p_ok) == len(task["FAIL_TO_PASS"]) and len(p2p_ok) == len(task["PASS_TO_PASS"])
    return {"resolved": resolved, "f2p": f"{len(f2p_ok)}/{len(task['FAIL_TO_PASS'])}",
            "p2p": f"{len(p2p_ok)}/{len(task['PASS_TO_PASS'])}"}


def solve(task: dict, args) -> dict:
    t0 = time.time()
    traj: list[dict] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    exit_status, diff, cid = "step_limit", "", None
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp) / "ws"
        make_workspace(task, ws)
        try:
            cid = docker_run(task["image"], ws)
            messages = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": INSTANCE.format(problem=task["problem_statement"])}]
            for step in range(args.max_steps):
                try:
                    msg, u = chat(args.base_url, args.model, messages, args)
                except RuntimeError as exc:
                    exit_status = f"model_error: {str(exc)[:200]}"
                    break
                for k in usage:
                    usage[k] += u.get(k, 0) or 0
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                assistant = {"role": "assistant", "content": content}
                if reasoning and args.keep_reasoning:
                    assistant["reasoning_content"] = reasoning
                messages.append(assistant)
                traj.append({"step": step, "reasoning_chars": len(reasoning), "content": content})
                blocks = BASH_BLOCK.findall(content)
                if len(blocks) != 1:
                    obs = ("Please always provide EXACTLY ONE action in triple backticks (```bash ... ```), "
                           f"found {len(blocks)} actions.")
                else:
                    command = blocks[0].strip()
                    if SUBMIT in command and command.startswith("echo"):
                        exit_status = "submitted"
                        break
                    obs = truncate(docker_exec(cid, command, args.command_timeout), args.obs_chars)
                traj[-1]["observation"] = obs[:4000]
                messages.append({"role": "user", "content": obs})
            diff = sh(["git", "diff", "HEAD"], cwd=ws).stdout
            # Mirror mini-SWE-agent: untracked new source files count too.
            sh(["git", "add", "-A"], cwd=ws)
            diff = sh(["git", "diff", "--cached", "HEAD"], cwd=ws).stdout
        finally:
            if cid:
                sh(["docker", "rm", "-f", cid])
    result = grade(task, diff) if diff.strip() else {"resolved": False, "reason": "empty diff"}
    return {"instance_id": task["instance_id"], "repo": task["repo"], "exit_status": exit_status,
            "steps": len(traj), "seconds": round(time.time() - t0, 1), **usage, **result,
            "diff": diff, "trajectory": traj}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return round(c - h, 3), round(c + h, 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="data/swe_live/validated.jsonl")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", default="reports/swe_live")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated instance ids")
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--max-statement-chars", type=int, default=30000)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--reasoning-effort", default="")
    ap.add_argument("--chat-template-kwargs", default="")
    ap.add_argument("--keep-reasoning", action="store_true", default=True)
    ap.add_argument("--no-keep-reasoning", dest="keep_reasoning", action="store_false")
    ap.add_argument("--command-timeout", type=float, default=120)
    ap.add_argument("--obs-chars", type=int, default=10000)
    ap.add_argument("--request-timeout", type=float, default=1800)
    args = ap.parse_args()

    tasks = [json.loads(l) for l in Path(args.tasks).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.only:
        keep = set(args.only.split(","))
        tasks = [t for t in tasks if t["instance_id"] in keep]
    random.Random(0).shuffle(tasks)
    if args.limit:
        tasks = tasks[: args.limit]
    # Applied after the seeded subset is drawn, so every model gets the same tasks; an issue longer than this
    # cannot fit a 131k context with room to work.
    tasks = [t for t in tasks if len(t["problem_statement"]) <= args.max_statement_chars]
    out_dir = Path(args.out_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in out_dir.glob("*.json") if p.stem != "summary"}
    todo = [t for t in tasks if t["instance_id"] not in done]
    print(f"{args.name}: {len(tasks)} tasks, {len(todo)} to run", flush=True)

    def work(task: dict) -> dict:
        try:
            res = solve(task, args)
        except Exception as exc:
            res = {"instance_id": task["instance_id"], "repo": task["repo"], "resolved": False,
                   "exit_status": f"harness_error: {exc}"[:400]}
        (out_dir / f"{task['instance_id']}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        return res

    with cf.ThreadPoolExecutor(args.workers) as pool:
        for res in pool.map(work, todo):
            print(res["instance_id"], "RESOLVED" if res["resolved"] else "no", res.get("exit_status"),
                  res.get("steps"), res.get("completion_tokens"), res.get("seconds"), flush=True)

    results = [json.loads(p.read_text(encoding="utf-8")) for p in out_dir.glob("*.json") if p.stem != "summary"]
    ids = {t["instance_id"] for t in tasks}
    results = [r for r in results if r["instance_id"] in ids]
    k = sum(r["resolved"] for r in results)
    summary = {"name": args.name, "model": args.model, "n": len(results), "resolved": k,
               "rate": round(k / max(1, len(results)), 4), "wilson95": wilson(k, len(results)),
               "harness_errors": sum(str(r.get("exit_status", "")).startswith("harness_error") for r in results),
               "completion_tokens": sum(r.get("completion_tokens", 0) or 0 for r in results),
               "settings": {k2: v for k2, v in vars(args).items() if k2 not in ("tasks",)},
               "per_task": {r["instance_id"]: r["resolved"] for r in sorted(results, key=lambda r: r["instance_id"])}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k2: summary[k2] for k2 in ("name", "n", "resolved", "rate", "wilson95", "harness_errors")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
