#!/usr/bin/env python3
"""Custom coding check for a served model: original Python tasks (custom_code_problems.py), graded by hidden tests.

The model gets the task prompt only and must answer with one ```python block (standard library only). The last
python block is extracted, concatenated with the hidden tests and run in a throwaway container with no network
(`docker run --rm --network none`, 512 MB, 1 CPU, 30 s). Results append to <out>/<name>.jsonl (resumable per
problem x sample); <out>/<name>.summary.json has pass rates per tier.

  custom_code_bench.py --check-refs                     # run every reference solution against its tests
  custom_code_bench.py --name pb27b_med --samples 2     # query http://127.0.0.1:8000/v1, model pb27
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from custom_code_problems import PROBLEMS  # noqa: E402

SYSTEM = ("You are an expert Python programmer. Answer with exactly one ```python code block containing the "
          "complete solution (imports included, standard library only, no example usage, no tests, no input()). "
          "Keep any explanation outside the code block short.")

_LAST_OPENCODE_BUSY = 0.0
_LAST_OPENCODE_CHECK = 0.0
_LAST_OPENCODE_COMPLETED: int | None = None
_OPENCODE_LOCK = threading.Lock()


class _OpenCodeYield(Exception):
    pass


def _opencode_active(in_flight: bool = False) -> bool:
    global _LAST_OPENCODE_BUSY, _LAST_OPENCODE_CHECK, _LAST_OPENCODE_COMPLETED
    with _OPENCODE_LOCK:
        now = time.time()
        if now - _LAST_OPENCODE_CHECK < 1.0:
            return (now - _LAST_OPENCODE_BUSY) < 1.0
        _LAST_OPENCODE_CHECK = now
        busy = False
        try:
            out = subprocess.check_output(
                ["ss", "-tnH", "state", "established", "( sport = :18080 )"],
                stderr=subprocess.DEVNULL, timeout=2
            ).decode("utf-8", errors="ignore").strip()
            if out:
                for line in out.splitlines():
                    parts = line.split()
                    if len(parts) >= 2 and (int(parts[0]) > 0 or int(parts[1]) > 0):
                        busy = True
        except Exception:
            pass
        try:
            with urllib.request.urlopen("http://127.0.0.1:18080/health", timeout=2) as r:
                comp = json.load(r).get("completed_requests", 0)
            if _LAST_OPENCODE_COMPLETED is not None and comp != _LAST_OPENCODE_COMPLETED:
                busy = True
            _LAST_OPENCODE_COMPLETED = comp
        except Exception:
            pass
        if busy:
            _LAST_OPENCODE_BUSY = time.time()
        return busy


def _wait_opencode_idle(idle_s: float) -> None:
    if idle_s <= 0:
        return
    while True:
        active = _opencode_active(in_flight=False)
        since = time.time() - _LAST_OPENCODE_BUSY
        if not active and since >= idle_s:
            return
        time.sleep(1.0)


def chat(args, messages):
    body = {"model": args.model, "messages": messages, "max_tokens": args.max_tokens,
            "temperature": args.temperature, "top_p": args.top_p}
    if args.top_k:
        body["top_k"] = args.top_k
    if args.reasoning_effort == "off":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    elif args.reasoning_effort:
        body["chat_template_kwargs"] = {"reasoning_effort": args.reasoning_effort}
    if getattr(args, "opencode_yield", 0) > 0:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        while True:
            _wait_opencode_idle(args.opencode_yield)
            req = urllib.request.Request(args.base_url + "/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "Authorization": "Bearer none"})
            try:
                content, finish_reason, usage = [], None, {}
                with urllib.request.urlopen(req, timeout=args.request_timeout) as r:
                    last_chk = time.time()
                    for raw in r:
                        now = time.time()
                        if now - last_chk >= 0.25:
                            last_chk = now
                            if _opencode_active(in_flight=True):
                                r.close()
                                raise _OpenCodeYield()
                        line = raw.decode().strip()
                        if not line.startswith("data:") or line == "data: [DONE]":
                            continue
                        ev = json.loads(line[5:])
                        if ev.get("usage"):
                            usage = ev["usage"]
                        for ch in ev.get("choices") or []:
                            if ch.get("finish_reason"):
                                finish_reason = ch["finish_reason"]
                            d = ch.get("delta") or {}
                            if d.get("content"):
                                content.append(d["content"])
                return {"choices": [{"message": {"content": "".join(content)}, "finish_reason": finish_reason}],
                        "usage": usage}
            except _OpenCodeYield:
                print("[opencode-yield] OpenCode active on :18080; aborted bench request and yielding GPU", flush=True)
                continue
            except Exception as exc:
                if _opencode_active(in_flight=False):
                    continue
                time.sleep(15)
    req = urllib.request.Request(args.base_url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer none"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=args.request_timeout) as r:
                return json.load(r)
        except Exception as exc:  # server restarts, timeouts
            err = exc
            time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"request failed: {err}")


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text or "", re.S)
    return blocks[-1] if blocks else None


def grade(code, tests, image, timeout):
    with tempfile.TemporaryDirectory() as d:
        os.chmod(d, 0o755)
        p = os.path.join(d, "t.py")
        with open(p, "w") as f:
            f.write(code + "\n\n# ---- hidden tests ----\n" + tests + "\nprint('ALL_TESTS_PASSED')\n")
        os.chmod(p, 0o644)
        cmd = ["docker", "run", "--rm", "--network", "none", "--memory", "512m", "--cpus", "1",
               "-v", f"{d}:/w:ro", image, "timeout", str(timeout), "python3", "/w/t.py"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 60)
        except subprocess.TimeoutExpired:
            return False, "timeout (docker)"
        ok = r.returncode == 0 and "ALL_TESTS_PASSED" in r.stdout
        if r.returncode == 124:
            return False, f"timeout ({timeout}s)"
        return ok, (r.stderr or r.stdout)[-1500:]


def summarize(rows, total):
    tiers = {}
    for r in rows:
        t = tiers.setdefault(r["tier"], [0, 0])
        t[0] += r["passed"]
        t[1] += 1
    passed = sum(r["passed"] for r in rows)
    probs = {}
    for r in rows:
        probs.setdefault(r["id"], []).append(r["passed"])
    return {"n": len(rows), "expected": total, "missing": total - len(rows), "passed": passed,
            "pass_rate": round(passed / max(1, len(rows)), 4),
            "problems_solved_any": sum(any(v) for v in probs.values()), "problems": len(probs),
            "by_tier": {k: {"passed": v[0], "n": v[1], "rate": round(v[0] / v[1], 4)} for k, v in sorted(tiers.items())},
            "no_code": sum(r["code"] is None for r in rows),
            "finish_length": sum(r.get("finish_reason") == "length" for r in rows),
            "mean_completion_tokens": round(sum(r.get("completion_tokens") or 0 for r in rows) / max(1, len(rows)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-refs", action="store_true")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="pb27")
    ap.add_argument("--name", default="pb27b")
    ap.add_argument("--out", default="/home/defnotean/pb27/reports/custom_code")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--reasoning-effort", default="medium", help="xhigh/medium/low, 'off' disables thinking")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--request-timeout", type=int, default=3600)
    ap.add_argument("--image", default="pbswe-msiemens-tinydb:env")
    ap.add_argument("--test-timeout", type=int, default=30)
    ap.add_argument("--only", default="", help="comma-separated problem ids")
    ap.add_argument("--opencode-yield", type=float, default=0.0,
                    help="if >0, wait for :18080 idle (seconds) and abort in-flight requests when :18080 is active")
    args = ap.parse_args()
    probs = [p for p in PROBLEMS if not args.only or p["id"] in args.only.split(",")]

    if args.check_refs:
        bad = 0
        with cf.ThreadPoolExecutor(args.workers) as ex:
            for p, (ok, log) in zip(probs, ex.map(lambda p: grade(p["ref"], p["tests"], args.image, args.test_timeout), probs)):
                print(("PASS " if ok else "FAIL ") + p["id"] + ("" if ok else "\n" + log))
                bad += not ok
        print(f"{len(probs) - bad}/{len(probs)} references pass")
        sys.exit(1 if bad else 0)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, args.name + ".jsonl")
    rows = []
    if os.path.exists(path):
        rows = [json.loads(l) for l in open(path) if l.strip()]
    done = {(r["id"], r["sample"]) for r in rows}
    todo = [(p, s) for p in probs for s in range(args.samples) if (p["id"], s) not in done]
    lock = threading.Lock()
    print(f"{len(done)} done, {len(todo)} to run", flush=True)

    def one(item):
        p, s = item
        t0 = time.time()
        resp = chat(args, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p["prompt"]}])
        ch = resp["choices"][0]
        text = ch["message"].get("content") or ""
        code = extract_code(text)
        ok, log = grade(code, p["tests"], args.image, args.test_timeout) if code else (False, "no python block")
        row = {"id": p["id"], "tier": p["tier"], "sample": s, "passed": bool(ok), "log": log if not ok else "",
               "code": code, "finish_reason": ch.get("finish_reason"),
               "completion_tokens": (resp.get("usage") or {}).get("completion_tokens"),
               "seconds": round(time.time() - t0, 1), "content_tail": text[-600:] if not code else ""}
        with lock:
            with open(path, "a") as f:
                f.write(json.dumps(row) + "\n")
            rows.append(row)
            print(f"{'PASS' if ok else 'FAIL'} {p['id']}#{s} {row['completion_tokens']} tok {row['seconds']}s", flush=True)

    with cf.ThreadPoolExecutor(args.workers) as ex:
        for fut in cf.as_completed([ex.submit(one, it) for it in todo]):
            try:
                fut.result()
            except Exception as exc:
                print("ERROR", exc, flush=True)
    summ = summarize([r for r in rows if r["id"] in {p["id"] for p in probs}], len(probs) * args.samples)
    summ.update(name=args.name, model=args.model, reasoning_effort=args.reasoning_effort,
                temperature=args.temperature, samples=args.samples, max_tokens=args.max_tokens)
    with open(os.path.join(args.out, args.name + ".summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ, indent=1))
    sys.exit(0 if summ["missing"] == 0 else 2)


if __name__ == "__main__":
    main()
