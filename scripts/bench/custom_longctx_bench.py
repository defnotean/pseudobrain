#!/usr/bin/env python3
"""Long-context check for a served model over a synthetic Python codebase (haystack), at increasing lengths.

Per length: three single-needle probes (a RELEASE_TOKEN constant at depth 0.1/0.5/0.9, among near-identical
distractor names), one multi-needle probe (4 LIMIT_* integers spread through the context, scored by fraction
recovered) and one multi-hop trace (a 4-function call chain plus constants scattered over the context; the answer is
an integer the model must compute). The haystack is deterministic (seeded), original, and sized with the server's
/tokenize endpoint; the actual prompt length is taken from the response usage. Requests run one at a time, thinking
disabled, greedy, streamed, so TTFT (~prefill) and decode tok/s are measured. Results append to <out>/<name>.jsonl
(resumable per length x probe) and <out>/<name>.summary.json.

  custom_longctx_bench.py --name pb27b_std --lengths 8192,32768,65536,120000
"""
import argparse
import json
import os
import random
import re
import subprocess
import threading
import time
import urllib.request

_LAST_OPENCODE_BUSY = 0.0
_LAST_OPENCODE_COMPLETED: int | None = None


class _OpenCodeYield(Exception):
    pass


def _opencode_active(in_flight: bool = False) -> bool:
    global _LAST_OPENCODE_BUSY, _LAST_OPENCODE_COMPLETED
    busy = False
    try:
        with urllib.request.urlopen("http://127.0.0.1:18080/health", timeout=2) as r:
            comp = json.load(r).get("completed_requests", 0)
        if _LAST_OPENCODE_COMPLETED is not None and comp != _LAST_OPENCODE_COMPLETED:
            busy = True
        _LAST_OPENCODE_COMPLETED = comp
    except Exception:
        pass
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=2) as r:
            txt = r.read().decode("utf-8", errors="ignore")
        active = 0.0
        for line in txt.splitlines():
            if line.startswith("vllm:num_requests_running{") or line.startswith("vllm:num_requests_waiting{"):
                active += float(line.rsplit(" ", 1)[-1])
        if active > (1.0 if in_flight else 0.0):
            busy = True
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

WORDS = ("account cache ledger invoice tenant report metric shard queue worker token session order batch quota "
         "region policy schema record buffer client stream window channel export import audit vendor parcel route "
         "budget snapshot profile catalog review signal payload cursor archive config filter").split()
MODS = ("billing shipping auth search inventory payments notify analytics gateway scheduler storage pricing "
        "loyalty refunds catalogsvc fraud identity reports exports webhooks").split()


def ident(r, n=2):
    return "_".join(r.choice(WORDS) for _ in range(n))


def filler_file(r, i):
    mod = f"{r.choice(MODS)}_{ident(r, 1)}_{i}"
    cls = "".join(w.title() for w in ident(r, 2).split("_"))
    a, b, c = ident(r), ident(r), ident(r)
    k = r.randint(2, 9)
    kind = r.randrange(4)
    if kind == 0:
        body = f'''class {cls}:
    """Keeps {a.replace("_", " ")} entries grouped by {b.replace("_", " ")}."""

    def __init__(self, limit={r.randint(8, 512)}):
        self.limit = limit
        self.items = {{}}

    def add(self, key, value):
        if len(self.items) >= self.limit:
            self.items.pop(next(iter(self.items)))
        self.items[key] = value

    def {c}(self, key, default=None):
        return self.items.get(key, default)
'''
    elif kind == 1:
        body = f'''def {a}(rows, threshold={r.randint(1, 99)}):
    """Return the {b.replace("_", " ")} rows whose score is at least threshold, sorted by score."""
    picked = [row for row in rows if row.get("score", 0) >= threshold]
    picked.sort(key=lambda row: (-row["score"], row.get("id", 0)))
    return picked[:{k * 10}]


def {c}(values):
    total = 0
    for v in values:
        total += v * {k}
    return total // max(1, len(values))
'''
    elif kind == 2:
        body = f'''from dataclasses import dataclass, field


@dataclass
class {cls}Config:
    name: str = "{a}"
    retries: int = {k}
    timeout_s: float = {r.randint(1, 60)}.0
    tags: list = field(default_factory=lambda: ["{b}", "{c}"])

    def describe(self):
        return f"{{self.name}} retries={{self.retries}} timeout={{self.timeout_s}}"
'''
    else:
        body = f'''import logging

log = logging.getLogger("{mod}")


def {a}(event, store):
    """Apply one {b.replace("_", " ")} event to the store; unknown kinds are logged and ignored."""
    kind = event.get("kind")
    if kind == "create":
        store[event["id"]] = event.get("data", {{}})
    elif kind == "delete":
        store.pop(event["id"], None)
    else:
        log.warning("ignored %s event for %s", kind, event.get("id"))
    return len(store)
'''
    return f"# ===== file: {mod.replace('_', '/', 1)}.py =====\n{body}\n"


class Haystack:
    def __init__(self, seed):
        self.r = random.Random(seed)
        self.files = []
        self.chars = 0

    def grow(self, chars):
        while self.chars < chars:
            f = filler_file(self.r, len(self.files))
            self.files.append(f)
            self.chars += len(f)

    def build(self, chars, inserts):
        """inserts: list of (depth 0..1, text). Returns the haystack text of ~chars characters."""
        self.grow(chars)
        out, n = [], 0
        for f in self.files:
            if n >= chars:
                break
            out.append(f)
            n += len(f)
        for depth, text in sorted(inserts, key=lambda x: -x[0]):
            out.insert(min(len(out), int(round(depth * len(out)))), text)
        return "".join(out)


def make_probes(r):
    """Probe specs for one length: (probe id, inserts, question, scorer -> (score, expected))."""
    probes = []
    for depth in (0.1, 0.5, 0.9):
        mod = r.choice(MODS)
        names = [f"RELEASE_TOKEN_{mod}_v{v}" for v in range(1, 7)]
        vals = {n: "%012x" % r.getrandbits(48) for n in names}
        target = r.choice(names)
        ins = [(depth, f'# ===== file: release/{target.lower()}.py =====\n{target} = "{vals[target]}"\n\n')]
        for n in names:
            if n != target:
                ins.append((r.random(), f'# ===== file: release/{n.lower()}.py =====\n{n} = "{vals[n]}"\n\n'))
        q = f"What is the exact string value assigned to {target} in the code above? Answer with only the value."
        probes.append((f"needle@{depth}", ins, q, (lambda exp: lambda a: (float(exp in a), exp))(vals[target])))
    lim = {f"LIMIT_{ident(r, 1).upper()}_{i}": r.randint(1000, 99999) for i in range(4)}
    ins = [((i + 0.5) / 4 * 0.9 + 0.05, f"# ===== file: limits/l{i}.py =====\n{k} = {v}\n\n")
           for i, (k, v) in enumerate(lim.items())]
    q = ("From file: limits/l0.py, file: limits/l1.py, file: limits/l2.py, and file: limits/l3.py above, "
         "write 4 lines mapping each LIMIT_ constant name to its integer value in the exact format:\n"
         'file: limits/l0.py -> "<NAME_0>": <v0>\n'
         'file: limits/l1.py -> "<NAME_1>": <v1>\n'
         'file: limits/l2.py -> "<NAME_2>": <v2>\n'
         'file: limits/l3.py -> "<NAME_3>": <v3>')

    def score_multi(a, lim=lim):
        got = sum(bool(re.search(rf'"?{k}"?\s*:\s*{v}\b', a)) for k, v in lim.items())
        return got / len(lim), lim
    probes.append(("multi4", ins, q, score_multi))
    tag = ident(r, 1)
    consts = [r.randint(2, 40) for _ in range(4)]
    mult = r.randint(2, 9)
    fns = [f"hop_{tag}_{i}" for i in range(4)]
    ins = []
    for i in range(4):
        nxt = f"{fns[i + 1]}(x + K_{tag.upper()}_{i})" if i < 3 else f"(x + K_{tag.upper()}_{i}) * {mult}"
        ins.append((0.05 + 0.9 * r.random(), f"# ===== file: chain/{fns[i]}.py =====\ndef {fns[i]}(x):\n    return {nxt}\n\n"))
        ins.append((0.05 + 0.9 * r.random(), f"# ===== file: chain/k{i}.py =====\nK_{tag.upper()}_{i} = {consts[i]}\n\n"))
    exp = (0 + sum(consts)) * mult
    t_up = tag.upper()
    q = (f"Using file: chain/{fns[0]}.py .. file: chain/{fns[3]}.py and file: chain/k0.py .. file: chain/k3.py above, "
         f"what does {fns[0]}(0) return? Write 4 lines in the exact format:\n"
         f"file: chain/k0.py ({fns[0]}): K_{t_up}_0 = <v0>, sum = <s0>\n"
         f"file: chain/k1.py ({fns[1]}): K_{t_up}_1 = <v1>, sum = <s1>\n"
         f"file: chain/k2.py ({fns[2]}): K_{t_up}_2 = <v2>, sum = <s2>\n"
         f"file: chain/k3.py ({fns[3]}): K_{t_up}_3 = <v3>, sum = <s3>, <s3> * <mult> = <result>\n"
         f"Do not write anything after <result>.")

    def score_trace(a, exp=exp):
        nums = re.findall(r"-?\d+", a.replace(",", ""))
        return float(bool(nums) and int(nums[-1]) == exp), exp
    probes.append(("trace4", ins, q, score_trace))
    return probes


def post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer none"})
    return urllib.request.urlopen(req, timeout=timeout)


def count_tokens(args, text):
    with post(args.base_url.rsplit("/v1", 1)[0] + "/tokenize", {"model": args.model, "prompt": text}, 600) as r:
        return json.load(r)["count"]


def ask(args, context, question):
    msg = ("Below is the source code of a Python repository.\n\n<repository>\n" + context + "</repository>\n\n"
           + question)
    body = {"model": args.model, "messages": [{"role": "user", "content": msg}], "max_tokens": args.max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "none"}}
    idle_s = getattr(args, "opencode_yield", 0.0)
    while True:
        if idle_s > 0:
            _wait_opencode_idle(idle_s)
        t0 = time.time()
        first = None
        text, usage = [], {}
        aborted = [False]
        stop_watch = threading.Event()
        resp_ref = [None]

        def _watcher():
            while not stop_watch.wait(0.25):
                if _opencode_active(in_flight=True):
                    aborted[0] = True
                    r_obj = resp_ref[0]
                    if r_obj is not None:
                        try:
                            r_obj.close()
                        except Exception:
                            pass
                    return

        wt = None
        if idle_s > 0:
            wt = threading.Thread(target=_watcher, daemon=True)
            wt.start()
        try:
            with post(args.base_url + "/chat/completions", body, args.request_timeout) as r:
                resp_ref[0] = r
                for line in r:
                    if aborted[0]:
                        raise _OpenCodeYield()
                    line = line.decode().strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    ev = json.loads(line[5:])
                    if ev.get("usage"):
                        usage = ev["usage"]
                    for ch in ev.get("choices") or []:
                        d = ch.get("delta") or {}
                        piece = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                        if piece and first is None:
                            first = time.time()
                        text.append(d.get("content") or "")
            if aborted[0]:
                raise _OpenCodeYield()
        except Exception as exc:
            if aborted[0] or isinstance(exc, _OpenCodeYield):
                print("[opencode-yield] OpenCode active on :18080; aborted longctx probe and yielding GPU", flush=True)
                continue
            raise
        finally:
            stop_watch.set()
            if wt is not None:
                wt.join(timeout=1.0)
        t1 = time.time()
        first = first or t1
        ct = usage.get("completion_tokens") or 0
        return {"answer": "".join(text).strip(), "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": ct,
                "ttft_s": round(first - t0, 2), "total_s": round(t1 - t0, 2),
                "decode_tok_s": round((ct - 1) / (t1 - first), 2) if ct > 1 and t1 > first else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="pb27")
    ap.add_argument("--name", default="pb27b_std")
    ap.add_argument("--out", default="/home/defnotean/pb27/reports/custom_longctx")
    ap.add_argument("--lengths", default="8192,32768,65536,120000", help="target prompt tokens")
    ap.add_argument("--seed", type=int, default=2710)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--request-timeout", type=int, default=7200)
    ap.add_argument("--fill", type=float, default=0.97, help="fraction of the target used by the haystack")
    ap.add_argument("--opencode-yield", type=float, default=0.0,
                    help="if >0, wait for :18080 idle (seconds) and abort in-flight requests when :18080 is active")
    ap.add_argument("--retry-failed", action="store_true",
                    help="re-run any existing probe rows whose score is < 1.0")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, args.name + ".jsonl")
    rows = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    if any(r.get("error") or not r.get("prompt_tokens") or (args.retry_failed and r.get("score", 0.0) < 1.0) for r in rows):
        rows = [r for r in rows if not r.get("error") and r.get("prompt_tokens")
                and (not args.retry_failed or r.get("score", 0.0) >= 1.0)]
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    done = {(r["length"], r["probe"]) for r in rows}

    hay = Haystack(args.seed)
    hay.grow(400_000)
    sample = "".join(hay.files)[:400_000]
    cpt = len(sample) / count_tokens(args, sample)
    print(f"calibration: {cpt:.3f} chars/token", flush=True)
    for L in [int(x) for x in args.lengths.split(",")]:
        probes = make_probes(random.Random(args.seed * 1000003 + L))
        for pid, ins, q, scorer in probes:
            if (L, pid) in done:
                continue
            ctx = hay.build(int(L * args.fill * cpt) - sum(len(t) for _, t in ins), ins)
            try:
                res = ask(args, ctx, q)
                score, exp = scorer(res["answer"])
                err = None
            except Exception as exc:
                res, score, exp, err = {}, 0.0, None, f"{type(exc).__name__}: {exc}"[:500]
            row = {"length": L, "probe": pid, "score": score, "expected": exp, "error": err, **res}
            rows.append(row)
            with open(path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row)[:400], flush=True)

    by = {}
    for r in rows:
        b = by.setdefault(r["length"], {"scores": {}, "prompt_tokens": [], "ttft_s": [], "decode_tok_s": []})
        b["scores"][r["probe"]] = r["score"]
        for k in ("prompt_tokens", "ttft_s", "decode_tok_s"):
            if r.get(k) is not None:
                b[k].append(r[k])
    summ = {"name": args.name, "model": args.model, "seed": args.seed, "lengths": {}}
    for L, b in sorted(by.items()):
        s = b["scores"]
        summ["lengths"][str(L)] = {
            "needle": round(sum(v for k, v in s.items() if k.startswith("needle")) / max(1, sum(k.startswith("needle") for k in s)), 3),
            "multi4": s.get("multi4"), "trace4": s.get("trace4"), "probes": s,
            "max_prompt_tokens": max(b["prompt_tokens"], default=None),
            "mean_ttft_s": round(sum(b["ttft_s"]) / len(b["ttft_s"]), 1) if b["ttft_s"] else None,
            "mean_decode_tok_s": round(sum(b["decode_tok_s"]) / len(b["decode_tok_s"]), 2) if b["decode_tok_s"] else None,
            "errors": sum(1 for r in rows if r["length"] == L and r.get("error"))}
    with open(os.path.join(args.out, args.name + ".summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
