"""Recall probes (passkey / multihop / copy) against a served model through the OpenAI completions API.

Same items as recall_probes.py (teacher tokenizer, corpus filler, depths 0.1/0.5/0.9 cycling), sent as token-id
prompts with greedy decoding; an item is exact when the generated text starts with the answer. Used to gate the exact
memory tier inside the vLLM serve against the PyTorch tier's numbers (docs/PB27_MEMORY_SERVE.md).

    python -m scripts.transplant.recall_http --model pb27 --name pb27_nvfp4_mem --lengths 32768,65536,131072
"""
from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
from pathlib import Path

from .common import TeacherTokenizer, teacher_dir
from .recall_probes import build_item, filler_tokens


def complete(base_url: str, model: str, ids: list[int], max_tokens: int) -> str:
    body = {"model": model, "prompt": ids, "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(f"{base_url}/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())["choices"][0]["text"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="pb27")
    ap.add_argument("--teacher", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--name", required=True)
    ap.add_argument("--lengths", default="32768,65536,131072")
    ap.add_argument("--tasks", default="passkey,multihop,copy")
    ap.add_argument("--items", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("~/pb27/reports/recall"))
    args = ap.parse_args()
    tok = TeacherTokenizer(teacher_dir(args.teacher))
    lengths = [int(x) for x in args.lengths.split(",")]
    filler = filler_tokens(tok, max(lengths) + 1024, args.seed)
    results: dict = {"name": args.name, "model": args.model, "cells": {}}
    for task in args.tasks.split(","):
        for length in lengths:
            rng = random.Random(args.seed * 1000 + length)
            hits, misses, t0 = 0, [], time.time()
            for i in range(args.items):
                depth = (0.1, 0.5, 0.9)[i % 3]
                ctx, ans_ids, ans = build_item(task, tok, filler, length - 64, depth, rng)
                ctx = ctx[: length - len(ans_ids) - 8]
                text = complete(args.base_url, args.model, ctx, len(ans_ids) + 2)
                ok = text.strip().startswith(ans.strip())
                hits += ok
                if not ok:
                    misses.append({"depth": depth, "want": ans.strip(), "got": text.strip()[:60]})
            rate = hits / args.items
            results["cells"][f"{task}@{length}"] = {"exact": rate, "n": args.items, "misses": misses,
                                                    "seconds": round(time.time() - t0)}
            print(f"{task:9s} {length:7d}: exact {rate:.2f} ({hits}/{args.items}) in {time.time() - t0:.0f}s"
                  + (f" misses {[m['depth'] for m in misses]}" if misses else ""), flush=True)
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
