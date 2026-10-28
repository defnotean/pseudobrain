"""Long-context recall probes for teachers and converted cores (programmatic, exact-match).

    PYTHONPATH=~/pylibs/fla ~/evalenv/bin/python -m scripts.transplant.recall_probes \\
        --model core:~/x2/runs/q08_rec/core.pt --name q08-rec --lengths 1024,4096,8192,16384,32768

Tasks (each item is a haystack of held-out corpus_v4 text of the given token length with
facts inserted at chosen depths; the answer is scored teacher-forced: exact = every answer
token is the greedy argmax, plus answer NLL in nats/token):
  passkey   "The pass key is <5 digits>. Remember it." at depth 10/50/90%; asked at the end
  multihop  "<Name>'s locker number is <3 digits>." early, "The code for locker <n> is <4
            digits>." later; asked for the code of <Name>'s locker
  copy      a 48-character random string early; "Repeat the string exactly:" at the end
Without any tool (the model must hold the information in its state / attention). The recall
tool (learning/recall_tool.py) is scored separately with the tool enabled.
Writes <out>/<name>.json.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import string
from pathlib import Path

import torch
import torch.nn.functional as F

from .common import TeacherTokenizer, holdout_texts, log
from .evaluate import build_core

NAMES = ("Alice", "Bruno", "Chandra", "Dmitri", "Elena", "Farid", "Greta", "Hiro", "Ines", "Jonas")


def filler_tokens(tok: TeacherTokenizer, n: int, seed: int) -> list[int]:
    texts = holdout_texts("~/corpus_v4", 400, 8192)
    rng = random.Random(seed)
    rng.shuffle(texts)
    out: list[int] = []
    for ids in tok.encode_batch(texts):
        out.extend(ids)
        if len(out) >= n:
            break
    return out[:n]


def build_item(task: str, tok, filler: list[int], length: int, depth: float, rng: random.Random):
    """(context ids, answer ids, answer text) with roughly `length` context tokens."""
    enc = lambda s: tok.encode_batch([s])[0]  # noqa: E731
    if task == "passkey":
        key = "".join(rng.choice(string.digits) for _ in range(5))
        facts = [(depth, enc(f" The pass key is {key}. Remember it. "))]
        question, answer = enc(" What is the pass key? The pass key is"), " " + key
    elif task == "multihop":
        name = rng.choice(NAMES)
        locker = str(rng.randrange(100, 1000))
        code = "".join(rng.choice(string.digits) for _ in range(4))
        facts = [(depth * 0.5, enc(f" {name}'s locker number is {locker}. ")),
                 (0.5 + depth * 0.4, enc(f" The code for locker {locker} is {code}. "))]
        question = enc(f" What is the code for {name}'s locker? The code for {name}'s locker is")
        answer = " " + code
    elif task == "copy":
        secret = "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(48))
        facts = [(depth, enc(f" Here is a string: {secret} "))]
        question, answer = enc(" Repeat the string exactly:"), " " + secret
    else:
        raise ValueError(task)
    budget = length - len(question) - sum(len(f) for _, f in facts)
    body = list(filler[:max(0, budget)])
    for frac, fact in sorted(facts, key=lambda x: -x[0]):
        at = int(len(body) * frac)
        body[at:at] = fact
    return [tok.bos] + body + question, enc(answer), answer


STREAM = {"chunk": 0, "question": 64, "decode": False}
FIRST_MISS: list = []


@torch.no_grad()
def score(core, context: list[int], answer: list[int]) -> tuple[bool, float]:
    ids = torch.tensor([context + answer], device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if STREAM["chunk"]:  # streaming prefill through the bounded caches (and the exact memory tier if on)
            state, finals, n = None, [], ids.shape[1]
            # context in large chunks; the last QUESTION_TOKENS of the context plus the answer as one small
            # final chunk, so retrieval (exact memory tier) is driven by the question, as in decoding
            cut = max(0, len(context) - STREAM["question"])
            bounds = [(s, min(s + STREAM["chunk"], cut)) for s in range(0, cut, STREAM["chunk"])]
            if STREAM["decode"]:  # question as one chunk, then each answer token alone (retrieval per token)
                bounds += [(cut, len(context))] + [(i, i + 1) for i in range(len(context), n)]
            else:
                bounds += [(cut, n)]
            for s, e in bounds:
                out, state = core.stream(ids[:, s:e], state, logits=False)
                if e > len(context) - 1:
                    finals.append((s, out["final"][0]))
            seq = torch.cat([f for _, f in finals])
            start = finals[0][0]
            final = seq[len(context) - 1 - start:n - 1 - start]
        else:
            final = core(ids, logits=False)["final"][0, len(context) - 1:-1]
    logits = F.linear(final.to(torch.bfloat16), core.head_weight().to(torch.bfloat16)).float()
    want = torch.tensor(answer, device="cuda")
    nll = float(F.cross_entropy(logits, want, reduction="mean"))
    wrong = (logits.argmax(-1) != want).nonzero().flatten()
    FIRST_MISS.append(int(wrong[0]) if len(wrong) else -1)  # index of the first wrong answer token (-1 = exact)
    return bool(len(wrong) == 0), nll


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--lengths", default="1024,4096,8192,16384,32768")
    ap.add_argument("--tasks", default="passkey,multihop,copy")
    ap.add_argument("--items", type=int, default=12, help="items per (task, length); depths cycle 0.1/0.5/0.9")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("~/x2/eval/recall"))
    ap.add_argument("--stream-chunk", type=int, default=0, help="score through stream() in chunks of this size")
    ap.add_argument("--decode", action="store_true", help="feed the answer one token at a time, as in generation")
    ap.add_argument("--exact-memory", default="", help="enable the exact memory tier: blocks,block_size e.g. 64,64")
    ap.add_argument("--no-sinks", action="store_true", help="serve parity: drop the 4 sink positions (as vLLM does)")
    args = ap.parse_args(argv)
    if args.no_sinks:
        from . import evaluate as ev
        ev.NO_SINKS = True
    STREAM["chunk"] = args.stream_chunk
    STREAM["decode"] = args.decode
    if args.exact_memory:
        from efficiency.exact_memory import EXACT_MEMORY
        parts = [int(v) for v in args.exact_memory.split(",")]
        EXACT_MEMORY.update(enabled=True, blocks=parts[0], block_size=parts[1],
                            decode_blocks=parts[2] if len(parts) > 2 else 0,
                            neighbors=parts[3] if len(parts) > 3 else 0,
                            rerank=parts[4] if len(parts) > 4 else 0)
        if not args.stream_chunk:
            raise SystemExit("--exact-memory needs --stream-chunk (the tier lives in the streaming caches)")
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    core, tdir = build_core(args.model, "cuda")
    core.eval()
    tok = TeacherTokenizer(tdir)
    lengths = [int(x) for x in args.lengths.split(",")]
    filler = filler_tokens(tok, max(lengths) + 64, args.seed)
    report = {"model": args.model, "results": {}}
    for task in args.tasks.split(","):
        for length in lengths:
            rng = random.Random(f"{args.seed}-{task}-{length}")
            exact, nll = 0, 0.0
            for i in range(args.items):
                depth = (0.1, 0.5, 0.9)[i % 3]
                ctx, ans, _ = build_item(task, tok, filler, length, depth, rng)
                ok, loss = score(core, ctx, ans)
                exact += ok
                nll += loss
            row = {"exact": exact / args.items, "answer_nll": nll / args.items,
                   "first_miss": FIRST_MISS[-args.items:]}
            report["results"][f"{task}@{length}"] = row
            log(f"{args.name} {task}@{length}: {row}")
            (out / f"{args.name}.json").write_text(json.dumps(report, indent=1))
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
