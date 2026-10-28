"""Dump a convert_token run's training batches to a file, so the run can train elsewhere (Colab) on exactly the
token stream the corpus samplers produce here, without a corpus replica. CPU only; run where the corpus is.

    PYTHONPATH=. ~/evalenv/bin/python -m scripts.transplant.dump_batches --teacher Qwen/Qwen3.8-27B \
        --batches 800 --batch 1 --seq 32768 --long-fraction 0.5 --chat-fraction 0.25 --long-join books \
        --out ~/pb27/colab/batches.npy

The sampler arguments mean the same as in convert_token.py (defaults identical). The output is a uint32 array
[batches, batch, seq]; give it to convert_token --batch-file with the same --batch/--seq. A sidecar JSON records
the arguments and the array's SHA-256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .common import STAGE3_MIXTURE, TeacherTokenizer, Windows, log, teacher_dir
from .convert_token import JoinedChunks, TokenBatches


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--corpus", default="~/corpus_v4")
    ap.add_argument("--mixture", default=STAGE3_MIXTURE)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--doc-bytes", type=int, default=32768)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--batches", type=int, required=True)
    ap.add_argument("--long-fraction", type=float, default=0.0)
    ap.add_argument("--long-mixture", default="books=0.5,openthoughts=0.2,code=0.15,openr1_math=0.1,wikipedia=0.05")
    ap.add_argument("--long-join-fraction", type=float, default=1.0)
    ap.add_argument("--long-join", default=None)
    ap.add_argument("--chat-fraction", type=float, default=0.0)
    ap.add_argument("--chat-mixture", default="conversation=1.0")
    ap.add_argument("--chat-template-from", default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tdir = teacher_dir(args.teacher)
    tok = TeacherTokenizer(tdir)
    # Same construction as convert_token.main.
    windows = Windows(args.corpus, args.mixture, args.seed, args.doc_bytes)
    long_windows = long_single = None
    if args.long_fraction > 0 and args.long_join:
        long_windows = JoinedChunks(args.corpus, tok, source=args.long_join, seed=args.seed + 1)
        if args.long_join_fraction < 1.0:
            long_single = Windows(args.corpus, args.long_mixture, args.seed + 3, args.seq * 8, min_bytes=args.seq * 3)
    elif args.long_fraction > 0:
        long_windows = Windows(args.corpus, args.long_mixture, args.seed + 1, args.seq * 8, min_bytes=args.seq * 3)
    chat_windows, render = None, None
    if args.chat_fraction > 0:
        from transformers import AutoTokenizer
        from .common import conversation_messages
        chat_tok = AutoTokenizer.from_pretrained(teacher_dir(args.chat_template_from or args.teacher))

        def render(text):
            msgs = conversation_messages(text)
            return None if msgs is None else chat_tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
        chat_windows = Windows(args.corpus, args.chat_mixture, args.seed + 2, args.seq * 8)
    batches = TokenBatches(windows, tok, args.batch, args.seq, long_windows, args.long_fraction,
                           chat_windows, args.chat_fraction, render, long_single, args.long_join_fraction)

    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(out, mode="w+", dtype=np.uint32, shape=(args.batches, args.batch, args.seq))
    for i in range(args.batches):
        arr[i] = batches.next().numpy().astype(np.uint32)
        if i % 50 == 0:
            log(f"batch {i}/{args.batches}")
    arr.flush()
    del arr
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    meta = {"args": {k: str(v) for k, v in vars(args).items()}, "sha256": digest,
            "padding_dropped_tokens": batches.dropped, "chat_skipped_documents": batches.chat_skipped}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    log(f"wrote {out} sha256 {digest[:16]}")


if __name__ == "__main__":
    main()
