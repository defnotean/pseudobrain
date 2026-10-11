"""Parity of a vLLM-served PB-27 against the PyTorch core: greedy continuations and per-step top-1 logprobs.

Two steps (the GPU cannot hold both the PyTorch core and the vLLM server at once):

  1. reference (PyTorch, before the server is up): greedy-continue held-out prompts of several lengths from the
     core with attention sinks disabled (vLLM serves without sinks), saving the prompt token ids, the generated
     ids and each step's chosen-token logprob:
       python -m scripts.transplant.vllm_parity reference --core core:~/x2/runs/pb27a/core.pt \
           --lengths 2048,20000 --new 48 --out ~/pb27/parity_ref.json
  2. compare (server up): send the same prompt ids to /v1/completions (temperature 0, logprobs) and report, per
     prompt, the first step where vLLM's greedy token differs, the match rate, and the mean |delta logprob| of the
     reference tokens over the agreeing prefix:
       python -m scripts.transplant.vllm_parity compare --ref ~/pb27/parity_ref.json --base-url http://127.0.0.1:8000/v1 --model pb27

Within the 16k window the two paths run the same math (bf16 vs vLLM kernels), so the first ~10 tokens should agree
and chosen-token logprobs should differ by < 0.1; a divergence at step 0 or a logprob gap > 0.5 means a weight or
layer-mapping bug. Beyond 16k the GDN branch carries the far context; agreement there checks the gdn mapping.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F


def reference(args) -> None:
    from .common import TeacherTokenizer, holdout_texts
    from . import evaluate as ev
    ev.NO_SINKS = True
    core, tdir = ev.build_core(args.core, "cuda")
    core.eval()
    for m in core.modules():
        if getattr(m, "window", None) is not None and getattr(m, "sinks", 0):
            m.sinks = 0
    tok = TeacherTokenizer(tdir)
    texts = holdout_texts(args.corpus, 40, 400000)
    text = "\n\n".join(texts)
    ids_all = tok.encode_batch([text])[0]
    out = {"core": args.core, "prompts": []}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for length in (int(x) for x in args.lengths.split(",")):
            prompt = [tok.bos] + ids_all[:length - 1]
            x = torch.tensor([prompt], device="cuda")
            o, state = core.stream(x, logits=False)
            last = o["final"][:, -1:].to(core.embed.weight.dtype)
            logits = F.linear(last, core.head_weight()).float()
            gen, lps = [], []
            for _ in range(args.new):
                lp = logits[0, -1].log_softmax(-1)
                t = int(lp.argmax())
                gen.append(t)
                lps.append(float(lp[t]))
                o, state = core.stream(torch.tensor([[t]], device="cuda"), state)
                logits = o["logits"].float()
            out["prompts"].append({"length": length, "prompt_ids": prompt, "generated": gen, "logprobs": lps})
            print(f"length {length}: {tok.tok.decode(gen)[:120]!r}", flush=True)
    Path(args.out).expanduser().write_text(json.dumps(out))


def compare(args) -> None:
    ref = json.loads(Path(args.ref).expanduser().read_text())
    for p in ref["prompts"]:
        body = {"model": args.model, "prompt": p["prompt_ids"], "max_tokens": len(p["generated"]),
                "temperature": 0, "logprobs": 1}
        req = urllib.request.Request(f"{args.base_url}/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            data = json.loads(r.read())
        ch = data["choices"][0]
        vl_tokens = ch["logprobs"]["tokens"]
        vl_lps = ch["logprobs"]["token_logprobs"]
        # token strings vs ids: compare via text of each reference id
        ref_tokens = [args_tok.tok.decode([t]) for t in p["generated"]] if (args_tok := _tok(args)) else None
        first_diff, agree = None, 0
        for i, (rt, vt) in enumerate(zip(ref_tokens, vl_tokens)):
            if rt != vt:
                first_diff = i
                break
            agree += 1
        gaps = [abs(a - b) for a, b in zip(p["logprobs"][:agree], vl_lps[:agree])]
        print(json.dumps({"length": p["length"], "steps": len(p["generated"]), "agree_prefix": agree,
                          "first_diff": first_diff, "mean_abs_dlogprob_on_prefix": sum(gaps) / max(1, len(gaps)),
                          "ref_text": "".join(ref_tokens)[:80], "vllm_text": "".join(vl_tokens)[:80]}), flush=True)


def _tok(args):
    from .common import TeacherTokenizer, teacher_dir
    return TeacherTokenizer(teacher_dir(args.teacher))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reference")
    r.add_argument("--core", required=True)
    r.add_argument("--corpus", default="~/corpus_v4")
    r.add_argument("--lengths", default="2048,20000")
    r.add_argument("--new", type=int, default=48)
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--ref", required=True)
    c.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    c.add_argument("--model", default="pb27")
    c.add_argument("--teacher", default="Qwen/Qwen3.8-27B")
    args = ap.parse_args()
    reference(args) if args.cmd == "reference" else compare(args)


if __name__ == "__main__":
    main()
