"""Recall probes with the `recall` tool enabled, as the deployed model runs (and without it).

    PYTHONPATH=~/pylibs/fla ~/evalenv/bin/python -m scripts.transplant.recall_eval \\
        --model exact:Qwen/Qwen3.5-0.8B --name q08i-teacher --chars 16000,64000

Items come from scripts/transplant/recall_data.make_item on held-out corpus_v4 filler (needle,
multi-hop, "what did I say earlier", long copy), at several context sizes. Greedy chat
decoding with the model's chat template (transformers apply_chat_template, tools=[recall]):
when the model emits a <tool_call> for recall, the runtime executes it on the session's
verbatim history (learning/recall_tool.py), appends the <tool_response> and lets the model
continue (at most --max-calls calls). Exact match = the answer string appears in the final
reply. 'without tool' runs the same items with no tool offered. Writes <out>/<name>.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import torch

from . import common  # noqa: F401  (puts the repo and engine/ on sys.path first)
from learning.recall_tool import SCHEMA, SessionHistory
from .common import load_byte_shards, log
from .evaluate import build_core
from .recall_data import answer_found, make_item

CALL = re.compile(r"<tool_call>\s*<function=([\w\-]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM = re.compile(r"<parameter=([\w\-]+)>\s*(.*?)\s*</parameter>", re.S)
TASKS = ("needle", "multihop", "earlier", "copy")
PREFILL_CHUNK = 65536


def evaluation_settings(args):
    chars, modes = [int(x) for x in args.chars.split(",")], args.modes.split(",")
    if not chars or any(x <= 0 for x in chars) or len(set(chars)) != len(chars):
        raise ValueError("recall context sizes must be positive and unique")
    if args.items < 1 or args.max_calls < 0:
        raise ValueError("recall item count must be positive and max calls nonnegative")
    if not modes or any(x not in ("tool", "no_tool") for x in modes) or len(set(modes)) != len(modes):
        raise ValueError("recall modes must be unique tool/no_tool entries")
    return {"chars": chars, "modes": modes, "items": args.items, "max_calls": args.max_calls,
            "seed": args.seed, "max_new": 96, "thinking": False}


def item_fingerprint(items):
    """Identify the exact ordered contexts, questions, answers and verified tool results."""
    raw = json.dumps(items, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_report(path, model, settings, resume=False):
    if not resume or not path.exists():
        return {"model": model, "settings": settings, "results": {}}
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("model") != model:
        raise ValueError("recall resume model mismatch")
    if report.get("settings") != settings:
        raise ValueError("recall resume settings differ or are missing; use a fresh output name")
    return report


@torch.no_grad()
def generate(core, tok, text: str, max_new: int, stop_ids: set[int]) -> str:
    ids = tok(text, add_special_tokens=False)["input_ids"]
    x = torch.tensor([ids], device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        # Prefill without logits; project only the last position (full-prompt logits are T x vocab).
        # A fully recurrent core prefills in chunks from the carried state (the same function as one
        # pass): a single ~1M-token pass exhausts the GB10's unified memory in activations.
        state = None
        step = PREFILL_CHUNK if all(m == "gated_deltanet" for m in core.spec.mixers) else x.shape[1]
        for i in range(0, x.shape[1], step):
            out, state = core.stream(x[:, i:i + step], state, logits=False)
        last = out["final"][:, -1:].to(core.embed.weight.dtype)
        token = torch.nn.functional.linear(last, core.head_weight()).argmax(-1)
        produced = []
        for _ in range(max_new):
            t = int(token)
            if t in stop_ids:
                break
            produced.append(t)
            out, state = core.stream(token, state)
            token = out["logits"][:, -1:].argmax(-1)
    return tok.decode(produced)


def run_item(core, tok, item: dict, with_tool: bool, max_calls: int, stop_ids: set[int]) -> dict:
    history = SessionHistory()
    msgs = []
    for i, turn in enumerate(item["turns"]):
        history.append("user", turn)
        msgs.append({"role": "user", "content": turn})
        if i < len(item["turns"]) - 1:
            msgs.append({"role": "assistant", "content": "Noted."})
    calls = 0
    call_trace = []
    reply = ""
    for _ in range(max_calls + 1):
        prompt = tok.apply_chat_template(msgs, tools=[SCHEMA] if with_tool else None, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        reply = generate(core, tok, prompt, 96, stop_ids)
        # A tool call rebuilds and prefills the full long prompt. Release the previous
        # prefill's cached blocks before the next call or item on unified-memory GPUs.
        torch.cuda.empty_cache()
        match = CALL.search(reply) if with_tool else None
        if not match or calls >= max_calls:
            break
        name, body = match.group(1), match.group(2)
        args = {k: v for k, v in PARAM.findall(body)}
        result = (history.recall(args.get("query", ""), args.get("near"))
                  if name == "recall" else {"ok": False, "error": "unknown_tool"})
        call_trace.append({"name": name, "query": args.get("query", ""),
                           "answer_in_result": any(item["answer"] in r.get("text", "")
                                                   for r in result.get("results", []))})
        calls += 1
        msgs.append({"role": "assistant", "content": reply[:match.start()].strip(),
                     "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args}}]})
        msgs.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False, separators=(",", ":"))})
    final = re.sub(r"<think>.*?</think>", "", reply, flags=re.S).strip()
    return {"exact": answer_found(item, final), "calls": calls, "call_trace": call_trace, "reply": final}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--chars", default="16000,64000", help="context sizes in characters (~4 chars/token)")
    ap.add_argument("--items", type=int, default=8, help="items per (task, size)")
    ap.add_argument("--max-calls", type=int, default=3)
    ap.add_argument("--modes", default="tool,no_tool")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", type=Path, default=Path("~/x2/eval/recall_tool"))
    ap.add_argument("--resume", action="store_true", help="skip completed cells in an existing output file")
    ap.add_argument("--corpus", default="~/corpus_v4_avail",
                    help="filler corpus (corpus_v4 still lists the deleted 'code' source)")
    args = ap.parse_args(argv)
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.name}.json"
    try:
        settings = evaluation_settings(args)
        report = load_report(path, args.model, settings, args.resume)
    except ValueError as exc:
        ap.error(str(exc))
    from transformers import AutoTokenizer
    core, tdir = build_core(args.model, "cuda")
    core.eval()
    tok = AutoTokenizer.from_pretrained(tdir)
    if tok.chat_template is None:
        raise SystemExit(f"{tdir} has no chat template; use the instruct model's tokenizer")
    stop_ids = {i for i in (tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id) if i is not None}
    bs = load_byte_shards()
    corpus = bs.ShardedByteCorpus(Path(args.corpus).expanduser(), "holdout")
    rng = random.Random(args.seed)
    for size in settings["chars"]:
        for task in TASKS:
            items = []
            while len(items) < args.items:
                parts, total = [], 0
                while total < size:
                    text = corpus.text(rng.randrange(len(corpus)))
                    parts.append(text)
                    total += len(text)
                item = make_item(task, "\n\n".join(parts)[:size], rng)
                if item["verified"]:
                    items.append(item)
            digest = item_fingerprint(items)
            for mode in settings["modes"]:
                key = f"{task}@{size}:{mode}"
                if key in report["results"]:
                    if report["results"][key].get("items_sha256") != digest:
                        ap.error(f"{key}: recall input items differ or fingerprint is missing; refusing resume")
                    log(f"{args.name} {key}: resumed")
                    continue
                rows = [run_item(core, tok, it, mode == "tool", args.max_calls, stop_ids) for it in items]
                report["results"][key] = {"items_sha256": digest,
                                          "exact": sum(r["exact"] for r in rows) / len(rows),
                                          "mean_calls": sum(r["calls"] for r in rows) / len(rows),
                                          "sample": rows[0]["reply"],
                                          "examples": [{"question": it["question"], "answer": it["answer"], **row}
                                                       for it, row in zip(items, rows)]}
                log(f"{args.name} {key}: exact={report['results'][key]['exact']:.3f} "
                    f"calls={report['results'][key]['mean_calls']:.2f}")
                path.write_text(json.dumps(report, indent=1))
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
