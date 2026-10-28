"""Evaluate teachers and transplant students on one battery (docs/SKILL_TRANSPLANT.md).

    PYTHONPATH=~/pylibs/fla ~/evalenv/bin/python -m scripts.transplant.evaluate \\
        --model exact:Qwen/Qwen3.5-0.8B-Base --name q08-teacher --limit 1000

Models:
  exact:<teacher>         the teacher as an exact TokenCore (its own layers in PB modules;
                          tested equal to Hugging Face transformers)
  swa:<teacher>:<W>       the teacher with every attention layer limited to a W-token window
  core:<core.pt>          a converted TokenCore (scripts/transplant/convert_token.py)
  hf:<repo>               a Hugging Face model through eval_base (reference)
  pb:<ckpt>[:<recipe>]    a byte-level PseudoBrain checkpoint through eval_base

Tasks (eval_base, same data and scoring as ~/eval_base/summary.md): hellaswag, piqa,
winogrande, arc_easy, arc_challenge, lambada (first --limit items), bpb (frozen corpus_v4
holdout, --holdout-docs x --holdout-bytes per domain); and the transplant probes:
  facts    facts_v1.json closed-book (prompt only) vs open-book (support sentence first):
           answer bits per byte and greedy exact match per item
  squad    SQuAD v1.1 validation (first --squad questions): open-book (passage given:
           grounded reading comprehension) vs closed-book (question only)
  speed    prefill tokens/s (batch 1 and 16 at 2048 tokens) and, for fully recurrent
           cores, step-by-step decode tokens/s from the carried state
Output: <out>/<name>/results.json (+ per-item facts.jsonl / squad.jsonl).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from .common import ROOT, far_copy_mask, log, parse_window, teacher_dir, token_rows_sha256
from efficiency import transplant as tp

sys.path.insert(0, str(ROOT))
try:
    from scripts.eval_base import models as eb_models  # noqa: E402
    from scripts.eval_base.run import run_task  # noqa: E402
except ImportError:
    eb_models = None
    run_task = None

LN2 = math.log(2)
EVAL_BASE_TASKS = ("hellaswag", "piqa", "winogrande", "arc_easy", "arc_challenge", "lambada", "bpb")


class CoreLM(torch.nn.Module):
    """A TokenCore behind the Hugging Face causal-LM surface eval_base's HFModel uses."""

    def __init__(self, core: tp.TokenCore, context: int = 8192):
        super().__init__()
        self.core = core
        self.config = SimpleNamespace(max_position_embeddings=context, _commit_hash=None)

    def forward(self, input_ids=None, attention_mask=None):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            final = self.core(input_ids, logits=False)["final"]
        logits = torch.nn.functional.linear(final.to(torch.bfloat16), self.core.head_weight().to(torch.bfloat16))
        return SimpleNamespace(logits=logits)

    def get_output_embeddings(self):
        return SimpleNamespace(weight=self.core.head_weight())


NO_SINKS = False


def build_core(spec: str, device: str) -> tuple[tp.TokenCore, Path]:
    if not any(spec.startswith(f"{k}:") for k in ("exact", "swa", "core", "grafted")):
        spec = f"core:{spec}"
    kind, _, rest = spec.partition(":")
    if kind == "exact":
        tdir = teacher_dir(rest)
        core, _ = tp.load_hf_core(tdir, None, dtype=torch.bfloat16, device=device)
        return core, tdir
    if kind == "swa":
        name, _, window = rest.rpartition(":")
        tdir = teacher_dir(name)
        c = tp.TeacherConfig.from_hf(json.loads((tdir / "config.json").read_text(encoding="utf-8")))
        window, sinks = parse_window(window)
        core, _ = tp.load_hf_core(tdir, tp.CoreSpec.exact(c, attention_window=window, attention_sinks=sinks),
                                  dtype=torch.bfloat16, device=device)
        return core, tdir
    if kind == "core":
        payload = torch.load(Path(rest).expanduser(), map_location="cpu", weights_only=False)
        args = payload.get("history", {}).get("args", {})
        tdir = teacher_dir(args.get("teacher", "Qwen/Qwen3.5-0.8B-Base"))
        if "mixers_only" in payload:  # a --shared conversion: teacher weights + the new mixers
            spec = tp.CoreSpec(**payload["config"]["spec"])
            core, _ = tp.load_hf_core(tdir, spec, dtype=torch.bfloat16, device=device)
            missing = core.load_state_dict(payload["mixers_only"], strict=False)
            if missing.unexpected_keys:
                raise ValueError(f"unexpected mixer keys {missing.unexpected_keys[:5]}")
            if "lora_adjacent" in payload:  # adapters on the MLPs next to converted layers: merge them in
                from efficiency.lora import add_pack, remove_pack
                pack = payload["lora_adjacent"]
                add_pack(core, "adjacent", alpha=pack["alpha"], tensors=pack["tensors"])
                remove_pack(core, "adjacent", merge=True)
            return core, tdir
        core = tp.TokenCore.from_config_dict(payload["config"])
        core.load_state_dict(payload["state_dict"])
        return core.to(device=device, dtype=torch.bfloat16), tdir
    if kind == "grafted":
        core_path, _, ws_path = rest.partition(":")
        core, tdir = build_core(f"core:{core_path}", device)
        from efficiency.token_workspace import GraftedCore, ThoughtWorkspace
        ws_ckpt = torch.load(Path(ws_path).expanduser(), map_location="cpu", weights_only=False)
        cfg = ws_ckpt["config"]
        ws = ThoughtWorkspace(cfg, core.c.vocab).to(device=device)
        ws.load_state_dict(ws_ckpt["state_dict"])
        grafted = GraftedCore(core, ws)
        return grafted, tdir
    raise ValueError(spec)


def teacher_dir_of(spec: str) -> Path:
    if not any(spec.startswith(f"{k}:") for k in ("exact", "swa", "core", "grafted")):
        spec = f"core:{spec}"
    kind, _, rest = spec.partition(":")
    if kind == "grafted":
        core_path, _, _ = rest.partition(":")
        return teacher_dir_of(f"core:{core_path}")
    if kind == "core":
        payload = torch.load(Path(rest).expanduser(), map_location="cpu", weights_only=False, mmap=True)
        return teacher_dir(payload.get("history", {}).get("args", {}).get("teacher", "Qwen/Qwen3.5-0.8B-Base"))
    return teacher_dir(rest.rpartition(":")[0] if kind == "swa" else rest)


def load_adapter(spec: str, args):
    kind = spec.partition(":")[0]
    if kind in ("hf", "pb"):
        return eb_models.load_model(spec, device="cuda", name=args.name, pb_repo=str(ROOT),
                                    memory_fraction=None, max_batch=args.max_batch,
                                    token_budget=args.token_budget), None
    from transformers import AutoTokenizer

    core, tdir = build_core(spec, "cuda")
    core.eval()
    if NO_SINKS:  # serve-parity check: vLLM on the GB10 (FA2) cannot keep the 4 sink positions
        n = 0
        for m in core.modules():
            if getattr(m, "window", None) is not None and getattr(m, "sinks", 0):
                m.sinks = 0; n += 1
        log(f"attention sinks disabled on {n} windowed attention layers")
    if args.quant:
        from efficiency.quantize import quantize_weights_
        args.quant_stats = quantize_weights_(core.blocks, args.quant)
        log(f"quantized: {args.quant_stats}")
    tokenizer = AutoTokenizer.from_pretrained(tdir)
    adapter = eb_models.HFModel(spec, name=args.name, model=CoreLM(core), tokenizer=tokenizer,
                                window=2048, max_batch=args.max_batch, memory_fraction=None,
                                logits_budget=int(args.logits_budget))
    return adapter, core


def item_flags(task: str, records: list[dict]) -> dict:
    """0/1 per item for the task's accuracy metrics (eval_base record formats)."""
    if task == "lambada":
        return {"acc": [int(bool(r["greedy"])) for r in records]}
    return {"acc": [int(r["pred"] == r["gold"]) for r in records],
            "acc_norm": [int(r["pred_norm"] == r["gold"]) for r in records]}


def load_facts() -> list[dict]:
    data = json.loads((Path(__file__).parent / "facts_v1.json").read_text(encoding="utf-8"))
    out = []
    for kind, x, y in data["items"]:
        prompt, answer, support = data["templates"][kind]
        out.append({"kind": kind, "closed": prompt.format(x=x, y=y), "answer": answer.format(x=x, y=y),
                    "support": support.format(x=x, y=y)})
    return out


def score_pairs(adapter, pairs: list[tuple[str, str]]) -> list[dict]:
    rows = adapter.loglikelihood(pairs)
    out = []
    for (ctx, cont), row in zip(pairs, rows):
        n = len(cont.encode("utf-8"))
        out.append({"bpb": -row["ll"] / LN2 / n, "greedy": bool(row["greedy"])})
    return out


def run_facts(adapter) -> tuple[list[dict], dict]:
    items = load_facts()
    closed = score_pairs(adapter, [(it["closed"], it["answer"]) for it in items])
    opened = score_pairs(adapter, [(it["support"] + "\n" + it["closed"], it["answer"]) for it in items])
    records = [{**it, "closed_bpb": a["bpb"], "closed_em": a["greedy"], "open_bpb": b["bpb"], "open_em": b["greedy"]}
               for it, a, b in zip(items, closed, opened)]
    n = len(records)
    metrics = {"items": n,
               "closed_em": sum(r["closed_em"] for r in records) / n,
               "closed_bpb": sum(r["closed_bpb"] for r in records) / n,
               "open_em": sum(r["open_em"] for r in records) / n,
               "open_bpb": sum(r["open_bpb"] for r in records) / n}
    return records, metrics


def load_squad(n: int) -> list[dict]:
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset

    data = load_dataset("rajpurkar/squad", split="validation")
    out, seen = [], set()
    for row in data:
        if row["context"] in seen:  # one question per passage
            continue
        seen.add(row["context"])
        out.append({"id": row["id"], "context": row["context"], "question": row["question"],
                    "answer": row["answers"]["text"][0]})
        if len(out) >= n:
            break
    return out


def run_squad(adapter, n: int) -> tuple[list[dict], dict]:
    items = load_squad(n)
    opened = score_pairs(adapter, [(f"{it['context']}\n\nQuestion: {it['question']}\nAnswer:", " " + it["answer"])
                                   for it in items])
    closed = score_pairs(adapter, [(f"Question: {it['question']}\nAnswer:", " " + it["answer"]) for it in items])
    records = [{"id": it["id"], "open_bpb": a["bpb"], "open_em": a["greedy"], "closed_bpb": b["bpb"],
                "closed_em": b["greedy"]} for it, a, b in zip(items, opened, closed)]
    k = len(records)
    metrics = {"items": k, **{f"{m}_{s}": sum(r[f"{m}_{s}"] for r in records) / k
                              for m in ("open", "closed") for s in ("em", "bpb")}}
    return records, metrics


@torch.no_grad()
def run_longnll(core: tp.TokenCore, tdir: Path, docs: int = 8, length: int = 16384, *, joined: bool = False) -> dict:
    """NLL (nats/token) by position bucket on long held-out documents (books: corpus_v4
    holdout), each cut to `length` tokens: where bounded windows / recurrence would lose
    what attention keeps. Compare against the teacher's numbers from the same task.
    joined=True (the 32k task): the holdout splits each book into consecutive ~60 KB chunks, so
    a row is consecutive chunks joined by a blank line (a chunk boundary may start a new book;
    both models read the same text)."""
    from .common import TeacherTokenizer, load_byte_shards
    bs = load_byte_shards()
    corpus = bs.ShardedByteCorpus(Path("~/corpus_v4").expanduser(), "holdout")
    tok = TeacherTokenizer(tdir)
    start, stop = corpus.source_range("books")
    rows = []
    i = start
    while i < stop and len(rows) < docs:
        if joined:
            ids, j = [], i
            while j < stop and len(ids) < length:
                ids.extend(tok.encode_batch([("\n\n" if j > i else "") + corpus.text(j)])[0])
                j += 1
            i = j
        else:
            ids = tok.encode_batch([corpus.text(i)[: length * 8]])[0]
            i += 1
        if len(ids) >= length:
            rows.append([tok.bos] + ids[:length - 1])
    buckets = {"0-2k": (0, 2048), "2k-8k": (2048, 8192), "8k-16k": (8192, 16384)}
    if length > 16384:
        buckets["16k-32k"] = (16384, length)
    sums = {k: [0.0, 0] for k in buckets}
    blocks = []  # mean NLL of every 1024-token block of every row: paired comparisons between models
    # per row and bucket: [NLL sum, count] over far-copy tokens (far_copy_mask), then over the rest
    split = []
    w = core.head_weight().to(torch.bfloat16)
    for row in rows:
        ids = torch.tensor([row], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            final = core(ids, logits=False)["final"][0, :-1]
        target = ids[0, 1:]
        per_token = []
        for s0 in range(0, final.shape[0], 2048):
            logits = torch.nn.functional.linear(final[s0:s0 + 2048].to(torch.bfloat16), w).float()
            nll = torch.nn.functional.cross_entropy(logits, target[s0:s0 + 2048], reduction="none")
            per_token.append(nll)
            pos = torch.arange(s0, s0 + nll.shape[0], device=nll.device)
            for k, (lo, hi) in buckets.items():
                m = (pos >= lo) & (pos < hi)
                sums[k][0] += float(nll[m].sum())
                sums[k][1] += int(m.sum())
        per_token = torch.cat(per_token)
        blocks.append([round(float(per_token[b0:b0 + 1024].mean()), 5) for b0 in range(0, per_token.numel(), 1024)])
        copy = torch.tensor(far_copy_mask(row), device=per_token.device)
        pos = torch.arange(per_token.numel(), device=per_token.device)
        cells = {}
        for k, (lo, hi) in buckets.items():
            m = (pos >= lo) & (pos < hi)
            a, b = m & copy, m & ~copy
            cells[k] = [round(float(per_token[a].sum()), 4), int(a.sum()), round(float(per_token[b].sum()), 4),
                        int(b.sum())]
        split.append(cells)
        del final
    out = {k: v[0] / max(1, v[1]) for k, v in sums.items()}
    for k in buckets:  # far-copy tokens: share and mean NLL; the rest: mean NLL
        c_sum, c_n, o_sum, o_n = (sum(r[k][i] for r in split) for i in range(4))
        out[f"{k}_far_copy_frac"] = c_n / max(1, c_n + o_n)
        out[f"{k}_far_copy_nll"] = c_sum / c_n if c_n else None
        out[f"{k}_other_nll"] = o_sum / o_n if o_n else None
    out["docs"] = len(rows)
    out["tokens"] = length
    out["data_sha256"] = token_rows_sha256(rows)
    out["data_identity"] = "token-rows-json-v1"
    out["block_nll_1024"] = blocks
    out["far_copy_split"] = {"n": 8, "distance": 1024, "rows": split}
    return out


@torch.no_grad()
def run_speed(core: tp.TokenCore) -> dict:
    """PyTorch (torch.compile) speeds of this module on the GB10: a lower bound for a tuned
    runtime, comparable across the arms."""
    out = {}
    forward = torch.compile(lambda ids: core(ids, logits=False)["final"], dynamic=False)
    for batch in (1, 16):
        ids = torch.randint(0, core.c.vocab, (batch, 2048), device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            forward(ids)
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(3):
                forward(ids)
            torch.cuda.synchronize()
        out[f"prefill_tok_per_s_b{batch}"] = 3 * ids.numel() / (time.time() - t)
    if all(m == "gated_deltanet" for m in core.spec.mixers):
        out.update(decode_speed(core))
    out.update(context_decode(core))
    out["parameters"] = sum(p.numel() for p in core.parameters())
    out["state_bytes_per_stream"] = state_bytes(core)
    return out


@torch.no_grad()
def context_decode(core: tp.TokenCore, contexts=(1024, 32768), steps: int = 32) -> dict:
    """Batch-1 greedy decode through TokenCore.stream (eager, the same path for every model:
    recurrent states, bounded windows and full KV caches) after a prefill of each context
    length: tokens/s and the streaming state's size (what constant memory means)."""
    out = {}
    for ctx in contexts:
        ids = torch.randint(0, core.c.vocab, (1, ctx), device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = None
            for s0 in range(0, ctx, 4096):  # chunked prefill
                o, state = core.stream(ids[:, s0:s0 + 4096], state, logits=False)
            tok = ids[:, -1:]
            for _ in range(4):
                o, state = core.stream(tok, state)
                tok = o["logits"][:, -1:].argmax(-1)
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(steps):
                o, state = core.stream(tok, state)
                tok = o["logits"][:, -1:].argmax(-1)
            torch.cuda.synchronize()
        out[f"eager_decode_tok_per_s_ctx{ctx}"] = steps / (time.time() - t)
        out[f"state_mb_ctx{ctx}"] = tp.TokenCore.state_bytes(state) / 2 ** 20
    return out


def state_bytes(core: tp.TokenCore) -> int | None:
    """Recurrent state size of a fully recurrent core (fp32 matrices + conv tails)."""
    if not all(m == "gated_deltanet" for m in core.spec.mixers):
        return None
    total = 0
    for b in core.blocks:
        total += 4 * (b.value_heads * b.key_dim * b.value_dim + b.tail_length * b.channels)
    return total


@torch.no_grad()
def decode_speed(core: tp.TokenCore, steps: int = 128) -> dict:
    """Greedy step-by-step decode from the carried state (batch 1 and 16)."""
    out = {}
    w = core.head_weight().to(torch.bfloat16)
    for batch in (1, 16):
        states = []
        for b in core.blocks:
            states.append(list(tp._zero_state(b, torch.zeros(batch, 1, device="cuda"))))
        tok = torch.zeros(batch, 1, dtype=torch.long, device="cuda")

        def raw_step(tok, flat):
            x = core.embed(tok).float()
            new = []
            for i, b in enumerate(core.blocks):
                y, h, m, tail = b(x, flat[3 * i], flat[3 * i + 1], flat[3 * i + 2])
                new += [h, m, tail]
                x = y
            logits = torch.nn.functional.linear(core.norm(x).to(torch.bfloat16), w)
            return logits.argmax(-1), new

        compiled = torch.compile(raw_step, dynamic=False)
        flat = [t for st in states for t in st]

        def step(tok):
            nonlocal flat
            tok, flat = compiled(tok, flat)
            return tok

        with torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(8):
                tok = step(tok)
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(steps):
                tok = step(tok)
            torch.cuda.synchronize()
        out[f"decode_tok_per_s_b{batch}"] = steps * batch / (time.time() - t)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--no-sinks", action="store_true", help="zero the sink positions of windowed attention layers")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", type=Path, default=Path("~/x2/eval"))
    ap.add_argument("--tasks", default="hellaswag,piqa,arc_easy,arc_challenge,winogrande,lambada,bpb,facts,squad,speed")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--squad", type=int, default=400)
    ap.add_argument("--data", type=Path, default=Path("~/eval_base/data"))
    ap.add_argument("--holdout-docs", type=int, default=200)
    ap.add_argument("--holdout-bytes", type=int, default=4096)
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--token-budget", type=int, default=16384)
    ap.add_argument("--logits-budget", type=float, default=2e9)
    ap.add_argument("--quant", default=None, choices=(None, "int8", "int4", "nvfp4"),
                    help="weight-only round-to-nearest fake quantization of every core Linear")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    global NO_SINKS
    NO_SINKS = args.no_sinks
    out = (args.out.expanduser() / args.name)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "results.json"
    results = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"tasks": {}}
    t0 = time.time()
    adapter, core = load_adapter(args.model, args)
    results.update(model=args.model, name=args.name, describe=adapter.describe(),
                   quant=getattr(args, "quant_stats", None))
    eb_args = SimpleNamespace(data=args.data.expanduser(), limit=args.limit, corpus="auto",
                              holdout_docs=args.holdout_docs, holdout_bytes=args.holdout_bytes, domains=None, seed=None)
    for task in [t for t in args.tasks.split(",") if t]:
        if task in results["tasks"] and not args.force:
            log(f"{task}: done already")
            continue
        t1 = time.time()
        if task in EVAL_BASE_TASKS:
            records, metrics = run_task(task, adapter, eb_args, log)
            if task == "bpb":
                metrics = {"all": metrics["all"]["bpb"],
                           **{d: m.get("bpb") for d, m in metrics["domains"].items()}}
            else:
                metrics = {k: v for k, v in metrics.items() if not isinstance(v, (dict, list))}
                # per-item correctness: paired student-vs-teacher differences (report.py)
                (out / f"{task}.items.json").write_text(json.dumps(item_flags(task, records)), encoding="utf-8")
        elif task == "facts":
            records, metrics = run_facts(adapter)
        elif task == "squad":
            records, metrics = run_squad(adapter, args.squad)
        elif task == "speed":
            if core is None:
                continue
            records, metrics = [], run_speed(core)
        elif task == "longnll":
            if core is None:
                continue
            records, metrics = [], run_longnll(core, teacher_dir_of(args.model))
        elif task == "longnll32":  # 4 rows of 32k tokens from consecutive holdout book chunks
            if core is None:
                continue
            records, metrics = [], run_longnll(core, teacher_dir_of(args.model), docs=4, length=32768, joined=True)
        else:
            raise ValueError(task)
        metrics["seconds"] = round(time.time() - t1, 1)
        results["tasks"][task] = metrics
        if task in ("facts", "squad"):
            with open(out / f"{task}.jsonl", "w", encoding="utf-8", newline="\n") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        log(f"{task}: {json.dumps(metrics)[:300]}")
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
