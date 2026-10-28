"""Numerics check for execution-only speed switches on a real recipe (GPU).

Runs the same weights and data (two carried chunks, forward + backward) as:

    fp32-ref   switches off, fp32 (no autocast, no TF32), batched segments
    fp32-new   switches on,  fp32, batched segments
    bf16-ref   switches off, bf16 autocast, compiled segments (the training path)
    bf16-new   switches on,  bf16 autocast, compiled segments

and reports log-prob / probability / gradient differences against fp32-ref.
"Same math" means fp32-new == fp32-ref to fp32 roundoff, and bf16-new's error
against fp32-ref is no larger than bf16-ref's.

    PB_REPO=$PWD python scripts/bench/check_exact_speedups.py --recipe configs/pb_s2_v32.json \
        --switch slow_factored --switch slow_tape_band --switch boundary_key_fold
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from efficiency.model import EfficiencyOptions, build_model  # noqa: E402
from lab.engine import Config  # noqa: E402


def activate(model, seed):
    """Open zero-initialized gates/projections so every path carries gradient."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in ("slot_tape_scale", "byte_memory_scale"):
                p.fill_(0.5)
            elif (name.endswith("mlp.down.weight") or name.endswith(".up.weight")
                  or name in ("medium_to_output.weight", "slot_tape_id.weight",
                              "reader_byte_query.weight", "reader_gate_proj.weight")
                  or name.startswith("source_gate.")):
                p.copy_(torch.randn(p.shape, generator=gen) * 0.02)


def run(model, ids, labels, chunks, amp):
    model.train()
    model.zero_grad(set_to_none=True)
    ext = torch.zeros_like(ids, dtype=torch.bool)
    state, outs, total, off = None, [], 0.0, 0
    for n in chunks:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            lp, state = model(ids[:, off:off + n], ext[:, off:off + n], ext[:, off:off + n],
                              state=state)[:2]
        loss = F.nll_loss(lp.float().flatten(0, 1), labels[:, off:off + n].flatten())
        total = total + loss
        outs.append(lp.detach().float())
        state = state.detach()
        off += n
    total.backward()
    grads = {n: p.grad.detach().float().clone() for n, p in model.named_parameters()
             if p.grad is not None}
    return torch.cat(outs, 1), grads, float(total)


def compare(ref, other):
    (o1, g1, l1), (o2, g2, l2) = ref, other
    assert set(g1) == set(g2), set(g1) ^ set(g2)
    num = den = 0.0
    worst = []
    for name in g1:
        d = float((g1[name] - g2[name]).norm())
        n = float(g1[name].norm())
        num += d * d
        den += n * n
        if n > 0:
            worst.append((d / n, name))
    worst.sort(reverse=True)
    return {
        "logprob_max_abs": float((o1 - o2).abs().max()),
        "prob_max_abs": float((o1.exp() - o2.exp()).abs().max()),
        "loss_diff": abs(l1 - l2),
        "grad_global_rel": (num ** 0.5) / max(den ** 0.5, 1e-30),
        "grad_worst_rel": [(round(r, 6), n) for r, n in worst[:3]],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", type=Path, default=ROOT / "configs" / "pb_s2_v32.json")
    ap.add_argument("--switch", action="append", default=[],
                    help="model attribute toggled False (ref) / True (new)")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--chunks", default="2048,2048")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-activate", action="store_true")
    args = ap.parse_args()
    recipe = json.loads(args.recipe.read_text())
    dev = torch.device("cuda")
    torch.manual_seed(args.seed)
    base = build_model(Config(**recipe["config"]), EfficiencyOptions(**recipe["options"])).to(dev)
    if not args.no_activate:
        activate(base, args.seed)
    chunks = [int(c) for c in args.chunks.split(",")]
    gen = torch.Generator().manual_seed(args.seed + 1)
    ids = torch.randint(9, 265, (args.batch, sum(chunks) + 1), generator=gen).to(dev)
    ids, labels = ids[:, :-1], ids[:, 1:]
    results = {}
    for name, on, amp, backend in (("fp32-ref", False, False, "batched"),
                                   ("fp32-new", True, False, "batched"),
                                   ("bf16-ref", False, True, "compiled"),
                                   ("bf16-new", True, True, "compiled")):
        model = copy.deepcopy(base)
        model.set_segment_backend(backend)
        for switch in args.switch:
            setattr(model, switch, on)
        torch.backends.cuda.matmul.allow_tf32 = amp
        torch.backends.cudnn.allow_tf32 = amp
        results[name] = run(model, ids, labels, chunks, amp)
        del model
        torch.cuda.empty_cache()
    report = {f"{k} vs fp32-ref": compare(results["fp32-ref"], results[k])
              for k in ("fp32-new", "bf16-ref", "bf16-new")}
    report["bf16-new vs bf16-ref"] = compare(results["bf16-ref"], results["bf16-new"])
    print(json.dumps({"switches": args.switch, **report}, indent=1))


if __name__ == "__main__":
    main()
