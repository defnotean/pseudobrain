# PseudoBrain

**Convert a stock dense LLM into a bounded-memory, long-context, fast hybrid — same intelligence, flat KV memory at any length.**

PseudoBrain takes an existing open model (first release: [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)) and
converts it, with ~25M tokens of distillation on one GPU, into a model whose attention working set never grows:
a fixed 16k-token window plus an exact, retrieved memory of everything older. Intelligence is kept (same benchmarks,
same real-world coding score), decode is several times faster on a single desktop GPU, and context length stops being
a property of the model.

> Status: private pre-release. Every number below has a receipt in `reports/`; the long-context ladder (1M+) is the open gate.

## What it is

| | Stock Qwen3.8-27B | PB-27 (this release) |
|---|---|---|
| Attention layers | 16 full attention (KV grows with context) | 16 × [16k sliding window **+** parallel Gated DeltaNet branch] |
| Everything else | 48 Gated DeltaNet + MLPs | unchanged |
| KV memory at N tokens | O(N) | O(16k) + retrieved blocks (constant) |
| Memory beyond the window | — | exact memory tier: evicted keys/values archived in 64-token blocks, top-k retrieved per step by Quest bounds + exact rescoring |
| Trained parameters | — | 1.85B (the 16 GDN branches); all stock weights untouched |
| Conversion cost | — | ~780 steps × 32k tokens, one 96 GB GPU, ≈ 1 day |

## Results (to be finalised from receipts)

- **Intelligence (served, like for like — both models NVFP4 + DFlash2 through vLLM, lm-eval 1000 items/task;
  `reports/lmeval/`):** HellaSwag 0.733 vs 0.740, PIQA 0.817 vs 0.816, ARC-E 0.809 vs 0.800, ARC-C 0.582 vs 0.578,
  Winogrande 0.759 vs 0.761, LAMBADA 0.728 vs 0.733 (max |Δ| 0.009, SE ≈ 0.014).
- **Real coding (PB-SWE-Live, 59 fail-to-pass tasks from PRs merged after the model's cutoff, 100-step agent loop;
  `reports/swe_live/`):** PB-27 26/59 vs stock 28/59; paired: 24 both, 5 stock-only, 3 PB-27-only (McNemar p ≈ 0.7).
- **Speed (single DGX Spark GB10, NVFP4 + DFlash2 drafter, single stream):** PB-27 code 58.7 tok/s, math 48.3,
  explanation 19.9, prose 17.1 — stock in the same config 84 / 53 / 24 / 19; the model alone 8–9; PB-27 FP8 + MTP 11–18.
- **Memory:** attention KV per request is flat (16k window per layer) instead of growing with context; the same GPU
  pool holds 1.68M cached tokens for PB-27 vs 1.04M for stock (NVFP4), 3.3M vs 0.88M in bf16.
- **Long context (exact-memory tier, PyTorch path, 12 items/cell, PB-27 vs stock):** passkey 1.00/1.00/1.00 vs
  1.00/1.00/1.00 at 32k/64k/128k; multihop 0.92/0.92/0.00 vs 0.67/0.67/0.17; copy 0.33/0.33/0.33 vs 0.58/0.33/0.33.
  The tier inside the vLLM serve is in progress (`docs/MEMORY_SERVE.md`); the 1M+ ladder follows it.

## Repository layout

```
pseudobrain/        conversion core: windowed-hybrid layers, exact memory tier, training (to be ported from the lab repo)
scripts/convert/    conversion, export (slim HF export, NVFP4 composition)
scripts/eval/       battery, long-NLL, recall probes, vLLM parity
scripts/bench/      PB-SWE-Live: harvest, validate, run (real post-cutoff PR tasks)
serving/pb27_vllm/  out-of-tree vLLM plug-in (vLLM v0.31.0) + serve script
docs/               architecture, vLLM plug-in design
checkpoints/        pointers to the released weights on Hugging Face
reports/            receipts for every number in this README
```

## Quick start (serving)

```bash
# vLLM v0.31.0 image, plug-in installed at start-up; model dir = slim export or NVFP4 composition
serving/pb27_vllm/serve.sh /path/to/pb27 pb27 8000
# fast: NVFP4 target + DFlash2 drafter
SPEC='{"method":"dflash","model":"z-lab/Qwen3.8-27B-DFlash2","num_speculative_tokens":16}' PB27_EAGER=0 \
  serving/pb27_vllm/serve.sh /path/to/pb27_nvfp4 pb27 8000
```

## License

Code: Apache-2.0 (see `LICENSE`). Converted weights inherit the base model's license (Qwen3.8-27B: Apache-2.0).

## Author

Ian Kurt Galeon ([@defnotean](https://github.com/defnotean)).
