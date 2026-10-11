# PseudoBrain

**Convert a stock dense LLM into a bounded-memory, long-context, fast hybrid — same intelligence, flat KV memory at any length.**

PseudoBrain takes an existing open model (first release: [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)) and
converts it, with ~25M tokens of distillation on one GPU, into a model whose attention working set never grows:
a fixed 16k-token window plus an exact, retrieved memory of everything older. Intelligence is kept (same benchmarks,
same real-world coding score), decode is several times faster on a single desktop GPU, and context length stops being
a property of the model.

> Status: private pre-release. Numbers below are filled in from the receipts in `reports/` before the public release.

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

- **Intelligence:** lm-eval battery within stock's 95% CI; PB-SWE-Live (fail-to-pass tasks from PRs merged after the
  model's cutoff): PB-27 __/59 vs stock 28/59.
- **Speed (single DGX Spark, GB10):** stock FP8 __ tok/s → PB-27 NVFP4 + DFlash2 drafter __ tok/s, identical greedy output.
- **Long context (exact rate, PB-27 vs stock):** passkey / multihop / copy at 32k, 64k, 128k, 1M: __.
- **Memory:** KV cache capacity 3.3M tokens vs 0.88M for stock in the same GPU memory.

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
