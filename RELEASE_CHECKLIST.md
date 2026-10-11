# Release checklist (private until every box is ticked)

## Code
- [ ] Port the conversion core from the lab repo into `pseudobrain/` with its dependency closure only
      (efficiency/transplant.py, efficiency/exact_memory.py, efficiency/scan.py and what they import; no byte/Minecraft code).
- [ ] `scripts/convert/`: convert_token.py (conversion), dump_batches.py, export_pb27_slim.py, make_pb27_nvfp4.py.
- [ ] `scripts/eval/`: evaluate.py (battery, long NLL), recall_probes.py (+ exact memory flags), recall_eval.py, compare_long.py, vllm_parity.py.
- [ ] `scripts/bench/`: PB-SWE-Live harvest/validate/run + the validated task list used in the paper table.
- [ ] One-command reproduction scripts: `make convert`, `make eval`, `make serve`.
- [ ] Remove every lab-only path (`~/x2`, `~/pb27`, Colab session names, Spark hostnames); config via env/args.
- [ ] pyproject with pinned deps (torch, flash-linear-attention, transformers, vllm) and a smoke test on CPU.

## Weights
- [ ] Upload PB-27 (slim HF export: gdn shard + index that references the stock shards) to HF under defnotean1, private.
- [ ] Upload PB-27 NVFP4 composition (or a script that builds it from the public NVFP4 checkpoint).
- [ ] Model card: base, what changed, conversion data (sources, token counts), evals with receipts, license.

## Receipts (every README number must point at a file in reports/)
- [ ] lm-eval battery: stock vs PB-27, same lm-eval version, served through vLLM for both.
- [ ] PB-SWE-Live: stock27_fp8_med60 vs pb27_fp8mtp_med60 summaries + per-task JSON; Wilson CIs.
- [ ] Speed: speed_smoke per workload for stock FP8, stock NVFP4+DFlash2, PB-27 FP8+MTP, PB-27 NVFP4+DFlash2.
- [ ] Parity: vllm_parity (bf16) and greedy agreement for NVFP4.
- [ ] Long context: recall probes (decode-faithful) at 32k/64k/128k/1M with and without the memory tier, stock, stock+RAG.
- [ ] Memory: KV capacity numbers from the vLLM start-up logs.

## Write-up
- [ ] Technical report (docs/REPORT.md → PDF): method, conversion recipe, results tables, limitations, related work
      (Qwen3-Next, Kimi Linear, Quest, InfLLM, DSA/QSA, DFlash).
- [ ] 60-second demo recording: 1M-token session on one Spark with the memory monitor visible.
- [ ] README numbers filled in; limitations section honest (what is not better than stock).

## Go public
- [ ] License files in place (Apache-2.0 code; weights per base model).
- [ ] All commits authored by defnotean only.
- [ ] Flip repo + HF repos to public in the same hour; post thread with the tables and the reproduction command.
