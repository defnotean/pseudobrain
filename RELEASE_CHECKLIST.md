# Release checklist

## Code
- [x] Port the conversion core from the lab repo into `pseudobrain/` with its dependency closure only
      (`pseudobrain/transplant.py`, `pseudobrain/exact_memory.py`, `pseudobrain/gated_deltanet.py`, `pseudobrain/moe_kernels.py`, `pseudobrain/scan.py`, `pseudobrain/lora.py`, `pseudobrain/recall_tool.py`; no byte/Minecraft code).
- [x] `scripts/convert/`: `convert_token.py` (conversion), `dump_batches.py`, `export_pb27_slim.py`, `make_pb27_nvfp4.py`, `prepare_hf_release.py`.
- [x] `scripts/eval/`: `evaluate.py` (battery, long NLL), `recall_probes.py` (+ exact memory flags), `recall_eval.py`, `compare_long.py`, `vllm_parity.py`.
- [x] `scripts/bench/`: `PB-SWE-Live` harvest/validate/run + `custom_longctx_bench.py`, `custom_longctx_pertoken.sh`, `custom_longctx_extended.sh`, `custom_code_bench.py`.
- [x] One-command reproduction scripts: `make convert`, `make eval`, `make serve`, `make smoke`.
- [x] Remove hard-coded lab-only requirements; config via env/args.
- [x] `pyproject.toml` with pinned deps and CPU smoke test (`make smoke`).

## Weights
- [x] Upload `PseudoBrain-Nex-35B-A3B-NVFP4` (`defnotean1/PseudoBrain-Nex-35B-A3B-NVFP4`) and `PseudoBrain-27B-NVFP4` (`defnotean1/PseudoBrain-27B-NVFP4`) + slim GDN shards (`defnotean1/pb27-checkpoints`) to Hugging Face under `defnotean1`.
- [x] Include `make_pb27_nvfp4.py` script that builds NVFP4 + BF16 GDN compositions from public NVFP4 checkpoints.
- [x] Model cards in `checkpoints/PseudoBrain-Nex-35B-A3B-NVFP4/README.md` and `checkpoints/PseudoBrain-27B-NVFP4/README.md`.

## Receipts (every README number points at a file in `reports/`)
- [x] `lm-eval` battery: stock vs PB-27, same `lm-eval` version, served through vLLM for both (`reports/lmeval/`).
- [x] `PB-SWE-Live`: `stock27_fp8_med60` vs `pb27_fp8mtp_med60` summaries + per-task JSON (`reports/swe_live/`).
- [x] Long context (`8K–1,000,000` tokens) & speed: `reports/pb27/nexc_pt.*`, `nexc_ext.*`, `nexc_fast_verify.*`, `pb27c_pt.*`, `pb27c_ext.*`.
- [x] Medium-hard Python coding benchmark (`33` problems × `2` samples): `reports/pb27/nexc_med.*`, `pb27c_med.*`.

## Write-up
- [x] Technical report (`docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.md` & `docs/REPORT.md`): method, conversion recipe, three-tier exact memory hierarchy, unrotated 2-hop bridge, bit-exact FA2 kernel fast paths, results tables, limitations.
- [x] `README.md` numbers filled in; limitations section honest (what is not better than stock).

## Go public
- [x] License files in place (Apache-2.0 code; weights per base model).
- [x] All commits authored by `defnotean <iangaleonofficial@gmail.com>` only.
