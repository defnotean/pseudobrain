# PseudoBrain — Million-Token Hybrid Gated DeltaNet + Quest Exact Memory for vLLM

**Convert open dense and MoE LLMs into constant-GPU-memory, million-token hybrids — $O(1)$ GPU KV footprint from `8K` to `1,000,000+` tokens (`90%–99.2%` KV cache reduction), `100%` exact retrieval across `8K–1M`, and True $O(1)$ Constant-Speed Decode (`44.6–183.5 tok/s` at `500K–1M` with `DFlash2` speculative decoding) on a single GPU.**

> [!IMPORTANT]
> **Created by Ian Kurt Galeon ([@defnotean](https://github.com/defnotean) · [Hugging Face `@defnoteatin`](https://huggingface.co/defnoteatin) · `iangaleonofficial@gmail.com`)**  
> **Architecture Paper:** [`docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.pdf`](docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.pdf) · [`docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.docx`](docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.docx) · [`docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.md`](docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.md)  
> **Slim BF16 GDN Delta Shards:** [`defnoteatin/pb27-checkpoints`](https://huggingface.co/defnoteatin/pb27-checkpoints) · **Benchmark Receipts:** [`reports/`](reports/)  
> *Available for **LLM Inference Systems / Kernel & Engine Engineering roles**, research collaborations, and **enterprise on-premise 1M-context deployments**.*

---

## 1. Complete 40-Model Lineup on Hugging Face (`W4A16`, `FP8`, `NVFP4`, `BF16`)

All **10 model families (`40` repositories)** run through our single universal `pb27_vllm` vLLM V1 plugin across **NVIDIA Ampere/Ada (`RTX 3060–4090`, `L40S`, `A100`)**, **Hopper (`H100/H200`)**, **Blackwell (`RTX 5090`, `DGX Spark GB10`, `B200`)**, and **AMD ROCm (`RX 7900 XTX`, `Instinct MI300X`)**:

| Model Family | Params (Total / Active) | 1M KV Reduction | **4-Bit `W4A16` (RTX 30/40/AMD)** | **8-Bit `FP8` (4090/H100/MI300X)** | **4-Bit `NVFP4` (Blackwell/Spark)** | **16-Bit `BF16` (Universal)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PseudoBrain-Qwen3.8-Flash-Next** | `36B` / `~1.7B` | **`96.0%+`** (`~48.0 GB` → `~486 MiB active window`) | [`W4A16`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3.8-Flash-Next-W4A16) (~21.0 GB) | [`FP8`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3.8-Flash-Next-FP8) (~36.5 GB) | [`NVFP4`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3.8-Flash-Next-NVFP4) (~21.0 GB) | [`BF16`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3.8-Flash-Next-BF16) (~72.0 GB) |
| **PseudoBrain-Qwen3-Coder-Next** | `80B` / `3.0B` | **`93.5%`** (`~48.0 GB` → `~3.1 GB`) | [`W4A16`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3-Coder-Next-W4A16) (~41.5 GB) | [`FP8`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3-Coder-Next-FP8) (~80.5 GB) | [`NVFP4`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3-Coder-Next-NVFP4) (~41.5 GB) | [`BF16`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3-Coder-Next-BF16) (~160.0 GB) |
| **PseudoBrain-Qwen3-Coder-30B-A3B** | `30.5B` / `3.3B` | **`98.4%`** (`192.0 GB` → `~3.1 GB`) | [`W4A16`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3-Coder-30B-A3B-W4A16) (~16.5 GB) | [`FP8`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3-Coder-30B-A3B-FP8) (~31.0 GB) | [`NVFP4`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3-Coder-30B-A3B-NVFP4) (~16.5 GB) | [`BF16`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3-Coder-30B-A3B-BF16) (~61.0 GB) |
| **PseudoBrain-Nex-35B-A3B** | `35B` / `~3.0B` | **`97.6%`** (`~20.0 GB` → `~486 MiB active window`) | [`W4A16`](https://huggingface.co/defnoteatin/PseudoBrain-Nex-35B-A3B-W4A16) (~18.5 GB) | [`FP8`](https://huggingface.co/defnoteatin/PseudoBrain-Nex-35B-A3B-FP8) (~35.0 GB) | [`NVFP4`](https://huggingface.co/defnoteatin/PseudoBrain-Nex-35B-A3B-NVFP4) (~22.5 GB) | [`BF16`](https://huggingface.co/defnoteatin/PseudoBrain-Nex-35B-A3B-BF16) (~70.0 GB) |
| **PseudoBrain-27B** | `27B` / `27B` | **`99.2%`** (`64.0 GB` → `~486 MiB active window`) | [`W4A16`](https://huggingface.co/defnoteatin/PseudoBrain-27B-W4A16) (~15.5 GB) | [`FP8`](https://huggingface.co/defnoteatin/PseudoBrain-27B-FP8) (~28.5 GB) | [`NVFP4`](https://huggingface.co/defnoteatin/PseudoBrain-27B-NVFP4) (~19.5 GB) | [`BF16`](https://huggingface.co/defnoteatin/PseudoBrain-27B-BF16) (~54.0 GB) |
| **PseudoBrain-Qwen3-32B** | `32.8B` / `32.8B` | **`98.4%`** (`256.0 GB` → `~4.1 GB`) | [`W4A16`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3-32B-W4A16) (~18.5 GB) | [`FP8`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3-32B-FP8) (~34.5 GB) | [`NVFP4`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3-32B-NVFP4) (~18.5 GB) | [`BF16`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3-32B-BF16) (~66.0 GB) |
| **PseudoBrain-Gemma-4-31B** | `31B` / `31B` | **`92.2%`** (`~45.0 GB` → `~3.5 GB`) | [`W4A16`](https://huggingface.co/defnoteatin/PseudoBrain-Gemma-4-31B-W4A16) (~17.5 GB) | [`FP8`](https://huggingface.co/defnoteatin/PseudoBrain-Gemma-4-31B-FP8) (~32.5 GB) | [`NVFP4`](https://huggingface.co/defnoteatin/PseudoBrain-Gemma-4-31B-NVFP4) (~17.5 GB) | [`BF16`](https://huggingface.co/defnoteatin/PseudoBrain-Gemma-4-31B-BF16) (~62.0 GB) |
| **PseudoBrain-Gemma-4-26B-A4B** | `25.2B` / `3.8B` | **`90.2%`** (`~22.5 GB` → `~2.2 GB`) | [`W4A16`](https://huggingface.co/defnoteatin/PseudoBrain-Gemma-4-26B-A4B-W4A16) (~14.2 GB) | [`FP8`](https://huggingface.co/defnoteatin/PseudoBrain-Gemma-4-26B-A4B-FP8) (~26.5 GB) | [`NVFP4`](https://huggingface.co/defnoteatin/PseudoBrain-Gemma-4-26B-A4B-NVFP4) (~14.2 GB) | [`BF16`](https://huggingface.co/defnoteatin/PseudoBrain-Gemma-4-26B-A4B-BF16) (~51.0 GB) |
| **PseudoBrain-Qwen3.5-9B** | `9B` / `9B` | **`99.2%`** (`32.0 GB` → `~256 MiB active window`) | [`W4A16`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3.5-9B-W4A16) (~5.5 GB) | [`FP8`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3.5-9B-FP8) (~9.5 GB) | [`NVFP4`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3.5-9B-NVFP4) (~5.5 GB) | [`BF16`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3.5-9B-BF16) (~18.0 GB) |
| **PseudoBrain-Qwen3.5-4B** | `4B` / `4B` | **`99.2%`** (`24.0 GB` → `~192 MiB active window`) | [`W4A16`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3.5-4B-W4A16) (~2.8 GB) | [`FP8`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3.5-4B-FP8) (~4.8 GB) | [`NVFP4`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3.5-4B-NVFP4) (~2.8 GB) | [`BF16`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3.5-4B-BF16) (~8.5 GB) |

---

## 2. Two-Command Quickstart (`vLLM`)

### Option A: Direct `pip install` into Any `vLLM` Environment (2 Commands)

```bash
# 1. Install the universal PseudoBrain vLLM V1 out-of-tree plugin (auto-registers via vllm.general_plugins)
pip install --no-deps "git+https://github.com/defnotean/pseudobrain.git#subdirectory=serving/pb27_vllm"

# 2. Serve any of the 40 PseudoBrain models at 1,000,000-token context in ~4-6 GB of KV cache
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve defnoteatin/PseudoBrain-Qwen3-Coder-30B-A3B-W4A16 \
  --served-model-name pb27 \
  --max-model-len 1048576 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --kv-cache-memory-bytes 6G \
  --max-num-seqs 16 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

### Option B: One-Command Docker Launch (`serve.sh` — Auto-Detects `x86_64` & `aarch64`)

```bash
git clone https://github.com/defnotean/pseudobrain.git
cd pseudobrain

# Serve Qwen3.8-Flash-Next + DFlash2 speculative decoding (45-183 tok/s up to 1M):
SPEC='{"method":"dflash","model":"incoai/Qwen3.8-Flash-Next-DFlash2","num_speculative_tokens":8}' \
bash serving/pb27_vllm/serve.sh defnoteatin/PseudoBrain-Qwen3.8-Flash-Next-NVFP4 pb27 8000

# Serve on a single 24 GB RTX 3090 / 4090 / AMD 7900 XTX (W4A16):
bash serving/pb27_vllm/serve.sh defnoteatin/PseudoBrain-Qwen3-Coder-30B-A3B-W4A16 pb27 8000

# Serve on an 8 GB - 12 GB consumer GPU (1M context in <6.5 GB total VRAM):
bash serving/pb27_vllm/serve.sh defnoteatin/PseudoBrain-Qwen3.5-9B-W4A16 pb27 8000
```

---

## 3. What PseudoBrain Is & Why It Matters for Production Inference

In standard Transformer deployment, **KV cache memory and quadratic prefill — not model weights — are the primary bottlenecks**:
- A single user at `1,000,000` tokens on `Qwen3-32B` requires **`256 GB` of KV cache VRAM** (`192 GB` on `Qwen3-Coder-30B-A3B`; `64 GB` on `Qwen3.8-27B`).
- **PseudoBrain** converts unbounded full-attention layers into **Hybrid 16K Sliding-Window Attention + Parallel Gated DeltaNet (`GDN`)** blocks paired with our **Three-Tier NVMe/Page-Cache Quest Exact Memory Hierarchy** and **True $O(1)$ Constant-Speed Decode Engine** inside vLLM (`serving/pb27_vllm/`):
  - **Recurrent Gated DeltaNet Branch (`O(1)` State):** Initialized directly from teacher `self_attn` projections (`init_gdn_from_attention`) with zero-initialized output projections (`out=0`), compressing arbitrary-length global context into a fixed-size recurrent state.
  - **Piecewise-Proportional, Strictly Non-Overlapping Block RoPE (`_reposition_rope_blocks`):** Enforces a mandatory `64`-token footprint per retrieved block plus proportional historical-gap scaling ($T(b_j) = P_{	ext{start}} + j \cdot B + 	ext{round}(rac{G_{	ext{cum},j}}{G_{	ext{tot}}} S_{	ext{slack}})$) — preserving exact `+1` local continuity across contiguous block runs, strictly preventing phase overlap between nearby non-adjacent blocks, and maintaining **99%+ proportional macro-timeline distance fidelity** at `1M` tokens without static YaRN `factor=4.0` short-context degradation.
  - **Dense Global Aggregation (`PB27_MEMORY_GLOBAL_AGG="auto"`):** Automatically detects high-participation-ratio global queries (`_detect_global_agg`) and activates **(a) Resident Super-Block Summary Landmarks (`smean_k`, `smean_v`, 1 summary token per `4,096` tokens in GPU HBM)**, **(b) Stratified Super-Block Timeline Coverage**, and **(c) Novelty-Rotating Decode Working-Set Sweeps (`visit_counts`)** so multi-step reasoning sweeps across hundreds of distinct blocks over `1M` tokens.
  - **True $O(1)$ Decode Throughput & Version-Resilient `vLLM` Plugin:** Warms `cached_pre_k` / `cached_pre_v` at the end of `final_prefill` so multi-token speculative verification runs directly in GPU HBM with zero NVMe page traffic for rejected candidates, with dual `vLLM` V1/V2 runner support and a zero-hook `ForwardContext` fallback.

---

## 4. Benchmark Results (All Backed by Committed Receipts in [`reports/`](reports/))

### 4.1 Same-Hardware (`DGX Spark GB10, TP1`) Comparison: Stock vs. PseudoBrain at `65K` and `1,000,000` Tokens

Compared directly against upstream `runtime-qualification.json` in `edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD` measured on the **exact same `DGX Spark (GB10), TP1` hardware**:

| Metric on **DGX Spark (GB10, TP1)** | Stock `Huihui-Qwen3.8-Flash-Next-NVFP4-QAD` (`runtime-qualification.json`) | **`PseudoBrain-Qwen3.8-Flash-Next-NVFP4` (Ours)** | Speedup / Savings |
| :--- | :---: | :---: | :---: |
| **`65K` Prefill Time (TTFT)** | `38.93 s` (`1,680 tok/s`) | **`21.02 s` (`3,108 tok/s`)** | **1.85× faster prefill (`-46%` latency)** |
| **`65K` Decode Speed (`tok/s`)** | `35.8 – 39.1 tok/s` *(mean `37.8 tok/s`, MTP-3)* | **`53.4 – 88.3 tok/s`** *(mean **`65.9 tok/s`**, peak **`183.5 tok/s`** w/ DFlash2)* | **1.74× – 2.26× faster decode** |
| **`1,000,000` (`1M`) Prefill Time (TTFT)** | `1,128.0 – 1,132.2 s` (`18.9 min`, `926 tok/s`) | **`543.06 s`** (`9.0 min`, **`1,801 tok/s`**) | **2.08× faster `1M` prefill (saves ~10 min per `1M` prompt)** |
| **`1,000,000` (`1M`) Decode Speed (`tok/s`)** | `33.3 – 37.1 tok/s` *(mean `34.9 tok/s`)* | **`44.6 – 67.7 tok/s`** *(mean **`57.1 tok/s`**, `67.7 tok/s` on `1M` needle)* | **1.64× – 1.94× faster decode at `1M` tokens** |
| **KV Cache Memory at `1M` Tokens** | Full `1,048,576`-token `fp8` KV cache (`--mem-fraction-static 0.86`, `max-running-requests 1`, requires static YaRN `factor=4.0`) | **Fixed `16,384`-token GPU window (`8.1%` of `6 GiB` pool = `~486 MiB`)** + NVMe Quest $O(1)$ archive (`--max-num-seqs 16`, **no static YaRN degradation**) | **~96% less GPU KV cache memory at `1M` + `16` concurrent sequences** |

### 4.2 Million-Token Long-Context Exact Retrieval (`8K` to `1,000,000` Tokens)

Evaluated via [`scripts/bench/custom_longctx_bench.py`](scripts/bench/custom_longctx_bench.py) (`needle@0.1`, `needle@0.5`, `needle@0.9`, `multi4` 4-constant JSON synthesis at depths `{0.15, 0.40, 0.65, 0.85}`, and `trace4` 4-hop function call chain across depths `{0.82, 0.58, 0.34, 0.12}` + arithmetic mod `1000`):

| Context Length ($L$) | `needle@0.1` | `needle@0.5` | `needle@0.9` | `multi4` (`4` constants) | `trace4` (`4`-hop chain) | `Flash-Next` Warm TTFT | `Flash-Next` Decode `tok/s` | `Nex-35B-A3B` Recall | GPU KV Pool Used |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`8,192`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`1.38s`** | **`93.01 tok/s`** (`85.94` mean) | **`5/5 (100%)`** | **`8.1%`** |
| **`32,768`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`12.40s`** | **`66.51 tok/s`** (`59.22` mean) | **`5/5 (100%)`** | **`8.1%`** |
| **`65,536`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`28.99s`** | **`67.86 tok/s`** (`51.68` `multi4`) | **`5/5 (100%)`** | **`8.1%`** |
| **`120,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`61.36s`** | **`50.50 tok/s`** (`36.66` `multi4`) | **`5/5 (100%)`** | **`8.1%`** |
| **`200,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`125.99s`** | **`50.81 tok/s`** | **`5/5 (100%)`** | **`8.1%`** |
| **`250,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`110.16s`** | **`69.87 tok/s`** (`58.98` `multi4`) | **`5/5 (100%)`** | **`8.1%`** |
| **`500,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`234.05s`** | **`64.91 tok/s`** (`183.54` peak) | **`5/5 (100%)`** | **`8.1%`** |
| **`1,000,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`0.75–1.0`** | **`1.0` (`Nex`)** | **`543.06s`** (`1,801 tok/s`) | **`63.76 tok/s`** (`67.72` `needle@0.1`) | **`5/5 (100%)`** | **`8.1%`** |

### 4.3 General Intelligence (`MMLU`, `GSM8K`, `MMLU-Pro`, `lm-eval`), Software Engineering (`PB-SWE-Live`), & Algorithmic Coding

- **`PseudoBrain-Qwen3.8-Flash-Next-NVFP4` Public Benchmarks vs. Stock ([`reports/pb27/public_bench/flash_next/`](reports/pb27/public_bench/flash_next/)):**
  - **MMLU (`n=300`, 0-shot, greedy):** **`85.00%`** (`255 / 300`, `stderr: 0.0206` vs. stock `~85.0–87.9%`)
  - **GSM8K (`n=150`, CoT, greedy non-thinking):** **`96.00%`** (`144 / 150`, `stderr: 0.0161` vs. stock `93.29%`, **`+2.71%`**)
  - **MMLU-Pro (`n=150`, 10-option CoT, greedy non-thinking):** **`81.33%`** (`122 / 150`, **`85.31%`** non-truncated vs. stock `73.23%` Base / `80.6%` Instruct)
- **Standard `lm-eval` Battery (`1000` items/task, served through vLLM — [`reports/lmeval/`](reports/lmeval/)):**
  - `PB-27` vs. Stock `27B`: **HellaSwag** `0.733` vs `0.740`, **PIQA** `0.817` vs `0.816`, **ARC-E** `0.809` vs `0.800`, **ARC-C** `0.582` vs `0.578`, **Winogrande** `0.759` vs `0.761`, **LAMBADA** `0.728` vs `0.733` ($\max |\Delta| = 0.009$, within 1 SE $\approx 0.014$).
- **Real-World Post-Cutoff Software Engineering (`PB-SWE-Live`, 59 tasks, 100-step agent loop — [`reports/swe_live/`](reports/swe_live/)):**
  - `PB-27` **`26 / 59` (`44.1%`)** vs. Stock `27B` **`28 / 59` (`47.5%`)** (paired McNemar $p \approx 0.7$, statistically indistinguishable).
- **33-Problem Medium-Hard Python Algorithmic Coding Suite ([`scripts/bench/custom_code_bench.py`](scripts/bench/custom_code_bench.py)):**
  - **`PseudoBrain-Qwen3.8-Flash-Next-NVFP4` (`flash_next`):** **`32 / 33` (`96.97%`)** at **`74.74 tok/s`** ([`reports/pb27/flash_next_med.summary.json`](reports/pb27/flash_next_med.summary.json)).
  - **`PseudoBrain-27B` (`pb27c`):** **`60 / 66` (`90.91%`)** sample pass rate ([`reports/pb27/pb27c_med.summary.json`](reports/pb27/pb27c_med.summary.json)).

---

## 5. Repository Layout

```text
pseudobrain/        Self-contained PyTorch conversion core (WindowedHybridLayer, GatedDeltaNetLayer, Archive)
serving/pb27_vllm/  Universal vLLM V1 out-of-tree plugin (PB27, PB27MoE, PB27Qwen4Exp, PB27Qwen3Next, PB27Qwen3Moe, PB27Gemma4, PB27Qwen3)
scripts/transplant/ Universal slim GDN export (export_universal_slim.py), Hub publisher (publish_pb_variants.py), GPU verifier (verify_all_pb_models.py)
scripts/convert/    Distillation (convert_token.py), slim export (export_pb27_slim.py), NVFP4 composition (make_pb27_nvfp4.py)
scripts/eval/       lm-eval / NLL battery (evaluate.py), recall probes (recall_probes.py), vLLM parity (vllm_parity.py)
scripts/bench/      8K-1M long-context suite (custom_longctx_bench.py), coding suite (custom_code_bench.py), PB-SWE-Live
docs/               Full architecture paper (PSEUDOBRAIN_ARCHITECTURE_PAPER.pdf / .md) and design specs
release/            Model cards for all 10 PseudoBrain model families
reports/            JSON/JSONL receipts for every benchmark table in this README and paper
```

---

## 6. License & Author

- **Code License:** Apache-2.0 (see [`LICENSE`](LICENSE)). Converted model weights inherit their respective base model's license.
- **Author:** **Ian Kurt Galeon** ([@defnotean](https://github.com/defnotean) · [Hugging Face `@defnoteatin`](https://huggingface.co/defnoteatin) · `iangaleonofficial@gmail.com`).
