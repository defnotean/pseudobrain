---
license: apache-2.0
library_name: vllm
pipeline_tag: text-generation
tags:
  - pseudobrain
  - qwen3.8-flash-next
  - qwen4-exp
  - abliterated
  - moe
  - gated-deltanet
  - dflash2
  - speculative-decoding
  - long-context
  - 1m-context
  - constant-memory
  - nvfp4
  - vllm
---

# PseudoBrain-Qwen3.8-Flash-Next-NVFP4 (`flash_next` / `PB27Qwen4ExpForConditionalGeneration`)

**GitHub Repository & vLLM Plugin:** [https://github.com/defnotean/pseudobrain](https://github.com/defnotean/pseudobrain)  
**Hugging Face Editions (Pick Your GPU):**
- **W4A16 (4-bit AWQ/Marlin — 18 Shards PLE-Pruned for RTX 3090 / 4090 & AMD 7900 XTX / ROCm):** [`defnoteatin/PseudoBrain-Qwen3.8-Flash-Next-W4A16`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3.8-Flash-Next-W4A16)
- **FP8 (RTX 4090 / H100 / AMD MI300X):** [`defnoteatin/PseudoBrain-Qwen3.8-Flash-Next-FP8`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3.8-Flash-Next-FP8)
- **NVFP4 (DGX Spark GB10 / RTX 5090 / Blackwell):** [`defnoteatin/PseudoBrain-Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3.8-Flash-Next-NVFP4)
- **BF16 (Universal NVIDIA & AMD ROCm):** [`defnoteatin/PseudoBrain-Qwen3.8-Flash-Next-BF16`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3.8-Flash-Next-BF16)

**Speculative Draft Model (`DFlash2`):** [`incoai/Qwen3.8-Flash-Next-DFlash2`](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)  
**Companion Model Families:** [`PseudoBrain-Nex-35B-A3B`](https://huggingface.co/defnoteatin/PseudoBrain-Nex-35B-A3B-NVFP4) | [`PseudoBrain-27B`](https://huggingface.co/defnoteatin/PseudoBrain-27B-NVFP4) | [`PseudoBrain-Nanbeige4.3-33B`](https://huggingface.co/defnoteatin/PseudoBrain-Nanbeige4.3-33B-W4A16)  
**Architecture Paper:** [`docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.md`](https://github.com/defnotean/pseudobrain/blob/main/docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.md)

## Overview

**PseudoBrain-Qwen3.8-Flash-Next-NVFP4 (`flash_next`)** is an ultra-sparse Mixture-of-Experts (`48` layers, `512` routed experts with `top-10` active per token + `1` shared expert, Per-Layer Embeddings / PLE) constant-GPU-memory hybrid long-context conversion of `edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD` (`huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated`, `Qwen4ExpForConditionalGeneration`). All `12` full-attention layers (`[3, 7, 11, ..., 47]`) are converted into **Hybrid Windowed-Attention + Parallel Gated DeltaNet** blocks (`PB27Qwen4ExpDecoderLayer`) alongside `36` native Gated DeltaNet layers, PseudoBrain's **Three-Tier NVMe/Page-Cache Exact Memory Archive** with **True $O(1)$ Constant-Speed Decode**, and native **`DFlash2` multi-layer speculative decoding** (`5–8` speculative draft tokens per step).

- **True $O(1)$ Constant-Speed Decode up to `1,000,000` Tokens (`44.6–183.5 tok/s` at `500K–1M`):** Decouples the context-scaled prefill retrieval budget (`PB27_MEMORY_SCALE`) from the constant decode budget (`PB27_MEMORY_DECODE_BLOCKS=64`, `PB27_MEMORY_DECODE_RERANK=0`, `PB27_MEMORY_DECODE_STRIDE=64`), warms the GPU LSE-merge cache (`cached_pre_k` / `cached_pre_v`) at the end of `final_prefill`, and uses **2-Level Hierarchical `bf16` Quest Super-Block Bounds** (`64` blocks = `4,096` tokens per super-block) for $O(1)$ (`~0.05 ms`) bound scoring.
- **Priority-Aware RoPE Sink Repositioning (`_reposition_rope_sinks`):** Dynamically re-rotates retrieved archive blocks into contiguous local RoPE positions just before the `16K` sliding window, reaching **`1,000,000` tokens without static YaRN `factor=4.0` short-context degradation**.
- **$O(1)$ Constant GPU KV Cache (`~28 GiB` Total GPU Footprint up to `1,000,000` Tokens):** Uses only `8.1%` (`~486 MiB`) of a fixed `6 GiB` KV cache pool even at `1,000,000` tokens on a single NVIDIA GB10 GPU.

## Stock `Qwen3.8-Flash-Next` vs. `PseudoBrain-Qwen3.8-Flash-Next-NVFP4` Comparison

### 1. Capability & Reasoning Benchmarks

| Benchmark | Stock `Qwen3.8-Flash-Next` (Official / Base / QAD) | **`PseudoBrain-Qwen3.8-Flash-Next-NVFP4` (Ours)** | Comparison |
| :--- | :---: | :---: | :---: |
| **MMLU** (`n=300`, 57 subjects) | `~85.0% – 87.9%` | **`85.00%`** (`255/300`, `±2.06%` stderr) | Within statistical stderr of stock |
| **MMLU-Pro** (`n=150`, 10-choice college reasoning) | `73.23%` *(Base)* / `80.6%` *(3-Next Instruct)* | **`81.33%`** (`122/150` raw) / **`85.31%`** *(non-truncated)* | **Matches / +0.7% to +4.7% over stock Instruct** |
| **GSM8K** (`n=150`, strict match) | `93.29%` *(Official tech report)* | **`96.00%`** (`144/150`, `±1.61%` stderr) | **+2.71% above official report** |
| **Coding** (`LiveCodeBench v6` / `custom_code_bench` 33-problem suite) | `91.9%` *(LCB v6)* / `82.9%` *(Terminal-Bench 2.1 NVFP4)* | **`96.97%`** (`32/33` exact pass at **`74.74 tok/s`**) | Zero degradation from GDN + NVMe PLE |
| **Long-Context Recall (`AA-LCR` vs. `8K–1M` NIAH/Multi4/Trace4)** | `71.9%` *(FP8)* / `74.1%` *(NVFP4)* / `79.4%` *(LIL QAD)* | **`100.0%` (`35/35`)** at `8K–500K`<br>**`100.0%` (`3/3`)** single-needle at **`1M`**<br>**`75.0%` (`3/4`)** `multi4` at **`1M`** | **Exact 100% single-needle recall all the way to `1,000,000` tokens** |

### 2. Same-Hardware (`DGX Spark GB10, TP1`) Speed & Memory at `65K` and `1,000,000` Tokens

Compared directly against the upstream `runtime-qualification.json` in `edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD` measured on the **exact same `DGX Spark (GB10), TP1` hardware**:

| Metric on **DGX Spark (GB10, TP1)** | Stock `Huihui-Qwen3.8-Flash-Next-NVFP4-QAD` (`runtime-qualification.json`) | **`PseudoBrain-Qwen3.8-Flash-Next-NVFP4` (Ours)** | Speedup / Savings |
| :--- | :---: | :---: | :---: |
| **`65K` Prefill Time (TTFT)** | `38.93 s` (`1,680 tok/s`) | **`21.02 s` (`3,108 tok/s`)** | **1.85× faster prefill (`-46%` latency)** |
| **`65K` Decode Speed (`tok/s`)** | `35.8 – 39.1 tok/s` *(mean `37.8 tok/s`, MTP-3)* | **`53.4 – 88.3 tok/s`** *(mean **`65.9 tok/s`**, peak **`183.5 tok/s`** w/ DFlash2)* | **1.74× – 2.26× faster decode** |
| **`1,000,000` (`1M`) Prefill Time (TTFT)** | `1,128.0 – 1,132.2 s` (`18.9 min`, `926 tok/s`) | **`543.06 s`** (`9.0 min`, **`1,801 tok/s`**) | **2.08× faster `1M` prefill (saves ~10 min per `1M` prompt)** |
| **`1,000,000` (`1M`) Decode Speed (`tok/s`)** | `33.3 – 37.1 tok/s` *(mean `34.9 tok/s`)* | **`44.6 – 67.7 tok/s`** *(mean **`57.1 tok/s`**, `67.7 tok/s` on `1M` needle)* | **1.64× – 1.94× faster decode at `1M` tokens** |
| **KV Cache Memory at `1M` Tokens** | Full `1,048,576`-token `fp8` KV cache (`--mem-fraction-static 0.86`, `max-running-requests 1`, requires static YaRN `factor=4.0`) | **Fixed `16,384`-token GPU window (`8.1%` of `6 GiB` pool = `~486 MiB`)** + NVMe Quest $O(1)$ archive (`--max-num-seqs 16`, **no static YaRN degradation**) | **~96% less GPU KV cache memory at `1M` + `16` concurrent sequences** |

## Long-Context Exact Retrieval & $O(1)$ Decode Benchmark (`8K` to `1,000,000` Tokens)

| Context Length ($L$) | `needle@0.1` | `needle@0.5` | `needle@0.9` | `multi4` (`4` constants) | `trace4` (`4`-hop chain) | Warm TTFT | `trace4` / Peak Decode `tok/s` | KV Pool Used |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`8,192`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`1.38s`** | **`93.01 tok/s`** (`79.38` `multi4`) | `8.1%` |
| **`32,768`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`12.40s`** | **`66.51 tok/s`** (`51.89` `multi4`) | `8.1%` |
| **`65,536`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`28.99s`** | **`67.86 tok/s`** (`51.68` `multi4`) | `8.1%` |
| **`120,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`61.36s`** | **`50.50 tok/s`** (`36.66` `multi4`) | `8.1%` |
| **`200,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`125.99s`** | **`50.81 tok/s`** | `8.1%` |
| **`250,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`110.16s`** | **`69.87 tok/s`** (`58.98` `multi4`) | `8.1%` |
| **`500,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`234.05s`** | **`64.91 tok/s`** (`183.54` `needle@0.1`, `47.69` `multi4`) | `8.1%` |
| **`1,000,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`0.75` (`3/4`)** | `0.0` | **`543.06s`** | **`63.76 tok/s`** (`67.72` `needle@0.1`, `57.1` needle mean, `44.60` overall mean) | `8.1%` |

## Quickstart with vLLM + `DFlash2` Speculative Decoding

1. Clone the PseudoBrain repository and install the `pb27_vllm` out-of-tree vLLM plugin:
   ```bash
   git clone https://github.com/defnotean/pseudobrain.git
   cd pseudobrain
   pip install -e serving/pb27_vllm
   ```

2. Serve `PseudoBrain-Qwen3.8-Flash-Next-NVFP4` with `DFlash2` speculative decoding and $O(1)$ constant-speed decode up to `1M`:
   ```bash
   MODEL_KIND=flash_next MAXLEN=1056768 bash serving/pb27_vllm/start_pb27_studio.sh
   ```
   Or directly via `serve.sh`:
   ```bash
   PB27_MEMORY_PREFILL=pt \
   PB27_MEMORY_DECODE_STRIDE=64 \
   PB27_MEMORY_DECODE_BLOCKS=64 \
   PB27_MEMORY_DECODE_RERANK=0 \
   PB27_MEMORY_SUPERBLOCK=64 \
   PB27_SPECULATIVE_CONFIG='{"method":"dflash","model":"incoai/Qwen3.8-Flash-Next-DFlash2","num_speculative_tokens":8}' \
   bash serving/pb27_vllm/serve.sh defnotean1/PseudoBrain-Qwen3.8-Flash-Next-NVFP4 8000
   ```
