---
license: apache-2.0
library_name: vllm
pipeline_tag: text-generation
tags:
  - pseudobrain
  - qwen3.5-moe
  - qwen3.8
  - abliterated
  - moe
  - gated-deltanet
  - long-context
  - 1m-context
  - constant-memory
  - nvfp4
  - vllm
---

# PseudoBrain-Nex-35B-A3B (`pb27-nexc` / `PB27MoEForConditionalGeneration`)

**GitHub Repository & vLLM Plugin:** [https://github.com/defnotean/pseudobrain](https://github.com/defnotean/pseudobrain)  
**Hugging Face Editions (Pick Your GPU):**
- **W4A16 (4-bit Marlin/ROCm — Single 24 GB RTX 3090 / 4090 & AMD 7900 XTX):** [`defnoteatin/PseudoBrain-Nex-35B-A3B-W4A16`](https://huggingface.co/defnoteatin/PseudoBrain-Nex-35B-A3B-W4A16)
- **FP8 (RTX 4090 / H100 / AMD MI300X):** [`defnoteatin/PseudoBrain-Nex-35B-A3B-FP8`](https://huggingface.co/defnoteatin/PseudoBrain-Nex-35B-A3B-FP8)
- **NVFP4 (DGX Spark GB10 / RTX 5090 / Blackwell):** [`defnoteatin/PseudoBrain-Nex-35B-A3B-NVFP4`](https://huggingface.co/defnoteatin/PseudoBrain-Nex-35B-A3B-NVFP4)
- **BF16 (Universal NVIDIA & AMD ROCm):** [`defnoteatin/PseudoBrain-Nex-35B-A3B-BF16`](https://huggingface.co/defnoteatin/PseudoBrain-Nex-35B-A3B-BF16)

**Companion Model Families:** [`PseudoBrain-Qwen3.8-Flash-Next`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3.8-Flash-Next-NVFP4) (`45–183 tok/s`) | [`PseudoBrain-27B`](https://huggingface.co/defnoteatin/PseudoBrain-27B-NVFP4) (`27B` Dense)  
**Architecture Paper:** [`docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.md`](https://github.com/defnotean/pseudobrain/blob/main/docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.md)

## Overview

**PseudoBrain-Nex-35B-A3B (`pb27-nexc`)** is a sparse Mixture-of-Experts (`35B` total parameters, `~3B` active parameters per token across `256` routed experts + `1` shared expert) constant-GPU-memory hybrid long-context conversion of `Huihui-Qwen3.5-35B-A3B-abliterated` / `Nex-N2.5` (`Qwen3_5MoeForConditionalGeneration`). All `10` full-attention layers (`[3, 7, 11, ..., 39]`) are converted into **Hybrid Windowed-Attention + Parallel Gated DeltaNet** blocks (`PB27MoEDecoderLayer`) alongside `30` native Gated DeltaNet layers and PseudoBrain's **Three-Tier NVMe/Page-Cache Exact Memory Archive** with **True $O(1)$ Constant-Speed Decode** (`pb27_vllm`).

- **True $O(1)$ Constant-Speed Decode up to `1,000,000` Tokens (`~25–33.6 tok/s` Flat):**
  - **Decoupled Prefill vs. Decode Retrieval Budget:** Full scaled retrieval (`PB27_MEMORY_SCALE=2048:384:4096:512:8192:1536`) runs once on the final prefill question chunk (`PB27_MEMORY_PREFILL=pt`) and pins the winning blocks in `Archive.prompt_top` + GPU-resident `cached_pre_k` / `cached_pre_v`.
  - **Zero-NVMe Stride-`64` Decode Cache:** During decode, `PB27_MEMORY_DECODE_STRIDE=64`, `PB27_MEMORY_DECODE_BLOCKS=64`, and `PB27_MEMORY_DECODE_RERANK=0` keep LSE-merge FlashAttention-2 running over the fixed GPU buffer (`[4 sinks + prompt_top + 16K window]`), eliminating the historical `500K–1M` NVMe decode slowdown (`~8–10 tok/s` → **`~25–33.6 tok/s` constant $O(1)$ decode speed**).
  - **2-Level Hierarchical `bf16` Quest Super-Block Bounds (`PB27_MEMORY_SUPERBLOCK=64`):** Stores block `kmax`/`kmin` in `bf16` with `4,096`-token super-block bounds (`256` super-blocks at `1M` tokens), reducing bound scoring to `<0.05 ms` ($O(1)$).
  - **Priority-Aware RoPE Sink Repositioning (`_reposition_rope_sinks`) & Cross-KV-Head Unification:** Dynamically re-rotates retrieved archive blocks into contiguous local RoPE positions just before the sliding window (`[sinks .. win_start - 1]`) with priority ordering preserved (`prio_rank`).
- **$O(1)$ Constant GPU KV Cache (`~27–31 GiB` Total GPU Footprint up to `1,000,000` Tokens):** Uses only `8.1%` of a fixed `6 GiB` KV cache pool even at `1,000,000` tokens.
- **100% (`40/40 = 1.000`) Exact Single-Needle, 4-Needle (`multi4`), and 4-Hop Call-Chain Trace (`trace4`) Retrieval from `8K` to `1,000,000` Tokens:** Combines median-centered Quest block bounds, per-query Z-score normalization, post-merge `[b, b+1]` successor expansion (`k=1024, c=8192` at 1M), immutable `prompt_top` + rolling `decode_top` retention, and position-invariant 2-hop key-to-key bridge retrieval over the $75\%$ unrotated key subspace (`d // 4 :`).
- **W4A16 Marlin NVFP4 MoE Scale Alignment:** Includes automatic `w1`/`w3` `weight_global_scale` harmonization (`_install_nvfp4_scale_hooks`), reducing fused SwiGLU scale discrepancy from $38.1\%$ to $2.9\%$.

## Benchmark Summary

| Context Length ($L$) | `needle@0.1` | `needle@0.5` | `needle@0.9` | `multi4` (`4` constants) | `trace4` (`4`-hop chain) | Mean TTFT | KV Pool Used |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`8,192`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`2.0s`** (`33.59 tok/s`) | `8.1%` |
| **`32,768`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`17.0s`** (`8.8s` warm) | `8.1%` |
| **`65,536`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`21.2s`** | `8.1%` |
| **`120,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`36.7s`** | `8.1%` |
| **`200,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`81.2s`** | `8.1%` |
| **`250,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`90.3s`** | `8.1%` |
| **`500,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`238.3s`** | `8.1%` |
| **`1,000,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`615.4s`** | `8.1%` |

- **Custom Medium-Hard Coding Benchmark (`nexc_med`, 33 problems × 2 samples):** **`26 / 33` problems solved (`78.8%`)**, **`43 / 66` sample pass rate (`65.2%`)** (`reports/pb27/nexc_med.summary.json`).

## Quickstart with vLLM

1. Clone the PseudoBrain repository and install the `pb27_vllm` out-of-tree vLLM plugin:
   ```bash
   git clone https://github.com/defnotean/pseudobrain.git
   cd pseudobrain
   pip install -e serving/pb27_vllm
   ```

2. Serve the NVFP4 + BF16 GDN checkpoint directly (`O(1)` constant-speed decode defaults are enabled automatically):
   ```bash
   bash serving/pb27_vllm/serve.sh defnotean1/PseudoBrain-Nex-35B-A3B-NVFP4 8000
   ```
