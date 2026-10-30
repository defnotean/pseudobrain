---
license: apache-2.0
library_name: vllm
pipeline_tag: text-generation
tags:
  - pseudobrain
  - qwen3.5-moe
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
**Hugging Face Weights:** [https://huggingface.co/defnotean1/PseudoBrain-Nex-35B-A3B-NVFP4](https://huggingface.co/defnotean1/PseudoBrain-Nex-35B-A3B-NVFP4)  
**Architecture Paper:** [`docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.md`](https://github.com/defnotean/pseudobrain/blob/main/docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.md)

## Overview

**PseudoBrain-Nex-35B-A3B (`pb27-nexc`)** is a sparse Mixture-of-Experts (`35B` total parameters, `~3B` active parameters per token across `256` routed experts + `1` shared expert) constant-GPU-memory hybrid long-context conversion of `Nex-N2.5-mini` (`Qwen3_5MoeForConditionalGeneration`). All `10` full-attention layers (`[3, 7, 11, ..., 39]`) are converted into **Hybrid Windowed-Attention + Parallel Gated DeltaNet** blocks (`PB27MoEDecoderLayer`) alongside `30` native Gated DeltaNet layers and PseudoBrain's **Three-Tier NVMe/Page-Cache Exact Memory Archive** (`pb27_vllm`).

- **2.6×–3.4× Faster Long-Context Prefill & 11.8× Faster Long-Context Decode:** Achieves **`33.59 tok/s`** decode at `8K` (`25.48 tok/s` under `1M` `max_model_len`), **`13.82 tok/s`** decode (`36.7s` TTFT) at `120K`, **`13.31 tok/s`** decode (`90.3s` TTFT) at `250K`, `500K` TTFT in **`238.3s`**, and `1,000,000` TTFT in **`615.4s`** on a single NVIDIA GB10 GPU with zero required `PB27_MEMORY_*` flags.
- **$O(1)$ Constant GPU KV Cache (`~27–31 GiB` Total GPU Footprint up to `1,000,000` Tokens):** Uses only `8.1%` of a fixed `6 GiB` KV cache pool even at `1,000,000` tokens.
- **100% (`40/40 = 1.000`) Exact Single-Needle, 4-Needle (`multi4`), and 4-Hop Call-Chain Trace (`trace4`) Retrieval from `8K` to `1,000,000` Tokens:** Combines median-centered Quest block bounds, per-query Z-score normalization, post-merge `[b, b+1]` successor expansion (`k=1024, c=8192` at 1M), immutable `prompt_top` + rolling `decode_top` retention, and position-invariant 2-hop key-to-key bridge retrieval over the $75\%$ unrotated key subspace (`d // 4 :`).
- **True $O(1)$ Constant-Speed Decode up to `1,000,000` Tokens:** Decouples prefill retrieval budget (`PB27_MEMORY_SCALE=2048:384:4096:512:8192:1536`, run once on the final prefill question chunk and pinned in `Archive.prompt_top` + GPU-cached `cached_pre_k`/`cached_pre_v`) from the incremental decode retrieval budget (`PB27_MEMORY_DECODE_BLOCKS=64`, `PB27_MEMORY_DECODE_RERANK=0`, `PB27_MEMORY_DECODE_STRIDE=64`) backed by 2-level hierarchical `bf16` super-block Quest bounds (`PB27_MEMORY_SUPERBLOCK=64`, `256` super-blocks at `1M`, `<0.05 ms` bound scoring, zero NVMe pages touched for rejected candidates during decode).
- **W4A16 Marlin NVFP4 MoE Scale Alignment:** Includes automatic `w1`/`w3` `weight_global_scale` harmonization (`_install_nvfp4_scale_hooks`), reducing fused SwiGLU scale discrepancy from $38.1\%$ to $2.9\%$.

## Benchmark Summary

| Context Length ($L$) | `needle@0.1` | `needle@0.5` | `needle@0.9` | `multi4` (`4` constants) | `trace4` (`4`-hop chain) | Mean TTFT | KV Pool Used |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`8,192`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`2.0s`** (`28.8 tok/s`) | `8.1%` |
| **`32,768`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`17.0s`** (`8.8s` warm) | `8.1%` |
| **`65,536`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`21.2s`** | `8.1%` |
| **`120,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`36.7s`** (`13.82 tok/s`) | `8.1%` |
| **`200,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`81.2s`** | `8.1%` |
| **`250,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`90.3s`** (`13.31 tok/s`) | `8.1%` |
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

2. Serve the NVFP4 + BF16 GDN checkpoint directly (or compose from slim + base NVFP4 using `scripts/convert/make_pb27_nvfp4.py`):
   ```bash
   bash serving/pb27_vllm/serve.sh defnotean1/PseudoBrain-Nex-35B-A3B-NVFP4 8000
   ```
