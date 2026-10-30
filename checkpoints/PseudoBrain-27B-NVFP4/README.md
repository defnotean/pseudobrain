---
license: apache-2.0
library_name: vllm
pipeline_tag: text-generation
tags:
  - pseudobrain
  - qwen3.5
  - qwen3.8
  - abliterated
  - gated-deltanet
  - long-context
  - 1m-context
  - constant-memory
  - nvfp4
  - vllm
---

# PseudoBrain-27B (`pb27c` / `PB27ForConditionalGeneration`)

**GitHub Repository & vLLM Plugin:** [https://github.com/defnotean/pseudobrain](https://github.com/defnotean/pseudobrain)  
**Hugging Face Weights:** [https://huggingface.co/defnotean1/PseudoBrain-27B-NVFP4](https://huggingface.co/defnotean1/PseudoBrain-27B-NVFP4)  
**Companion Models:** [`PseudoBrain-Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/defnotean1/PseudoBrain-Qwen3.8-Flash-Next-NVFP4) (`45–183 tok/s`) | [`PseudoBrain-Nex-35B-A3B-NVFP4`](https://huggingface.co/defnotean1/PseudoBrain-Nex-35B-A3B-NVFP4) (`25–34 tok/s`)  
**Architecture Paper:** [`docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.md`](https://github.com/defnotean/pseudobrain/blob/main/docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.md)

## Overview

**PseudoBrain-27B (`pb27c`)** is a constant-GPU-memory hybrid long-context conversion of the 27B dense abliterated Qwen architecture (`Huihui-Qwen3.8-27B-abliterated`). Every fourth layer (`16` full-attention layers `[3, 7, 11, ..., 63]`) is converted into a **Hybrid Windowed-Attention + Parallel Gated DeltaNet** block (`PB27DecoderLayer`) paired with PseudoBrain's **Three-Tier NVMe/Page-Cache Exact Memory Archive** and **True $O(1)$ Constant-Speed Decode Engine** (`pb27_vllm`).

- **$O(1)$ Constant GPU KV Cache from `8K` to `1,000,000+` Tokens:** Local sliding window $W = 16,384$ + $4$ StreamingLLM sinks in GPU HBM (`--kv-cache-memory-bytes 6G–8G`, total GPU footprint `~29–33 GiB` even at $1,000,000$ tokens — vs. `64 GiB` KV cache alone on stock 27B).
- **True $O(1)$ Constant-Speed Decode up to `1,000,000` Tokens (`~13.1 tok/s` Flat from `8K` to `1M`):**
  - **Decoupled Prefill vs. Decode Retrieval Budget:** Full scaled multi-block retrieval (`PB27_MEMORY_SCALE=2048:384:4096:512:8192:1536`) runs once on the final prefill question chunk (`PB27_MEMORY_PREFILL=pt`) and pins the retrieved blocks in `Archive.prompt_top` + GPU-resident `cached_pre_k` / `cached_pre_v`.
  - **Zero-NVMe Stride-`64` Decode Cache:** During generation, `PB27_MEMORY_DECODE_STRIDE=64`, `PB27_MEMORY_DECODE_BLOCKS=64`, and `PB27_MEMORY_DECODE_RERANK=0` keep LSE-merge FlashAttention-2 running over the fixed GPU buffer (`[4 sinks + prompt_top + 16K window]`), eliminating `~98%` of decode retrieval overhead and touching zero NVMe pages for rejected candidates.
  - **2-Level Hierarchical `bf16` Quest Super-Block Bounds (`PB27_MEMORY_SUPERBLOCK=64`):** Stores block `kmax`/`kmin` in `bf16` with `4,096`-token super-block bounds (`256` super-blocks at `1M` tokens), reducing bound scoring to `<0.05 ms` ($O(1)$).
  - **Priority-Aware RoPE Sink Repositioning (`_reposition_rope_sinks`) & Cross-KV-Head Unification:** Dynamically re-rotates retrieved archive blocks into contiguous local RoPE positions just before the sliding window (`[sinks .. win_start - 1]`) with priority ordering preserved (`prio_rank`), enabling `1,000,000`-token retrieval without static YaRN short-context degradation.
- **Parallel Gated DeltaNet Recurrent Branches (`16` layers + `48` native GDN layers):** Carries dense global discourse state across the entire context.

## Benchmark Summary

| Benchmark / Metric | Result |
| :--- | :---: |
| **Standard `lm-eval` Parity vs. Stock `27B` (`HellaSwag`, `PIQA`, `ARC-E`, `ARC-C`, `Winogrande`, `LAMBADA`)** | **Within `1 SE` ($\max |\Delta| = 0.009$)** |
| **Real-World Software Engineering (`PB-SWE-Live`, 59 tasks)** | **`44.1%` (`26 / 59`, statistically matched to stock `27B` `47.5%`)** |
| **Medium Coding Benchmark (`custom_code_bench.py`, 33 problems × 2)** | **`90.91%` (`60 / 66`)** |
| **Single-Needle Exact Recall (`8K, 32K, 65K, 120K, 200K, 250K, 500K, 1M`)** | **`100.0%` (`24 / 24` across all depths `0.1, 0.5, 0.9`)** |
| **Multi-Needle 4-Constant JSON Recall (`8K–250K`)** | **`100.0%` (`4 / 4` at every length)** |
| **GPU KV Pool Usage at `1,000,000` Tokens** | **`8.1%` of `6 GiB` pool (`~33 GiB` total GPU memory)** |
| **Constant $O(1)$ Decode Throughput (`8K` to `1,000,000` tokens, batch 1)** | **`13.07 tok/s` (flat $O(1)$ decode cost across context length)** |

## Quickstart with vLLM

1. Clone the PseudoBrain repository and install the `pb27_vllm` out-of-tree vLLM plugin:
   ```bash
   git clone https://github.com/defnotean/pseudobrain.git
   cd pseudobrain
   pip install -e serving/pb27_vllm
   ```

2. Launch the constant-memory, constant-speed vLLM server (`O(1)` decode defaults are enabled automatically):
   ```bash
   bash serving/pb27_vllm/serve.sh defnotean1/PseudoBrain-27B-NVFP4 8000
   ```
