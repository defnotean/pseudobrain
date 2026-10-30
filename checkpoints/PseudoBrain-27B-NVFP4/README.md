---
license: apache-2.0
library_name: vllm
pipeline_tag: text-generation
tags:
  - pseudobrain
  - qwen3.5
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
**Architecture Paper:** [`docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.md`](https://github.com/defnotean/pseudobrain/blob/main/docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.md)

## Overview

**PseudoBrain-27B (`pb27c`)** is a constant-GPU-memory hybrid long-context conversion of the 27B dense Qwen3.5 architecture (`Huihui-Qwen3.8-27B-abliterated`). Every fourth layer (`16` full-attention layers `[3, 7, 11, ..., 63]`) is converted into a **Hybrid Windowed-Attention + Parallel Gated DeltaNet** block (`PB27DecoderLayer`) paired with PseudoBrain's **Three-Tier NVMe/Page-Cache Exact Memory Archive** (`pb27_vllm`).

- **$O(1)$ Constant GPU KV Cache from `8K` to `1,000,000+` Tokens:** Local sliding window $W = 16,384$ + $4$ StreamingLLM sinks in GPU HBM (`--kv-cache-memory-bytes 6G–8G`, total GPU footprint `~29–33 GiB` even at $1,000,000$ tokens).
- **Parallel Gated DeltaNet Recurrent Branches (`16` layers + `48` native GDN layers):** Carries dense global discourse state across the entire context.
- **Unrotated Two-Hop Exact Memory Retrieval:** Evicted post-RoPE keys/values are archived in $64$-token blocks in sparse unlinked `mmap` files (`PB27_MEMORY_OFFLOAD=nvme`, `PB27_MEMORY_NVME_DTYPE=bf16`) with GPU-resident Quest min/max bounds, per-query Z-score + query-centered bound scoring, `[b, b+1]` successor expansion, and position-invariant 2-hop key-to-key bridge retrieval over the $75\%$ unrotated key subspace (`d // 4 :`), merged with the sliding window via FlashAttention-2 exact LSE rescaling.
- **True $O(1)$ Constant-Speed Decode up to `1,000,000` Tokens:** Decouples prefill retrieval budget (`PB27_MEMORY_SCALE=2048:384:4096:512:8192:1536`, run once on the final prefill question chunk and pinned in `Archive.prompt_top` + GPU-cached `cached_pre_k`/`cached_pre_v`) from the incremental decode retrieval budget (`PB27_MEMORY_DECODE_BLOCKS=64`, `PB27_MEMORY_DECODE_RERANK=0`, `PB27_MEMORY_DECODE_STRIDE=64`) backed by 2-level hierarchical `bf16` super-block Quest bounds (`PB27_MEMORY_SUPERBLOCK=64`, `256` super-blocks at `1M`, `<0.05 ms` bound scoring, zero NVMe pages touched for rejected candidates during decode).

## Benchmark Summary

| Benchmark / Metric | Result |
| :--- | :---: |
| **Medium Coding Benchmark (`custom_code_bench.py`, 33 problems × 2)** | **`90.91%` (`60 / 66`)** |
| **Single-Needle Exact Recall (`8K, 32K, 65K, 120K, 200K, 250K, 500K, 1M`)** | **`100.0%` (`24 / 24` across all depths `0.1, 0.5, 0.9`)** |
| **Multi-Needle 4-Constant JSON Recall (`8K–250K`)** | **`100.0%` (`4 / 4` at every length)** |
| **GPU KV Pool Usage at `1,000,000` Tokens** | **`8.1%` of `6 GiB` pool (`~33 GiB` total GPU memory)** |
| **Standing Decode Throughput (`8K` context, batch 1)** | **`13.07 tok/s`** |

## Quickstart with vLLM

1. Clone the PseudoBrain repository and install the `pb27_vllm` out-of-tree vLLM plugin:
   ```bash
   git clone https://github.com/defnotean/pseudobrain.git
   cd pseudobrain
   pip install -e serving/pb27_vllm
   ```

2. Launch the constant-memory vLLM server:
   ```bash
   bash serving/pb27_vllm/serve.sh defnotean1/PseudoBrain-27B-NVFP4 8000
   ```

3. For $1,000,000$-token extended context serving (`6 GiB` KV pool + NVMe BF16 exact memory tier):
   ```bash
   bash scripts/bench/custom_longctx_extended.sh pb27c_ext /path/to/pb27c_nvfp4
   ```
