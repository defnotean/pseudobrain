---
license: apache-2.0
library_name: vllm
pipeline_tag: text-generation
tags:
  - pseudobrain
  - nanbeige
  - llama
  - gated-deltanet
  - long-context
  - 1m-context
  - constant-memory
  - vllm
---

# PseudoBrain-Nanbeige4.3-33B (`PB27LlamaForCausalLM`)

**GitHub Repository & vLLM Plugin:** [https://github.com/defnotean/pseudobrain](https://github.com/defnotean/pseudobrain)  
**Hugging Face Editions:**
- **W4A16 (4-bit AWQ/Marlin — Single 24 GB RTX 3090 / 4090 & AMD 7900 XTX):** [`defnoteatin/PseudoBrain-Nanbeige4.3-33B-W4A16`](https://huggingface.co/defnoteatin/PseudoBrain-Nanbeige4.3-33B-W4A16)
- **FP8 (RTX 4090 / H100 / AMD MI300X):** [`defnoteatin/PseudoBrain-Nanbeige4.3-33B-FP8`](https://huggingface.co/defnoteatin/PseudoBrain-Nanbeige4.3-33B-FP8)
- **NVFP4 (DGX Spark / RTX 5090 / Blackwell):** [`defnoteatin/PseudoBrain-Nanbeige4.3-33B-NVFP4`](https://huggingface.co/defnoteatin/PseudoBrain-Nanbeige4.3-33B-NVFP4)
- **BF16 (Universal NVIDIA & AMD ROCm):** [`defnoteatin/PseudoBrain-Nanbeige4.3-33B-BF16`](https://huggingface.co/defnoteatin/PseudoBrain-Nanbeige4.3-33B-BF16)

**Architecture Paper:** [`docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.md`](https://github.com/defnotean/pseudobrain/blob/main/docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.md)

## Overview

**PseudoBrain-Nanbeige4.3-33B** is a full-model (`52 / 52` layers) constant-GPU-memory hybrid long-context conversion of `Nanbeige/Nanbeige4.3-33B` (`LlamaForCausalLM` → `PB27LlamaForCausalLM`). Because stock `Nanbeige4.3-33B` is a pure 52-layer full-attention dense model (`hidden_size=7168`, `56` query heads, `8` KV heads), its standard KV cache explodes linearly to **`~104 GiB` at `1,000,000` tokens** with quadratic $O(L^2)$ prefill latency.

PseudoBrain converts **all 52 full-attention layers (`pb_hybrid_layers = [0..51]`)** into **Hybrid Windowed-Attention + Parallel Gated DeltaNet** blocks (`PB27LlamaDecoderLayer`) paired with PseudoBrain's **Three-Tier NVMe/Page-Cache Exact Memory Archive** and **True $O(1)$ Constant-Speed Decode Engine** (`pb27_vllm`):

- **98.4% GPU KV Cache Reduction at `1,000,000` Tokens:** Caps on-GPU KV cache to a fixed $W = 16,384$ sliding window + $4$ StreamingLLM sinks (`~1.6 GiB` fixed GPU KV cache regardless of whether the prompt is `16K` or `1,000,000` tokens).
- **Runs `1,000,000` Context on a Single `24 GB` GPU (`W4A16` Edition):** In 4-bit `W4A16` (`compressed-tensors` Marlin / ROCm), model weights use `~17.5 GiB` and the fixed `16K` KV cache uses `~1.6 GiB`, fitting comfortably inside a **single `24 GB` RTX 3090, RTX 4090, or AMD Radeon RX 7900 XTX** at `1,000,000` tokens without OOM.
- **Parallel Gated DeltaNet Recurrent Branches (`52` layers, `bf16`):** Trained via Stage-1 hidden-state alignment + Stage-2 full-model knowledge distillation (`25.56M` tokens, final `val_kl = 0.0261`, student validation NLL **`1.5218`** outperforming teacher NLL `1.5306`).
- **True $O(1)$ Constant-Speed Decode:** Decoupled prefill vs. decode Quest retrieval (`PB27_MEMORY_PREFILL=pt`, `PB27_MEMORY_DECODE_STRIDE=64`, `PB27_MEMORY_DECODE_BLOCKS=64`, `PB27_MEMORY_DECODE_RERANK=0`, `PB27_MEMORY_SUPERBLOCK=64`) + Priority-Aware RoPE Sink Repositioning (`_reposition_rope_sinks`).

## Quickstart with vLLM (NVIDIA & AMD ROCm)

1. Clone the PseudoBrain repository and install the `pb27_vllm` out-of-tree vLLM plugin:
   ```bash
   git clone https://github.com/defnotean/pseudobrain.git
   cd pseudobrain
   pip install -e serving/pb27_vllm
   ```

2. Launch with `serve.sh` (pick the edition matching your GPU):
   ```bash
   # Single 24 GB GPU (RTX 3090 / RTX 4090 / AMD RX 7900 XTX):
   bash serving/pb27_vllm/serve.sh defnoteatin/PseudoBrain-Nanbeige4.3-33B-W4A16 8000

   # FP8 (RTX 4090 / H100 / AMD MI300X):
   bash serving/pb27_vllm/serve.sh defnoteatin/PseudoBrain-Nanbeige4.3-33B-FP8 8000

   # NVFP4 (DGX Spark GB10 / RTX 5090 / Blackwell):
   bash serving/pb27_vllm/serve.sh defnoteatin/PseudoBrain-Nanbeige4.3-33B-NVFP4 8000

   # BF16 (Universal Multi-GPU / 80GB+ HBM / AMD Instinct):
   bash serving/pb27_vllm/serve.sh defnoteatin/PseudoBrain-Nanbeige4.3-33B-BF16 8000
   ```
