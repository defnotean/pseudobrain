---
license: apache-2.0
library_name: vllm
pipeline_tag: text-generation
tags:
  - pseudobrain
  - qwen3.8-flash-next
  - qwen4-exp
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
**Hugging Face Weights:** [https://huggingface.co/defnotean1/PseudoBrain-Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/defnotean1/PseudoBrain-Qwen3.8-Flash-Next-NVFP4)  
**Speculative Draft Model (`DFlash2`):** [`incoai/Qwen3.8-Flash-Next-DFlash2`](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)  
**Architecture Paper:** [`docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.md`](https://github.com/defnotean/pseudobrain/blob/main/docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.md)

## Overview

**PseudoBrain-Qwen3.8-Flash-Next-NVFP4 (`flash_next`)** is an ultra-sparse Mixture-of-Experts (`48` layers, `512` routed experts with `top-10` active per token + `1` shared expert, Per-Layer Embeddings / PLE) constant-GPU-memory hybrid long-context conversion of `Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD` (`Qwen4ExpForConditionalGeneration`). All `12` full-attention layers (`[3, 7, 11, ..., 47]`) are converted into **Hybrid Windowed-Attention + Parallel Gated DeltaNet** blocks (`PB27Qwen4ExpDecoderLayer`) alongside `36` native Gated DeltaNet layers, PseudoBrain's **Three-Tier NVMe/Page-Cache Exact Memory Archive** with **True $O(1)$ Constant-Speed Decode**, and native **`DFlash2` multi-layer speculative decoding** (`5` speculative draft tokens per step).

- **True $O(1)$ Constant-Speed Decode up to `1,000,000` Tokens (`64–183 tok/s` at `500K`):** Decouples the context-scaled prefill retrieval budget (`PB27_MEMORY_SCALE`) from the constant decode budget (`PB27_MEMORY_DECODE_BLOCKS=64`, `PB27_MEMORY_DECODE_RERANK=0`, `PB27_MEMORY_DECODE_STRIDE=64`), warms the GPU LSE-merge cache (`cached_pre_k` / `cached_pre_v`) at the end of `final_prefill`, and uses **2-Level Hierarchical `bf16` Quest Super-Block Bounds** (`64` blocks = `4,096` tokens per super-block) for $O(1)$ (`~0.05 ms`) bound scoring.
- **$O(1)$ Constant GPU KV Cache (`~28 GiB` Total GPU Footprint up to `1,000,000` Tokens):** Uses only `8.1%` of a fixed `6 GiB` KV cache pool even at `1,000,000` tokens on a single NVIDIA GB10 GPU.
- **Strong General Reasoning, Math & Coding Capabilities:**
  - **MMLU (`n=300`, 0-shot, greedy):** **`85.00%`** (`255 / 300`, `stderr: 0.0206`)
  - **GSM8K (`n=150`, CoT, greedy non-thinking):** **`96.00%`** (`144 / 150`, `stderr: 0.0161`)
  - **MMLU-Pro (`n=150`, 10-option CoT, greedy non-thinking):** **`81.33%`** (`122 / 150`, `85.31%` non-truncated)
  - **Custom Medium-Hard Coding Benchmark (`flash_next_med`, 33 problems):** **`96.97%` (`32 / 33`)** (`easy: 8/8`, `medium: 12/12`, `fix: 3/3`, `hard: 9/10`, `0` syntax/truncation errors)

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
| **`1,000,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`0.75` (`3/4`)** | `0.0` | **`543.06s`** | **`63.76 tok/s`** (`44.60` mean, `37.27` `multi4`) | `8.1%` |

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
   PB27_SPECULATIVE_CONFIG='{"method":"dflash","model":"incoai/Qwen3.8-Flash-Next-DFlash2","num_speculative_tokens":5}' \
   bash serving/pb27_vllm/serve.sh defnotean1/PseudoBrain-Qwen3.8-Flash-Next-NVFP4 8000
   ```
