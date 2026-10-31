---
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-Flash-Next
  - edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD
  - cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4
  - Qwen/Qwen3.8-Flash-Next-FP8
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
  - vllm
  - nvfp4
---

# PseudoBrain-Qwen3.8-Flash-Next-NVFP4 — 1M-Context Ultra-Sparse 36B-A1.7B MoE + PLE + DFlash2 Speculative Decoding (`45–183 tok/s`)

> [!IMPORTANT]
> **Created by Ian Kurt Galeon ([@defnotean](https://github.com/defnotean) · [Hugging Face `@defnoteatin`](https://huggingface.co/defnoteatin) · `iangaleonofficial@gmail.com`)**  
> **GitHub & Universal `pb27_vllm` Engine Plugin:** [https://github.com/defnotean/pseudobrain](https://github.com/defnotean/pseudobrain)  
> **Architecture Paper:** [`PSEUDOBRAIN_ARCHITECTURE_PAPER.pdf`](https://github.com/defnotean/pseudobrain/blob/main/docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.pdf) · [`PSEUDOBRAIN_ARCHITECTURE_PAPER.md`](https://github.com/defnotean/pseudobrain/blob/main/docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.md) · **Benchmark Receipts:** [`reports/`](https://github.com/defnotean/pseudobrain/tree/main/reports)  
> *Open to **LLM Inference Systems / Kernel & Engine Engineering roles**, research collaborations, and **enterprise on-premise 1M-context deployments**.*

---

## 1. Why `PseudoBrain-Qwen3.8-Flash-Next` Exists

In standard Transformer serving, **model weights are paid once, while KV cache memory grows linearly ($O(L)$) and prefill compute grows quadratically ($O(L^2)$)** — forcing long-context (`128K–1,000,000` token) workloads to OOM even when the GPU easily fits the model weights.

**`PseudoBrain-Qwen3.8-Flash-Next-NVFP4`** (`PB27Qwen4ExpForConditionalGeneration`) solves this via the **PseudoBrain architecture**:
1. **Hybrid Windowed-Attention + Parallel Gated DeltaNet (`GDN`):** 48 layers: 36 native Gated DeltaNet + 12 full-attention layers (`[3, 7, 11, ..., 47]`) converted into 16K SWA + Parallel Gated DeltaNet + Quest Exact Memory. Recurrent GDN state is **$O(1)$ fixed size** regardless of whether context is `4K` or `1,000,000` tokens.
2. **Three-Tier NVMe/Page-Cache Quest Exact Memory (`piece_attention`):** On the sparse anchor layers, queries attend over `[4 StreamingLLM Sinks + Top-k Quest Exact-Memory Pages + 16,384 Sliding Window]` via FlashAttention-2 LSE-merge (`pb27_vllm/memory.py`) with **2-Level Hierarchical `bf16` Quest Super-Block Bounds (`PB27_MEMORY_SUPERBLOCK=64`)** and **Priority-Aware RoPE Sink Repositioning (`_reposition_rope_sinks`)**.
3. **True $O(1)$ Constant-Speed Decode up to `1,000,000` Tokens:** Decouples the scaled prefill retrieval budget from the constant decode budget (`PB27_MEMORY_DECODE_STRIDE=64`, `PB27_MEMORY_DECODE_BLOCKS=64`, `PB27_MEMORY_DECODE_RERANK=0`), cutting **1M-token KV cache memory by `96.0%+`** (`~48.0 GB (or static YaRN 4.0x degradation)` → **`~486 MiB active window (4.7 GB pool for 1.15M tokens)`**).

### Hardware & Precision Editions for `PseudoBrain-Qwen3.8-Flash-Next`

| Repository | Precision | Model Weights | **Total VRAM at `1,000,000` Tokens** | Target GPUs |
| :--- | :---: | :---: | :---: | :--- |
| [`defnoteatin/PseudoBrain-Qwen3.8-Flash-Next-W4A16`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3.8-Flash-Next-W4A16) | **`W4A16`** | `~21.0 GB` | **`~26.0 GB`** | RTX 3090 / 4090 (24 GB, 18 shards PLE-pruned), RTX 5090 (32 GB), AMD RX 7900 XTX / MI300X |
| [`defnoteatin/PseudoBrain-Qwen3.8-Flash-Next-FP8`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3.8-Flash-Next-FP8) | **`FP8`** | `~36.5 GB` | **`~42.0 GB`** | L40S / A100 / H100, 2x RTX 4090, DGX Spark (128 GB), AMD Instinct MI300X |
| [`defnoteatin/PseudoBrain-Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3.8-Flash-Next-NVFP4) **(This Repo)** | **`NVFP4`** | `~21.0 GB` | **`~27.5 GB`** | NVIDIA Blackwell (DGX Spark GB10, RTX 5090 32 GB, RTX PRO 6000, B200) |
| [`defnoteatin/PseudoBrain-Qwen3.8-Flash-Next-BF16`](https://huggingface.co/defnoteatin/PseudoBrain-Qwen3.8-Flash-Next-BF16) | **`BF16`** | `~72.0 GB` | **`~78.0 GB`** | H100 / H200 (80 GB+), DGX Spark (128 GB), AMD MI300X, Multi-GPU CUDA / ROCm |

---

## 2. Two-Command Quickstart (`vLLM` on NVIDIA CUDA & AMD ROCm)

### Option A: Install Plugin Directly into Existing `vLLM` Environment (2 Commands)

```bash
# 1. Install the universal PseudoBrain vLLM V1 plugin (auto-registers via vllm.general_plugins)
pip install --no-deps "git+https://github.com/defnotean/pseudobrain.git#subdirectory=serving/pb27_vllm"

# 2. Launch 1,000,000-token OpenAI-compatible server
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve defnoteatin/PseudoBrain-Qwen3.8-Flash-Next-NVFP4 \
  --served-model-name pb27 \
  --max-model-len 1048576 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --kv-cache-memory-bytes 6G \
  --max-num-seqs 16 \
  --speculative-config '{"method":"dflash","model":"incoai/Qwen3.8-Flash-Next-DFlash2","num_speculative_tokens":8}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

*(Note: Every Hugging Face checkpoint also bundles `pb27_vllm_plugin/` directly inside the repository if you prefer `pip install --no-deps -e ./pb27_vllm_plugin` from a local snapshot download.)*

### Option B: One-Command Docker Launch (`serve.sh` — Auto-Detects `x86_64` & `aarch64`)

```bash
git clone https://github.com/defnotean/pseudobrain.git
cd pseudobrain
bash serving/pb27_vllm/serve.sh defnoteatin/PseudoBrain-Qwen3.8-Flash-Next-NVFP4 pb27 8000
```

### Option C: Connect `OpenCode` / Agentic Coding Harnesses (`~/.config/opencode/opencode.jsonc`)

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "model": "pseudobrain/pb27",
  "small_model": "pseudobrain/pb27",
  "compaction": { "auto": true, "prune": true },
  "provider": {
    "pseudobrain": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "PseudoBrain 1M Local Server",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "local-loopback",
        "timeout": 600000
      },
      "models": {
        "pb27": {
          "name": "PseudoBrain-Qwen3.8-Flash-Next-NVFP4 (1M Context)",
          "tool_call": true,
          "limit": { "context": 1048576, "output": 16384 }
        }
      }
    }
  }
}
```

---

## 3. Verified PseudoBrain Benchmark Results (Committed Receipts in [`reports/`](https://github.com/defnotean/pseudobrain/tree/main/reports))

### 3.1 Same-Hardware (`DGX Spark GB10, TP1`) Speed & Memory vs. Stock at `65K` and `1,000,000` Tokens

Measured directly against upstream `runtime-qualification.json` on the **exact same `DGX Spark (GB10, 128 GB Unified Memory), TP1` hardware**:

| Metric on **DGX Spark (GB10, TP1)** | Stock `Qwen3.8-Flash-Next-NVFP4` (`runtime-qualification.json`) | **`PseudoBrain-Qwen3.8-Flash-Next-NVFP4` (Ours)** | Speedup / Savings |
| :--- | :---: | :---: | :---: |
| **`65K` Prefill Time (TTFT)** | `38.93 s` (`1,680 tok/s`) | **`21.02 s` (`3,108 tok/s`)** | **1.85× faster prefill (`-46%` latency)** |
| **`65K` Decode Speed (`tok/s`)** | `35.8 – 39.1 tok/s` *(mean `37.8 tok/s`, MTP-3)* | **`53.4 – 88.3 tok/s`** *(mean **`65.9 tok/s`**, peak **`183.5 tok/s`** w/ DFlash2)* | **1.74× – 2.26× faster decode** |
| **`1,000,000` (`1M`) Prefill Time (TTFT)** | `1,128.0 – 1,132.2 s` (`18.9 min`, `926 tok/s`) | **`543.06 s`** (`9.0 min`, **`1,801 tok/s`**) | **2.08× faster `1M` prefill (saves ~10 min per `1M` prompt)** |
| **`1,000,000` (`1M`) Decode Speed (`tok/s`)** | `33.3 – 37.1 tok/s` *(mean `34.9 tok/s`)* | **`44.6 – 67.7 tok/s`** *(mean **`57.1 tok/s`**, `67.7 tok/s` on `1M` needle)* | **1.64× – 1.94× faster decode at `1M` tokens** |
| **KV Cache Memory at `1M` Tokens** | Full `1,048,576`-token KV cache (`max-running-requests 1`, requires static YaRN `factor=4.0`) | **Fixed `16,384`-token GPU window (`~486 MiB` active)** + NVMe Quest $O(1)$ archive (`--max-num-seqs 16`, **no static YaRN degradation**) | **~96%–99.2% less GPU KV memory at `1M` + `16` concurrent sequences** |

### 3.2 `8K` to `1,000,000` Token Exact Retrieval (`needle@0.1/0.5/0.9`, `multi4`, `trace4`)

| Context Length ($L$) | `needle@0.1` | `needle@0.5` | `needle@0.9` | `multi4` (`4` constants) | `trace4` (`4`-hop chain) | `Flash-Next` Decode `tok/s` | `Nex-35B-A3B` Recall | GPU KV Pool Used |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`8,192`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`93.01 tok/s`** | **`5/5 (100%)`** | **`8.1%`** |
| **`32,768`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`66.51 tok/s`** | **`5/5 (100%)`** | **`8.1%`** |
| **`65,536`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`67.86 tok/s`** | **`5/5 (100%)`** | **`8.1%`** |
| **`120,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`50.50 tok/s`** | **`5/5 (100%)`** | **`8.1%`** |
| **`200,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`50.81 tok/s`** | **`5/5 (100%)`** | **`8.1%`** |
| **`250,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`69.87 tok/s`** | **`5/5 (100%)`** | **`8.1%`** |
| **`500,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`64.91 tok/s`** (`183.5` peak) | **`5/5 (100%)`** | **`8.1%`** |
| **`1,000,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`0.75–1.0`** | **`1.0` (`Nex`)** | **`63.76 tok/s`** (`67.7` needle) | **`5/5 (100%)`** | **`8.1%`** |

### 3.3 General Intelligence, Reasoning, & Software Engineering Parity

- **MMLU (`n=300`, 0-shot):** **`85.00%`** (`255/300`, within statistical stderr of stock `~85.0–87.9%`).
- **GSM8K (`n=150`, strict CoT):** **`96.00%`** (`144/150`, **`+2.71%`** above official `93.29%` report).
- **MMLU-Pro (`n=150`, 10-choice college reasoning):** **`81.33%`** raw / **`85.31%`** non-truncated (vs. `73.23%` Base / `80.6%` Instruct).
- **Algorithmic Coding (`custom_code_bench`, 33 medium-hard Python problems):** **`96.97%` (`32/33`)** on `PseudoBrain-Qwen3.8-Flash-Next` at **`74.74 tok/s`**; **`90.91%` (`60/66`)** on `PseudoBrain-27B`.
- **Real-World Post-Cutoff Software Engineering (`PB-SWE-Live`, 59 tasks, 100-step agent loop):** **`26 / 59` (`44.1%`)** vs. Stock **`28 / 59` (`47.5%`)** (paired McNemar $p \approx 0.7$, statistically indistinguishable).

---

## 4. Complete PseudoBrain 1M-Context Model Lineup (10 Families × 4 Precisions = 40 Models)

All 40 repositories share the single unified `pb27_vllm` plugin (`pip install --no-deps "git+https://github.com/defnotean/pseudobrain.git#subdirectory=serving/pb27_vllm"`), with slim BF16 GDN delta overlays hosted at [`defnoteatin/pb27-checkpoints`](https://huggingface.co/defnoteatin/pb27-checkpoints):

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

## 5. License, Citation & Contact

- **Code & Plugin License:** Apache-2.0 ([`https://github.com/defnotean/pseudobrain`](https://github.com/defnotean/pseudobrain)). Model weights inherit their upstream base model license (`apache-2.0`).
- **Author:** **Ian Kurt Galeon**
  - **GitHub:** [@defnotean](https://github.com/defnotean)
  - **Hugging Face:** [@defnoteatin](https://huggingface.co/defnoteatin)
  - **Email (Recruiting, Research & Enterprise Inquiries):** `iangaleonofficial@gmail.com`
