# `pb27-vllm`: Out-of-Tree vLLM v0.31.0 Plugin for PseudoBrain (`PB27` & `PB-Nex`)

`pb27_vllm` registers the PseudoBrain dense (`PB27ForConditionalGeneration` / `PB27ForCausalLM`) and sparse MoE (`PB27MoEForConditionalGeneration` / `PB27MoEForCausalLM`) hybrid architectures alongside the **Three-Tier Exact Memory Hierarchy** (`pb27_vllm/memory.py`) as an out-of-tree vLLM plugin (`vllm.general_plugins`).

## Supported Architectures

| Model Class | Base Model Type | Hybrid Layers | Architecture Summary |
| :--- | :--- | :--- | :--- |
| **`PB27ForConditionalGeneration`** | `qwen3_5` (27B Dense) | `16` layers (`[3, 7, ..., 63]`) | `48` native Gated DeltaNet layers + `16` `PB27DecoderLayer` (`16K` SWA + `4` sinks + parallel GDN + Exact Memory Archive) |
| **`PB27MoEForConditionalGeneration`** | `qwen3_5_moe` (35B-A3B MoE) | `10` layers (`[3, 7, ..., 39]`) | `30` native Gated DeltaNet layers + `10` `PB27MoEDecoderLayer` (`16K` SWA + `4` sinks + parallel GDN + Exact Memory Archive, `256` routed experts + `1` shared expert, `top-8` active) |

## Key Features

1. **Zero-Config Production Defaults:** Importing `pb27_vllm.memory` automatically enables:
   - `PB27_MEMORY=256,64,256,0,512` and `PB27_MEMORY_SCALE=2048:384:4096:512:8192:1536`
   - `PB27_MEMORY_PREFILL=pt`, `PB27_MEMORY_PERTOKEN=1`, `PB27_MEMORY_SINKS=4`, `PB27_MEMORY_FAITHFUL=1`
   - `PB27_MEMORY_GPU_ARC_CAP=262144` (256K GPU-first archive tier, spilling to NVMe `mmap` up to `1,000,000+` tokens via `PB27_MEMORY_OFFLOAD=nvme`, `PB27_MEMORY_NVME_DTYPE=bf16`)
   - `PB27_MEMORY_DECODE_STRIDE=4` (reuses cached `pre_k`/`pre_v` in FlashAttention-2 `[SINKS + R, KV, D]` layout on non-refresh decode steps)
2. **100% Exact `8K–1M` Retrieval (`40/40 = 1.000`):**
   - Full `c = 16,384` token-level exact archive scan at the end of prefill (`lead < 0`) + query-axis relative Z-score normalization + `[b, b+1]` successor expansion + unrotated (`d // 4 :`) 2-hop key-to-key bridge retrieval.
3. **Real-Time Long-Context Decode (`13.82 tok/s` at `120K`, `13.31 tok/s` at `250K`):**
   - $O(M \log M)$ stable-sort priority deduplication (`_dedup_priority`), preallocated GPU `cu_seqlens` tables (`_CU_TABLES`), non-causal `varlen_fwd` (`is_causal=False`) on pre-sliced `t == 1` windows, and 2-kernel `torch.sigmoid` + `torch.lerp` LSE merge.
4. **W4A16 Marlin NVFP4 MoE Scale Harmonization:**
   - `_install_nvfp4_scale_hooks` in `pb27_vllm/model.py` automatically aligns `w1` (`gate_proj`) and `w3` (`up_proj`) `weight_global_scale` tensors before vLLM's fused `w13` Marlin SwiGLU kernel runs.

## Quickstart

```bash
# Install in an existing vLLM environment:
pip install -e serving/pb27_vllm

# Or launch via the Docker wrapper (installs the plugin automatically inside vllm/vllm-openai:v0.31.0-aarch64):
bash serving/pb27_vllm/serve.sh defnotean1/PseudoBrain-Nex-35B-A3B-NVFP4 pb27-nex 8000
bash serving/pb27_vllm/serve.sh defnotean1/PseudoBrain-27B-NVFP4 pb27 8000
```
