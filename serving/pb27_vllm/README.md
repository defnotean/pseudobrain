# `pb27-vllm`: Universal PseudoBrain 1M-Context vLLM V1 Out-of-Tree Plugin

**Author:** Ian Kurt Galeon ([@defnotean](https://github.com/defnotean) · [Hugging Face `@defnoteatin`](https://huggingface.co/defnoteatin) · `iangaleonofficial@gmail.com`)

`pb27-vllm` (`v0.2.0`) is the universal out-of-tree vLLM V1 engine and memory plugin for all **10 PseudoBrain model families (`40` repositories across `W4A16`, `FP8`, `NVFP4`, and `BF16`)**. It registers all 7 PseudoBrain hybrid architectures via `vllm.general_plugins` (`pb27 = "pb27_vllm:register"`):

- `PB27ForConditionalGeneration` / `PB27ForCausalLM` (`PseudoBrain-27B`, `PseudoBrain-Qwen3.5-9B`, `PseudoBrain-Qwen3.5-4B`)
- `PB27MoEForConditionalGeneration` (`PseudoBrain-Nex-35B-A3B`)
- `PB27Qwen4ExpForConditionalGeneration` (`PseudoBrain-Qwen3.8-Flash-Next` + `DFlash2` speculative decoding)
- `PB27Qwen3NextForCausalLM` (`PseudoBrain-Qwen3-Coder-Next` 80B-A3B)
- `PB27Qwen3MoeForCausalLM` (`PseudoBrain-Qwen3-Coder-30B-A3B`)
- `PB27Gemma4ForConditionalGeneration` (`PseudoBrain-Gemma-4-31B`, `PseudoBrain-Gemma-4-26B-A4B`)
- `PB27Qwen3ForCausalLM` (`PseudoBrain-Qwen3-32B`)

---

## 1. Install in 1 Command

```bash
pip install --no-deps "git+https://github.com/defnotean/pseudobrain.git#subdirectory=serving/pb27_vllm"
```

Verify registration:
```bash
python3 -c "import pb27_vllm; pb27_vllm.register(); print('pb27_vllm registered successfully')"
```

---

## 2. Serve Any PseudoBrain Model up to `1,000,000` Tokens

### Direct `vllm serve`
```bash
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

### Docker Helper (`serve.sh` — Auto-Detects `x86_64` & `aarch64`)
```bash
bash serving/pb27_vllm/serve.sh defnoteatin/PseudoBrain-Qwen3.5-9B-W4A16 pb27 8000
```

---

## 3. Architecture & Exact Memory Tier (`pb27_vllm/memory.py`)

1. **Bounded 16K Sliding Window + Parallel Gated DeltaNet (`GDN`):**
   - Each converted full-attention layer runs `self_attn(positions, hidden_states) + gdn(hidden_states)` where `self_attn` advertises a `16,384`-token `SlidingWindowSpec` and `gdn` maintains an $O(1)$ recurrent `MambaSpec` state.
2. **Three-Tier NVMe/Page-Cache Quest Exact Memory + LSE Merge (`piece_attention`):**
   - Archives rotated keys/values per request in `64`-token blocks with **2-Level Hierarchical `bf16` Quest Super-Block Bounds (`PB27_MEMORY_SUPERBLOCK=64`)** and **Priority-Aware RoPE Sink Repositioning (`_reposition_rope_sinks`)**.
   - Recomputes attention over `[4 StreamingLLM Sinks + Top-k Quest Pages + 16,384 Window]` using FlashAttention-2 LSE merge (`d <= 256`) or exact PyTorch SDPA (`head_dim = 512` on Gemma 4).
3. **True $O(1)$ Constant-Speed Decode (`DECODE_STRIDE=64`, `DECODE_BLOCKS=64`, `DECODE_RERANK=0`):**
   - Warms `cached_pre_k` / `cached_pre_v` at the end of `final_prefill` so decode and multi-token speculative verification run directly in GPU HBM with zero NVMe page traffic for rejected drafts.
