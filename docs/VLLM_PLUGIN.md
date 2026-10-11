# PB-27 on vLLM v0.31.0 as an out-of-tree model (design, 2026-10-10)

Findings from reading vLLM v0.31.0 (`vllm/vllm-openai:v0.31.0-aarch64`); paths are under
`/usr/local/lib/python3.12/dist-packages/vllm/`.

## What PB-27 needs from the engine
Each of the 16 converted layers (indices 3,7,…,63) = the stock attention weights restricted to a 16384-token
sliding window (+4 StreamingLLM sink positions in the PyTorch model) **plus** a parallel Gated DeltaNet branch
on the same pre-norm input, outputs summed. Everything else is stock Qwen3.8-27B.

## 1. Layer construction and caches in vLLM
- `models/qwen3_5.py:252-261` builds layers from `config.layer_types`; `Qwen3_5DecoderLayer.__init__` (122-206)
  makes `QwenGatedDeltaNetAttention(prefix=f"{prefix}.linear_attn")` or `Qwen3NextAttention(prefix=f"{prefix}.self_attn")`.
- GDN layers are `MambaBase`; they self-register in `static_forward_context[prefix]`
  (`layers/mamba/gdn/qwen_gdn_linear_attn.py:535-538`) and expose a `MambaSpec` via `get_state_shape`
  (conv state (10240, 3+spec), ssm state (48,128,128)).
- Attention layers register `Attention(prefix=f"{prefix}.attn")` (`layers/attention/attention.py:443-445`);
  `get_kv_cache_spec` returns `SlidingWindowSpec` when `per_layer_sliding_window` is set.
- The runner collects cache-bearing modules by prefix string only (`v1/worker/gpu_model_runner.py:7424-7440`).

## 2. Sliding window and sinks
- `Attention(per_layer_sliding_window=16384)` → FA `window_size=(W-1, 0)`, same semantics as PB's `qi-ki < W`
  (`flash_attn.py:1127-1132`); mixed per-layer windows are supported.
- gpt-oss "sinks" are learned per-head logit sinks (FA3/FA4), not kept positions. `StaticSinkAttention` needs
  `sink_len % block_size == 0` and is full-attention; R-SWA (`rswa_attention.py`) is FA4-only.
- On the GB10 (sm_121) `get_flash_attn_version` picks FA2 (`fa_utils.py:169-177`): no logit sinks, no mask_mod.
- **Realistic option:** `per_layer_sliding_window=16384` with the 4 sinks dropped. First measure PB-27 with
  `attention_sinks=0` in PyTorch (`evaluate.py --no-sinks`); if the loss is unacceptable, retrain run B with sinks=0.

## 3. Two cache modules in one decoder layer
Supported: KV-cache groups are formed by spec, not decoder index (`v1/core/kv_cache_utils.py:2300-2358`,
`_get_kv_cache_groups_uniform_page_size` 1453-1596). Names like `layers.3.self_attn.attn` and `layers.3.gdn` both
work (`extract_layer_index` needs exactly one integer). Start with `--no-enable-prefix-caching` (cache-hit lookup
supports full attention + one other type only); `mamba_cache_mode="all"` is rejected for Qwen3.5.

## 4. Plugin skeleton
- `pyproject.toml`: `[project.entry-points."vllm.general_plugins"] pb27 = "pb27_vllm:register"`;
  `register()` calls `ModelRegistry.register_model("PB27ForConditionalGeneration", "pb27_vllm.model:PB27ForConditionalGeneration")`
  and adds the arch to `MODELS_CONFIG_MAP` with `Qwen3_5ForConditionalGenerationConfig`.
- `config.json`: `architectures=["PB27ForConditionalGeneration"]`, keep `model_type` qwen3_5 / qwen3_5_text,
  add `pb_hybrid_layers=[3,7,…,63]`, `pb_attention_window=16384`.
- Classes (copy `__init__` bodies; replacing modules after `super().__init__` trips "Duplicate layer name"):
  `PB27WindowedAttention(Qwen3NextAttention)` with `per_layer_sliding_window`; `PB27DecoderLayer` with
  `self.self_attn` + `self.gdn = QwenGatedDeltaNetAttention(prefix=f"{prefix}.gdn")`, forward
  `h = self_attn(h, positions) + gdn(h)`; `PB27Model(Qwen3_5Model)` (re-apply `@support_torch_compile`),
  `PB27ForCausalLM`, `PB27ForConditionalGeneration`.
- Weights: extend `efficiency/transplant.py: export_hf_qwen35` to emit HF-named safetensors for
  `WindowedHybridLayer`; vLLM's `hf_to_vllm_mapper` fuses by suffix. Per hybrid block i (root = sqrt(64),
  PB out-matrices are stored ×root; zero-centered norms: `w_hf = w_pb - 1`):
  - `self_attn.q_proj` ← `attn.q` [12288,5120]; `k_proj`/`v_proj` ← `attn.k/v` [1024,5120]; `o_proj` ← `attn.out/root`;
    `q_norm`/`k_norm` ← `attn.q_norm/k_norm - 1`.
  - `gdn.in_proj_qkv` ← `gdn.proj.weight[:4096+6144]`, `in_proj_z` ← next 6144 rows, `in_proj_a` ← next 48,
    `in_proj_b` ← last 48 (PB order `[qkv|z|a|b]`); vLLM fuses to `in_proj_qkvz` [16384,5120] and `in_proj_ba` [96,5120] (`[b|a]`).
  - `gdn.conv1d.weight` ← `gdn.conv_weight[:, None, :]` [10240,1,4]; `A_log`, `dt_bias` [48]; `norm.weight` ← `gdn.o_norm` [128];
    `out_proj` ← `gdn.out/root`.
  - `input_layernorm` ← `blocks.i.norm - 1`; `post_attention_layernorm` ← `mlp.norm - 1`; MLP gate/up split, `down/root`.

## 5. Risks and order
Risks: sinks dropped on GB10; prefix-caching limits with SWA+Mamba; Mamba page padding vs SWA block size;
cudagraph/compile with the two-mixer forward; `AutoWeightsLoader` fails on any unmapped key.
Order: (a) `--no-sinks` eval; (b) exporter; (c) plugin, serve `--enforce-eager --no-enable-prefix-caching`;
(d) logit parity vs PyTorch at 2k and >16k; (e) cudagraphs, prefix caching, MTP; (f) exact sinks if FA4 runs on sm_121.
