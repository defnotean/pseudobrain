# Exact memory tier inside the vLLM serve (design, 2026-10-12)

Goal: the served PB-27 (vLLM v0.31.0 plug-in, NVFP4 + DFlash2) gets the same exact-memory tier the PyTorch path has
(efficiency/exact_memory.py): keys/values that leave the 16k window are archived in 64-token blocks with Quest bounds,
and every decode step attends exactly to the top-k retrieved blocks in addition to the window. Later stages (trained
selector, local positions, INT2 archive) plug into the same seams.

## What vLLM gives us (read from the image, paths under vllm/)
- `v1/attention/backends/flash_attn.py`: the SWA layers use `SlidingWindowSpec`; FA2 is called with
  `window_size=(W-1, 0)` and `return_softmax_lse=False` (line ~249). The impl's `forward(layer, query, key, value,
  kv_cache, attn_metadata, output)` receives post-RoPE k/v for the scheduled tokens and the paged `kv_cache`.
- `v1/attention/ops/merge_attn_states.py: merge_attn_states(output, prefix_output, prefix_lse, suffix_output,
  suffix_lse)` — exact LSE merge of two partial attentions (used by cascade attention). Shapes: outputs
  [T, H, D], lse [H, T].
- `vllm_flash_attn.flash_attn_varlen_func(..., return_softmax_lse=True)` returns (out, lse) for a paged call
  (block_table + seqused_k) — the window attention with LSE is one call away.
- Custom ops registered with `direct_register_custom_op` and listed in `compilation_config.splitting_ops` run eagerly
  between CUDA-graph pieces (that is how `unified_attention_with_output` runs under PIECEWISE graphs).
- `forward_context.get_forward_context().attn_metadata[layer_name]` gives the per-layer metadata (query_start_loc,
  seq_lens, block_table, num_actual_tokens, max_query_len) inside the forward.

## Design
1. **Archive capture, per request.** In `PB27WindowedAttention.forward` (plug-in), after RoPE and before `self.attn`,
   append each scheduled token's k/v to `ARCHIVE[req_id]` in the order of `query_start_loc`. Request identity: a tiny
   hook on `GPUModelRunner._prepare_inputs` stores `input_batch.req_ids` (ordered as the batch) in a global that the
   layer reads; requests that disappear from the batch free their archive (scheduler finished/aborted → runner
   `input_batch.remove_request`, hooked the same way). Archive = per layer, per request: `keys [KV, N, D]`,
   `values [KV, N, D]` (fp8 as stored in the cache, or bf16 first), block bounds `kmax/kmin [KV, N/64, D]` updated
   incrementally (exact_memory.archive_append). Tokens still inside the window are also in the archive (harmless:
   retrieval only considers blocks older than the window).
2. **Decode retrieval with exact merge.** A custom op `pb27_memory_attention(q, window_out?, ...)` registered as a
   splitting op: for each decode request (query length ≤ 64), `retrieve()` picks the top-k blocks (Quest bound → exact
   rescoring), gathers their k/v, runs `flash_attn_varlen_func(q, k_ret, v_ret, return_softmax_lse=True)` (no mask:
   all retrieved keys are older than the query), and merges with the window attention via `merge_attn_states`.
   The window attention's LSE comes from calling the FA impl ourselves with `return_softmax_lse=True` for the hybrid
   layers (a subclass of FlashAttentionImpl selected via the layer's `attn_backend` override), so the window call is
   not duplicated. Prefill chunks skip retrieval in v1 (the PyTorch "decode-faithful" result is the reference).
3. **Positions.** v1 stores post-RoPE keys at true positions (what the PyTorch tier does; verified to 128k). v2 stores
   pre-RoPE keys and rotates retrieved blocks to local pseudo-positions just outside the window (InfLLM style) — the
   requirement for 1M+ (docs/PB27_RESEARCH_DEEP_2026-10.md item 3).
4. **Placement.** Archive on GPU first (bounded by `--max-model-len`), then pinned host memory with async gathers,
   then INT2 (OSCAR) for the 10M+ ladder.
5. **Config.** `PB27_MEMORY=blocks,block_size,decode_blocks,neighbors,rerank` env on serve.sh (same tuple as the
   PyTorch `--exact-memory`); 0 disables → identical to today's serve.

## Order and gates
a. Capture + bookkeeping under `--enforce-eager`; unit check: archive length == tokens seen per request.
b. Decode retrieval + merge; parity with the PyTorch exact-memory path on the recall probes at 32k/64k/128k through
   the server (passkey ≥ 1.00/1.00/1.00, multihop ≥ 0.92/0.92, copy ≥ 0.33 — the current PyTorch numbers).
c. CUDA graphs: the op in `splitting_ops`; speed smoke must stay within 10% of today's 48–59 tok/s.
d. Host-memory archive; length ladder starts.
