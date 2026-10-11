# pb27-vllm: PB-27 as a vLLM v0.31.0 out-of-tree model

Design and vLLM findings: `docs/PB27_VLLM_PLUGIN.md`. Checkpoint exporter:
`efficiency/transplant.py: export_hf_qwen35` (+ `pb27_config`).

PB-27 is Qwen3.8-27B where the 16 full-attention layers in `pb_hybrid_layers`
keep their attention weights under a `pb_attention_window`-token sliding window
and add a parallel Gated DeltaNet branch (`layers.i.gdn.*`, stock `linear_attn`
tensor names) on the same pre-norm input; the two outputs are summed. The
plug-in registers `PB27ForConditionalGeneration` / `PB27ForCausalLM`; the
checkpoint keeps `model_type` qwen3_5 / qwen3_5_text.

What vLLM runs differs from the PyTorch core in one known way: the 4
StreamingLLM sink positions are dropped (plain sliding window, FA2 on the GB10).
Compare against `evaluate.py --no-sinks` numbers.

## Layout

```
serving/pb27_vllm/
  pyproject.toml            package pb27-vllm, entry point vllm.general_plugins -> pb27_vllm:register
  pb27_vllm/__init__.py     register(): ModelRegistry + MODELS_CONFIG_MAP
  pb27_vllm/model.py        PB27WindowedAttention, PB27DecoderLayer, PB27Model, PB27ForCausalLM,
                            PB27ForConditionalGeneration
  serve.sh                  docker run + pip install -e + vllm serve (Spark, ~/gpu_run.sh)
```

## Install and run on the Spark

The exported checkpoint lives at `~/models/pb27a` (config.json has
`architectures=["PB27ForConditionalGeneration"]`, `pb_hybrid_layers`,
`pb_attention_window`, `pb_attention_sinks` on both the top level and
`text_config`).

```
# from the repo checkout on the Spark
serving/pb27_vllm/serve.sh ~/models/pb27a pb27 8000
# extra vllm args pass through, e.g. a smaller context for a first smoke test:
MAXLEN=32768 serving/pb27_vllm/serve.sh ~/models/pb27a pb27 8000 --max-num-seqs 4
docker stop pb27-vllm
```

`serve.sh` bind-mounts this directory at `/pb27_vllm`, runs
`pip install --no-deps -e /pb27_vllm` inside the container (writes the
`vllm.general_plugins` entry point; vLLM loads it in the API server, engine
core and worker processes), then runs

```
vllm serve /models/pb27a --served-model-name pb27 --max-model-len 131072 \
  --enforce-eager --no-enable-prefix-caching \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

`PB27_INSTALL=shim` skips pip: it puts the package on `PYTHONPATH` and adds a
`sitecustomize.py` that calls `pb27_vllm.register()` in every Python process.

Check the plug-in loaded: the engine log prints
`PB27: 16 hybrid layers [3, 7, ...], attention window 16384` and, with
`VLLM_LOGGING_LEVEL=DEBUG`, `Available plugins for group vllm.general_plugins:
pb27 -> pb27_vllm:register`.

Flags to keep for now (design doc sections 3 and 5): `--enforce-eager` (two
mixers in one layer have not been exercised under cudagraph capture /
torch.compile), `--no-enable-prefix-caching` (prefix-cache lookup supports full
attention plus one other cache type; PB-27 has sliding-window plus mamba), no
`--speculative-config` (MTP is step (e)). `--mamba-cache-mode all` is rejected,
as for Qwen3.5.

## How the plug-in maps onto vLLM

- `PB27WindowedAttention` = `Qwen3NextAttention.__init__` with
  `Attention(..., per_layer_sliding_window=W)`; the layer advertises a
  `SlidingWindowSpec` so the KV cache for these 16 layers is bounded by W.
- `PB27DecoderLayer` = `Qwen3_5DecoderLayer.__init__` plus
  `self.gdn = QwenGatedDeltaNetAttention(config, vllm_config, prefix=f"{prefix}.gdn",
  gqa_interleaved_layout=False)` for hybrid indices; forward does
  `self_attn(positions, h) + gdn(h)`.
- Caches: vLLM enumerates cache-bearing modules by prefix string
  (`v1/worker/gpu_model_runner.py: get_kv_cache_spec`), so
  `layers.3.self_attn.attn` (SlidingWindowSpec) and `layers.3.gdn` (MambaSpec)
  coexist; nothing counts one cache module per decoder layer. The mamba page
  size uses the per-model `get_mamba_state_shape_from_config` (one GDN state
  shape, identical for `linear_attn` and `gdn`).
- Weights: the stock `Qwen3_5Model.hf_to_vllm_mapper` fuses by substring
  (`.in_proj_qkv`/`.in_proj_z` -> `.in_proj_qkvz`, `.in_proj_b`/`.in_proj_a`
  -> `.in_proj_ba`), so `gdn.*` tensors load like `linear_attn.*` without
  extra rules. `AutoWeightsLoader` fails on any unmapped key, so an exporter
  name mismatch shows up at load time, not as a silent miss.

## Parity test (to implement)

Goal: the vLLM logits for PB-27 match the PyTorch core run with
`attention_sinks=0`, at a prompt that fits the window (2k tokens) and one that
exceeds it (>16k), so the window, the GDN branch and the sum are all exercised.

1. Reference logits. On the Spark (`scripts/tools/spark_test.sh`), load the
   PB-27 core in PyTorch with sinks disabled (the `--no-sinks` path of
   `evaluate.py`), run the fixed token ids in bf16, save the final-position
   logits (and, for the long prompt, logits at a few positions past the window,
   e.g. every 2048 tokens) as a `.pt` with the token ids.
2. vLLM logits. Start `serve.sh` and request the same token ids through
   `/v1/completions` with `"prompt": [ids]`, `max_tokens: 1`, `temperature: 0`,
   `logprobs: 20` (top-20 logprobs at the next-token position), or run an
   offline `LLM(...).generate` inside the container with
   `SamplingParams(max_tokens=1, logprobs=20, prompt_logprobs=0)` to get
   per-position prompt logprobs for the long-prompt comparison.
3. Compare: top-1 token agreement at every compared position; max absolute
   difference of the top-20 log-probs (expect bf16-level noise, roughly 1e-2;
   a sign of a wrong branch is a difference of order 1). Use
   `logprobs=20` because the API does not return full logits; for a full-vocab
   check run the offline path and patch `compute_logits` to dump the tensor.
4. Also compare against the same prompts with the plug-in disabled on the
   stock Qwen3.8-27B checkpoint: PB-27 must differ from stock on the >16k
   prompt (window in effect) and the hybrid branch must change the 2k logits
   relative to a run with `gdn` zeroed (sanity that the branch is live).
5. Two negative controls: (a) `pb_attention_window` doubled in a copied
   config.json should change the >16k logits; (b) a checkpoint with a renamed
   `gdn.` tensor must fail to load (unmapped key), not load silently.

Keep `--enforce-eager --no-enable-prefix-caching` for the parity run; a second
pass with cudagraphs and prefix caching is step (e) in the design doc.
