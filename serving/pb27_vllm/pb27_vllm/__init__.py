"""vLLM general plug-in for PB-27 (docs/PB27_VLLM_PLUGIN.md).

Loaded through the `vllm.general_plugins` entry point in every vLLM process
(front end, engine core, workers), so `register()` must be idempotent: it only
touches registries and imports nothing that initialises CUDA. The model classes
are referenced lazily by "<module>:<class>" strings.
"""

ARCHS = (
    "PB27ForConditionalGeneration",
    "PB27ForCausalLM",
    "PB27MoeForConditionalGeneration",
    "PB27MoeForCausalLM",
)
LLAMA_ARCHS = ("PB27LlamaForCausalLM",)  # a Llama teacher (Nanbeige4.3-33B), llama_model.py
QWEN4_EXP_ARCHS = (
    "PB27Qwen4ExpForConditionalGeneration",
    "PB27Qwen4ExpForCausalLM",
)


def register() -> None:
    from vllm import ModelRegistry
    from vllm.config import engram as vllm_engram
    from vllm.config import vllm as vllm_cfg
    from vllm.model_executor.models.config import (
        MODELS_CONFIG_MAP,
        Qwen3_5ForCausalLMConfig,
        Qwen3_5ForConditionalGenerationConfig,
        _strip_qwen4_exp_mrope,
    )

    class PB27Qwen4ExpForConditionalGenerationConfig(
        Qwen3_5ForConditionalGenerationConfig
    ):
        """Qwen4ExpForConditionalGenerationConfig (config.py:876-916) plus
        DFlash2 speculative decoding support."""

        @staticmethod
        def verify_and_update_config(vllm_config) -> None:
            Qwen3_5ForConditionalGenerationConfig.verify_and_update_config(vllm_config)
            text_config = vllm_config.model_config.hf_text_config
            if text_config.hc_count <= 1:
                raise ValueError("Qwen4Exp requires hc_count > 1")
            parallel_config = vllm_config.parallel_config
            uses_ple_or_qsa = bool(text_config.ple_layer_ids) or (
                getattr(text_config, "indexer_n_heads", None) is not None
            )
            if uses_ple_or_qsa and (
                parallel_config.enable_dbo or parallel_config.ubatch_size > 1
            ):
                raise NotImplementedError(
                    "Qwen4Exp PLE/QSA does not support dual-batch overlap or microbatching"
                )
            if text_config.ple_layer_ids and parallel_config.pipeline_parallel_size > 1:
                raise NotImplementedError(
                    "Qwen4Exp N-gram PLE embedding requires pipeline_parallel_size=1 "
                    "because non-first pipeline ranks do not receive the raw input_ids "
                    "it needs. Please run with PP=1."
                )
            multimodal_config = vllm_config.model_config.multimodal_config
            if multimodal_config is not None and multimodal_config.language_model_only:
                _strip_qwen4_exp_mrope(vllm_config.model_config)
            spec_config = vllm_config.speculative_config
            if spec_config is not None:
                if spec_config.method not in {
                    "mtp",
                    "ngram",
                    "ngram_gpu",
                    "dflash",
                }:
                    raise NotImplementedError(
                        "PB27Qwen4Exp speculative decoding supports only its native MTP, "
                        "DFlash2, and linear n-gram proposers"
                    )
                draft_mc = getattr(spec_config, "draft_model_config", None)
                if draft_mc is not None and getattr(draft_mc, "hf_text_config", None) is not None:
                    draft_hf = draft_mc.hf_text_config
                    max_len = int(getattr(vllm_config.model_config, "max_model_len", 0) or 0)
                    if max_len > int(getattr(draft_hf, "max_position_embeddings", 0) or 0):
                        draft_hf.max_position_embeddings = max_len

    class PB27Qwen4ExpForCausalLMConfig(PB27Qwen4ExpForConditionalGenerationConfig):
        @staticmethod
        def verify_and_update_config(vllm_config) -> None:
            PB27Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(
                vllm_config
            )
            _strip_qwen4_exp_mrope(vllm_config.model_config)

    for arch in ARCHS:
        if arch not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(arch, f"pb27_vllm.model:{arch}")
    for arch in LLAMA_ARCHS:
        if arch not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(arch, f"pb27_vllm.llama_model:{arch}")
    for arch in QWEN4_EXP_ARCHS:
        if arch not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(arch, f"pb27_vllm.qwen4_exp_model:{arch}")

    # The verify-and-update hooks of the stock Qwen3.5 classes (mamba SSM cache
    # dtype from the HF config; the CausalLM variant also drops the M-RoPE
    # fields). PB-27 keeps model_type qwen3_5(_moe) / qwen3_5(_moe)_text so they
    # apply as is; the stock MoE classes map to the same hooks (config.py:1061-1062).
    for arch in ("PB27ForConditionalGeneration", "PB27MoeForConditionalGeneration"):
        MODELS_CONFIG_MAP.setdefault(arch, Qwen3_5ForConditionalGenerationConfig)
    for arch in ("PB27ForCausalLM", "PB27MoeForCausalLM"):
        MODELS_CONFIG_MAP.setdefault(arch, Qwen3_5ForCausalLMConfig)
    # The Llama path only needs the mamba SSM cache dtype hook (no M-RoPE fields to drop).
    for arch in LLAMA_ARCHS:
        MODELS_CONFIG_MAP.setdefault(arch, Qwen3_5ForConditionalGenerationConfig)
    MODELS_CONFIG_MAP.setdefault(
        "PB27Qwen4ExpForConditionalGeneration",
        PB27Qwen4ExpForConditionalGenerationConfig,
    )
    MODELS_CONFIG_MAP.setdefault(
        "PB27Qwen4ExpForCausalLM",
        PB27Qwen4ExpForCausalLMConfig,
    )

    for arch in QWEN4_EXP_ARCHS:
        vllm_engram._NGRAM_LAYER_FIELDS[arch] = "ple_layer_ids"
    vllm_cfg.DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES = (
        vllm_cfg.DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES | frozenset(QWEN4_EXP_ARCHS)
    )
    vllm_cfg.default_breakable_cudagraph_architectures.cache_clear()

    # Exact memory tier (PB27_MEMORY env): archive bookkeeping hooks on the GPU model runner.
    from . import memory

    memory.configure_from_env()
    if memory.MEMORY["enabled"]:
        try:
            memory.install_runner_hooks()
        except Exception as exc:  # the API-server process has no runner; the engine/worker processes do
            import logging

            logging.getLogger(__name__).info("pb27 memory hooks not installed here: %s", exc)
