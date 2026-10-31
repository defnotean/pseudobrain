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
UNIVERSAL_ARCHS = (
    "PB27Qwen3NextForCausalLM",
    "PB27Qwen3MoeForCausalLM",
    "PB27Qwen3ForCausalLM",
    "PB27Gemma4ForCausalLM",
    "PB27Gemma4ForConditionalGeneration",
)


def _fallback_strip_mrope(model_config) -> None:
    """Version-tolerant fallback if vLLM moves or renames `_strip_qwen4_exp_mrope`."""
    hf_text = getattr(model_config, "hf_text_config", None)
    for cfg in (getattr(model_config, "hf_config", None), hf_text):
        if cfg is None:
            continue
        for container_name in ("rope_parameters", "rope_scaling"):
            rope_cfg = getattr(cfg, container_name, None)
            if isinstance(rope_cfg, dict):
                rope_cfg.pop("mrope_section", None)
                rope_cfg.pop("mrope_interleaved", None)


def register() -> None:
    import logging

    from vllm import ModelRegistry

    log = logging.getLogger(__name__)

    try:
        from vllm.config import engram as vllm_engram
    except Exception:
        vllm_engram = None

    try:
        from vllm.config import vllm as vllm_cfg
    except Exception:
        vllm_cfg = None

    try:
        from vllm.model_executor.models.config import MODELS_CONFIG_MAP
    except Exception:
        MODELS_CONFIG_MAP = {}

    try:
        from vllm.model_executor.models.config import (
            Qwen3_5ForCausalLMConfig,
            Qwen3_5ForConditionalGenerationConfig,
        )
    except Exception:
        class Qwen3_5ForConditionalGenerationConfig:  # type: ignore[no-redef]
            @staticmethod
            def verify_and_update_config(vllm_config) -> None:
                pass

        class Qwen3_5ForCausalLMConfig(Qwen3_5ForConditionalGenerationConfig):  # type: ignore[no-redef]
            pass

    try:
        from vllm.model_executor.models.config import Gemma4Config
    except Exception:
        Gemma4Config = Qwen3_5ForConditionalGenerationConfig

    try:
        from vllm.model_executor.models.config import _strip_qwen4_exp_mrope
    except Exception:
        _strip_qwen4_exp_mrope = _fallback_strip_mrope

    class PB27Gemma4Config(Gemma4Config):
        @staticmethod
        def verify_and_update_config(vllm_config) -> None:
            Qwen3_5ForConditionalGenerationConfig.verify_and_update_config(vllm_config)
            if Gemma4Config is not Qwen3_5ForConditionalGenerationConfig:
                Gemma4Config.verify_and_update_config(vllm_config)

    class PB27Qwen4ExpForConditionalGenerationConfig(
        Qwen3_5ForConditionalGenerationConfig
    ):
        """Qwen4ExpForConditionalGenerationConfig plus DFlash2 speculative decoding support."""

        @staticmethod
        def verify_and_update_config(vllm_config) -> None:
            Qwen3_5ForConditionalGenerationConfig.verify_and_update_config(vllm_config)
            text_config = vllm_config.model_config.hf_text_config
            if getattr(text_config, "hc_count", 4) <= 1:
                raise ValueError("Qwen4Exp requires hc_count > 1")
            parallel_config = vllm_config.parallel_config
            uses_ple_or_qsa = bool(getattr(text_config, "ple_layer_ids", None)) or (
                getattr(text_config, "indexer_n_heads", None) is not None
            )
            if uses_ple_or_qsa and (
                getattr(parallel_config, "enable_dbo", False)
                or getattr(parallel_config, "ubatch_size", 1) > 1
            ):
                raise NotImplementedError(
                    "Qwen4Exp PLE/QSA does not support dual-batch overlap or microbatching"
                )
            if getattr(text_config, "ple_layer_ids", None) and parallel_config.pipeline_parallel_size > 1:
                raise NotImplementedError(
                    "Qwen4Exp N-gram PLE embedding requires pipeline_parallel_size=1 "
                    "because non-first pipeline ranks do not receive the raw input_ids "
                    "it needs. Please run with PP=1."
                )
            multimodal_config = vllm_config.model_config.multimodal_config
            if multimodal_config is not None and getattr(multimodal_config, "language_model_only", False):
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

    supported = set(ModelRegistry.get_supported_archs())
    for arch in ARCHS:
        if arch not in supported:
            ModelRegistry.register_model(arch, f"pb27_vllm.model:{arch}")
    for arch in LLAMA_ARCHS:
        if arch not in supported:
            ModelRegistry.register_model(arch, f"pb27_vllm.llama_model:{arch}")
    for arch in QWEN4_EXP_ARCHS:
        if arch not in supported:
            ModelRegistry.register_model(arch, f"pb27_vllm.qwen4_exp_model:{arch}")
    for arch in UNIVERSAL_ARCHS:
        if arch not in supported:
            ModelRegistry.register_model(arch, f"pb27_vllm.universal_models:{arch}")

    # The verify-and-update hooks of the stock Qwen3.5 classes (mamba SSM cache
    # dtype from the HF config; the CausalLM variant also drops the M-RoPE
    # fields). PB-27 keeps model_type qwen3_5(_moe) / qwen3_5(_moe)_text so they
    # apply as is; the stock MoE classes map to the same hooks.
    for arch in ("PB27ForConditionalGeneration", "PB27MoeForConditionalGeneration"):
        MODELS_CONFIG_MAP.setdefault(arch, Qwen3_5ForConditionalGenerationConfig)
    for arch in ("PB27ForCausalLM", "PB27MoeForCausalLM"):
        MODELS_CONFIG_MAP.setdefault(arch, Qwen3_5ForCausalLMConfig)
    for arch in (
        *LLAMA_ARCHS,
        "PB27Qwen3NextForCausalLM",
        "PB27Qwen3MoeForCausalLM",
        "PB27Qwen3ForCausalLM",
    ):
        MODELS_CONFIG_MAP.setdefault(arch, Qwen3_5ForConditionalGenerationConfig)
    for arch in ("PB27Gemma4ForCausalLM", "PB27Gemma4ForConditionalGeneration"):
        MODELS_CONFIG_MAP.setdefault(arch, PB27Gemma4Config)
    MODELS_CONFIG_MAP.setdefault(
        "PB27Qwen4ExpForConditionalGeneration",
        PB27Qwen4ExpForConditionalGenerationConfig,
    )
    MODELS_CONFIG_MAP.setdefault(
        "PB27Qwen4ExpForCausalLM",
        PB27Qwen4ExpForCausalLMConfig,
    )

    if vllm_engram is not None and hasattr(vllm_engram, "_NGRAM_LAYER_FIELDS"):
        for arch in QWEN4_EXP_ARCHS:
            vllm_engram._NGRAM_LAYER_FIELDS[arch] = "ple_layer_ids"
    if vllm_cfg is not None and hasattr(vllm_cfg, "DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES"):
        vllm_cfg.DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES = (
            vllm_cfg.DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES | frozenset(QWEN4_EXP_ARCHS)
        )
        if hasattr(vllm_cfg, "default_breakable_cudagraph_architectures") and hasattr(
            vllm_cfg.default_breakable_cudagraph_architectures, "cache_clear"
        ):
            vllm_cfg.default_breakable_cudagraph_architectures.cache_clear()

    # Exact memory tier (PB27_MEMORY env): archive bookkeeping hooks on the GPU model runner.
    from . import memory

    memory.configure_from_env()
    if memory.MEMORY["enabled"]:
        try:
            memory.install_runner_hooks()
        except Exception as exc:  # the API-server process has no runner; the engine/worker processes do
            log.info("pb27 memory hooks not installed here: %s", exc)
