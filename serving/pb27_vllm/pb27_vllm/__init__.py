"""vLLM general plug-in for PB-27 (docs/PB27_VLLM_PLUGIN.md).

Loaded through the `vllm.general_plugins` entry point in every vLLM process
(front end, engine core, workers), so `register()` must be idempotent: it only
touches registries and imports nothing that initialises CUDA. The model classes
are referenced lazily by "<module>:<class>" strings.
"""

ARCHS = ("PB27ForConditionalGeneration", "PB27ForCausalLM")


def register() -> None:
    from vllm import ModelRegistry
    from vllm.model_executor.models.config import (
        MODELS_CONFIG_MAP,
        Qwen3_5ForCausalLMConfig,
        Qwen3_5ForConditionalGenerationConfig,
    )

    for arch in ARCHS:
        if arch not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(arch, f"pb27_vllm.model:{arch}")

    # The verify-and-update hooks of the stock Qwen3.5 classes (mamba SSM cache
    # dtype from the HF config; the CausalLM variant also drops the M-RoPE
    # fields). PB-27 keeps model_type qwen3_5 / qwen3_5_text so they apply as is.
    MODELS_CONFIG_MAP.setdefault(
        "PB27ForConditionalGeneration", Qwen3_5ForConditionalGenerationConfig
    )
    MODELS_CONFIG_MAP.setdefault("PB27ForCausalLM", Qwen3_5ForCausalLMConfig)

    # Exact memory tier (PB27_MEMORY env): archive bookkeeping hooks on the GPU model runner.
    from . import memory

    memory.configure_from_env()
    if memory.MEMORY["enabled"]:
        try:
            memory.install_runner_hooks()
        except Exception as exc:  # the API-server process has no runner; the engine/worker processes do
            import logging

            logging.getLogger(__name__).info("pb27 memory hooks not installed here: %s", exc)
