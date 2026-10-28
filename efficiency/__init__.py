"""Compatibility alias forwarding `efficiency.*` imports to `pseudobrain.*`."""

from pseudobrain import (
    Archive,
    EXACT_MEMORY,
    GatedDeltaNetLayer,
    TeacherConfig,
    TokenCore,
    WindowedHybridLayer,
    archive_append,
    exact_memory,
    export_hf_qwen35,
    gated_deltanet,
    lora,
    moe_kernels,
    pb27_config,
    recall_tool,
    retrieve,
    scan,
    transplant,
)
