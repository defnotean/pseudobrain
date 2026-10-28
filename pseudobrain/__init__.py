"""PseudoBrain conversion core, hybrid windowed-attention + Gated DeltaNet layers, and reference exact-memory tier."""

from . import exact_memory, gated_deltanet, lora, moe_kernels, recall_tool, scan, transplant
from .exact_memory import EXACT_MEMORY, Archive, archive_append, retrieve
from .gated_deltanet import GatedDeltaNetLayer
from .transplant import (
    TeacherConfig,
    TokenCore,
    WindowedHybridLayer,
    export_hf_qwen35,
    pb27_config,
)

__all__ = [
    "Archive",
    "EXACT_MEMORY",
    "GatedDeltaNetLayer",
    "TeacherConfig",
    "TokenCore",
    "WindowedHybridLayer",
    "archive_append",
    "exact_memory",
    "export_hf_qwen35",
    "gated_deltanet",
    "lora",
    "moe_kernels",
    "pb27_config",
    "recall_tool",
    "retrieve",
    "scan",
    "transplant",
]
