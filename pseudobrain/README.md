# `pseudobrain`: Conversion Core & Reference Exact Memory Tier

This package contains the self-contained PyTorch conversion core for PseudoBrain (`PB-27` dense and `PB-Nex` 35B-A3B MoE):

- [`transplant.py`](transplant.py): `TransplantConfig`, `TransplantModel`, `WindowedHybridLayer` (sliding-window attention + 4 StreamingLLM sinks + parallel Gated DeltaNet branch), weight loader (`load_qwen35_weights`), and Hugging Face safetensors exporter (`export_hf_qwen35`).
- [`gated_deltanet.py`](gated_deltanet.py): Reference and `flash-linear-attention` (`fla`) fused Gated DeltaNet layer (`GatedDeltaNetLayer`).
- [`exact_memory.py`](exact_memory.py): Reference PyTorch exact-memory tier (`Archive`, `archive_append`, `retrieve` with Quest bounding boxes and exact block rescoring).
- [`moe_kernels.py`](moe_kernels.py) & [`scan.py`](scan.py): Sparse MoE dispatch and recurrent scan primitives.
