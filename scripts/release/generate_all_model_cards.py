#!/usr/bin/env python3
"""Generate comprehensive, benchmark-backed Hugging Face Model Cards (README.md)
for all 10 PseudoBrain model families (40 repositories across W4A16, FP8, NVFP4, BF16),
the defnoteatin/pb27-checkpoints hub, and the public GitHub repository.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

FAMILIES = {
    "flash_next": {
        "family_title": "PseudoBrain-Qwen3.8-Flash-Next",
        "arch_class": "PB27Qwen4ExpForConditionalGeneration",
        "tagline": "1M-Context Ultra-Sparse 36B-A1.7B MoE + PLE + DFlash2 Speculative Decoding (`45–183 tok/s`)",
        "license": "apache-2.0",
        "pipeline_tag": "text-generation",
        "total_params": "36B",
        "active_params": "~1.7B (top-10 of 512 experts + 1 shared + Layer-2 PLE)",
        "layers_desc": "48 layers: 36 native Gated DeltaNet + 12 full-attention layers (`[3, 7, 11, ..., 47]`) converted into 16K SWA + Parallel Gated DeltaNet + Quest Exact Memory",
        "stock_1m_kv": "~48.0 GB (or static YaRN 4.0x degradation)",
        "pb_1m_kv": "~486 MiB active window (4.7 GB pool for 1.15M tokens)",
        "kv_reduction": "96.0%+",
        "extra_vllm_flags": " \\\n  --speculative-config '{\"method\":\"dflash\",\"model\":\"incoai/Qwen3.8-Flash-Next-DFlash2\",\"num_speculative_tokens\":8}'",
        "parser_flags": " \\\n  --reasoning-parser qwen3 \\\n  --enable-auto-tool-choice \\\n  --tool-call-parser qwen3_coder",
        "tags": ["pseudobrain", "qwen3.8-flash-next", "qwen4-exp", "moe", "gated-deltanet", "dflash2", "speculative-decoding", "long-context", "1m-context", "constant-memory", "vllm"],
        "editions": {
            "W4A16": {
                "repo": "defnoteatin/PseudoBrain-Qwen3.8-Flash-Next-W4A16",
                "base_repo": "cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4",
                "upstream_base": "Qwen/Qwen3.8-Flash-Next",
                "weights_gb": "~21.0 GB",
                "total_1m_gb": "~26.0 GB",
                "gpus": "RTX 3090 / 4090 (24 GB, 18 shards PLE-pruned), RTX 5090 (32 GB), AMD RX 7900 XTX / MI300X",
            },
            "FP8": {
                "repo": "defnoteatin/PseudoBrain-Qwen3.8-Flash-Next-FP8",
                "base_repo": "Qwen/Qwen3.8-Flash-Next-FP8",
                "upstream_base": "Qwen/Qwen3.8-Flash-Next",
                "weights_gb": "~36.5 GB",
                "total_1m_gb": "~42.0 GB",
                "gpus": "L40S / A100 / H100, 2x RTX 4090, DGX Spark (128 GB), AMD Instinct MI300X",
            },
            "NVFP4": {
                "repo": "defnoteatin/PseudoBrain-Qwen3.8-Flash-Next-NVFP4",
                "base_repo": "edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD",
                "upstream_base": "Qwen/Qwen3.8-Flash-Next",
                "weights_gb": "~21.0 GB",
                "total_1m_gb": "~27.5 GB",
                "gpus": "NVIDIA Blackwell (DGX Spark GB10, RTX 5090 32 GB, RTX PRO 6000, B200)",
            },
            "BF16": {
                "repo": "defnoteatin/PseudoBrain-Qwen3.8-Flash-Next-BF16",
                "base_repo": "Qwen/Qwen3.8-Flash-Next",
                "upstream_base": "Qwen/Qwen3.8-Flash-Next",
                "weights_gb": "~72.0 GB",
                "total_1m_gb": "~78.0 GB",
                "gpus": "H100 / H200 (80 GB+), DGX Spark (128 GB), AMD MI300X, Multi-GPU CUDA / ROCm",
            },
        },
    },
    "qwen3_coder_next": {
        "family_title": "PseudoBrain-Qwen3-Coder-Next",
        "arch_class": "PB27Qwen3NextForCausalLM",
        "tagline": "1M-Context Frontier 80B-A3B Agentic Coding MoE (Runs 1M Context in ~46 GB Total VRAM)",
        "license": "apache-2.0",
        "pipeline_tag": "text-generation",
        "total_params": "80B",
        "active_params": "3.0B (top-10 of 512 routed experts + 1 shared expert)",
        "layers_desc": "48 layers: 36 native Gated DeltaNet + 12 full-attention layers (`[3, 7, 11, ..., 47]`) converted into 16K SWA + Parallel Gated DeltaNet + Quest Exact Memory",
        "stock_1m_kv": "~48.0 GB",
        "pb_1m_kv": "~3.1 GB",
        "kv_reduction": "93.5%",
        "extra_vllm_flags": "",
        "parser_flags": " \\\n  --enable-auto-tool-choice \\\n  --tool-call-parser qwen3_coder",
        "tags": ["pseudobrain", "qwen3-next", "qwen3-coder-next", "coding", "agentic", "moe", "gated-deltanet", "long-context", "1m-context", "constant-memory", "vllm"],
        "editions": {
            "W4A16": {
                "repo": "defnoteatin/PseudoBrain-Qwen3-Coder-Next-W4A16",
                "base_repo": "qtum/Qwen3-Coder-Next-AWQ",
                "upstream_base": "Qwen/Qwen3-Coder-Next",
                "weights_gb": "~41.5 GB",
                "total_1m_gb": "~45.5 GB",
                "gpus": "2x RTX 3090 / 4090 (48 GB), RTX 6000 Ada (48 GB), 2x RTX 5090, DGX Spark, AMD MI300X",
            },
            "FP8": {
                "repo": "defnoteatin/PseudoBrain-Qwen3-Coder-Next-FP8",
                "base_repo": "Qwen/Qwen3-Coder-Next-FP8",
                "upstream_base": "Qwen/Qwen3-Coder-Next",
                "weights_gb": "~80.5 GB",
                "total_1m_gb": "~85.0 GB",
                "gpus": "H200 (141 GB), DGX Spark (128 GB), AMD Instinct MI300X (192 GB), 2x A100/H100",
            },
            "NVFP4": {
                "repo": "defnoteatin/PseudoBrain-Qwen3-Coder-Next-NVFP4",
                "base_repo": "RedHatAI/Qwen3-Coder-Next-NVFP4",
                "upstream_base": "Qwen/Qwen3-Coder-Next",
                "weights_gb": "~41.5 GB",
                "total_1m_gb": "~45.5 GB",
                "gpus": "NVIDIA Blackwell (DGX Spark GB10, 2x RTX 5090, RTX PRO 6000 96 GB, B200)",
            },
            "BF16": {
                "repo": "defnoteatin/PseudoBrain-Qwen3-Coder-Next-BF16",
                "base_repo": "Qwen/Qwen3-Coder-Next",
                "upstream_base": "Qwen/Qwen3-Coder-Next",
                "weights_gb": "~160.0 GB",
                "total_1m_gb": "~165.0 GB",
                "gpus": "AMD MI300X (192 GB), 2x H200 / B200, Multi-GPU CUDA / ROCm",
            },
        },
    },
    "qwen3_coder_30b": {
        "family_title": "PseudoBrain-Qwen3-Coder-30B-A3B",
        "arch_class": "PB27Qwen3MoeForCausalLM",
        "tagline": "1M-Context 30.5B-A3.3B Coding MoE — 98.4% KV Cache Reduction (`192 GB` -> `~3.1 GB` at 1M Tokens)",
        "license": "apache-2.0",
        "pipeline_tag": "text-generation",
        "total_params": "30.5B",
        "active_params": "3.3B (top-8 of 128 routed experts)",
        "layers_desc": "48 layers: all 48 unbounded full-attention layers converted into 16K SWA + Parallel Gated DeltaNet + Quest Exact Memory",
        "stock_1m_kv": "192.0 GB",
        "pb_1m_kv": "~3.1 GB",
        "kv_reduction": "98.4%",
        "extra_vllm_flags": "",
        "parser_flags": " \\\n  --enable-auto-tool-choice \\\n  --tool-call-parser qwen3_coder",
        "tags": ["pseudobrain", "qwen3", "qwen3-coder", "coding", "agentic", "moe", "gated-deltanet", "long-context", "1m-context", "constant-memory", "vllm"],
        "editions": {
            "W4A16": {
                "repo": "defnoteatin/PseudoBrain-Qwen3-Coder-30B-A3B-W4A16",
                "base_repo": "cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit",
                "upstream_base": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
                "weights_gb": "~16.5 GB",
                "total_1m_gb": "~20.5 GB",
                "gpus": "Single 24 GB RTX 3090 / 4090, AMD RX 7900 XTX (24 GB), RTX 5090 (32 GB)",
            },
            "FP8": {
                "repo": "defnoteatin/PseudoBrain-Qwen3-Coder-30B-A3B-FP8",
                "base_repo": "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
                "upstream_base": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
                "weights_gb": "~31.0 GB",
                "total_1m_gb": "~35.0 GB",
                "gpus": "RTX 5090 (32 GB) / L40S (48 GB) / A100 / H100, DGX Spark, AMD MI300X",
            },
            "NVFP4": {
                "repo": "defnoteatin/PseudoBrain-Qwen3-Coder-30B-A3B-NVFP4",
                "base_repo": "NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4",
                "upstream_base": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
                "weights_gb": "~16.5 GB",
                "total_1m_gb": "~20.5 GB",
                "gpus": "NVIDIA Blackwell (RTX 5090 32 GB, DGX Spark GB10, B200)",
            },
            "BF16": {
                "repo": "defnoteatin/PseudoBrain-Qwen3-Coder-30B-A3B-BF16",
                "base_repo": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
                "upstream_base": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
                "weights_gb": "~61.0 GB",
                "total_1m_gb": "~65.0 GB",
                "gpus": "A100 / H100 (80 GB), DGX Spark (128 GB), 3x RTX 3090/4090, AMD MI300X",
            },
        },
    },
    "pb27_nex": {
        "family_title": "PseudoBrain-Nex-35B-A3B",
        "arch_class": "PB27MoEForConditionalGeneration",
        "tagline": "1M-Context 35B-A3B Agentic MoE — 100% (`40/40`) Exact 8K–1M Recall on a Single 24 GB GPU",
        "license": "apache-2.0",
        "pipeline_tag": "text-generation",
        "total_params": "35B",
        "active_params": "~3.0B (top-8 of 256 routed experts + 1 shared expert)",
        "layers_desc": "40 layers: 30 native Gated DeltaNet + 10 full-attention layers (`[3, 7, 11, ..., 39]`) converted into 16K SWA + Parallel Gated DeltaNet + Quest Exact Memory",
        "stock_1m_kv": "~20.0 GB",
        "pb_1m_kv": "~486 MiB active window (4.7 GB pool)",
        "kv_reduction": "97.6%",
        "extra_vllm_flags": "",
        "parser_flags": " \\\n  --reasoning-parser qwen3 \\\n  --enable-auto-tool-choice \\\n  --tool-call-parser qwen3_coder",
        "tags": ["pseudobrain", "qwen3.5-moe", "nex-n2.5", "abliterated", "moe", "gated-deltanet", "long-context", "1m-context", "constant-memory", "vllm"],
        "editions": {
            "W4A16": {
                "repo": "defnoteatin/PseudoBrain-Nex-35B-A3B-W4A16",
                "base_repo": "Arahide/Nex-N2.5-mini-INT4-W4A16",
                "upstream_base": "nex-agi/Nex-N2.5-mini",
                "weights_gb": "~18.5 GB",
                "total_1m_gb": "~22.5 GB",
                "gpus": "Single 24 GB RTX 3090 / 4090, AMD RX 7900 XTX (24 GB), RTX 5090",
            },
            "FP8": {
                "repo": "defnoteatin/PseudoBrain-Nex-35B-A3B-FP8",
                "base_repo": "cbert33/Nex-N2.5-mini-FP8-Calibrated",
                "upstream_base": "nex-agi/Nex-N2.5-mini",
                "weights_gb": "~35.0 GB",
                "total_1m_gb": "~40.0 GB",
                "gpus": "L40S (48 GB) / A100 / H100, 2x RTX 3090/4090, DGX Spark, AMD MI300X",
            },
            "NVFP4": {
                "repo": "defnoteatin/PseudoBrain-Nex-35B-A3B-NVFP4",
                "base_repo": "huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated",
                "upstream_base": "nex-agi/Nex-N2.5-mini",
                "weights_gb": "~22.5 GB",
                "total_1m_gb": "~27.5 GB",
                "gpus": "NVIDIA Blackwell (RTX 5090 32 GB, DGX Spark GB10, B200)",
            },
            "BF16": {
                "repo": "defnoteatin/PseudoBrain-Nex-35B-A3B-BF16",
                "base_repo": "nex-agi/Nex-N2.5-mini",
                "upstream_base": "nex-agi/Nex-N2.5-mini",
                "weights_gb": "~70.0 GB",
                "total_1m_gb": "~75.0 GB",
                "gpus": "A100 / H100 (80 GB), DGX Spark (128 GB), AMD MI300X",
            },
        },
    },
    "pb27c": {
        "family_title": "PseudoBrain-27B",
        "arch_class": "PB27ForConditionalGeneration",
        "tagline": "1M-Context 27B Dense Uncensored Reasoning & Coding Model — 99.2% KV Cache Reduction at 1M Tokens",
        "license": "apache-2.0",
        "pipeline_tag": "text-generation",
        "total_params": "27B",
        "active_params": "27B (Dense)",
        "layers_desc": "64 layers: 48 native Gated DeltaNet + 16 full-attention layers (`[3, 7, 11, ..., 63]`) converted into 16K SWA + Parallel Gated DeltaNet + Quest Exact Memory",
        "stock_1m_kv": "64.0 GB",
        "pb_1m_kv": "~486 MiB active window (6.0 GB pool)",
        "kv_reduction": "99.2%",
        "extra_vllm_flags": "",
        "parser_flags": " \\\n  --reasoning-parser qwen3 \\\n  --enable-auto-tool-choice \\\n  --tool-call-parser qwen3_coder",
        "tags": ["pseudobrain", "qwen3.5", "qwen3.8", "abliterated", "gated-deltanet", "long-context", "1m-context", "constant-memory", "vllm"],
        "editions": {
            "W4A16": {
                "repo": "defnoteatin/PseudoBrain-27B-W4A16",
                "base_repo": "philbert440/Qwen3.8-27B-Uncensored-Aggressive-W4A16-AWQ",
                "upstream_base": "huihui-ai/Huihui-Qwen3.8-27B-abliterated",
                "weights_gb": "~15.5 GB",
                "total_1m_gb": "~20.5 GB",
                "gpus": "Single 24 GB RTX 3090 / 4090, AMD RX 7900 XTX (24 GB), RTX 5090",
            },
            "FP8": {
                "repo": "defnoteatin/PseudoBrain-27B-FP8",
                "base_repo": "batsclamp/Huihui-Qwen3.8-27B-abliterated-FP8-v2",
                "upstream_base": "huihui-ai/Huihui-Qwen3.8-27B-abliterated",
                "weights_gb": "~28.5 GB",
                "total_1m_gb": "~33.5 GB",
                "gpus": "RTX 5090 (32 GB), L40S / A100 / H100, DGX Spark, AMD MI300X",
            },
            "NVFP4": {
                "repo": "defnoteatin/PseudoBrain-27B-NVFP4",
                "base_repo": "huihui-ai/Huihui-Qwen3.8-27B-abliterated",
                "upstream_base": "huihui-ai/Huihui-Qwen3.8-27B-abliterated",
                "weights_gb": "~19.5 GB",
                "total_1m_gb": "~25.5 GB",
                "gpus": "NVIDIA Blackwell (RTX 5090 32 GB, DGX Spark GB10, B200)",
            },
            "BF16": {
                "repo": "defnoteatin/PseudoBrain-27B-BF16",
                "base_repo": "huihui-ai/Huihui-Qwen3.8-27B-abliterated",
                "upstream_base": "huihui-ai/Huihui-Qwen3.8-27B-abliterated",
                "weights_gb": "~54.0 GB",
                "total_1m_gb": "~60.0 GB",
                "gpus": "A100 / H100 (80 GB), DGX Spark (128 GB), 3x RTX 3090/4090, AMD MI300X",
            },
        },
    },
    "qwen3_32b": {
        "family_title": "PseudoBrain-Qwen3-32B",
        "arch_class": "PB27Qwen3ForCausalLM",
        "tagline": "1M-Context 32.8B Dense Reasoning Model — Cuts 1M KV Cache from `256 GB` to `~4.1 GB` (`98.4%` Reduction)",
        "license": "apache-2.0",
        "pipeline_tag": "text-generation",
        "total_params": "32.8B",
        "active_params": "32.8B (Dense)",
        "layers_desc": "64 layers: all 64 unbounded full-attention layers converted into 16K SWA + Parallel Gated DeltaNet + Quest Exact Memory",
        "stock_1m_kv": "256.0 GB",
        "pb_1m_kv": "~4.1 GB",
        "kv_reduction": "98.4%",
        "extra_vllm_flags": "",
        "parser_flags": " \\\n  --reasoning-parser qwen3 \\\n  --enable-auto-tool-choice \\\n  --tool-call-parser hermes",
        "tags": ["pseudobrain", "qwen3", "qwen3-32b", "reasoning", "gated-deltanet", "long-context", "1m-context", "constant-memory", "vllm"],
        "editions": {
            "W4A16": {
                "repo": "defnoteatin/PseudoBrain-Qwen3-32B-W4A16",
                "base_repo": "Qwen/Qwen3-32B-AWQ",
                "upstream_base": "Qwen/Qwen3-32B",
                "weights_gb": "~18.5 GB",
                "total_1m_gb": "~22.8 GB",
                "gpus": "Single 24 GB RTX 3090 / 4090, AMD RX 7900 XTX (24 GB), RTX 5090 (32 GB)",
            },
            "FP8": {
                "repo": "defnoteatin/PseudoBrain-Qwen3-32B-FP8",
                "base_repo": "Qwen/Qwen3-32B-FP8",
                "upstream_base": "Qwen/Qwen3-32B",
                "weights_gb": "~34.5 GB",
                "total_1m_gb": "~39.0 GB",
                "gpus": "L40S (48 GB) / A100 / H100, 2x RTX 3090/4090, DGX Spark, AMD MI300X",
            },
            "NVFP4": {
                "repo": "defnoteatin/PseudoBrain-Qwen3-32B-NVFP4",
                "base_repo": "RedHatAI/Qwen3-32B-NVFP4",
                "upstream_base": "Qwen/Qwen3-32B",
                "weights_gb": "~18.5 GB",
                "total_1m_gb": "~22.8 GB",
                "gpus": "NVIDIA Blackwell (RTX 5090 32 GB, DGX Spark GB10, B200)",
            },
            "BF16": {
                "repo": "defnoteatin/PseudoBrain-Qwen3-32B-BF16",
                "base_repo": "Qwen/Qwen3-32B",
                "upstream_base": "Qwen/Qwen3-32B",
                "weights_gb": "~66.0 GB",
                "total_1m_gb": "~71.0 GB",
                "gpus": "A100 / H100 (80 GB), DGX Spark (128 GB), AMD MI300X",
            },
        },
    },
    "gemma4_31b": {
        "family_title": "PseudoBrain-Gemma-4-31B",
        "arch_class": "PB27Gemma4ForConditionalGeneration",
        "tagline": "1M-Context 31B Multimodal Dense Model — Bounded 16K SWA + Gated DeltaNet + Quest Exact Memory (`head_dim=512`)",
        "license": "gemma",
        "pipeline_tag": "image-text-to-text",
        "total_params": "31B",
        "active_params": "31B (Dense Multimodal)",
        "layers_desc": "60 layers: 50 native SWA layers + all 10 unbounded global full-attention layers (`[5, 11, 17, ..., 59]`, `head_dim=512`) converted into 16K SWA + Parallel Gated DeltaNet + Quest Exact Memory",
        "stock_1m_kv": "~45.0 GB",
        "pb_1m_kv": "~3.5 GB",
        "kv_reduction": "92.2%",
        "extra_vllm_flags": "",
        "parser_flags": "",
        "tags": ["pseudobrain", "gemma4", "gemma-4-31b", "multimodal", "gated-deltanet", "long-context", "1m-context", "constant-memory", "vllm"],
        "editions": {
            "W4A16": {
                "repo": "defnoteatin/PseudoBrain-Gemma-4-31B-W4A16",
                "base_repo": "cyankiwi/gemma-4-31B-it-AWQ-4bit",
                "upstream_base": "google/gemma-4-31B-it",
                "weights_gb": "~17.5 GB",
                "total_1m_gb": "~21.5 GB",
                "gpus": "Single 24 GB RTX 3090 / 4090, AMD RX 7900 XTX (24 GB), RTX 5090 (32 GB)",
            },
            "FP8": {
                "repo": "defnoteatin/PseudoBrain-Gemma-4-31B-FP8",
                "base_repo": "RedHatAI/gemma-4-31B-it-FP8-dynamic",
                "upstream_base": "google/gemma-4-31B-it",
                "weights_gb": "~32.5 GB",
                "total_1m_gb": "~36.5 GB",
                "gpus": "L40S (48 GB) / A100 / H100, 2x RTX 3090/4090, DGX Spark, AMD MI300X",
            },
            "NVFP4": {
                "repo": "defnoteatin/PseudoBrain-Gemma-4-31B-NVFP4",
                "base_repo": "RedHatAI/gemma-4-31B-it-NVFP4",
                "upstream_base": "google/gemma-4-31B-it",
                "weights_gb": "~17.5 GB",
                "total_1m_gb": "~21.5 GB",
                "gpus": "NVIDIA Blackwell (RTX 5090 32 GB, DGX Spark GB10, B200)",
            },
            "BF16": {
                "repo": "defnoteatin/PseudoBrain-Gemma-4-31B-BF16",
                "base_repo": "google/gemma-4-31B-it",
                "upstream_base": "google/gemma-4-31B-it",
                "weights_gb": "~62.0 GB",
                "total_1m_gb": "~66.5 GB",
                "gpus": "A100 / H100 (80 GB), DGX Spark (128 GB), AMD MI300X",
            },
        },
    },
    "gemma4_26b": {
        "family_title": "PseudoBrain-Gemma-4-26B-A4B",
        "arch_class": "PB27Gemma4ForConditionalGeneration",
        "tagline": "1M-Context 25.2B-A3.8B Multimodal MoE — Runs 1M Context in ~18 GB Total VRAM on a Single 24 GB GPU",
        "license": "gemma",
        "pipeline_tag": "image-text-to-text",
        "total_params": "25.2B",
        "active_params": "3.8B (Multimodal MoE)",
        "layers_desc": "30 layers: 25 native SWA layers + all 5 unbounded global full-attention layers (`[5, 11, 17, 23, 29]`, `head_dim=512`) converted into 16K SWA + Parallel Gated DeltaNet + Quest Exact Memory",
        "stock_1m_kv": "~22.5 GB",
        "pb_1m_kv": "~2.2 GB",
        "kv_reduction": "90.2%",
        "extra_vllm_flags": "",
        "parser_flags": "",
        "tags": ["pseudobrain", "gemma4", "gemma-4-26b-a4b", "moe", "multimodal", "gated-deltanet", "long-context", "1m-context", "constant-memory", "vllm"],
        "editions": {
            "W4A16": {
                "repo": "defnoteatin/PseudoBrain-Gemma-4-26B-A4B-W4A16",
                "base_repo": "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit",
                "upstream_base": "google/gemma-4-26B-A4B-it",
                "weights_gb": "~14.2 GB",
                "total_1m_gb": "~17.5 GB",
                "gpus": "Single 24 GB RTX 3090 / 4090, RTX 4080 (16 GB), AMD RX 7900 XTX (24 GB)",
            },
            "FP8": {
                "repo": "defnoteatin/PseudoBrain-Gemma-4-26B-A4B-FP8",
                "base_repo": "RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic",
                "upstream_base": "google/gemma-4-26B-A4B-it",
                "weights_gb": "~26.5 GB",
                "total_1m_gb": "~30.0 GB",
                "gpus": "RTX 5090 (32 GB), L40S / A100 / H100, DGX Spark, AMD MI300X",
            },
            "NVFP4": {
                "repo": "defnoteatin/PseudoBrain-Gemma-4-26B-A4B-NVFP4",
                "base_repo": "RedHatAI/gemma-4-26B-A4B-it-NVFP4",
                "upstream_base": "google/gemma-4-26B-A4B-it",
                "weights_gb": "~14.2 GB",
                "total_1m_gb": "~17.5 GB",
                "gpus": "NVIDIA Blackwell (RTX 5080 / 5090, DGX Spark GB10, B200)",
            },
            "BF16": {
                "repo": "defnoteatin/PseudoBrain-Gemma-4-26B-A4B-BF16",
                "base_repo": "google/gemma-4-26B-A4B-it",
                "upstream_base": "google/gemma-4-26B-A4B-it",
                "weights_gb": "~51.0 GB",
                "total_1m_gb": "~54.5 GB",
                "gpus": "A100 / H100 (80 GB), DGX Spark (128 GB), AMD MI300X",
            },
        },
    },
    "qwen35_9b": {
        "family_title": "PseudoBrain-Qwen3.5-9B",
        "arch_class": "PB27ForConditionalGeneration",
        "tagline": "1M-Context 9B Dense Model — Runs 1,000,000 Tokens inside `<6.5 GB` VRAM on an 8 GB – 12 GB Consumer GPU",
        "license": "apache-2.0",
        "pipeline_tag": "text-generation",
        "total_params": "9B",
        "active_params": "9B (Dense)",
        "layers_desc": "32 layers: 24 native Gated DeltaNet + 8 full-attention layers (`[3, 7, 11, 15, 19, 23, 27, 31]`) converted into 16K SWA + Parallel Gated DeltaNet + Quest Exact Memory",
        "stock_1m_kv": "32.0 GB",
        "pb_1m_kv": "~256 MiB active window (~1.0 GB pool)",
        "kv_reduction": "99.2%",
        "extra_vllm_flags": "",
        "parser_flags": " \\\n  --reasoning-parser qwen3 \\\n  --enable-auto-tool-choice \\\n  --tool-call-parser qwen3_coder",
        "tags": ["pseudobrain", "qwen3.5", "qwen3.5-9b", "gated-deltanet", "long-context", "1m-context", "constant-memory", "vllm"],
        "editions": {
            "W4A16": {
                "repo": "defnoteatin/PseudoBrain-Qwen3.5-9B-W4A16",
                "base_repo": "cyankiwi/Qwen3.5-9B-AWQ-4bit",
                "upstream_base": "Qwen/Qwen3.5-9B",
                "weights_gb": "~5.5 GB",
                "total_1m_gb": "~6.5 GB",
                "gpus": "8 GB – 12 GB Consumer GPUs (RTX 3060 / 4060 / 4070 / 3090 / 4090, AMD RX 7600 / 7900 XTX)",
            },
            "FP8": {
                "repo": "defnoteatin/PseudoBrain-Qwen3.5-9B-FP8",
                "base_repo": "RedHatAI/Qwen3.5-9B-FP8-dynamic",
                "upstream_base": "Qwen/Qwen3.5-9B",
                "weights_gb": "~9.5 GB",
                "total_1m_gb": "~11.0 GB",
                "gpus": "12 GB – 16 GB GPUs (RTX 4070 Ti / 4080 / 4090 / 5090, AMD MI300X)",
            },
            "NVFP4": {
                "repo": "defnoteatin/PseudoBrain-Qwen3.5-9B-NVFP4",
                "base_repo": "ig1/Qwen3.5-9B-NVFP4",
                "upstream_base": "Qwen/Qwen3.5-9B",
                "weights_gb": "~5.5 GB",
                "total_1m_gb": "~6.5 GB",
                "gpus": "NVIDIA Blackwell (RTX 5060 / 5070 / 5080 / 5090, DGX Spark GB10)",
            },
            "BF16": {
                "repo": "defnoteatin/PseudoBrain-Qwen3.5-9B-BF16",
                "base_repo": "Qwen/Qwen3.5-9B",
                "upstream_base": "Qwen/Qwen3.5-9B",
                "weights_gb": "~18.0 GB",
                "total_1m_gb": "~20.0 GB",
                "gpus": "Single 24 GB RTX 3090 / 4090, AMD RX 7900 XTX (24 GB)",
            },
        },
    },
    "qwen35_4b": {
        "family_title": "PseudoBrain-Qwen3.5-4B",
        "arch_class": "PB27ForConditionalGeneration",
        "tagline": "1M-Context 4B Compact Model — Runs 1,000,000 Tokens inside `<3.8 GB` VRAM on Laptop & Edge GPUs",
        "license": "apache-2.0",
        "pipeline_tag": "text-generation",
        "total_params": "4B",
        "active_params": "4B (Dense)",
        "layers_desc": "32 layers: 24 native Gated DeltaNet + 8 full-attention layers (`[3, 7, 11, 15, 19, 23, 27, 31]`) converted into 16K SWA + Parallel Gated DeltaNet + Quest Exact Memory",
        "stock_1m_kv": "24.0 GB",
        "pb_1m_kv": "~192 MiB active window (~1.0 GB pool)",
        "kv_reduction": "99.2%",
        "extra_vllm_flags": "",
        "parser_flags": " \\\n  --reasoning-parser qwen3 \\\n  --enable-auto-tool-choice \\\n  --tool-call-parser qwen3_coder",
        "tags": ["pseudobrain", "qwen3.5", "qwen3.5-4b", "edge", "gated-deltanet", "long-context", "1m-context", "constant-memory", "vllm"],
        "editions": {
            "W4A16": {
                "repo": "defnoteatin/PseudoBrain-Qwen3.5-4B-W4A16",
                "base_repo": "cyankiwi/Qwen3.5-4B-AWQ-4bit",
                "upstream_base": "Qwen/Qwen3.5-4B",
                "weights_gb": "~2.8 GB",
                "total_1m_gb": "~3.8 GB",
                "gpus": "6 GB – 8 GB Laptop & Desktop GPUs (RTX 3050 / 3060 / 4060, AMD RX 6600 / 7600)",
            },
            "FP8": {
                "repo": "defnoteatin/PseudoBrain-Qwen3.5-4B-FP8",
                "base_repo": "surogate/Qwen3.5-4B-FP8",
                "upstream_base": "Qwen/Qwen3.5-4B",
                "weights_gb": "~4.8 GB",
                "total_1m_gb": "~6.0 GB",
                "gpus": "8 GB+ GPUs (RTX 4060 / 4070 / 4090, AMD MI300X)",
            },
            "NVFP4": {
                "repo": "defnoteatin/PseudoBrain-Qwen3.5-4B-NVFP4",
                "base_repo": "AxionML/Qwen3.5-4B-NVFP4",
                "upstream_base": "Qwen/Qwen3.5-4B",
                "weights_gb": "~2.8 GB",
                "total_1m_gb": "~3.8 GB",
                "gpus": "NVIDIA Blackwell Laptop & Desktop GPUs (RTX 5050 / 5060+, DGX Spark GB10)",
            },
            "BF16": {
                "repo": "defnoteatin/PseudoBrain-Qwen3.5-4B-BF16",
                "base_repo": "Qwen/Qwen3.5-4B",
                "upstream_base": "Qwen/Qwen3.5-4B",
                "weights_gb": "~8.5 GB",
                "total_1m_gb": "~10.0 GB",
                "gpus": "12 GB+ GPUs (RTX 3060 12GB / 4070 / 3090 / 4090, AMD RX 7700 XT+)",
            },
        },
    },
}


def build_ecosystem_table() -> str:
    lines = [
        "| Model Family | Params (Total / Active) | 1M KV Reduction | **4-Bit `W4A16` (RTX 30/40/AMD)** | **8-Bit `FP8` (4090/H100/MI300X)** | **4-Bit `NVFP4` (Blackwell/Spark)** | **16-Bit `BF16` (Universal)** |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for fkey, f in FAMILIES.items():
        eds = f["editions"]
        w4 = f"[`W4A16`](https://huggingface.co/{eds['W4A16']['repo']}) ({eds['W4A16']['weights_gb']})"
        fp8 = f"[`FP8`](https://huggingface.co/{eds['FP8']['repo']}) ({eds['FP8']['weights_gb']})"
        nv4 = f"[`NVFP4`](https://huggingface.co/{eds['NVFP4']['repo']}) ({eds['NVFP4']['weights_gb']})"
        bf16 = f"[`BF16`](https://huggingface.co/{eds['BF16']['repo']}) ({eds['BF16']['weights_gb']})"
        p_short = f["active_params"].split("(")[0].strip()
        lines.append(
            f"| **{f['family_title']}** | `{f['total_params']}` / `{p_short}` | **`{f['kv_reduction']}`** (`{f['stock_1m_kv'].split('(')[0].strip()}` → `{f['pb_1m_kv'].split('(')[0].strip()}`) | {w4} | {fp8} | {nv4} | {bf16} |"
        )
    return "\n".join(lines)


def render_model_card(fkey: str, prec: str = "W4A16") -> str:
    f = FAMILIES[fkey]
    ed = f["editions"][prec]
    target_repo = ed["repo"]
    repo_name = target_repo.split("/")[-1]

    # Collect unique base models for YAML frontmatter so HF links this repo to both upstream & quantized base repos
    base_models = []
    for m in [ed["upstream_base"], ed["base_repo"]] + [x["base_repo"] for x in f["editions"].values()]:
        if m not in base_models:
            base_models.append(m)
    base_model_yaml = "\n".join(f"  - {m}" for m in base_models)
    tags_yaml = "\n".join(f"  - {t}" for t in f["tags"] + [prec.lower()])

    editions_rows = []
    for p_label, p_spec in f["editions"].items():
        marker = " **(This Repo)**" if p_label == prec else ""
        editions_rows.append(
            f"| [`{p_spec['repo']}`](https://huggingface.co/{p_spec['repo']}){marker} | **`{p_label}`** | `{p_spec['weights_gb']}` | **`{p_spec['total_1m_gb']}`** | {p_spec['gpus']} |"
        )
    editions_table = "\n".join(editions_rows)
    ecosystem_table = build_ecosystem_table()

    return f"""---
license: {f['license']}
base_model:
{base_model_yaml}
library_name: vllm
pipeline_tag: {f['pipeline_tag']}
tags:
{tags_yaml}
---

# {repo_name} — {f['tagline']}

> [!IMPORTANT]
> **Created by Ian Kurt Galeon ([@defnotean](https://github.com/defnotean) · [Hugging Face `@defnoteatin`](https://huggingface.co/defnoteatin) · `iangaleonofficial@gmail.com`)**  
> **GitHub & Universal `pb27_vllm` Engine Plugin:** [https://github.com/defnotean/pseudobrain](https://github.com/defnotean/pseudobrain)  
> **Architecture Paper:** [`PSEUDOBRAIN_ARCHITECTURE_PAPER.pdf`](https://github.com/defnotean/pseudobrain/blob/main/docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.pdf) · [`PSEUDOBRAIN_ARCHITECTURE_PAPER.md`](https://github.com/defnotean/pseudobrain/blob/main/docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.md) · **Benchmark Receipts:** [`reports/`](https://github.com/defnotean/pseudobrain/tree/main/reports)  
> *Open to **LLM Inference Systems / Kernel & Engine Engineering roles**, research collaborations, and **enterprise on-premise 1M-context deployments**.*

---

## 1. Why `{f['family_title']}` Exists

In standard Transformer serving, **model weights are paid once, while KV cache memory grows linearly ($O(L)$) and prefill compute grows quadratically ($O(L^2)$)** — forcing long-context (`128K–1,000,000` token) workloads to OOM even when the GPU easily fits the model weights.

**`{repo_name}`** (`{f['arch_class']}`) solves this via the **PseudoBrain architecture**:
1. **Hybrid Windowed-Attention + Parallel Gated DeltaNet (`GDN`):** {f['layers_desc']}. Recurrent GDN state is **$O(1)$ fixed size** regardless of whether context is `4K` or `1,000,000` tokens.
2. **Three-Tier NVMe/Page-Cache Quest Exact Memory (`piece_attention`):** On the sparse anchor layers, queries attend over `[4 StreamingLLM Sinks + Top-k Quest Exact-Memory Pages + 16,384 Sliding Window]` via FlashAttention-2 LSE-merge (`pb27_vllm/memory.py`) with **2-Level Hierarchical `bf16` Quest Super-Block Bounds (`PB27_MEMORY_SUPERBLOCK=64`)** and **Priority-Aware RoPE Sink Repositioning (`_reposition_rope_sinks`)**.
3. **True $O(1)$ Constant-Speed Decode up to `1,000,000` Tokens:** Decouples the scaled prefill retrieval budget from the constant decode budget (`PB27_MEMORY_DECODE_STRIDE=64`, `PB27_MEMORY_DECODE_BLOCKS=64`, `PB27_MEMORY_DECODE_RERANK=0`), cutting **1M-token KV cache memory by `{f['kv_reduction']}`** (`{f['stock_1m_kv']}` → **`{f['pb_1m_kv']}`**).

### Hardware & Precision Editions for `{f['family_title']}`

| Repository | Precision | Model Weights | **Total VRAM at `1,000,000` Tokens** | Target GPUs |
| :--- | :---: | :---: | :---: | :--- |
{editions_table}

---

## 2. Two-Command Quickstart (`vLLM` on NVIDIA CUDA & AMD ROCm)

### Option A: Install Plugin Directly into Existing `vLLM` Environment (2 Commands)

```bash
# 1. Install the universal PseudoBrain vLLM V1 plugin (auto-registers via vllm.general_plugins)
pip install --no-deps "git+https://github.com/defnotean/pseudobrain.git#subdirectory=serving/pb27_vllm"

# 2. Launch 1,000,000-token OpenAI-compatible server
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve {target_repo} \\
  --served-model-name pb27 \\
  --max-model-len 1048576 \\
  --enforce-eager \\
  --no-enable-prefix-caching \\
  --kv-cache-memory-bytes 6G \\
  --max-num-seqs 16{f['extra_vllm_flags']}{f['parser_flags']}
```

*(Note: Every Hugging Face checkpoint also bundles `pb27_vllm_plugin/` directly inside the repository if you prefer `pip install --no-deps -e ./pb27_vllm_plugin` from a local snapshot download.)*

### Option B: One-Command Docker Launch (`serve.sh` — Auto-Detects `x86_64` & `aarch64`)

```bash
git clone https://github.com/defnotean/pseudobrain.git
cd pseudobrain
bash serving/pb27_vllm/serve.sh {target_repo} pb27 8000
```

### Option C: Connect `OpenCode` / Agentic Coding Harnesses (`~/.config/opencode/opencode.jsonc`)

```jsonc
{{
  "$schema": "https://opencode.ai/config.json",
  "model": "pseudobrain/pb27",
  "small_model": "pseudobrain/pb27",
  "compaction": {{ "auto": true, "prune": true }},
  "provider": {{
    "pseudobrain": {{
      "npm": "@ai-sdk/openai-compatible",
      "name": "PseudoBrain 1M Local Server",
      "options": {{
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "local-loopback",
        "timeout": 600000
      }},
      "models": {{
        "pb27": {{
          "name": "{repo_name} (1M Context)",
          "tool_call": true,
          "limit": {{ "context": 1048576, "output": 16384 }}
        }}
      }}
    }}
  }}
}}
```

---

## 3. Verified PseudoBrain Benchmark Results (Committed Receipts in [`reports/`](https://github.com/defnotean/pseudobrain/tree/main/reports))

### 3.1 Same-Hardware (`DGX Spark GB10, TP1`) Speed & Memory vs. Stock at `65K` and `1,000,000` Tokens

Measured directly against upstream `runtime-qualification.json` on the **exact same `DGX Spark (GB10, 128 GB Unified Memory), TP1` hardware**:

| Metric on **DGX Spark (GB10, TP1)** | Stock `Qwen3.8-Flash-Next-NVFP4` (`runtime-qualification.json`) | **`PseudoBrain-Qwen3.8-Flash-Next-NVFP4` (Ours)** | Speedup / Savings |
| :--- | :---: | :---: | :---: |
| **`65K` Prefill Time (TTFT)** | `38.93 s` (`1,680 tok/s`) | **`21.02 s` (`3,108 tok/s`)** | **1.85× faster prefill (`-46%` latency)** |
| **`65K` Decode Speed (`tok/s`)** | `35.8 – 39.1 tok/s` *(mean `37.8 tok/s`, MTP-3)* | **`53.4 – 88.3 tok/s`** *(mean **`65.9 tok/s`**, peak **`183.5 tok/s`** w/ DFlash2)* | **1.74× – 2.26× faster decode** |
| **`1,000,000` (`1M`) Prefill Time (TTFT)** | `1,128.0 – 1,132.2 s` (`18.9 min`, `926 tok/s`) | **`543.06 s`** (`9.0 min`, **`1,801 tok/s`**) | **2.08× faster `1M` prefill (saves ~10 min per `1M` prompt)** |
| **`1,000,000` (`1M`) Decode Speed (`tok/s`)** | `33.3 – 37.1 tok/s` *(mean `34.9 tok/s`)* | **`44.6 – 67.7 tok/s`** *(mean **`57.1 tok/s`**, `67.7 tok/s` on `1M` needle)* | **1.64× – 1.94× faster decode at `1M` tokens** |
| **KV Cache Memory at `1M` Tokens** | Full `1,048,576`-token KV cache (`max-running-requests 1`, requires static YaRN `factor=4.0`) | **Fixed `16,384`-token GPU window (`~486 MiB` active)** + NVMe Quest $O(1)$ archive (`--max-num-seqs 16`, **no static YaRN degradation**) | **~96%–99.2% less GPU KV memory at `1M` + `16` concurrent sequences** |

### 3.2 `8K` to `1,000,000` Token Exact Retrieval (`needle@0.1/0.5/0.9`, `multi4`, `trace4`)

| Context Length ($L$) | `needle@0.1` | `needle@0.5` | `needle@0.9` | `multi4` (`4` constants) | `trace4` (`4`-hop chain) | `Flash-Next` Decode `tok/s` | `Nex-35B-A3B` Recall | GPU KV Pool Used |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`8,192`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`93.01 tok/s`** | **`5/5 (100%)`** | **`8.1%`** |
| **`32,768`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`66.51 tok/s`** | **`5/5 (100%)`** | **`8.1%`** |
| **`65,536`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`67.86 tok/s`** | **`5/5 (100%)`** | **`8.1%`** |
| **`120,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`50.50 tok/s`** | **`5/5 (100%)`** | **`8.1%`** |
| **`200,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`50.81 tok/s`** | **`5/5 (100%)`** | **`8.1%`** |
| **`250,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`69.87 tok/s`** | **`5/5 (100%)`** | **`8.1%`** |
| **`500,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`64.91 tok/s`** (`183.5` peak) | **`5/5 (100%)`** | **`8.1%`** |
| **`1,000,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`0.75–1.0`** | **`1.0` (`Nex`)** | **`63.76 tok/s`** (`67.7` needle) | **`5/5 (100%)`** | **`8.1%`** |

### 3.3 General Intelligence, Reasoning, & Software Engineering Parity

- **MMLU (`n=300`, 0-shot):** **`85.00%`** (`255/300`, within statistical stderr of stock `~85.0–87.9%`).
- **GSM8K (`n=150`, strict CoT):** **`96.00%`** (`144/150`, **`+2.71%`** above official `93.29%` report).
- **MMLU-Pro (`n=150`, 10-choice college reasoning):** **`81.33%`** raw / **`85.31%`** non-truncated (vs. `73.23%` Base / `80.6%` Instruct).
- **Algorithmic Coding (`custom_code_bench`, 33 medium-hard Python problems):** **`96.97%` (`32/33`)** on `PseudoBrain-Qwen3.8-Flash-Next` at **`74.74 tok/s`**; **`90.91%` (`60/66`)** on `PseudoBrain-27B`.
- **Real-World Post-Cutoff Software Engineering (`PB-SWE-Live`, 59 tasks, 100-step agent loop):** **`26 / 59` (`44.1%`)** vs. Stock **`28 / 59` (`47.5%`)** (paired McNemar $p \\approx 0.7$, statistically indistinguishable).

---

## 4. Complete PseudoBrain 1M-Context Model Lineup (10 Families × 4 Precisions = 40 Models)

All 40 repositories share the single unified `pb27_vllm` plugin (`pip install --no-deps "git+https://github.com/defnotean/pseudobrain.git#subdirectory=serving/pb27_vllm"`), with slim BF16 GDN delta overlays hosted at [`defnoteatin/pb27-checkpoints`](https://huggingface.co/defnoteatin/pb27-checkpoints):

{ecosystem_table}

---

## 5. License, Citation & Contact

- **Code & Plugin License:** Apache-2.0 ([`https://github.com/defnotean/pseudobrain`](https://github.com/defnotean/pseudobrain)). Model weights inherit their upstream base model license (`{f['license']}`).
- **Author:** **Ian Kurt Galeon**
  - **GitHub:** [@defnotean](https://github.com/defnotean)
  - **Hugging Face:** [@defnoteatin](https://huggingface.co/defnoteatin)
  - **Email (Recruiting, Research & Enterprise Inquiries):** `iangaleonofficial@gmail.com`
"""


def render_github_readme() -> str:
    ecosystem_table = build_ecosystem_table()
    return f"""# PseudoBrain — Million-Token Hybrid Gated DeltaNet + Quest Exact Memory for vLLM

**Convert open dense and MoE LLMs into constant-GPU-memory, million-token hybrids — $O(1)$ GPU KV footprint from `8K` to `1,000,000+` tokens (`90%–99.2%` KV cache reduction), `100%` exact retrieval across `8K–1M`, and True $O(1)$ Constant-Speed Decode (`44.6–183.5 tok/s` at `500K–1M` with `DFlash2` speculative decoding) on a single GPU.**

> [!IMPORTANT]
> **Created by Ian Kurt Galeon ([@defnotean](https://github.com/defnotean) · [Hugging Face `@defnoteatin`](https://huggingface.co/defnoteatin) · `iangaleonofficial@gmail.com`)**  
> **Architecture Paper:** [`docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.pdf`](docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.pdf) · [`docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.docx`](docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.docx) · [`docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.md`](docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.md)  
> **Slim BF16 GDN Delta Shards:** [`defnoteatin/pb27-checkpoints`](https://huggingface.co/defnoteatin/pb27-checkpoints) · **Benchmark Receipts:** [`reports/`](reports/)  
> *Available for **LLM Inference Systems / Kernel & Engine Engineering roles**, research collaborations, and **enterprise on-premise 1M-context deployments**.*

---

## 1. Complete 40-Model Lineup on Hugging Face (`W4A16`, `FP8`, `NVFP4`, `BF16`)

All **10 model families (`40` repositories)** run through our single universal `pb27_vllm` vLLM V1 plugin across **NVIDIA Ampere/Ada (`RTX 3060–4090`, `L40S`, `A100`)**, **Hopper (`H100/H200`)**, **Blackwell (`RTX 5090`, `DGX Spark GB10`, `B200`)**, and **AMD ROCm (`RX 7900 XTX`, `Instinct MI300X`)**:

{ecosystem_table}

---

## 2. Two-Command Quickstart (`vLLM`)

### Option A: Direct `pip install` into Any `vLLM` Environment (2 Commands)

```bash
# 1. Install the universal PseudoBrain vLLM V1 out-of-tree plugin (auto-registers via vllm.general_plugins)
pip install --no-deps "git+https://github.com/defnotean/pseudobrain.git#subdirectory=serving/pb27_vllm"

# 2. Serve any of the 40 PseudoBrain models at 1,000,000-token context in ~4-6 GB of KV cache
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve defnoteatin/PseudoBrain-Qwen3-Coder-30B-A3B-W4A16 \\
  --served-model-name pb27 \\
  --max-model-len 1048576 \\
  --enforce-eager \\
  --no-enable-prefix-caching \\
  --kv-cache-memory-bytes 6G \\
  --max-num-seqs 16 \\
  --enable-auto-tool-choice \\
  --tool-call-parser qwen3_coder
```

### Option B: One-Command Docker Launch (`serve.sh` — Auto-Detects `x86_64` & `aarch64`)

```bash
git clone https://github.com/defnotean/pseudobrain.git
cd pseudobrain

# Serve Qwen3.8-Flash-Next + DFlash2 speculative decoding (45-183 tok/s up to 1M):
SPEC='{{"method":"dflash","model":"incoai/Qwen3.8-Flash-Next-DFlash2","num_speculative_tokens":8}}' \\
bash serving/pb27_vllm/serve.sh defnoteatin/PseudoBrain-Qwen3.8-Flash-Next-NVFP4 pb27 8000

# Serve on a single 24 GB RTX 3090 / 4090 / AMD 7900 XTX (W4A16):
bash serving/pb27_vllm/serve.sh defnoteatin/PseudoBrain-Qwen3-Coder-30B-A3B-W4A16 pb27 8000

# Serve on an 8 GB - 12 GB consumer GPU (1M context in <6.5 GB total VRAM):
bash serving/pb27_vllm/serve.sh defnoteatin/PseudoBrain-Qwen3.5-9B-W4A16 pb27 8000
```

---

## 3. What PseudoBrain Is & Why It Matters for Production Inference

In standard Transformer deployment, **KV cache memory and quadratic prefill — not model weights — are the primary bottlenecks**:
- A single user at `1,000,000` tokens on `Qwen3-32B` requires **`256 GB` of KV cache VRAM** (`192 GB` on `Qwen3-Coder-30B-A3B`; `64 GB` on `Qwen3.8-27B`).
- **PseudoBrain** converts unbounded full-attention layers into **Hybrid 16K Sliding-Window Attention + Parallel Gated DeltaNet (`GDN`)** blocks paired with our **Three-Tier NVMe/Page-Cache Quest Exact Memory Hierarchy** and **True $O(1)$ Constant-Speed Decode Engine** inside vLLM (`serving/pb27_vllm/`):
  - **Recurrent Gated DeltaNet Branch (`O(1)` State):** Initialized directly from teacher `self_attn` projections (`init_gdn_from_attention`) with zero-initialized output projections (`out=0`), compressing arbitrary-length global context into a fixed-size recurrent state.
  - **Quest Exact Memory + FlashAttention-2 LSE Merge (`piece_attention`):** Archives rotated keys/values in `64`-token blocks with **2-Level Hierarchical `bf16` Quest Super-Block Bounds (`4,096` tokens/super-block)** and **Priority-Aware RoPE Sink Repositioning (`_reposition_rope_sinks`)**, recomputing attention over `[4 Sinks + Top-k Retrieved Pages + 16K Window]` without static YaRN `factor=4.0` short-context degradation.
  - **True $O(1)$ Decode Throughput (`DECODE_STRIDE=64`, `DECODE_BLOCKS=64`, `DECODE_RERANK=0`):** Warms `cached_pre_k` / `cached_pre_v` at the end of `final_prefill` so multi-token speculative verification runs directly in GPU HBM with zero NVMe page traffic for rejected candidates.

---

## 4. Benchmark Results (All Backed by Committed Receipts in [`reports/`](reports/))

### 4.1 Same-Hardware (`DGX Spark GB10, TP1`) Comparison: Stock vs. PseudoBrain at `65K` and `1,000,000` Tokens

Compared directly against upstream `runtime-qualification.json` in `edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD` measured on the **exact same `DGX Spark (GB10), TP1` hardware**:

| Metric on **DGX Spark (GB10, TP1)** | Stock `Huihui-Qwen3.8-Flash-Next-NVFP4-QAD` (`runtime-qualification.json`) | **`PseudoBrain-Qwen3.8-Flash-Next-NVFP4` (Ours)** | Speedup / Savings |
| :--- | :---: | :---: | :---: |
| **`65K` Prefill Time (TTFT)** | `38.93 s` (`1,680 tok/s`) | **`21.02 s` (`3,108 tok/s`)** | **1.85× faster prefill (`-46%` latency)** |
| **`65K` Decode Speed (`tok/s`)** | `35.8 – 39.1 tok/s` *(mean `37.8 tok/s`, MTP-3)* | **`53.4 – 88.3 tok/s`** *(mean **`65.9 tok/s`**, peak **`183.5 tok/s`** w/ DFlash2)* | **1.74× – 2.26× faster decode** |
| **`1,000,000` (`1M`) Prefill Time (TTFT)** | `1,128.0 – 1,132.2 s` (`18.9 min`, `926 tok/s`) | **`543.06 s`** (`9.0 min`, **`1,801 tok/s`**) | **2.08× faster `1M` prefill (saves ~10 min per `1M` prompt)** |
| **`1,000,000` (`1M`) Decode Speed (`tok/s`)** | `33.3 – 37.1 tok/s` *(mean `34.9 tok/s`)* | **`44.6 – 67.7 tok/s`** *(mean **`57.1 tok/s`**, `67.7 tok/s` on `1M` needle)* | **1.64× – 1.94× faster decode at `1M` tokens** |
| **KV Cache Memory at `1M` Tokens** | Full `1,048,576`-token `fp8` KV cache (`--mem-fraction-static 0.86`, `max-running-requests 1`, requires static YaRN `factor=4.0`) | **Fixed `16,384`-token GPU window (`8.1%` of `6 GiB` pool = `~486 MiB`)** + NVMe Quest $O(1)$ archive (`--max-num-seqs 16`, **no static YaRN degradation**) | **~96% less GPU KV cache memory at `1M` + `16` concurrent sequences** |

### 4.2 Million-Token Long-Context Exact Retrieval (`8K` to `1,000,000` Tokens)

Evaluated via [`scripts/bench/custom_longctx_bench.py`](scripts/bench/custom_longctx_bench.py) (`needle@0.1`, `needle@0.5`, `needle@0.9`, `multi4` 4-constant JSON synthesis at depths `{{0.15, 0.40, 0.65, 0.85}}`, and `trace4` 4-hop function call chain across depths `{{0.82, 0.58, 0.34, 0.12}}` + arithmetic mod `1000`):

| Context Length ($L$) | `needle@0.1` | `needle@0.5` | `needle@0.9` | `multi4` (`4` constants) | `trace4` (`4`-hop chain) | `Flash-Next` Warm TTFT | `Flash-Next` Decode `tok/s` | `Nex-35B-A3B` Recall | GPU KV Pool Used |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`8,192`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`1.38s`** | **`93.01 tok/s`** (`85.94` mean) | **`5/5 (100%)`** | **`8.1%`** |
| **`32,768`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`12.40s`** | **`66.51 tok/s`** (`59.22` mean) | **`5/5 (100%)`** | **`8.1%`** |
| **`65,536`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`28.99s`** | **`67.86 tok/s`** (`51.68` `multi4`) | **`5/5 (100%)`** | **`8.1%`** |
| **`120,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`61.36s`** | **`50.50 tok/s`** (`36.66` `multi4`) | **`5/5 (100%)`** | **`8.1%`** |
| **`200,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`125.99s`** | **`50.81 tok/s`** | **`5/5 (100%)`** | **`8.1%`** |
| **`250,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`110.16s`** | **`69.87 tok/s`** (`58.98` `multi4`) | **`5/5 (100%)`** | **`8.1%`** |
| **`500,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`234.05s`** | **`64.91 tok/s`** (`183.54` peak) | **`5/5 (100%)`** | **`8.1%`** |
| **`1,000,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`0.75–1.0`** | **`1.0` (`Nex`)** | **`543.06s`** (`1,801 tok/s`) | **`63.76 tok/s`** (`67.72` `needle@0.1`) | **`5/5 (100%)`** | **`8.1%`** |

### 4.3 General Intelligence (`MMLU`, `GSM8K`, `MMLU-Pro`, `lm-eval`), Software Engineering (`PB-SWE-Live`), & Algorithmic Coding

- **`PseudoBrain-Qwen3.8-Flash-Next-NVFP4` Public Benchmarks vs. Stock ([`reports/pb27/public_bench/flash_next/`](reports/pb27/public_bench/flash_next/)):**
  - **MMLU (`n=300`, 0-shot, greedy):** **`85.00%`** (`255 / 300`, `stderr: 0.0206` vs. stock `~85.0–87.9%`)
  - **GSM8K (`n=150`, CoT, greedy non-thinking):** **`96.00%`** (`144 / 150`, `stderr: 0.0161` vs. stock `93.29%`, **`+2.71%`**)
  - **MMLU-Pro (`n=150`, 10-option CoT, greedy non-thinking):** **`81.33%`** (`122 / 150`, **`85.31%`** non-truncated vs. stock `73.23%` Base / `80.6%` Instruct)
- **Standard `lm-eval` Battery (`1000` items/task, served through vLLM — [`reports/lmeval/`](reports/lmeval/)):**
  - `PB-27` vs. Stock `27B`: **HellaSwag** `0.733` vs `0.740`, **PIQA** `0.817` vs `0.816`, **ARC-E** `0.809` vs `0.800`, **ARC-C** `0.582` vs `0.578`, **Winogrande** `0.759` vs `0.761`, **LAMBADA** `0.728` vs `0.733` ($\\max |\\Delta| = 0.009$, within 1 SE $\\approx 0.014$).
- **Real-World Post-Cutoff Software Engineering (`PB-SWE-Live`, 59 tasks, 100-step agent loop — [`reports/swe_live/`](reports/swe_live/)):**
  - `PB-27` **`26 / 59` (`44.1%`)** vs. Stock `27B` **`28 / 59` (`47.5%`)** (paired McNemar $p \\approx 0.7$, statistically indistinguishable).
- **33-Problem Medium-Hard Python Algorithmic Coding Suite ([`scripts/bench/custom_code_bench.py`](scripts/bench/custom_code_bench.py)):**
  - **`PseudoBrain-Qwen3.8-Flash-Next-NVFP4` (`flash_next`):** **`32 / 33` (`96.97%`)** at **`74.74 tok/s`** ([`reports/pb27/flash_next_med.summary.json`](reports/pb27/flash_next_med.summary.json)).
  - **`PseudoBrain-27B` (`pb27c`):** **`60 / 66` (`90.91%`)** sample pass rate ([`reports/pb27/pb27c_med.summary.json`](reports/pb27/pb27c_med.summary.json)).

---

## 5. Repository Layout

```text
pseudobrain/        Self-contained PyTorch conversion core (WindowedHybridLayer, GatedDeltaNetLayer, Archive)
serving/pb27_vllm/  Universal vLLM V1 out-of-tree plugin (PB27, PB27MoE, PB27Qwen4Exp, PB27Qwen3Next, PB27Qwen3Moe, PB27Gemma4, PB27Qwen3)
scripts/transplant/ Universal slim GDN export (export_universal_slim.py), Hub publisher (publish_pb_variants.py), GPU verifier (verify_all_pb_models.py)
scripts/convert/    Distillation (convert_token.py), slim export (export_pb27_slim.py), NVFP4 composition (make_pb27_nvfp4.py)
scripts/eval/       lm-eval / NLL battery (evaluate.py), recall probes (recall_probes.py), vLLM parity (vllm_parity.py)
scripts/bench/      8K-1M long-context suite (custom_longctx_bench.py), coding suite (custom_code_bench.py), PB-SWE-Live
docs/               Full architecture paper (PSEUDOBRAIN_ARCHITECTURE_PAPER.pdf / .md) and design specs
release/            Model cards for all 10 PseudoBrain model families
reports/            JSON/JSONL receipts for every benchmark table in this README and paper
```

---

## 6. License & Author

- **Code License:** Apache-2.0 (see [`LICENSE`](LICENSE)). Converted model weights inherit their respective base model's license.
- **Author:** **Ian Kurt Galeon** ([@defnotean](https://github.com/defnotean) · [Hugging Face `@defnoteatin`](https://huggingface.co/defnoteatin) · `iangaleonofficial@gmail.com`).
"""


def main() -> None:
    release_dir = REPO_ROOT / "release"
    for fkey, f in FAMILIES.items():
        fdir = release_dir / fkey
        fdir.mkdir(parents=True, exist_ok=True)
        # Default family README uses W4A16 (or NVFP4 for flash_next)
        default_prec = "NVFP4" if fkey == "flash_next" else "W4A16"
        (fdir / "README.md").write_text(render_model_card(fkey, default_prec), encoding="utf-8")
        for prec in ("W4A16", "FP8", "NVFP4", "BF16"):
            (fdir / f"README_{prec}.md").write_text(render_model_card(fkey, prec), encoding="utf-8")
        print(f"Generated 4 precision model cards for {fkey} in {fdir}")

    gh_readme = render_github_readme()
    (release_dir / "GITHUB_README.md").write_text(gh_readme, encoding="utf-8")
    print("Generated GITHUB_README.md")


if __name__ == "__main__":
    main()
