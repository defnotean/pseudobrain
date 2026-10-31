"""Universal PB-27 vLLM model classes for Qwen3Next, Qwen3Moe, Qwen3, and Gemma4.

Each class wraps the stock vLLM model implementation so layers listed in
`pb_hybrid_layers` run:
  1. Windowed `self_attn` (`per_layer_sliding_window = pb_attention_window`)
     with Quest/Landmark exact memory retrieval (`memory.windowed_attention_with_memory`)
     and Priority-Aware RoPE sink repositioning.
  2. A parallel Gated DeltaNet recurrent state branch (`self.gdn`,
     `QwenGatedDeltaNetAttention`) summed into the post-attention hidden state.
All non-hybrid layers, MoE routers/experts, norms, and multimodal towers remain
100% identical to the stock vLLM implementation.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe.utils import (
    is_model_fused_shared_expert_compatible,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.gemma4 import (
    Gemma4Attention,
    Gemma4CrossDecoderLayers,
    Gemma4DecoderLayer,
    Gemma4ForCausalLM,
    Gemma4Model,
    Gemma4SelfDecoderLayers,
    _gemma4_layer_weights_mapper,
    _get_text_config,
    gemma4_layer_config,
)
from vllm.model_executor.models.gemma4_mm import (
    Gemma4DummyInputsBuilder,
    Gemma4ForConditionalGeneration,
    Gemma4MultiModalProcessor,
    Gemma4ProcessingInfo,
)
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.qwen3 import (
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3Model,
)
from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeMLP,
    Qwen3MoeModel,
    Qwen3MoeSparseMoeBlock,
)
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextForCausalLM,
    Qwen3NextMLP,
    Qwen3NextModel,
    Qwen3NextRMSNorm,
    Qwen3NextSparseMoeBlock,
    _should_use_sequence_parallel,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    init_vllm_registered_model,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from . import memory
from .model import PB27WindowedAttention, pb27_hybrid_config

logger = init_logger(__name__)

GDN_STACKED_MAPPER = WeightsMapper(
    orig_to_new_stacked={
        ".gdn.in_proj_qkv": (".gdn.in_proj_qkvz", (0, 1, 2)),
        ".gdn.in_proj_z": (".gdn.in_proj_qkvz", 3),
        ".gdn.in_proj_b": (".gdn.in_proj_ba", 0),
        ".gdn.in_proj_a": (".gdn.in_proj_ba", 1),
    }
)


def _ensure_linear_gdn_config(config: Any) -> Any:
    """Ensure non-Qwen3.5/Qwen3Next HF configs have linear_* GDN attributes set."""
    for attr, default in (
        ("linear_num_key_heads", 16),
        ("linear_key_head_dim", 128),
        ("linear_num_value_heads", 16),
        ("linear_value_head_dim", 128),
        ("linear_conv_kernel_dim", 4),
        ("hidden_act", "silu"),
    ):
        if getattr(config, attr, None) is None:
            setattr(config, attr, default)
    return config


class _PB27MambaStateMixin(HasInnerState, IsHybrid):
    """Shared Mamba/GDN state shape, dtype, and copy calculators."""

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        hf_config = _ensure_linear_gdn_config(vllm_config.model_config.hf_text_config)
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()


# ============================================================================
# 1. Qwen3-Next (e.g. Qwen3-Coder-Next 80B-A3B)
# ============================================================================


class PB27Qwen3NextDecoderLayer(Qwen3NextDecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
        hybrid_layers: frozenset[int] = frozenset(),
        attention_window: int | None = None,
    ) -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)
        self.is_pb_hybrid = self.layer_idx in hybrid_layers

        mlp_only_layers = getattr(config, "mlp_only_layers", []) or []
        is_moe_layer = (self.layer_idx not in mlp_only_layers) and (
            config.num_experts > 0
            and (self.layer_idx + 1) % config.decoder_sparse_step == 0
        )
        self.use_attn_reduce_scatter_for_moe = _should_use_sequence_parallel(
            vllm_config
        )

        if self.is_pb_hybrid:
            self.self_attn = PB27WindowedAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                reduce_results=not self.use_attn_reduce_scatter_for_moe,
                prefix=f"{prefix}.self_attn",
                sliding_window=attention_window,
            )
            self.gdn = QwenGatedDeltaNetAttention(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.gdn",
                gqa_interleaved_layout=False,
                reduce_results=not self.use_attn_reduce_scatter_for_moe,
            )
        elif self.layer_type == "linear_attention":
            self.linear_attn = QwenGatedDeltaNetAttention(
                config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=True,
                reduce_results=not self.use_attn_reduce_scatter_for_moe,
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                reduce_results=not self.use_attn_reduce_scatter_for_moe,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        if is_moe_layer:
            self.mlp = Qwen3NextSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(
                torch.zeros(1, 1, config.hidden_size)
            )
            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(1, 1, config.hidden_size)
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        **kwargs: object,
    ):
        full_num_tokens = positions.shape[-1]
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if self.use_attn_reduce_scatter_for_moe:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            hidden_states = hidden_states[:full_num_tokens]

        if self.is_pb_hybrid:
            attn_out = self.self_attn(hidden_states=hidden_states, positions=positions)
            gdn_out = self.gdn(hidden_states=hidden_states)
            hidden_states = attn_out + gdn_out
        elif self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states=hidden_states)
        elif self.layer_type == "full_attention":
            hidden_states = self.self_attn(
                hidden_states=hidden_states, positions=positions
            )
        else:
            raise ValueError("Invalid layer_type")

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype) + 1
                )

        if self.use_attn_reduce_scatter_for_moe:
            tp_world_size = get_tensor_model_parallel_world_size()
            sp_pad = (-hidden_states.shape[0]) % tp_world_size
            hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, sp_pad))
            hidden_states = tensor_model_parallel_reduce_scatter(hidden_states, 0)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if self.use_attn_reduce_scatter_for_moe:
            hidden_states = self.mlp(hidden_states, already_sequence_parallel=True)
        else:
            hidden_states = self.mlp(hidden_states)

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype) + 1
                )

        return hidden_states, residual


@support_torch_compile
class PB27Qwen3NextModel(Qwen3NextModel):
    hf_to_vllm_mapper = Qwen3NextModel.hf_to_vllm_mapper | GDN_STACKED_MAPPER

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        max_model_len = int(getattr(vllm_config.model_config, "max_model_len", 0) or 0)
        if max_model_len > int(getattr(config, "max_position_embeddings", 0) or 0):
            config.max_position_embeddings = max_model_len
        parallel_config = vllm_config.parallel_config
        self.num_redundant_experts = parallel_config.eplb_config.num_redundant_experts
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, config.hidden_size)

        hybrid_layers, attention_window = pb27_hybrid_config(vllm_config)
        logger.info(
            "PB27 (Qwen3Next): %d hybrid layers %s, window %s",
            len(hybrid_layers),
            sorted(hybrid_layers),
            attention_window,
        )

        def get_layer(prefix: str):
            return PB27Qwen3NextDecoderLayer(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
                hybrid_layers=hybrid_layers,
                attention_window=attention_window,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.is_fused_shared_expert_enabled = is_model_fused_shared_expert_compatible(
            self.layers, Qwen3NextSparseMoeBlock, "mlp"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        from vllm.distributed import get_pp_group

        if get_pp_group().is_last_rank:
            self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.aux_hidden_state_layers: tuple[int, ...] = ()


class PB27Qwen3NextForCausalLM(Qwen3NextForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        if vllm_config.cache_config.mamba_cache_mode == "all":
            raise NotImplementedError("PB27Qwen3Next does not support mamba_cache_mode='all'")
        self.quant_config = vllm_config.quant_config
        nn.Module.__init__(self)
        self.config = config
        self.scheduler_config = vllm_config.scheduler_config
        self.model = PB27Qwen3NextModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
        self.set_moe_parameters()


# ============================================================================
# 2. Qwen3-MoE (e.g. Qwen3-Coder-30B-A3B-Instruct)
# ============================================================================


class PB27Qwen3MoeAttention(Qwen3MoeAttention):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict[str, Any],
        max_position_embeddings: int = 8192,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        dual_chunk_attention_config: dict[str, Any] | None = None,
        per_layer_sliding_window: int | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.dual_chunk_attention_config = dual_chunk_attention_config
        self.sliding_window = per_layer_sliding_window

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        attention_kwargs: dict[str, Any] = {}
        if dual_chunk_attention_config:
            attention_kwargs = {
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": dual_chunk_attention_config,
            }
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=per_layer_sliding_window,
            prefix=f"{prefix}.attn",
            **attention_kwargs,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q = self.q_norm(q_by_head).view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k = self.k_norm(k_by_head).view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        if self.sliding_window is not None and memory.MEMORY["enabled"]:
            attn_output = memory.windowed_attention_with_memory(self.attn, q, k, v, positions)
        else:
            attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class PB27Qwen3MoeDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        is_fused_checkpoint_transposed: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        hybrid, window = pb27_hybrid_config(vllm_config)
        layer_idx = extract_layer_index(prefix)
        self.layer_idx = layer_idx
        self.is_hybrid = layer_idx in hybrid

        self.hidden_size = config.hidden_size
        max_pos = max(
            int(getattr(config, "max_position_embeddings", 8192) or 8192),
            int(getattr(vllm_config.model_config, "max_model_len", 0) or 0),
        )
        dual_chunk_attention_config = getattr(config, "dual_chunk_attention_config", None)
        self.self_attn = PB27Qwen3MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_parameters=config.rope_parameters,
            max_position_embeddings=max_pos,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            dual_chunk_attention_config=dual_chunk_attention_config,
            per_layer_sliding_window=window,
        )
        if self.is_hybrid:
            _ensure_linear_gdn_config(config)
            self.gdn = QwenGatedDeltaNetAttention(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.gdn",
                gqa_interleaved_layout=False,
            )

        mlp_only_layers = getattr(config, "mlp_only_layers", []) or []
        if (layer_idx not in mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3MoeSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
                is_fused_checkpoint_transposed=is_fused_checkpoint_transposed,
            )
        else:
            self.mlp = Qwen3MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        attn_out = self.self_attn(positions=positions, hidden_states=hidden_states)
        if self.is_hybrid:
            attn_out = attn_out + self.gdn(hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(attn_out, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class PB27Qwen3MoeModel(Qwen3MoeModel):
    hf_to_vllm_mapper = Qwen3MoeModel.hf_to_vllm_mapper | GDN_STACKED_MAPPER

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        decoder_layer_type: type[nn.Module] = PB27Qwen3MoeDecoderLayer,
    ):
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            decoder_layer_type=decoder_layer_type,
        )
        hybrid, window = pb27_hybrid_config(vllm_config)
        logger.info(
            "PB27 (Qwen3Moe): %d hybrid layers %s, window %s",
            len(hybrid),
            sorted(hybrid),
            window,
        )


class PB27Qwen3MoeForCausalLM(Qwen3MoeForCausalLM, _PB27MambaStateMixin):
    hf_to_vllm_mapper = PB27Qwen3MoeModel.hf_to_vllm_mapper
    packed_modules_mapping = {
        **Qwen3MoeForCausalLM.packed_modules_mapping,
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

        if vllm_config.cache_config.mamba_cache_mode == "all":
            raise NotImplementedError("PB27Qwen3Moe does not support mamba_cache_mode='all'")
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        if getattr(config, "mlp_only_layers", []):
            self.packed_modules_mapping["gate_up_proj"] = ["gate_proj", "up_proj"]
        self.model = PB27Qwen3MoeModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if self.config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        self.moe_layers = []
        example_layer = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
                example_layer = layer.mlp
                self.moe_layers.append(layer.mlp.experts)
        if example_layer is None:
            raise RuntimeError("No Qwen3MoE layer found in the model.layers.")
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_layer.n_logical_experts
        self.num_physical_experts = example_layer.n_physical_experts
        self.num_local_physical_experts = example_layer.n_local_physical_experts
        self.num_routed_experts = example_layer.n_routed_experts
        self.num_redundant_experts = example_layer.n_redundant_experts


# ============================================================================
# 3. Qwen3 Dense (e.g. Qwen3-32B)
# ============================================================================


class PB27Qwen3DecoderLayer(Qwen3DecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        hybrid, window = pb27_hybrid_config(vllm_config)
        layer_idx = extract_layer_index(prefix)
        self.layer_idx = layer_idx
        self.is_hybrid = layer_idx in hybrid
        max_len = int(getattr(vllm_config.model_config, "max_model_len", 0) or 0)
        if max_len > int(getattr(config, "max_position_embeddings", 0) or 0):
            config.max_position_embeddings = max_len
        super().__init__(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            per_layer_sliding_window=window,
        )
        sa = self.self_attn

        def _forward_with_memory(
            positions: torch.Tensor, hidden_states: torch.Tensor
        ) -> torch.Tensor:
            qkv, _ = sa.qkv_proj(hidden_states)
            q, k, v = qkv.split([sa.q_size, sa.kv_size, sa.kv_size], dim=-1)
            q_by_head = q.view(
                *q.shape[:-1], q.shape[-1] // sa.head_dim, sa.head_dim
            )
            q = sa.q_norm(q_by_head).view(q.shape)
            k_by_head = k.view(
                *k.shape[:-1], k.shape[-1] // sa.head_dim, sa.head_dim
            )
            k = sa.k_norm(k_by_head).view(k.shape)
            q, k = sa.rotary_emb(positions, q, k)
            if memory.MEMORY["enabled"]:
                attn_output = memory.windowed_attention_with_memory(
                    sa.attn, q, k, v, positions
                )
            else:
                attn_output = sa.attn(q, k, v)
            output, _ = sa.o_proj(attn_output)
            return output

        sa.forward = _forward_with_memory
        if self.is_hybrid:
            _ensure_linear_gdn_config(config)
            self.gdn = QwenGatedDeltaNetAttention(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.gdn",
                gqa_interleaved_layout=False,
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        attn_out = self.self_attn(positions=positions, hidden_states=hidden_states)
        if self.is_hybrid:
            attn_out = attn_out + self.gdn(hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(attn_out, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class PB27Qwen3Model(Qwen3Model):
    hf_to_vllm_mapper = Qwen3Model.hf_to_vllm_mapper | GDN_STACKED_MAPPER

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        Qwen2Model.__init__(
            self,
            vllm_config=vllm_config,
            prefix=prefix,
            decoder_layer_type=lambda **kw: PB27Qwen3DecoderLayer(
                vllm_config=vllm_config, **kw
            ),
        )
        hybrid, window = pb27_hybrid_config(vllm_config)
        logger.info(
            "PB27 (Qwen3): %d hybrid layers %s, window %s",
            len(hybrid),
            sorted(hybrid),
            window,
        )


class PB27Qwen3ForCausalLM(Qwen3ForCausalLM, _PB27MambaStateMixin):
    hf_to_vllm_mapper = PB27Qwen3Model.hf_to_vllm_mapper
    packed_modules_mapping = {
        **Qwen3ForCausalLM.packed_modules_mapping,
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        from vllm.distributed import get_pp_group
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

        if vllm_config.cache_config.mamba_cache_mode == "all":
            raise NotImplementedError("PB27Qwen3 does not support mamba_cache_mode='all'")
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.vllm_config = vllm_config
        self.quant_config = quant_config
        self.model = PB27Qwen3Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors


# ============================================================================
# 4. Gemma 4 (e.g. Gemma-4-26B-A4B-it & Gemma-4-31B-it)
# ============================================================================


class PB27Gemma4Attention(Gemma4Attention):
    def __init__(
        self,
        config,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        attn_logits_soft_cap: float | None = None,
        prefix: str = "",
        pb_sliding_window: int | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = hidden_size

        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = 1.0

        layer_idx = extract_layer_index(prefix)
        num_kv_shared_layers = getattr(config, "num_kv_shared_layers", 0)
        first_kv_shared_layer_idx = config.num_hidden_layers - num_kv_shared_layers
        self.is_kv_shared_layer = (
            num_kv_shared_layers > 0 and layer_idx >= first_kv_shared_layer_idx
        )

        if self.is_kv_shared_layer:
            self.q_proj = ColumnParallelLinear(
                hidden_size,
                self.total_num_heads * self.head_dim,
                bias=config.attention_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
        else:
            self.qkv_proj = QKVParallelLinear(
                hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=config.attention_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
            )
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.v_norm = RMSNorm(
                self.head_dim, eps=config.rms_norm_eps, has_weight=False
            )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        layer_type = config.layer_types[layer_idx]
        self.is_sliding = layer_type == "sliding_attention"
        sliding_window = (
            config.sliding_window
            if self.is_sliding
            else pb_sliding_window
        )
        self.pb_windowed = pb_sliding_window is not None and not self.is_kv_shared_layer

        if layer_type in config.rope_parameters:
            rope_parameters = dict(config.rope_parameters[layer_type])
        else:
            rope_parameters = dict(config.rope_parameters.copy())
            if self.is_sliding:
                rope_parameters["rope_theta"] = getattr(
                    config, "rope_local_base_freq", 10000.0
                )

        kv_sharing_target_layer_name = None
        if self.is_kv_shared_layer:
            prev_layers = config.layer_types[:first_kv_shared_layer_idx]
            current_layer_type = config.layer_types[layer_idx]
            kv_shared_layer_index = (
                len(prev_layers) - 1 - prev_layers[::-1].index(current_layer_type)
            )
            if kv_shared_layer_index >= 0:
                param_name_before_layers = prefix.split(".layers.")[0]
                kv_sharing_target_layer_name = (
                    f"{param_name_before_layers}.layers."
                    f"{kv_shared_layer_index}.self_attn.attn"
                )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
            is_neox_style=True,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            logits_soft_cap=attn_logits_soft_cap,
            per_layer_sliding_window=sliding_window,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            mm_prefix_clamp_sliding_window=self.is_sliding,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if self.is_kv_shared_layer:
            q, _ = self.q_proj(hidden_states)
            q = q.unflatten(-1, (self.num_heads, self.head_dim))
            q = self.q_norm(q)
            q = q.flatten(-2, -1)
            q, _ = self.rotary_emb(positions, q, None)
            attn_output = self.attn(q, None, None)
        else:
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q = q.unflatten(-1, (self.num_heads, self.head_dim))
            q = self.q_norm(q)
            q = q.flatten(-2, -1)
            k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
            k = self.k_norm(k)
            k = k.flatten(-2, -1)
            q, k = self.rotary_emb(positions, q, k)
            v = v.unflatten(-1, (self.num_kv_heads, self.head_dim))
            v = self.v_norm(v)
            v = v.flatten(-2, -1)
            if self.pb_windowed and memory.MEMORY["enabled"]:
                attn_output = memory.windowed_attention_with_memory(
                    self.attn, q, k, v, positions
                )
            else:
                attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class PB27Gemma4DecoderLayer(Gemma4DecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory
        from vllm.model_executor.layers.linear import ReplicatedLinear
        from vllm.model_executor.models.gemma4 import Gemma4MLP, Gemma4Router, getattr_iter

        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.hidden_size_per_layer_input = getattr(
            config, "hidden_size_per_layer_input", 0
        )

        layer_idx = extract_layer_index(prefix)
        self.layer_idx = layer_idx
        hybrid, window = pb27_hybrid_config(vllm_config)
        first_kv_shared_layer_idx = config.num_hidden_layers - getattr(
            config, "num_kv_shared_layers", 0
        )
        is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        self.is_hybrid = (layer_idx in hybrid) and not is_kv_shared_layer

        layer_config = gemma4_layer_config(config, layer_idx)
        head_dim = layer_config.head_dim
        num_kv_heads = layer_config.num_key_value_heads
        max_pos = max(
            int(getattr(config, "max_position_embeddings", 131072) or 131072),
            int(getattr(vllm_config.model_config, "max_model_len", 0) or 0),
        )

        self.self_attn = PB27Gemma4Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_position_embeddings=max_pos,
            cache_config=cache_config,
            quant_config=quant_config,
            attn_logits_soft_cap=getattr(config, "attn_logit_softcapping", None),
            prefix=f"{prefix}.self_attn",
            pb_sliding_window=window if self.is_hybrid else None,
        )
        if self.is_hybrid:
            _ensure_linear_gdn_config(config)
            self.gdn = QwenGatedDeltaNetAttention(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.gdn",
                gqa_interleaved_layout=False,
            )

        use_double_wide_mlp = (
            getattr(config, "use_double_wide_mlp", False) and is_kv_shared_layer
        )
        layer_intermediate_size = config.intermediate_size * (
            2 if use_double_wide_mlp else 1
        )
        self.mlp = Gemma4MLP(
            hidden_size=self.hidden_size,
            intermediate_size=layer_intermediate_size,
            hidden_activation=config.hidden_activation,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.enable_moe_block = getattr(config, "enable_moe_block", False) or getattr(
            config, "use_second_mlp_block", False
        )
        if self.enable_moe_block:
            self.router = Gemma4Router(
                config,
                quant_config=quant_config,
                prefix=f"{prefix}.router",
            )
            names = ("moe_intermediate_size", "expert_intermediate_size")
            intermediate_size = getattr_iter(config, names)
            if intermediate_size is None:
                raise ValueError("Gemma4 MoE requires an expert intermediate size")
            self.experts = FusedMoEFactory(
                num_experts=config.num_experts,
                top_k=config.top_k_experts,
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                renormalize=True,
                quant_config=quant_config,
                prefix=f"{prefix}.experts",
                custom_routing_function=self.router.routing_function,
                activation="gelu_tanh",
            )
            self.post_feedforward_layernorm_1 = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_feedforward_layernorm_2 = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.pre_feedforward_layernorm_2 = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.router = None
            self.experts = None
            self.post_feedforward_layernorm_1 = None
            self.post_feedforward_layernorm_2 = None
            self.pre_feedforward_layernorm_2 = None

        if (
            self.hidden_size_per_layer_input is not None
            and self.hidden_size_per_layer_input > 0
        ):
            self.per_layer_input_gate = ReplicatedLinear(
                self.hidden_size,
                self.hidden_size_per_layer_input,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.per_layer_input_gate",
                return_bias=False,
            )
            self.per_layer_projection = ReplicatedLinear(
                self.hidden_size_per_layer_input,
                self.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.per_layer_projection",
                return_bias=False,
            )
            self.post_per_layer_input_norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.per_layer_input_gate = None
            self.per_layer_projection = None
            self.post_per_layer_input_norm = None

        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        per_layer_input: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        normed_input = self.input_layernorm(residual)
        attn_out = self.self_attn(
            positions=positions,
            hidden_states=normed_input,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(attn_out)
        if self.is_hybrid:
            hidden_states = hidden_states + self.gdn(hidden_states=normed_input)
        hidden_states = hidden_states + residual
        residual = hidden_states

        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if self.enable_moe_block:
            hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)
            hidden_states_2 = self.pre_feedforward_layernorm_2(residual)
            router_logits = self.router(residual)
            hidden_states_2 = self.experts(hidden_states_2, router_logits)
            hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)
            hidden_states = hidden_states_1 + hidden_states_2

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        if per_layer_input is not None and self.per_layer_input_gate is not None:
            gate = self.per_layer_input_gate(hidden_states)
            gate = torch.nn.functional.gelu(gate, approximate="tanh")
            gated_per_layer = gate * per_layer_input
            per_layer_contribution = self.per_layer_projection(gated_per_layer)
            per_layer_contribution = self.post_per_layer_input_norm(
                per_layer_contribution
            )
            hidden_states = hidden_states + per_layer_contribution

        hidden_states = hidden_states * self.layer_scalar
        return hidden_states, None


@support_torch_compile(
    enable_if=lambda vllm_config: not vllm_config.cache_config.kv_sharing_fast_prefill
)
class PB27Gemma4Model(Gemma4Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = _get_text_config(vllm_config.model_config.hf_config)
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.hf_to_vllm_mapper = _gemma4_layer_weights_mapper(config) | GDN_STACKED_MAPPER

        self.hidden_size_per_layer_input = getattr(
            config, "hidden_size_per_layer_input", 0
        )
        self.vocab_size_per_layer_input = getattr(
            config, "vocab_size_per_layer_input", config.vocab_size
        )

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )

        if (
            self.hidden_size_per_layer_input is not None
            and self.hidden_size_per_layer_input > 0
        ):
            total_ple_dim = self.hidden_size_per_layer_input * config.num_hidden_layers
            self.embed_tokens_per_layer = VocabParallelEmbedding(
                self.vocab_size_per_layer_input,
                total_ple_dim,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens_per_layer",
            )
            self.register_buffer(
                "embed_scale_per_layer",
                torch.tensor(self.hidden_size_per_layer_input**0.5),
                persistent=False,
            )
            self.per_layer_model_projection = ColumnParallelLinear(
                config.hidden_size,
                total_ple_dim,
                bias=False,
                gather_output=True,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.per_layer_model_projection",
            )
            self.per_layer_projection_norm = RMSNorm(
                self.hidden_size_per_layer_input,
                eps=config.rms_norm_eps,
            )
            self.register_buffer(
                "per_layer_input_scale",
                torch.rsqrt(torch.tensor(2.0)),
                persistent=False,
            )
            self.register_buffer(
                "per_layer_projection_scale",
                torch.tensor(config.hidden_size**-0.5),
                persistent=False,
            )
        else:
            self.embed_tokens_per_layer = None
            self.embed_scale_per_layer = None
            self.per_layer_model_projection = None
            self.per_layer_projection_norm = None
            self.per_layer_input_scale = None
            self.per_layer_projection_scale = None

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: PB27Gemma4DecoderLayer(
                vllm_config=vllm_config,
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.register_buffer(
            "normalizer",
            torch.tensor(
                config.hidden_size**0.5,
                dtype=vllm_config.model_config.dtype,
            ),
            persistent=False,
        )

        first_kv_shared_layer_idx = config.num_hidden_layers - getattr(
            config, "num_kv_shared_layers", 0
        )
        from vllm.compilation.backends import set_model_tag

        with set_model_tag("self_decoder"):
            self.self_decoder = Gemma4SelfDecoderLayers(
                vllm_config=vllm_config,
                prefix=f"{prefix}.self_decoder",
                decoder_layers=self.layers[:first_kv_shared_layer_idx],
                layer_idx_start=0,
                embed_tokens=self.embed_tokens,
                normalizer=self.normalizer,
                embed_tokens_per_layer=getattr(self, "embed_tokens_per_layer", None),
                embed_scale_per_layer=getattr(self, "embed_scale_per_layer", None),
                per_layer_model_projection=getattr(
                    self, "per_layer_model_projection", None
                ),
                per_layer_projection_norm=getattr(
                    self, "per_layer_projection_norm", None
                ),
                per_layer_input_scale=getattr(self, "per_layer_input_scale", None),
                per_layer_projection_scale=getattr(
                    self, "per_layer_projection_scale", None
                ),
            )
        with set_model_tag("cross_decoder"):
            self.cross_decoder = Gemma4CrossDecoderLayers(
                vllm_config=vllm_config,
                prefix=f"{prefix}.cross_decoder",
                decoder_layers=self.layers[first_kv_shared_layer_idx:],
                layer_idx_start=first_kv_shared_layer_idx,
            )

        self.fast_prefill_enabled = cache_config.kv_sharing_fast_prefill
        if self.fast_prefill_enabled:
            max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            device = next(self.parameters()).device
            self.positions = torch.zeros(
                max_num_tokens, dtype=torch.int64, device=device
            )
            self.hidden_states = torch.zeros(
                (max_num_tokens, config.hidden_size),
                dtype=vllm_config.model_config.dtype,
                device=device,
            )
            if (
                self.hidden_size_per_layer_input
                and self.hidden_size_per_layer_input > 0
            ):
                self.per_layer_inputs = torch.zeros(
                    (
                        max_num_tokens,
                        config.num_hidden_layers,
                        self.hidden_size_per_layer_input,
                    ),
                    dtype=vllm_config.model_config.dtype,
                    device=device,
                )
            else:
                self.per_layer_inputs = None

        ple_dim = self.hidden_size_per_layer_input
        num_layers = config.num_hidden_layers
        hidden_size = config.hidden_size
        from vllm.sequence import IntermediateTensors

        def _make_empty_intermediate_tensors(
            batch_size: int,
            dtype: torch.dtype,
            device: torch.device,
        ) -> IntermediateTensors:
            tensors: dict[str, torch.Tensor] = {
                "hidden_states": torch.zeros(
                    (batch_size, hidden_size),
                    dtype=dtype,
                    device=device,
                ),
            }
            if ple_dim and ple_dim > 0:
                tensors["per_layer_inputs"] = torch.zeros(
                    (batch_size, num_layers, ple_dim),
                    dtype=dtype,
                    device=device,
                )
            return IntermediateTensors(tensors)

        self.make_empty_intermediate_tensors = _make_empty_intermediate_tensors
        hybrid, window = pb27_hybrid_config(vllm_config)
        logger.info(
            "PB27 (Gemma4): %d hybrid layers %s, window %s",
            len(hybrid),
            sorted(hybrid),
            window,
        )


class PB27Gemma4ForCausalLM(Gemma4ForCausalLM, _PB27MambaStateMixin):
    packed_modules_mapping = {
        **Gemma4ForCausalLM.packed_modules_mapping,
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

        if vllm_config.cache_config.mamba_cache_mode == "all":
            raise NotImplementedError("PB27Gemma4 does not support mamba_cache_mode='all'")
        config = _get_text_config(vllm_config.model_config.hf_config)
        quant_config = vllm_config.quant_config
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config
        self.model = PB27Gemma4Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(
            config.vocab_size,
            soft_cap=getattr(config, "final_logit_softcapping", None),
        )
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        self.moe_layers: list[nn.Module] = [
            layer.experts
            for layer in self.model.layers
            if not isinstance(layer, PPMissingLayer) and layer.experts is not None
        ]
        self.num_moe_layers = len(self.moe_layers)
        num_experts = config.num_experts if self.num_moe_layers else 0
        self.num_logical_experts = num_experts
        self.num_physical_experts = num_experts
        self.num_local_physical_experts = num_experts
        self.num_routed_experts = num_experts
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_redundant_experts = 0


@MULTIMODAL_REGISTRY.register_processor(
    Gemma4MultiModalProcessor,
    info=Gemma4ProcessingInfo,
    dummy_inputs=Gemma4DummyInputsBuilder,
)
class PB27Gemma4ForConditionalGeneration(
    Gemma4ForConditionalGeneration, _PB27MambaStateMixin
):
    packed_modules_mapping = {
        **Gemma4ForConditionalGeneration.packed_modules_mapping,
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        from transformers import AutoModel
        from vllm.model_executor.models.gemma4_mm import (
            Gemma4MultimodalEmbedder,
            recursive_replace_linear,
        )

        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.quant_config = quant_config
        self.multimodal_config = multimodal_config
        self.model_dtype = vllm_config.model_config.dtype
        self.vllm_config = vllm_config
        lora_config = vllm_config.lora_config
        self._enable_mm_lora = bool(
            lora_config is not None and lora_config.enable_tower_connector_lora
        )

        tower_quant: QuantizationConfig | None
        if quant_config and quant_config.get_name() in [
            "bitsandbytes",
            "torchao",
            "compressed-tensors",
        ]:
            tower_quant = quant_config
        else:
            vision_cfg = config.vision_config
            quantizable = (
                vision_cfg.hidden_size % 64 == 0
                and vision_cfg.intermediate_size % 64 == 0
            )
            tower_quant = quant_config if quantizable else None

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.vision_tower = AutoModel.from_config(config=config.vision_config)
            self.embed_vision = Gemma4MultimodalEmbedder(
                config.vision_config,
                config.text_config,
                quant_config=tower_quant,
                prefix=maybe_prefix(prefix, "embed_vision"),
            )
            recursive_replace_linear(
                self.vision_tower,
                tower_quant,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )

        self.embed_audio: Gemma4MultimodalEmbedder | None
        if config.audio_config is not None:
            with self._mark_tower_model(vllm_config, "audio"):
                self.audio_tower = AutoModel.from_config(config=config.audio_config)
                self.audio_tower.post_init()
                self.embed_audio = Gemma4MultimodalEmbedder(
                    config.audio_config,
                    config.text_config,
                    quant_config=tower_quant,
                    prefix=maybe_prefix(prefix, "embed_audio"),
                )
                recursive_replace_linear(
                    self.audio_tower,
                    tower_quant,
                    prefix=maybe_prefix(prefix, "audio_tower"),
                )
        else:
            self.audio_tower = None
            self.embed_audio = None

        with self._mark_language_model(vllm_config):
            self.language_model: PB27Gemma4ForCausalLM = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["PB27Gemma4ForCausalLM"],
            )
            ple_dim = config.text_config.hidden_size_per_layer_input
            if ple_dim is not None and ple_dim > 0:
                embed = self.language_model.model.embed_tokens
                self.per_layer_embeddings = torch.zeros(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    config.text_config.num_hidden_layers,
                    ple_dim,
                    device=next(embed.parameters()).device,
                    dtype=vllm_config.model_config.dtype,
                )
            else:
                self.per_layer_embeddings = None

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        self._full_attn_layer_idxs: frozenset[int] = frozenset()
        text_config = config.text_config
        if getattr(text_config, "use_bidirectional_attention", None) == "vision":
            layer_types = getattr(text_config, "layer_types", None)
            if layer_types:
                self._full_attn_layer_idxs = frozenset(
                    i for i, lt in enumerate(layer_types) if lt != "sliding_attention"
                )

        self.moe_layers = self.language_model.moe_layers
        self.num_moe_layers = self.language_model.num_moe_layers
        self.num_logical_experts = self.language_model.num_logical_experts
        self.num_physical_experts = self.language_model.num_physical_experts
        self.num_local_physical_experts = self.language_model.num_local_physical_experts
        self.num_routed_experts = self.language_model.num_routed_experts
        self.num_expert_groups = self.language_model.num_expert_groups
        self.num_shared_experts = self.language_model.num_shared_experts
        self.num_redundant_experts = self.language_model.num_redundant_experts

        gen_cfg = vllm_config.model_config.try_get_generation_config()
        self._suppress_token_ids = gen_cfg.get("suppress_tokens") if gen_cfg else None
