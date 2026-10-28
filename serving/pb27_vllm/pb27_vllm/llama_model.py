"""PB-27 for vLLM v0.31.0 on a Llama teacher (Nanbeige4.3-33B): the layers listed
in `pb_hybrid_layers` run the stock Llama attention under a sliding window of
`pb_attention_window` tokens plus a parallel Gated DeltaNet branch (`gdn`,
Qwen3.5 linear_attn naming, dims from the `linear_*` fields that
export_pb27_slim writes into config.json) on the same pre-norm input; the two
outputs are summed, and exact memory retrieval (`memory.py`) is wired into
each hybrid layer's attention forward. Everything else is the stock Llama
implementation.

Unlike model.py, the parent `__init__` bodies are called: the window reaches
LlamaAttention through a per-layer config copy (layer_types /
sliding_window, llama.py:185-204), so every Attention is constructed once, and
the GDN is added after the parent constructor, which builds none (duplicate
prefixes raise, attention.py:443-445, qwen_gdn_linear_attn.py:535-538).

Reference lines are into /usr/local/lib/python3.12/dist-packages/vllm/ of the
vllm/vllm-openai:v0.31.0-aarch64 image.
"""

import copy

import torch

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.model_executor.models.llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
)
from vllm.model_executor.models.utils import WeightsMapper, extract_layer_index

from . import memory
from .model import pb27_hybrid_config

logger = init_logger(__name__)


class PB27LlamaDecoderLayer(LlamaDecoderLayer):
    """LlamaDecoderLayer (llama.py:251-330); a hybrid layer's mixer is
    `self_attn(h) + gdn(h)` with self_attn windowed + exact memory tier."""

    def __init__(self, vllm_config: VllmConfig, prefix: str = "", config=None, **kwargs) -> None:
        hf_config = config or vllm_config.model_config.hf_config
        hybrid, window = pb27_hybrid_config(vllm_config)
        layer_idx = extract_layer_index(prefix)
        layer_config = copy.copy(hf_config)
        max_len = getattr(vllm_config.model_config, "max_model_len", 0) or 0
        if max_len > getattr(layer_config, "max_position_embeddings", 0):
            layer_config.max_position_embeddings = max_len
        if window is not None:
            layer_config.layer_types = [
                "sliding_attention" if i in hybrid else "full_attention"
                for i in range(hf_config.num_hidden_layers)
            ]
            layer_config.sliding_window = window
        super().__init__(vllm_config, prefix=prefix, config=layer_config, **kwargs)
        self.layer_idx = layer_idx
        self.is_hybrid = layer_idx in hybrid
        if self.is_hybrid:
            sa = self.self_attn

            def _forward_with_memory(positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
                qkv, _ = sa.qkv_proj(hidden_states)
                q, k, v = qkv.split([sa.q_size, sa.kv_size, sa.kv_size], dim=-1)
                q, k = sa.rotary_emb(positions, q, k)
                if memory.MEMORY["enabled"]:
                    attn_output = memory.windowed_attention_with_memory(sa.attn, q, k, v, positions)
                else:
                    attn_output = sa.attn(q, k, v)
                output, _ = sa.o_proj(attn_output)
                return output

            sa.forward = _forward_with_memory
            self.gdn = QwenGatedDeltaNetAttention(
                config=hf_config,
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
        "input_ids": {0: "b"},
        "positions": {0: "b"},
        "intermediate_tensors": {0: "b"},
        "inputs_embeds": {0: "b"},
    },
)
class PB27LlamaModel(LlamaModel):
    """LlamaModel (llama.py:337-461) whose weight mapper also fuses the GDN
    in_proj tensors as Qwen3_5Model does (qwen3_5.py:222-229)."""

    hf_to_vllm_mapper = LlamaModel.hf_to_vllm_mapper | WeightsMapper(
        orig_to_new_stacked={
            ".in_proj_qkv": (".in_proj_qkvz", (0, 1, 2)),
            ".in_proj_z": (".in_proj_qkvz", 3),
            ".in_proj_b": (".in_proj_ba", 0),
            ".in_proj_a": (".in_proj_ba", 1),
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", layer_type=PB27LlamaDecoderLayer):
        super().__init__(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)
        hybrid, window = pb27_hybrid_config(vllm_config)
        logger.info("PB27 (Llama): %d hybrid layers %s, attention window %s", len(hybrid), sorted(hybrid), window)


class PB27LlamaForCausalLM(LlamaForCausalLM, HasInnerState, IsHybrid):
    """LlamaForCausalLM (llama.py:464-571) with the Gated DeltaNet state
    interface of Qwen3_5ForCausalLMBase (qwen3_5.py:389-430)."""

    hf_to_vllm_mapper = PB27LlamaModel.hf_to_vllm_mapper
    packed_modules_mapping = {
        **LlamaForCausalLM.packed_modules_mapping,
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", layer_type=PB27LlamaDecoderLayer):
        if vllm_config.cache_config.mamba_cache_mode == "all":
            raise NotImplementedError("PB27 (Llama) does not support mamba_cache_mode='all'")
        super().__init__(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)

    def _init_model(self, vllm_config: VllmConfig, prefix: str = "", layer_type=PB27LlamaDecoderLayer):
        return PB27LlamaModel(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig) -> tuple[tuple[int, int], tuple[int, int, int]]:
        hf_config = vllm_config.model_config.hf_text_config
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
