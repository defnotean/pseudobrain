"""PB-27 for vLLM v0.31.0: Qwen3.5/3.8 where the layers listed in
`pb_hybrid_layers` run the stock attention weights under a sliding window of
`pb_attention_window` tokens (no sink positions) plus a parallel Gated DeltaNet
branch (`gdn`, stock linear_attn naming) on the same pre-norm input; the two
outputs are summed. Everything else is the stock Qwen3.5 implementation.

Every class below copies the corresponding vLLM `__init__` body instead of
calling the parent `__init__` and swapping modules afterwards: the Attention and
GDN layers register themselves in `compilation_config.static_forward_context`
under their prefix at construction time and a second construction with the same
prefix raises "Duplicate layer name" (layers/attention/attention.py:443-445,
layers/mamba/gdn/qwen_gdn_linear_attn.py:535-538).

Reference lines are into /usr/local/lib/python3.12/dist-packages/vllm/ of the
vllm/vllm-openai:v0.31.0-aarch64 image.
"""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe.utils import (
    is_model_fused_shared_expert_compatible,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm as Qwen3_5RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding, get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5ForCausalLMBase,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
    Qwen3_5ProcessingInfo,
)
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextModel,
    Qwen3NextSparseMoeBlock,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.tokenizers.registry import cached_tokenizer_from_config
from vllm.v1.attention.backend import AttentionType

from . import memory

logger = init_logger(__name__)


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


def _pb_field(vllm_config: VllmConfig, name: str, default=None):
    """`pb_*` fields live in text_config and on the top level; prefer the text
    config (what the decoder layers read everything else from)."""
    model_config = vllm_config.model_config
    for cfg in (model_config.hf_text_config, model_config.hf_config):
        if cfg is not None and hasattr(cfg, name):
            value = getattr(cfg, name)
            if value is not None:
                return value
    return default


def pb27_hybrid_config(vllm_config: VllmConfig) -> tuple[frozenset[int], int | None]:
    hybrid = _pb_field(vllm_config, "pb_hybrid_layers")
    if hybrid is None:
        raise ValueError(
            "PB27 config needs `pb_hybrid_layers` (list of decoder-layer indices) "
            "in text_config or at the top level of config.json"
        )
    window = _pb_field(vllm_config, "pb_attention_window")
    if window is not None:
        window = int(window)
        if window <= 0:
            raise ValueError(f"pb_attention_window must be positive, got {window}")
    sinks = int(_pb_field(vllm_config, "pb_attention_sinks", 0) or 0)
    if sinks > 0:
        logger.warning_once(
            "PB27: config lists pb_attention_sinks=%d but vLLM serves the hybrid "
            "layers with a plain sliding window (no kept sink positions); expect "
            "the `--no-sinks` PyTorch numbers (docs/PB27_VLLM_PLUGIN.md section 2).",
            sinks,
        )
    return frozenset(int(i) for i in hybrid), window


# --------------------------------------------------------------------------
# windowed attention
# --------------------------------------------------------------------------


class PB27WindowedAttention(Qwen3NextAttention):
    """Qwen3NextAttention (qwen3_next.py:273-385) with a per-layer sliding window.

    Only `Attention(..., per_layer_sliding_window=sliding_window)` differs; the
    Attention layer then advertises a SlidingWindowSpec (attention.py:606-668)
    and FA masks `q_idx - k_idx < W`, PB's semantics. `_project_qkv_gate` and
    `forward(positions, hidden_states)` (qwen3_next.py:449-460) are inherited.
    """

    def __init__(
        self,
        config,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
        sliding_window: int | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        self.attn_output_gate = getattr(config, "attn_output_gate", True)
        self.sliding_window = sliding_window

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=getattr(config, "qkv_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
            dual_chunk_attention_config=self.dual_chunk_attention_config,
        )

        attn_type = (
            AttentionType.DECODER
            if getattr(config, "is_causal", True)
            else AttentionType.ENCODER_ONLY
        )
        # Attention.__init__ (attention.py:241-260): per_layer_sliding_window
        # overrides cache_config.sliding_window (attention.py:265-273).
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": self.dual_chunk_attention_config,
            }
            if self.dual_chunk_attention_config
            else {},
        )

        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        mm_config = model_config.multimodal_config if model_config else None
        text_only = mm_config is None or mm_config.language_model_only
        mrope_section = getattr(self.rotary_emb, "mrope_section", None)
        supports_mrope = bool(
            type(self.rotary_emb) is MRotaryEmbedding
            and mrope_section
            and len(mrope_section) == 3
            and sum(mrope_section) == self.rotary_emb.rotary_dim // 2
            and getattr(self.rotary_emb, "mrope_interleaved", False)
        )
        supports_dtype = getattr(self.rotary_emb, "dtype", None) in (
            torch.float16,
            torch.bfloat16,
        )
        self.use_fused_qk_norm_rope_gate = (
            self.attn_output_gate
            and getattr(self.rotary_emb, "is_neox_style", False)
            and (current_platform.is_cuda() or current_platform.is_xpu())
            and supports_dtype
            and (text_only or supports_mrope)
        )


# --------------------------------------------------------------------------
# decoder layer
# --------------------------------------------------------------------------


    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        # Qwen3NextAttention.forward (qwen3_next.py:449-460) with the exact memory tier spliced in around the
        # attention call when PB27_MEMORY is set (pb27_vllm/memory.py); otherwise identical to the parent.
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        if memory.MEMORY["enabled"]:
            attn_output = memory.windowed_attention_with_memory(self.attn, q, k, v, positions)
        else:
            attn_output = self.attn(q, k, v)
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)
        output, _ = self.o_proj(attn_output)
        return output


class PB27DecoderLayer(Qwen3_5DecoderLayer):
    """Qwen3_5DecoderLayer.__init__ (qwen3_5.py:122-206) plus, for indices in
    `pb_hybrid_layers`, a windowed self_attn and a parallel `gdn` branch.

    `layer_type` keeps the stock meaning ("full_attention" for the hybrid
    indices, since the checkpoint's layer_types is unchanged); `is_pb_hybrid`
    selects the summed two-mixer forward.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
        hybrid_layers: frozenset[int] = frozenset(),
        attention_window: int | None = None,
    ) -> None:
        super(Qwen3NextDecoderLayer, self).__init__()

        config = vllm_config.model_config.hf_text_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)
        self.is_pb_hybrid = self.layer_idx in hybrid_layers
        is_moe_layer = config.model_type == "qwen3_5_moe_text"
        self.use_attn_reduce_scatter_for_moe = (
            parallel_config.use_sequence_parallel_moe
            and parallel_config.pipeline_parallel_size == 1
            and is_moe_layer
        )

        if self.is_pb_hybrid:
            if self.layer_type != "full_attention":
                raise ValueError(
                    f"PB27 hybrid layer {self.layer_idx} must be a full_attention "
                    f"layer in layer_types, got {self.layer_type!r}"
                )
            # Stock attention weights, windowed. Prefix `<layer>.self_attn` so
            # the HF names load unchanged.
            self.self_attn = PB27WindowedAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                reduce_results=not self.use_attn_reduce_scatter_for_moe,
                sliding_window=attention_window,
            )
            # Parallel Gated DeltaNet branch; same constructor call as the stock
            # linear_attention layer (qwen3_5.py:147-153) under prefix `<layer>.gdn`.
            # The module registers itself as a MambaBase cache-bearing layer under
            # that prefix (qwen_gdn_linear_attn.py:535-538).
            self.gdn = QwenGatedDeltaNetAttention(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.gdn",
                gqa_interleaved_layout=False,
                reduce_results=not self.use_attn_reduce_scatter_for_moe,
            )
        elif self.layer_type == "linear_attention":
            self.linear_attn = QwenGatedDeltaNetAttention(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,
                reduce_results=not self.use_attn_reduce_scatter_for_moe,
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                reduce_results=not self.use_attn_reduce_scatter_for_moe,
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        if config.model_type == "qwen3_5_moe_text":
            self.mlp = Qwen3NextSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        elif config.model_type == "qwen3_5_text":
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            raise ValueError(f"Invalid model_type {config.model_type}")

        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(
                torch.zeros(1, 1, config.hidden_size),
            )
            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(1, 1, config.hidden_size),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        **kwargs: object,
    ):
        # Qwen3NextDecoderLayer.forward (qwen3_next.py:549-620) with the hybrid
        # branch added. The model calls layer(positions=, hidden_states=,
        # residual=) (qwen3_next.py:717-721).
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
            # Both mixers read the same post-norm input; outputs summed.
            # Qwen3NextAttention.forward(positions, hidden_states) (qwen3_next.py:449)
            # QwenGatedDeltaNetAttention.forward(hidden_states) (qwen_gdn_linear_attn.py:839)
            attn_out = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
            )
            gdn_out = self.gdn(hidden_states=hidden_states)
            hidden_states = attn_out + gdn_out
        elif self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states=hidden_states)
        elif self.layer_type == "full_attention":
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
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
            hidden_states = self.mlp(
                hidden_states,
                already_sequence_parallel=True,
            )
        else:
            hidden_states = self.mlp(hidden_states)

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                assert len(hidden_states.shape) == len(self.ffn_layer_scale.shape)
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype) + 1
                )

        return hidden_states, residual


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class PB27Model(Qwen3_5Model):
    """Qwen3_5Model.__init__ (qwen3_5.py:231-276) with the PB27 layer factory.

    `hf_to_vllm_mapper` is inherited: its stacked rules match by substring
    (utils.py:126-129), so `layers.i.gdn.in_proj_qkv` / `in_proj_z` fuse into
    `gdn.in_proj_qkvz` and `in_proj_b` / `in_proj_a` into `gdn.in_proj_ba`
    exactly like `linear_attn.*` (qwen3_5.py:222-229); no gdn-specific rule.
    `forward` and `load_weights` are inherited too.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Qwen3NextModel, self).__init__()

        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config

        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.config = config
        self.quant_config = vllm_config.quant_config

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        hybrid_layers, attention_window = pb27_hybrid_config(vllm_config)
        self.pb_hybrid_layers = hybrid_layers
        self.pb_attention_window = attention_window
        logger.info(
            "PB27: %d hybrid layers %s, attention window %s",
            len(hybrid_layers),
            sorted(hybrid_layers),
            attention_window,
        )

        def get_layer(prefix: str):
            return PB27DecoderLayer(
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
            self.layers,
            Qwen3NextSparseMoeBlock,
            "mlp",
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        if get_pp_group().is_last_rank:
            self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.aux_hidden_state_layers: tuple[int, ...] = ()


# --------------------------------------------------------------------------
# causal LM / conditional generation wrappers
# --------------------------------------------------------------------------


class PB27ForCausalLM(Qwen3_5ForCausalLM):
    """Qwen3_5ForCausalLMBase.__init__ (qwen3_5.py:327-363) with PB27Model.

    Inherits packed_modules_mapping, the `model.language_model.` -> `model.`
    prefix mapper, the mamba state shape/dtype/copy classmethods (they describe
    one GDN layer's state; every GDN module, `linear_attn` or `gdn`, has the
    same shape) and compute_logits/load_weights.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        scheduler_config = vllm_config.scheduler_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "PB27 (like Qwen3.5) does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )
        self.quant_config = vllm_config.quant_config

        super(Qwen3_5ForCausalLMBase, self).__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = PB27Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class PB27ForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """Qwen3_5ForConditionalGeneration.__init__ (qwen3_5.py:488-530) with
    PB27ForCausalLM as the language model. The hf_to_vllm_mapper
    (`model.language_model.` -> `language_model.model.`, `mtp.` dropped),
    packed_modules_mapping, forward, embed_input_ids, load_weights and the
    mamba state classmethods are inherited unchanged.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.get_multimodal_config()

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )
        self.video_pruning_rate = self.multimodal_config.video_pruning_rate
        self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)

        self.use_deepstack = hasattr(config.vision_config, "deepstack_visual_indexes")
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = PB27ForCausalLM(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
