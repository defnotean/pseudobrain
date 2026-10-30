"""PB-27 for vLLM v0.31.0 on a Qwen4Exp teacher (Qwen3.8-Flash-Next): the layers
listed in `pb_hybrid_layers` run `PB27WindowedAttention` (`pb_attention_window`
sliding window + Exact Memory retrieval from `memory.py`) in parallel with a
Gated DeltaNet branch (`gdn`, Qwen3.5 `linear_attn` naming) on the same
HyperConnection `block_input`, summing the two outputs before the MLP
HyperConnection combine-and-mix. Everything else (36 native `linear_attention`
GDN layers, Layer 2 PLE n-gram lookup, 4-stream HyperConnections, 256-expert
MoE, Qwen3-VL vision tower, native `Qwen4ExpMTP` and `DFlash2` Eagle3 auxiliary
hidden-state capture) follows the stock `vllm.models.qwen4_exp.nvidia.model`
implementation.
"""

from collections.abc import Iterable
import glob
from itertools import islice
import json
import mmap
import os
import struct
import warnings

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.utils import (
    is_model_fused_shared_expert_compatible,
)
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import EagleModelMixin, SupportsEagle3
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextMLP,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
)
from vllm.model_executor.models.utils import (
    StageMissingLayer,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.models.qwen4_exp.config import (
    ATTENTION_LAYER_TYPES,
    QSA_LAYER_TYPE,
    Qwen4ExpConfig,
)
from vllm.models.qwen4_exp.nvidia.hyperconnection import (
    GatedResidual,
    HyperConnectionConfig,
)
from vllm.models.qwen4_exp.nvidia.low_latency_gemm import (
    enable_qwen4_exp_low_latency_gemm,
)
from vllm.models.qwen4_exp.nvidia.model import (
    Qwen4ExpDecoderLayer,
    Qwen4ExpForCausalLM,
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpModel,
    Qwen4ExpProcessingInfo,
    Qwen4ExpSparseMoeBlock,
)
from vllm.models.qwen4_exp.nvidia.ngram_embedding import Qwen4ExpNGramEmbedding
from vllm.models.qwen4_exp.nvidia.ops.cute_dsl.hc_down_silu import (
    request_hc_down_silu_warmup,
)
from vllm.models.qwen4_exp.nvidia.ple_layer import (
    Qwen4ExpPLEGroupedNorm,
    Qwen4ExpPLELayer,
)
from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAAttention
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.tokenizers.registry import cached_tokenizer_from_config
from vllm.transformers_utils.configs.qwen4_exp import Qwen4ExpTextConfig
from vllm.triton_utils import tl, triton

from .model import (
    PB27WindowedAttention,
    pb27_hybrid_config,
    pb27_spatial_config,
)

logger = init_logger(__name__)


@triton.jit
def _nvfp4_ple_mmap_gather_kernel(
    ptr_table_w,
    ptr_table_s,
    ids_ptr,
    out_ptr,
    scale2_ptr,
    lut_ptr,
    shard_rows,
    total_rows,
):
    row_id = tl.program_id(0)
    idx = tl.load(ids_ptr + row_id)
    valid = (idx >= 0) & (idx < total_rows)
    idx = tl.where(valid, idx, 0)
    shard_id = idx // shard_rows
    row_in_shard = idx % shard_rows
    w_base = tl.load(ptr_table_w + shard_id).to(tl.pointer_type(tl.uint8))
    s_base = tl.load(ptr_table_s + shard_id).to(tl.pointer_type(tl.uint8))
    offs80 = tl.arange(0, 128)
    mask80 = offs80 < 80
    wb = tl.load(w_base + row_in_shard * 80 + offs80, mask=mask80, other=0)
    sb_u8 = tl.load(s_base + row_in_shard * 10 + (offs80 // 8), mask=mask80, other=0)
    sb = sb_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    s2 = tl.load(scale2_ptr).to(tl.float32)
    scale = tl.where(valid & mask80, sb * s2, 0.0)
    lo = wb & 0x0F
    hi = wb >> 4
    v_lo = tl.load(lut_ptr + lo) * scale
    v_hi = tl.load(lut_ptr + hi) * scale
    out_base = out_ptr + row_id * 160
    tl.store(out_base + offs80 * 2, v_lo.to(tl.bfloat16), mask=mask80)
    tl.store(out_base + offs80 * 2 + 1, v_hi.to(tl.bfloat16), mask=mask80)


class PB27Qwen4ExpNVFP4PLEEmbedding(nn.Module):
    """Zero-copy mmap NVFP4 PLE embedding lookup + on-the-fly GPU Triton dequantization."""

    supports_prefetch = False

    def __init__(
        self,
        model_path: str,
        vocab_size: int,
        embedding_dim: int,
        split_ngram_parts: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        self.org_vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.split_ngram_parts = split_ngram_parts
        self.params_dtype = params_dtype
        self.prefix = prefix
        self.shard_rows = (vocab_size + split_ngram_parts - 1) // split_ngram_parts
        self.weight_scale_2 = nn.Parameter(
            torch.ones(1, dtype=torch.float32), requires_grad=False
        )
        self.register_buffer(
            "weight", torch.empty(0, dtype=torch.uint8), persistent=False
        )
        self.register_buffer(
            "_fp4_lut",
            torch.tensor(
                [
                    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
                ],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self._mmaps: list[mmap.mmap] = []
        self._keepalive: list[torch.Tensor] = []
        self._tokens_seen: int = 0
        self._init_mmap_shards(model_path, prefix)

    def _init_mmap_shards(self, model_path: str, prefix: str) -> None:
        hf_prefix_candidates = [prefix]
        if prefix.startswith("model.layers."):
            hf_prefix_candidates.insert(
                0, "model.language_model." + prefix[len("model.") :]
            )
        elif prefix.startswith("language_model.model.layers."):
            hf_prefix_candidates.insert(
                0, "model." + prefix.replace("language_model.model.", "language_model.")
            )
        elif prefix.startswith("model.language_model.model.layers."):
            hf_prefix_candidates.insert(
                0, prefix.replace("model.language_model.model.", "model.language_model.")
            )

        w_ptrs = [0] * self.split_ngram_parts
        s_ptrs = [0] * self.split_ngram_parts
        found_w = 0
        found_s = 0

        st_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        for path in st_files:
            with open(path, "rb") as f:
                raw_len = f.read(8)
                if len(raw_len) < 8:
                    continue
                (hlen,) = struct.unpack("<Q", raw_len)
                header = json.loads(f.read(hlen).decode("utf-8"))

            matched_prefix = None
            for cand in hf_prefix_candidates:
                if any(k.startswith(f"{cand}.shard_") for k in header):
                    matched_prefix = cand
                    break
            if matched_prefix is None:
                continue

            fd = os.open(path, os.O_RDONLY)
            fsize = os.fstat(fd).st_size
            mm = mmap.mmap(fd, fsize, access=mmap.ACCESS_READ)
            os.close(fd)
            if hasattr(mm, "madvise") and hasattr(mmap, "MADV_RANDOM"):
                mm.madvise(mmap.MADV_RANDOM)
            self._mmaps.append(mm)
            base_offset = 8 + hlen

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                base_view = torch.frombuffer(
                    mm, dtype=torch.uint8, count=1, offset=0
                )
            self._keepalive.append(base_view)
            mm_base_ptr = base_view.data_ptr()

            for s in range(self.split_ngram_parts):
                wk = f"{matched_prefix}.shard_{s}.weight"
                sk = f"{matched_prefix}.shard_{s}.weight_scale"
                if wk in header:
                    ws, _ = header[wk]["data_offsets"]
                    w_ptrs[s] = mm_base_ptr + base_offset + ws
                    found_w += 1
                if sk in header:
                    ss, _ = header[sk]["data_offsets"]
                    s_ptrs[s] = mm_base_ptr + base_offset + ss
                    found_s += 1

        if found_w != self.split_ngram_parts or found_s != self.split_ngram_parts:
            raise RuntimeError(
                f"PB27Qwen4ExpNVFP4PLEEmbedding: expected {self.split_ngram_parts} "
                f"PLE shards in {model_path} for {hf_prefix_candidates}, found "
                f"weight={found_w}, weight_scale={found_s}"
            )
        self.register_buffer(
            "ptr_table_w",
            torch.tensor(w_ptrs, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "ptr_table_s",
            torch.tensor(s_ptrs, dtype=torch.int64),
            persistent=False,
        )

    def dequantize(
        self, embeddings: torch.Tensor, output_dtype: torch.dtype
    ) -> torch.Tensor:
        return embeddings.to(output_dtype)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        orig_shape = input_ids.shape
        flat_ids = input_ids.reshape(-1)
        n = flat_ids.numel()
        out = torch.empty(
            (n, self.embedding_dim),
            dtype=torch.bfloat16,
            device=input_ids.device,
        )
        if n > 0:
            _nvfp4_ple_mmap_gather_kernel[(n,)](
                self.ptr_table_w,
                self.ptr_table_s,
                flat_ids,
                out,
                self.weight_scale_2,
                self._fp4_lut,
                self.shard_rows,
                self.org_vocab_size,
                num_warps=4,
            )
            if n >= 1024:
                self._tokens_seen += n
                if self._tokens_seen >= 65536:
                    self._tokens_seen = 0
                    torch.cuda.current_stream().synchronize()
                    for mm in self._mmaps:
                        try:
                            mm.madvise(mmap.MADV_DONTNEED)
                        except OSError:
                            pass
                    if (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) > (2 << 30):
                        torch.cuda.empty_cache()
        return out.view(*orig_shape, self.embedding_dim).to(self.params_dtype)


class PB27Qwen4ExpNGramEmbedding(Qwen4ExpNGramEmbedding):
    """Qwen4ExpNGramEmbedding with support for zero-copy mmap `ple_embedding_dtype='nvfp4'`."""

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        embedding_dim: int,
        ple_dense_layer_id: int,
        max_total_tokens: int,
        *,
        data_parallel_rank: int,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
        params_dtype: torch.dtype | None = None,
    ) -> None:
        if getattr(config, "ple_embedding_dtype", None) != "nvfp4":
            super().__init__(
                config,
                embedding_dim,
                ple_dense_layer_id,
                max_total_tokens,
                data_parallel_rank=data_parallel_rank,
                prefix=prefix,
                quant_config=quant_config,
                params_dtype=params_dtype,
            )
            return

        nn.Module.__init__(self)
        self.embedding_dim = embedding_dim
        self.ngram_size = int(config.ngram_size)
        self.heads_per_ngram = int(config.heads_per_ngram)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.head_dim = embedding_dim // self.ngram_heads
        self.eos_token_id = int(config.eos_token_id)
        self.unigram_vocab_size = int(config.vocab_size)
        self.split_ngram_parts = int(getattr(config, "split_ngram_parts", 512))

        multipliers = self._make_layer_multipliers(
            ngram_size=self.ngram_size,
            unigram_vocab_size=self.unigram_vocab_size,
            seed=int(getattr(config, "seed", 1234)),
            ple_dense_layer_id=ple_dense_layer_id,
        )
        self.register_buffer(
            "layer_multipliers",
            torch.tensor(multipliers, dtype=torch.long),
            persistent=True,
        )
        sizes, offsets, total_vocab_size = self._make_vocab_layout(
            ngram_vocab_size_base=int(config.ngram_vocab_size_base),
            ngram_heads=self.ngram_heads,
            ple_dense_layer_id=ple_dense_layer_id,
        )
        self.register_buffer(
            "ngram_heads_vocab_sizes",
            torch.tensor(sizes, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "ngram_heads_offsets",
            torch.tensor(offsets, dtype=torch.long),
            persistent=True,
        )
        divisor = int(config.make_ngram_vocab_size_divisible_by)
        padded_vocab_size = ((total_vocab_size + divisor - 1) // divisor) * divisor
        embedding_prefix = f"{prefix}.ngram_embedding"
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        model_path = get_current_vllm_config().model_config.model
        self.ngram_embedding = PB27Qwen4ExpNVFP4PLEEmbedding(
            model_path=model_path,
            vocab_size=padded_vocab_size,
            embedding_dim=self.head_dim,
            split_ngram_parts=self.split_ngram_parts,
            params_dtype=params_dtype,
            prefix=embedding_prefix,
        )
        logger.info(
            "Initialized zero-copy mmap NVFP4 PLE embedding %s from %s (%d shards, vocab=%d, dim=%d)",
            embedding_prefix,
            model_path,
            self.split_ngram_parts,
            padded_vocab_size,
            self.head_dim,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if isinstance(self.ngram_embedding, PB27Qwen4ExpNVFP4PLEEmbedding):
            weights = (
                (name, loaded_weight)
                for name, loaded_weight in weights
                if not name.startswith("ngram_embedding.shard_")
            )
        return super().load_weights(weights)


class PB27Qwen4ExpPLELayer(Qwen4ExpPLELayer):
    """Qwen4ExpPLELayer backed by PB27Qwen4ExpNGramEmbedding."""

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        vllm_config: VllmConfig,
        layer_idx: int = 0,
        ple_dense_layer_id: int | None = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.model_config = model_config
        self.cache_config = cache_config
        self.layer_idx = layer_idx
        self.ple_dense_layer_id = (
            int(ple_dense_layer_id)
            if ple_dense_layer_id is not None
            else int(layer_idx)
        )
        self.prefix = prefix
        self.hidden_size = int(config.hidden_size)
        self.hc_count = config.hc_count
        self.hc_hidden_size = self.hidden_size * self.hc_count
        self.conv_kernel_size = int(config.ple_conv_kernel_size)
        self.short_conv_dilation = int(config.ngram_size)
        self.conv_state_len = (self.conv_kernel_size - 1) * self.short_conv_dilation
        self.num_spec_tokens = vllm_config.num_speculative_tokens
        self.activation = "silu"
        self.ple_embedding = PB27Qwen4ExpNGramEmbedding(
            config,
            int(config.ple_embed_dim),
            self.ple_dense_layer_id,
            vllm_config.scheduler_config.max_num_batched_tokens,
            data_parallel_rank=vllm_config.parallel_config.data_parallel_rank,
            prefix=f"{prefix}.ple_embedding",
            quant_config=quant_config,
            params_dtype=model_config.dtype,
        )
        self.kv_proj = MergedColumnParallelLinear(
            int(config.ple_embed_dim),
            [self.hc_hidden_size, self.hidden_size],
            bias=False,
            params_dtype=model_config.dtype,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_proj",
            disable_tp=True,
        )
        norm_args = (
            self.hc_hidden_size,
            config.rms_norm_eps,
            self.hidden_size,
            model_config.dtype,
        )
        self.norm_key = Qwen4ExpPLEGroupedNorm(*norm_args)
        self.norm_query = Qwen4ExpPLEGroupedNorm(*norm_args)
        self.norm_conv = Qwen4ExpPLEGroupedNorm(*norm_args)
        self.conv1d = nn.Conv1d(
            self.hc_hidden_size,
            self.hc_hidden_size,
            self.conv_kernel_size,
            groups=self.hc_hidden_size,
            padding=self.conv_state_len,
            dilation=self.short_conv_dilation,
            bias=False,
            dtype=model_config.dtype,
        )
        nn.init.zeros_(self.conv1d.weight)
        self.conv1d.weight._no_reinit = True
        self.kv_cache = (torch.tensor([]),)
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self


class PB27Qwen4ExpDecoderLayer(Qwen4ExpDecoderLayer):
    """Qwen4ExpDecoderLayer with windowed attention + Exact Memory + parallel
    `gdn` branch on layers in `pb_hybrid_layers`."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
        hybrid_layers: frozenset[int] = frozenset(),
        attention_window: int | None = None,
        spatial_layers: frozenset[int] = frozenset(),
        spatial_intermediate_size: int = 0,
    ) -> None:
        nn.Module.__init__(self)
        config: Qwen4ExpTextConfig = vllm_config.model_config.hf_text_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)
        self.is_pb_hybrid = self.layer_idx in hybrid_layers
        self.is_pb_spatial = (
            self.layer_idx in spatial_layers and spatial_intermediate_size > 0
        )
        if vllm_config.parallel_config.use_sequence_parallel_moe:
            raise NotImplementedError(
                "Qwen4Exp HC does not support sequence-parallel MoE"
            )
        self.ple: Qwen4ExpPLELayer | None = None
        ple_layer_ids = config.ple_layer_ids
        if (self.layer_idx + 1) in ple_layer_ids:
            ple_layer_ids_sorted = sorted(set(ple_layer_ids))
            ple_dense_layer_id_map = {
                abs_id: idx for idx, abs_id in enumerate(ple_layer_ids_sorted)
            }
            ple_dense_layer_id = ple_dense_layer_id_map[self.layer_idx + 1]
            self.ple = PB27Qwen4ExpPLELayer(
                config,
                vllm_config=vllm_config,
                layer_idx=self.layer_idx,
                ple_dense_layer_id=ple_dense_layer_id,
                prefix=f"{prefix}.ple",
            )

        if self.is_pb_hybrid:
            if layer_type not in ATTENTION_LAYER_TYPES:
                raise ValueError(
                    f"PB27 (Qwen4Exp) hybrid layer {self.layer_idx} must be an "
                    f"attention layer in layer_types, got {layer_type!r}"
                )
            self.self_attn = PB27WindowedAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                sliding_window=attention_window,
            )
            self.gdn = QwenGatedDeltaNetAttention(
                config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.gdn",
                gqa_interleaved_layout=False,
            )
        elif layer_type == "linear_attention":
            self.linear_attn = QwenGatedDeltaNetAttention(
                config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,
            )
        elif layer_type in ATTENTION_LAYER_TYPES:
            use_qsa = (
                layer_type == QSA_LAYER_TYPE
                or getattr(config, "indexer_n_heads", None) is not None
            )
            if not use_qsa:
                self.self_attn = Qwen3NextAttention(
                    config,
                    model_config=model_config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.self_attn",
                )
            else:
                self.self_attn = Qwen4ExpQSAAttention(
                    vllm_config=vllm_config,
                    config=config,
                    layer_id=self.layer_idx,
                    quant_config=quant_config,
                    prefix=f"{prefix}.self_attn",
                )
        else:
            raise ValueError(f"Invalid layer_type {layer_type}")

        mlp_only_layers = getattr(config, "mlp_only_layers", [])
        num_experts = getattr(config, "num_experts", 0) or 0
        absolute_layer_id = self.layer_idx + 1
        is_moe_layer = self.layer_idx not in mlp_only_layers and (
            num_experts > 0 and absolute_layer_id % config.decoder_sparse_step == 0
        )
        if is_moe_layer:
            self.mlp = Qwen4ExpSparseMoeBlock(
                vllm_config=vllm_config, prefix=f"{prefix}.mlp"
            )
        else:
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        if self.is_pb_spatial:
            self.spatial_mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=spatial_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.spatial_mlp",
            )
        else:
            self.spatial_mlp = None

        hc_config = HyperConnectionConfig(
            hc_count=config.hc_count,
            hidden_size=config.hidden_size,
            params_dtype=torch.bfloat16,
            hc_lowrank=config.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
            hc_per_branch_norm=True,
        )
        self.attn_hyper_connection = GatedResidual(
            hc_config,
            prefix=maybe_prefix(prefix, "attn_hyper_connection"),
        )
        self.mlp_hyper_connection = GatedResidual(
            hc_config,
            prefix=maybe_prefix(prefix, "mlp_hyper_connection"),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor | None,
        prev_injection: torch.Tensor | None,
        positions: torch.Tensor,
        *,
        input_ids: torch.Tensor | None,
        query_start_loc: torch.Tensor | None,
        ngram_context: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if prev_block_output is None:
            assert prev_injection is None
        attn_hc = self.attn_hyper_connection
        if self.ple is not None:
            if prev_block_output is not None:
                hidden_states = attn_hc.combine(
                    hidden_states, prev_block_output, prev_injection
                )
                prev_block_output = prev_injection = None

            if input_ids is None or query_start_loc is None or ngram_context is None:
                raise RuntimeError("PLE inputs were not prepared")
            hidden_states = self.ple(
                hidden_states,
                input_ids,
                query_start_loc,
                ngram_context,
            )

        if prev_block_output is not None:
            hidden_states, block_input, injection = attn_hc.combine_and_mix(
                hidden_states, prev_block_output, prev_injection
            )
        else:
            hidden_states, block_input, injection = attn_hc.mix(hidden_states)

        if self.is_pb_hybrid:
            attn_out = self.self_attn(
                positions=positions,
                hidden_states=block_input,
            ) + self.gdn(hidden_states=block_input)
        elif self.layer_type == "linear_attention":
            attn_out = self.linear_attn(hidden_states=block_input)
        elif self.layer_type in ATTENTION_LAYER_TYPES:
            attn_out = self.self_attn(
                hidden_states=block_input,
                positions=positions,
            )
        else:
            raise ValueError("Invalid layer_type")

        mlp_hc = self.mlp_hyper_connection
        hidden_states, block_input, injection = mlp_hc.combine_and_mix(
            hidden_states, attn_out, injection
        )
        mlp_out = self.mlp(block_input)
        if self.spatial_mlp is not None:
            mlp_out = mlp_out + self.spatial_mlp(block_input)
        return hidden_states, mlp_out, injection


class PB27Qwen4ExpModel(Qwen4ExpModel, EagleModelMixin):
    """Qwen4ExpModel with the PB27Qwen4ExpDecoderLayer factory and Eagle3/DFlash2
    auxiliary hidden-state capture."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        config: Qwen4ExpTextConfig = vllm_config.model_config.hf_text_config
        max_model_len = int(getattr(vllm_config.model_config, "max_model_len", 0) or 0)
        if max_model_len > int(getattr(config, "max_position_embeddings", 0) or 0):
            config.max_position_embeddings = max_model_len
        self.config = config
        self.num_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, config.hidden_size)

        hybrid_layers, attention_window = pb27_hybrid_config(vllm_config)
        spatial_layers, spatial_dim = pb27_spatial_config(vllm_config)
        self._hybrid_layer_ids = hybrid_layers
        self._qsa_layer_ids = frozenset(
            layer_idx
            for layer_idx, layer_type in enumerate(config.layer_types)
            if layer_idx not in hybrid_layers
            and (
                layer_type == QSA_LAYER_TYPE
                or (
                    layer_type == "full_attention"
                    and getattr(config, "indexer_n_heads", None) is not None
                )
            )
        )
        logger.info(
            "PB27 (Qwen4Exp): %d hybrid layers %s, attention window %s, spatial layers %s (dim %d)",
            len(hybrid_layers),
            sorted(hybrid_layers),
            attention_window,
            sorted(spatial_layers),
            spatial_dim,
        )

        def get_layer(prefix: str) -> PB27Qwen4ExpDecoderLayer:
            layer_idx = extract_layer_index(prefix)
            return PB27Qwen4ExpDecoderLayer(
                vllm_config,
                layer_type=config.layer_types[layer_idx],
                prefix=prefix,
                hybrid_layers=hybrid_layers,
                attention_window=attention_window,
                spatial_layers=spatial_layers,
                spatial_intermediate_size=spatial_dim,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.is_fused_shared_expert_enabled = is_model_fused_shared_expert_compatible(
            self.layers,
            Qwen4ExpSparseMoeBlock,
            "mlp",
        )
        intermediate_size = config.hidden_size * config.hc_count
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], intermediate_size
        )

        self.hyper_connection_mixer: GatedResidual | None
        if get_pp_group().is_last_rank:
            hc_config = HyperConnectionConfig(
                hc_count=config.hc_count,
                hidden_size=config.hidden_size,
                params_dtype=torch.bfloat16,
                hc_lowrank=config.hc_lowrank,
                rms_norm_eps=config.rms_norm_eps,
                hc_per_branch_norm=True,
            )
            self.hyper_connection_mixer = GatedResidual(
                hc_config,
                use_combine=False,
                prefix=maybe_prefix(prefix, "hyper_connection_mixer"),
            )
        else:
            self.hyper_connection_mixer = None

        spec_config = vllm_config.speculative_config
        needs_mtp_hidden = (
            spec_config is not None
            and getattr(spec_config, "method", None) == "mtp"
            and get_pp_group().is_last_rank
        )
        if needs_mtp_hidden:
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.hc_count * config.hidden_size,
                dtype=vllm_config.model_config.dtype,
            )
        else:
            self._mtp_hidden_buffer = None

        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        ngram_context: torch.Tensor | None = None,
        deepstack_input_embeds: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError("input_ids or inputs_embeds is required")
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = hidden_states.repeat(1, self.config.hc_count)
        else:
            if intermediate_tensors is None:
                raise ValueError("pipeline stage requires intermediate tensors")
            hidden_states = intermediate_tensors["hidden_states"]

        aux_hidden_states: list[torch.Tensor] = []
        block_output = None
        injection = None
        last_layer = None
        if self.start_layer < self.end_layer:
            self._start_layer_ple_prefetch(
                self.layers[self.start_layer],
                hidden_states,
                input_ids,
                query_start_loc,
                ngram_context,
            )
        for layer_idx, layer in islice(
            enumerate(self.layers), self.start_layer, self.end_layer
        ):
            last_layer = layer
            if layer_idx + 1 < self.end_layer:
                self._start_layer_ple_prefetch(
                    self.layers[layer_idx + 1],
                    hidden_states,
                    input_ids,
                    query_start_loc,
                    ngram_context,
                )
            hidden_states, block_output, injection = layer(
                hidden_states=hidden_states,
                prev_block_output=block_output,
                prev_injection=injection,
                positions=positions,
                input_ids=input_ids,
                query_start_loc=query_start_loc,
                ngram_context=ngram_context,
            )
            if deepstack_input_embeds is not None and layer_idx < len(
                deepstack_input_embeds
            ):
                deepstack_embed = deepstack_input_embeds[
                    f"deepstack_input_embeds_{layer_idx}"
                ]
                deepstack_embed = (
                    deepstack_embed.unsqueeze(-2)
                    .expand(
                        *deepstack_embed.shape[:-1],
                        self.config.hc_count,
                        self.config.hidden_size,
                    )
                    .flatten(-2)
                )
                hidden_states = layer.mlp_hyper_connection.combine(
                    hidden_states, block_output, injection
                )
                block_output = None
                injection = None
                hidden_states = hidden_states + deepstack_embed

            if layer_idx + 1 in self.aux_hidden_state_layers:
                completed = (
                    hidden_states
                    if block_output is None
                    else layer.mlp_hyper_connection.combine(
                        hidden_states, block_output, injection
                    )
                )
                aux_hidden_states.append(
                    completed.unflatten(
                        -1, (self.config.hc_count, self.config.hidden_size)
                    ).mean(dim=-2)
                )

        if not get_pp_group().is_last_rank:
            if last_layer is not None and block_output is not None:
                hidden_states = last_layer.mlp_hyper_connection.combine(
                    hidden_states, block_output, injection
                )
            return IntermediateTensors({"hidden_states": hidden_states})

        final_mixer = self.hyper_connection_mixer
        assert final_mixer is not None
        multi_hidden, sample_hidden_states, _ = final_mixer.combine_and_mix(
            hidden_states, block_output, injection
        )
        if self._mtp_hidden_buffer is not None:
            num_tokens = multi_hidden.shape[0]
            self._mtp_hidden_buffer[:num_tokens].copy_(multi_hidden)
        if self.aux_hidden_state_layers:
            return sample_hidden_states, aux_hidden_states
        return sample_hidden_states

    def load_weights(self, weights) -> set[str]:
        hybrid_indexer_markers = tuple(
            f"layers.{layer_idx}.self_attn.indexer."
            for layer_idx in sorted(self._hybrid_layer_ids)
        )
        if hybrid_indexer_markers:
            weights = (
                (name, weight)
                for name, weight in weights
                if not any(m in name for m in hybrid_indexer_markers)
            )
        return super().load_weights(weights)


class PB27Qwen4ExpForCausalLM(Qwen4ExpForCausalLM, SupportsEagle3):
    """Qwen4ExpForCausalLM backed by PB27Qwen4ExpModel."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        config: Qwen4ExpTextConfig = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.quant_config = vllm_config.quant_config
        self.config = config
        self.scheduler_config = vllm_config.scheduler_config
        if vllm_config.cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "PB27 (Qwen4Exp) does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )
        self.model = PB27Qwen4ExpModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.set_moe_parameters(self.model.layers)
        enable_qwen4_exp_low_latency_gemm(self, self.model_config.dtype)
        if self.model_config.dtype == torch.bfloat16:
            request_hc_down_silu_warmup(
                vllm_config.compilation_config.cudagraph_capture_sizes or (),
                self.config.hc_lowrank,
                self.config.hc_count,
                self.config.hidden_size * self.config.hc_count,
            )

    @property
    def get_mtp_target_hidden_states(self):
        if self.model._mtp_hidden_buffer is None:
            raise AttributeError(
                "get_mtp_target_hidden_states is only active for method='mtp'"
            )
        return super().get_mtp_target_hidden_states


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen4ExpProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class PB27Qwen4ExpForConditionalGeneration(Qwen4ExpForConditionalGeneration):
    """Qwen4ExpForConditionalGeneration backed by PB27Qwen4ExpForCausalLM."""

    @property
    def get_mtp_target_hidden_states(self):
        if self.language_model.model._mtp_hidden_buffer is None:
            raise AttributeError(
                "get_mtp_target_hidden_states is only active for method='mtp'"
            )
        return super().get_mtp_target_hidden_states

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None:
        nn.Module.__init__(self)
        config: Qwen4ExpConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        if multimodal_config is None:
            raise ValueError(
                "PB27Qwen4ExpForConditionalGeneration requires multimodal_config"
            )

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.language_model_only = multimodal_config.language_model_only
        if self.language_model_only:
            self.use_data_parallel = False
            self.is_multimodal_pruning_enabled = False
            self.video_pruning_method = None
            self.video_pruning_rate = 0.0
            self._tokenizer = None
            self.visual = StageMissingLayer("vision_tower")
            self._tower_model_names = []
        else:
            self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
            self._init_video_pruning(multimodal_config)
            self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)

            with self._mark_tower_model(vllm_config, {"image", "video"}):
                self.visual = Qwen3_VisionTransformer(
                    config.vision_config,
                    norm_eps=config.text_config.rms_norm_eps,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "visual"),
                )

        self.use_deepstack = (
            not self.language_model_only
            and bool(config.vision_config.deepstack_visual_indexes)
            and not isinstance(self.visual, StageMissingLayer)
        )
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        if self.use_deepstack:
            self.deepstack_input_embeds = [
                torch.zeros(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    config.text_config.hidden_size,
                )
                for _ in range(self.deepstack_num_level)
            ]
            self.deepstack_input_embeds_num_tokens = 0

        with self._mark_language_model(vllm_config):
            self.language_model = PB27Qwen4ExpForCausalLM(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        if not get_pp_group().is_first_rank and self.use_deepstack:
            assert self.language_model.model.start_layer >= len(
                config.vision_config.deepstack_visual_indexes
            ), (
                "start_layer should be greater than or equal to "
                "len(deepstack_visual_indexes)"
            )
        self.set_moe_parameters(self.language_model.model.layers)
