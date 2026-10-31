"""Skill transplant: a pretrained Llama-family transformer -> a PseudoBrain core.

docs/SKILL_TRANSPLANT.md has the method, the citations and the results. This module
holds the pieces that need only torch (no transformers / tokenizers), so tests and
training harnesses can import it anywhere:

* ``SafeTensors`` / ``read_safetensors``: a dependency-free, lazy safetensors reader
  (single file or a sharded ``model.safetensors.index.json``).
* ``TeacherConfig`` / ``LlamaTeacher``: the Llama / Qwen2 decoder (RMSNorm, rotary
  q/k with the rotate-half convention, grouped-query attention, SwiGLU MLP, tied or
  separate head), with per-layer taps for alignment: each layer's residual input and
  its attention output (the o_proj output added to the residual). ``load_hf_core`` maps
  a Qwen3.5 / Qwen3.8 hybrid (3 Gated DeltaNet layers : 1 gated attention layer)
  straight into PB modules: its Gated DeltaNet layers ARE PB layers (copied exactly).
* ``TokenCore``: the token-level student. Its blocks are PseudoBrain heavy-core
  layers (efficiency/gated_deltanet.py GatedDeltaNetLayer or efficiency/scan.py
  ScanLayer, each with the core SwiGLU MLP and RMS pre-norms: EfficiencyOptions
  core_norm='rms'), optionally with some layers kept as (sliding-window) softmax
  attention (``TransplantAttentionLayer``, the hybrid). The teacher's embeddings,
  norms and MLPs are copied exactly (``transplant_from_teacher``); every attention
  is replaced by a PB mixer initialized from the teacher's q/k/v/o projections
  (Mamba-in-the-Llama / RADLADS-style).
* Alignment: ``mixer_delta`` (a block's mixer branch alone), ``normalized_mse``,
  ``topk_targets`` / ``topk_kd_loss`` (sparse top-k KL), ``chunked_head_kd``.
* Byte <-> token alignment for the byte-level PB student: ``patch_token_index``
  (the teacher position each 4-byte heavy patch is aligned to) and
  ``memory_mask_words`` (the "memory mask": tokens of capitalized words and
  multi-digit numbers that do not occur earlier in the context).
* ``load_core_into_hierarchy``: copies a converted all-Gated-DeltaNet ``TokenCore``
  into a hierarchical PseudoBrain (core_readout='residual').
"""
from __future__ import annotations

import json
import math
import re
import struct
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from . import moe_kernels
from .gated_deltanet import GatedDeltaNetLayer
from .scan import ScanLayer, mlp_hidden_width

# --------------------------------------------------------------------------- safetensors

_ST_DTYPES = {
    "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
    "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
    "BOOL": torch.bool,
}


class SafeTensors:
    """Lazy reader of one safetensors file or a sharded checkpoint directory.

    ``get(name)`` reads only that tensor's bytes (a 27B teacher never has to fit in
    memory at once). Format: 8-byte little-endian header length, a JSON header
    {name: {dtype, shape, data_offsets}}, then the raw little-endian data."""

    def __init__(self, path: str | Path):
        path = Path(path).expanduser()
        self.files: dict[str, tuple[Path, dict, int]] = {}
        if path.is_dir():
            index = path / "model.safetensors.index.json"
            if index.exists():
                weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
                shards = sorted(set(weight_map.values()))
            else:
                shards = sorted(p.name for p in path.glob("*.safetensors"))
            if not shards:
                raise FileNotFoundError(f"no safetensors files in {path}")
            for shard in shards:
                self._add(path / shard)
        else:
            self._add(path)

    def _add(self, file: Path) -> None:
        with open(file, "rb") as handle:
            (length,) = struct.unpack("<Q", handle.read(8))
            header = json.loads(handle.read(length))
        header.pop("__metadata__", None)
        for name, info in header.items():
            self.files[name] = (file, info, 8 + length)

    def keys(self) -> list[str]:
        return list(self.files)

    def __contains__(self, name: str) -> bool:
        return name in self.files

    def get(self, name: str, dtype: torch.dtype | None = None) -> torch.Tensor:
        file, info, base = self.files[name]
        start, end = info["data_offsets"]
        with open(file, "rb") as handle:
            handle.seek(base + start)
            raw = bytearray(handle.read(end - start))
        kind = _ST_DTYPES[info["dtype"]]
        tensor = torch.frombuffer(raw, dtype=kind) if raw else torch.empty(0, dtype=kind)
        tensor = tensor.reshape(info["shape"])
        return tensor if dtype is None else tensor.to(dtype)


def read_safetensors(path: str | Path, dtype: torch.dtype | None = None) -> dict[str, torch.Tensor]:
    reader = SafeTensors(path)
    return {name: reader.get(name, dtype) for name in reader.keys()}


def write_safetensors(path: str | Path, tensors: Mapping[str, torch.Tensor]) -> None:
    """Minimal writer (tests and small exports)."""
    names = {v: k for k, v in _ST_DTYPES.items()}
    header, chunks, offset = {}, [], 0
    for name, tensor in tensors.items():
        data = tensor.detach().cpu().contiguous()
        raw = data.reshape(-1).view(torch.uint8).numpy().tobytes() if data.numel() else b""
        header[name] = {"dtype": names[data.dtype], "shape": list(data.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        chunks.append(raw)
        offset += len(raw)
    blob = json.dumps(header).encode("utf-8")
    blob += b" " * ((-len(blob)) % 8)
    with open(Path(path), "wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)
        for raw in chunks:
            handle.write(raw)


# --------------------------------------------------------------------------- teacher

FULL, LINEAR = "full_attention", "linear_attention"


@dataclass(frozen=True)
class TeacherConfig:
    """Shape of a Llama / Qwen2 / Mistral transformer or a Qwen3.5 / Qwen3.8 hybrid
    (Gated DeltaNet 'linear_attention' layers + gated 'full_attention' layers)."""
    vocab: int
    width: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    hidden: int
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    qkv_bias: bool = False
    bos_id: int | None = None
    eos_id: int | None = None
    model_type: str = "llama"
    # Qwen3.5-style fields (defaults = a plain Llama transformer).
    layer_types: tuple = ()          # per layer FULL | LINEAR; () = every layer FULL
    rotary_dim: int = 0              # rotated dims per head (0 = head_dim; partial RoPE)
    attn_output_gate: bool = False   # q_proj also yields a per-channel sigmoid output gate
    qk_norm: bool = False            # per-head RMSNorm on q and k
    zero_centered_norm: bool = False  # checkpoint norms store w with scale (1 + w)
    linear_key_heads: int = 0
    linear_value_heads: int = 0
    linear_key_dim: int = 0
    linear_value_dim: int = 0
    linear_conv: int = 4
    # Multimodal RoPE (Qwen3.5 vision inputs): rotary pairs split into (t, h, w) sections,
    # interleaved. Text-only positions make it the plain 1D RoPE.
    mrope_section: tuple = ()
    image_token_id: int | None = None
    vision_start_id: int | None = None
    vision_end_id: int | None = None
    # Sparse mixture-of-experts MLPs (Qwen3.5/3.6-MoE): 0 experts = dense SwiGLU of `hidden`.
    num_experts: int = 0
    experts_per_token: int = 0
    moe_hidden: int = 0
    shared_expert_hidden: int = 0

    def types(self) -> tuple:
        return tuple(self.layer_types) or (FULL,) * self.layers

    def gdn_dims(self) -> dict:
        """Gated DeltaNet layer shape used for every student GDN layer: the teacher's own
        linear-attention shape when it has one, else the attention's heads."""
        if self.linear_key_heads:
            return dict(heads=self.linear_key_heads, key_dim=self.linear_key_dim,
                        value_heads=self.linear_value_heads, value_dim=self.linear_value_dim,
                        conv=self.linear_conv)
        return dict(heads=self.heads, key_dim=self.head_dim, value_heads=self.heads,
                    value_dim=self.head_dim, conv=4)

    @classmethod
    def from_hf(cls, config: Mapping) -> "TeacherConfig":
        text = config.get("text_config") or config
        kind = config.get("model_type", "llama")
        if kind not in ("llama", "qwen2", "mistral", "qwen3_5", "qwen3_5_text", "qwen3_5_moe",
                        "qwen3_5_moe_text", "qwen4_exp", "qwen4_exp_text"):
            raise ValueError(f"unsupported teacher model_type {kind!r}")
        heads = int(text["num_attention_heads"])
        width = int(text["hidden_size"])
        head_dim = int(text.get("head_dim") or width // heads)
        rope = text.get("rope_parameters") or {}
        if text.get("rope_scaling") or rope.get("rope_type", "default") != "default":
            raise ValueError("scaled RoPE teachers are not supported")
        theta = float(rope.get("rope_theta") or text.get("rope_theta") or 10000.0)
        factor = float(rope.get("partial_rotary_factor") or text.get("partial_rotary_factor") or 1.0)
        tie = text.get("tie_word_embeddings", config.get("tie_word_embeddings"))
        hybrid = kind.startswith(("qwen3_5", "qwen4_exp"))
        return cls(
            vocab=int(text["vocab_size"]), width=width, layers=int(text["num_hidden_layers"]),
            heads=heads, kv_heads=int(text.get("num_key_value_heads") or heads), head_dim=head_dim,
            hidden=int(text.get("intermediate_size") or text.get("moe_intermediate_size")), rope_theta=theta,
            norm_eps=float(text.get("rms_norm_eps", 1e-5)), tie_embeddings=bool(tie),
            qkv_bias=kind == "qwen2" or bool(text.get("attention_bias", False)),
            bos_id=text.get("bos_token_id", config.get("bos_token_id")),
            eos_id=text.get("eos_token_id", config.get("eos_token_id")), model_type=kind,
            layer_types=tuple(text.get("layer_types") or ()),
            rotary_dim=int(head_dim * factor) if factor != 1.0 else 0,
            attn_output_gate=bool(text.get("attn_output_gate", bool(text.get("output_gate_type")))), qk_norm=hybrid,
            zero_centered_norm=hybrid,
            linear_key_heads=int(text.get("linear_num_key_heads") or 0),
            linear_value_heads=int(text.get("linear_num_value_heads") or 0),
            linear_key_dim=int(text.get("linear_key_head_dim") or 0),
            linear_value_dim=int(text.get("linear_value_head_dim") or 0),
            linear_conv=int(text.get("linear_conv_kernel_dim") or 4),
            mrope_section=tuple(rope.get("mrope_section") or ()),
            image_token_id=config.get("image_token_id"), vision_start_id=config.get("vision_start_token_id"),
            vision_end_id=config.get("vision_end_token_id"),
            num_experts=int(text.get("num_experts") or 0),
            experts_per_token=int(text.get("num_experts_per_tok") or 0),
            moe_hidden=int(text.get("moe_intermediate_size") or 0),
            shared_expert_hidden=int(text.get("shared_expert_intermediate_size") or 0))


def rope_cos_sin(length: int, head_dim: int, theta: float, device, offset: int = 0):
    """HF rotary tables (non-interleaved halves): cos/sin [length, head_dim], fp32."""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    angles = torch.outer(torch.arange(offset, offset + length, device=device, dtype=torch.float32), inv)
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos(), angles.sin()


def rope_cos_sin_positions(positions: torch.Tensor, head_dim: int, theta: float,
                           mrope_section: Sequence[int] = ()):
    """Rotary tables for explicit positions: [B,T] (1D) or [3,B,T] (multimodal t/h/w, Qwen3.5
    interleaved M-RoPE: rotary pair i takes its angle from the h positions when i % 3 == 1
    and i < 3*section[1], from w when i % 3 == 2 and i < 3*section[2], else from t, exactly
    as transformers' Qwen3_5TextRotaryEmbedding.recomposition_frequencies). Returns cos/sin
    [B,1,T,head_dim] (broadcast over heads), fp32."""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.float32) / head_dim))
    if positions.ndim == 2:
        positions = positions[None].expand(3, -1, -1)
    angles = positions.float()[..., None] * inv  # [3,B,T,half]
    mixed = angles[0].clone()
    if mrope_section:
        for dim in (1, 2):
            idx = slice(dim, mrope_section[dim] * 3, 3)
            mixed[..., idx] = angles[dim][..., idx]
    mixed = torch.cat((mixed, mixed), dim=-1)[:, None]
    return mixed.cos(), mixed.sin()


def mrope_positions(token_types: Sequence[int], grids: Sequence[Sequence[int]], merge: int = 2) -> torch.Tensor:
    """[3, T] Qwen3.5 M-RoPE position ids of one sequence (transformers' get_rope_index for
    images): text runs count up on all three axes; an image of grid (t, h, w) patches
    (h, w before the spatial merge) takes (start + t_idx, start + h_idx, start + w_idx) over
    its (t, h/merge, w/merge) tokens and advances the position by max(h, w)/merge.
    token_types: 0 text, 1 image (one run per image, in order of `grids`)."""
    out, pos, i, img = [], 0, 0, 0
    types = list(token_types)
    while i < len(types):
        j = i
        while j < len(types) and types[j] == types[i]:
            j += 1
        if types[i] == 0:
            out.append(torch.arange(j - i).view(1, -1).expand(3, -1) + pos)
            pos += j - i
        else:
            t, h, w = (int(v) for v in grids[img])
            img += 1
            gh, gw = h // merge, w // merge
            tt, hh, ww = torch.meshgrid(torch.arange(t), torch.arange(gh) + pos, torch.arange(gw) + pos,
                                        indexing="ij")
            block = torch.stack((tt + pos, hh, ww)).reshape(3, -1)
            if block.shape[1] != j - i:
                raise ValueError(f"image run of {j - i} tokens does not match grid {grids[img - 1]}")
            out.append(block)
            pos += max(h, w) // merge
        i = j
    return torch.cat(out, dim=1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [B,H,T,hd]; rotate_half convention (HF Llama/Qwen) on the first cos.shape[-1]
    dims (partial RoPE), the rest passed through."""
    rd = cos.shape[-1]
    rot, rest = x[..., :rd], x[..., rd:]
    half = rd // 2
    rotated = torch.cat((-rot[..., half:], rot[..., :half]), dim=-1)
    out = (rot.float() * cos + rotated.float() * sin).to(x.dtype)
    return out if rest.shape[-1] == 0 else torch.cat((out, rest), dim=-1)


# Sliding-window training attention through FlexAttention's block-sparse kernel (skips the masked
# blocks: 4.9x faster forward+backward than SDPA's dense-mask path at 4096 tokens, window 1024,
# head dim 256 on the GB10). Its default backward tiles need 112 KB of shared memory; the GB10 has
# 99 KB, hence the 32x32 tiles. Used with gradients enabled on CUDA only (fixed training shapes);
# inference and evaluation keep SDPA (shapes vary, and a recompiling flex falls back to a slow path).
FLEX_WINDOW = {"enabled": True, "max_shapes": 8,
               "kernel_options": {"BLOCK_M1": 32, "BLOCK_N1": 32, "BLOCK_M2": 32, "BLOCK_N2": 32, "num_stages": 1}}
_FLEX_STATE: dict = {"fn": None, "masks": {}}


@torch.compiler.disable
def _flex_window_attention(q, k, v, window: int, sinks: int) -> torch.Tensor:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    if _FLEX_STATE["fn"] is None:
        _FLEX_STATE["fn"] = torch.compile(flex_attention, dynamic=False)
    length = q.shape[2]
    key = (length, window, sinks, str(q.device))
    mask = _FLEX_STATE["masks"].get(key)
    if mask is None:
        def mask_mod(b, h, qi, ki):
            return (ki <= qi) & ((qi - ki < window) | (ki < sinks))
        mask = create_block_mask(mask_mod, None, None, length, length, device=q.device)
        _FLEX_STATE["masks"][key] = mask
    return _FLEX_STATE["fn"](q, k, v, block_mask=mask, enable_gqa=q.shape[1] != k.shape[1],
                             kernel_options=FLEX_WINDOW["kernel_options"])


def _use_flex(q, window, length) -> bool:
    return (FLEX_WINDOW["enabled"] and q.is_cuda and torch.is_grad_enabled() and length >= 2 * window
            and (len(_FLEX_STATE["masks"]) < FLEX_WINDOW["max_shapes"]
                 or any(key[0] == length for key in _FLEX_STATE["masks"])))


# Windowed attention without FlexAttention (evaluation, CPU): queries in chunks of this many
# positions, each against only the keys it can see (its sinks and the window before it), so
# memory and work stay O(T * (window + chunk)) instead of a dense [T, T] mask; exact.
WINDOW_CHUNK = {"queries": 4096}


def _chunked_window_attention(q, k, v, window: int, sinks: int, chunk: int) -> torch.Tensor:
    """q/k/v [B,H,T,hd] (k/v already repeated to H heads): sliding window + sinks, exact."""
    length = q.shape[2]
    outs = []
    for s in range(0, length, chunk):
        e = min(length, s + chunk)
        lo = max(0, s - window + 1)
        sink_end = min(sinks, lo)  # sink positions before the window run [lo, e)
        kpos = torch.cat((torch.arange(sink_end, device=q.device), torch.arange(lo, e, device=q.device)))
        kk = torch.cat((k[:, :, :sink_end], k[:, :, lo:e]), dim=2)
        vv = torch.cat((v[:, :, :sink_end], v[:, :, lo:e]), dim=2)
        qpos = torch.arange(s, e, device=q.device)
        mask = (kpos[None, :] <= qpos[:, None]) & (((qpos[:, None] - kpos[None, :]) < window)
                                                  | (kpos[None, :] < sinks))
        outs.append(F.scaled_dot_product_attention(q[:, :, s:e], kk, vv, attn_mask=mask))
    return torch.cat(outs, dim=2)


@torch.compiler.disable
def _flex_cached_attention(q, k, v, offset: int, held: int, window: int | None, sinks: int) -> torch.Tensor:
    """cached_attention through FlexAttention's block-sparse kernel (training on CUDA: no masked
    SDPA under autograd). k/v keep their KV heads (GQA)."""
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    if _FLEX_STATE["fn"] is None:
        _FLEX_STATE["fn"] = torch.compile(flex_attention, dynamic=False)
    t, total = q.shape[2], k.shape[2]
    run_start = offset + t - (total - held)
    key = ("cached", t, total, offset, held, window, sinks, str(q.device))
    mask = _FLEX_STATE["masks"].get(key)
    if mask is None:
        def mask_mod(b, h, qi, ki):
            kp = torch.where(ki < held, ki, ki - held + run_start)
            qp = qi + offset
            visible = kp <= qp
            if window is not None:
                visible = visible & (((qp - kp) < window) | (kp < sinks))
            return visible
        mask = create_block_mask(mask_mod, None, None, t, total, device=q.device)
        _FLEX_STATE["masks"][key] = mask
    return _FLEX_STATE["fn"](q, k, v, block_mask=mask, enable_gqa=q.shape[1] != k.shape[1],
                             kernel_options=FLEX_WINDOW["kernel_options"])


def cached_attention(q, k, v, offset: int, held: int, window: int | None, sinks: int,
                     chunk: int | None = None) -> torch.Tensor:
    """Queries at absolute positions offset..offset+t-1 against cached + new keys, exactly as the
    masked path: q [B,H,t,hd], k/v [B,H,S,hd] (already repeated to H heads) laid out as `held`
    sink entries (positions 0..held-1) then a contiguous run ending at position offset+t-1. Query
    chunks attend only to the keys they can see (sinks + window, or everything for full attention),
    so a long chunk continuing a stream (truncated-BPTT training, chunked prefill) needs no dense
    [t, S] mask."""
    t, total = q.shape[2], k.shape[2]
    run_start = offset + t - (total - held)  # absolute position of the first run entry
    chunk = chunk or WINDOW_CHUNK["queries"]
    outs = []
    for s in range(0, t, chunk):
        e = min(t, s + chunk)
        a_pos, b_pos = offset + s, offset + e  # query positions [a_pos, b_pos)
        lo = run_start if window is None else max(run_start, a_pos - window + 1)
        sink_run = 0 if window is None else max(0, min(sinks - run_start, lo - run_start))  # sinks inside the run
        pieces = [(0, held), (held, held + sink_run), (held + lo - run_start, held + b_pos - run_start)]
        index = torch.cat([torch.arange(i, j, device=q.device) for i, j in pieces if j > i])
        kpos = torch.where(index < held, index, index - held + run_start)
        qpos = torch.arange(a_pos, b_pos, device=q.device)
        mask = kpos[None, :] <= qpos[:, None]
        if window is not None:
            mask &= ((qpos[:, None] - kpos[None, :]) < window) | (kpos[None, :] < sinks)
        outs.append(F.scaled_dot_product_attention(q[:, :, s:e], k[:, :, index], v[:, :, index], attn_mask=mask))
    return torch.cat(outs, dim=2)


def causal_attention(q, k, v, window: int | None = None, sinks: int = 0) -> torch.Tensor:
    """q [B,H,T,hd], k/v [B,KV,T,hd] (GQA repeat); full causal or a sliding window of
    `window` keys (the query's own position included), plus the first `sinks` positions,
    which stay visible to every query (attention sinks, StreamingLLM: a window that evicts
    the first tokens drives softmax attention out of distribution)."""
    length = q.shape[2]
    if window is not None and window < length and _use_flex(q, window, length):
        return _flex_window_attention(q, k, v, window, sinks)
    groups = q.shape[1] // k.shape[1]
    if groups > 1:
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    if window is None or window >= length:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    chunk = WINDOW_CHUNK["queries"]
    if length > chunk:
        return _chunked_window_attention(q, k, v, window, sinks, chunk)
    pos = torch.arange(length, device=q.device)
    mask = (pos[None, :] <= pos[:, None]) & ((pos[:, None] - pos[None, :] < window) | (pos[None, :] < sinks))
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


class TeacherLayer(nn.Module):
    def __init__(self, c: TeacherConfig):
        super().__init__()
        self.c = c
        self.input_norm = nn.RMSNorm(c.width, eps=c.norm_eps)
        self.q = nn.Linear(c.width, c.heads * c.head_dim, bias=c.qkv_bias)
        self.k = nn.Linear(c.width, c.kv_heads * c.head_dim, bias=c.qkv_bias)
        self.v = nn.Linear(c.width, c.kv_heads * c.head_dim, bias=c.qkv_bias)
        self.o = nn.Linear(c.heads * c.head_dim, c.width, bias=False)
        self.post_norm = nn.RMSNorm(c.width, eps=c.norm_eps)
        self.gate = nn.Linear(c.width, c.hidden, bias=False)
        self.up = nn.Linear(c.width, c.hidden, bias=False)
        self.down = nn.Linear(c.hidden, c.width, bias=False)

    def attention(self, x, cos, sin, window=None):
        c = self.c
        b, t, _ = x.shape
        h = self.input_norm(x)
        q = self.q(h).view(b, t, c.heads, c.head_dim).transpose(1, 2)
        k = self.k(h).view(b, t, c.kv_heads, c.head_dim).transpose(1, 2)
        v = self.v(h).view(b, t, c.kv_heads, c.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = causal_attention(q, k, v, window)
        return self.o(a.transpose(1, 2).reshape(b, t, c.heads * c.head_dim))

    def mlp(self, x):
        h = self.post_norm(x)
        return self.down(F.silu(self.gate(h)) * self.up(h))

    def forward(self, x, cos, sin):
        attn = self.attention(x, cos, sin)
        x = x + attn
        return x + self.mlp(x), attn


class LlamaTeacher(nn.Module):
    """Llama / Qwen2 / Mistral decoder (no KV cache: whole sequences only). The reference
    the exact TokenCore is tested against; hybrid teachers load straight into a TokenCore
    (load_hf_core)."""

    def __init__(self, c: TeacherConfig):
        super().__init__()
        if any(t != FULL for t in c.types()):
            raise ValueError("LlamaTeacher is attention-only; load hybrids with load_hf_core")
        self.c = c
        self.embed = nn.Embedding(c.vocab, c.width)
        self.layers = nn.ModuleList(TeacherLayer(c) for _ in range(c.layers))
        self.norm = nn.RMSNorm(c.width, eps=c.norm_eps)
        self.head = None if c.tie_embeddings else nn.Linear(c.width, c.vocab, bias=False)

    def head_weight(self) -> torch.Tensor:
        return self.embed.weight if self.head is None else self.head.weight

    def forward(self, ids: torch.Tensor, *, collect: bool = False, hidden_layers: Sequence[int] = (),
                logits: bool = True):
        """ids [B,T]. Returns dict: 'logits' [B,T,V] (if logits), 'final' (normed last hidden),
        with collect=True 'inputs' (each layer's residual input) and 'attn' (each layer's
        attention output), and 'hidden' {l: residual output of layer l} for hidden_layers."""
        cos, sin = rope_cos_sin(ids.shape[1], self.c.head_dim, self.c.rope_theta, ids.device)
        x = self.embed(ids)
        out = {"inputs": [], "attn": [], "hidden": {}}
        wanted = set(hidden_layers)
        for i, layer in enumerate(self.layers):
            if collect:
                out["inputs"].append(x)
            x, attn = layer(x, cos, sin)
            if collect:
                out["attn"].append(attn)
            if i in wanted:
                out["hidden"][i] = x
        final = self.norm(x)
        out["final"] = final
        if logits:
            out["logits"] = F.linear(final, self.head_weight())
        return out

    @classmethod
    def from_pretrained(cls, path: str | Path, dtype: torch.dtype = torch.bfloat16,
                        device="cpu") -> "LlamaTeacher":
        path = Path(path).expanduser()
        config = TeacherConfig.from_hf(json.loads((path / "config.json").read_text(encoding="utf-8")))
        with torch.device("meta"):
            model = cls(config)
        reader = SafeTensors(path)
        state = {}
        for name, param in model.named_parameters():
            state[name] = reader.get(hf_name(name), dtype)
        model.load_state_dict(state, strict=True, assign=True)
        return model.to(device)


_HF_LAYER = {"input_norm": "input_layernorm", "post_norm": "post_attention_layernorm",
             "q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj",
             "o": "self_attn.o_proj", "gate": "mlp.gate_proj", "up": "mlp.up_proj",
             "down": "mlp.down_proj"}


def hf_name(name: str) -> str:
    """LlamaTeacher parameter name -> Hugging Face checkpoint name."""
    if name == "embed.weight":
        return "model.embed_tokens.weight"
    if name == "norm.weight":
        return "model.norm.weight"
    if name == "head.weight":
        return "lm_head.weight"
    parts = name.split(".")
    if parts[0] == "layers":
        return f"model.layers.{parts[1]}.{_HF_LAYER[parts[2]]}.{parts[3]}"
    raise KeyError(name)


# --------------------------------------------------------------------------- student blocks

class TransplantAttentionLayer(nn.Module):
    """A teacher attention layer kept as softmax attention (optionally a sliding window)
    inside the PB core, with the PB core-layer conventions (pre-RMSNorm, residual branch
    scaled by residual_scale, core SwiGLU MLP). Covers Llama attention and Qwen3.5's
    gated attention (per-head q/k RMSNorm, partial RoPE, sigmoid output gate)."""

    def __init__(self, c: TeacherConfig, residual_scale: float, window: int | None = None, sinks: int = 0):
        super().__init__()
        self.c = c
        self.window = window
        self.sinks = sinks if window is not None else 0
        self.norm = nn.RMSNorm(c.width, eps=c.norm_eps)
        q_rows = c.heads * c.head_dim * (2 if c.attn_output_gate else 1)
        self.q = nn.Linear(c.width, q_rows, bias=c.qkv_bias)
        self.k = nn.Linear(c.width, c.kv_heads * c.head_dim, bias=c.qkv_bias)
        self.v = nn.Linear(c.width, c.kv_heads * c.head_dim, bias=c.qkv_bias)
        if c.qk_norm:
            self.q_norm = nn.RMSNorm(c.head_dim, eps=c.norm_eps)
            self.k_norm = nn.RMSNorm(c.head_dim, eps=c.norm_eps)
        else:
            self.q_norm = self.k_norm = None
        self.out = nn.Linear(c.heads * c.head_dim, c.width, bias=False)
        self.residual_scale = residual_scale
        self.mlp = make_mlp(c)

    def mix(self, x, cache: "AttentionCache | None" = None, positions: torch.Tensor | None = None):
        """The residual branch of the attention alone (residual_scale included). With a
        cache (streaming), queries sit at absolute positions cache.position.. and attend to
        the cached keys/values too; returns (branch, new cache) then. positions: explicit
        RoPE positions [B,T] or M-RoPE [3,B,T] for these tokens (vision inputs); causality
        always follows the sequence order."""
        c = self.c
        b, t, _ = x.shape
        h = self.norm(x)
        gate = None
        if c.attn_output_gate:
            q, gate = self.q(h).view(b, t, c.heads, 2 * c.head_dim).chunk(2, dim=-1)
            gate = gate.reshape(b, t, c.heads * c.head_dim)
        else:
            q = self.q(h).view(b, t, c.heads, c.head_dim)
        k = self.k(h).view(b, t, c.kv_heads, c.head_dim)
        v = self.v(h).view(b, t, c.kv_heads, c.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        offset = 0 if cache is None else cache.position
        if positions is None:
            cos, sin = rope_cos_sin(t, c.rotary_dim or c.head_dim, c.rope_theta, x.device, offset)
        else:
            cos, sin = rope_cos_sin_positions(positions, c.rotary_dim or c.head_dim, c.rope_theta,
                                              c.mrope_section)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is None:
            a = causal_attention(q, k, v, self.window, self.sinks)
        else:
            held = 0  # cached sink entries (positions 0..held-1) sit before the recent run
            fresh = cache.keys is None and offset == 0
            if cache.keys is not None:
                k = torch.cat((cache.keys.to(k.dtype), k), dim=2)
                v = torch.cat((cache.values.to(v.dtype), v), dim=2)
                held = cache.sinks
            total = k.shape[2]
            kpos = torch.cat((torch.arange(held, device=x.device),
                              torch.arange(offset + t - (total - held), offset + t, device=x.device)))
            from .exact_memory import EXACT_MEMORY, archive_append, retrieve
            use_memory = EXACT_MEMORY["enabled"] and self.window is not None
            retrieved = retrieve(cache.archive, q, c.heads // c.kv_heads) if use_memory else None
            if retrieved is not None:  # archived blocks (all older than the window) + sinks/window, exact
                rk, rv = retrieved
                qpos = torch.arange(offset, offset + t, device=x.device)
                mask = (kpos[None, :] <= qpos[:, None]) & (((qpos[:, None] - kpos[None, :]) < self.window)
                                                           | (kpos[None, :] < self.sinks))
                mask = torch.cat((torch.ones(t, rk.shape[2], dtype=torch.bool, device=x.device), mask), dim=1)
                groups = c.heads // c.kv_heads
                kk = torch.cat((rk.to(k.dtype), k), dim=2)
                vv = torch.cat((rv.to(v.dtype), v), dim=2)
                if groups > 1:
                    kk, vv = kk.repeat_interleave(groups, dim=1), vv.repeat_interleave(groups, dim=1)
                a = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask)
            elif fresh:  # the first prefill chunk is a whole sequence: the forward's kernels (no [t, t] mask)
                a = causal_attention(q, k, v, self.window, self.sinks)
            elif t > WINDOW_CHUNK["queries"] and FLEX_WINDOW["enabled"] and q.is_cuda and torch.is_grad_enabled():
                a = _flex_cached_attention(q, k, v, offset, held, self.window, self.sinks)  # TBPTT training
            elif t > WINDOW_CHUNK["queries"]:  # a long chunk after a cache: query chunks, no [t, S] mask
                groups = c.heads // c.kv_heads
                kk = k.repeat_interleave(groups, dim=1) if groups > 1 else k
                vv = v.repeat_interleave(groups, dim=1) if groups > 1 else v
                a = cached_attention(q, kk, vv, offset, held, self.window, self.sinks)
            else:
                qpos = torch.arange(offset, offset + t, device=x.device)
                mask = kpos[None, :] <= qpos[:, None]
                if self.window is not None:
                    mask &= ((qpos[:, None] - kpos[None, :]) < self.window) | (kpos[None, :] < self.sinks)
                groups = c.heads // c.kv_heads
                kk = k.repeat_interleave(groups, dim=1) if groups > 1 else k
                vv = v.repeat_interleave(groups, dim=1) if groups > 1 else v
                a = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask)
            if self.window is None:
                cache = AttentionCache(k, v, offset + t)
            else:  # keep the sink positions and the last window-1 positions after them
                sink = int((kpos < self.sinks).sum())
                keep = min(total - sink, self.window - 1)
                index = torch.cat((torch.arange(sink, device=x.device),
                                   torch.arange(total - keep, total, device=x.device)))
                archive = cache.archive
                if use_memory:  # positions leaving the window go to the archive instead of being dropped
                    archive = archive_append(archive, k[:, :, sink:total - keep], v[:, :, sink:total - keep])
                cache = AttentionCache(k[:, :, index], v[:, :, index], offset + t, sink, archive)
        a = a.transpose(1, 2).reshape(b, t, -1)
        if gate is not None:
            a = a * torch.sigmoid(gate)
        branch = self.residual_scale * self.out(a)
        return branch if cache is None else (branch, cache)

    def forward(self, x):
        x = x + self.mix(x).to(x.dtype)
        mlp = self.mlp
        return x if mlp is None else x + self.residual_scale * mlp(x).to(x.dtype)


MIXERS = ("gated_deltanet", "mingru", "attention", "swa_gdn")


class SparseMoE(nn.Module):
    """A PB core block's MLP as a sparse mixture of experts (the Qwen3.5/3.6-MoE layout):
    pre-RMSNorm, softmax router over `experts` with the top-k probabilities renormalized,
    SwiGLU experts stored as stacked [E, 2I, W] / [E, W, I] tensors (the checkpoint's own
    layout), plus an always-on shared SwiGLU expert scaled by a sigmoid gate. Returns the
    residual branch like efficiency/scan.py SwiGLU. record_routing=True keeps the last
    top-k expert indices (last_routing) for router-agreement checks; added experts
    (grow_experts) start unreachable from the router and contribute nothing until trained."""

    def __init__(self, width: int, experts: int, top_k: int, hidden: int, shared_hidden: int,
                 norm_eps: float = 1e-5):
        super().__init__()
        self.norm = nn.RMSNorm(width, eps=norm_eps)
        self.router = nn.Linear(width, experts, bias=False)
        self.gate_up = nn.Parameter(torch.empty(experts, 2 * hidden, width))
        self.down = nn.Parameter(torch.empty(experts, width, hidden))
        bound = 1.0 / math.sqrt(width)
        nn.init.uniform_(self.gate_up, -bound, bound)
        nn.init.uniform_(self.down, -1.0 / math.sqrt(hidden), 1.0 / math.sqrt(hidden))
        self.top_k = top_k
        if shared_hidden:
            self.shared_gate_up = nn.Linear(width, 2 * shared_hidden, bias=False)
            self.shared_down = nn.Linear(shared_hidden, width, bias=False)
            self.shared_gate = nn.Linear(width, 1, bias=False)
        else:
            self.shared_gate_up = self.shared_down = self.shared_gate = None
        self.record_routing = False
        self.last_routing = None
        self.router_bias = None  # created by grow_experts (old experts 0, new ones closed)

    # Execution only: how the routed experts run. 'auto': grouped GEMM over the active experts
    # (torch._grouped_mm: only the routed (token, expert) rows, no padding) when it works on
    # this device, else a loop over the active experts (the transformers reference);
    # 'padded': one padded bmm per projection (fast when routing is balanced, wasteful when
    # one expert takes many tokens); 'loop'.
    expert_backend = "auto"
    batched = True  # False forces the loop (kept for older callers/tests)
    _grouped_ok: bool | None = None
    # The grouped path's GEMM: 'triton' (moe_kernels.grouped_mm: one launch, no host sync; frozen experts
    # only), 'torch' (torch._grouped_mm), 'auto': triton where torch has no native grouped GEMM (anything
    # but sm_90/sm_100, e.g. GB10 and RTX PRO 6000) and the experts take no gradient, else torch.
    grouped_kernel = "auto"
    last_grouped_kernel: str | None = None  # what the grouped path last ran, for run logs
    # Memory only: without grad, run the routed experts over at most this many tokens at a time
    # (0 = all at once). Bounds the transient [tokens * top_k, width] buffers of long no-grad passes.
    token_chunk = 0

    @property
    def num_experts(self) -> int:
        return self.gate_up.shape[0]

    def _experts(self, h, weight, index):
        if moe_kernels.STATE["regular_single"] and h.shape[0] == 1:
            dtype = self._grouped_dtype(h)
            if dtype is not None:
                return self._experts_single(h, weight, index, dtype)
        if (moe_kernels.STATE["decode"] and index.numel() <= moe_kernels.STATE["max_rows"]
                and (h.is_cuda or moe_kernels.interpreting())):
            return moe_kernels.moe_decode(h, weight, index, self.gate_up, self.down)  # decode: capturable
        backend = self.expert_backend if self.batched else "loop"
        low = self._grouped_dtype(h)
        if backend == "auto" and h.is_cuda and low is not None and SparseMoE._grouped_ok is not False                 and hasattr(torch, "_grouped_mm"):
            try:
                out = self._experts_grouped(h, weight, index, low)
                SparseMoE._grouped_ok = True
                return out
            except (RuntimeError, NotImplementedError) as err:
                SparseMoE._grouped_ok = False  # unsupported here: fall back for good
                warnings.warn(f"SparseMoE: grouped experts failed ({err}); using the per-expert loop from now on")
        if backend == "padded":
            return self._experts_batched(h, weight, index)
        return self._experts_loop(h, weight, index)

    def _grouped_dtype(self, h):
        """The 16-bit dtype the grouped GEMMs run in: the input's, else autocast's, else the
        expert weights'. None (an fp32 model outside autocast) keeps the exact fp32 loop. The
        block's residual stream is fp32 under autocast, so the norm hands the MoE fp32 rows."""
        low = (torch.bfloat16, torch.float16)
        if h.dtype in low:
            return h.dtype
        if h.is_cuda and torch.is_autocast_enabled("cuda"):
            return torch.get_autocast_dtype("cuda")
        return self.gate_up.dtype if self.gate_up.dtype in low else None

    def _use_triton_grouped(self, h) -> bool:
        kernel = SparseMoE.grouped_kernel
        if kernel == "torch" or not moe_kernels.available():
            return False
        if torch.is_grad_enabled() and (self.gate_up.requires_grad or self.down.requires_grad):
            return False  # the Triton kernel has no weight gradient
        if kernel == "triton":
            return True
        return torch.cuda.get_device_capability(h.device)[0] not in (9, 10)

    def _experts_grouped(self, h, weight, index, dtype=None):
        """Only the routed rows, grouped by expert: two grouped GEMMs (moe_kernels' fused Triton path, or
        torch._grouped_mm with row offsets) in `dtype`, routing-weighted sum back into h's dtype."""
        dtype = dtype or h.dtype
        flat = index.reshape(-1)
        order = flat.argsort()
        # rows per expert with a static shape (bincount's output size is data-dependent, which
        # breaks torch.compile graphs)
        counts = torch.zeros(self.num_experts, dtype=torch.int64, device=flat.device)
        counts.index_add_(0, flat, torch.ones_like(flat))
        offs = torch.cumsum(counts, 0).to(torch.int32)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=order.device)
        if self._use_triton_grouped(h):
            # the token gather and the routing-weighted slot sum are fused into the GEMMs and their
            # gradients: no [T*k, W] gathered or permuted copies, no index_put accumulation
            SparseMoE.last_grouped_kernel = "triton"
            rows = order // self.top_k
            gate_up = moe_kernels.grouped_gather_mm(h.to(dtype), self.gate_up.to(dtype), counts, rows, inverse,
                                                    self.top_k)
            out = moe_kernels.grouped_mm_combine(moe_kernels.swiglu(gate_up), self.down.to(dtype), counts, rows,
                                                 order, inverse, weight)
            return out.to(h.dtype)
        SparseMoE.last_grouped_kernel = "torch"
        x = h.to(dtype)[order // self.top_k]  # cast the T rows, then gather the T*k routed copies
        gate, up = torch._grouped_mm(x, self.gate_up.transpose(1, 2).to(dtype), offs=offs).chunk(2, dim=-1)
        y = torch._grouped_mm(F.silu(gate) * up, self.down.transpose(1, 2).to(dtype), offs=offs)
        # back to (token, slot) order, then the routing-weighted sum over the k slots as one
        # batched [1,k]x[k,W] product per token (fp32 accumulation, no [T*k, W] fp32 buffer)
        y = y[inverse].view(-1, self.top_k, y.shape[-1])
        out = torch.bmm(weight.to(dtype)[:, None, :], y)[:, 0]
        return out.to(h.dtype)

    def _experts_single(self, h, weight, index, dtype):
        """Only selected experts, using the regular grouped fallback's GEMMs.

        PyTorch's grouped_mm fallback walks all experts and scalar offsets.
        A single token selects unique experts: one small index transfer lets us
        skip empty groups while preserving BF16 projection/activation/bmm
        rounding and the routing-slot reduction order. Serving only.
        """
        order = index.reshape(-1).argsort()
        active = index.reshape(-1)[order].tolist()
        x = h.to(dtype)
        rows = []
        for expert in active:
            gate, up = F.linear(x, self.gate_up[expert]).chunk(2, dim=-1)
            rows.append(F.linear(F.silu(gate) * up, self.down[expert]))
        y = torch.cat(rows, dim=0)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=order.device)
        y = y[inverse].view(1, self.top_k, -1)
        return torch.bmm(weight.to(dtype)[:, None, :], y)[:, 0].to(h.dtype)

    def _experts_loop(self, h, weight, index):
        out = torch.zeros_like(h)
        order = index.reshape(-1).argsort()
        token = (torch.arange(index.numel(), device=h.device) // self.top_k)[order]
        slot = order % self.top_k
        counts = torch.bincount(index.reshape(-1), minlength=self.num_experts).tolist()
        start = 0
        for expert, count in enumerate(counts):
            if not count:
                continue
            rows = token[start:start + count]
            gate, up = F.linear(h[rows], self.gate_up[expert]).chunk(2, dim=-1)
            y = F.linear(F.silu(gate) * up, self.down[expert])
            w = weight[rows, slot[start:start + count]][:, None]
            out.index_add_(0, rows, (y * w).to(out.dtype))
            start += count
        return out

    def _experts_batched(self, h, weight, index):
        """Every routed (token, expert) pair grouped by expert into [E, C, W] (C = the largest
        group, padded), two bmm calls, weighted scatter-add back."""
        experts = self.num_experts
        flat = index.reshape(-1)
        order = flat.argsort()
        sorted_expert = flat[order]
        token = (torch.arange(flat.numel(), device=h.device) // self.top_k)[order]
        w = weight.reshape(-1)[order]
        counts = torch.bincount(flat, minlength=experts)
        starts = torch.cumsum(counts, 0) - counts
        slot = torch.arange(flat.numel(), device=h.device) - starts[sorted_expert]
        capacity = int(counts.max())
        x = h.new_zeros(experts, capacity, h.shape[-1])
        x[sorted_expert, slot] = h[token]
        gate, up = torch.bmm(x, self.gate_up.transpose(1, 2).to(x.dtype)).chunk(2, dim=-1)
        y = torch.bmm(F.silu(gate) * up, self.down.transpose(1, 2).to(x.dtype))
        y = y[sorted_expert, slot] * w[:, None].to(y.dtype)
        out = torch.zeros_like(h)
        out.index_add_(0, token, y.to(out.dtype))
        return out

    @torch.no_grad()
    def grow_experts(self, count: int, *, copy_from: Sequence[int] | None = None, closed: float = -1e4) -> list[int]:
        """Add `count` routed experts for new skills, function-preserving: their router logits
        get a bias of `closed` (so no token selects them; top-k weights are renormalized over
        the selected experts only, so the old outputs are unchanged bit for bit), their
        weights are copies of `copy_from` experts (or fresh), and their router rows start as
        the copied experts' rows. Training the new experts' weights, router rows and bias on
        new-skill data opens them; the old experts can stay frozen, so old skills are not
        overwritten. Returns the new expert indices."""
        old = self.num_experts
        device, dtype = self.gate_up.device, self.gate_up.dtype
        src = list(copy_from) if copy_from is not None else []
        gate_up = torch.empty(count, *self.gate_up.shape[1:], device=device, dtype=dtype)
        down = torch.empty(count, *self.down.shape[1:], device=device, dtype=dtype)
        rows = torch.zeros(count, self.router.in_features, device=device, dtype=self.router.weight.dtype)
        for j in range(count):
            if src:
                e = src[j % len(src)]
                gate_up[j], down[j], rows[j] = self.gate_up[e], self.down[e], self.router.weight[e]
            else:
                nn.init.uniform_(gate_up[j], -1 / math.sqrt(gate_up.shape[-1]), 1 / math.sqrt(gate_up.shape[-1]))
                down[j].zero_()
        self.gate_up = nn.Parameter(torch.cat((self.gate_up.data, gate_up)))
        self.down = nn.Parameter(torch.cat((self.down.data, down)))
        router = nn.Linear(self.router.in_features, old + count, bias=False).to(device=device,
                                                                               dtype=self.router.weight.dtype)
        router.weight.copy_(torch.cat((self.router.weight, rows)))
        self.router = router
        bias = torch.zeros(old + count, device=device, dtype=torch.float32)
        if self.router_bias is not None:
            bias[:old] = self.router_bias
        bias[old:] = closed
        self.router_bias = nn.Parameter(bias)
        return list(range(old, old + count))

    def route(self, h: torch.Tensor):
        logits = self.router(h)
        if self.router_bias is not None:
            logits = logits + self.router_bias.to(logits.dtype)
        probs = torch.softmax(logits.float(), dim=-1)
        weight, index = torch.topk(probs, self.top_k, dim=-1)
        weight = weight / weight.sum(-1, keepdim=True)
        return weight.to(logits.dtype), index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        h = self.norm(x).reshape(-1, shape[-1])
        weight, index = self.route(h)
        if self.record_routing:
            self.last_routing = index.detach()
        chunk = SparseMoE.token_chunk
        if chunk and h.shape[0] > chunk and not torch.is_grad_enabled():
            # memory only: every token's routed experts are computed independently of the others
            out = torch.cat([self._experts(h[s:s + chunk], weight[s:s + chunk], index[s:s + chunk])
                             for s in range(0, h.shape[0], chunk)])
        else:
            out = self._experts(h, weight, index)
        if self.shared_gate_up is not None:
            gate, up = self.shared_gate_up(h).chunk(2, dim=-1)
            shared = self.shared_down(F.silu(gate) * up)
            out = out + torch.sigmoid(self.shared_gate(h)) * shared
        return out.reshape(shape)


def make_mlp(c: "TeacherConfig"):
    """The core MLP a teacher layer needs: dense SwiGLU or SparseMoE."""
    from .scan import SwiGLU
    if c.num_experts:
        return SparseMoE(c.width, c.num_experts, c.experts_per_token, c.moe_hidden, c.shared_expert_hidden,
                         c.norm_eps)
    return SwiGLU(c.width, c.hidden, zero_output=False, norm="rms", norm_eps=c.norm_eps)


@dataclass
class AttentionCache:
    """Streaming state of a kept attention layer: rotated keys/values [B,KV,S,hd] (all of
    them, or for a sliding window the `sinks` first positions then the last window-1) and
    the next absolute position."""
    keys: torch.Tensor | None
    values: torch.Tensor | None
    position: int = 0
    sinks: int = 0  # how many leading cached entries are sink positions 0..sinks-1
    archive: object = None  # exact memory tier (efficiency.exact_memory.Archive): evicted keys/values


@dataclass
class HybridState:
    """Streaming state of a WindowedHybridLayer: its bounded KV window and its GDN state."""
    cache: AttentionCache
    gdn: list


class WindowedHybridLayer(nn.Module):
    """A converted attention layer as the teacher's OWN attention restricted to a sliding
    window of W tokens (bounded KV cache) PLUS a Gated DeltaNet mixer for what lies beyond
    the window (LoLCATs' sliding-window + linear split, with PB's mixer). Both read the
    block's single pre-norm output. The GDN's out projection starts at zero, so at
    initialization the layer is exactly the windowed teacher attention; attention transfer
    then fits the GDN to (full attention - windowed attention). Memory per stream stays
    constant: W-1 cached keys/values plus the GDN matrix state."""

    def __init__(self, c: TeacherConfig, residual_scale: float, window: int, gdn: dict, conv: int,
                 backend: str, sinks: int = 0):
        super().__init__()
        if window is None or window < 1:
            raise ValueError("swa_gdn needs CoreSpec.attention_window")
        self.norm = nn.RMSNorm(c.width, eps=c.norm_eps)
        self.attn = TransplantAttentionLayer(c, residual_scale, window, sinks)
        self.attn.norm = nn.Identity()
        self.attn.mlp = None
        self.gdn = GatedDeltaNetLayer(c.width, residual_scale, gdn["heads"], gdn["key_dim"], conv, backend, 0,
                                      "rms", value_heads=gdn["value_heads"], value_dim=gdn["value_dim"],
                                      norm_eps=c.norm_eps, block_norm_eps=c.norm_eps, l2norm="hf")
        self.gdn.norm = nn.Identity()
        with torch.no_grad():
            self.gdn.out.weight.zero_()
        self.residual_scale = residual_scale
        self.mlp = make_mlp(c)

    def _gdn(self, h, state):
        g = self.gdn
        _, vec, matrix, tail, seq = g(h, *state, return_states=True, return_output=False)
        if g.value_width == g.width:
            seq = g.out(seq)
        return self.residual_scale * seq, [vec, matrix, tail]

    def mix(self, x, state: "HybridState | None" = None, positions: torch.Tensor | None = None):
        h = self.norm(x)
        if state is None:
            a = self.attn.mix(h, None, positions)
            g, _ = self._gdn(h, _zero_state(self.gdn, h))
            return a + g.to(a.dtype)
        a, cache = self.attn.mix(h, state.cache, positions)
        g, gdn_state = self._gdn(h, state.gdn)
        return a + g.to(a.dtype), HybridState(cache, gdn_state)

    def forward(self, x):
        x = x + self.mix(x).to(x.dtype)
        return x + mlp_branch(self, x)


def _zero_state(block: GatedDeltaNetLayer, x: torch.Tensor):
    b = x.shape[0]
    return (x.new_zeros(b, block.width, dtype=torch.float32),
            x.new_zeros(b, block.value_heads, block.key_dim, block.value_dim, dtype=torch.float32),
            x.new_zeros(b, block.tail_length, block.channels, dtype=torch.float32))


def mixer_delta(block: nn.Module, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
    """The block's mixer residual branch alone (what replaces a teacher attention output),
    from a fresh recurrent state: x + mixer_delta(x) is the block before its MLP. Same ops
    as the block's forward (no module mutation, so it traces under torch.compile)."""
    if isinstance(block, GatedDeltaNetLayer):
        _, _, _, _, seq = block(x, *_zero_state(block, x), return_states=True, return_output=False)
        if block.value_width == block.width:
            seq = block.out(seq)
        return block.residual_scale * seq.to(x.dtype)
    if isinstance(block, ScanLayer):
        _, _, hs = block(x, x.new_zeros(x.shape[0], x.shape[-1], dtype=torch.float32),
                         return_states=True, return_output=False)
        return block.residual_scale * block.out(hs).to(x.dtype)
    return block.mix(x, None, positions).to(x.dtype)


def mlp_branch(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = block.mlp(x).to(x.dtype)
    s_mlp = getattr(block, "spatial_mlp", None)
    if s_mlp is not None and getattr(s_mlp, "enabled", True):
        h = block.mlp.norm(x).to(s_mlp.gate_up.weight.dtype)
        gate, up = s_mlp.gate_up(h).chunk(2, dim=-1)
        out = out + s_mlp.down(F.silu(gate) * up).to(x.dtype)
    return block.residual_scale * out


def _state_step(block: nn.Module, x: torch.Tensor, st, positions=None):
    """One block from its streaming state: (output, new state). Gated DeltaNet [last vector,
    matrix, conv tail], minGRU [hidden], attention / windowed hybrid caches."""
    if isinstance(block, GatedDeltaNetLayer):
        x, hidden, matrix, tail = block(x, *st)
        return x, [hidden, matrix, tail]
    if isinstance(block, ScanLayer):
        x, hidden = block(x, st[0])
        return x, [hidden]
    branch, cache = block.mix(x, st, positions)
    x = x + branch.to(x.dtype)
    return x + mlp_branch(block, x), cache


def _block_step(block: nn.Module, x: torch.Tensor, positions=None) -> torch.Tensor:
    x = x + mixer_delta(block, x, positions)
    return x + mlp_branch(block, x)


def shared_student(teacher: "TokenCore", spec: "CoreSpec", *, mixer_init: bool = True,
                   trainable_dtype: torch.dtype = torch.float32) -> tuple["TokenCore", list[int]]:
    """A student that SHARES every unconverted block, the embedding, final norm and head with
    the (frozen) teacher, and gets new mixers only where `spec` replaces an attention layer
    (each new block reuses the teacher block's pre-norm and MLP modules). Memory: the teacher
    plus the new mixers - what makes a 27B conversion fit on one GB10. Returns (student,
    converted layer indices). Only the new mixers are trainable."""
    c = teacher.c
    kinds = CoreSpec.exact(c).mixers
    student = TokenCore.__new__(TokenCore)
    nn.Module.__init__(student)
    student.c, student.spec = c, spec
    student.embed, student.norm, student.head = teacher.embed, teacher.norm, teacher.head
    blocks, converted = [], []
    device = teacher.embed.weight.device
    for i, (mixer, kind, tblock) in enumerate(zip(spec.mixers, kinds, teacher.blocks)):
        if mixer == kind:
            blocks.append(tblock)
            continue
        if mixer not in ("gated_deltanet", "swa_gdn") or not isinstance(tblock, TransplantAttentionLayer):
            raise ValueError(f"layer {i}: only attention -> gated_deltanet / swa_gdn conversion is shared")
        fresh = TokenCore._fresh_block(c, spec, i)
        fresh.mlp = None
        fresh = fresh.to(device=device, dtype=trainable_dtype)
        gdn = fresh.gdn if mixer == "swa_gdn" else fresh
        if mixer_init:
            with torch.no_grad():
                gate = None
                q = tblock.q.weight.float()
                if c.attn_output_gate:
                    both = q.reshape(c.heads, 2, c.head_dim, -1)
                    q = both[:, 0].reshape(c.heads * c.head_dim, -1)
                    gate = both[:, 1].reshape(c.heads * c.head_dim, -1)
                # tblock.out already carries the PB residual scale (sqrt(L) folded in)
                init_gdn_from_attention(gdn, c, q, tblock.k.weight.float(), tblock.v.weight.float(),
                                        tblock.out.weight.float(), gate, root=1.0)
        if mixer == "swa_gdn":
            with torch.no_grad():
                gdn.out.weight.zero_()  # start as the windowed teacher attention
            for name in ("q", "k", "v", "out", "q_norm", "k_norm"):  # the teacher's own attention
                setattr(fresh.attn, name, getattr(tblock, name))
        fresh.norm = tblock.norm
        fresh.mlp = tblock.mlp
        blocks.append(fresh)
        converted.append(i)
    student.blocks = nn.ModuleList(blocks)
    for p in student.parameters():
        p.requires_grad_(False)
    for p in student.mixer_parameters(converted):
        p.requires_grad_(True)
    return student, converted


def run_block(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """One block over a whole sequence from a fresh (zero) recurrent state (the block's
    own forward)."""
    if isinstance(block, GatedDeltaNetLayer):
        return block(x, *_zero_state(block, x))[0]
    if isinstance(block, ScanLayer):
        return block(x, x.new_zeros(x.shape[0], x.shape[-1], dtype=torch.float32))[0]
    return block(x)


@dataclass
class CoreSpec:
    """Student core: one mixer per layer (len == teacher layers) + mixer options."""
    mixers: list[str]
    conv: int = 4
    attention_window: int | None = None  # window of the kept attention layers (None = full)
    backend: str = "fused"
    attention_sinks: int = 0  # with a window: first positions every query still sees (StreamingLLM)
    # Mixers that REPLACE an attention layer (gated_deltanet / swa_gdn on a full-attention layer):
    # Gated DeltaNet key dim (0 = the teacher's GDN shape). The state is value_heads x dk x dv,
    # so a larger dk is a larger far-context memory (with dk <= head_dim the init stays mapped).
    converted_key_dim: int = 0
    # Per-layer attention windows overriding attention_window: ((layer, window), ...).
    layer_windows: tuple = ()
    # Mixers that replace an attention layer get this many times the teacher's GDN heads (key and
    # value), a proportionally larger state. widen_gdn_state warm-starts them from a narrower core.
    converted_head_mult: int = 1
    # Per-layer head multipliers overriding converted_head_mult: ((layer, mult), ...).
    layer_head_mults: tuple = ()
    # Mixers that replace an attention layer get this many key and value heads (0 = gdn_dims, one per
    # query head). The teacher's KV heads keep a GQA teacher's GDN small (Nanbeige4.3-33B: 8 heads,
    # 37M parameters per layer instead of 257M); init_gdn_from_attention then maps each KV head 1:1.
    converted_heads: int = 0

    def window(self, index: int) -> int | None:
        return dict((int(i), int(w)) for i, w in self.layer_windows).get(index, self.attention_window)

    def head_mult(self, index: int) -> int:
        return dict((int(i), int(m)) for i, m in self.layer_head_mults).get(index, self.converted_head_mult)

    @classmethod
    def uniform(cls, mixer: str, layers: int, **kwargs) -> "CoreSpec":
        return cls([mixer] * layers, **kwargs)

    @classmethod
    def hybrid(cls, layers: int, every: int, **kwargs) -> "CoreSpec":
        """Gated DeltaNet everywhere except every `every`-th layer (1-based multiples),
        which keeps softmax attention."""
        return cls(["attention" if (i + 1) % every == 0 else "gated_deltanet" for i in range(layers)],
                   **kwargs)

    @classmethod
    def exact(cls, c: TeacherConfig, **kwargs) -> "CoreSpec":
        """The teacher itself in PB modules: Gated DeltaNet where the teacher has linear
        attention, attention where it has attention."""
        return cls(["gated_deltanet" if t == LINEAR else "attention" for t in c.types()], **kwargs)


class TokenCore(nn.Module):
    """Token-level PB core: the teacher's embedding/head around PB core blocks."""

    def __init__(self, teacher_config: TeacherConfig, spec: CoreSpec):
        super().__init__()
        c = teacher_config
        if len(spec.mixers) != c.layers or any(m not in MIXERS for m in spec.mixers):
            raise ValueError(f"spec.mixers must list one of {MIXERS} per teacher layer")
        self.c, self.spec = c, spec
        self.embed = nn.Embedding(c.vocab, c.width)
        scale = 1.0 / math.sqrt(c.layers)
        if not c.num_experts and mlp_hidden_width(c.width, c.hidden / c.width) != c.hidden:
            raise AssertionError("core MLP width does not reproduce the teacher's")
        dense_hidden = 0 if c.num_experts else c.hidden
        gdn = c.gdn_dims()
        conv = gdn.pop("conv") if c.linear_key_heads else spec.conv
        gdn.pop("conv", None)
        blocks = []
        kinds = c.types()
        for i, mixer in enumerate(spec.mixers):
            dims = gdn
            if kinds[i] == FULL and spec.converted_key_dim:  # a mixer that replaces attention
                dims = {**gdn, "key_dim": spec.converted_key_dim}
            if kinds[i] == FULL and spec.converted_heads:
                dims = {**dims, "heads": spec.converted_heads, "value_heads": spec.converted_heads}
            if kinds[i] == FULL and spec.head_mult(i) > 1:
                m = spec.head_mult(i)
                dims = {**dims, "heads": dims["heads"] * m, "value_heads": dims["value_heads"] * m}
            if mixer == "gated_deltanet":
                blocks.append(GatedDeltaNetLayer(
                    c.width, scale, dims["heads"], dims["key_dim"], conv, spec.backend, dense_hidden, "rms",
                    value_heads=dims["value_heads"], value_dim=dims["value_dim"], norm_eps=c.norm_eps,
                    block_norm_eps=c.norm_eps, l2norm="hf"))
                if c.num_experts:
                    blocks[-1].mlp = make_mlp(c)
            elif mixer == "mingru":
                blocks.append(ScanLayer(c.width, scale, spec.backend, dense_hidden, "rms"))
                if c.num_experts:
                    blocks[-1].mlp = make_mlp(c)
            elif mixer == "swa_gdn":
                blocks.append(WindowedHybridLayer(c, scale, spec.window(i), dims, conv, spec.backend,
                                                  spec.attention_sinks))
            else:
                blocks.append(TransplantAttentionLayer(c, scale, spec.window(i), spec.attention_sinks))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(c.width, eps=c.norm_eps)
        self.head = None if c.tie_embeddings else nn.Linear(c.width, c.vocab, bias=False)

    def head_weight(self) -> torch.Tensor:
        return self.embed.weight if self.head is None else self.head.weight

    @staticmethod
    def _fresh_block(c: "TeacherConfig", spec: "CoreSpec", index: int) -> nn.Module:
        """Block `index` of TokenCore(c, spec), freshly initialized, built alone."""
        one = CoreSpec([spec.mixers[index]], spec.conv, spec.window(index), spec.backend, spec.attention_sinks,
                       spec.converted_key_dim, converted_head_mult=spec.head_mult(index),
                       converted_heads=spec.converted_heads)
        kind = c.types()[index]
        single = TeacherConfig(**{**asdict(c), "layers": 1, "layer_types": (kind,) if c.layer_types else ()})
        block = TokenCore(single, one).blocks[0]
        for module in block.modules():  # composite blocks scale their parts too
            if hasattr(module, "residual_scale"):
                module.residual_scale = 1.0 / math.sqrt(c.layers)
        return block

    def initial_state(self, batch: int, device=None) -> list:
        """Fresh streaming state, one entry per block: Gated DeltaNet [last output vector,
        matrix state [B,Hv,dk,dv], conv tail] (fp32), minGRU [hidden], attention
        AttentionCache (empty)."""
        state = []
        for block in self.blocks:
            if isinstance(block, GatedDeltaNetLayer):
                state.append(list(_zero_state(block, torch.zeros(batch, 1, device=device))))
            elif isinstance(block, ScanLayer):
                state.append([torch.zeros(batch, self.c.width, device=device)])
            elif isinstance(block, WindowedHybridLayer):
                state.append(HybridState(AttentionCache(None, None, 0),
                                         list(_zero_state(block.gdn, torch.zeros(batch, 1, device=device)))))
            else:
                state.append(AttentionCache(None, None, 0))
        return state

    def _inputs(self, ids, inputs_embeds):
        if inputs_embeds is not None:
            return inputs_embeds.float()
        return self.embed(ids).float()

    def stream(self, ids: torch.Tensor | None, state: list | None = None, *, logits: bool = True,
               inputs_embeds: torch.Tensor | None = None, positions: torch.Tensor | None = None):
        """Prefill or step: run ids [B,T] (or inputs_embeds [B,T,D], e.g. with image
        embeddings scattered in) from `state` (None = fresh) and return (out dict, new
        state). Feeding a sequence in any chunking gives the full-sequence forward.
        positions: RoPE positions of these tokens for kept attention layers (M-RoPE [3,B,T]
        for vision inputs); recurrent layers need none."""
        x = self._inputs(ids, inputs_embeds)
        state = self.initial_state(x.shape[0], x.device) if state is None else state
        new_state = []
        for block, st in zip(self.blocks, state):
            x, st = _state_step(block, x, st, positions)
            new_state.append(st)
        final = self.norm(x)
        out = {"final": final}
        if logits:
            out["logits"] = F.linear(final.to(self.embed.weight.dtype), self.head_weight())
        return out, new_state

    def forward_state(self, ids: torch.Tensor, state: list | None = None) -> tuple[torch.Tensor, list]:
        """Training forward of one chunk of a longer row from the streaming state the previous
        chunks left (truncated backpropagation through time): (normed final hidden [B,T,W], new
        state). Blocks are activation-checkpointed when checkpoint_blocks is set and grad is on;
        the caller detaches the state between chunks. Chunk by chunk this computes the same
        function as forward() on the whole row (stream() semantics), with gradients inside each
        chunk only."""
        x = self._inputs(ids, None)
        state = self.initial_state(x.shape[0], x.device) if state is None else state
        remat = getattr(self, "checkpoint_blocks", False) and torch.is_grad_enabled()
        new_state = []
        for block, st in zip(self.blocks, state):
            if remat:
                x, st = checkpoint(_state_step, block, x, st, None, use_reentrant=False)
            else:
                x, st = _state_step(block, x, st, None)
            new_state.append(st)
        return self.norm(x), new_state

    @staticmethod
    def detach_state(state: list) -> list:
        """The streaming state with every tensor detached (the truncation point of TBPTT)."""
        def det(t):
            return None if t is None else t.detach()

        def det_archive(a):  # exact memory tier: keep the archive across TBPTT chunks, detached
            if a is None:
                return None
            from .exact_memory import Archive
            return Archive(det(a.keys), det(a.values), det(a.kmax), det(a.kmin))
        out = []
        for st in state:
            if isinstance(st, HybridState):
                c = st.cache
                out.append(HybridState(AttentionCache(det(c.keys), det(c.values), c.position, c.sinks,
                                                      det_archive(c.archive)),
                                       [det(t) for t in st.gdn]))
            elif isinstance(st, AttentionCache):
                out.append(AttentionCache(det(st.keys), det(st.values), st.position, st.sinks,
                                          det_archive(st.archive)))
            else:
                out.append([det(t) for t in st])
        return out

    @staticmethod
    def state_bytes(state: list) -> int:
        """Bytes held by a streaming state (recurrent matrices, conv tails, KV caches)."""
        total = 0
        for st in state:
            if isinstance(st, HybridState):
                tensors = [st.cache.keys, st.cache.values, *st.gdn]
            elif isinstance(st, AttentionCache):
                tensors = [st.keys, st.values]
            else:
                tensors = st
            total += sum(t.numel() * t.element_size() for t in tensors if t is not None)
        return total

    def forward(self, ids: torch.Tensor | None, *, collect: bool | Sequence[int] = False,
                hidden_layers: Sequence[int] = (), logits: bool = True,
                inputs_embeds: torch.Tensor | None = None, positions: torch.Tensor | None = None):
        """dict: 'final' (normed last hidden), 'logits' (if logits), 'hidden' {l: residual
        output of layer l}, and with collect=True 'inputs'/'mix' (each layer's residual
        input and its mixer branch: the alignment taps; collect=[layers] keeps only those
        layers' taps, None elsewhere). A whole sequence from a fresh state; stream() carries
        state across calls. inputs_embeds/positions as in stream()."""
        x = self._inputs(ids, inputs_embeds)
        out = {"hidden": {}, "inputs": [], "mix": []}
        wanted = set(hidden_layers)
        taps = set(range(len(self.blocks))) if collect is True else set(collect or ())
        remat = getattr(self, "checkpoint_blocks", False) and torch.is_grad_enabled() and not collect
        for i, block in enumerate(self.blocks):
            if remat:  # activation checkpointing per block (27B end-to-end distillation)
                x = checkpoint(_block_step, block, x, positions, use_reentrant=False)
            else:
                delta = mixer_delta(block, x, positions)
                if collect:
                    out["inputs"].append(x if i in taps else None)
                    out["mix"].append(delta if i in taps else None)
                x = x + delta
                x = x + mlp_branch(block, x)
            if i in wanted:
                out["hidden"][i] = x
        final = self.norm(x)
        out["final"] = final
        if logits:
            out["logits"] = F.linear(final.to(self.embed.weight.dtype), self.head_weight())
        return out

    def config_dict(self) -> dict:
        return {"teacher": asdict(self.c), "spec": asdict(self.spec)}

    @classmethod
    def from_config_dict(cls, data: Mapping) -> "TokenCore":
        teacher = dict(data["teacher"])
        for key in ("layer_types", "mrope_section"):  # JSON lists back to the frozen tuples
            teacher[key] = tuple(teacher.get(key) or ())
        spec = dict(data["spec"])
        spec["layer_windows"] = tuple(tuple(p) for p in spec.get("layer_windows") or ())
        spec["layer_head_mults"] = tuple(tuple(p) for p in spec.get("layer_head_mults") or ())
        return cls(TeacherConfig(**teacher), CoreSpec(**spec))

    def mixer_parameters(self, layers: Iterable[int] | None = None) -> list[nn.Parameter]:
        """Mixer parameters (everything except the copied embedding/head, norms and MLPs),
        of all layers or of `layers`."""
        out = []
        chosen = range(len(self.blocks)) if layers is None else layers
        for i in chosen:
            block = self.blocks[i]
            if isinstance(block, WindowedHybridLayer):  # the windowed attention stays the teacher's
                out.extend(block.gdn.parameters())
                continue
            for name, p in block.named_parameters():
                if not (name.startswith("mlp.") or name.startswith("norm.")):
                    out.append(p)
        return out


def _repeat_kv(weight: torch.Tensor, kv_heads: int, heads: int, head_dim: int) -> torch.Tensor:
    """[KV*hd, ...] -> [H*hd, ...]: query head h reads key/value head h // (H/KV)."""
    rest = weight.shape[1:]
    return weight.reshape(kv_heads, head_dim, *rest).repeat_interleave(heads // kv_heads, 0).reshape(
        heads * head_dim, *rest)


@torch.no_grad()
def widen_gdn_state(state: dict, model: "TokenCore", noise: float = 0.02, seed: int = 0) -> list[int]:
    """Fit a narrower core's state dict to `model` in place: every Gated DeltaNet block whose
    heads are an integer multiple m of the checkpoint's gets each head repeated m times (q, k,
    v, output gate, decay, beta, conv channels, A_log, dt_bias) and its out-projection columns
    divided by m, so all m copies compute the same head output and the block is unchanged.
    Identical copies would also get identical gradients, so the copies after the first get
    relative noise `noise` on their q/k rows (a small change to the function; measure the
    widened core's val KL before training on it). Returns the widened layers."""
    gen = torch.Generator().manual_seed(seed)
    own = model.state_dict()
    widened = []
    for i, block in enumerate(model.blocks):
        if not isinstance(block, GatedDeltaNetLayer):
            continue
        key = f"blocks.{i}.proj.weight"
        if key not in state or state[key].shape == own[key].shape:
            continue
        h, vh, dk, dv = block.heads, block.value_heads, block.key_dim, block.value_dim
        old_vh = state[f"blocks.{i}.A_log"].shape[0]
        m = vh // old_vh
        # (with value heads sharing a key head the copies would land on other key heads)
        if m * old_vh != vh or h != vh:
            raise ValueError(f"block {i}: only equal key/value heads widen ({old_vh} -> {vh} value, {h} key)")
        oh = h // m

        def rep(t, n, d):  # [n*d, ...] -> each d-row group repeated m times
            return t.reshape(n, d, *t.shape[1:]).repeat_interleave(m, 0).reshape(n * m * d, *t.shape[1:])

        p = state[key]
        q, k, v, gate, a, b = p.split((oh * dk, oh * dk, old_vh * dv, old_vh * dv, old_vh, old_vh))
        q, k = rep(q, oh, dk), rep(k, oh, dk)
        for t in (q, k):  # break the copies' symmetry; copy 0 of each head stays exact
            g = t.view(oh, m, dk, -1)
            g[:, 1:] += noise * g[:, 1:].std() * torch.randn(g[:, 1:].shape, generator=gen, dtype=g.dtype)
        state[key] = torch.cat((q, k, rep(v, old_vh, dv), rep(gate, old_vh, dv), rep(a, old_vh, 1),
                                rep(b, old_vh, 1)))
        ck = f"blocks.{i}.conv_weight"
        if ck in state and state[ck] is not None:
            cq, ckk, cv = state[ck].split((oh * dk, oh * dk, old_vh * dv))
            state[ck] = torch.cat((rep(cq, oh, dk), rep(ckk, oh, dk), rep(cv, old_vh, dv)))
        for name in ("A_log", "dt_bias"):
            state[f"blocks.{i}.{name}"] = state[f"blocks.{i}.{name}"].repeat_interleave(m)
        ok = f"blocks.{i}.out.weight"
        state[ok] = rep(state[ok].t(), old_vh, dv).t().contiguous() / m
        widened.append(i)
    return widened


@torch.no_grad()
def init_gdn_from_attention(block: GatedDeltaNetLayer, c: TeacherConfig, q: torch.Tensor, k: torch.Tensor,
                            v: torch.Tensor, o: torch.Tensor, gate: torch.Tensor | None = None,
                            root: float = 1.0) -> list[str]:
    """INITIALIZE (not convert) a Gated DeltaNet mixer from a softmax attention layer's
    projections, in the spirit of Mamba-in-the-Llama / RADLADS. A delta-rule recurrence is
    not a softmax attention, so this only gives the new mixer a starting point that already
    reads and writes the residual stream like the attention did; the conversion is the
    alignment / distillation training that follows (judge it only by retention after that).

    Value side (exact copies where the widths match, attention heads*head_dim == the GDN
    value width): GDN value head vh takes the attention value dims [vh*dv, (vh+1)*dv) of
    the KV-repeated layout, i.e. query head h = vh*dv // head_dim at offset o; the output
    projection columns and the output-gate rows (Qwen3.5) follow the same dims.
    Key side: GDN key head kh serves value heads [kh*r, (kh+1)*r) (r = value/key heads);
    its query (key) row block is the MEAN over those served (head h, offset o) slices of
    the attention's query (KV-repeated key) rows [o, o+dk) - every query head contributes
    (Qwen3.5-0.8B: 1 slice per key head = the head halves; Qwen3.8-27B: 3 slices from the
    24 query heads per key head; nothing is dropped). Without a clean value map, q/k fall
    back to the first rows and v/o/gate keep their fresh initialization. The short
    convolution starts as the identity."""
    done = []
    hk = block.heads * block.key_dim
    vw, dv, dk = block.value_width, block.value_dim, block.key_dim
    q_all = q.reshape(c.heads * c.head_dim, -1)
    k_all = _repeat_kv(k, c.kv_heads, c.heads, c.head_dim)
    v_all = _repeat_kv(v, c.kv_heads, c.heads, c.head_dim)
    proj = block.proj.weight
    mapped = v_all.shape[0] == vw and dk <= c.head_dim
    gqa = (block.heads == block.value_heads == c.kv_heads < c.heads and dk == dv == c.head_dim
           and vw == c.kv_heads * c.head_dim)
    if gqa:  # one GDN head per KV head: k, v exact; q = mean of the group's query heads
        group = c.heads // c.kv_heads
        proj[:hk].copy_(q.reshape(c.kv_heads, group, c.head_dim, -1).mean(1).reshape(hk, -1))
        proj[hk:2 * hk].copy_(k.reshape(hk, -1))
        proj[2 * hk:2 * hk + vw].copy_(v.reshape(vw, -1))
        done += ["q(mean of group)", "k", "v"]
        if o.shape[1] == c.heads * c.head_dim:  # the group's output columns summed (one head per group)
            block.out.weight.copy_(o.reshape(o.shape[0], c.kv_heads, group, c.head_dim).sum(2)
                                   .reshape(o.shape[0], vw) * root)
            done.append("o(group sum)")
        if gate is not None:
            proj[2 * hk + vw:2 * hk + 2 * vw].copy_(gate.reshape(c.kv_heads, group, c.head_dim, -1).mean(1)
                                                    .reshape(vw, -1))
            done.append("gate(mean of group)")
        if block.conv_weight is not None:
            block.conv_weight.zero_()
            block.conv_weight[:, -1] = 1.0
            done.append("conv=identity")
        return done
    if mapped:
        ratio = block.value_heads // block.heads
        for kh in range(block.heads):
            qs, ks = [], []
            for vh in range(kh * ratio, (kh + 1) * ratio):
                h, off = divmod(vh * dv, c.head_dim)
                off = min(off, c.head_dim - dk)
                rows = slice(h * c.head_dim + off, h * c.head_dim + off + dk)
                qs.append(q_all[rows])
                ks.append(k_all[rows])
            proj[kh * dk:(kh + 1) * dk].copy_(torch.stack(qs).mean(0))
            proj[hk + kh * dk:hk + (kh + 1) * dk].copy_(torch.stack(ks).mean(0))
        done += ["q(mean of served heads)", "k(mean of served heads)"]
        proj[2 * hk:2 * hk + vw].copy_(v_all)
        done.append("v")
    else:
        n = min(q_all.shape[0], hk)
        proj[:n].copy_(q_all[:n])
        proj[hk:hk + n].copy_(k_all[:n])
        done += ["q", "k"] if n == hk else [f"q[:{n}]", f"k[:{n}]"]
    if o.shape[1] == vw:
        block.out.weight.copy_(o * root)
        done.append("o")
    if gate is not None and gate.shape[0] == vw:
        proj[2 * hk + vw:2 * hk + 2 * vw].copy_(gate)
        done.append("gate")
    if block.conv_weight is not None:
        block.conv_weight.zero_()
        block.conv_weight[:, -1] = 1.0
        done.append("conv=identity")
    return done


@torch.no_grad()
def transplant_from_teacher(student: TokenCore, teacher: LlamaTeacher, *, mixer_init: bool = True) -> dict:
    """Llama-family teacher: copy embeddings, norms, MLPs (exact) and initialize every
    mixer from the teacher's attention projections. Returns a summary."""
    c = teacher.c
    if student.c.width != c.width or student.c.layers != c.layers:
        raise ValueError("student/teacher shapes differ")
    root = math.sqrt(c.layers)  # undo the PB residual scale 1/sqrt(L)
    student.embed.weight.copy_(teacher.embed.weight)
    student.norm.weight.copy_(teacher.norm.weight)
    if student.head is not None:
        student.head.weight.copy_(teacher.head.weight)
    copied = {"exact": ["embed", "norm", "head" if student.head is not None else "head(tied)"],
              "mixer_init": []}
    for i, (block, t) in enumerate(zip(student.blocks, teacher.layers)):
        block.norm.weight.copy_(t.input_norm.weight)
        block.mlp.norm.weight.copy_(t.post_norm.weight)
        block.mlp.gate_up.weight.copy_(torch.cat((t.gate.weight, t.up.weight), 0))
        block.mlp.down.weight.copy_(t.down.weight * root)
        kind = student.spec.mixers[i]
        if kind == "attention":
            for name in ("q", "k", "v"):
                getattr(block, name).weight.copy_(getattr(t, name).weight)
                if c.qkv_bias:
                    getattr(block, name).bias.copy_(getattr(t, name).bias)
            block.out.weight.copy_(t.o.weight * root)
            copied["mixer_init"].append("attention(exact)")
            continue
        if not mixer_init:
            copied["mixer_init"].append(f"{kind}(random)")
            continue
        block.out.weight.copy_(t.o.weight * root)
        if kind == "gated_deltanet":
            done = init_gdn_from_attention(block, c, t.q.weight, t.k.weight, t.v.weight, t.o.weight,
                                           root=root)
            copied["mixer_init"].append("gated_deltanet(" + ",".join(done) + ")")
        elif kind == "mingru":
            width = c.width
            v = _repeat_kv(t.v.weight, c.kv_heads, c.heads, c.head_dim)
            if v.shape[0] != width:
                raise ValueError("mingru proposal init needs heads*head_dim == width")
            block.gates.weight[width:].copy_(v)
            if c.qkv_bias:
                block.gates.bias[width:].copy_(_repeat_kv(t.v.bias, c.kv_heads, c.heads, c.head_dim))
            else:
                block.gates.bias[width:].zero_()
            copied["mixer_init"].append("mingru(v,o)")
    return copied


# --------------------------------------------------------------------------- Qwen3.5 / Qwen3.8 hybrids

def _qwen_prefix(reader: "SafeTensors") -> str:
    for prefix in ("model.language_model.", "model."):
        if prefix + "embed_tokens.weight" in reader:
            return prefix
    raise KeyError("no embed_tokens.weight in the checkpoint")


@torch.no_grad()
def _copy_qwen_layer(block: nn.Module, i: int, kind: str, mixer: str, c: TeacherConfig, spec: CoreSpec, get, has,
                     norm_weight, root: float, mixer_init: bool, summary: dict) -> None:
    """Checkpoint layer i -> PB block (the Qwen3.5/3.6/3.8 mapping of load_hf_core); get/has
    read checkpoint tensors by their name below the language-model prefix."""
    base = f"layers.{i}."
    block.norm.weight.copy_(norm_weight(base + "input_layernorm.weight"))
    block.mlp.norm.weight.copy_(norm_weight(base + "post_attention_layernorm.weight"))
    if c.num_experts:
        moe = block.mlp
        moe.router.weight.copy_(get(base + "mlp.gate.weight"))
        if has(base + "mlp.experts.gate_up_proj"):  # the Hub's fused layout
            moe.gate_up.copy_(get(base + "mlp.experts.gate_up_proj"))
            moe.down.copy_(get(base + "mlp.experts.down_proj") * root)
        else:  # per-expert tensors (transformers' save_pretrained)
            for e in range(c.num_experts):
                ex = base + f"mlp.experts.{e}."
                moe.gate_up[e].copy_(torch.cat((get(ex + "gate_proj.weight"), get(ex + "up_proj.weight")), 0))
                moe.down[e].copy_(get(ex + "down_proj.weight") * root)
        if moe.shared_gate_up is not None:
            moe.shared_gate_up.weight.copy_(torch.cat((get(base + "mlp.shared_expert.gate_proj.weight"),
                                                       get(base + "mlp.shared_expert.up_proj.weight")), 0))
            moe.shared_down.weight.copy_(get(base + "mlp.shared_expert.down_proj.weight") * root)
            moe.shared_gate.weight.copy_(get(base + "mlp.shared_expert_gate.weight"))
    else:
        block.mlp.gate_up.weight.copy_(torch.cat((get(base + "mlp.gate_proj.weight"),
                                                  get(base + "mlp.up_proj.weight")), 0))
        block.mlp.down.weight.copy_(get(base + "mlp.down_proj.weight") * root)
    if kind == LINEAR:
        if mixer != "gated_deltanet":
            raise ValueError(f"layer {i} is a Gated DeltaNet layer; spec must keep gated_deltanet")
        la = base + "linear_attn."
        block.proj.weight.copy_(torch.cat((get(la + "in_proj_qkv.weight"), get(la + "in_proj_z.weight"),
                                           get(la + "in_proj_a.weight"), get(la + "in_proj_b.weight")), 0))
        block.conv_weight.copy_(get(la + "conv1d.weight").reshape(block.channels, -1))
        block.A_log.copy_(get(la + "A_log"))
        block.dt_bias.copy_(get(la + "dt_bias"))
        block.o_norm.copy_(get(la + "norm.weight"))
        block.out.weight.copy_(get(la + "out_proj.weight") * root)
        summary["layers"].append("gated_deltanet(exact)")
        return
    sa = base + "self_attn."
    q, k, v, o = (get(sa + n + "_proj.weight") for n in ("q", "k", "v", "o"))
    if mixer == "swa_gdn":
        attn = block.attn
        attn.q.weight.copy_(q)
        attn.k.weight.copy_(k)
        attn.v.weight.copy_(v)
        attn.out.weight.copy_(o * root)
        if c.qk_norm:
            attn.q_norm.weight.copy_(norm_weight(sa + "q_norm.weight"))
            attn.k_norm.weight.copy_(norm_weight(sa + "k_norm.weight"))
        done = []
        if mixer_init:
            gate = None
            qq = q
            if c.attn_output_gate:
                both = q.reshape(c.heads, 2, c.head_dim, -1)
                qq, gate = (both[:, 0].reshape(c.heads * c.head_dim, -1),
                            both[:, 1].reshape(c.heads * c.head_dim, -1))
            done = init_gdn_from_attention(block.gdn, c, qq, k, v, o, gate, root)
        block.gdn.out.weight.zero_()
        summary["layers"].append(f"swa_gdn(window {spec.window(i)} exact; gdn: "
                                 + ",".join(done) + ", out=0)")
        return
    if mixer == "attention":
        block.q.weight.copy_(q)
        block.k.weight.copy_(k)
        block.v.weight.copy_(v)
        block.out.weight.copy_(o * root)
        if c.qk_norm:
            block.q_norm.weight.copy_(norm_weight(sa + "q_norm.weight"))
            block.k_norm.weight.copy_(norm_weight(sa + "k_norm.weight"))
        summary["layers"].append("attention(exact)")
    elif mixer == "gated_deltanet":
        gate = None
        if c.attn_output_gate:
            both = q.reshape(c.heads, 2, c.head_dim, -1)
            q, gate = both[:, 0].reshape(c.heads * c.head_dim, -1), both[:, 1].reshape(c.heads * c.head_dim, -1)
        done = init_gdn_from_attention(block, c, q, k, v, o, gate, root) if mixer_init else []
        summary["layers"].append("gated_deltanet(from attention: " + ",".join(done) + ")")
    else:
        raise ValueError(f"mixer {mixer!r} is not supported for a hybrid teacher's attention layer")


@torch.no_grad()
def load_hf_core(path: str | Path, spec: CoreSpec | None = None, dtype: torch.dtype = torch.float32,
                 device="cpu", *, mixer_init: bool = True) -> tuple["TokenCore", dict]:
    """A Hugging Face checkpoint -> TokenCore. Llama/Qwen2/Mistral go through LlamaTeacher;
    a Qwen3.5 / Qwen3.8 hybrid is mapped directly: its Gated DeltaNet layers are copied
    exactly into PB GatedDeltaNetLayers, its gated attention layers into
    TransplantAttentionLayers (spec 'attention') or used to initialize new Gated DeltaNet
    mixers (spec 'gated_deltanet' on a full_attention layer). spec=None: the exact teacher
    (CoreSpec.exact). Zero-centered norms (1 + w) become plain RMSNorm weights; the PB
    residual scale 1/sqrt(L) is folded into every out/down projection. Vision tower and
    the MTP head are not loaded."""
    path = Path(path).expanduser()
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    c = TeacherConfig.from_hf(config)
    reader = SafeTensors(path)
    # Llama/Mistral without biases take the direct path below in any dtype but fp32 (a 33B
    # Llama teacher never exists in fp32 either); LlamaTeacher stays the fp32 reference path.
    direct = not c.qkv_bias and dtype != torch.float32
    if not c.model_type.startswith("qwen3_5") and not direct:
        teacher = LlamaTeacher.from_pretrained(path, dtype=torch.float32)
        spec = spec or CoreSpec.exact(c)
        core = TokenCore(c, spec)
        summary = transplant_from_teacher(core, teacher, mixer_init=mixer_init)
        return core.to(device=device, dtype=dtype), summary
    prefix = _qwen_prefix(reader)
    # The checkpoint decides tying (Qwen3.5-27B's config leaves tie_word_embeddings null).
    c = TeacherConfig(**{**asdict(c), "tie_embeddings": "lm_head.weight" not in reader})
    spec = spec or CoreSpec.exact(c)
    kinds = c.types()
    fresh = [i for i, (kind, mixer) in enumerate(zip(kinds, spec.mixers))
             if kind == FULL and mixer not in ("attention",)]
    # Built on the meta device and materialized straight on `device` in `dtype` (a 27B
    # teacher never exists in fp32): every tensor of an imported layer comes from the
    # checkpoint. Only the mixers initialized from attention are built for real (one layer
    # at a time, fresh initialization, then moved), so their uncopied projections keep a
    # proper initialization.
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device("meta"):
            core = TokenCore(c, spec)
    finally:
        torch.set_default_dtype(previous)
    template = core
    core = core.to_empty(device=device)
    for i in fresh:
        with torch.device("cpu"):
            core.blocks[i] = TokenCore._fresh_block(c, spec, i).to(device=device, dtype=dtype)
    del template

    def get(name):
        return reader.get(prefix + name, torch.float32).to(device)

    def norm_weight(name):
        w = get(name)
        return 1.0 + w if c.zero_centered_norm else w

    root = math.sqrt(c.layers)
    core.embed.weight.copy_(get("embed_tokens.weight"))
    core.norm.weight.copy_(norm_weight("norm.weight"))
    if core.head is not None:
        core.head.weight.copy_(reader.get("lm_head.weight", torch.float32).to(device))
    summary = {"exact": ["embed", "norm", "head" if core.head is not None else "head(tied)"], "layers": []}
    for i, (block, kind) in enumerate(zip(core.blocks, c.types())):
        _copy_qwen_layer(block, i, kind, spec.mixers[i], c, spec, get, lambda name: prefix + name in reader,
                         norm_weight, root, mixer_init, summary)
    return core.to(device=device, dtype=dtype), summary


@torch.no_grad()
def load_hf_block(path: str | Path, index: int, dtype: torch.dtype = torch.float32, device="cpu") -> nn.Module:
    """Layer `index` of a Qwen3.5/3.6/3.8 checkpoint alone, as the exact PB block (attention
    kept), with residual scale 1 and no folded sqrt(L): block(x) = x + mixer + MLP exactly as
    the teacher's layer. For parity checks of teachers too big to hold twice (run the PB model
    one block at a time next to the reference's layers)."""
    path = Path(path).expanduser()
    c = TeacherConfig.from_hf(json.loads((path / "config.json").read_text(encoding="utf-8")))
    if not c.model_type.startswith("qwen3_5"):
        raise ValueError("load_hf_block maps Qwen3.5/3.6/3.8 checkpoints")
    reader = SafeTensors(path)
    prefix = _qwen_prefix(reader)
    kind = c.types()[index]
    single = TeacherConfig(**{**asdict(c), "layers": 1, "layer_types": (kind,)})
    spec = CoreSpec.exact(single)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device("meta"):
            block = TokenCore(single, spec).blocks[0]
    finally:
        torch.set_default_dtype(previous)
    block = block.to_empty(device=device)

    def name_of(name):
        return prefix + name.replace("layers.0.", f"layers.{index}.", 1)

    def get(name):
        return reader.get(name_of(name), torch.float32).to(device)

    def norm_weight(name):
        w = get(name)
        return 1.0 + w if c.zero_centered_norm else w

    _copy_qwen_layer(block, 0, kind, spec.mixers[0], single, spec, get, lambda name: name_of(name) in reader,
                     norm_weight, 1.0, False, {"layers": []})
    return block.to(device=device, dtype=dtype)


def _clone_state(state: list) -> list:
    out = []
    for st in state:
        if isinstance(st, HybridState):
            out.append(HybridState(AttentionCache(st.cache.keys, st.cache.values, st.cache.position, st.cache.sinks),
                                   list(st.gdn)))
        elif isinstance(st, AttentionCache):
            out.append(AttentionCache(st.keys, st.values, st.position, st.sinks))  # tensors are never mutated
        else:
            out.append(list(st))
    return out


@torch.no_grad()
def greedy_decode(core: "TokenCore", ids: torch.Tensor, new_tokens: int) -> torch.Tensor:
    """Plain greedy decoding through stream() (batch 1): the reference for speculative_greedy."""
    out, state = core.stream(ids)
    token = out["logits"][:, -1:].argmax(-1)
    produced = [token]
    for _ in range(new_tokens - 1):
        out, state = core.stream(token, state)
        token = out["logits"][:, -1:].argmax(-1)
        produced.append(token)
    return torch.cat(produced, 1)


@torch.no_grad()
def speculative_greedy(target: "TokenCore", draft: "TokenCore", ids: torch.Tensor, new_tokens: int,
                       k: int = 4) -> tuple[torch.Tensor, dict]:
    """Greedy speculative decoding with recurrent (or hybrid) cores sharing a tokenizer
    (batch 1). The draft proposes k tokens step by step; the target scores them in ONE
    chunked pass (its weights are read once for k+1 tokens). The longest agreeing prefix
    plus the target's own next token are kept. Recurrent state cannot be "un-appended",
    so on a rejection both models re-stream the accepted tokens from the states saved
    before the block (a second, shorter target pass). Returns the target's greedy
    continuation (up to float summation order) and acceptance statistics."""
    out_t, state_t = target.stream(ids)
    out_d, state_d = draft.stream(ids)
    token = out_t["logits"][:, -1:].argmax(-1)
    produced = [token]
    proposed = accepted = target_passes = 0
    while sum(t.shape[1] for t in produced) < new_tokens:
        saved_t, saved_d = _clone_state(state_t), _clone_state(state_d)
        block, d_state, d_token = [], state_d, token
        for _ in range(k):  # draft proposals
            o, d_state = draft.stream(d_token, d_state)
            d_token = o["logits"][:, -1:].argmax(-1)
            block.append(d_token)
        chunk = torch.cat([token] + block, 1)  # target scores token + k proposals
        o, full_state = target.stream(chunk, saved_t)
        target_passes += 1
        greedy = o["logits"].argmax(-1)  # greedy[:, j] follows chunk[:, :j+1]
        m = 0
        while m < k and int(greedy[0, m]) == int(block[m][0, 0]):
            m += 1
        proposed += k
        accepted += m
        keep = torch.cat([token] + block[:m], 1)
        bonus = greedy[:, m:m + 1]
        if m == k:
            state_t = full_state
            _, state_d = draft.stream(block[-1], d_state)  # the draft has not consumed its last proposal
        else:
            _, state_t = target.stream(keep, saved_t)
            target_passes += 1
            _, state_d = draft.stream(keep, saved_d)
        produced.extend(block[:m])
        produced.append(bonus)
        token = bonus
    result = torch.cat(produced, 1)[:, :new_tokens]
    return result, {"proposed": proposed, "accepted": accepted, "acceptance": accepted / max(1, proposed),
                    "tokens_per_target_pass": result.shape[1] / max(1, target_passes),
                    "target_passes": target_passes}


def gdn_hf_tensors(prefix: str, *, proj: torch.Tensor, conv: torch.Tensor, A_log: torch.Tensor,
                   dt_bias: torch.Tensor, o_norm: torch.Tensor, out: torch.Tensor, heads: int, key_dim: int,
                   value_heads: int, value_dim: int, root: float) -> dict[str, torch.Tensor]:
    """A PB GatedDeltaNetLayer's tensors under HF `linear_attn` names below `prefix`
    (e.g. 'model.language_model.layers.3.gdn.'): the proj's [qkv|z|a|b] rows split, conv
    [C,1,K], o_norm as norm.weight, the out projection with the PB residual scale
    (1/sqrt(L), stored folded into `out`) undone by `root` = sqrt(L). Values keep their
    dtype (fp32 math on the scaled matrix); shared by export_hf_qwen35 and the PB-27 slim
    export (scripts/transplant/export_pb27_slim.py)."""
    hk, vw, hv = 2 * heads * key_dim, value_heads * value_dim, value_heads
    if tuple(proj.shape[:1]) != (hk + 2 * vw + 2 * hv,):
        raise ValueError(f"{prefix}: proj has {proj.shape[0]} rows, expected {hk + 2 * vw + 2 * hv}")
    if tuple(conv.shape[:1]) != (hk + vw,) or A_log.numel() != hv or o_norm.numel() != value_dim:
        raise ValueError(f"{prefix}: conv/A_log/o_norm shapes disagree with the GDN dims")
    return {
        prefix + "in_proj_qkv.weight": proj[:hk + vw],
        prefix + "in_proj_z.weight": proj[hk + vw:hk + 2 * vw],
        prefix + "in_proj_a.weight": proj[hk + 2 * vw:hk + 2 * vw + hv],
        prefix + "in_proj_b.weight": proj[hk + 2 * vw + hv:],
        prefix + "conv1d.weight": conv[:, None, :],
        prefix + "A_log": A_log,
        prefix + "dt_bias": dt_bias,
        prefix + "norm.weight": o_norm,
        prefix + "out_proj.weight": out.float() / root,
    }


def pb27_config(config: dict, hybrid: Sequence[int], window: int | None, sinks: int,
                gdn: Mapping | None = None) -> dict:
    """Mark a Qwen3.5 config.json dict for the vLLM out-of-tree PB27 model: architectures
    PB27ForConditionalGeneration, or PB27MoeForConditionalGeneration for a qwen3_5_moe
    teacher (model_type unchanged), and pb_hybrid_layers / pb_attention_window /
    pb_attention_sinks on both the top level and text_config (docs/PB27_VLLM_PLUGIN.md
    section 4). A Llama teacher (no linear attention of its own) becomes
    PB27LlamaForCausalLM and also gets the gdn shape `gdn` (heads / key_dim /
    value_heads / value_dim / conv) as Qwen3.5 linear_* fields plus mamba_ssm_dtype.
    Returns `config` (edited in place)."""
    text = config.get("text_config", config)
    kind = str(text.get("model_type", config.get("model_type", "")))
    if kind == "llama":
        if not gdn:
            raise ValueError("a Llama teacher needs the gdn shape")
        config["architectures"] = ["PB27LlamaForCausalLM"]
        config.update(linear_num_key_heads=int(gdn["heads"]), linear_key_head_dim=int(gdn["key_dim"]),
                      linear_num_value_heads=int(gdn["value_heads"]), linear_value_head_dim=int(gdn["value_dim"]),
                      linear_conv_kernel_dim=int(gdn["conv"]), mamba_ssm_dtype="float32")
    elif kind.startswith("qwen4_exp"):
        config["architectures"] = ["PB27Qwen4ExpForConditionalGeneration"]
    else:
        moe = "moe" in kind
        config["architectures"] = ["PB27MoeForConditionalGeneration" if moe else "PB27ForConditionalGeneration"]
    for target in {id(config): config, id(text): text}.values():
        target["pb_hybrid_layers"] = [int(i) for i in hybrid]
        target["pb_attention_window"] = window
        target["pb_attention_sinks"] = int(sinks)
        if gdn:
            target.setdefault("linear_num_key_heads", int(gdn["heads"]))
            target.setdefault("linear_key_head_dim", int(gdn["key_dim"]))
            target.setdefault("linear_num_value_heads", int(gdn["value_heads"]))
            target.setdefault("linear_value_head_dim", int(gdn["value_dim"]))
            target.setdefault("linear_conv_kernel_dim", int(gdn["conv"]))
            target.setdefault("mamba_ssm_dtype", "float32")
    return config


@torch.no_grad()
def export_hf_qwen35(core: "TokenCore", teacher_path: str | Path, out_dir: str | Path,
                     dtype: torch.dtype = torch.bfloat16, shard_bytes: int = 4 << 30) -> dict:
    """Write a TokenCore of a Qwen3.5/3.8 teacher back as a Hugging Face qwen3_5 checkpoint
    (the inverse of load_hf_core), so the converted model loads in transformers and in
    serving stacks that run the qwen3_5 architecture. Converted layers become
    'linear_attention' layers in layer_types (a fully recurrent core has no full_attention
    layer left). The teacher's vision tower and MTP weights are copied through unchanged;
    tokenizer and processor files are copied. Only Qwen-layout Gated DeltaNet and kept
    gated-attention blocks are exportable, plus PB-27's WindowedHybridLayers (a --shared
    swa_gdn conversion): those stay 'full_attention' in layer_types, their windowed teacher
    attention goes under the stock self_attn names and their Gated DeltaNet branch under
    `layers.i.gdn.` (linear_attn naming); config.json then names the vLLM out-of-tree
    architecture PB27ForConditionalGeneration and lists pb_hybrid_layers /
    pb_attention_window / pb_attention_sinks (docs/PB27_VLLM_PLUGIN.md section 4)."""
    import shutil

    src, out = Path(teacher_path).expanduser(), Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    c = core.c
    if not c.model_type.startswith("qwen3_5"):
        raise ValueError("export_hf_qwen35 needs a Qwen3.5/3.8 TokenCore")
    reader = SafeTensors(src)
    prefix = _qwen_prefix(reader)
    root = math.sqrt(c.layers)
    tensors: dict[str, torch.Tensor] = {}

    def put(name, value, full=False):
        tensors[name if full else prefix + name] = value.detach().to("cpu", dtype).contiguous()

    def norm(value):
        return value.float() - 1.0 if c.zero_centered_norm else value.float()

    def put_gdn(i, prefix_name, block):
        """A PB GatedDeltaNetLayer under `prefix_name` (HF linear_attn naming: the proj's
        [qkv|z|a|b] rows split, conv [C,1,K], out projection with the residual scale undone)."""
        if (block.heads, block.key_dim, block.value_heads, block.value_dim) != (
                c.linear_key_heads, c.linear_key_dim, c.linear_value_heads, c.linear_value_dim):
            raise ValueError(f"layer {i}: Gated DeltaNet shape differs from the teacher's linear attention")
        hf = gdn_hf_tensors(prefix_name, proj=block.proj.weight, conv=block.conv_weight, A_log=block.A_log,
                            dt_bias=block.dt_bias, o_norm=block.o_norm, out=block.out.weight,
                            heads=block.heads, key_dim=block.key_dim, value_heads=block.value_heads,
                            value_dim=block.value_dim, root=root)
        for name, value in hf.items():
            put(name, value)

    def put_attention(prefix_name, block):
        """A PB TransplantAttentionLayer's projections under `prefix_name` (HF self_attn naming)."""
        put(prefix_name + "q_proj.weight", block.q.weight)
        put(prefix_name + "k_proj.weight", block.k.weight)
        put(prefix_name + "v_proj.weight", block.v.weight)
        put(prefix_name + "o_proj.weight", block.out.weight.float() / root)
        if block.q_norm is not None:
            put(prefix_name + "q_norm.weight", norm(block.q_norm.weight))
            put(prefix_name + "k_norm.weight", norm(block.k_norm.weight))

    put("embed_tokens.weight", core.embed.weight)
    put("norm.weight", norm(core.norm.weight))
    if core.head is not None:
        put("lm_head.weight", core.head.weight, full=True)
    types, hybrid = [], []
    for i, block in enumerate(core.blocks):
        base = f"layers.{i}."
        put(base + "input_layernorm.weight", norm(block.norm.weight))
        put(base + "post_attention_layernorm.weight", norm(block.mlp.norm.weight))
        if isinstance(block.mlp, SparseMoE):
            moe = block.mlp
            put(base + "mlp.gate.weight", moe.router.weight)
            put(base + "mlp.experts.gate_up_proj", moe.gate_up)
            put(base + "mlp.experts.down_proj", moe.down.float() / root)
            if moe.shared_gate_up is not None:
                gate, up = moe.shared_gate_up.weight.chunk(2, dim=0)
                put(base + "mlp.shared_expert.gate_proj.weight", gate)
                put(base + "mlp.shared_expert.up_proj.weight", up)
                put(base + "mlp.shared_expert.down_proj.weight", moe.shared_down.weight.float() / root)
                put(base + "mlp.shared_expert_gate.weight", moe.shared_gate.weight)
        else:
            gate, up = block.mlp.gate_up.weight.chunk(2, dim=0)
            put(base + "mlp.gate_proj.weight", gate)
            put(base + "mlp.up_proj.weight", up)
            put(base + "mlp.down_proj.weight", block.mlp.down.weight.float() / root)
        if isinstance(block, GatedDeltaNetLayer):
            put_gdn(i, base + "linear_attn.", block)
            types.append(LINEAR)
        elif isinstance(block, WindowedHybridLayer):
            # PB-27 hybrid (docs/PB27_VLLM_PLUGIN.md section 4): the teacher's own attention
            # under the stock self_attn names (the plugin restricts it to the window) plus the
            # far-context Gated DeltaNet under `gdn.` with the linear_attn naming.
            put_attention(base + "self_attn.", block.attn)
            put_gdn(i, base + "gdn.", block.gdn)
            types.append(FULL)
            hybrid.append(i)
        elif isinstance(block, TransplantAttentionLayer):
            put_attention(base + "self_attn.", block)
            types.append(FULL)
        else:
            raise ValueError(f"layer {i}: {type(block).__name__} has no qwen3_5 equivalent")
    for name in reader.keys():  # vision tower and MTP head, unchanged
        if name.startswith(("model.visual.", "mtp.")):
            tensors[name] = reader.get(name).to(dtype) if reader.get(name).is_floating_point() else reader.get(name)
    shards, current, size = [], {}, 0
    for name in sorted(tensors):
        nbytes = tensors[name].numel() * tensors[name].element_size()
        if current and size + nbytes > shard_bytes:
            shards.append(current)
            current, size = {}, 0
        current[name] = tensors[name]
        size += nbytes
    shards.append(current)
    weight_map = {}
    for k, shard in enumerate(shards):
        fname = f"model-{k + 1:05d}-of-{len(shards):05d}.safetensors"
        write_safetensors(out / fname, shard)
        weight_map.update({name: fname for name in shard})
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
         "weight_map": weight_map}, indent=1), encoding="utf-8")
    config = json.loads((src / "config.json").read_text(encoding="utf-8"))
    text = config.get("text_config", config)
    text["layer_types"] = types
    if all(t == LINEAR for t in types):
        text["full_attention_interval"] = c.layers + 1
    if hybrid:
        # Only the vLLM out-of-tree PB27 model reads these (model_type stays qwen3_5*, so the
        # stock config class parses the rest); the hybrid layers are listed as full_attention
        # above and the plugin adds their gdn branch and sliding window from these fields.
        pb27_config(config, hybrid, core.spec.attention_window, core.spec.attention_sinks)
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    for extra in src.iterdir():
        if extra.suffix in (".json", ".txt", ".jinja", ".model") and extra.name not in (
                "config.json", "model.safetensors.index.json"):
            shutil.copy(extra, out / extra.name)
    return {"layers": types, "hybrid": hybrid, "shards": len(shards), "tensors": len(tensors)}


@torch.no_grad()
def load_pb27_gdn_shard(core: "TokenCore", shard_path: str | Path) -> list[int]:
    """Load a PB-27 `pb27-gdn-00001-of-00001.safetensors` shard directly into a `TokenCore`'s
    `WindowedHybridLayer.gdn` branches (undoing the HF `1/sqrt(L)` division by multiplying `out_proj` by `sqrt(L)`)."""
    reader = SafeTensors(shard_path)
    root = math.sqrt(core.c.layers)
    loaded = []
    for i, block in enumerate(core.blocks):
        if not isinstance(block, WindowedHybridLayer):
            continue
        p = f"model.language_model.layers.{i}.gdn."
        if p + "in_proj_qkv.weight" not in reader:
            p = f"model.layers.{i}.gdn."
        if p + "in_proj_qkv.weight" not in reader:
            continue
        gdn = block.gdn
        dev, dt = gdn.proj.weight.device, gdn.proj.weight.dtype
        qkv = reader.get(p + "in_proj_qkv.weight", torch.float32).to(dev)
        z = reader.get(p + "in_proj_z.weight", torch.float32).to(dev)
        a = reader.get(p + "in_proj_a.weight", torch.float32).to(dev)
        b = reader.get(p + "in_proj_b.weight", torch.float32).to(dev)
        gdn.proj.weight.copy_(torch.cat((qkv, z, a, b), dim=0).to(dt))
        gdn.conv_weight.copy_(reader.get(p + "conv1d.weight", torch.float32).to(dev).reshape(gdn.channels, -1).to(dt))
        gdn.A_log.copy_(reader.get(p + "A_log", torch.float32).to(dev).to(gdn.A_log.dtype))
        gdn.dt_bias.copy_(reader.get(p + "dt_bias", torch.float32).to(dev).to(gdn.dt_bias.dtype))
        gdn.o_norm.copy_(reader.get(p + "norm.weight", torch.float32).to(dev).to(gdn.o_norm.dtype))
        gdn.out.weight.copy_((reader.get(p + "out_proj.weight", torch.float32).to(dev) * root).to(dt))
        loaded.append(i)
    return loaded


def attach_spatial_mlp(core: "TokenCore", layers: Sequence[int], hidden: int = 7552,
                       dtype: torch.dtype = torch.bfloat16) -> list[nn.Parameter]:
    """Attach zero-output parallel SwiGLU `spatial_mlp` branches (`+1.856B` params at `hidden=7552` across 16 layers)
    to the specified decoder layers of `core`, freezing all existing parameters and returning the trainable parameters."""
    from .scan import SwiGLU
    for p in core.parameters():
        p.requires_grad_(False)
    params: list[nn.Parameter] = []
    dev = core.embed.weight.device
    for i in layers:
        block = core.blocks[int(i)]
        s_mlp = SwiGLU(core.c.width, hidden, zero_output=True, norm="rms", norm_eps=core.c.norm_eps).to(device=dev, dtype=dtype)
        s_mlp.norm = nn.Identity()
        s_mlp.enabled = True
        with torch.no_grad():
            s_mlp.down.weight.zero_()
        block.spatial_mlp = s_mlp
        for p in s_mlp.parameters():
            p.requires_grad_(True)
            params.append(p)
    return params


import contextlib as _contextlib


@_contextlib.contextmanager
def spatial_bypassed(core: "TokenCore"):
    """Temporarily disable attached `spatial_mlp` branches so `core` computes the exact base `pb27c` outputs."""
    mods = [getattr(b, "spatial_mlp", None) for b in core.blocks if getattr(b, "spatial_mlp", None) is not None]
    prev = [getattr(m, "enabled", True) for m in mods]
    for m in mods:
        m.enabled = False
    try:
        yield
    finally:
        for m, p in zip(mods, prev):
            m.enabled = p


@torch.no_grad()
def export_pb27_spatial_shard(core: "TokenCore", layers: Sequence[int], out_path: str | Path,
                              dtype: torch.dtype = torch.bfloat16) -> dict:
    """Export the trained `spatial_mlp` branches of `core` to an HF safetensors shard
    (`model.language_model.layers.{i}.spatial_mlp.{gate_proj,up_proj,down_proj}.weight`)."""
    root = math.sqrt(core.c.layers)
    tensors: dict[str, torch.Tensor] = {}
    for i in layers:
        block = core.blocks[int(i)]
        s_mlp = getattr(block, "spatial_mlp", None)
        if s_mlp is None:
            continue
        base = f"model.language_model.layers.{int(i)}.spatial_mlp."
        gate, up = s_mlp.gate_up.weight.detach().float().chunk(2, dim=0)
        down = s_mlp.down.weight.detach().float() / root
        tensors[base + "gate_proj.weight"] = gate.to("cpu", dtype).contiguous()
        tensors[base + "up_proj.weight"] = up.to("cpu", dtype).contiguous()
        tensors[base + "down_proj.weight"] = down.to("cpu", dtype).contiguous()
    out_p = Path(out_path).expanduser()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    write_safetensors(out_p, tensors)
    return {"path": str(out_p), "layers": [int(i) for i in layers], "tensors": len(tensors),
            "params": sum(t.numel() for t in tensors.values())}


def copy_exact_layers(dst: "TokenCore", src: "TokenCore") -> list[int]:
    """Copy every block whose mixer type matches between two cores of the same teacher
    (e.g. a hybrid teacher's Gated DeltaNet layers into the fully recurrent student), plus
    embeddings, final norm and head. Returns the copied layer indices."""
    with torch.no_grad():
        dst.embed.weight.copy_(src.embed.weight)
        dst.norm.weight.copy_(src.norm.weight)
        if dst.head is not None and src.head is not None:
            dst.head.weight.copy_(src.head.weight)
    out = []
    for i, (a, b) in enumerate(zip(dst.blocks, src.blocks)):
        if type(a) is type(b):
            a.load_state_dict(b.state_dict(), strict=True)
            out.append(i)
        else:
            a.norm.load_state_dict(b.norm.state_dict())
            a.mlp.load_state_dict(b.mlp.state_dict())
    return out


# --------------------------------------------------------------------------- losses

def normalized_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """mean ||pred - target||^2 / mean ||target||^2 over (masked) rows; fp32."""
    pred, target = pred.float(), target.float()
    err = (pred - target).square().sum(-1)
    ref = target.square().sum(-1)
    if mask is not None:
        m = mask.float()
        return (err * m).sum() / (ref * m).sum().clamp_min(1e-6)
    return err.sum() / ref.sum().clamp_min(1e-6)


@torch.no_grad()
def topk_targets(logits: torch.Tensor, k: int = 64, block: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher logits [..., V] -> (top-k indices [..., k], log-probs [..., k+1]); the last
    log-prob is the remaining mass (log(1 - sum top-k), floored)."""
    shape = logits.shape[:-1]
    flat = logits.reshape(-1, logits.shape[-1])
    idx = torch.empty(flat.shape[0], k, dtype=torch.long, device=logits.device)
    lp = torch.empty(flat.shape[0], k + 1, dtype=torch.float32, device=logits.device)
    for s in range(0, flat.shape[0], block):
        chunk = flat[s:s + block].float().log_softmax(-1)
        top, index = chunk.topk(k, dim=-1)
        idx[s:s + block] = index
        lp[s:s + block, :k] = top
        rest = (1.0 - top.exp().sum(-1)).clamp_min(1e-8)
        lp[s:s + block, k] = rest.log()
    return idx.reshape(*shape, k), lp.reshape(*shape, k + 1)


def topk_kd_loss(student_logits: torch.Tensor, idx: torch.Tensor, target_lp: torch.Tensor,
                 weight: torch.Tensor | None = None) -> torch.Tensor:
    """KL(teacher || student) over the teacher's top-k tokens plus one 'rest' bucket
    (a lower bound of the full KL). student_logits [N,V]; idx [N,k]; target_lp [N,k+1];
    weight [N] (0 drops a row). Mean over weighted rows."""
    s = student_logits.float().log_softmax(-1)
    s_top = s.gather(-1, idx)
    s_rest = torch.log1p(-s_top.exp().sum(-1).clamp(max=1 - 1e-6))
    s_all = torch.cat((s_top, s_rest[:, None]), -1)
    p = target_lp.exp()
    kl = (p * (target_lp - s_all)).sum(-1)
    if weight is None:
        return kl.mean()
    return (kl * weight).sum() / weight.sum().clamp_min(1e-6)


def chunked_head_kd(features: torch.Tensor, head_weight: torch.Tensor, idx: torch.Tensor,
                    target_lp: torch.Tensor, weight: torch.Tensor | None = None,
                    chunk: int = 4096) -> torch.Tensor:
    """topk_kd_loss of features @ head_weight^T without holding [N,V] logits for all N
    rows at once (each chunk is recomputed in backward)."""
    n = features.shape[0]
    total = features.new_zeros((), dtype=torch.float32)
    w_all = torch.ones(n, device=features.device) if weight is None else weight.float()
    denom = w_all.sum().clamp_min(1e-6)

    def piece(f, i, t, w):
        logits = F.linear(f, head_weight.to(f.dtype))
        return topk_kd_loss(logits, i, t, w) * w.sum()

    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        w = w_all[s:e]
        if float(w.sum()) == 0.0:
            continue
        if features.requires_grad or head_weight.requires_grad:
            total = total + checkpoint(piece, features[s:e], idx[s:e], target_lp[s:e], w, use_reentrant=False)
        else:
            total = total + piece(features[s:e], idx[s:e], target_lp[s:e], w)
    return total / denom


def _chunk_kl(fs: torch.Tensor, ft: torch.Tensor, w_s: torch.Tensor, w_t: torch.Tensor,
              temperature: float = 1.0, reverse: float = 0.0):
    """(sum over rows of the distillation loss, d sum / d student logits) for one chunk.

    loss = T^2 [KL(p_t || p_s) + reverse * KL(p_s || p_t)], p = softmax(logits / T): the
    forward KL covers the teacher's mass, the reverse term sharpens the student on the
    teacher's modes. Gradients are analytic: T [(p_s - p_t) + reverse * p_s (log p_s -
    log p_t - KL_r)] per row."""
    dt = torch.float64 if fs.dtype == torch.float64 else torch.float32
    log_s = (F.linear(fs, w_s).to(dt) / temperature).log_softmax(-1)
    log_t = (F.linear(ft, w_t).to(dt) / temperature).log_softmax(-1)
    p_t, p_s = log_t.exp(), log_s.exp()
    kl = (p_t * (log_t - log_s)).sum()
    grad = p_s - p_t
    if reverse:
        diff = log_s - log_t
        kl_r = (p_s * diff).sum(-1, keepdim=True)
        kl = kl + reverse * kl_r.sum()
        grad = grad + reverse * p_s * (diff - kl_r)
    return kl * temperature ** 2, grad * temperature


_COMPILED_CHUNK_KL = {}


class _FusedHeadKL(torch.autograd.Function):
    """KL(teacher || student) through both LM heads, chunk by chunk, with the gradient
    formed in the forward pass (softmax(s) - softmax(t) per row): no [N,V] tensor outlives
    a chunk and no logit recomputation in backward. Gradients flow to the student
    features (and to the student head when it requires grad)."""

    @staticmethod
    def forward(ctx, fs, ft, w_s, w_t, chunk, compiled, temperature=1.0, reverse=0.0):
        kernel = _chunk_kl
        if compiled:
            key = (fs.device.type,)
            if key not in _COMPILED_CHUNK_KL:
                _COMPILED_CHUNK_KL[key] = torch.compile(_chunk_kl, dynamic=False)
            kernel = _COMPILED_CHUNK_KL[key]
        n = fs.shape[0]
        acc = torch.float64 if fs.dtype == torch.float64 else torch.float32
        total = torch.zeros((), dtype=acc, device=fs.device)
        grad_fs = torch.empty(fs.shape, dtype=acc, device=fs.device)
        grad_w = torch.zeros(w_s.shape, dtype=acc, device=fs.device) if ctx.needs_input_grad[2] else None
        for s in range(0, n, chunk):
            a, b = fs[s:s + chunk], ft[s:s + chunk]
            pad = chunk - a.shape[0]
            if pad and compiled:  # one static shape for the compiled kernel; padded rows have equal
                a = F.pad(a, (0, 0, 0, pad))  # teacher/student features -> zero KL, zero gradient
                b = F.pad(b, (0, 0, 0, pad))
            kl, g = kernel(a, b, w_s, w_t, temperature, reverse)
            g = g[:chunk - pad]
            total = total + kl
            grad_fs[s:s + chunk] = (g.to(w_s.dtype) @ w_s).to(acc)
            if grad_w is not None:
                grad_w += (g.to(w_s.dtype).t() @ fs[s:s + chunk].to(w_s.dtype)).to(acc)
        ctx.save_for_backward(grad_fs, grad_w if grad_w is not None else torch.empty(0))
        ctx.has_w = grad_w is not None
        ctx.n = n
        ctx.f_dtype, ctx.w_dtype = fs.dtype, w_s.dtype
        return total / n

    @staticmethod
    def backward(ctx, grad_out):
        grad_fs, grad_w = ctx.saved_tensors
        scale = grad_out / ctx.n
        gw = (grad_w * scale).to(ctx.w_dtype) if ctx.has_w else None
        return (grad_fs * scale).to(ctx.f_dtype), None, gw, None, None, None, None, None


def fused_head_kl(fs: torch.Tensor, ft: torch.Tensor, w_s: torch.Tensor, w_t: torch.Tensor,
                  chunk: int = 2048, compiled: bool = False, temperature: float = 1.0,
                  reverse: float = 0.0) -> torch.Tensor:
    """Mean over rows of KL(softmax(ft W_t^T) || softmax(fs W_s^T)), memory-bounded
    (see _FusedHeadKL). fs/ft [N,D] (the matmuls run in w_s/w_t's dtype)."""
    return _FusedHeadKL.apply(fs.to(w_s.dtype), ft.to(w_t.dtype), w_s, w_t, chunk, compiled,
                              float(temperature), float(reverse))


def full_kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                 mask: torch.Tensor | None = None) -> torch.Tensor:
    """Exact KL(teacher || student) over the full vocabulary, mean over (masked) rows."""
    t = teacher_logits.float().log_softmax(-1)
    s = student_logits.float().log_softmax(-1)
    kl = (t.exp() * (t - s)).sum(-1)
    if mask is None:
        return kl.mean()
    m = mask.float()
    return (kl * m).sum() / m.sum().clamp_min(1e-6)


# --------------------------------------------------------------------------- byte <-> token alignment

def token_end_offsets(token_byte_lengths: Sequence[int]) -> torch.Tensor:
    """Cumulative byte end offsets e_1..e_M (exclusive) of a token sequence."""
    return torch.tensor(list(token_byte_lengths), dtype=torch.long).cumsum(0)


def patch_token_index(ends: torch.Tensor, n_patches: int, patch: int = 4, lead: int = 2) -> torch.Tensor:
    """Teacher position aligned to each heavy patch of a PB text trajectory.

    PB frames a text as ids [BOS, TEXT, byte_0, byte_1, ...]: byte k sits at position
    k + lead (lead = 2). Heavy patch j covers positions [patch*j, patch*j + patch - 1] and
    has consumed text bytes [0, patch*j + patch - lead). The teacher input is
    [BOS, t_1..t_M]; teacher position i has consumed t_1..t_i, so the aligned position is
    tau(j) = max{i : e_i <= patch*j + patch - lead} (0 = the BOS state).
    ends [M] (token_end_offsets). Returns tau [n_patches] (long)."""
    consumed = torch.arange(n_patches, dtype=torch.long) * patch + (patch - lead)
    return torch.searchsorted(ends.to(torch.long), consumed, right=True)


_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")


def memory_mask_words(text: str) -> list[tuple[int, int]]:
    """Character spans of 'knowledge' words for the memory mask: capitalized words that
    are not sentence-initial, and numbers of 2+ digits, that do NOT occur earlier in the
    text (so reading/copying from context is never masked)."""
    spans = []
    seen: set[str] = set()
    for m in _WORD.finditer(text):
        word = m.group(0)
        start = m.start()
        before = text[:start].rstrip(" \t\"'(“‘")
        initial = (not before) or before[-1] in ".!?:\n;"
        capital = word[0].isupper() and not initial
        number = sum(ch.isdigit() for ch in word) >= 2
        if (capital or number) and word not in seen:
            spans.append((start, m.end()))
        seen.add(word)
    return spans


def char_to_byte_offsets(text: str) -> list[int]:
    """offset[i] = UTF-8 byte offset of character i (len(text)+1 entries)."""
    out = [0]
    for ch in text:
        out.append(out[-1] + len(ch.encode("utf-8")))
    return out


def masked_token_flags(text: str, ends: Sequence[int]) -> torch.Tensor:
    """flag[i] for teacher tokens t_1..t_M: the token overlaps a memory-mask word."""
    offsets = char_to_byte_offsets(text)
    spans = [(offsets[a], offsets[b]) for a, b in memory_mask_words(text)]
    flags = torch.zeros(len(ends), dtype=torch.bool)
    start = 0
    j = 0
    for i, end in enumerate(ends):
        while j < len(spans) and spans[j][1] <= start:
            j += 1
        k = j
        while k < len(spans) and spans[k][0] < end:
            if spans[k][1] > start:
                flags[i] = True
                break
            k += 1
        start = int(end)
    return flags


def masked_byte_flags(text: str) -> torch.Tensor:
    """flag per UTF-8 byte of text: inside a memory-mask word."""
    offsets = char_to_byte_offsets(text)
    flags = torch.zeros(offsets[-1], dtype=torch.bool)
    for a, b in memory_mask_words(text):
        flags[offsets[a]:offsets[b]] = True
    return flags


# --------------------------------------------------------------------------- byte-level PB

def byte_student_recipe(c: TeacherConfig, *, patch_width: int = 512, micro_width: int = 384,
                        local_attention_layers: int = 2, local_decoder_layers: int = 2,
                        local_heads: int = 4, conv: int = 4, segment_backend: str = "batched") -> dict:
    """Hierarchical PB recipe whose heavy core has the teacher's shape: width, layers,
    the Gated DeltaNet shape of TeacherConfig.gdn_dims (the hybrid teacher's own linear
    attention, else the attention heads), core SwiGLU MLP of the teacher's hidden width,
    RMS norms (teacher epsilon) and the residual readout. The byte path follows the Stage 3
    recipe (pb_s3_330m.json) at a smaller width."""
    gdn = c.gdn_dims()
    conv = gdn["conv"] if c.linear_key_heads else conv
    value_layout = {}
    if gdn["value_heads"] != gdn["heads"] or gdn["value_heads"] * gdn["value_dim"] != c.width:
        value_layout = {"core_mixer_value_heads": gdn["value_heads"], "core_mixer_value_dim": gdn["value_dim"]}
    return {
        "config": {"width": c.width, "layers": c.layers, "slots": 8, "stride": 4, "cycles": 1,
                   "memory_dim": 64, "tape": 256, "vocab": 265, "image_size": 32,
                   "use_slow": True, "use_reader": True},
        "options": {
            "backbone": "scan", "text_only_fast_path": True, "patch_size": 4,
            "patch_width": patch_width, "micro_patch_size": 2, "micro_patch_width": micro_width,
            "tape_attention_mode": "low_rank", "scan_backend": "fused", "reader_backend": "windowed",
            "text_memory_writes": True, "local_decoder_layers": local_decoder_layers,
            "segment_backend": segment_backend, "reader_window": 2048, "reader_prev_byte_keys": True,
            "reader_gate_features": True, "slow_backend": "parallel",
            "local_attention_layers": local_attention_layers, "local_attention_window": 256,
            "local_attention_heads": local_heads,
            "core_mixer": "gated_deltanet", "core_mixer_heads": gdn["heads"],
            "core_mixer_head_dim": gdn["key_dim"], "core_mixer_conv": conv,
            "core_mlp_ratio": c.hidden / c.width, "core_norm": "rms", "core_readout": "residual",
            "core_norm_eps": c.norm_eps, "core_mixer_norm_eps": c.norm_eps, "core_mixer_l2norm": "hf",
            **value_layout,
        },
    }


@torch.no_grad()
def load_core_into_hierarchy(model: nn.Module, core: TokenCore) -> list[str]:
    """Copy an all-Gated-DeltaNet TokenCore's blocks and final norm into a hierarchical
    PB built from byte_student_recipe (same module classes, so the keys match)."""
    if any(m != "gated_deltanet" for m in core.spec.mixers):
        raise ValueError("only an all-gated_deltanet core can be loaded into the hierarchy")
    if getattr(model, "core_final_norm", None) is None:
        raise ValueError("the hierarchy must use core_readout='residual'")
    loaded = []
    for i, (dst, src) in enumerate(zip(model.fast.blocks, core.blocks)):
        dst.load_state_dict(src.state_dict(), strict=True)
        loaded.append(f"fast.blocks.{i}")
    model.core_final_norm.load_state_dict(core.norm.state_dict(), strict=True)
    loaded.append("core_final_norm")
    return loaded


def parameter_count(module: nn.Module, names: Iterable[str] | None = None) -> int:
    return sum(p.numel() for n, p in module.named_parameters() if names is None or n in names)
