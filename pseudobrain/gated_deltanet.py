"""Gated DeltaNet heavy core (opt-in: EfficiencyOptions.core_mixer='gated_deltanet').

The default heavy core (efficiency/scan.py ScanLayer) is a minGRU-style
DIAGONAL recurrence: each layer carries a vector of `width` numbers. This
module replaces every core layer by a Gated DeltaNet token mixer (Yang, Kautz &
Hatamizadeh 2024, "Gated Delta Networks: Improving Mamba2 with Delta Rule",
arXiv:2412.06464) whose state is a MATRIX per head, S in R^{dk x dv}:

    S_t = a_t S_{t-1} + k_t u_t^T,   u_t = b_t (v_t - a_t S_{t-1}^T k_t)
        (= a_t (I - b_t k_t k_t^T) S_{t-1} + b_t k_t v_t^T)
    o_t = S_t^T q_t / sqrt(dk)

with L2-normalized q/k (after a short causal depthwise convolution and SiLU),
a_t = exp(g_t), g_t = -exp(A_log) * softplus(W_a x_t + dt_bias) (Mamba2 decay),
b_t = sigmoid(W_b x_t), and the output o_t = RMSNorm(o_t) * SiLU(W_g x_t) per head
(the layout of flash-linear-attention's GatedDeltaNet layer, S stored [dk, dv]).

Interface. A GatedDeltaNetLayer is a drop-in for ScanLayer inside the stack:
pre-LN, residual out projection scaled by 1/sqrt(layers), optional SwiGLU MLP,
and a per-step output VECTOR hs_t = the gated, normalized o_t ([B,T,width]; the
value heads tile the width). hs plays the role of the minGRU state sequence:
the hierarchy reads the top layer's hs as the core context and carries each
layer's last hs in state.fast [L,B,width] (unchanged layout). The recurrent
memory lives in two extra state fields: core_mixer_state [L,B,H,dk,dv] (fp32)
and core_mixer_conv [L,B,conv-1,channels] (the short-convolution input tail).
Chunked == full holds by construction: any chunking carries S and the tail.

Kernels (all compute the same recurrence; only float summation order differs):
  * sequential  per-step recurrence (tests, `serial=True`, short sequences and
                one-step decode: T <= SEQUENTIAL_MAX)
  * chunked     chunkwise-parallel WY algorithm in plain PyTorch (fp32; one
                small triangular solve per chunk; autograd), the default
  * fla         flash-linear-attention's Triton chunk kernels (scan_backend
                'fused' on CUDA when `fla` imports), wrapped as a torch custom
                op so torch.compile traces through it. Optional dependency.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .scan import SCAN_BACKENDS, SwiGLU, make_norm, mlp_hidden_width

# Chunk length of the chunkwise PyTorch algorithm, and of fla's kernels.
CHUNK = 64
FLA_CHUNK = 64
# Sequences up to this length use the per-step recurrence (e.g. the one-patch
# step of byte-by-byte decode): cheaper than a chunk solve, and plain torch ops
# (CUDA-graph capturable).
SEQUENTIAL_MAX = 4
L2_EPS = 1e-6
L2_MODES = ("max", "hf")


def l2_normalize(x: torch.Tensor, mode: str = "max", eps: float = L2_EPS) -> torch.Tensor:
    """q/k normalization over the last dim. 'max' (the PB default since the core mixer
    was added): x / max(||x||, eps) (F.normalize). 'hf': x * rsqrt(sum x^2 + eps), computed
    in fp32, as Hugging Face transformers' qwen3_5 l2norm and flash-linear-attention's
    use_qk_l2norm_in_kernel do. They differ for small vectors: [1e-4, 0] -> [1, 0] (max)
    vs [0.0995, 0] (hf)."""
    if mode == "max":
        return F.normalize(x, dim=-1, eps=eps)
    if mode == "hf":
        xf = x.float()
        return (xf * torch.rsqrt(xf.square().sum(-1, keepdim=True) + eps)).to(x.dtype)
    raise ValueError(f"l2 normalization mode must be one of {L2_MODES}")
NORM_EPS = 1e-5


# ----------------------------------------------------------------------------
# The recurrence. q, k [B,T,H,K]; v [B,T,H,V]; g, beta [B,T,H] (g <= 0 is the
# log decay); h0 [B,H,K,V]. Returns (o [B,T,H,V] fp32, final state fp32).
# ----------------------------------------------------------------------------
def _compute_dtype(x: torch.Tensor) -> torch.dtype:
    """fp32 arithmetic (fp64 when the inputs are fp64, for reference checks)."""
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def _full_precision(fn):
    """Run a PyTorch kernel with autocast off: the recurrence stays at fp32 even
    inside an autocast region (the projections around it still use AMP)."""
    def wrapper(q, *args, **kwargs):
        with torch.autocast(q.device.type, enabled=False):
            return fn(q, *args, **kwargs)
    wrapper.__name__, wrapper.__doc__ = fn.__name__, fn.__doc__
    return wrapper


@_full_precision
def delta_rule_sequential(q, k, v, g, beta, scale, h0):
    """Reference per-step recurrence (fp32)."""
    dtype = _compute_dtype(q)
    state = h0.to(dtype)
    q, k, v = q.to(dtype) * scale, k.to(dtype), v.to(dtype)
    decay, beta = g.to(dtype).exp(), beta.to(dtype)
    rows = []
    for t in range(q.shape[1]):
        state = state * decay[:, t, :, None, None]
        kt = k[:, t]
        u = (v[:, t] - torch.einsum("bhkv,bhk->bhv", state, kt)) * beta[:, t, :, None]
        state = state + kt[..., None] * u[..., None, :]
        rows.append(torch.einsum("bhkv,bhk->bhv", state, q[:, t]))
    return torch.stack(rows, 1), state


@_full_precision
def delta_rule_chunked(q, k, v, g, beta, scale, h0, chunk: int | None = None):
    """Chunkwise-parallel gated delta rule (WY representation), fp32.

    Within a chunk (positions 1..C, cumulative log decay c_t, G_t = exp(c_t)):
      u_t = b_t (v_t - G_t S0^T k_t - sum_{i<t} (G_t/G_i) (k_t.k_i) u_i)
      => (I + A) U = diag(b) V - diag(b G) K S0,  A_ti = b_t (G_t/G_i) k_t.k_i (i<t)
      => U = T diag(b) V - T diag(b G) K S0 = U' - W S0,  T = (I + A)^-1
      O    = diag(G) Q S0 + (Q K^T o D) U,  D_ti = G_t/G_i (i<=t)
      S_C  = G_C S0 + K^T diag(G_C/G) U.
    Every exponent is <= 0 (g <= 0), so nothing overflows. U', W, Q K^T o D
    are computed for all chunks at once; only S0 -> S_C runs chunk by chunk.
    """
    batch, length, heads, dk = q.shape
    dv = v.shape[-1]
    size = min(CHUNK if chunk is None else chunk, length)
    pad = (-length) % size
    # [B,H,T,*] fp32; padded steps have k = v = 0, g = 0, b = 0: the identity.
    dtype = _compute_dtype(q)
    q = q.to(dtype).transpose(1, 2) * scale
    k = k.to(dtype).transpose(1, 2)
    v = v.to(dtype).transpose(1, 2)
    g = g.to(dtype).transpose(1, 2)
    beta = beta.to(dtype).transpose(1, 2)
    if pad:
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        g, beta = F.pad(g, (0, pad)), F.pad(beta, (0, pad))
    count = (length + pad) // size
    q = q.reshape(batch, heads, count, size, dk)
    k = k.reshape(batch, heads, count, size, dk)
    v = v.reshape(batch, heads, count, size, dv)
    g = g.reshape(batch, heads, count, size).cumsum(-1)
    beta = beta.reshape(batch, heads, count, size)
    inclusive = torch.ones(size, size, dtype=torch.bool, device=q.device).tril()
    strict = inclusive.tril(-1)
    decay = (g[..., :, None] - g[..., None, :]).masked_fill(~inclusive, float("-inf")).exp()
    a = (beta[..., :, None] * (k @ k.transpose(-1, -2)) * decay).masked_fill(~strict, 0.0)
    eye = torch.eye(size, dtype=q.dtype, device=q.device)
    solve = torch.linalg.solve_triangular(
        eye + a, eye.expand_as(a), upper=False, unitriangular=True
    )
    gamma = g.exp()
    w = solve @ (k * (beta * gamma)[..., None])
    u0 = solve @ (v * beta[..., None])
    qk = (q @ k.transpose(-1, -2)) * decay
    qg = q * gamma[..., None]
    tail = (g[..., -1:] - g).exp()[..., None] * k  # (G_C/G_i) k_i
    last = gamma[..., -1]
    state = h0.to(dtype)
    rows = []
    for n in range(count):
        u = u0[:, :, n] - w[:, :, n] @ state
        rows.append(qg[:, :, n] @ state + qk[:, :, n] @ u)
        state = state * last[:, :, n, None, None] + tail[:, :, n].transpose(-1, -2) @ u
    out = torch.stack(rows, 2).reshape(batch, heads, count * size, dv)[:, :, :length]
    return out.transpose(1, 2), state


# ----------------------------------------------------------------------------
# flash-linear-attention chunk kernels as a torch custom op (optional).
# ----------------------------------------------------------------------------
_FLA: dict = {}


def fla_functions():
    """(forward, backward) of fla's gated delta rule chunk kernels, or None."""
    if "functions" not in _FLA:
        try:
            from fla.ops.gated_delta_rule.chunk import (
                chunk_gated_delta_rule_bwd,
                chunk_gated_delta_rule_fwd,
            )
            _FLA["functions"] = (chunk_gated_delta_rule_fwd, chunk_gated_delta_rule_bwd)
        except Exception as error:  # noqa: BLE001 - optional dependency
            _FLA["functions"] = None
            _FLA["error"] = f"{type(error).__name__}: {error}"
    return _FLA["functions"]


def fla_usable(*tensors: torch.Tensor) -> bool:
    """fla's chunk kernels apply: CUDA tensors and head widths <= 256. tensors[0] is the
    matrix state [..., dk, dv] (both head widths are checked)."""
    return (
        all(t.is_cuda for t in tensors)
        and tensors[0].shape[-1] <= 256
        and (tensors[0].ndim < 2 or tensors[0].shape[-2] <= 256)
        and fla_functions() is not None
    )


@torch.library.custom_op(
    "pseudobrain::gated_delta_chunk",
    mutates_args=(),
    schema="(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor h0, float scale) -> (Tensor, Tensor, Tensor, Tensor)",
)
def _fla_chunk_op(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
    beta: torch.Tensor, h0: torch.Tensor, scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(o [B,T,H,V] in q's dtype, final state fp32, fla's chunk-cumulated g, WY A)."""
    forward, _ = fla_functions()
    gc, o, a, final, _, _ = forward(
        q=q.contiguous(), k=k.contiguous(), v=v.contiguous(), g=g.contiguous(),
        beta=beta.contiguous(), scale=scale, initial_state=h0.contiguous(),
        output_final_state=True, chunk_size=FLA_CHUNK,
    )
    return o.to(q.dtype), final.float(), gc, a


@_fla_chunk_op.register_fake
def _(q, k, v, g, beta, h0, scale):
    batch, length, heads, dk = q.shape
    return (
        q.new_empty(batch, length, heads, v.shape[-1]),
        h0.new_empty(batch, heads, dk, v.shape[-1], dtype=torch.float32),
        g.new_empty(batch, length, heads, dtype=torch.float32),
        k.new_empty(batch, length, heads, FLA_CHUNK),
    )


@torch.library.custom_op(
    "pseudobrain::gated_delta_chunk_backward",
    mutates_args=(),
    schema="(Tensor q, Tensor k, Tensor v, Tensor gc, Tensor beta, Tensor a, Tensor h0, Tensor grad_o, Tensor grad_final, float scale) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)",
)
def _fla_chunk_backward_op(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gc: torch.Tensor,
    beta: torch.Tensor, a: torch.Tensor, h0: torch.Tensor, grad_o: torch.Tensor,
    grad_final: torch.Tensor, scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _, backward = fla_functions()
    dq, dk, dv, dbeta, dg, dh0, _, _ = backward(
        q=q.contiguous(), k=k.contiguous(), v=v.contiguous(), g=gc.contiguous(),
        beta=beta.contiguous(), A=a.contiguous(), scale=scale,
        initial_state=h0.contiguous(), do=grad_o.contiguous(),
        dht=grad_final.contiguous(), chunk_size=FLA_CHUNK,
    )
    return (dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dg.float(),
            dbeta.to(beta.dtype), dh0.to(h0.dtype))


@_fla_chunk_backward_op.register_fake
def _(q, k, v, gc, beta, a, h0, grad_o, grad_final, scale):
    return (torch.empty_like(q), torch.empty_like(k), torch.empty_like(v),
            torch.empty_like(gc, dtype=torch.float32), torch.empty_like(beta),
            torch.empty_like(h0))


def _fla_setup(ctx, inputs, output):
    q, k, v, _g, beta, h0, scale = inputs
    _o, _final, gc, a = output
    ctx.save_for_backward(q, k, v, gc, beta, a, h0)
    ctx.scale = scale


def _fla_grad(ctx, grad_o, grad_final, _grad_gc, _grad_a):
    q, k, v, gc, beta, a, h0 = ctx.saved_tensors
    if grad_o is None:
        grad_o = torch.zeros(*q.shape[:3], v.shape[-1], dtype=q.dtype, device=q.device)
    if grad_final is None:
        grad_final = torch.zeros(h0.shape, dtype=torch.float32, device=h0.device)
    dq, dk, dv, dg, dbeta, dh0 = _fla_chunk_backward_op(
        q, k, v, gc, beta, a, h0, grad_o.to(q.dtype), grad_final.float(), ctx.scale
    )
    return dq, dk, dv, dg, dbeta, dh0, None


_fla_chunk_op.register_autograd(_fla_grad, setup_context=_fla_setup)


def delta_rule_fla(q, k, v, g, beta, scale, h0):
    """fla chunk kernels (Triton): q/k/v at their own precision (bf16 under
    autocast; fp32 inputs use TF32 dot products), fp32 state and decay."""
    o, final, _, _ = _fla_chunk_op(q, k, v.to(q.dtype), g.float(), beta.to(q.dtype),
                                   h0.float(), float(scale))
    return o.float(), final


KERNELS = ("sequential", "chunked", "fla")


def select_kernel(backend: str, serial: bool, length: int, *tensors: torch.Tensor) -> str:
    """Kernel for one call (execution only; see the module notes)."""
    if serial or length <= SEQUENTIAL_MAX:
        return "sequential"
    if backend == "fused" and fla_usable(*tensors):
        return "fla"
    return "chunked"


def gated_delta_rule(q, k, v, g, beta, scale, h0, kernel: str = "chunked"):
    """The recurrence with the given kernel ('sequential', 'chunked' or 'fla')."""
    if kernel == "sequential":
        return delta_rule_sequential(q, k, v, g, beta, scale, h0)
    if kernel == "fla":
        return delta_rule_fla(q, k, v, g, beta, scale, h0)
    if kernel == "chunked":
        return delta_rule_chunked(q, k, v, g, beta, scale, h0)
    raise ValueError(f"gated delta rule kernel must be one of {KERNELS}")


# ----------------------------------------------------------------------------
# Layer and stack.
# ----------------------------------------------------------------------------
def short_convolution(x, tail, weight):
    """Causal depthwise convolution over time with a carried input tail, then SiLU.

    x [B,T,C]; tail [B,K-1,C] (the last K-1 inputs before x); weight [C,K]
    (weight[:, -1] multiplies the current input) or None (no convolution).
    Computed in fp32. Returns (y [B,T,C] fp32, new tail [B,K-1,C] = the last
    K-1 inputs, in the tail's dtype)."""
    dtype = _compute_dtype(x)
    if weight is None:
        return F.silu(x.to(dtype)), tail
    size = weight.shape[1]
    length = x.shape[1]
    extended = torch.cat((tail.to(x.dtype), x), dim=1) if size > 1 else x
    new_tail = extended[:, extended.shape[1] - (size - 1):] if size > 1 else tail
    extended, weight = extended.to(dtype), weight.to(dtype)
    if extended.is_cuda and not torch.compiler.is_compiling():
        # eager CUDA: one depthwise conv kernel instead of K multiply-add passes over [B,T,C]
        # (under torch.compile the loop below fuses into one kernel by itself)
        with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):  # fp32 stays fp32
            y = F.conv1d(extended.transpose(1, 2), weight[:, None, :], groups=weight.shape[0])
        return F.silu(y.transpose(1, 2).contiguous()), new_tail.to(tail.dtype)
    y = extended[:, :length] * weight[:, 0]
    for j in range(1, size):
        y = y + extended[:, j:j + length] * weight[:, j]
    return F.silu(y), new_tail.to(tail.dtype)


class GatedDeltaNetLayer(nn.Module):
    """Pre-LN Gated DeltaNet token mixer with ScanLayer's residual conventions.

    One input projection yields [q | k | v | output gate | decay | beta] (the
    short convolution covers q, k, v). By default heads * value head width ==
    width. value_heads > heads (a multiple) gives the Qwen3-Next / Qwen3.5 layout:
    `heads` query/key heads, each shared by value_heads/heads value heads (q and k
    repeated after the L2 norm), value width value_heads * value_dim (any size;
    the out projection maps it back to width), decay/beta per value head.
    norm_eps is the per-head output RMSNorm epsilon, block_norm_eps the pre-norm's.
    """

    def __init__(self, width: int, residual_scale: float, heads: int, key_dim: int,
                 conv: int, backend: str = "reference", mlp_hidden: int = 0, norm: str = "layer",
                 *, value_heads: int = 0, value_dim: int = 0, norm_eps: float = NORM_EPS,
                 block_norm_eps: float = 1e-5, l2norm: str = "max"):
        super().__init__()
        if l2norm not in L2_MODES:
            raise ValueError(f"l2norm must be one of {L2_MODES}")
        self.l2norm = l2norm
        value_heads = value_heads or heads
        if heads < 1 or value_heads % heads:
            raise ValueError("gated deltanet value heads must be a multiple of the key heads")
        if not value_dim and width % value_heads:
            raise ValueError(
                f"core_mixer_heads={value_heads} must divide the core width {width}"
            )
        if key_dim < 1 or conv < 0:
            raise ValueError("gated deltanet key width must be positive and conv >= 0")
        self.width, self.heads = width, heads
        self.value_heads = value_heads
        self.key_dim, self.value_dim = key_dim, value_dim or width // value_heads
        self.value_width = value_heads * self.value_dim
        self.norm_eps = norm_eps
        self.conv_size = conv
        self.channels = 2 * heads * key_dim + self.value_width  # q, k, v
        self.norm = make_norm(width, norm, block_norm_eps)
        self.proj = nn.Linear(width, self.channels + self.value_width + 2 * value_heads, bias=False)
        # Depthwise causal convolution (nn.Conv1d's default initialization);
        # conv = 0: none (SiLU only).
        if conv:
            self.conv_weight = nn.Parameter(torch.empty(self.channels, conv))
            bound = 1.0 / math.sqrt(conv)
            nn.init.uniform_(self.conv_weight, -bound, bound)
        else:
            self.register_parameter("conv_weight", None)
        # Mamba2-style decay: g = -exp(A_log) * softplus(a + dt_bias); A ~ U(0, 16),
        # dt ~ logU(1e-3, 1e-1) (flash-linear-attention's initialization).
        self.A_log = nn.Parameter(torch.empty(value_heads).uniform_(0, 16).log())
        dt = torch.exp(torch.rand(value_heads) * (math.log(0.1) - math.log(0.001))
                       + math.log(0.001)).clamp_min(1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.o_norm = nn.Parameter(torch.ones(self.value_dim))
        self.out = nn.Linear(self.value_width, width, bias=False)
        self.residual_scale = residual_scale
        self.backend = backend
        self.mlp = (SwiGLU(width, mlp_hidden, norm=norm, norm_eps=block_norm_eps)
                    if mlp_hidden > 0 else None)

    @property
    def tail_length(self) -> int:
        return max(self.conv_size - 1, 0)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor, matrix: torch.Tensor,
                tail: torch.Tensor, *, serial: bool = False, return_states: bool = False,
                return_output: bool = True):
        """x [B,T,W]; hidden [B,W] (last output vector; not read); matrix
        [B,Hv,dk,dv]; tail [B,conv-1,channels]. Returns (output or None, last
        hs [B,W], new matrix, new tail[, hs [B,T,W]]); hs is the gated, normed
        per-step output (out-projected when the value width is not W)."""
        batch, length, width = x.shape
        heads, dk, dv = self.heads, self.key_dim, self.value_dim
        vheads = getattr(self, "value_heads", heads)  # modules pickled pre-option
        vwidth = getattr(self, "value_width", width)
        z = self.proj(self.norm(x))
        qkv, gate, decay = z.split((self.channels, vwidth, 2 * vheads), dim=-1)
        qkv, new_tail = short_convolution(qkv, tail, self.conv_weight)
        q, k, v = qkv.split((heads * dk, heads * dk, vwidth), dim=-1)
        mode = getattr(self, "l2norm", "max")  # modules pickled pre-option
        q = l2_normalize(q.view(batch, length, heads, dk), mode)
        k = l2_normalize(k.view(batch, length, heads, dk), mode)
        if vheads != heads:
            q = q.repeat_interleave(vheads // heads, dim=2)
            k = k.repeat_interleave(vheads // heads, dim=2)
        v = v.view(batch, length, vheads, dv)
        dtype = _compute_dtype(hidden)  # fp32 (fp64 for an fp64 model)
        a, b = decay.to(dtype).chunk(2, dim=-1)
        g = -self.A_log.to(dtype).exp() * F.softplus(a + self.dt_bias.to(dtype))
        beta = torch.sigmoid(b)
        # The matrix state goes first: fla_usable checks its head widths (dk, dv).
        # (Passing x checked the model width, so any core wider than 256 silently
        # fell back to the plain-PyTorch chunked kernel.)
        kernel = select_kernel(self.backend, serial, length, matrix, x)
        if kernel == "fla" and z.dtype in (torch.bfloat16, torch.float16):
            # Under autocast the Triton kernels read q/k/v at the matmul
            # precision (fp32 accumulation, fp32 state and decay).
            q, k, v = q.to(z.dtype), k.to(z.dtype), v.to(z.dtype)
        o, new_matrix = gated_delta_rule(q, k, v, g, beta, dk ** -0.5, matrix, kernel)
        o = o * torch.rsqrt(o.square().mean(-1, keepdim=True) + getattr(self, "norm_eps", NORM_EPS))
        o = o * self.o_norm.to(o.dtype) * F.silu(gate.to(o.dtype).view(batch, length, vheads, dv))
        hs = o.reshape(batch, length, vwidth).to(hidden.dtype)
        new_matrix = new_matrix.to(matrix.dtype)
        if vwidth != width:
            # The state.fast row / top state sequence keeps width W.
            projected = self.out(hs).to(hidden.dtype)
            vector, sequence = projected[:, -1], projected
        else:
            vector, sequence = hs[:, -1], hs
        if not return_output:
            result = (None, vector, new_matrix, new_tail)
            return result + (sequence,) if return_states else result
        projection = projected if vwidth != width and not torch.is_grad_enabled() else self.out(hs)
        output = x + self.residual_scale * projection.to(x.dtype)
        if self.mlp is not None:
            output = output + self.residual_scale * self.mlp(output).to(x.dtype)
        result = (output, vector, new_matrix, new_tail)
        return result + (sequence,) if return_states else result


class GatedDeltaNetStack(nn.Module):
    """The heavy core with Gated DeltaNet layers (the ScanFast counterpart).

    forward(x, hidden, matrix, tail) -> (outputs, hidden [L,B,W], matrix
    [L,B,H,dk,dv], tail [L,B,conv-1,C][, top layer hs [B,T,W]]). All three
    states are required together (None for all = a fresh stream).
    """
    # Execution only, as ScanFast.compile_training (set by the hierarchy under
    # segment_backend='compiled').
    compile_training = False

    def __init__(self, width: int, layers: int, heads: int, key_dim: int, conv: int,
                 backend: str = "reference", checkpoint_layers: bool = False,
                 mlp_ratio: float = 0.0, mlp_last_layer: bool = True, norm: str = "layer",
                 *, value_heads: int = 0, value_dim: int = 0, norm_eps: float = NORM_EPS,
                 block_norm_eps: float = 1e-5, l2norm: str = "max"):
        super().__init__()
        if width < 1 or layers < 1:
            raise ValueError("width and layers must be positive")
        if backend not in SCAN_BACKENDS:
            raise ValueError("invalid scan backend")
        if type(checkpoint_layers) is not bool:
            raise ValueError("checkpoint_layers must be boolean")
        if (isinstance(mlp_ratio, bool) or not isinstance(mlp_ratio, (int, float))
                or not math.isfinite(mlp_ratio) or mlp_ratio < 0):
            raise ValueError("mlp_ratio must be a finite nonnegative number")
        self.backend = backend
        self.checkpoint_layers = checkpoint_layers
        self.width, self.num_layers = width, layers
        key_dim = key_dim or width // max(value_heads or heads, 1)
        hidden = mlp_hidden_width(width, mlp_ratio)
        self.blocks = nn.ModuleList(
            GatedDeltaNetLayer(width, 1.0 / math.sqrt(layers), heads, key_dim, conv, backend,
                               hidden if (mlp_last_layer or i < layers - 1) else 0, norm,
                               value_heads=value_heads, value_dim=value_dim, norm_eps=norm_eps,
                               block_norm_eps=block_norm_eps, l2norm=l2norm)
            for i in range(layers))
        first = self.blocks[0]
        self.heads, self.key_dim, self.value_dim = first.heads, first.key_dim, first.value_dim
        self.value_heads = first.value_heads
        self.channels, self.tail_length = first.channels, first.tail_length

    def state_shapes(self, batch: int):
        """(matrix, tail) shapes of the extra recurrent state for `batch` streams."""
        layers = self.num_layers
        return ((layers, batch, getattr(self, "value_heads", self.heads), self.key_dim, self.value_dim),
                (layers, batch, self.tail_length, self.channels))

    def empty_state(self, batch: int, like: torch.Tensor):
        matrix, tail = self.state_shapes(batch)
        return like.new_zeros(matrix), like.new_zeros(tail)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None,
                matrix: torch.Tensor | None = None, tail: torch.Tensor | None = None,
                *, serial: bool = False, depth: int | None = None,
                return_state_sequence: bool = False, return_output: bool = True):
        if hidden is None and matrix is None and tail is None:
            hidden = x.new_zeros(self.num_layers, x.shape[0], self.width)
            matrix, tail = self.empty_state(x.shape[0], hidden)
        elif matrix is None or tail is None or hidden is None:
            raise ValueError(
                "the gated_deltanet core needs its matrix and convolution states "
                "(core_mixer_state / core_mixer_conv) with the vector state; the motor "
                "path (frame events) does not carry them and is not supported with "
                "core_mixer='gated_deltanet'"
            )
        if self.backend == "fused" and x.is_cuda and not torch.compiler.is_compiling():
            fla_functions()  # import the optional kernels outside any compiled region
        if (self.compile_training and not serial and torch.is_grad_enabled()
                and not torch.compiler.is_compiling()):
            from .hierarchical_patches import _compiled
            return _compiled(_gated_deltanet_stack)(
                self, x, hidden, matrix, tail, depth, return_state_sequence, False,
                return_output,
            )
        return _gated_deltanet_stack(self, x, hidden, matrix, tail, depth,
                                     return_state_sequence, serial, return_output)


def _gated_deltanet_stack(module: GatedDeltaNetStack, x, hidden, matrix, tail, depth,
                          return_state_sequence: bool, serial: bool = False,
                          return_output: bool = True):
    """GatedDeltaNetStack.forward's layer stack (compiled under compile_training)."""
    depth = module.num_layers if depth is None else depth
    if not 1 <= depth <= module.num_layers:
        raise ValueError("depth outside recurrent stack")
    if x.ndim != 3 or x.shape[1] < 1 or x.shape[-1] != module.width:
        raise ValueError("expected nonempty [B,T,width]")
    batch = x.shape[0]
    matrix_shape, tail_shape = module.state_shapes(batch)
    if (hidden.shape != (module.num_layers, batch, module.width)
            or matrix.shape != matrix_shape or tail.shape != tail_shape):
        raise ValueError("wrong recurrent state shape")
    hiddens, matrices, tails = [], [], []
    top = None
    checkpointed = module.checkpoint_layers and module.training and torch.is_grad_enabled()
    for i, block in enumerate(module.blocks[:depth]):
        need_states = return_state_sequence and i == depth - 1
        output = return_output or i < depth - 1
        if checkpointed:
            result = checkpoint(block, x, hidden[i], matrix[i], tail[i], use_reentrant=False,
                                serial=serial, return_states=need_states,
                                return_output=output)
        else:
            result = block(x, hidden[i], matrix[i], tail[i], serial=serial,
                           return_states=need_states, return_output=output)
        x = result[0]
        hiddens.append(result[1])
        matrices.append(result[2])
        tails.append(result[3])
        if need_states:
            top = result[4]
    for i in range(depth, module.num_layers):
        hiddens.append(hidden[i])
        matrices.append(matrix[i])
        tails.append(tail[i])
    result = (x, torch.stack(hiddens), torch.stack(matrices), torch.stack(tails))
    return result + (top,) if return_state_sequence else result
