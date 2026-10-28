"""Triton kernels for sparse-MoE decode (Track X2 serving, docs/SKILL_TRANSPLANT.md).

At decode a token routes to top_k experts; only those experts' weights should be read, once. The
training/prefill paths (SparseMoE._experts_grouped: torch._grouped_mm over token-sorted rows; the
loop) either synchronise with the host (the loop's bincount().tolist(); _grouped_mm on the GB10)
or pay for sorting and scatter that one token does not need, and neither can be captured in a
CUDA graph. `expert_gemv` is one kernel: program (r, n-block) reads the expert id of routed row r
from a device tensor and multiplies that expert's [BLOCK_N, K] weight tile with the row's input,
accumulating in fp32. `moe_decode` = gate_up GEMV -> SiLU(gate) * up -> down GEMV -> routing-
weighted sum over the k slots: no host synchronisation, static shapes, capturable.

Used by SparseMoE only inside `decode_mode()` (efficiency/fast_decode.StaticDecoder), so training,
prefill and the evaluation paths keep their numerics. The kernel accumulates in fp32 and rounds the
SiLU product to the weight dtype as the grouped path does; it differs from that path by bf16
rounding only.

`grouped_mm` is the training/prefill grouped GEMM for frozen experts. On GPUs without a native
torch._grouped_mm (GB10 sm_121, RTX PRO 6000 sm_120) torch reads the group offsets back to the host
and runs one small mm per expert (256 launches per projection, a host sync per call). Here each
program finds its expert and row tile on the device from the per-expert row counts, so a projection
is one launch with no synchronisation; the input gradient is the same kernel on the transposed
weights. Weight gradients are not implemented: SparseMoE uses it only while its experts are frozen.
`grouped_gather_mm` reads each routed row straight from its token (no [T*k, W] gathered copy),
`swiglu` is silu(gate)*up in one pass (no chunk/cat in its gradient), and `grouped_mm_combine` does the
routing-weighted sum over each token's k slots from the expert-sorted outputs (no permuted copy, no
index_put in the backward; the routing weight scales the input gradient inside the GEMM).
"""
from __future__ import annotations

import contextlib
import os

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only installs (CI): decode_mode() then keeps the regular paths
    triton = None
    tl = None

STATE = {"decode": False, "regular_single": False,
         "block_n": 32, "block_k": 256, "warps": 4, "max_rows": 128}


def available() -> bool:
    return triton is not None


def interpreting() -> bool:
    """Triton's CPU interpreter (TRITON_INTERPRET=1, set before import): the kernels run on CPU
    tensors, for tests without a GPU."""
    return available() and os.environ.get("TRITON_INTERPRET") == "1"


@contextlib.contextmanager
def decode_mode(enabled: bool = True, *, regular_single: bool = False):
    """SparseMoE layers use moe_decode for small batches (<= max_rows routed rows) inside."""
    old = STATE["decode"]
    old_regular = STATE["regular_single"]
    STATE["decode"] = enabled and available()
    STATE["regular_single"] = regular_single
    try:
        yield
    finally:
        STATE["decode"] = old
        STATE["regular_single"] = old_regular


if triton is not None:
    @triton.jit
    def _expert_gemv_kernel(x_ptr, idx_ptr, w_ptr, out_ptr, N, K, stride_x, stride_we, stride_wn, stride_o,
                            XDIV: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        r = tl.program_id(0)
        nb = tl.program_id(1)
        e = tl.load(idx_ptr + r).to(tl.int64)
        offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        x_row = x_ptr + (r // XDIV).to(tl.int64) * stride_x
        w_base = w_ptr + e * stride_we + offs_n[:, None].to(tl.int64) * stride_wn
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            xv = tl.load(x_row + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            w = tl.load(w_base + offs_k[None, :], mask=mask_n[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            acc += tl.sum(w * xv[None, :], axis=1)
        tl.store(out_ptr + r.to(tl.int64) * stride_o + offs_n, acc, mask=mask_n)


if triton is not None:
    @triton.autotune(configs=[triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w, num_stages=st)
                              for bm, bn, bk, w, st in ((128, 128, 64, 8, 3), (128, 128, 64, 4, 3), (64, 128, 64, 4, 3),
                                                        (128, 128, 32, 8, 4), (128, 64, 64, 4, 3), (64, 256, 64, 8, 2),
                                                        (64, 64, 64, 4, 4))],
                     key=["N", "K"])
    @triton.heuristics({"EVEN_N": lambda a: a["N"] % a["BLOCK_N"] == 0,
                        "EVEN_K": lambda a: a["K"] % a["BLOCK_K"] == 0})
    @triton.jit
    def _grouped_mm_kernel(x_ptr, w_ptr, out_ptr, counts_ptr, rows_ptr, scale_ptr, N, K, E, stride_xm, stride_we,
                           stride_wn, stride_wk, stride_om, EP: tl.constexpr, HAS_ROWS: tl.constexpr,
                           HAS_SCALE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                           BLOCK_K: tl.constexpr, EVEN_N: tl.constexpr, EVEN_K: tl.constexpr):
        # The masks are per row only when N and K divide into tiles (every MoE shape here): a mask that
        # varies along a tile's contiguous dimension stops the loads from vectorising. A row block's N tiles
        # are consecutive programs, so its A tile is read from DRAM once and then hits L2 (GB10 has ~273 GB/s:
        # M-major order re-streamed the [M, K] rows once per N tile and was bandwidth bound).
        n_tiles = tl.cdiv(N, BLOCK_N)
        pid = tl.program_id(0) // n_tiles
        pid_n = tl.program_id(0) % n_tiles
        eo = tl.arange(0, EP)
        cnt = tl.load(counts_ptr + eo, mask=eo < E, other=0).to(tl.int32)
        tiles = (cnt + BLOCK_M - 1) // BLOCK_M
        e = tl.sum((tl.cumsum(tiles, 0) <= pid).to(tl.int32))   # the expert whose row tiles contain pid
        if e < E:
            before = eo < e
            m0 = tl.sum(tl.where(before, cnt, 0)) + (pid - tl.sum(tl.where(before, tiles, 0))) * BLOCK_M
            m_end = tl.sum(tl.where(eo <= e, cnt, 0))
            rm = m0 + tl.arange(0, BLOCK_M)
            rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
            rk = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_K), BLOCK_K), BLOCK_K)
            mask_m = rm < m_end
            mask_n = rn < N
            # HAS_ROWS: output row r reads input row rows[r] (a token's routed copy is never materialised)
            src = tl.load(rows_ptr + rm, mask=mask_m, other=0) if HAS_ROWS else rm
            a_ptr = x_ptr + src[:, None].to(tl.int64) * stride_xm + rk[None, :]
            b_ptr = w_ptr + e.to(tl.int64) * stride_we + rn[None, :].to(tl.int64) * stride_wn + rk[:, None] * stride_wk
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k0 in range(0, K, BLOCK_K):
                if EVEN_K:
                    a = tl.load(a_ptr, mask=mask_m[:, None], other=0.0)
                    b = tl.load(b_ptr) if EVEN_N else tl.load(b_ptr, mask=mask_n[None, :], other=0.0)
                else:
                    mask_k = rk < K - k0
                    a = tl.load(a_ptr, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                    b = tl.load(b_ptr, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
                acc = tl.dot(a, b, acc)
                a_ptr += BLOCK_K
                b_ptr += BLOCK_K * stride_wk
            if HAS_SCALE:
                acc = acc * tl.load(scale_ptr + rm, mask=mask_m, other=0.0).to(tl.float32)[:, None]
            o_ptr = out_ptr + rm[:, None].to(tl.int64) * stride_om + rn[None, :]
            if EVEN_N:
                tl.store(o_ptr, acc.to(out_ptr.dtype.element_ty), mask=mask_m[:, None])
            else:
                tl.store(o_ptr, acc.to(out_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


    @triton.jit
    def _slot_sum_kernel(y_ptr, inv_ptr, w_ptr, out_ptr, W, stride_y, stride_o, TOPK: tl.constexpr,
                         HAS_W: tl.constexpr, BLOCK_W: tl.constexpr):
        # out[t] = sum_j w[t, j] * y[inv[t*TOPK + j]], fp32 accumulation
        t = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
        mask = cols < W
        acc = tl.zeros((BLOCK_W,), dtype=tl.float32)
        for j in tl.static_range(TOPK):
            r = tl.load(inv_ptr + t * TOPK + j).to(tl.int64)
            v = tl.load(y_ptr + r * stride_y + cols, mask=mask, other=0.0).to(tl.float32)
            if HAS_W:
                v = v * tl.load(w_ptr + t * TOPK + j).to(tl.float32)
            acc += v
        tl.store(out_ptr + t.to(tl.int64) * stride_o + cols, acc.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _slot_dot_kernel(y_ptr, inv_ptr, g_ptr, out_ptr, W, stride_y, stride_g, TOPK: tl.constexpr,
                         BLOCK_W: tl.constexpr):
        # out[t, j] = <g[t], y[inv[t*TOPK + j]]>: the routing-weight gradient of _slot_sum_kernel
        t = tl.program_id(0)
        for j in tl.static_range(TOPK):
            r = tl.load(inv_ptr + t * TOPK + j).to(tl.int64)
            acc = tl.zeros((BLOCK_W,), dtype=tl.float32)
            for c0 in range(0, W, BLOCK_W):
                cols = c0 + tl.arange(0, BLOCK_W)
                mask = cols < W
                g = tl.load(g_ptr + t.to(tl.int64) * stride_g + cols, mask=mask, other=0.0).to(tl.float32)
                acc += g * tl.load(y_ptr + r * stride_y + cols, mask=mask, other=0.0).to(tl.float32)
            tl.store(out_ptr + t * TOPK + j, tl.sum(acc, 0).to(out_ptr.dtype.element_ty))


    @triton.jit
    def _swiglu_kernel(gu_ptr, out_ptr, n, H, BLOCK: tl.constexpr):
        # out[r, c] = silu(gu[r, c]) * gu[r, H + c] for gu [M, 2H] contiguous, rounded once
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        src = (offs // H) * (2 * H) + offs % H
        g = tl.load(gu_ptr + src, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(gu_ptr + src + H, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, (g * tl.sigmoid(g) * u).to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _swiglu_bwd_kernel(gu_ptr, grad_ptr, out_ptr, n, H, BLOCK: tl.constexpr):
        # out [M, 2H]: d/dgate and d/dup of _swiglu_kernel, written in place of torch's chunk + cat
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        src = (offs // H) * (2 * H) + offs % H
        g = tl.load(gu_ptr + src, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(gu_ptr + src + H, mask=mask, other=0.0).to(tl.float32)
        d = tl.load(grad_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        sig = tl.sigmoid(g)
        tl.store(out_ptr + src, (d * u * sig * (1.0 + g * (1.0 - sig))).to(out_ptr.dtype.element_ty), mask=mask)
        tl.store(out_ptr + src + H, (d * g * sig).to(out_ptr.dtype.element_ty), mask=mask)


def _grouped_mm_launch(x: torch.Tensor, w: torch.Tensor, counts: torch.Tensor, rows: torch.Tensor | None = None,
                       scale: torch.Tensor | None = None) -> torch.Tensor:
    """out[r] = scale[r] * w[e(r)] @ x[rows[r]] for output rows sorted by expert (rows, scale optional:
    identity, 1), x [*, K], w [E, N, K] (any strides), counts [E]."""
    x = x.contiguous()
    m = x.shape[0] if rows is None else rows.numel()
    k = x.shape[1]
    e, n, _ = w.shape
    out = torch.empty(m, n, device=x.device, dtype=x.dtype)
    if m == 0:
        return out
    grid = lambda meta: ((triton.cdiv(m, meta["BLOCK_M"]) + e) * triton.cdiv(n, meta["BLOCK_N"]),)
    _grouped_mm_kernel[grid](x, w, out, counts, x if rows is None else rows, x if scale is None else scale,
                             n, k, e, x.stride(0), w.stride(0), w.stride(1), w.stride(2), out.stride(0),
                             EP=triton.next_power_of_2(e), HAS_ROWS=rows is not None, HAS_SCALE=scale is not None)
    return out


def _slot_sum(y: torch.Tensor, inverse: torch.Tensor, weight: torch.Tensor | None, top_k: int) -> torch.Tensor:
    """out[t] = sum_j weight[t, j] * y[inverse[t*k + j]] (weight None: 1) in y's dtype."""
    t, w = inverse.numel() // top_k, y.shape[1]
    out = torch.empty(t, w, device=y.device, dtype=y.dtype)
    if t:
        block = min(1024, triton.next_power_of_2(w))
        _slot_sum_kernel[(t, triton.cdiv(w, block))](y, inverse, y if weight is None else weight, out, w,
                                                    y.stride(0), out.stride(0), TOPK=top_k,
                                                    HAS_W=weight is not None, BLOCK_W=block, num_warps=4)
    return out


class _GroupedMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, counts):
        ctx.save_for_backward(w, counts)
        return _grouped_mm_launch(x, w, counts)

    @staticmethod
    def backward(ctx, grad):
        w, counts = ctx.saved_tensors
        if ctx.needs_input_grad[1]:
            raise NotImplementedError("grouped_mm: expert weight gradients (the experts must be frozen)")
        return _grouped_mm_launch(grad, w.transpose(1, 2), counts), None, None


class _GatherGroupedMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, w, counts, rows, inverse, top_k):
        ctx.save_for_backward(w, counts, inverse)
        ctx.top_k = top_k
        return _grouped_mm_launch(h, w, counts, rows=rows)

    @staticmethod
    def backward(ctx, grad):
        w, counts, inverse = ctx.saved_tensors
        if ctx.needs_input_grad[1]:
            raise NotImplementedError("grouped_mm: expert weight gradients (the experts must be frozen)")
        # the gather's gradient: each token's k routed rows summed (no index_put accumulation)
        grad_rows = _grouped_mm_launch(grad.contiguous(), w.transpose(1, 2), counts)
        return _slot_sum(grad_rows, inverse, None, ctx.top_k), None, None, None, None, None


class _GroupedMMCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, counts, rows, order, inverse, weight):
        y = _grouped_mm_launch(x, w, counts)
        weight = weight.contiguous()
        # the expert outputs are kept only for the routing-weight gradient
        ctx.save_for_backward(w, counts, rows, inverse, weight.reshape(-1)[order],
                              y if ctx.needs_input_grad[6] else None)
        ctx.weight_shape = weight.shape
        return _slot_sum(y, inverse, weight, weight.shape[1])

    @staticmethod
    def backward(ctx, grad):
        w, counts, rows, inverse, scale, y = ctx.saved_tensors
        if ctx.needs_input_grad[1]:
            raise NotImplementedError("grouped_mm: expert weight gradients (the experts must be frozen)")
        grad = grad.contiguous()
        grad_weight = None
        if y is not None:
            t, k = ctx.weight_shape
            grad_weight = torch.empty(t, k, device=grad.device, dtype=scale.dtype)
            if t:
                _slot_dot_kernel[(t,)](y, inverse, grad, grad_weight, y.shape[1], y.stride(0), grad.stride(0),
                                       TOPK=k, BLOCK_W=min(1024, triton.next_power_of_2(y.shape[1])), num_warps=4)
        # a routed row's gradient: its token's output gradient times its routing weight
        grad_x = _grouped_mm_launch(grad, w.transpose(1, 2), counts, rows=rows, scale=scale)
        return grad_x, None, None, None, None, None, grad_weight


class _SwiGLU(torch.autograd.Function):
    BLOCK = 1024

    @staticmethod
    def forward(ctx, gate_up):
        gate_up = gate_up.contiguous()
        m, h = gate_up.shape[0], gate_up.shape[1] // 2
        out = torch.empty(m, h, device=gate_up.device, dtype=gate_up.dtype)
        if out.numel():
            _swiglu_kernel[(triton.cdiv(out.numel(), _SwiGLU.BLOCK),)](gate_up, out, out.numel(), h,
                                                                     BLOCK=_SwiGLU.BLOCK, num_warps=4)
        ctx.save_for_backward(gate_up)
        return out

    @staticmethod
    def backward(ctx, grad):
        (gate_up,) = ctx.saved_tensors
        grad = grad.contiguous()
        out = torch.empty_like(gate_up)
        if grad.numel():
            _swiglu_bwd_kernel[(triton.cdiv(grad.numel(), _SwiGLU.BLOCK),)](gate_up, grad, out, grad.numel(),
                                                                          gate_up.shape[1] // 2,
                                                                          BLOCK=_SwiGLU.BLOCK, num_warps=4)
        return out


def swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    """silu(gate) * up for gate_up = [gate | up] [M, 2H] -> [M, H], fp32 inside, one rounding."""
    return _SwiGLU.apply(gate_up)


def grouped_mm(x: torch.Tensor, w: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Expert-grouped x @ w[e].T: x [M, K] rows sorted by expert (counts[e] rows each, counts on the device),
    w [E, N, K] frozen expert weights in x's dtype. Returns [M, N] in x's dtype (fp32 accumulation); input
    gradients only."""
    return _GroupedMM.apply(x, w, counts)


def grouped_gather_mm(h: torch.Tensor, w: torch.Tensor, counts: torch.Tensor, rows: torch.Tensor,
                      inverse: torch.Tensor, top_k: int) -> torch.Tensor:
    """grouped_mm(h[rows], w, counts) without materialising h[rows]: h [T, K] tokens, rows [T*k] the token of
    each expert-sorted routed row, inverse [T*k] the sorted position of (token, slot) t*k + j. The input
    gradient sums each token's k rows."""
    return _GatherGroupedMM.apply(h, w, counts, rows, inverse, top_k)


def grouped_mm_combine(x: torch.Tensor, w: torch.Tensor, counts: torch.Tensor, rows: torch.Tensor,
                       order: torch.Tensor, inverse: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """out[t] = sum_j weight[t, j] * grouped_mm(x, w, counts)[inverse[t*k + j]] -> [T, N] in x's dtype, without
    permuting the [T*k, N] expert outputs (order: the (token, slot) index of each sorted row, rows = order // k).
    Gradients for x and the routing weight [T, k]."""
    return _GroupedMMCombine.apply(x, w, counts, rows, order, inverse, weight)


def expert_gemv(x: torch.Tensor, index: torch.Tensor, weight: torch.Tensor, xdiv: int = 1) -> torch.Tensor:
    """out[r] = weight[index[r]] @ x[r // xdiv] for every routed row r, in fp32.
    x [R / xdiv, K]; index [R] (expert ids, on the device); weight [E, N, K] (last dim contiguous).
    Returns [R, N] fp32."""
    if weight.stride(-1) != 1:
        raise ValueError("expert weights need a contiguous last dimension")
    x = x.contiguous()
    index = index.reshape(-1).contiguous()
    rows = index.numel()
    _, n, k = weight.shape
    out = torch.empty(rows, n, device=x.device, dtype=torch.float32)
    grid = (rows, triton.cdiv(n, STATE["block_n"]))
    _expert_gemv_kernel[grid](x, index, weight, out, n, k, x.stride(0), weight.stride(0), weight.stride(1),
                              out.stride(0), XDIV=xdiv, BLOCK_N=STATE["block_n"], BLOCK_K=STATE["block_k"],
                              num_warps=STATE["warps"])
    return out


def moe_decode(h: torch.Tensor, route_weight: torch.Tensor, index: torch.Tensor, gate_up: torch.Tensor,
               down: torch.Tensor) -> torch.Tensor:
    """The routed experts of SparseMoE for a few tokens: h [T, W], route_weight/index [T, k],
    gate_up [E, 2I, W], down [E, W, I]. Returns [T, W] in h's dtype."""
    t, k = index.shape
    dtype = gate_up.dtype
    gu = expert_gemv(h.to(dtype), index, gate_up, xdiv=k)            # [T*k, 2I] fp32
    gate, up = gu.to(dtype).chunk(2, dim=-1)
    act = F.silu(gate) * up                                          # rounded like the grouped path
    # The reference grouped GEMM rounds the down projection and routed bmm to
    # the expert dtype before adding the residual stream. Match both boundaries.
    y = expert_gemv(act, index, down).to(dtype)                       # [T*k, W]
    out = torch.bmm(route_weight.to(dtype).view(t, 1, k), y.view(t, k, -1))[:, 0]
    return out.to(h.dtype)
