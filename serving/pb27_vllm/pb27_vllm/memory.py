"""Exact memory tier for the PB-27 vLLM serve (docs/PB27_MEMORY_SERVE.md), backend-agnostic v1.

Every hybrid layer archives the rotated keys/values of every token it sees, per request, indexed by position (so
rejected speculative tokens are simply overwritten), in 64-token blocks with Quest bounds (elementwise min/max of the
keys). For every query piece of <= 64 tokens whose window no longer covers the start of the context, the top-k
archived blocks older than the piece's window are retrieved by bound (exact rescoring for decode pieces) and the
piece's attention is recomputed exactly over [retrieved blocks + its sliding window] from the archive, replacing
vLLM's window-only output for those queries. vLLM's own attention still runs (it owns the paged cache and the
backend kernels), so v1 pays a second attention for the pieces that retrieve; the LSE-merge variant comes later.
Same math as efficiency/exact_memory.py; the archive lives on the GPU in v1.

Enable with PB27_MEMORY="blocks,block_size,decode_blocks,neighbors,rerank" (e.g. "64,64,0,0,256"); unset = off.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

log = logging.getLogger("vllm")

MEMORY: dict = {"enabled": False, "blocks": 64, "block_size": 64, "decode_blocks": 0, "neighbors": 0, "rerank": 0,
                "chunk": 64, "window": int(os.environ.get("PB27_WINDOW", "16384"))}
DEBUG = os.environ.get("PB27_MEMORY_DEBUG", "") == "1"
BATCH: dict = {"ib": None}                       # the runner's current InputBatch (req_ids, query_start_loc, ...)
ARCHIVES: dict[tuple[str, str], "Archive"] = {}  # (req_id, layer_name) -> Archive
STATS: dict = {"calls": 0, "fallback": 0, "retrievals": 0, "pieces": 0, "tokens_archived": 0}


def configure_from_env() -> None:
    spec = os.environ.get("PB27_MEMORY", "").strip()
    if not spec:
        return
    parts = [int(x) for x in spec.split(",")]
    for key, val in zip(["blocks", "block_size", "decode_blocks", "neighbors", "rerank"], parts):
        MEMORY[key] = val
    MEMORY["enabled"] = MEMORY["blocks"] > 0


@dataclass
class Archive:
    keys: torch.Tensor          # [KV, cap, D], valid up to `size`
    values: torch.Tensor
    kmax: torch.Tensor          # [KV, cap // block, D]
    kmin: torch.Tensor
    size: int = 0               # 1 + highest position written
    dirty_lo: int = field(default=1 << 60)  # block range whose bounds need refreshing
    dirty_hi: int = -1

    @staticmethod
    def new(kv: int, d: int, cap: int, block: int, dtype, device) -> "Archive":
        return Archive(torch.empty(kv, cap, d, dtype=dtype, device=device), torch.empty(kv, cap, d, dtype=dtype, device=device),
                       torch.empty(kv, cap // block, d, dtype=torch.float32, device=device),
                       torch.empty(kv, cap // block, d, dtype=torch.float32, device=device))

    def grow(self, need: int, block: int) -> None:
        cap = self.keys.shape[1]
        if need <= cap:
            return
        new_cap = cap
        while new_cap < need:
            new_cap *= 2
        kv, d = self.keys.shape[0], self.keys.shape[2]
        for name in ("keys", "values"):
            old = getattr(self, name)
            buf = torch.empty(kv, new_cap, d, dtype=old.dtype, device=old.device)
            buf[:, :cap] = old
            setattr(self, name, buf)
        for name in ("kmax", "kmin"):
            old = getattr(self, name)
            buf = torch.empty(kv, new_cap // block, d, dtype=old.dtype, device=old.device)
            buf[:, : cap // block] = old
            setattr(self, name, buf)

    def write(self, pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block: int) -> None:
        """pos [N] long, k/v [N, KV, D] -> store at those positions (overwriting re-verified speculative slots)."""
        hi = int(pos.max()) + 1
        self.grow(hi, block)
        self.keys[:, pos] = k.transpose(0, 1).to(self.keys.dtype)
        self.values[:, pos] = v.transpose(0, 1).to(self.values.dtype)
        self.size = max(self.size, hi)
        lo_b, hi_b = int(pos.min()) // block, (hi - 1) // block
        self.dirty_lo, self.dirty_hi = min(self.dirty_lo, lo_b), max(self.dirty_hi, hi_b)
        STATS["tokens_archived"] += int(pos.numel())

    def refresh_bounds(self, block: int) -> None:
        if self.dirty_hi < self.dirty_lo:
            return
        lo, hi = self.dirty_lo * block, min(self.size, (self.dirty_hi + 1) * block)
        seg = self.keys[:, lo:hi].float()
        pad = (-seg.shape[1]) % block
        if pad:
            seg = torch.cat((seg, seg[:, -1:].expand(-1, pad, -1)), dim=1)
        blocks = seg.view(seg.shape[0], -1, block, seg.shape[2])
        self.kmax[:, self.dirty_lo: self.dirty_lo + blocks.shape[1]] = blocks.amax(2)
        self.kmin[:, self.dirty_lo: self.dirty_lo + blocks.shape[1]] = blocks.amin(2)
        self.dirty_lo, self.dirty_hi = 1 << 60, -1


def drop_request(req_id: str) -> None:
    for key in [k for k in ARCHIVES if k[0] == req_id]:
        del ARCHIVES[key]


def retrieve_blocks(arc: Archive, q: torch.Tensor, groups: int, limit: int, rerank: bool) -> torch.Tensor | None:
    """Indices of the top blocks among archived positions [0, limit) for query piece q [T, H, D] -> [KV, k] or None."""
    block = MEMORY["block_size"]
    full = limit // block
    if full <= 0:
        return None
    arc.refresh_bounds(block)
    t, h, d = q.shape
    kv = h // groups
    qg = q.float().permute(1, 0, 2).reshape(kv, groups * t, d)                         # [KV, G*T, D]
    pos, neg = qg.clamp(min=0), qg.clamp(max=0)
    score = (pos @ arc.kmax[:, :full].transpose(-1, -2) + neg @ arc.kmin[:, :full].transpose(-1, -2)).amax(1)
    budget = MEMORY["decode_blocks"] if (rerank and MEMORY["decode_blocks"]) else MEMORY["blocks"]
    k = min(budget, full)
    if rerank and MEMORY["rerank"] > k:
        c = min(MEMORY["rerank"], full)
        cand = score.topk(c, dim=-1).indices                                              # [KV, c]
        idx = (cand.unsqueeze(-1) * block + torch.arange(block, device=q.device)).view(kv, -1)
        ck = arc.keys.gather(1, idx.unsqueeze(-1).expand(-1, -1, d)).float()
        exact = torch.einsum("gqd,gnd->gqn", qg, ck).amax(1).view(kv, c, block).amax(-1)
        top = cand.gather(-1, exact.topk(k, dim=-1).indices)
    else:
        top = score.topk(k, dim=-1).indices
    if MEMORY["neighbors"]:
        top = torch.cat([top] + [(top + j).clamp(max=full - 1) for j in range(1, MEMORY["neighbors"] + 1)], dim=-1)
    STATS["retrievals"] += 1
    return top.sort(-1).values


def piece_attention(arc: Archive, q: torch.Tensor, q_pos: torch.Tensor, top: torch.Tensor | None, groups: int,
                    window: int, scale: float) -> torch.Tensor:
    """Exact attention of piece q [T, H, D] at positions q_pos over its sliding window (from the archive) plus the
    retrieved blocks `top` [KV, k]; returns [T, H, D]."""
    block = MEMORY["block_size"]
    t, h, d = q.shape
    kv = h // groups
    lo = max(0, int(q_pos.min()) - window + 1)
    hi = int(q_pos.max()) + 1
    wk, wv = arc.keys[:, lo:hi], arc.values[:, lo:hi]                                 # [KV, Wn, D]
    wpos = torch.arange(lo, hi, device=q.device)
    allow = (wpos[None, :] <= q_pos[:, None]) & (wpos[None, :] >= q_pos[:, None] - window + 1)   # [T, Wn]
    if top is not None:
        idx = (top.unsqueeze(-1) * block + torch.arange(block, device=q.device)).view(kv, -1)
        gather = idx.unsqueeze(-1).expand(-1, -1, d)
        rk, rv = arc.keys.gather(1, gather), arc.values.gather(1, gather)              # [KV, R, D]
        keys = torch.cat((rk, wk), dim=1)
        vals = torch.cat((rv, wv), dim=1)
        allow = torch.cat((torch.ones(t, rk.shape[1], dtype=torch.bool, device=q.device), allow), dim=1)
    else:
        keys, vals = wk, wv
    keys = keys.repeat_interleave(groups, dim=0)                                        # [H, N, D]
    vals = vals.repeat_interleave(groups, dim=0)
    out = F.scaled_dot_product_attention(q.transpose(0, 1).unsqueeze(0), keys.unsqueeze(0), vals.unsqueeze(0),
                                         attn_mask=allow[None, None], scale=scale)
    return out[0].transpose(0, 1).to(q.dtype)


def install_runner_hooks() -> None:
    """Expose the runner's current InputBatch and free archives of finished requests (V2 GPU runner)."""
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.states import RequestState

    if getattr(GPUModelRunner, "_pb27_hooked", False):
        return
    orig_prepare = GPUModelRunner.prepare_inputs
    orig_remove = RequestState.remove_request

    def prepare_inputs(self, *args, **kwargs):
        ib = orig_prepare(self, *args, **kwargs)
        BATCH["ib"] = ib
        return ib

    def remove_request(self, req_id, *args, **kwargs):
        drop_request(req_id)
        return orig_remove(self, req_id, *args, **kwargs)

    GPUModelRunner.prepare_inputs = prepare_inputs
    RequestState.remove_request = remove_request
    GPUModelRunner._pb27_hooked = True
    log.info("pb27 memory tier: runner hooks installed (pid %d), config %s", os.getpid(), MEMORY)


def windowed_attention_with_memory(attn_module, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                   positions: torch.Tensor) -> torch.Tensor:
    """vLLM's window attention, then exact recomputation over [retrieved + window] for the pieces that retrieve."""
    out = attn_module(q, k, v)
    ib = BATCH["ib"]
    STATS["calls"] += 1
    if DEBUG and STATS["calls"] % 500 == 1:
        log.info("pb27 memory: call %d ib=%s archives=%d stats=%s", STATS["calls"], ib is not None, len(ARCHIVES), STATS)
    if ib is None or not ib.req_ids:
        STATS["fallback"] += 1
        return out
    impl = attn_module.impl
    H, KV, D = impl.num_heads, impl.num_kv_heads, impl.head_size
    groups = H // KV
    window = getattr(attn_module, "sliding_window", None) or MEMORY["window"]
    T = ib.num_tokens
    qh, kh, vh = q[:T].view(T, H, D), k[:T].view(T, KV, D), v[:T].view(T, KV, D)
    pos = positions[0, :T] if positions.dim() == 2 else positions[:T]
    qsl = ib.query_start_loc_np
    block, piece = MEMORY["block_size"], MEMORY["chunk"]
    layer = attn_module.layer_name
    out_h = out[:T].view(T, H, D)
    for i, req_id in enumerate(ib.req_ids):
        s, e = int(qsl[i]), int(qsl[i + 1])
        if e <= s:
            continue
        key = (req_id, layer)
        arc = ARCHIVES.get(key)
        if arc is None:
            arc = ARCHIVES[key] = Archive.new(KV, D, max(block * 64, 1 << 15), block, kh.dtype, kh.device)
        arc.write(pos[s:e], kh[s:e], vh[s:e], block)
        if int(pos[e - 1]) + 1 <= window:
            continue                                      # the whole context still fits the window
        rerank = (e - s) <= piece
        for p0 in range(s, e, piece):
            p1 = min(e, p0 + piece)
            q_pos = pos[p0:p1]
            limit = int(q_pos.min()) - window + 1         # archived positions older than every query's window
            top = retrieve_blocks(arc, qh[p0:p1], groups, limit, rerank) if limit >= block else None
            if top is None:
                continue
            out_h[p0:p1] = piece_attention(arc, qh[p0:p1], q_pos, top, groups, window, impl.scale)
            STATS["pieces"] += 1
    return out
