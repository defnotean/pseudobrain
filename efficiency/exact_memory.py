"""Exact memory tier for bounded-window attention (PB-27 S4.6).

A windowed attention layer keeps its sink positions and the last window-1 keys/values. With the exact memory
tier enabled, the keys/values that leave the window are not dropped: they go into an archive of fixed-size
blocks (rotated keys, so positions are preserved), each block summarized by the elementwise min and max of its
keys (Quest, Tang et al. 2024). For every query chunk, each archived block gets an upper bound on its attention
logit, sum_d max(q_d * kmax_d, q_d * kmin_d), maximized over the chunk's queries and the query heads that share
the key head; the top `blocks` blocks per key head are gathered and attended exactly together with the window.
Retrieved keys are all older than the window, so every query in the chunk may see them.

The archive grows with the context (host memory in a full deployment; here it stays on the device), while the
attention working set per step is bounded: sinks + window + blocks * block_size.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

EXACT_MEMORY = {"enabled": False, "blocks": 64, "block_size": 64, "decode_blocks": 64, "neighbors": 0, "rerank": 0,
                "decode_rerank": 0, "superblock": 64, "rope_horizon": 262144, "rope_arc_span": 16384,
                "rope_theta": 10000000.0, "keys_like_values": False}  # decode_blocks: O(1) budget when the
# query chunk is short (<= 64 tokens, i.e. decoding); 0 = same as blocks. superblock: 2-level hierarchical Quest
# super-block size (64 blocks = 4,096 tokens per super-block, 256 super-blocks at 1M). keys_like_values: archive keys
# in the values' dtype (bf16 under autocast, as the serving KV cache stores them).


def _reposition_rope_blocks(
    rk: torch.Tensor, blk_idx: torch.Tensor, block: int, ref_pos: int, span: int | None = None
) -> torch.Tensor:
    horizon = int(EXACT_MEMORY.get("rope_horizon", 262144) or 0)
    arc_span = int(EXACT_MEMORY.get("rope_arc_span", 16384) or 16384) if span is None else span
    theta = float(EXACT_MEMORY.get("rope_theta", 10000000.0) or 10000000.0)
    if horizon <= 0 or ref_pos <= horizon or rk.numel() == 0:
        return rk
    b, kv, total_t, d = rk.shape
    n_blk = blk_idx.shape[2]
    if n_blk == 0 or total_t != n_blk * block:
        return rk
    d_rot = d // 4
    half = d_rot // 2
    idx = blk_idx.to(device=rk.device, dtype=torch.long).clone()
    if n_blk > 1:
        is_succ = torch.zeros_like(idx, dtype=torch.bool)
        is_succ[:, :, 1:] = idx[:, :, 1:] == (idx[:, :, :-1] + 1)
        step = 1
        while step < n_blk:
            prev_idx = torch.roll(idx, shifts=step, dims=2)
            prev_succ = torch.roll(is_succ, shifts=step, dims=2)
            prev_succ[:, :, :step] = False
            idx = torch.where(is_succ, prev_idx, idx)
            is_succ = is_succ & prev_succ
            if not bool(is_succ.any()):
                break
            step *= 2
    orig_pos = (idx * block).to(torch.float32)
    scale = 1.0 - min(1.0, float(arc_span) / float(max(1, ref_pos)))
    max_shift = (float(max(0, ref_pos - 2 * block)) - orig_pos).clamp(min=0.0)
    delta = torch.minimum(torch.round((float(ref_pos) - orig_pos).clamp(min=0.0) * scale), max_shift)
    inv_freq = 1.0 / (theta ** (torch.arange(0, d_rot, 2, device=rk.device, dtype=torch.float32) / float(d_rot)))
    angles = delta.unsqueeze(-1) * inv_freq
    cos, sin = angles.cos().unsqueeze(3), angles.sin().unsqueeze(3)
    rk_v = rk.view(b, kv, n_blk, block, d)
    x1, x2 = rk_v[..., :half].float(), rk_v[..., half:d_rot].float()
    out = rk.clone().view(b, kv, n_blk, block, d)
    out[..., :half] = (x1 * cos - x2 * sin).to(rk.dtype)
    out[..., half:d_rot] = (x2 * cos + x1 * sin).to(rk.dtype)
    return out.view(b, kv, total_t, d)


@dataclass
class Archive:
    """Evicted rotated keys/values [B,KV,N,hd], per-block bounds [B,KV,nblocks,hd], and super-block bounds [B,KV,nsb,hd]."""
    keys: torch.Tensor
    values: torch.Tensor
    kmax: torch.Tensor
    kmin: torch.Tensor
    smax: torch.Tensor | None = None
    smin: torch.Tensor | None = None
    prompt_top: torch.Tensor | None = None

    @property
    def length(self) -> int:
        return self.keys.shape[2]


def _bounds(keys: torch.Tensor, block: int) -> tuple[torch.Tensor, torch.Tensor]:
    b, h, n, d = keys.shape
    pad = (-n) % block
    if pad:  # the trailing partial block: pad with its own last key so min/max are unaffected
        keys = torch.cat((keys, keys[:, :, -1:].expand(b, h, pad, d)), dim=2)
    blocks = keys.view(b, h, -1, block, d)
    b_dtype = torch.bfloat16 if (keys.is_cuda and keys.dtype in (torch.bfloat16, torch.float16)) else keys.dtype
    return blocks.amax(3).to(b_dtype), blocks.amin(3).to(b_dtype)


def _superbounds(kmax: torch.Tensor, kmin: torch.Tensor, sblk: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if sblk <= 1 or kmax.shape[2] == 0:
        return None, None
    b, h, nb, d = kmax.shape
    pad = (-nb) % sblk
    if pad:
        kmax_p = torch.cat((kmax, kmax[:, :, -1:].expand(b, h, pad, d)), dim=2)
        kmin_p = torch.cat((kmin, kmin[:, :, -1:].expand(b, h, pad, d)), dim=2)
    else:
        kmax_p, kmin_p = kmax, kmin
    return kmax_p.view(b, h, -1, sblk, d).amax(3), kmin_p.view(b, h, -1, sblk, d).amin(3)


def archive_append(archive: Archive | None, keys: torch.Tensor, values: torch.Tensor) -> Archive | None:
    """Append evicted keys/values (in position order) and refresh the bounds of affected blocks and super-blocks."""
    if keys.shape[2] == 0:
        return archive
    block = EXACT_MEMORY["block_size"]
    sblk = int(EXACT_MEMORY.get("superblock", 64) or 0)
    if EXACT_MEMORY["keys_like_values"]:
        keys = keys.to(values.dtype)
    if archive is None:
        kmax, kmin = _bounds(keys, block)
        smax, smin = _superbounds(kmax, kmin, sblk)
        return Archive(keys, values, kmax, kmin, smax, smin)
    full = (archive.length // block) * block          # complete blocks are final; redo the partial tail
    new_k = torch.cat((archive.keys, keys), dim=2)
    new_v = torch.cat((archive.values, values), dim=2)
    tmax, tmin = _bounds(new_k[:, :, full:], block)
    kmax = torch.cat((archive.kmax[:, :, : full // block], tmax), dim=2)
    kmin = torch.cat((archive.kmin[:, :, : full // block], tmin), dim=2)
    smax, smin = _superbounds(kmax, kmin, sblk)
    return Archive(new_k, new_v, kmax, kmin, smax, smin, archive.prompt_top)


def retrieve(archive: Archive | None, q: torch.Tensor, groups: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Top blocks per key head for query chunk q [B,H,T,hd]: (keys, values) [B,KV,R,hd] or None."""
    if archive is None or archive.length == 0:
        return None
    block = EXACT_MEMORY["block_size"]
    sblk = int(EXACT_MEMORY.get("superblock", 64) or 0)
    b, h, t, d = q.shape
    kv = h // groups
    s_dtype = q.dtype if (q.is_cuda and q.dtype in (torch.bfloat16, torch.float16)) else torch.float32
    qg_s = q.to(s_dtype).view(b, kv, groups * t, d)
    pos, neg = qg_s.clamp(min=0), qg_s.clamp(max=0)
    full = archive.length // block                                        # complete blocks are candidates;
    tail = archive.length - full * block                                  # the partial tail is always kept
    parts_k, parts_v = [], []
    if full:
        is_decode = t <= 64
        budget = EXACT_MEMORY["decode_blocks"] if (is_decode and EXACT_MEMORY["decode_blocks"]) else EXACT_MEMORY["blocks"]
        k = min(budget, full)
        rerank_cfg = int(EXACT_MEMORY.get("decode_rerank", 0) if is_decode else EXACT_MEMORY.get("rerank", 0))
        if is_decode and rerank_cfg <= k and sblk > 1 and archive.smax is not None and archive.smin is not None and full > sblk * 16:
            n_sb = (full + sblk - 1) // sblk
            smax_t = archive.smax[:, :, :n_sb].to(s_dtype).transpose(-1, -2)
            smin_t = archive.smin[:, :, :n_sb].to(s_dtype).transpose(-1, -2)
            sb_score = (pos @ smax_t + neg @ smin_t).amax(2).float()
            m_sb = min(max(16, (2 * k + sblk - 1) // sblk), n_sb)
            top_sb = sb_score.topk(m_sb, dim=-1).indices
            sb_off = torch.arange(sblk, device=q.device, dtype=top_sb.dtype)
            fine_raw = (top_sb.unsqueeze(-1) * sblk + sb_off).reshape(b, kv, m_sb * sblk)
            fine_valid = fine_raw < full
            fine_idx = fine_raw.clamp(max=full - 1)
            idx_d = fine_idx.unsqueeze(-1).expand(b, kv, m_sb * sblk, d)
            kmax_sub = archive.kmax[:, :, :full].gather(2, idx_d).to(s_dtype).transpose(-1, -2)
            kmin_sub = archive.kmin[:, :, :full].gather(2, idx_d).to(s_dtype).transpose(-1, -2)
            sub_score = (pos @ kmax_sub + neg @ kmin_sub).amax(2).float().masked_fill(~fine_valid, float("-inf"))
            top = fine_idx.gather(-1, sub_score.topk(min(k, m_sb * sblk), dim=-1).indices)
        else:
            bound = pos @ archive.kmax[:, :, :full].to(s_dtype).transpose(-1, -2) + neg @ archive.kmin[:, :, :full].to(s_dtype).transpose(-1, -2)
            score = bound.amax(2).float()
            if rerank_cfg > k:
                qg = q.float().view(b, kv, groups * t, d)
                c = min(rerank_cfg, full)
                cand = score.topk(c, dim=-1).indices                               # [B,KV,c]
                idx = (cand.unsqueeze(-1) * block + torch.arange(block, device=q.device)).view(b, kv, -1)
                ck = archive.keys.gather(2, idx.unsqueeze(-1).expand(-1, -1, -1, d)).float()   # [B,KV,c*block,d]
                ck = _reposition_rope_blocks(ck, cand, block, archive.length, span=0)
                exact = torch.einsum("bgqd,bgnd->bgqn", qg, ck).amax(2).view(b, kv, c, block).amax(-1)  # [B,KV,c]
                top = cand.gather(-1, exact.topk(k, dim=-1).indices)
            else:
                top = score.topk(k, dim=-1).indices
        if not is_decode:
            archive.prompt_top = top
        elif archive.prompt_top is not None:
            p_prev = archive.prompt_top.clamp(max=full - 1)
            top = torch.cat((p_prev, top), dim=-1)
        if EXACT_MEMORY["neighbors"]:  # also fetch the blocks right after each hit (a cue phrase precedes its content)
            nb = [top] + [(top + j).clamp(max=full - 1) for j in range(1, EXACT_MEMORY["neighbors"] + 1)]
            top = torch.cat(nb, dim=-1)
        top = top.sort(-1).values                                         # keep position order (duplicates harmless)
        idx = (top.unsqueeze(-1) * block + torch.arange(block, device=q.device)).view(b, kv, -1)
        gather = idx.unsqueeze(-1).expand(-1, -1, -1, d)
        rk = _reposition_rope_blocks(archive.keys.gather(2, gather), top, block, archive.length)
        parts_k.append(rk)
        parts_v.append(archive.values.gather(2, gather))
    if tail:
        parts_k.append(archive.keys[:, :, full * block:])
        parts_v.append(archive.values[:, :, full * block:])
    return torch.cat(parts_k, dim=2), torch.cat(parts_v, dim=2)
