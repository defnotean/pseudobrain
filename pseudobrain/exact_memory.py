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

EXACT_MEMORY = {"enabled": False, "blocks": 64, "block_size": 64, "decode_blocks": 0, "neighbors": 0, "rerank": 0,
                "keys_like_values": False}  # decode_blocks: budget when the
# query chunk is short (<= 64 tokens, i.e. decoding); 0 = same as blocks. keys_like_values: archive keys in the values'
# dtype (bf16 under autocast, as the serving KV cache stores them); k_norm's RMSNorm runs in fp32 under autocast, so
# the keys otherwise arrive fp32 and a 131K-token archive of the 16 window layers takes 12.9 GB instead of 8.6 GB.


@dataclass
class Archive:
    """Evicted rotated keys/values [B,KV,N,hd] and per-block key bounds [B,KV,nblocks,hd]."""
    keys: torch.Tensor
    values: torch.Tensor
    kmax: torch.Tensor
    kmin: torch.Tensor

    @property
    def length(self) -> int:
        return self.keys.shape[2]


def _bounds(keys: torch.Tensor, block: int) -> tuple[torch.Tensor, torch.Tensor]:
    b, h, n, d = keys.shape
    pad = (-n) % block
    if pad:  # the trailing partial block: pad with its own last key so min/max are unaffected
        keys = torch.cat((keys, keys[:, :, -1:].expand(b, h, pad, d)), dim=2)
    blocks = keys.view(b, h, -1, block, d)
    return blocks.amax(3), blocks.amin(3)


def archive_append(archive: Archive | None, keys: torch.Tensor, values: torch.Tensor) -> Archive | None:
    """Append evicted keys/values (in position order) and refresh the bounds of affected blocks."""
    if keys.shape[2] == 0:
        return archive
    block = EXACT_MEMORY["block_size"]
    if EXACT_MEMORY["keys_like_values"]:
        keys = keys.to(values.dtype)
    if archive is None:
        kmax, kmin = _bounds(keys, block)
        return Archive(keys, values, kmax, kmin)
    full = (archive.length // block) * block          # complete blocks are final; redo the partial tail
    new_k = torch.cat((archive.keys, keys), dim=2)
    new_v = torch.cat((archive.values, values), dim=2)
    tmax, tmin = _bounds(new_k[:, :, full:], block)
    kmax = torch.cat((archive.kmax[:, :, : full // block], tmax), dim=2)
    kmin = torch.cat((archive.kmin[:, :, : full // block], tmin), dim=2)
    return Archive(new_k, new_v, kmax, kmin)


def retrieve(archive: Archive | None, q: torch.Tensor, groups: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Top blocks per key head for query chunk q [B,H,T,hd]: (keys, values) [B,KV,R,hd] or None."""
    if archive is None or archive.length == 0:
        return None
    block = EXACT_MEMORY["block_size"]
    b, h, t, d = q.shape
    kv = h // groups
    qg = q.float().view(b, kv, groups * t, d)
    pos, neg = qg.clamp(min=0), qg.clamp(max=0)
    # upper bound of q.k over each block: [B,KV,G*T,nblocks] -> max over the chunk's queries
    bound = pos @ archive.kmax.float().transpose(-1, -2) + neg @ archive.kmin.float().transpose(-1, -2)
    score = bound.amax(2)
    full = archive.length // block                                        # complete blocks are candidates;
    tail = archive.length - full * block                                  # the partial tail is always kept
    parts_k, parts_v = [], []
    if full:
        budget = EXACT_MEMORY["decode_blocks"] if (t <= 64 and EXACT_MEMORY["decode_blocks"]) else EXACT_MEMORY["blocks"]
        k = min(budget, full)
        if t <= 64 and EXACT_MEMORY.get("rerank", 0) > k:  # decode: exact rescoring of the bound's top candidates
            c = min(EXACT_MEMORY["rerank"], full)
            cand = score[..., :full].topk(c, dim=-1).indices                       # [B,KV,c]
            idx = (cand.unsqueeze(-1) * block + torch.arange(block, device=q.device)).view(b, kv, -1)
            ck = archive.keys.gather(2, idx.unsqueeze(-1).expand(-1, -1, -1, d)).float()   # [B,KV,c*block,d]
            exact = torch.einsum("bgqd,bgnd->bgqn", qg, ck).amax(2).view(b, kv, c, block).amax(-1)  # [B,KV,c]
            top = cand.gather(-1, exact.topk(k, dim=-1).indices)
        else:
            top = score[..., :full].topk(k, dim=-1).indices
        if EXACT_MEMORY["neighbors"]:  # also fetch the blocks right after each hit (a cue phrase precedes its content)
            nb = [top] + [(top + j).clamp(max=full - 1) for j in range(1, EXACT_MEMORY["neighbors"] + 1)]
            top = torch.cat(nb, dim=-1)
        top = top.sort(-1).values                                         # keep position order (duplicates harmless)
        idx = (top.unsqueeze(-1) * block + torch.arange(block, device=q.device)).view(b, kv, -1)
        gather = idx.unsqueeze(-1).expand(-1, -1, -1, d)
        parts_k.append(archive.keys.gather(2, gather))
        parts_v.append(archive.values.gather(2, gather))
    if tail:
        parts_k.append(archive.keys[:, :, full * block:])
        parts_v.append(archive.values[:, :, full * block:])
    return torch.cat(parts_k, dim=2), torch.cat(parts_v, dim=2)
