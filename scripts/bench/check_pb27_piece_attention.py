"""Equivalence + speed check for piece_attention variants (runs inside the pb27-vllm container)."""
import os, sys, time
os.environ.setdefault("PB27_MEMORY", "64,64,0,0,256")
os.environ.setdefault("PB27_MEMORY_SCALE", "256:128")
sys.path.insert(0, "/tmp/newpkg")  # new memory.py
import torch
import torch.nn.functional as F
from pb27_vllm import memory as M
M.configure_from_env()

H, KV, D, W, block = 24, 4, 256, 16384, 64
G = H // KV
dev = torch.device("cuda")
scale = D ** -0.5

def ref_attend(q, keys, vals, allow):
    k = keys.repeat_interleave(G, dim=0); v = vals.repeat_interleave(G, dim=0)
    out = F.scaled_dot_product_attention(q.transpose(0, 1).unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                                         attn_mask=allow[None, None], scale=scale)
    return out[0].transpose(0, 1)

def gqa_batch_attend(q, keys, vals, allow):
    t = q.shape[0]
    qb = q.transpose(0, 1).reshape(KV, G, t, D)
    out = F.scaled_dot_product_attention(qb, keys.unsqueeze(1), vals.unsqueeze(1), attn_mask=allow[None, None],
                                         scale=scale, enable_gqa=True)
    return out.reshape(H, t, D).transpose(0, 1)

def timeit(fn, *a, n=10):
    fn(*a); torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(n):
        fn(*a)
    torch.cuda.synchronize(); return (time.perf_counter() - s) / n * 1000

ctx = 98304
arc = M.Archive.new(KV, D, 131072, block, torch.bfloat16, dev)
for s in range(0, ctx, 8192):
    pos = torch.arange(s, s + 8192, device=dev)
    arc.write(pos, torch.randn(8192, KV, D, device=dev, dtype=torch.bfloat16),
              torch.randn(8192, KV, D, device=dev, dtype=torch.bfloat16), block)
arc.prompt_len = ctx
arc.refresh_bounds(block)

for T in (17, 2048):
    pos = torch.arange(ctx - T, ctx, device=dev)
    q = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
    top = M.retrieve_blocks(arc, q, G, int(pos.min()) - W + 1, True, 0)
    new = M.piece_attention(arc, q, pos, top, G, W, scale)
    # rebuild the same keys/mask the function builds, for the reference and the batch variant
    lo = max(0, int(pos.min()) - W + 1); hi = int(pos.max()) + 1
    idx = (top.unsqueeze(-1) * block + torch.arange(block, device=dev)).view(KV, -1)
    sinks_k, sinks_v = arc.keys[:, :4], arc.values[:, :4]
    rk, rv = M._gather_tokens(arc.keys, idx), M._gather_tokens(arc.values, idx)
    keys = torch.cat([sinks_k, rk, arc.keys[:, lo:hi]], 1); vals = torch.cat([sinks_v, rv, arc.values[:, lo:hi]], 1)
    wpos = torch.arange(lo, hi, device=dev)
    allow_w = (wpos[None] <= pos[:, None]) & (wpos[None] >= pos[:, None] - W + 1)
    allow = torch.cat([torch.ones(T, 4 + rk.shape[1], dtype=torch.bool, device=dev), allow_w], 1)
    ref = ref_attend(q, keys, vals, allow)
    alt = gqa_batch_attend(q, keys, vals, allow)
    d_new = (new.float() - ref.float()).abs().max().item(); d_alt = (alt.float() - ref.float()).abs().max().item()
    t_ref = timeit(ref_attend, q, keys, vals, allow)
    t_new = timeit(lambda: M.piece_attention(arc, q, pos, top, G, W, scale))
    t_alt = timeit(gqa_batch_attend, q, keys, vals, allow)
    print(f"T={T:>5} keys={keys.shape[1]}  max|new-ref|={d_new:.2e} max|batchgqa-ref|={d_alt:.2e}  "
          f"ms: ref(attn only)={t_ref:.2f} new(full fn)={t_new:.2f} batchgqa(attn only)={t_alt:.2f}", flush=True)
