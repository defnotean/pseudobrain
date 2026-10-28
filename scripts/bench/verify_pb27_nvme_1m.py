"""CPU verification of the flat working set: one real-size layer (4 KV heads, D=256) filled to 1M tokens in the
file-backed archive. Reports resident memory, MemAvailable and cold/warm retrieval+attention latency.
Run on Spark outside the container: PB27_MEMORY=64,64,0,0,256 PB27_MEMORY_OFFLOAD=nvme python verify_pb27_nvme_1m.py"""
import mmap, os, sys, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "serving", "pb27_vllm"))
import pb27_vllm.memory as M
M.configure_from_env()
maps = []
_orig = mmap.mmap
M.mmap = type("m", (), {"mmap": staticmethod(lambda *a, **k: (maps.append(_orig(*a, **k)) or maps[-1]))})


def meminfo():
    d = {l.split(":")[0]: int(l.split()[1]) // 1024 for l in open("/proc/meminfo")}
    rss = next(int(l.split()[1]) // 1024 for l in open("/proc/self/status") if l.startswith("VmRSS"))
    return f"rss={rss}MB avail={d['MemAvailable']}MB cached={d['Cached']}MB dirty={d['Dirty']}MB"


N, KV, D, G, W = int(os.environ.get("N", 1_048_576)), 4, 256, 6, 16384
block = M.MEMORY["block_size"]
g = torch.Generator().manual_seed(1)
arc = M.Archive.new(KV, D, 1 << 10, block, torch.bfloat16, torch.device("cpu"))
assert arc.mapped
print("start", meminfo(), flush=True)
t0, pos = time.time(), 0
while pos < N:
    t = min(2048, N - pos)
    arc.write(torch.arange(pos, pos + t), torch.randn(t, KV, D, generator=g).bfloat16(),
              torch.randn(t, KV, D, generator=g).bfloat16(), block)
    pos += t
    if pos % (1 << 18) == 0:
        print(f"wrote {pos}", meminfo(), f"{time.time()-t0:.0f}s", flush=True)
arc.refresh_bounds(block)
for m in maps:
    m.flush()
    m.madvise(21)          # MADV_PAGEOUT: evict the mapped pages (cold start)
time.sleep(5)
print("after pageout", meminfo(), flush=True)
q = torch.randn(17, KV * G, D, generator=g).bfloat16()
q_pos = torch.arange(N - 17, N)
for label in ("cold", "warm", "warm"):
    t0 = time.time()
    top = M.retrieve_blocks(arc, q, G, N - W, rerank=True)
    t1 = time.time()
    out = M.piece_attention(arc, q, q_pos, top, G, W, D ** -0.5)
    print(f"{label}: retrieve {t1-t0:.3f}s attend {time.time()-t1:.3f}s top={tuple(top.shape)}", meminfo(), flush=True)
print("DONE")
