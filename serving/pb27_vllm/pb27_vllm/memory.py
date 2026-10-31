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
import mmap
import os
import tempfile
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

try:
    from vllm.vllm_flash_attn import _vllm_fa2_C  # noqa: F401
    HAS_FA2 = hasattr(torch.ops, "_vllm_fa2_C") and hasattr(torch.ops._vllm_fa2_C, "varlen_fwd")
except Exception:
    HAS_FA2 = False

log = logging.getLogger("vllm")

MEMORY: dict = globals().get("MEMORY", {"enabled": True, "blocks": 256, "block_size": 64, "decode_blocks": 256, "neighbors": 0, "rerank": 512,
                "chunk": 64, "window": int(os.environ.get("PB27_WINDOW", "16384"))})
DEBUG = os.environ.get("PB27_MEMORY_DEBUG", "") == "1"
NORETRIEVE = os.environ.get("PB27_MEMORY_NORETRIEVE", "") == "1"   # diagnostic: recompute pieces from the archive without retrieval
SINKS = int(os.environ.get("PB27_MEMORY_SINKS", "4") or 0)         # PB-27 was trained with 4 always-visible sink positions; vLLM
                                                                   # drops them, the recomputed pieces restore positions 0..SINKS-1
PREFILL = os.environ.get("PB27_MEMORY_PREFILL", "pt")             # prefill retrieval: "1" every piece, "0" none, "last"
                                                                   # or "lastN" only the final N pieces, "pt" final question piece
TAIL = int(PREFILL[4:] or 1) if PREFILL.startswith("last") else 1
PT_CHUNK = int(os.environ.get("PB27_MEMORY_PT_CHUNK", "2048") or 2048)  # prefill chunk size of the PyTorch tier
PREFILL_BLOCKS = int(os.environ.get("PB27_MEMORY_PREFILL_BLOCKS", "0") or 0)  # block budget for prefill pieces (0 = as decode)
ORACLE = os.environ.get("PB27_MEMORY_ORACLE", "") == "1"         # diagnostic: select blocks by exact attention mass
FAITHFUL = os.environ.get("PB27_MEMORY_FAITHFUL", "1") != "0"    # mirror efficiency/exact_memory.py: archive starts
ORIGIN = SINKS if FAITHFUL else 0                                  # after the sinks (blocks at 4+64j) and keeps tail
QWIN = os.environ.get("PB27_MEMORY_QWIN", "") == "1"             # decode retrieval uses trailing verified query window
DSTEP = os.environ.get("PB27_MEMORY_DSTEP", "") == "1"           # step-1 decode uses `blocks`; step-2+ decode uses `decode_blocks`
SCALE = [2048, 384, 4096, 512, 8192, 1024]
ROPE_HORIZON = 262144
ROPE_ARC_SPAN = 98304
ROPE_THETA = float(os.environ.get("PB27_MEMORY_ROPE_THETA", "10000000.0") or 10000000.0)
OFFLOAD = os.environ.get("PB27_MEMORY_OFFLOAD", "nvme").strip().lower()  # "", "cpu", "fp8", "cpu_fp8", or "nvme"
# "nvme": fp8/bf16 K/V live in an unlinked sparse file mapped into memory (page cache is evictable, so resident memory
# stays flat while disk holds the history); block bounds stay resident. NVME_CAP is the max tokens per request.
NVME_DIR = os.environ.get("PB27_MEMORY_NVME_DIR", "") or None
NVME_CAP = int(os.environ.get("PB27_MEMORY_NVME_CAP", "1114112") or 1114112)
# "bf16" keeps the file-backed archive in the model dtype (2x disk/page cache, no e4m3 rounding of the recomputed
# window and retrieved blocks); "fp8" is the original e4m3 archive.
NVME_DTYPE = os.environ.get("PB27_MEMORY_NVME_DTYPE", "bf16").strip().lower()
# Per-query retrieval for multi-token decode pieces (speculative verify): each query scores blocks over its own G
# heads and attends to its own top-k, as non-speculative decode would, instead of one block set shared by the
# verified token and every draft. PERTOKEN_CAND scales the rerank candidate pool; PERTOKEN_UNION caps the gathered
# union at UNION x k blocks.
PERTOKEN = os.environ.get("PB27_MEMORY_PERTOKEN", "1") != "0"
PERTOKEN_CAND = int(os.environ.get("PB27_MEMORY_PERTOKEN_CAND", "4") or 4)
PERTOKEN_UNION = int(os.environ.get("PB27_MEMORY_PERTOKEN_UNION", "2") or 2)
# LEAD: the query whose output is always emitted (the verified token of a verify piece, the last query of the final
# prefill piece) keeps its full own top-k from its own rerank pool; the other queries fill the remaining union slots.
# OPEN: every query of the piece attends to the whole union instead of only its own blocks (same cost as the masked
# form; a query never sees fewer blocks than the lead's top-k). PREFILL_UNION > 0: the final prefill piece also
# retrieves per query, with the union capped at PREFILL_UNION x k (once per request and layer).
PERTOKEN_LEAD = os.environ.get("PB27_MEMORY_PERTOKEN_LEAD", "1") != "0"
PERTOKEN_OPEN = os.environ.get("PB27_MEMORY_PERTOKEN_OPEN", "1") != "0"
PERTOKEN_PREFILL_UNION = int(os.environ.get("PB27_MEMORY_PERTOKEN_PREFILL_UNION", "2") or 2)
DECODE_STRIDE = int(os.environ.get("PB27_MEMORY_DECODE_STRIDE", "64") or 64)
DECODE_BLOCKS = int(os.environ.get("PB27_MEMORY_DECODE_BLOCKS", "64") or 64)
DECODE_RERANK = os.environ.get("PB27_MEMORY_DECODE_RERANK", "0") == "1"
SBLK = int(os.environ.get("PB27_MEMORY_SUPERBLOCK", "64") or 64)
# Dense Global Aggregation ("auto", "1", or "0"): detects high-entropy global queries across the archive and activates
# (1) stratified super-block timeline coverage, (2) novelty-rotating decode working sets across decode steps, and
# (3) resident super-block summary KV landmarks (`smean_k` / `smean_v`, 1 summary token per 4,096 tokens).
GLOBAL_AGG = os.environ.get("PB27_MEMORY_GLOBAL_AGG", "auto").strip().lower()
GLOBAL_DECODE_STRIDE = int(os.environ.get("PB27_MEMORY_GLOBAL_DECODE_STRIDE", "16") or 16)
NOVELTY_ALPHA = float(os.environ.get("PB27_MEMORY_NOVELTY_ALPHA", "0.75") or 0.75)
LAYER_STRIDE = int(os.environ.get("PB27_MEMORY_LAYER_STRIDE", "1") or 1)
GPU_ARC_CAP = int(os.environ.get("PB27_MEMORY_GPU_ARC_CAP", "262144") or 262144)
QBUF_CAP = 256
GPU_WIN_CAP = 32768
LSE = os.environ.get("PB27_MEMORY_LSE", "1") != "0"               # FlashAttention-2 LSE-merge fast path for piece_attention
RERANK_ALL = os.environ.get("PB27_MEMORY_RERANK_ALL", "1") != "0"  # exact rescoring for every piece, not only decode
BATCH: dict = globals().get("BATCH", {"ib": None, "pos_list": None})     # the runner's current InputBatch (req_ids, query_start_loc, ...)
PROMPT_LENS: dict[str, int] = globals().get("PROMPT_LENS", {})           # req_id -> initial prompt_len from RequestState.add_request
ARCHIVES: dict[tuple[str, str], "Archive"] = globals().get("ARCHIVES", {})  # (req_id, layer_name) -> Archive
STATS: dict = globals().get("STATS", {"calls": 0, "fallback": 0, "retrievals": 0, "pieces": 0, "tokens_archived": 0,
               "adopt_full": 0, "adopt_partial": 0, "adopt_none": 0, "hole_skips": 0,
               "pertoken": 0, "pertoken_capped": 0, "global_agg": 0})
# Prefix caching: vLLM skips computing the cached prefix of a new request, so the archive of the request that
# produced that prefix is retained after it finishes and adopted by the next request whose tokens match it.
RETAIN = int(os.environ.get("PB27_MEMORY_RETAIN", "0") or 0)   # max retired archive sets kept (0 = off)
RETIRED: list[dict] = globals().get("RETIRED", [])        # [{"tokens": int32 tensor of the finished request, "archives": {layer: Archive}}]
ADOPTED: dict[str, tuple] = globals().get("ADOPTED", {})  # req_id -> (archives by layer, hole_lo, hole_hi, num_computed)
_MTIME: float = os.path.getmtime(__file__) if os.path.exists(__file__) else 0.0


def configure_from_env() -> None:
    spec = os.environ.get("PB27_MEMORY", "").strip()
    if spec.lower() in ("0", "off", "false", "none"):
        MEMORY["enabled"] = False
        return
    if not spec:
        MEMORY["enabled"] = MEMORY["blocks"] > 0
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
    qbuf: torch.Tensor | None = None  # [QBUF_CAP, H, D] rolling query buffer indexed by position % QBUF_CAP
    size: int = 0               # 1 + highest position written
    prompt_len: int = 0         # 1 + highest prefill position written
    dirty_lo: int = field(default=1 << 60)  # block range whose bounds need refreshing
    dirty_hi: int = -1
    hole_lo: int = 0            # positions [hole_lo, hole_hi) were served from vLLM's prefix cache but are not in
    hole_hi: int = 0            # this archive (no matching retained archive); they are never read
    mapped: bool = False        # K/V are file-backed (OFFLOAD=nvme) at fixed capacity
    gpu_k: torch.Tensor | None = None    # [GPU_WIN_CAP, KV, D] resident tail on GPU when keys live on CPU/NVMe
    gpu_v: torch.Tensor | None = None
    gpu_lo: int = 0
    gpu_hi: int = 0
    sink_k: torch.Tensor | None = None   # [SINKS, KV, D] resident sinks on GPU when keys live on CPU/NVMe
    sink_v: torch.Tensor | None = None
    last_top: torch.Tensor | None = None
    last_rk: torch.Tensor | None = None
    prompt_top: torch.Tensor | None = None  # [KV, U] priority-ordered blocks from final prefill question piece
    decode_top: torch.Tensor | None = None  # [KV, U] rolling priority-ordered blocks accumulated during decode
    decode_step: int = 0                    # per-layer decode token counter
    last_hop_head: torch.Tensor | None = None  # [KV, 8] previous first-hop decode seeds for 2-hop change detection
    cached_top: torch.Tensor | None = None  # [KV, k] active retrieved block indices cached on GPU
    cached_rk: torch.Tensor | None = None   # [KV, k*B, D] cached retrieved keys on GPU
    cached_rv: torch.Tensor | None = None   # [KV, k*B, D] cached retrieved values on GPU
    cached_ns: int = -1                     # cached sink count in cached_pre_k / cached_pre_v
    cached_pre_k: torch.Tensor | None = None  # [ns + k*B, KV, D] FA2-ready pre_k cached on GPU
    cached_pre_v: torch.Tensor | None = None  # [ns + k*B, KV, D] FA2-ready pre_v cached on GPU
    last_prio_rank: torch.Tensor | None = None  # [KV, k] priority rank of each block in sorted `top`
    cached_cu: tuple | None = None          # (t, w_len, p_len, cu_q, cu_w, cu_p) cached int32 cu_seqlens on GPU
    smax: torch.Tensor | None = None        # [KV, (cap // block + SBLK - 1) // SBLK, D] coarse bf16 super-block max
    smin: torch.Tensor | None = None        # [KV, (cap // block + SBLK - 1) // SBLK, D] coarse bf16 super-block min
    bmean_k: torch.Tensor | None = None     # [KV, cap // block, D] per-block mean key on GPU for hierarchical summaries
    bmean_v: torch.Tensor | None = None     # [KV, cap // block, D] per-block mean value on GPU for hierarchical summaries
    smean_k: torch.Tensor | None = None     # [KV, n_sb, D] coarse super-block summary key landmarks on GPU
    smean_v: torch.Tensor | None = None     # [KV, n_sb, D] coarse super-block summary value landmarks on GPU
    visit_counts: torch.Tensor | None = None  # [KV, cap // block] int16 decode visit counts for novelty sweeping
    is_global_agg: bool = False             # True when Dense Global Aggregation mode is active for this request
    mapped_files: list[tuple[int, mmap.mmap, int]] = field(default_factory=list)
    last_evict_pos: int = 0

    def evict_page_cache(self) -> None:
        for fd, mm, nbytes in self.mapped_files:
            try:
                os.fdatasync(fd)
                mm.madvise(mmap.MADV_DONTNEED)
                os.posix_fadvise(fd, 0, nbytes, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass

    @staticmethod
    def new(kv: int, d: int, cap: int, block: int, dtype, device, max_hint: int = 0) -> "Archive":
        b_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        gpu_cap = GPU_ARC_CAP if kv <= 2 else min(GPU_ARC_CAP, 65536)
        if OFFLOAD == "nvme" and (max_hint <= 0 or max_hint > gpu_cap or cap > gpu_cap):
            big = max(cap, -(-NVME_CAP // block) * block)
            arc_dtype = torch.float8_e4m3fn if NVME_DTYPE == "fp8" else dtype
            n_blk = cap // block
            n_sb = (n_blk + SBLK - 1) // SBLK
            mk, mf_k = _mapped(kv, big, d, arc_dtype)
            mv, mf_v = _mapped(kv, big, d, arc_dtype)
            return Archive(mk, mv,
                           torch.empty(kv, n_blk, d, dtype=b_dtype, device=device),
                           torch.empty(kv, n_blk, d, dtype=b_dtype, device=device),
                           mapped=True,
                           smax=torch.empty(kv, n_sb, d, dtype=b_dtype, device=device),
                           smin=torch.empty(kv, n_sb, d, dtype=b_dtype, device=device),
                           bmean_k=torch.zeros(kv, n_blk, d, dtype=b_dtype, device=device),
                           bmean_v=torch.zeros(kv, n_blk, d, dtype=b_dtype, device=device),
                           smean_k=torch.zeros(kv, n_sb, d, dtype=b_dtype, device=device),
                           smean_v=torch.zeros(kv, n_sb, d, dtype=b_dtype, device=device),
                           mapped_files=[mf_k, mf_v])
        if OFFLOAD == "nvme":
            cap = min(max(cap, -(-max_hint // block) * block), gpu_cap)
            n_blk = cap // block
            n_sb = (n_blk + SBLK - 1) // SBLK
            return Archive(torch.empty(kv, cap, d, dtype=dtype, device=device),
                           torch.empty(kv, cap, d, dtype=dtype, device=device),
                           torch.empty(kv, n_blk, d, dtype=b_dtype, device=device),
                           torch.empty(kv, n_blk, d, dtype=b_dtype, device=device),
                           mapped=False,
                           smax=torch.empty(kv, n_sb, d, dtype=b_dtype, device=device),
                           smin=torch.empty(kv, n_sb, d, dtype=b_dtype, device=device),
                           bmean_k=torch.zeros(kv, n_blk, d, dtype=b_dtype, device=device),
                           bmean_v=torch.zeros(kv, n_blk, d, dtype=b_dtype, device=device),
                           smean_k=torch.zeros(kv, n_sb, d, dtype=b_dtype, device=device),
                           smean_v=torch.zeros(kv, n_sb, d, dtype=b_dtype, device=device))
        kv_dev = torch.device("cpu") if "cpu" in OFFLOAD else device
        kv_dtype = torch.float8_e4m3fn if "fp8" in OFFLOAD else dtype
        n_blk = cap // block
        n_sb = (n_blk + SBLK - 1) // SBLK
        return Archive(torch.empty(kv, cap, d, dtype=kv_dtype, device=kv_dev),
                       torch.empty(kv, cap, d, dtype=kv_dtype, device=kv_dev),
                       torch.empty(kv, n_blk, d, dtype=b_dtype, device=device),
                       torch.empty(kv, n_blk, d, dtype=b_dtype, device=device),
                       smax=torch.empty(kv, n_sb, d, dtype=b_dtype, device=device),
                       smin=torch.empty(kv, n_sb, d, dtype=b_dtype, device=device),
                       bmean_k=torch.zeros(kv, n_blk, d, dtype=b_dtype, device=device),
                       bmean_v=torch.zeros(kv, n_blk, d, dtype=b_dtype, device=device),
                       smean_k=torch.zeros(kv, n_sb, d, dtype=b_dtype, device=device),
                       smean_v=torch.zeros(kv, n_sb, d, dtype=b_dtype, device=device))

    def grow(self, need: int, block: int) -> None:
        cap = self.kmax.shape[1] * block if self.mapped else self.keys.shape[1]
        if need <= cap:
            return
        new_cap = cap
        while new_cap < need:
            new_cap *= 2
        kv, d = self.keys.shape[0], self.keys.shape[2]
        gpu_cap = GPU_ARC_CAP if kv <= 2 else min(GPU_ARC_CAP, 65536)
        if OFFLOAD == "nvme" and not self.mapped and need > gpu_cap:
            big = max(new_cap, -(-NVME_CAP // block) * block)
            arc_dtype = torch.float8_e4m3fn if NVME_DTYPE == "fp8" else self.keys.dtype
            mk, mf_k = _mapped(kv, big, d, arc_dtype)
            mv, mf_v = _mapped(kv, big, d, arc_dtype)
            if self.size > 0:
                mk[:, :self.size] = self.keys[:, :self.size].to(device=mk.device, dtype=arc_dtype)
                mv[:, :self.size] = self.values[:, :self.size].to(device=mv.device, dtype=arc_dtype)
                if SINKS > 0:
                    s_end = min(self.size, SINKS)
                    self.sink_k = self.keys[:, :s_end].transpose(0, 1).contiguous()
                    self.sink_v = self.values[:, :s_end].transpose(0, 1).contiguous()
            self.keys, self.values, self.mapped = mk, mv, True
            self.mapped_files = [mf_k, mf_v]
            new_cap = min(new_cap, self.keys.shape[1] // block * block)
        elif self.mapped:
            if need > self.keys.shape[1]:
                raise RuntimeError(f"PB27 nvme archive full: need {need} > PB27_MEMORY_NVME_CAP {self.keys.shape[1]}")
            new_cap = min(new_cap, self.keys.shape[1] // block * block)
        else:
            if OFFLOAD == "nvme":
                new_cap = min(new_cap, gpu_cap)
            for name in ("keys", "values"):
                old = getattr(self, name)
                buf = torch.empty(kv, new_cap, d, dtype=old.dtype, device=old.device)
                buf[:, :cap] = old
                setattr(self, name, buf)
        new_n_blk = new_cap // block
        for name in ("kmax", "kmin", "bmean_k", "bmean_v"):
            old = getattr(self, name, None)
            if old is not None:
                buf = torch.zeros(kv, new_n_blk, d, dtype=old.dtype, device=old.device)
                buf[:, : old.shape[1]] = old
                setattr(self, name, buf)
        if self.visit_counts is not None and self.visit_counts.shape[1] < new_n_blk:
            vc = torch.zeros(kv, new_n_blk, dtype=self.visit_counts.dtype, device=self.visit_counts.device)
            vc[:, : self.visit_counts.shape[1]] = self.visit_counts
            self.visit_counts = vc
        new_n_sb = (new_n_blk + SBLK - 1) // SBLK
        for name in ("smax", "smin", "smean_k", "smean_v"):
            old = getattr(self, name, None)
            if old is not None and old.shape[1] < new_n_sb:
                buf = torch.zeros(kv, new_n_sb, d, dtype=old.dtype, device=old.device)
                buf[:, : old.shape[1]] = old
                setattr(self, name, buf)

    def _update_superbounds(self, lo_b: int, hi_b: int, block: int) -> None:
        if self.smax is None or self.smin is None or hi_b < lo_b or self.size <= ORIGIN:
            return
        n_blk = min(self.kmax.shape[1], (self.size - ORIGIN + block - 1) // block)
        sb0 = max(0, lo_b // SBLK)
        sb1 = min(self.smax.shape[1] - 1, min(hi_b, max(0, n_blk - 1)) // SBLK)
        if sb1 < sb0:
            return
        full_sb_end = min(sb1 + 1, n_blk // SBLK)
        if full_sb_end > sb0:
            b0, b1 = sb0 * SBLK, full_sb_end * SBLK
            kv, _, d = self.kmax.shape
            n_full = full_sb_end - sb0
            self.smax[:, sb0:full_sb_end] = self.kmax[:, b0:b1].view(kv, n_full, SBLK, d).amax(2)
            self.smin[:, sb0:full_sb_end] = self.kmin[:, b0:b1].view(kv, n_full, SBLK, d).amin(2)
            if self.bmean_k is not None and self.smean_k is not None:
                self.smean_k[:, sb0:full_sb_end] = self.bmean_k[:, b0:b1].view(kv, n_full, SBLK, d).mean(2).to(self.smean_k.dtype)
            if self.bmean_v is not None and self.smean_v is not None:
                self.smean_v[:, sb0:full_sb_end] = self.bmean_v[:, b0:b1].view(kv, n_full, SBLK, d).mean(2).to(self.smean_v.dtype)
        for sb in range(max(sb0, full_sb_end), sb1 + 1):
            b0, b1 = sb * SBLK, min(n_blk, (sb + 1) * SBLK)
            if b1 > b0:
                self.smax[:, sb] = self.kmax[:, b0:b1].amax(1)
                self.smin[:, sb] = self.kmin[:, b0:b1].amin(1)
                if self.bmean_k is not None and self.smean_k is not None:
                    self.smean_k[:, sb] = self.bmean_k[:, b0:b1].mean(1).to(self.smean_k.dtype)
                if self.bmean_v is not None and self.smean_v is not None:
                    self.smean_v[:, sb] = self.bmean_v[:, b0:b1].mean(1).to(self.smean_v.dtype)

    def write(self, pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block: int,
              p0: int | None = None, hi: int | None = None) -> None:
        """pos [N] long, k/v [N, KV, D] -> store at those positions (overwriting re-verified speculative slots)."""
        if p0 is None or hi is None:
            hi = int(pos.max()) + 1
            p0 = int(pos.min())
        n = int(pos.numel())
        contig = (hi - p0 == n)
        self.grow(hi, block)
        kt = k.transpose(0, 1)
        vt = v.transpose(0, 1)
        if self.keys.dtype != torch.float8_e4m3fn:
            if SINKS > 0 and p0 < SINKS and contig:
                s_end = min(hi, SINKS)
                if self.sink_k is None:
                    self.sink_k = torch.empty(SINKS, k.shape[1], k.shape[2], dtype=k.dtype, device=k.device)
                    self.sink_v = torch.empty(SINKS, v.shape[1], v.shape[2], dtype=v.dtype, device=k.device)
                self.sink_k[p0:s_end] = k[:s_end - p0]
                self.sink_v[p0:s_end] = v[:s_end - p0]
            if contig and n <= GPU_WIN_CAP and (self.gpu_k is None or (p0 >= self.gpu_lo and p0 <= self.gpu_hi)):
                if self.gpu_k is None:
                    self.gpu_k = torch.empty(GPU_WIN_CAP, k.shape[1], k.shape[2], dtype=k.dtype, device=k.device)
                    self.gpu_v = torch.empty(GPU_WIN_CAP, v.shape[1], v.shape[2], dtype=v.dtype, device=k.device)
                    self.gpu_lo = p0
                elif hi - self.gpu_lo > GPU_WIN_CAP:
                    keep_lo = max(self.gpu_lo, p0 - MEMORY["window"] - block)
                    keep_n = p0 - keep_lo
                    if keep_n > 0:
                        self.gpu_k[:keep_n] = self.gpu_k[keep_lo - self.gpu_lo : p0 - self.gpu_lo].clone()
                        self.gpu_v[:keep_n] = self.gpu_v[keep_lo - self.gpu_lo : p0 - self.gpu_lo].clone()
                    self.gpu_lo = keep_lo
                self.gpu_k[p0 - self.gpu_lo : hi - self.gpu_lo] = k
                self.gpu_v[p0 - self.gpu_lo : hi - self.gpu_lo] = v
                self.gpu_hi = hi
            else:
                self.gpu_k = self.gpu_v = None
        fast_bounds = False
        b_start = b_end = 0
        if hi - 1 >= ORIGIN and contig and n >= block:
            b_start = max(0, (p0 - ORIGIN + block - 1) // block)
            b_end = max(b_start, (hi - ORIGIN) // block)
            if b_end > b_start:
                s0 = ORIGIN + b_start * block - p0
                s1 = ORIGIN + b_end * block - p0
                k_full = kt[:, s0:s1].view(kt.shape[0], b_end - b_start, block, kt.shape[2])
                v_full = vt[:, s0:s1].view(vt.shape[0], b_end - b_start, block, vt.shape[2])
                self.kmax[:, b_start:b_end] = k_full.amax(2).to(self.kmax.dtype)
                self.kmin[:, b_start:b_end] = k_full.amin(2).to(self.kmin.dtype)
                if self.bmean_k is not None:
                    self.bmean_k[:, b_start:b_end] = k_full.mean(2).to(self.bmean_k.dtype)
                if self.bmean_v is not None:
                    self.bmean_v[:, b_start:b_end] = v_full.mean(2).to(self.bmean_v.dtype)
                fast_bounds = True
        if contig:
            if self.keys.device.type == "cpu" and kt.is_cuda and self.keys.dtype == kt.dtype and n <= 8192:
                _copy_d2h_staged(self.keys[:, p0:hi], self.values[:, p0:hi], kt, vt)
            else:
                self.keys[:, p0:hi] = kt if (self.keys.device == kt.device and self.keys.dtype == kt.dtype) else kt.to(device=self.keys.device, dtype=self.keys.dtype)
                self.values[:, p0:hi] = vt if (self.values.device == vt.device and self.values.dtype == vt.dtype) else vt.to(device=self.values.device, dtype=self.values.dtype)
        else:
            p_idx = pos if self.keys.device == pos.device else pos.to(self.keys.device)
            self.keys[:, p_idx] = kt.to(device=self.keys.device, dtype=self.keys.dtype)
            self.values[:, p_idx] = vt.to(device=self.values.device, dtype=self.values.dtype)
        self.size = max(self.size, hi)
        if self.mapped and self.size - self.last_evict_pos >= 65536:
            self.evict_page_cache()
            self.last_evict_pos = self.size
        if hi - 1 < ORIGIN:
            return
        lo_b, hi_b = max(0, p0 - ORIGIN) // block, (hi - 1 - ORIGIN) // block
        if fast_bounds:
            if lo_b < b_start:
                self._refresh_range(lo_b, b_start - 1, block)
            if hi_b >= b_end:
                self._refresh_range(b_end, hi_b, block)
            self._update_superbounds(lo_b, hi_b, block)
            STATS["tokens_archived"] += n
            return
        self.dirty_lo, self.dirty_hi = min(self.dirty_lo, lo_b), max(self.dirty_hi, hi_b)
        STATS["tokens_archived"] += n

    def write_q(self, pos: torch.Tensor, q: torch.Tensor) -> None:
        """pos [N] long, q [N, H, D] -> store trailing queries in ring buffer by position."""
        if self.qbuf is None:
            self.qbuf = torch.zeros(QBUF_CAP, q.shape[1], q.shape[2], dtype=q.dtype, device=q.device)
        if pos.numel() > QBUF_CAP:
            pos, q = pos[-QBUF_CAP:], q[-QBUF_CAP:]
        self.qbuf[pos % QBUF_CAP] = q.to(self.qbuf.dtype)

    def _refresh_range(self, lo_b: int, hi_b: int, block: int) -> None:
        if hi_b < lo_b:
            return
        lo, hi = ORIGIN + lo_b * block, min(self.size, ORIGIN + (hi_b + 1) * block)
        if hi <= lo:
            return
        if self.gpu_k is not None and lo >= self.gpu_lo and hi <= self.gpu_hi:
            seg = self.gpu_k[lo - self.gpu_lo : hi - self.gpu_lo].transpose(0, 1).float()
            seg_v = self.gpu_v[lo - self.gpu_lo : hi - self.gpu_lo].transpose(0, 1).float() if self.gpu_v is not None else None
        else:
            seg = self.keys[:, lo:hi].to(device=self.kmax.device).float()
            seg_v = self.values[:, lo:hi].to(device=self.kmax.device).float() if self.bmean_v is not None else None
        pad = (-seg.shape[1]) % block
        if pad:
            seg = torch.cat((seg, seg[:, -1:].expand(-1, pad, -1)), dim=1)
            if seg_v is not None:
                seg_v = torch.cat((seg_v, seg_v[:, -1:].expand(-1, pad, -1)), dim=1)
        blocks = seg.view(seg.shape[0], -1, block, seg.shape[2])
        self.kmax[:, lo_b: lo_b + blocks.shape[1]] = blocks.amax(2).to(self.kmax.dtype)
        self.kmin[:, lo_b: lo_b + blocks.shape[1]] = blocks.amin(2).to(self.kmin.dtype)
        if self.bmean_k is not None:
            self.bmean_k[:, lo_b: lo_b + blocks.shape[1]] = blocks.mean(2).to(self.bmean_k.dtype)
        if self.bmean_v is not None and seg_v is not None:
            blocks_v = seg_v.view(seg_v.shape[0], -1, block, seg_v.shape[2])
            self.bmean_v[:, lo_b: lo_b + blocks_v.shape[1]] = blocks_v.mean(2).to(self.bmean_v.dtype)

    def refresh_bounds(self, block: int) -> None:
        if self.dirty_hi < self.dirty_lo:
            return
        self._refresh_range(self.dirty_lo, self.dirty_hi, block)
        self._update_superbounds(self.dirty_lo, self.dirty_hi, block)
        self.dirty_lo, self.dirty_hi = 1 << 60, -1


_MMAP_POOL: dict[tuple[int, int, int, torch.dtype], list[torch.Tensor]] = {}
_MMAP_POOL_MAX = 32
_STAGE_KV: dict[tuple[int, int, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}


def _copy_d2h_staged(dst_k: torch.Tensor, dst_v: torch.Tensor, kt: torch.Tensor, vt: torch.Tensor) -> None:
    kv, n, d = kt.shape
    key = (kv, d, kt.dtype)
    pair = _STAGE_KV.get(key)
    if pair is None or pair[0].shape[1] < n:
        cap_n = max(8192, n)
        pair = (
            torch.empty((kv, cap_n, d), dtype=kt.dtype, pin_memory=True),
            torch.empty((kv, cap_n, d), dtype=vt.dtype, pin_memory=True),
        )
        _STAGE_KV[key] = pair
    sk, sv = pair[0][:, :n], pair[1][:, :n]
    sk.copy_(kt.contiguous(), non_blocking=True)
    sv.copy_(vt.contiguous(), non_blocking=True)
    torch.cuda.current_stream().synchronize()
    dst_k.copy_(sk)
    dst_v.copy_(sv)


def _mapped(kv: int, cap: int, d: int, dtype) -> tuple[torch.Tensor, tuple[int, mmap.mmap, int]]:
    """[kv, cap, d] tensor backed by an unlinked sparse file; only touched pages use disk or evictable page cache."""
    nbytes = kv * cap * d * torch.empty(0, dtype=dtype).element_size()
    target_dir = NVME_DIR if (NVME_DIR and os.path.isdir(NVME_DIR)) else None
    fd, path = tempfile.mkstemp(dir=target_dir, prefix="pb27_arc_")
    try:
        os.unlink(path)
        os.ftruncate(fd, nbytes)
        mm = mmap.mmap(fd, nbytes)
    except Exception:
        os.close(fd)
        raise
    return torch.frombuffer(mm, dtype=torch.uint8).view(dtype).view(kv, cap, d), (fd, mm, nbytes)


def _free(archives) -> None:
    for arc in archives.values():
        for fd, mm, nbytes in getattr(arc, "mapped_files", ()):
            try:
                mm.madvise(mmap.MADV_DONTNEED)
                os.posix_fadvise(fd, 0, nbytes, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
            try:
                os.close(fd)
            except OSError:
                pass
        if hasattr(arc, "mapped_files"):
            arc.mapped_files.clear()
    archives.clear()
    if torch.cuda.is_available() and (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) > (2 << 30):
        torch.cuda.empty_cache()


def drop_request(req_id: str, tokens: torch.Tensor | None = None) -> None:
    """A request finished. With RETAIN > 0 and its token ids known, keep its archives for prefix reuse."""
    PROMPT_LENS.pop(req_id, None)
    ADOPTED.pop(req_id, None)
    keys = [k for k in ARCHIVES if k[0] == req_id]
    archives = {k[1]: ARCHIVES.pop(k) for k in keys}
    if not archives:
        return
    if RETAIN <= 0 or tokens is None or len(tokens) < 2 or any(a.hole_hi > a.hole_lo for a in archives.values()):
        _free(archives)
        return
    for arc in archives.values():
        arc.prompt_top = arc.decode_top = arc.last_top = arc.last_rk = None
        arc.cached_top = arc.cached_rk = arc.cached_rv = None
        arc.cached_pre_k = arc.cached_pre_v = arc.cached_cu = None
        arc.cached_ns = -1
        if arc.mapped:
            arc.evict_page_cache()
    RETIRED.append({"tokens": tokens, "archives": archives})
    while len(RETIRED) > RETAIN:
        _free(RETIRED.pop(0)["archives"])


def adopt_for_request(req_id: str, all_token_ids, num_computed: int) -> None:
    """New request whose first `num_computed` tokens come from vLLM's prefix cache: adopt the retired archive set
    with the longest exact token match. Positions not covered by it become a hole that is never read."""
    if num_computed <= 0 or not MEMORY["enabled"]:
        return
    new = torch.as_tensor(list(all_token_ids[:num_computed]), dtype=torch.int32)
    best, best_len = None, 0
    for i, ent in enumerate(RETIRED):
        old = ent["tokens"]
        n = min(len(old) - 1, num_computed)      # the last sampled token of a finished request was never computed
        if n <= best_len:
            continue
        neq = (old[:n].cpu() != new[:n]).nonzero()
        lcp = int(neq[0]) if len(neq) else n
        if lcp > best_len:
            best, best_len = i, lcp
    if best is None:
        STATS["adopt_none"] += 1
        ADOPTED[req_id] = ({}, 0, num_computed, num_computed)
        return
    ent = RETIRED.pop(best)
    hole = (best_len, num_computed) if best_len < num_computed else (0, 0)
    STATS["adopt_full" if hole == (0, 0) else "adopt_partial"] += 1
    ADOPTED[req_id] = (ent["archives"], *hole, num_computed)


def oracle_block_mass(arc: Archive, qg: torch.Tensor, full: int, block: int, qchunk: int = 256) -> torch.Tensor:
    """Diagnostic selector: the true softmax attention mass of each archived block, normalised over the archived
    blocks per query and summed over the piece's queries. qg [KV, G*T, D] -> [KV, full]. Small per-head chunks keep
    the extra allocation to tens of MB (GB10 memory is shared with the host)."""
    kv, n, d = qg.shape
    mass = torch.zeros(kv, full, device=qg.device)
    for g in range(kv):
        keys = arc.keys[g, ORIGIN:ORIGIN + full * block]                                             # [N, D]
        for c0 in range(0, n, qchunk):
            logits = (qg[g, c0:c0 + qchunk].to(keys.dtype) @ keys.T).float() * d ** -0.5  # [q, N]
            mass[g] += logits.softmax(-1).view(-1, full, block).sum((0, 2))
    return mass


def _gather_tokens(t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.float8_e4m3fn:
        return t[torch.arange(t.shape[0], device=t.device)[:, None], idx]
    return t.gather(1, idx.unsqueeze(-1).expand(-1, -1, t.shape[2]))


_ROPE_INV_FREQ: dict[tuple[torch.device, int], torch.Tensor] = {}


def _get_rope_inv_freq(device: torch.device, d_rot: int) -> torch.Tensor:
    key = (device, d_rot)
    inv_freq = _ROPE_INV_FREQ.get(key)
    if inv_freq is None:
        inv_freq = 1.0 / (
            ROPE_THETA ** (torch.arange(0, d_rot, 2, device=device, dtype=torch.float32) / float(d_rot))
        )
        _ROPE_INV_FREQ[key] = inv_freq
    return inv_freq


def _reposition_rope_sinks(
    sk: torch.Tensor,
    ref_pos: int,
    span: int | None = None,
    n_blk: int = 0,
    block: int = 64,
) -> torch.Tensor:
    """Reposition the sink keys `sk` [KV, ns, D] from original positions `0..ns-1` to `[base_start - ns, base_start)`,
    strictly before the earliest repositioned archive block `[base_start, ref_pos)` when `ref_pos > ROPE_HORIZON`."""
    if ROPE_HORIZON <= 0 or ref_pos <= ROPE_HORIZON or sk.numel() == 0:
        return sk
    kv, ns, d = sk.shape
    d_rot = d // 4
    half = d_rot // 2
    if half <= 0:
        return sk
    arc_span = ROPE_ARC_SPAN if (span is None or span <= 0) else span
    eff_span = max(arc_span, n_blk * block)
    base_start = max(ORIGIN, ref_pos - eff_span)
    delta_val = float(max(0, base_start - ns))
    if delta_val <= 0.0:
        return sk
    inv_freq = _get_rope_inv_freq(sk.device, d_rot)
    angles = delta_val * inv_freq
    cos = angles.cos().view(1, 1, half)
    sin = angles.sin().view(1, 1, half)
    x1 = sk[..., :half].float()
    x2 = sk[..., half:d_rot].float()
    r1 = (x1 * cos - x2 * sin).to(sk.dtype)
    r2 = (x2 * cos + x1 * sin).to(sk.dtype)
    out = sk.clone()
    out[..., :half] = r1
    out[..., half:d_rot] = r2
    return out


def _reposition_rope_blocks(
    rk: torch.Tensor,
    blk_idx: torch.Tensor,
    block: int,
    ref_pos: int,
    contiguous_runs: bool = True,
    span: int | None = None,
    prio_rank: torch.Tensor | None = None,
) -> torch.Tensor:
    """Piecewise-Proportional, Strictly Non-Overlapping Block RoPE into `[ref_pos - eff_span, ref_pos)`.

    Guarantees three simultaneous invariants when `ref_pos > ROPE_HORIZON`:
    1. Exact Intra-Block & Contiguous-Run Continuity: All `block` tokens inside block `b` rotate by the same shift
       `delta_b` (preserving exact `+1` local steps), and contiguous blocks `b_j = b_{j-1} + 1` have inter-block
       historical gap `g_mid = 0`, receiving the exact same `delta` across the block boundary.
    2. Strict Chronological Non-Overlap: Every block `j` is allocated a mandatory `block`-token footprint plus its
       proportional share of the remaining RoPE slack, guaranteeing `T(b_j) >= T(b_{j-1}) + block` so nearby
       non-adjacent blocks (`b_j = b_{j-1} + 2`) never invert or overlap in rotary phase space.
    3. Proportional Macro-Timeline Preservation: True historical gaps `g_j = (b_j - b_{j-1} - 1) * block` scale
       proportionally across the `ROPE_ARC_SPAN` horizon instead of flattening relative timeline distances.
    """
    if ROPE_HORIZON <= 0 or ref_pos <= ROPE_HORIZON or rk.numel() == 0:
        return rk
    kv, total_t, d = rk.shape
    n_blk = blk_idx.shape[1]
    if n_blk == 0 or total_t != n_blk * block:
        return rk
    d_rot = d // 4
    half = d_rot // 2
    if half <= 0:
        return rk
    idx = blk_idx.to(device=rk.device, dtype=torch.long)
    arc_span = ROPE_ARC_SPAN if (span is None or span <= 0) else span
    orig_pos = (ORIGIN + idx * block).to(torch.float32)
    if not contiguous_runs:
        scale = 1.0 - min(1.0, float(arc_span) / float(max(1, ref_pos)))
        target_pos = orig_pos + torch.round((float(ref_pos) - orig_pos).clamp(min=0.0) * scale)
        delta = (target_pos - orig_pos).clamp(min=0.0)
    else:
        eff_span = max(arc_span, n_blk * block)
        base_start = float(max(ORIGIN, ref_pos - eff_span))
        slack = float(max(0.0, (ref_pos - base_start) - n_blk * block))
        g_pre = (orig_pos[:, :1] - float(ORIGIN)).clamp(min=0.0)
        if n_blk > 1:
            g_mid = ((idx[:, 1:] - idx[:, :-1] - 1).clamp(min=0) * block).to(torch.float32)
            g_cum = torch.cat((g_pre, g_pre + torch.cumsum(g_mid, dim=1)), dim=1)
            g_mid_sum = g_mid.sum(dim=1, keepdim=True)
        else:
            g_cum = g_pre
            g_mid_sum = torch.zeros_like(g_pre)
        g_post = (float(ref_pos) - (orig_pos[:, -1:] + float(block))).clamp(min=0.0)
        g_tot = (g_pre + g_mid_sum + g_post).clamp(min=1.0)
        j_steps = (torch.arange(n_blk, device=rk.device, dtype=torch.float32) * float(block)).unsqueeze(0)
        target_pos = base_start + j_steps + torch.round(g_cum * (slack / g_tot))
        delta = (target_pos - orig_pos).clamp(min=0.0)
    inv_freq = _get_rope_inv_freq(rk.device, d_rot)
    angles = delta.unsqueeze(-1) * inv_freq
    cos = angles.cos().unsqueeze(2)
    sin = angles.sin().unsqueeze(2)
    rk_v = rk.view(kv, n_blk, block, d)
    x1 = rk_v[..., :half].float()
    x2 = rk_v[..., half:d_rot].float()
    r1 = (x1 * cos - x2 * sin).to(rk.dtype)
    r2 = (x2 * cos + x1 * sin).to(rk.dtype)
    out = rk.clone().view(kv, n_blk, block, d)
    out[..., :half] = r1
    out[..., half:d_rot] = r2
    return out.view(kv, total_t, d)


def _reposition_rope_landmarks(
    smk: torch.Tensor,
    block: int,
    ref_pos: int,
    span: int | None = None,
) -> torch.Tensor:
    """Reposition super-block centroid landmark keys `smk` [KV, n_sb, D] proportionally into `[ref_pos - arc_span, ref_pos)`."""
    if ROPE_HORIZON <= 0 or ref_pos <= ROPE_HORIZON or smk.numel() == 0:
        return smk
    kv, n_sb, d = smk.shape
    d_rot = d // 4
    half = d_rot // 2
    if half <= 0:
        return smk
    arc_span = ROPE_ARC_SPAN if (span is None or span <= 0) else span
    sb_stride = float(SBLK * block)
    orig_pos = float(ORIGIN) + (torch.arange(n_sb, device=smk.device, dtype=torch.float32) + 0.5) * sb_stride
    scale = 1.0 - min(1.0, float(arc_span) / float(max(1, ref_pos)))
    target_pos = orig_pos + torch.round((float(ref_pos) - orig_pos).clamp(min=0.0) * scale)
    delta = (target_pos - orig_pos).clamp(min=0.0).unsqueeze(0)  # [1, n_sb]
    inv_freq = _get_rope_inv_freq(smk.device, d_rot)
    angles = delta.unsqueeze(-1) * inv_freq  # [1, n_sb, half]
    cos = angles.cos()
    sin = angles.sin()
    x1 = smk[..., :half].float()
    x2 = smk[..., half:d_rot].float()
    r1 = (x1 * cos - x2 * sin).to(smk.dtype)
    r2 = (x2 * cos + x1 * sin).to(smk.dtype)
    out = smk.clone()
    out[..., :half] = r1
    out[..., half:d_rot] = r2
    return out


def _gather_blocks(t: torch.Tensor, blk_idx: torch.Tensor, block: int) -> torch.Tensor:
    """Gather contiguous blocks `blk_idx` [KV, B] starting at ORIGIN from `t` [KV, cap, D] -> [KV, B * block, D]."""
    kv, cap, d = t.shape
    b_idx = blk_idx.to(t.device)
    n_blk = (cap - ORIGIN) // block
    tb = t[:, ORIGIN:ORIGIN + n_blk * block].view(kv, n_blk, block * d)
    if not t.is_cuda or t.dtype == torch.float8_e4m3fn:
        return tb[torch.arange(kv, device=t.device)[:, None], b_idx].view(kv, -1, d)
    return tb.gather(1, b_idx.unsqueeze(-1).expand(-1, -1, block * d)).view(kv, -1, d)


def _gather_cached_delta(arc: Archive, top: torch.Tensor, block: int, dtype, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Reuse blocks already resident in `arc.cached_rk`/`arc.cached_rv` on GPU and gather only `top \\ arc.cached_top`."""
    if (arc.cached_top is not None and arc.cached_rk is not None and arc.cached_rv is not None
            and arc.cached_top.shape == top.shape and (top is arc.cached_top or torch.equal(top, arc.cached_top))):
        return arc.cached_rk.to(dtype=dtype), arc.cached_rv.to(dtype=dtype)
    if (arc.keys.is_cuda and arc.keys.dtype != torch.float8_e4m3fn
            or arc.cached_top is None or arc.cached_rk is None or arc.cached_rv is None
            or arc.cached_top.shape != top.shape):
        rk = _gather_blocks(arc.keys, top, block).to(device=device, dtype=dtype)
        rv = _gather_blocks(arc.values, top, block).to(device=device, dtype=dtype)
        return rk, rv
    kv, k_len = top.shape
    d = arc.keys.shape[2]
    pos_in_cache = torch.searchsorted(arc.cached_top, top)
    pos_clamped = pos_in_cache.clamp(max=k_len - 1)
    hit = (pos_in_cache < k_len) & (arc.cached_top.gather(1, pos_clamped) == top)
    idx_exp = pos_clamped.unsqueeze(-1).expand(kv, k_len, block * d)
    rk = arc.cached_rk.view(kv, k_len, block * d).gather(1, idx_exp).view(kv, k_len, block, d).to(dtype=dtype)
    rv = arc.cached_rv.view(kv, k_len, block * d).gather(1, idx_exp).view(kv, k_len, block, d).to(dtype=dtype)
    hit_cpu = hit.cpu()
    if not bool(hit_cpu.all()):
        top_cpu = top.to(arc.keys.device)
        n_blk = (arc.keys.shape[1] - ORIGIN) // block
        kb_all = arc.keys[:, ORIGIN:ORIGIN + n_blk * block].view(kv, n_blk, block, d)
        vb_all = arc.values[:, ORIGIN:ORIGIN + n_blk * block].view(kv, n_blk, block, d)
        for g in range(kv):
            miss_cpu = (~hit_cpu[g]).nonzero(as_tuple=True)[0]
            if miss_cpu.numel() > 0:
                miss_blks = top_cpu[g, miss_cpu]
                miss_gpu = miss_cpu.to(device=device)
                rk[g, miss_gpu] = kb_all[g, miss_blks].to(device=device, dtype=dtype)
                rv[g, miss_gpu] = vb_all[g, miss_blks].to(device=device, dtype=dtype)
    return rk.view(kv, k_len * block, d), rv.view(kv, k_len * block, d)


def retrieve_blocks(arc: Archive, q: torch.Tensor, groups: int, limit: int, rerank: bool,
                    budget: int = 0) -> torch.Tensor | None:
    """Indices of the top blocks among archived positions [0, limit) for query piece q [T, H, D] -> [KV, k] or None."""
    block = MEMORY["block_size"]
    full = (limit - ORIGIN) // block
    if full <= 0:
        return None
    arc.refresh_bounds(block)
    t, h, d = q.shape
    kv = h // groups
    n_valid = full
    hole_b0 = hole_b1 = 0
    if arc.hole_hi > arc.hole_lo:
        hole_b0 = max(0, arc.hole_lo - ORIGIN) // block
        hole_b1 = min(full, (arc.hole_hi - ORIGIN + block - 1) // block)
        if hole_b1 > hole_b0:
            n_valid = full - (hole_b1 - hole_b0)
        if n_valid <= 0:
            return None
    if not budget:
        budget = MEMORY["decode_blocks"] if (rerank and MEMORY["decode_blocks"]) else MEMORY["blocks"]
    in_prompt_mid = not rerank and limit + MEMORY["window"] < arc.prompt_len
    if not in_prompt_mid:
        n_arch = max(full, (arc.prompt_len - MEMORY["window"] - ORIGIN) // block)
        for s_thresh, s_blocks in zip(SCALE[0::2], SCALE[1::2]):
            if n_arch > s_thresh:
                budget = max(budget, s_blocks)
    k = min(budget, n_valid)
    rerank_pool = max(MEMORY["rerank"], 2 * budget) if MEMORY["rerank"] > 0 else 0
    qg = None
    if ORACLE:
        qg = q.float().permute(1, 0, 2).reshape(kv, groups * t, d)                         # [KV, G*T, D]
        score = oracle_block_mass(arc, qg, full, block)
        if hole_b1 > hole_b0:
            score[:, hole_b0:hole_b1] = float("-inf")
    else:
        s_dtype = q.dtype if (q.is_cuda and q.dtype in (torch.bfloat16, torch.float16)) else torch.float32
        qg_s = q.to(s_dtype).permute(1, 0, 2).reshape(kv, groups * t, d)
        if ROPE_HORIZON > 0 and limit > ROPE_HORIZON:
            qg_s = qg_s.clone()
            qg_s[..., : d // 4] = 0
        pos, neg = qg_s.clamp(min=0), qg_s.clamp(max=0)
        if not rerank and SBLK > 1 and arc.smax is not None and arc.smin is not None and full > SBLK * 16:
            n_sb = (full + SBLK - 1) // SBLK
            smax_t = arc.smax[:, :n_sb].to(s_dtype).transpose(-1, -2)
            smin_t = arc.smin[:, :n_sb].to(s_dtype).transpose(-1, -2)
            if pos.shape[1] <= 2048:
                sb_score = (pos @ smax_t + neg @ smin_t).amax(1).float()
            else:
                sb_score = torch.full((kv, n_sb), float("-inf"), device=q.device, dtype=s_dtype)
                for c0 in range(0, pos.shape[1], 2048):
                    s_c = (pos[:, c0:c0 + 2048] @ smax_t + neg[:, c0:c0 + 2048] @ smin_t).amax(1)
                    sb_score = torch.maximum(sb_score, s_c)
                sb_score = sb_score.float()
            m_sb = min(max(16, (2 * k + SBLK - 1) // SBLK), n_sb)
            top_sb = sb_score.topk(m_sb, dim=-1).indices
            sb_off = torch.arange(SBLK, device=q.device, dtype=top_sb.dtype)
            fine_raw = (top_sb.unsqueeze(-1) * SBLK + sb_off).reshape(kv, m_sb * SBLK)
            fine_valid = fine_raw < full
            fine_idx = fine_raw.clamp(max=full - 1)
            idx_d = fine_idx.unsqueeze(-1).expand(kv, m_sb * SBLK, d)
            kmax_t = arc.kmax[:, :full].gather(1, idx_d).to(s_dtype).transpose(-1, -2)
            kmin_t = arc.kmin[:, :full].gather(1, idx_d).to(s_dtype).transpose(-1, -2)
            if pos.shape[1] <= 2048:
                sub_score = (pos @ kmax_t + neg @ kmin_t).amax(1).float()
            else:
                sub_score = torch.full((kv, m_sb * SBLK), float("-inf"), device=q.device, dtype=s_dtype)
                for c0 in range(0, pos.shape[1], 2048):
                    s_c = (pos[:, c0:c0 + 2048] @ kmax_t + neg[:, c0:c0 + 2048] @ kmin_t).amax(1)
                    sub_score = torch.maximum(sub_score, s_c)
                sub_score = sub_score.float()
            sub_score = sub_score.masked_fill(~fine_valid, float("-inf"))
            if hole_b1 > hole_b0:
                in_hole = (fine_idx >= hole_b0) & (fine_idx < hole_b1)
                sub_score = sub_score.masked_fill(in_hole, float("-inf"))
            arc.last_top = arc.last_rk = None
            top = fine_idx.gather(-1, sub_score.topk(min(k, m_sb * SBLK), dim=-1).indices)
            if MEMORY["neighbors"]:
                top = torch.cat([top] + [(top + j).clamp(max=full - 1) for j in range(1, MEMORY["neighbors"] + 1)], dim=-1)
            STATS["retrievals"] += 1
            return top.sort(-1).values
        kmax_t = arc.kmax[:, :full].to(s_dtype).transpose(-1, -2)
        kmin_t = arc.kmin[:, :full].to(s_dtype).transpose(-1, -2)
        if pos.shape[1] <= 2048:
            score = (pos @ kmax_t + neg @ kmin_t).amax(1).float()
        else:
            score = torch.full((kv, full), float("-inf"), device=q.device, dtype=s_dtype)
            for c0 in range(0, pos.shape[1], 2048):
                s_c = (pos[:, c0:c0 + 2048] @ kmax_t + neg[:, c0:c0 + 2048] @ kmin_t).amax(1)
                score = torch.maximum(score, s_c)
            score = score.float()
        if hole_b1 > hole_b0:
            score[:, hole_b0:hole_b1] = float("-inf")
    arc.last_top = arc.last_rk = None
    if rerank and rerank_pool > k and not ORACLE:
        if qg is None:
            qg = q.float().permute(1, 0, 2).reshape(kv, groups * t, d)
        c = min(rerank_pool, n_valid)
        cand = score.topk(c, dim=-1).indices                                              # [KV, c]
        ck_raw = _gather_blocks(arc.keys, cand, block).to(device=q.device)
        ck = ck_raw.float()
        if ROPE_HORIZON > 0 and limit > ROPE_HORIZON:
            ck = _reposition_rope_blocks(ck, cand, block, limit, contiguous_runs=False, span=0)
        exact = torch.einsum("gqd,gnd->gqn", qg, ck).amax(1).view(kv, c, block).amax(-1)
        keep = exact.topk(k, dim=-1).indices
        top = cand.gather(-1, keep)
        if not MEMORY["neighbors"]:
            order = top.argsort(-1)
            top, keep = top.gather(-1, order), keep.gather(-1, order)
            arc.last_top = top
            arc.last_rk = ck_raw.view(kv, c, block * d).gather(1, keep.unsqueeze(-1).expand(-1, -1, block * d)).view(kv, -1, d)
            STATS["retrievals"] += 1
            return top
    else:
        top = score.topk(k, dim=-1).indices
    if MEMORY["neighbors"]:
        top = torch.cat([top] + [(top + j).clamp(max=full - 1) for j in range(1, MEMORY["neighbors"] + 1)], dim=-1)
    STATS["retrievals"] += 1
    return top.sort(-1).values


def _ranks(score: torch.Tensor) -> torch.Tensor:
    """Rank (0 = best) of every entry along the last axis."""
    order = score.argsort(-1, descending=True)
    ar = torch.arange(score.shape[-1], device=score.device).expand_as(order)
    return torch.empty_like(order).scatter_(-1, order, ar)


def _dedup_priority(seq: torch.Tensor, u: int) -> torch.Tensor:
    """Select the first `u` unique block indices per row of `seq` [KV, M] in priority order -> [KV, u_eff]."""
    kv, m = seq.shape
    u_eff = min(u, m)
    vals, perm = seq.sort(dim=-1, stable=True)
    dup = torch.zeros_like(vals, dtype=torch.bool)
    dup[:, 1:] = vals[:, 1:] == vals[:, :-1]
    score = torch.where(dup, perm + m, perm)
    keep = score.topk(u_eff, dim=-1, largest=False).values % m
    return seq.gather(-1, keep)


def _expand_successors_dedup(blk_ord: torch.Tensor, full: int, u: int,
                             m_succ: int = 80) -> tuple[torch.Tensor, torch.Tensor]:
    """Interleave each of the first `m_succ` blocks in `blk_ord` [KV, M] with its successor `(b + 1).clamp(max=full - 1)`
    and select the first `u` unique blocks per row in priority order -> (top_prio [KV, u], top_sorted [KV, u])."""
    kv, m = blk_ord.shape
    ms = min(m_succ, m)
    head = blk_ord[:, :ms]
    succ = (head + 1).clamp(max=full - 1)
    pairs = torch.stack((head, succ), dim=-1).view(kv, 2 * ms)
    seq = torch.cat((pairs, blk_ord[:, ms:]), dim=-1) if ms < m else pairs
    top_prio = _dedup_priority(seq, u)
    return top_prio, top_prio.sort(-1).values


def _second_hop_blocks(arc: Archive, blk_ord: torch.Tensor, full: int, n_valid: int,
                       block: int, k2: int = 64, c2_cap: int = 512) -> torch.Tensor:
    """Two-hop key-to-key bridge retrieval over the position-invariant unrotated key slice `d // 4 :`."""
    kv = blk_ord.shape[0]
    d = arc.keys.shape[2]
    d0 = d // 4
    m1 = min(6, blk_ord.shape[1])
    b1 = blk_ord[:, :m1]
    b_seed = _dedup_priority(
        torch.stack((b1, (b1 + 1).clamp(max=full - 1)), dim=-1).view(kv, 2 * m1),
        min(10, full),
    )
    k_seed = _gather_blocks(arc.keys, b_seed, block).to(device=arc.kmax.device).float()[..., d0:]   # [KV, S, D_u]
    kmax_u = arc.kmax[:, :full, d0:].float()
    kmin_u = arc.kmin[:, :full, d0:].float()
    k_mean = 0.5 * (kmax_u + kmin_u).mean(1, keepdim=True)                                          # [KV, 1, D_u]
    q2 = F.normalize(k_seed - k_mean, dim=-1)                                                       # [KV, S, D_u]
    kmax2 = (kmax_u - k_mean).transpose(-1, -2)                                                     # [KV, D_u, full]
    kmin2 = (kmin_u - k_mean).transpose(-1, -2)
    s2 = q2.clamp(min=0) @ kmax2 + q2.clamp(max=0) @ kmin2                                          # [KV, S, full]
    if arc.hole_hi > arc.hole_lo:
        b0 = max(0, arc.hole_lo - ORIGIN) // block
        b_hi = min(full, (arc.hole_hi - ORIGIN + block - 1) // block)
        if b_hi > b0:
            s2[..., b0:b_hi] = float("-inf")
    s2_fin = s2.masked_fill(s2.isinf(), 0.0)
    s2_rel = s2 - s2_fin.mean(1, keepdim=True)
    s2_rel_fin = s2_rel.masked_fill(s2.isinf(), 0.0)
    z2 = (s2_rel - s2_rel_fin.mean(-1, keepdim=True)) / s2_rel_fin.std(-1, keepdim=True).clamp(min=1e-6)  # [KV, S, full]
    c2 = min(c2_cap, n_valid)
    cand2 = z2.amax(1).topk(c2, dim=-1).indices                                                     # [KV, c2]
    ck2 = _gather_blocks(arc.keys, cand2, block).to(device=arc.kmax.device).float()[..., d0:] - k_mean
    ck2 = F.normalize(ck2, dim=-1)                                                                  # [KV, c2*B, D_u]
    n_q2 = min(128, q2.shape[1])
    top_q_idx = z2.amax(-1).topk(n_q2, dim=-1).indices                                              # [KV, n_q2]
    q2_top = q2.gather(1, top_q_idx.unsqueeze(-1).expand(-1, -1, q2.shape[-1]))                     # [KV, n_q2, D_u]
    ex2 = torch.bmm(q2_top, ck2.transpose(-1, -2)).view(kv, n_q2, c2, block).amax(-1)               # [KV, n_q2, c2]
    ex2_rel = ex2 - ex2.mean(1, keepdim=True)
    ze2 = (ex2_rel - ex2_rel.mean(-1, keepdim=True)) / ex2_rel.std(-1, keepdim=True).clamp(min=1e-6)
    er2 = _ranks(ex2)
    e2_tier = er2.amin(1) * c2 + _ranks(ex2.amax(1))
    e2_z = _ranks(ze2.amax(1))
    u2 = min(k2, c2)
    inter2 = torch.stack(((-e2_z).topk(u2, dim=-1).indices,
                          (-e2_tier).topk(u2, dim=-1).indices), dim=-1).view(kv, 2 * u2)
    keep2 = _dedup_priority(inter2, u2)
    return cand2.gather(-1, keep2)


def _detect_global_agg(z_raw: torch.Tensor, full: int) -> bool:
    """Detect Dense Global Aggregation queries (high participation ratio across the archived timeline)."""
    if GLOBAL_AGG in ("0", "false", "off", "no"):
        return False
    if GLOBAL_AGG in ("1", "true", "on", "yes", "force"):
        return True
    if full < 128:
        return False
    z_tail = z_raw[:, -min(z_raw.shape[1], 16):].amax(1)  # [KV, full]
    z_1d = (z_tail - z_tail.mean(-1, keepdim=True)) / z_tail.std(-1, keepdim=True).clamp(min=1e-6)
    pos_mass = (z_1d - 1.0).clamp(min=0.0)
    l1 = pos_mass.sum(dim=-1)
    l2_sq = (pos_mass * pos_mass).sum(dim=-1).clamp(min=1e-6)
    n_eff = float((l1 * l1 / l2_sq).mean().item())
    z_peak = float(z_1d.amax().item())
    return n_eff > max(64.0, 0.12 * float(full)) and z_peak < 4.5


def _stratified_superblock_peaks(score_1d: torch.Tensor, full: int, max_sb: int) -> torch.Tensor:
    """Pick the highest-scoring block inside each uniform timeline segment across `[0, full)` -> `[KV, n_seg]`."""
    kv = score_1d.shape[0]
    n_seg = min(max(1, max_sb), full // max(1, SBLK))
    if n_seg <= 1:
        return score_1d.argmax(dim=-1, keepdim=True)
    seg_len = full // n_seg
    usable = n_seg * seg_len
    s_seg = score_1d[:, :usable].view(kv, n_seg, seg_len)
    local_best = s_seg.argmax(dim=-1)
    seg_base = (torch.arange(n_seg, device=score_1d.device, dtype=local_best.dtype) * seg_len).unsqueeze(0)
    return seg_base + local_best


def _record_block_visits(arc: Archive, blk: torch.Tensor, full: int) -> None:
    """Increment per-block decode visit counts for novelty-penalized global aggregation sweeps."""
    kv = blk.shape[0]
    cap_blk = arc.kmax.shape[1]
    if arc.visit_counts is None or arc.visit_counts.shape[1] < cap_blk:
        vc = torch.zeros(kv, cap_blk, dtype=torch.int16, device=blk.device)
        if arc.visit_counts is not None:
            vc[:, : arc.visit_counts.shape[1]] = arc.visit_counts
        arc.visit_counts = vc
    b_clamped = blk.clamp(min=0, max=cap_blk - 1)
    ones = torch.ones_like(b_clamped, dtype=torch.int16)
    arc.visit_counts.scatter_add_(1, b_clamped, ones)


def retrieve_blocks_per_token(arc: Archive, q: torch.Tensor, groups: int, limit: int, budget: int = 0,
                              lead: int | None = None, union: int = 0,
                              q_hist: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None] | None:
    """Per-query Z-score + tiered + 2-hop retrieval with [b, b+1] successor expansion, prompt-question retention,
    and Dense Global Aggregation support (stratified timeline coverage + novelty-rotating decode sweep)."""
    block = MEMORY["block_size"]
    full = (limit - ORIGIN) // block
    if full <= 0:
        return None
    eff_stride = GLOBAL_DECODE_STRIDE if arc.is_global_agg else DECODE_STRIDE
    if lead == 0 and q.shape[0] <= MEMORY["chunk"]:
        arc.decode_step += 1
        if (eff_stride > 1 and arc.cached_top is not None
                and arc.decode_step % eff_stride != 0):
            return arc.cached_top, None
    elif lead is not None and lead < 0:
        arc.decode_step = 0
        arc.last_hop_head = None
        arc.cached_top = arc.cached_rk = arc.cached_rv = None
        arc.cached_pre_k = arc.cached_pre_v = None
        arc.cached_ns = -1
        if arc.visit_counts is not None:
            arc.visit_counts.zero_()
    arc.refresh_bounds(block)
    q_step = q[:1] if (lead == 0 and q.shape[0] > 1) else q
    q_eval = (torch.cat((q_hist[:-1], q_step), dim=0)
              if (q_hist is not None and lead == 0 and q_hist.shape[0] > 1) else q_step)
    t_ev, h, d = q_eval.shape
    kv = h // groups
    ref_pos = limit
    s_dtype = q_eval.dtype if (q_eval.is_cuda and q_eval.dtype in (torch.bfloat16, torch.float16)) else torch.float32
    qg_s = q_eval.to(s_dtype).permute(1, 0, 2).reshape(kv, groups * t_ev, d)
    if ROPE_HORIZON > 0 and ref_pos > ROPE_HORIZON:
        qg_s = qg_s.clone()
        qg_s[..., : d // 4] = 0
    pos, neg = qg_s.clamp(min=0), qg_s.clamp(max=0)
    n_valid = full
    hole_b0 = hole_b1 = 0
    if arc.hole_hi > arc.hole_lo:
        hole_b0 = max(0, arc.hole_lo - ORIGIN) // block
        hole_b1 = min(full, (arc.hole_hi - ORIGIN + block - 1) // block)
        if hole_b1 > hole_b0:
            n_valid = full - (hole_b1 - hole_b0)
        if n_valid <= 0:
            return None
    if lead == 0 and not DECODE_RERANK:
        k_dec = min(DECODE_BLOCKS if DECODE_BLOCKS > 0 else (MEMORY["decode_blocks"] or MEMORY["blocks"]), n_valid)
        fine_idx = None
        if SBLK > 1 and arc.smax is not None and arc.smin is not None and full > SBLK * 16 and not arc.is_global_agg:
            n_sb = (full + SBLK - 1) // SBLK
            smax_t = arc.smax[:, :n_sb].to(s_dtype).transpose(-1, -2)
            smin_t = arc.smin[:, :n_sb].to(s_dtype).transpose(-1, -2)
            sb_score = (pos @ smax_t + neg @ smin_t).view(kv, groups, t_ev, n_sb).amax(1).float()
            m_sb = min(max(16, (2 * k_dec + SBLK - 1) // SBLK), n_sb)
            top_sb = sb_score.amax(1).topk(m_sb, dim=-1).indices
            sb_off = torch.arange(SBLK, device=q.device, dtype=top_sb.dtype)
            fine_raw = (top_sb.unsqueeze(-1) * SBLK + sb_off).reshape(kv, m_sb * SBLK)
            fine_valid = fine_raw < full
            fine_idx = fine_raw.clamp(max=full - 1)
            idx_d = fine_idx.unsqueeze(-1).expand(kv, m_sb * SBLK, d)
            kmax_t = arc.kmax[:, :full].gather(1, idx_d).to(s_dtype).transpose(-1, -2)
            kmin_t = arc.kmin[:, :full].gather(1, idx_d).to(s_dtype).transpose(-1, -2)
            score = (pos @ kmax_t + neg @ kmin_t).view(kv, groups, t_ev, m_sb * SBLK).amax(1).float()
            score = score.masked_fill(~fine_valid.unsqueeze(1), float("-inf"))
            if hole_b1 > hole_b0:
                in_hole = (fine_idx >= hole_b0) & (fine_idx < hole_b1)
                score = score.masked_fill(in_hole.unsqueeze(1), float("-inf"))
        else:
            kmax_t = arc.kmax[:, :full].to(s_dtype).transpose(-1, -2)
            kmin_t = arc.kmin[:, :full].to(s_dtype).transpose(-1, -2)
            score = (pos @ kmax_t + neg @ kmin_t).view(kv, groups, t_ev, full).amax(1).float()
            if hole_b1 > hole_b0:
                score[..., hole_b0:hole_b1] = float("-inf")
        n_pool = score.shape[-1]
        k_sel = min(k_dec, n_pool)
        s_fin = score.masked_fill(score.isinf(), 0.0)
        s_rel = score - s_fin.mean(1, keepdim=True) if t_ev > 1 else score
        s_rel_fin = s_rel.masked_fill(score.isinf(), 0.0)
        z_rel = (s_rel - s_rel_fin.mean(-1, keepdim=True)) / s_rel_fin.std(-1, keepdim=True).clamp(min=1e-6)
        z_raw = (score - s_fin.mean(-1, keepdim=True)) / s_fin.std(-1, keepdim=True).clamp(min=1e-6)
        if arc.is_global_agg and arc.visit_counts is not None and fine_idx is None:
            vc_pen = NOVELTY_ALPHA * torch.log1p(arc.visit_counts[:, :full].float()).unsqueeze(1)
            z_raw = z_raw - vc_pen
            z_rel = z_rel - vc_pen
        ranks = _ranks(score)
        r_tier = ranks.amin(1) * n_pool + _ranks(score.amax(1))
        if t_ev > 8:
            t_dec = min(t_ev, 16)
            r_tail = ranks[:, -8:].amin(1) * n_pool + ranks[:, -t_dec:].amin(1)
            r_z_rel = _ranks(z_rel[:, -t_dec:].amax(1))
            r_z_raw = _ranks(z_raw[:, -t_dec:].amax(1))
            keep_inter = torch.stack(((-r_tail).topk(k_sel, dim=-1).indices,
                                      (-r_z_rel).topk(k_sel, dim=-1).indices,
                                      (-r_z_raw).topk(k_sel, dim=-1).indices,
                                      (-r_tier).topk(k_sel, dim=-1).indices), dim=-1).view(kv, 4 * k_sel)
        else:
            r_z_rel = _ranks(z_rel.amax(1))
            r_z_raw = _ranks(z_raw.amax(1))
            keep_inter = torch.stack(((-r_z_rel).topk(k_sel, dim=-1).indices,
                                      (-r_z_raw).topk(k_sel, dim=-1).indices,
                                      (-r_tier).topk(k_sel, dim=-1).indices), dim=-1).view(kv, 3 * k_sel)
        keep_ord = _dedup_priority(keep_inter, k_sel)
        blk_ord = fine_idx.gather(-1, keep_ord) if fine_idx is not None else keep_ord
        if arc.is_global_agg and fine_idx is None and full >= SBLK * 2:
            strat = _stratified_superblock_peaks(z_raw.amax(1), full, max(8, k_sel // 2))
            n_top = max(1, k_sel // 2)
            blk_ord = _dedup_priority(torch.cat((blk_ord[:, :n_top], strat, blk_ord[:, n_top:]), dim=-1), k_sel)
        if arc.decode_top is not None and not arc.is_global_agg:
            d_prev = arc.decode_top.clamp(max=full - 1)
            k_d = min(d_prev.shape[1], blk_ord.shape[1])
            d_inter = torch.stack((blk_ord[:, :k_d], d_prev[:, :k_d]), dim=-1).view(kv, 2 * k_d)
            d_merged = torch.cat((d_inter, blk_ord[:, k_d:]), dim=-1) if k_d < blk_ord.shape[1] else d_inter
            blk_ord = _dedup_priority(d_merged, k_sel)
        arc.decode_top = blk_ord
        if kv > 1:
            blk_ord = _dedup_priority(blk_ord.transpose(0, 1).reshape(1, -1), k_sel).expand(kv, -1).contiguous()
        dec_prio, dec_sorted = _expand_successors_dedup(blk_ord, full, k_sel, m_succ=k_sel)
        if arc.prompt_top is not None:
            p_prev = arc.prompt_top.clamp(max=full - 1)
            k_out = min(p_prev.shape[1], n_valid)
            n_pin = (k_out // 2) if arc.is_global_agg else max(0, k_out - dec_prio.shape[1])
            merged = torch.cat((p_prev[:, :n_pin], dec_prio, p_prev[:, n_pin:]), dim=-1)
            top_prio = _dedup_priority(merged, k_out)
            blk, arc.last_prio_rank = top_prio.sort(-1)
        else:
            _, arc.last_prio_rank = dec_prio.sort(-1)
            blk = dec_sorted
        if arc.is_global_agg:
            _record_block_visits(arc, dec_prio, full)
        STATS["retrievals"] += 1
        STATS["pertoken"] += 1
        return blk, None
    qg = q_eval.float().permute(1, 0, 2).reshape(kv, groups * t_ev, d)                 # rows ordered (g, t)
    kmax_t = arc.kmax[:, :full].to(s_dtype).transpose(-1, -2)
    kmin_t = arc.kmin[:, :full].to(s_dtype).transpose(-1, -2)
    score = (pos @ kmax_t + neg @ kmin_t).view(kv, groups, t_ev, full).amax(1).float() # [KV, T_eval, full]
    if hole_b1 > hole_b0:
        score[..., hole_b0:hole_b1] = float("-inf")
    if not budget:
        budget = MEMORY["decode_blocks"] or MEMORY["blocks"]
    n_arch = max(full, (arc.prompt_len - MEMORY["window"] - ORIGIN) // block)
    for s_thresh, s_blocks in zip(SCALE[0::2], SCALE[1::2]):
        if n_arch > s_thresh:
            budget = max(budget, s_blocks)
    k = min(max(budget, 1024 if n_arch > 8192 else (512 if n_arch > 4096 else (384 if n_arch > 2048 else budget))), n_valid)
    rerank_pool = max(MEMORY["rerank"], 2 * k) if MEMORY["rerank"] > 0 else 0
    c0 = min(max(rerank_pool, k), n_valid)
    cand_mult = max(PERTOKEN_CAND, 2) if (lead is not None or t_ev > 1) else max(PERTOKEN_CAND, 1)
    if lead == 0 and arc.decode_step > 2 and arc.prompt_top is not None:
        c = min(max(k, 768), n_valid)
    elif lead == 0 and arc.prompt_top is not None:
        c = min(max(k, 1536), n_valid)
    else:
        c_floor = ((16384 if (lead is not None and lead < 0) else 8192) if n_arch > 8192
                   else (4096 if n_arch > 4096 else (3072 if n_arch > 2048 else 1536)))
        c = min(max(c0 * cand_mult, c_floor), n_valid)
    s_fin = score.masked_fill(score.isinf(), 0.0)
    s_rel = score - s_fin.mean(1, keepdim=True) if t_ev > 1 else score
    s_rel_fin = s_rel.masked_fill(score.isinf(), 0.0)
    z_score_rel = (s_rel - s_rel_fin.mean(-1, keepdim=True)) / s_rel_fin.std(-1, keepdim=True).clamp(min=1e-6)
    z_score_raw = (score - s_fin.mean(-1, keepdim=True)) / s_fin.std(-1, keepdim=True).clamp(min=1e-6)
    if lead is not None and lead < 0:
        arc.is_global_agg = _detect_global_agg(z_score_raw, full)
        if arc.is_global_agg:
            STATS["global_agg"] = STATS.get("global_agg", 0) + 1
    elif lead == 0 and arc.is_global_agg and arc.visit_counts is not None:
        vc_pen = NOVELTY_ALPHA * torch.log1p(arc.visit_counts[:, :full].float()).unsqueeze(1)
        z_score_raw = z_score_raw - vc_pen
        z_score_rel = z_score_rel - vc_pen
    ranks = _ranks(score)                                                              # [KV, T_eval, full]
    r_tier = ranks.amin(1) * full + _ranks(score.amax(1))                              # [KV, full]
    if lead is not None and lead < 0 and t_ev > 36:
        t_tail = min(t_ev, 96)
        r_tail = ranks[:, -t_tail:].amin(1) * full + _ranks(score[:, -t_tail:].amax(1))
        r_z_rel = _ranks(z_score_rel[:, -t_tail:].amax(1))
        r_z_raw = _ranks(z_score_raw[:, -t_tail:].amax(1))
        cand_inter = torch.stack(((-r_tail).topk(c, dim=-1).indices,
                                  (-r_z_rel).topk(c, dim=-1).indices,
                                  (-r_z_raw).topk(c, dim=-1).indices,
                                  (-r_tier).topk(c, dim=-1).indices), dim=-1).view(kv, 4 * c)
    elif lead == 0 and t_ev > 8:
        t_dec = min(t_ev, 16)
        r_tail = ranks[:, -8:].amin(1) * full + ranks[:, -t_dec:].amin(1)
        r_z_rel = _ranks(z_score_rel[:, -t_dec:].amax(1))
        r_z_raw = _ranks(z_score_raw[:, -t_dec:].amax(1))
        cand_inter = torch.stack(((-r_tail).topk(c, dim=-1).indices,
                                  (-r_z_rel).topk(c, dim=-1).indices,
                                  (-r_z_raw).topk(c, dim=-1).indices,
                                  (-r_tier).topk(c, dim=-1).indices), dim=-1).view(kv, 4 * c)
    else:
        r_z_rel = _ranks(z_score_rel.amax(1))
        r_z_raw = _ranks(z_score_raw.amax(1))
        cand_inter = torch.stack(((-r_z_rel).topk(c, dim=-1).indices,
                                  (-r_z_raw).topk(c, dim=-1).indices,
                                  (-r_tier).topk(c, dim=-1).indices), dim=-1).view(kv, 3 * c)
    cand = _dedup_priority(cand_inter, c)                                              # [KV, c]
    arc.last_top = arc.last_rk = None
    if MEMORY["rerank"]:
        exact_parts = []
        for c_s in range(0, c, 1024):
            c_e = min(c, c_s + 1024)
            cand_part = cand[:, c_s:c_e]
            ck_part = _gather_blocks(arc.keys, cand_part, block).to(device=q.device, dtype=torch.float32)
            if ROPE_HORIZON > 0 and ref_pos > ROPE_HORIZON:
                ck_part = _reposition_rope_blocks(ck_part, cand_part, block, ref_pos, contiguous_runs=False, span=0)
            ex_part = torch.bmm(qg, ck_part.transpose(-1, -2)).view(kv, groups, t_ev, c_e - c_s, block).amax(-1).amax(1)
            exact_parts.append(ex_part)
        exact = torch.cat(exact_parts, dim=-1) if len(exact_parts) > 1 else exact_parts[0]
    else:
        exact = score.gather(-1, cand.unsqueeze(1).expand(kv, t_ev, c))
    e_fin = exact.masked_fill(exact.isinf(), 0.0)
    e_rel = exact - e_fin.mean(1, keepdim=True) if t_ev > 1 else exact
    e_rel_fin = e_rel.masked_fill(exact.isinf(), 0.0)
    z_exact_rel = (e_rel - e_rel_fin.mean(-1, keepdim=True)) / e_rel_fin.std(-1, keepdim=True).clamp(min=1e-6)
    z_exact_raw = (exact - e_fin.mean(-1, keepdim=True)) / e_fin.std(-1, keepdim=True).clamp(min=1e-6)
    er = _ranks(exact)                                                                 # [KV, T_eval, c]
    e_tier = er.amin(1) * c + _ranks(exact.amax(1))                                    # [KV, c]
    if lead is not None and lead < 0 and t_ev > 36:
        t_tail = min(t_ev, 96)
        e_tail = er[:, -t_tail:].amin(1) * c + _ranks(exact[:, -t_tail:].amax(1))
        e_z_rel = _ranks(z_exact_rel[:, -t_tail:].amax(1))
        e_z_raw = _ranks(z_exact_raw[:, -t_tail:].amax(1))
        keep_inter = torch.stack(((-e_z_rel).topk(k, dim=-1).indices,
                                  (-e_z_raw).topk(k, dim=-1).indices,
                                  (-e_tail).topk(k, dim=-1).indices,
                                  (-e_tier).topk(k, dim=-1).indices), dim=-1).view(kv, 4 * k)
    elif lead == 0 and t_ev > 8:
        t_dec = min(t_ev, 16)
        e_tail = er[:, -8:].amin(1) * c + er[:, -t_dec:].amin(1)
        e_z_rel = _ranks(z_exact_rel[:, -t_dec:].amax(1))
        e_z_raw = _ranks(z_exact_raw[:, -t_dec:].amax(1))
        keep_inter = torch.stack(((-e_tail).topk(k, dim=-1).indices,
                                  (-e_z_rel).topk(k, dim=-1).indices,
                                  (-e_z_raw).topk(k, dim=-1).indices,
                                  (-e_tier).topk(k, dim=-1).indices), dim=-1).view(kv, 4 * k)
    else:
        e_z_rel = _ranks(z_exact_rel.amax(1))
        e_z_raw = _ranks(z_exact_raw.amax(1))
        keep_inter = torch.stack(((-e_z_rel).topk(k, dim=-1).indices,
                                  (-e_z_raw).topk(k, dim=-1).indices,
                                  (-e_tier).topk(k, dim=-1).indices), dim=-1).view(kv, 3 * k)
    keep_ord = _dedup_priority(keep_inter, k)                                          # [KV, k]
    blk_ord = cand.gather(-1, keep_ord)                                                # [KV, k]
    if arc.is_global_agg and full >= SBLK * 2:
        strat = _stratified_superblock_peaks(z_score_raw.amax(1), full, max(16, k // 4))
        n_keep_top = max(1, (3 * k) // 4)
        blk_ord = _dedup_priority(torch.cat((blk_ord[:, :n_keep_top], strat, blk_ord[:, n_keep_top:]), dim=-1), k)
    if MEMORY["rerank"] and n_valid > 16:
        n_seed = min(8, blk_ord.shape[1])
        cur_head = blk_ord[:, :n_seed]
        run_hop2 = (lead is None or lead < 0 or arc.decode_step <= 2 or arc.last_hop_head is None
                    or cur_head.shape != arc.last_hop_head.shape or not torch.equal(cur_head, arc.last_hop_head))
        if run_hop2:
            arc.last_hop_head = cur_head
            c2_cap = 256 if (lead == 0 and arc.decode_step > 2) else (384 if lead == 0 else 512)
            hop2 = _second_hop_blocks(arc, blk_ord, full, n_valid, block, k2=min(64, max(32, k // 4)), c2_cap=c2_cap)
            n_head = min(16, blk_ord.shape[1])
            blk_ord = _dedup_priority(torch.cat((blk_ord[:, :n_head], hop2, blk_ord[:, n_head:]), dim=-1), k)
    if lead is not None and lead < 0:
        arc.decode_top = None
        merged_ord = blk_ord
    elif arc.prompt_top is not None:
        if arc.decode_top is not None and not arc.is_global_agg:
            d_prev = arc.decode_top.clamp(max=full - 1)
            k_d = min(d_prev.shape[1], blk_ord.shape[1])
            d_inter = torch.stack((blk_ord[:, :k_d], d_prev[:, :k_d]), dim=-1).view(kv, 2 * k_d)
            d_merged = torch.cat((d_inter, blk_ord[:, k_d:]), dim=-1) if k_d < blk_ord.shape[1] else d_inter
            blk_ord = _dedup_priority(d_merged, k)
        arc.decode_top = blk_ord
        p_prev = arc.prompt_top.clamp(max=full - 1)
        k_p = min(p_prev.shape[1], blk_ord.shape[1])
        inter = torch.stack((p_prev[:, :k_p], blk_ord[:, :k_p]), dim=-1).view(kv, 2 * k_p)
        merged = torch.cat((inter, blk_ord[:, k_p:]), dim=-1) if k_p < blk_ord.shape[1] else inter
        merged_ord = _dedup_priority(merged, k)
    else:
        merged_ord = blk_ord
    m_succ = min(640, k) if k > 1024 else k
    top_prio, _ = _expand_successors_dedup(merged_ord, full, k, m_succ=m_succ)
    blk, arc.last_prio_rank = top_prio.sort(-1)
    if lead is not None and lead < 0:
        arc.prompt_top = top_prio
    if arc.is_global_agg:
        _record_block_visits(arc, top_prio, full)
    STATS["retrievals"] += 1
    STATS["pertoken"] += 1
    return blk, None


_CU_TABLES: dict[torch.device, torch.Tensor] = {}


def _get_cu(arc: Archive, t: int, w_len: int, p_len: int, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    tbl = _CU_TABLES.get(device)
    max_n = max(t, w_len, p_len) + 1
    if tbl is None or tbl.shape[0] < max_n:
        cap = max(131072, 1 << max_n.bit_length())
        tbl = torch.stack((torch.zeros(cap, device=device, dtype=torch.int32),
                           torch.arange(cap, device=device, dtype=torch.int32)), dim=-1)
        _CU_TABLES[device] = tbl
    return tbl[t], tbl[w_len], (tbl[p_len] if p_len > 0 else None)


def piece_attention(arc: Archive, q: torch.Tensor, q_pos: torch.Tensor, top: torch.Tensor | None, groups: int,
                    window: int, scale: float, own: torch.Tensor | None = None,
                    q_min: int | None = None, q_max: int | None = None) -> torch.Tensor:
    """Exact attention of piece q [T, H, D] at positions q_pos over its sliding window (from the archive) plus the
    retrieved blocks `top` [KV, k] (and super-block summary landmarks when `arc.is_global_agg` is active);
    with `own` [KV, T, k] each query sees only its own retrieved blocks. Returns [T, H, D]."""
    block = MEMORY["block_size"]
    t, h, d = q.shape
    kv = h // groups
    if q_min is None or q_max is None:
        q_min, q_max = int(q_pos.min()), int(q_pos.max())
        contig_q = (t == 1 or (q_max - q_min == t - 1 and int(q_pos[0]) == q_min and int(q_pos[-1]) == q_max))
    else:
        contig_q = (t == 1 or (q_max - q_min == t - 1))
    lo = max(0, q_min - window + 1)
    hi = q_max + 1
    use_lse = (own is None and LSE and HAS_FA2 and d <= 256 and q.is_cuda and q.dtype in (torch.bfloat16, torch.float16) and contig_q)
    tail0 = ORIGIN + ((lo - ORIGIN) // block) * block
    fold_tail = bool(use_lse and t <= MEMORY["chunk"] and FAITHFUL and top is not None and lo > tail0)
    w_lo = tail0 if fold_tail else lo
    ns = min(SINKS, lo)                                                                # sink positions below the window
    n_sb_valid = max(0, ((lo - ORIGIN) // block) // SBLK) if (arc.is_global_agg and arc.smean_k is not None and arc.smean_v is not None) else 0
    if use_lse:
        if arc.gpu_k is not None and w_lo >= arc.gpu_lo and hi <= arc.gpu_hi:
            wk_t = arc.gpu_k[w_lo - arc.gpu_lo : hi - arc.gpu_lo].to(dtype=q.dtype)    # [Wn, KV, D] zero-copy slice
            wv_t = arc.gpu_v[w_lo - arc.gpu_lo : hi - arc.gpu_lo].to(dtype=q.dtype)
        else:
            wk_t = arc.keys[:, w_lo:hi].to(device=q.device, dtype=q.dtype).transpose(0, 1)
            wv_t = arc.values[:, w_lo:hi].to(device=q.device, dtype=q.dtype).transpose(0, 1)
        if t <= MEMORY["chunk"] and top is not None:
            if (arc.cached_top is not None and arc.cached_pre_k is not None and arc.cached_pre_v is not None
                    and arc.cached_ns == ns and arc.cached_top.shape == top.shape
                    and (top is arc.cached_top or torch.equal(top, arc.cached_top))):
                pre_k = arc.cached_pre_k.to(dtype=q.dtype)
                pre_v = arc.cached_pre_v.to(dtype=q.dtype)
            else:
                rk_raw, rv = _gather_cached_delta(arc, top, block, q.dtype, q.device)
                arc.last_top = arc.last_rk = None
                arc.cached_top = top
                arc.cached_rk = rk_raw
                arc.cached_rv = rv
                ref_pos = lo
                rk = _reposition_rope_blocks(rk_raw, top, block, ref_pos, contiguous_runs=True, prio_rank=arc.last_prio_rank)
                pre_k_list, pre_v_list = [], []
                if ns > 0:
                    if arc.sink_k is not None and ns <= arc.sink_k.shape[0]:
                        sk_kv = arc.sink_k[:ns].transpose(0, 1).to(dtype=q.dtype)
                        sv_t = arc.sink_v[:ns].to(dtype=q.dtype)
                    else:
                        sk_kv = arc.keys[:, :ns].to(device=q.device, dtype=q.dtype)
                        sv_t = arc.values[:, :ns].to(device=q.device, dtype=q.dtype).transpose(0, 1)
                    sk_t = _reposition_rope_sinks(sk_kv, ref_pos, n_blk=top.shape[1], block=block).transpose(0, 1)
                    pre_k_list.append(sk_t)
                    pre_v_list.append(sv_t)
                if n_sb_valid > 0:
                    smk = _reposition_rope_landmarks(arc.smean_k[:, :n_sb_valid].to(dtype=q.dtype), block, ref_pos).transpose(0, 1)
                    smv = arc.smean_v[:, :n_sb_valid].to(dtype=q.dtype).transpose(0, 1)
                    pre_k_list.append(smk)
                    pre_v_list.append(smv)
                pre_k_list.append(rk.transpose(0, 1))
                pre_v_list.append(rv.transpose(0, 1))
                pre_k = pre_k_list[0].contiguous() if len(pre_k_list) == 1 else torch.cat(pre_k_list, dim=0).contiguous()
                pre_v = pre_v_list[0].contiguous() if len(pre_v_list) == 1 else torch.cat(pre_v_list, dim=0).contiguous()
                arc.cached_ns = ns
                arc.cached_pre_k = pre_k
                arc.cached_pre_v = pre_v
            cu_q, cu_w, cu_p = _get_cu(arc, t, wk_t.shape[0], pre_k.shape[0], q.device)
            causal_w = t > 1
            win_r = 0 if causal_w else -1
            out_w, lse_w = torch.ops._vllm_fa2_C.varlen_fwd(
                q, wk_t, wv_t, None, cu_q, cu_w, None, None, None, None,
                t, wk_t.shape[0], 0.0, scale, False, causal_w, -1, win_r, 0.0, False, 0, None
            )
            out_p, lse_p = torch.ops._vllm_fa2_C.varlen_fwd(
                q, pre_k, pre_v, None, cu_q, cu_p, None, None, None, None,
                t, pre_k.shape[0], 0.0, scale, False, False, -1, -1, 0.0, False, 0, None
            )
            w = torch.sigmoid(lse_p - lse_w).transpose(0, 1).unsqueeze(-1)
            return torch.lerp(out_w.float(), out_p.float(), w).to(q.dtype)
    elif arc.gpu_k is not None and lo >= arc.gpu_lo and hi <= arc.gpu_hi:
        wk = arc.gpu_k[lo - arc.gpu_lo : hi - arc.gpu_lo].transpose(0, 1).to(dtype=q.dtype)
        wv = arc.gpu_v[lo - arc.gpu_lo : hi - arc.gpu_lo].transpose(0, 1).transpose(0, 1) if False else arc.gpu_v[lo - arc.gpu_lo : hi - arc.gpu_lo].transpose(0, 1).to(dtype=q.dtype)
    else:
        wk = arc.keys[:, lo:hi].to(device=q.device).to(dtype=q.dtype)                  # [KV, Wn, D]
        wv = arc.values[:, lo:hi].to(device=q.device).to(dtype=q.dtype)
    unmasked = (t == 1 and own is None)
    need_mask = not (use_lse or unmasked)
    parts_k, parts_v, parts_allow = [], [], []
    ref_pos = lo
    n_blk_cur = top.shape[1] if top is not None else 0
    if ns > 0:
        if arc.sink_k is not None and ns <= arc.sink_k.shape[0]:
            sk_kv = arc.sink_k[:ns].transpose(0, 1).to(dtype=q.dtype)
            parts_v.append(arc.sink_v[:ns].transpose(0, 1).to(dtype=q.dtype))
        else:
            sk_kv = arc.keys[:, :ns].to(device=q.device).to(dtype=q.dtype)
            parts_v.append(arc.values[:, :ns].to(device=q.device).to(dtype=q.dtype))
        parts_k.append(_reposition_rope_sinks(sk_kv, ref_pos, n_blk=n_blk_cur, block=block))
        if need_mask:
            parts_allow.append(torch.ones(t, ns, dtype=torch.bool, device=q.device))
    if n_sb_valid > 0:
        smk = _reposition_rope_landmarks(arc.smean_k[:, :n_sb_valid].to(dtype=q.dtype), block, ref_pos)
        smv = arc.smean_v[:, :n_sb_valid].to(dtype=q.dtype)
        parts_k.append(smk)
        parts_v.append(smv)
        if need_mask:
            parts_allow.append(torch.ones(t, n_sb_valid, dtype=torch.bool, device=q.device))
    if top is not None:
        if arc.last_top is top and arc.last_rk is not None:
            rk_raw = arc.last_rk.to(dtype=q.dtype)
        else:
            rk_raw = _gather_blocks(arc.keys, top, block).to(device=q.device).to(dtype=q.dtype)       # [KV, R, D]
        arc.last_top = arc.last_rk = None
        rv = _gather_blocks(arc.values, top, block).to(device=q.device).to(dtype=q.dtype)
        rk = _reposition_rope_blocks(rk_raw, top, block, ref_pos, contiguous_runs=True, prio_rank=arc.last_prio_rank)
        parts_k.append(rk); parts_v.append(rv)
        if own is None and arc.prompt_top is not None:
            arc.cached_top = top
            arc.cached_rk = rk_raw
            arc.cached_rv = rv
            arc.cached_ns = ns
            arc.cached_pre_k = (parts_k[0] if len(parts_k) == 1 else torch.cat(parts_k, dim=1)).transpose(0, 1).contiguous()
            arc.cached_pre_v = (parts_v[0] if len(parts_v) == 1 else torch.cat(parts_v, dim=1)).transpose(0, 1).contiguous()
        else:
            arc.cached_top = arc.cached_rk = arc.cached_rv = None
            arc.cached_pre_k = arc.cached_pre_v = None
        if need_mask:
            if own is not None:
                parts_allow.append(own.to(q.device).repeat_interleave(block, dim=-1))      # [KV, T, R]
            else:
                parts_allow.append(torch.ones(t, rk.shape[1], dtype=torch.bool, device=q.device))
        if FAITHFUL and lo > tail0:   # the PyTorch tier always keeps the partial block just older than the window
            if arc.gpu_k is not None and tail0 >= arc.gpu_lo and lo <= arc.gpu_hi:
                parts_k.append(arc.gpu_k[tail0 - arc.gpu_lo : lo - arc.gpu_lo].transpose(0, 1).to(dtype=q.dtype))
                parts_v.append(arc.gpu_v[tail0 - arc.gpu_lo : lo - arc.gpu_lo].transpose(0, 1).to(dtype=q.dtype))
            else:
                parts_k.append(arc.keys[:, tail0:lo].to(device=q.device).to(dtype=q.dtype))
                parts_v.append(arc.values[:, tail0:lo].to(device=q.device).to(dtype=q.dtype))
            if need_mask:
                parts_allow.append(torch.ones(t, lo - tail0, dtype=torch.bool, device=q.device))
    if use_lse:
        if not parts_k:
            cu_q, cu_w, _ = _get_cu(arc, t, wk_t.shape[0], 0, q.device)
            out_w, _ = torch.ops._vllm_fa2_C.varlen_fwd(
                q, wk_t, wv_t, None, cu_q, cu_w, None, None, None, None,
                t, wk_t.shape[0], 0.0, scale, False, True, window - 1, 0, 0.0, False, 0, None
            )
            return out_w.to(q.dtype)
        pre_k = (parts_k[0] if len(parts_k) == 1 else torch.cat(parts_k, dim=1)).transpose(0, 1)
        pre_v = (parts_v[0] if len(parts_v) == 1 else torch.cat(parts_v, dim=1)).transpose(0, 1)
        cu_q, cu_w, cu_p = _get_cu(arc, t, wk_t.shape[0], pre_k.shape[0], q.device)
        out_w, lse_w = torch.ops._vllm_fa2_C.varlen_fwd(
            q, wk_t, wv_t, None, cu_q, cu_w, None, None, None, None,
            t, wk_t.shape[0], 0.0, scale, False, True, window - 1, 0, 0.0, False, 0, None
        )
        out_p, lse_p = torch.ops._vllm_fa2_C.varlen_fwd(
            q, pre_k, pre_v, None, cu_q, cu_p, None, None, None, None,
            t, pre_k.shape[0], 0.0, scale, False, False, -1, -1, 0.0, False, 0, None
        )
        w = torch.sigmoid(lse_p - lse_w).transpose(0, 1).unsqueeze(-1)
        return torch.lerp(out_w.float(), out_p.float(), w).to(q.dtype)
    parts_k.append(wk); parts_v.append(wv)
    if need_mask:
        wpos = torch.arange(lo, hi, device=q.device)
        allow = (wpos[None, :] <= q_pos[:, None]) & (wpos[None, :] >= q_pos[:, None] - window + 1)   # [T, Wn]
        parts_allow.append(allow)
    keys, vals = torch.cat(parts_k, dim=1), torch.cat(parts_v, dim=1)
    # Grouped-query layout: query head h uses KV head h // groups, so stack each group's queries along the
    # sequence axis ([KV, groups*T, D]) instead of replicating K/V per query head.
    qg = q.transpose(0, 1).reshape(kv, groups * t, d)                                  # [KV, G*T, D]
    if not need_mask:
        mask = None
    elif own is not None:   # per-KV-head masks [KV, G*T, N]
        allow = torch.cat([a if a.dim() == 3 else a.expand(kv, -1, -1) for a in parts_allow], dim=-1)
        mask = allow.repeat(1, groups, 1)[None]
    else:
        mask = torch.cat(parts_allow, dim=1).repeat(groups, 1)[None, None]
    out = F.scaled_dot_product_attention(qg.unsqueeze(0), keys.unsqueeze(0), vals.unsqueeze(0),
                                         attn_mask=mask, scale=scale)
    return out[0].reshape(h, t, d).transpose(0, 1).to(q.dtype)


@dataclass
class _SyntheticInputBatch:
    """Zero-hook fallback batch descriptor synthesized directly from ForwardContext / positions when
    upstream vLLM runner internals change."""
    num_tokens: int
    req_ids: list[str]
    query_start_loc_np: list[int]


def _fallback_batch_from_forward_context(positions: torch.Tensor, q: torch.Tensor) -> _SyntheticInputBatch | None:
    """Synthesize an InputBatch-compatible object from vLLM ForwardContext or position resets when runner hooks
    are unavailable."""
    pos = positions[0] if positions.dim() == 2 else positions
    t = int(min(pos.numel(), q.shape[0]))
    if t <= 0:
        return None
    try:
        from vllm.forward_context import get_forward_context
        ctx = get_forward_context()
        attn_meta = getattr(ctx, "attn_metadata", None)
        if isinstance(attn_meta, dict) and attn_meta:
            attn_meta = next(iter(attn_meta.values()))
        qsl = getattr(attn_meta, "query_start_loc", None)
        if qsl is not None and isinstance(qsl, torch.Tensor) and qsl.numel() >= 2:
            qsl_list = [int(x) for x in qsl.detach().cpu().tolist() if int(x) <= t]
            if len(qsl_list) >= 2 and qsl_list[-1] > 0:
                n_req = len(qsl_list) - 1
                return _SyntheticInputBatch(
                    num_tokens=qsl_list[-1],
                    req_ids=[f"pb27_ctx_{i}" for i in range(n_req)],
                    query_start_loc_np=qsl_list,
                )
    except Exception:
        pass
    return _SyntheticInputBatch(num_tokens=t, req_ids=["pb27_default_req"], query_start_loc_np=[0, t])


def install_runner_hooks() -> None:
    """Expose the runner's current InputBatch and free archives of finished requests across both vLLM V1
    (`vllm.v1.worker.gpu_model_runner`) and modular V1/V2 (`vllm.v1.worker.gpu.model_runner`) layouts."""
    try:
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        from vllm.v1.worker.gpu.states import RequestState

        if getattr(GPUModelRunner, "_pb27_hooked", False):
            return
        orig_prepare = GPUModelRunner.prepare_inputs
        orig_add = RequestState.add_request
        orig_remove = RequestState.remove_request

        def prepare_inputs(self, *args, **kwargs):
            ib = orig_prepare(self, *args, **kwargs)
            BATCH["ib"] = ib
            BATCH["pos_list"] = None
            BATCH["req_bounds"] = None
            return ib

        def add_request(self, req_id, prompt_len, *args, **kwargs):
            PROMPT_LENS[req_id] = int(prompt_len)
            all_token_ids = kwargs.get("all_token_ids", args[0] if len(args) > 0 else None)
            num_computed = kwargs.get("num_computed_tokens", args[1] if len(args) > 1 else 0)
            if all_token_ids is not None and num_computed:
                adopt_for_request(req_id, all_token_ids, int(num_computed))
            if int(prompt_len) > 262144 and not ADOPTED.get(req_id, ({},))[0] and len(RETIRED) >= max(1, RETAIN):
                while RETIRED:
                    _free(RETIRED.pop(0)["archives"])
            return orig_add(self, req_id, prompt_len, *args, **kwargs)

        def remove_request(self, req_id, *args, **kwargs):
            tokens = None
            if RETAIN > 0 and req_id in self.req_id_to_index:
                idx = self.req_id_to_index[req_id]
                try:
                    n = int(self.total_len.gpu[idx])
                    tokens = self.all_token_ids.gpu[idx, :n].clone()
                except Exception as exc:
                    log.warning("pb27 memory: cannot read token ids for %s (%s); not retaining", req_id, exc)
            drop_request(req_id, tokens)
            return orig_remove(self, req_id, *args, **kwargs)

        GPUModelRunner.prepare_inputs = prepare_inputs
        RequestState.add_request = add_request
        RequestState.remove_request = remove_request
        GPUModelRunner._pb27_hooked = True
        log.info("pb27 memory tier: runner hooks installed (V2 layout, pid %d), config %s", os.getpid(), MEMORY)
        return
    except ImportError:
        pass

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner as V1GPUModelRunner

    if getattr(V1GPUModelRunner, "_pb27_hooked", False):
        return
    orig_update = getattr(V1GPUModelRunner, "_update_states", None)
    if orig_update is not None:
        def _update_states(self, scheduler_output, *args, **kwargs):
            for req_id in getattr(scheduler_output, "finished_req_ids", ()):
                drop_request(req_id, None)
            for new_req in getattr(scheduler_output, "scheduled_new_reqs", ()):
                r_id = getattr(new_req, "req_id", None)
                p_ids = getattr(new_req, "prompt_token_ids", None)
                if r_id is not None and p_ids is not None:
                    PROMPT_LENS[r_id] = len(p_ids)
            res = orig_update(self, scheduler_output, *args, **kwargs)
            BATCH["ib"] = getattr(self, "input_batch", None)
            BATCH["pos_list"] = None
            BATCH["req_bounds"] = None
            return res

        V1GPUModelRunner._update_states = _update_states
        V1GPUModelRunner._pb27_hooked = True
        log.info("pb27 memory tier: runner hooks installed (V1 layout, pid %d), config %s", os.getpid(), MEMORY)


def _layer_enabled(layer_name: str) -> bool:
    if LAYER_STRIDE <= 1:
        return True
    for part in layer_name.split("."):
        if part.isdigit():
            return (int(part) % LAYER_STRIDE) == 0
    return True


def _kv_cache_update_only(attn_module, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor | None:
    """Update vLLM's paged KV cache without running the base sliding-window attention kernel when every output
    row in `out[:T]` will be overwritten by `piece_attention`."""
    try:
        if (
            getattr(attn_module.attn_backend, "forward_includes_kv_cache_update", True)
            or getattr(attn_module, "kv_sharing_target_layer_name", None) is not None
            or k is None
            or v is None
        ):
            return None
        from vllm.model_executor.layers.attention.attention import (
            _encode_layer_name,
            unified_kv_cache_update,
        )
        key = k.view(-1, attn_module.num_kv_heads, attn_module.head_size)
        val = v.view(-1, attn_module.num_kv_heads, attn_module.head_size_v)
        if getattr(attn_module, "use_direct_call", False):
            unified_kv_cache_update(key, val, attn_module.layer_name)
        else:
            torch.ops.vllm.unified_kv_cache_update(key, val, _encode_layer_name(attn_module.layer_name))
        return torch.empty((q.shape[0], attn_module.num_heads * attn_module.head_size_v), dtype=q.dtype, device=q.device)
    except Exception:
        return None


def windowed_attention_with_memory(attn_module, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                   positions: torch.Tensor) -> torch.Tensor:
    """vLLM's window attention, then exact recomputation over [retrieved + window] for the pieces that retrieve."""
    ib = BATCH["ib"]
    STATS["calls"] += 1
    if DEBUG and STATS["calls"] % 500 == 1:
        log.info("pb27 memory: call %d ib=%s archives=%d stats=%s", STATS["calls"], ib is not None, len(ARCHIVES), STATS)
    if ib is None or not getattr(ib, "req_ids", None):
        ib = _fallback_batch_from_forward_context(positions, q)
        if ib is None or not ib.req_ids:
            STATS["fallback"] += 1
            return attn_module(q, k, v)
    layer = attn_module.layer_name
    if not _layer_enabled(layer):
        return attn_module(q, k, v)
    impl = attn_module.impl
    H, KV, D = impl.num_heads, impl.num_kv_heads, impl.head_size
    groups = H // KV
    window = getattr(attn_module, "sliding_window", None) or MEMORY["window"]
    T = ib.num_tokens
    qh, kh, vh = q[:T].view(T, H, D), k[:T].view(T, KV, D), v[:T].view(T, KV, D)
    pos = positions[0, :T] if positions.dim() == 2 else positions[:T]
    qsl = ib.query_start_loc_np
    block, piece = MEMORY["block_size"], MEMORY["chunk"]
    req_bounds = BATCH.get("req_bounds")
    if req_bounds is None or BATCH.get("req_bounds_T") != T:
        idx_pts = []
        for i in range(len(ib.req_ids)):
            s, e = int(qsl[i]), int(qsl[i + 1])
            if e > s:
                idx_pts.extend((s, e - 1))
        pts = pos[torch.tensor(idx_pts, device=pos.device, dtype=torch.long)].tolist() if idx_pts else []
        req_bounds = {}
        p_it = iter(pts)
        for i in range(len(ib.req_ids)):
            s, e = int(qsl[i]), int(qsl[i + 1])
            if e > s:
                req_bounds[i] = (int(next(p_it)), int(next(p_it)))
        BATCH["req_bounds"] = req_bounds
        BATCH["req_bounds_T"] = T
        if any(p0 == 0 for p0, _ in req_bounds.values()):
            try:
                mt = os.path.getmtime(__file__)
                if mt > _MTIME:
                    import importlib
                    import sys
                    importlib.reload(sys.modules[__name__])
                    return sys.modules[__name__].windowed_attention_with_memory(attn_module, q, k, v, positions)
            except Exception:
                pass
    pos_list = BATCH.get("pos_list")
    skip_base = not NORETRIEVE and not ORACLE
    if skip_base:
        for i, req_id in enumerate(ib.req_ids):
            s, e = int(qsl[i]), int(qsl[i + 1])
            if e <= s:
                continue
            p_first, p_last = req_bounds[i]
            arc_existing = ARCHIVES.get((req_id, layer))
            has_hole = req_id in ADOPTED or (arc_existing is not None and arc_existing.hole_hi > arc_existing.hole_lo)
            full_cov = (e - s) <= piece or PREFILL == "1"
            if has_hole or not full_cov or (p_last - p_first != e - s - 1) or (p_first - window + 1 < ORIGIN + block):
                skip_base = False
                break
    out = _kv_cache_update_only(attn_module, q, k, v) if skip_base else None
    if out is None:
        out = attn_module(q, k, v)
    out_h = out[:T].view(T, H, D)
    for i, req_id in enumerate(ib.req_ids):
        s, e = int(qsl[i]), int(qsl[i + 1])
        if e <= s:
            continue
        p_first, p_last = req_bounds[i]
        contig_req = (p_last - p_first == e - s - 1)
        if not contig_req and (pos_list is None or len(pos_list) != T):
            pos_list = pos.tolist()
            BATCH["pos_list"] = pos_list
        key = (req_id, layer)
        arc = ARCHIVES.get(key)
        if arc is None:
            plen = PROMPT_LENS.get(req_id, 0)
            if max(plen, p_last + 1) + 1024 <= window and req_id not in ADOPTED and RETAIN <= 0:
                continue
            adopted, h_lo, h_hi, n_cached = ADOPTED.get(req_id, ({}, 0, 0, 0))
            arc = adopted.pop(layer, None)
            if arc is None:
                max_hint = max(plen, p_last + 1) + 2048
                init_cap = max(block * 16, 1 << 10, 1 << (max_hint - 1).bit_length())
                arc = Archive.new(KV, D, init_cap, block, kh.dtype, kh.device, max_hint=max_hint)
                h_lo, h_hi = (0, n_cached) if n_cached else (0, 0)   # cached prefix with no archive
            else:
                valid_end = n_cached if h_hi <= h_lo else h_lo
                arc.size = min(arc.size, valid_end)
                if arc.gpu_k is not None:
                    if valid_end < arc.gpu_lo:
                        arc.gpu_k = arc.gpu_v = None
                    else:
                        arc.gpu_hi = min(arc.gpu_hi, valid_end)
                if valid_end > ORIGIN:
                    b_tail = (valid_end - 1 - ORIGIN) // block
                    arc.dirty_lo, arc.dirty_hi = min(arc.dirty_lo, b_tail), max(arc.dirty_hi, b_tail)
            arc.hole_lo, arc.hole_hi = h_lo, h_hi
            arc.prompt_len = plen
            arc.prompt_top = None
            arc.decode_top = None
            arc.decode_step = 0
            arc.last_hop_head = None
            arc.cached_top = arc.cached_rk = arc.cached_rv = None
            arc.cached_pre_k = arc.cached_pre_v = None
            arc.cached_ns = -1
            arc.cached_cu = None
            ARCHIVES[key] = arc
        arc.write(pos[s:e], kh[s:e], vh[s:e], block, p0=p_first if contig_req else None, hi=(p_last + 1) if contig_req else None)
        if QWIN or PERTOKEN:
            arc.write_q(pos[s:e], qh[s:e])
        rerank = (e - s) <= piece or RERANK_ALL   # the PyTorch tier scores the question in its own <=64-token
        is_prefill = (e - s) > piece               # chunk with exact rescoring; here it sits inside a prefill chunk
        if is_prefill:
            arc.prompt_len = max(arc.prompt_len, p_last + 1)
        if p_last + 1 <= window:
            continue                                      # the whole context still fits the window
        if is_prefill and PREFILL == "0":
            continue
        # pieces are aligned to the END of the chunk, so the last piece is always a full window of the most recent
        # tokens (the question) instead of a remainder of a few tokens
        bounds = [(max(s, p1 - piece), p1) for p1 in range(e, s, -piece)]
        q_piece = max(piece, 160)
        if is_prefill and PREFILL == "pt":
            # Keep vLLM's clean shift-invariant sliding-window output for haystack chunks and run exact rescoring
            # retrieval + piece_attention only on the final question piece `(cut, e)` at the end of the prompt.
            at_prompt_end = p_last + 1 >= arc.prompt_len
            cut = (e - q_piece if e - s > q_piece else s) if at_prompt_end else e
            bounds = [(cut, e)] if cut < e else []
        elif is_prefill and PREFILL not in ("0", "1"):
            bounds = bounds[:TAIL]                 # retrieval during prefill only in the last TAIL pieces of the chunk
        for p0, p1 in bounds:
            q_pos = pos[p0:p1]
            if contig_req:
                q_min, q_max = p_first + (p0 - s), p_first + (p1 - 1 - s)
            else:
                q_min, q_max = int(pos_list[p0]), int(pos_list[p1 - 1])
            limit = q_min - window + 1                        # archived positions older than every query's window
            if arc.hole_hi > arc.hole_lo:
                w_lo, w_hi = max(0, limit), q_max + 1
                sinks_missing = SINKS > 0 and w_lo > 0 and arc.hole_lo < SINKS
                if (arc.hole_lo < w_hi and arc.hole_hi > w_lo) or sinks_missing:
                    STATS["hole_skips"] += 1              # keep vLLM's window-only output for this piece
                    continue
            if is_prefill:
                budget = PREFILL_BLOCKS
            elif DSTEP and q_min <= arc.prompt_len:
                budget = MEMORY["blocks"]
            else:
                budget = 0
            rr = (p1 - p0) <= q_piece if PREFILL == "pt" else rerank
            q_hist = None
            eff_stride = GLOBAL_DECODE_STRIDE if arc.is_global_agg else DECODE_STRIDE
            need_q_hist = (
                (QWIN or PERTOKEN)
                and not is_prefill
                and arc.qbuf is not None
                and (
                    QWIN
                    or eff_stride <= 1
                    or arc.cached_top is None
                    or (arc.decode_step + 1) % eff_stride == 0
                )
            )
            if need_q_hist:
                p_end = q_min + 1
                p_start = max(0, p_end - (piece if QWIN else min(32, piece)))
                q_hist = arc.qbuf[torch.arange(p_start, p_end, device=qh.device) % QBUF_CAP]
            q_ret = q_hist if (q_hist is not None and QWIN) else qh[p0:p1]
            own = None
            in_prompt = q_max + 1 <= arc.prompt_len
            final_prefill = (PERTOKEN_PREFILL_UNION > 0 and (is_prefill or in_prompt) and p1 == e
                             and q_max + 1 >= arc.prompt_len)
            if (PERTOKEN and rr and (not is_prefill or final_prefill) and not QWIN and not ORACLE
                    and not MEMORY["neighbors"]):
                lead = (-1 if in_prompt else 0) if PERTOKEN_LEAD else None
                res = (retrieve_blocks_per_token(arc, qh[p0:p1], groups, limit, budget, lead,
                                                 PERTOKEN_PREFILL_UNION if final_prefill else 0,
                                                 None if in_prompt else q_hist)
                       if limit >= block else None)
                top, own = res if res is not None else (None, None)
            else:
                top = retrieve_blocks(arc, q_ret, groups, limit, rr, budget) if limit >= block else None
            if top is None:
                continue
            if NORETRIEVE:
                top, own = None, None
            out_h[p0:p1] = piece_attention(arc, qh[p0:p1], q_pos, top, groups, window, impl.scale, own,
                                           q_min=q_min, q_max=q_max)
            STATS["pieces"] += 1
    return out
