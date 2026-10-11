# PB-27 architecture spec

The target design for the PseudoBrain conversion of Qwen/Qwen3.8-27B. Program, gates and ledger:
docs/PB27_PROGRAM.md. Every claim here becomes true only when its gate passes with receipts.

## Starting point (stock Qwen3.8-27B, measured from its config)

- 64 layers, width 5120, MLP 17408 (SwiGLU), vocab 248320, one MTP layer.
- 48 Gated DeltaNet layers (16 key heads x 128, 48 value heads x 128, conv 4): fixed-size recurrent state.
- 16 gated full-attention layers (every 4th; 24 query heads, 4 KV heads, head dim 256, partial RoPE 25%):
  KV cache grows 64 KB per token (16 layers x 4 heads x 256 x 2 x bf16) — 8 GB at 128k, 64 GB at 1M.

## PB-27 layer stack

| Layer kind | Count | What it is | Memory | Exactness vs stock |
|---|---|---|---|---|
| DeltaNet | 48 | stock weights, unchanged | fixed | exact |
| PB attention | 16 | stock attention weights over three sources: (1) sliding window W + 4 sinks on GPU; (2) **exact memory tier**: older K/V in 64-token blocks in host RAM/NVMe, each block summarized by min/max key bounds; per query the top-k blocks by upper-bound score are fetched and attended exactly (Quest/InfLLM-style); (3) a parallel DeltaNet branch, zero-initialized and distilled, carrying what retrieval misses | GPU: W + k blocks; host: grows, cheap | exact inside W; near-exact beyond (gated) |
| MLP | 64 | stock SwiGLU + optional trained activation-sparsity gate (skip reading inactive neuron rows) | — | exact without the gate; gated with it |
| MTP | 1 | stock draft head; proposes several tokens verified in one pass of the full model | — | lossless (verification) |

Plus the **recall tool** (learning/recall_tool.py) for exact verbatim lookup of anything in the session log.

## Speed design (S4.5)

1. Multi-token verification with the stock MTP head (vLLM `qwen3_5_mtp`; DFlash2 block drafts as an option).
   Recurrent state rollback for rejected drafts.
2. Bounded attention: per-token attention cost is O(W + k·64) regardless of context length.
3. Weights: FP8 (official) first; NVFP4 for MLP/projection matrices if the S1 gate holds.
4. Activation sparsity in the MLPs, trained against stock logits (KD), threshold per layer set by the gate.

## Serving plan

Run PB-27 on vLLM (which already has DeltaNet, sliding-window and sink attention kernels, FP8/NVFP4 and MTP)
as an out-of-tree model; PB-specific parts (exact memory tier, DeltaNet branch in attention layers, sparsity)
are added as custom layers there, so PB-27 gets mainstream kernel speed instead of a from-scratch server.

## Measurements required (each is a gate)

- Retention: PB-SWE-Live and lm-eval within the stock model's 95% CI.
- Long context: NLL by position bucket to 128k, needle/copy/multi-hop probes 32k–1M: ≥ stock where stock runs.
- Memory: GPU memory flat from 32k to 1M tokens.
- Speed: batch-1 tok/s and first-token latency at 2k/32k/128k, PB-27 vs stock under the same engine.
