# Deep research: what can make PseudoBrain significantly better (2026-10-11)

Owner asked for real papers with real mechanisms, not headlines. Every item below has the mechanism, the evidence,
and a concrete way it plugs into PB-27 (16k-window attention + parallel GDN branch in 16 of 64 layers, exact memory
tier over evicted KV, served on one GB10). Ranked by expected gain per unit of our compute. Numbers are the papers'
own unless marked; several are self-reported and were not independently reproduced.

## Tier 1 — directly fixes a measured PB-27 weakness, cheap, proven on Qwen3-class models

### 1. Train a block selector on the frozen model by KL to its own attention (SpotAttention, arXiv 2606.22874; RTPurbo, 2605.16928)
- **Mechanism (SpotAttention):** a 4-head, dim-128 query–key scorer with a 64-dim RoPE component per full-attention
  layer; ReLU score → softmax over keys = an *estimate of the layer's attention distribution*. Blocks of 16 tokens, per-
  block max. Trained with forward KL from the head-averaged backbone attention, **backbone frozen**, 100M FineWeb-Edu
  tokens at 16k, 763 steps (loss ≈ 0 after ~200). Because the estimate is a calibrated distribution, a *dual top-p*
  rule (sink 128 + recent 256 always kept, nucleus p=0.9 over the rest, floor 512 tokens) sets a per-query budget.
- **Evidence:** on Qwen3-8B/32B at 32k and **Qwen3.5-4B/9B hybrids (GDN + 8 attention layers) at 128k** it matches or
  beats dense: Qwen3.5-9B RULER 0.995 vs dense 0.948, InfiniteBench 0.883 vs 0.866; decode at 128k 3.9× FlashAttention.
- **RTPurbo** goes further for a dense→sparse conversion: find *retrieval heads* offline (needle self-attention score;
  top 15% of heads), give only those full KV + a rank-16 Q/K indexer trained by KL (≈4k params/head, 30M tokens,
  600 steps), all other heads get an 8k sliding window + 4 sinks; then ~600 steps of self-distillation on the top-10
  logits. Qwen3-Coder-30B: LongBench 54.2 vs 53.8 dense, RULER-64k 85.5 vs 86.2, AIME unchanged; 2× decode at 1M.
- **For PB-27:** this is exactly S8 item 1, with the recipe and numbers now known. Our exact tier already keeps the
  archive and does Quest-bound + exact rescoring; replace the bound with a SpotAttention-style scorer trained by KL
  against PB-27's own windowed+full attention (teacher = the dense attention over the archive, which `retrieve()`'s
  rescoring stage already computes). ~100M tokens on Colab ≈ 10 units. Expected: close copy@32k and multihop@128k,
  and a per-query budget instead of a fixed 64 blocks. Also adopt the sink+recent reservation.

### 2. Keep a compressed memory of what was evicted — we already have the slot (IndexMem, arXiv 2605.25475, ICML 2026)
- **Mechanism:** a fixed-size fast-weight matrix M (d/8 × d) per layer, written with evicted keys/values by a decayed
  outer-product recurrence (M ← λM + η Σ φ(k) vᵀ, with a normaliser b), read by m(q) = φ(q)ᵀM / (φ(q)²ᵀb), added as
  o = o_attn + g(q)·m(q). Backbone frozen; the memory (0.5M params/layer) is trained by MSE against the attention
  residual lost to eviction. Qwen3-8B RULER-16k at 50% eviction: 80.0 vs SnapKV 62.8; at 90%: 56.0 vs 26.8.
- **For PB-27:** our parallel GDN branch *is* this object (a gated delta-rule fast-weight memory over everything,
  trained by KD) — but it was trained to reproduce the teacher's logits, not the *attention residual that the window
  drops*. IndexMem's loss is the sharper target: train the branch (or a small additional M) on exactly the attention
  mass the 16k window loses, with the exact tier as the "retained" set. Cheap (Colab, hours), function-preserving.

### 3. Positions for retrieved memory — the ladder's make-or-break (InfLLM, NeurIPS 2024; Leng et al. ICLR 2026, 2510.17196)
- **Evidence that it matters:** training-free RoPE scaling extrapolates only a small multiple of training length;
  chunk retrieval that avoids absolute positions generalises far: Leng et al. train at **4k** and hold RULER 85.7 @128k,
  85.8 @1M, 82.2 @8M, 79.8 @32M (Llama-YaRN, Mamba-2 and Landmark score 0 at 128k). Keys to their result: retrieved
  chunks carry no far-away absolute position, a non-linear chunk encoder with a CLS landmark, and *enforced selection
  sparsity in training* (top-8 chunks, not 64 — 79.8 vs 69.7 at 32M).
- **For PB-27:** our archive stores RoPE-rotated keys at true positions, so a query at 900k attends to keys rotated by
  distances the model never saw. Change: store *unrotated* keys in the archive and rotate retrieved blocks to local
  pseudo-positions (just outside the window, InfLLM style) at retrieval time. Pure serving code; no training. Then
  the train-time matching trick: when we train the selector/branch (items 1–2), use a *small* block budget so the
  model learns to rely on few, precise blocks — the same enforced-sparsity effect.

### 4. 2-bit archive (OSCAR, arXiv 2605.17757; ParisKV, 2602.07721)
- **OSCAR:** offline attention-aware covariance → fixed rotations + clipping for INT2 KV; Qwen3-32B "on par with
  BF16", Qwen3-8B −1.4 points, RULER-NIAH robust to 128k; ~8× KV memory, 3× batch-1 decode; INT2 kernels for
  vLLM/SGLang paged caches. **ParisKV:** data-independent sign-pattern centroids (no learned index, so no drift as
  decode adds keys), 4-bit codes on GPU, full KV on CPU fetched by UVA; matches/exceeds full attention on 7/9 long
  reasoning tasks, 49 ms/step at 1M on Llama-3.1-8B. Code: github.com/amy-77/ParisKV.
- **For PB-27:** the archive is the only thing that grows with context. OSCAR-style INT2 for archived K/V (8× less
  host memory per token → 1B tokens of archive ≈ tens of GB per layer group instead of hundreds) plus ParisKV's
  drift-free coarse index as the candidate filter. Serving work, no training; gate on the recall ladder.

## Tier 2 — architecture upgrades for a run B (one Colab conversion, ~350 units)

### 5. Gated DeltaNet-2 for the parallel branches (NVIDIA, arXiv 2605.22791; code NVlabs/GatedDeltaNet-2)
- **Mechanism:** S_t = (I − k_t (b_t ⊙ k_t)ᵀ) D_t S_{t−1} + k_t (w_t ⊙ v_t)ᵀ — a channel-wise *erase* gate on the key
  side, a channel-wise *write* gate on the value side, KDA-style channel-wise decay; reduces to KDA/GDN as special cases.
- **Evidence (1.3B, 100B tokens, matched state size):** Wiki ppl 15.90 vs GDN 16.40; commonsense 53.11 vs 52.07;
  hybrid-with-SWA average 53.97 vs GDN 52.25; **multi-key NIAH 72.6/51.4/37.8 vs GDN 58.0/37.0/27.8** at 1k/2k/4k.
- **For PB-27:** the 16 added branches are the only trained part; swapping their rule for GDN-2 is a one-line change
  in the conversion (same state shape), and multi-key retrieval is precisely where PB-27 without the tier is weakest.
  Risk: Triton kernels must run on sm_121/Blackwell; test on the Spark first.

### 6. SpectralShift init for the branches' decay (RUCAIBox, arXiv 2609.14320)
- **Mechanism:** split the GDN alpha (decay) projection into a shared mean + input-dependent deviation and scale the
  deviation by s = (L_ref/L_target)^½ (s = 0.25 for 8k→128k) so the slow spectral band covers the target length;
  scale only alpha's LR by s during long-context training. Applied to an existing checkpoint at the start of CPT.
- **Evidence:** 1.5B-A0.6B GDN-MoE, 8k→128k: RULER 55.2 vs 52.4; 32k single stage 45.8 vs 40.6; pure GDN S-NIAH at
  32k 10.6 vs 6.4. General benchmarks unchanged.
- **For PB-27:** our branches were trained on 32k rows; for run B targeting 128k+ rows, initialise/scale alpha this way.
  Free.

### 7. Richer fixed-state recall if run B can afford it (Switching Linear Attention, COLM 2026, 2609.39034)
- **Mechanism:** per output dim, J delta-rule components with EM-style responsibilities (prior from the key, Gaussian
  likelihood of the prediction error); gated and *temporal* variants (mixture persists across steps). Code released;
  only a sequential Triton kernel today (2× slower training, 1.25–1.5× slower inference than GDN).
- **Evidence (374M, 15B tokens):** recall-intensive avg: hybrid temporal SwiLA 31.8 vs hybrid GDN 29.5; LM parity.
- **For PB-27:** a candidate for the branch rule *after* kernels mature; not before GDN-2.

### 8. One checkpoint, many speeds (Super Apriel, arXiv 2604.19877)
- **Mechanism:** every layer holds four trained mixers (full attention, SWA, KDA, GDN); a per-request "placement" picks
  one per layer; stochastic distillation from the frozen teacher; a surrogate predicts placement quality; shared weights
  also give self-speculative decoding. 15B: all-attention preset = teacher; presets at 2.9–10.7× decode keep 96–77%.
- **For PB-27:** we already have two mixers per hybrid layer (window+branch and, implicitly, the stock full attention).
  Adding "placement" = letting the server choose per request between full attention (short prompts) and windowed+tier
  (long) with the *same* weights is cheap and makes the short-context case exactly stock. Worth doing in serving.

## Tier 3 — evidence that shapes decisions (no action)

- **RADLADS (COLM 2025):** attention→linear conversion in 3 stages (hidden-state L2 → logits KL → CE at 16k; 350–700M
  tokens) keeps MMLU to within ~3–6 points but **RULER collapses** (7B: 39.7@4k, 16.5@16k vs teacher 69.7@16k).
  Confirms our choice: keep real attention (windowed) + a memory tier; pure linear conversion loses exact recall.
- **"Impossibility triangle" (2605.05066, withdrawn — proof error):** efficiency × compact state × recall ∝ length
  can't all hold for a fixed-state model. Even without the proof, the practical reading stands: unbounded exact recall
  needs an archive that grows (ours does, off-GPU), and the fixed-state part should be judged on *what it adds inside
  the budget*, not asked to carry everything.
- **InfLLM-V2 (MiniCPM4.1):** parameter-free dense↔sparse switch retains 98.1% long-context / 99.7% CoT at 4× speed —
  supports item 8 (switch by length) and shows self-distillation on the model's own attention is enough.
- **Beyond a Million Tokens (BEAM, 2510.27246):** 100 conversations up to 10M tokens, 2,000 probing questions — the
  right *real-task* benchmark for the ladder beyond needles; LIGHT (episodic + working + scratchpad memory) +3.5–12.7%.
  Use BEAM as the owner-facing proof at 1M+ alongside RULER/BABILong.
- **KV quantization below 4-bit:** naive 2-bit (KIVI) collapses on reasoning (MATH 88→47 on Qwen3-8B); only
  covariance/rotation-aware schemes (OSCAR, RotateKV) survive — hence item 4 names OSCAR, not KIVI.

## Proposed order (after the current gates: DFlash2/NVFP4 speed, final SWE-Live)
1. Positions for retrieved blocks (item 3) — serving change, prerequisite for the ladder.
2. Trained selector + dual top-p (item 1) and residual-targeted branch/memory loss (item 2) — one Colab session.
3. INT2 archive + drift-free candidate index (item 4) — serving.
4. Length ladder with receipts: RULER/BABILong/needles + BEAM at 128k → 1M → 8M/32M, vs stock and stock+RAG.
5. Run B: GDN-2 branches + SpectralShift init (items 5–6) if the ladder shows the branch is the limiting factor.
6. Placement presets (item 8) as a serving feature; SwiLA (item 7) only when kernels are production-grade.

## Sources
SpotAttention arXiv 2606.22874 · RTPurbo arXiv 2605.16928 · IndexMem arXiv 2605.25475 · Leng et al. arXiv 2510.17196
(github.com/jacky-leng/length-generalizable-sparse-attention) · InfLLM arXiv 2402.04617 · OSCAR arXiv 2605.17757 ·
ParisKV arXiv 2602.07721 (github.com/amy-77/ParisKV) · Gated DeltaNet-2 arXiv 2605.22791 (NVlabs/GatedDeltaNet-2) ·
SpectralShift arXiv 2609.14320 (RUCAIBox/GDN-SpectralShift) · Switching Linear Attention arXiv 2609.39034
(lindermanlab/switching-linear-attention) · Super Apriel arXiv 2604.19877 · RADLADS arXiv 2505.03005 ·
Impossibility Triangle arXiv 2605.05066 (withdrawn) · InfLLM-V2 arXiv 2509.24663 · BEAM arXiv 2510.27246 ·
KIVI arXiv 2402.02750 · RotateKV IJCAI 2025 · Qwen3.8-Next design arXiv 2608.30320.
