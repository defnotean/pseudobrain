# PseudoBrain (`PB-27` & `PB-Nex`)

**Convert open dense or MoE LLMs into constant-GPU-memory, million-token hybrids — $O(1)$ GPU KV footprint from `8K` to `1,000,000+` tokens, `100%` (`40/40`) exact retrieval & 4-hop code trace across `8K–1M`, and up to `3.4×` faster prefill & `11.8×` faster long-context decode on a single GPU.**

- **Architecture Paper:** [`docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.pdf`](docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.pdf) | [`docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.docx`](docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.docx) | [`docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.md`](docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.md)
- **Released Weights on Hugging Face:**
  - **PseudoBrain-Nex-35B-A3B (`pb27-nexc` / `PB27MoEForConditionalGeneration`):** [`defnotean1/PseudoBrain-Nex-35B-A3B-NVFP4`](https://huggingface.co/defnotean1/PseudoBrain-Nex-35B-A3B-NVFP4) (`35B` total, `~3B` active MoE, `23 GiB`)
  - **PseudoBrain-27B (`pb27c` / `PB27ForConditionalGeneration`):** [`defnotean1/PseudoBrain-27B-NVFP4`](https://huggingface.co/defnotean1/PseudoBrain-27B-NVFP4) (`27B` dense, `20 GiB`)
  - **Slim BF16 GDN-Only Delta Shards:** [`defnotean1/pb27-checkpoints`](https://huggingface.co/defnotean1/pb27-checkpoints) (`644 MB` for `nexc`, `3.5 GB` for `pb27c`)

---

## 1. What PseudoBrain Is

Stock Qwen3.5 (`27B` dense and `35B-A3B` MoE) interleaves 3 Gated DeltaNet layers with 1 full softmax attention layer (`16` full-attention layers in `27B`; `10` in `35B-A3B`). While the Gated DeltaNet layers have fixed recurrent state, the full-attention layers still require a linearly growing $O(L)$ KV cache (`64 GiB` at `1M` tokens on `27B`; `20 GiB` on `35B-A3B`) and $O(L^2)$ prefill / $O(L)$ decode attention compute.

**PseudoBrain** converts every full-attention layer into a **Hybrid Windowed-Attention + Parallel Gated DeltaNet** block paired with our **Three-Tier Exact Memory Hierarchy** inside vLLM (`serving/pb27_vllm/`):

| Feature | Stock `Qwen3.5-27B` / `Nex-N2.5-mini` (`35B-A3B`) | **PseudoBrain-27B (`pb27c`)** | **PseudoBrain-Nex-35B-A3B (`pb27-nexc`)** |
| :--- | :---: | :---: | :---: |
| **Attention Layers** | `16` / `10` full attention ($O(L)$ KV cache) | `16` × (`16K` SWA + `4` sinks + parallel GDN + Exact Memory Archive) | `10` × (`16K` SWA + `4` sinks + parallel GDN + Exact Memory Archive) |
| **Native Recurrent Layers** | `48` / `30` Gated DeltaNet | `48` Gated DeltaNet (unchanged) | `30` Gated DeltaNet (unchanged) |
| **Active Parameters / Token** | `27B` / `~3B` (`top-8` of `256` experts + `1` shared) | `27B` + `1.85B` GDN branches | `~3B` + `0.32B` GDN branches |
| **GPU KV Cache at `1,000,000` Tokens** | `64 GiB` (`27B`) / `20 GiB` (`35B-A3B`) — OOMs | **`486 MiB` (`8.1%` of fixed `6 GiB` pool)** | **`486 MiB` (`8.1%` of fixed `6 GiB` pool)** |
| **Total GPU Memory (`8K` to `1,000,000` Tokens)** | Grows linearly until OOM (`>262K`) | **Flat `~29–33 GiB`** | **Flat `~27–31 GiB`** |
| **Exact Memory Hierarchy** | None | **Tier 1:** `16K` SWA + `4` sinks in GPU HBM<br>**Tier 2:** `<= 256K` GPU-First Archive (`273 GB/s`)<br>**Tier 3:** `256K–1M+` NVMe/Page-Cache `mmap` + GPU Quest bounds | **Tier 1:** `16K` SWA + `4` sinks in GPU HBM<br>**Tier 2:** `<= 256K` GPU-First Archive (`273 GB/s`)<br>**Tier 3:** `256K–1M+` NVMe/Page-Cache `mmap` + GPU Quest bounds |
| **Multi-Hop Long-Context Retrieval** | Degrades past `256K` | Single-pass exact rescoring (`100%` needle to `1M`) | **Unrotated (`d // 4 :`) 2-Hop Key-to-Key Bridge (`100%` `needle`, `multi4`, `trace4` to `1M`)** |

---

## 2. Benchmark Results (All Backed by Committed Receipts in [`reports/`](reports/))

### 2.1 Million-Token Long-Context Exact Retrieval (`8K` to `1,000,000` Tokens)

Evaluated via [`scripts/bench/custom_longctx_bench.py`](scripts/bench/custom_longctx_bench.py) (`needle@0.1`, `needle@0.5`, `needle@0.9`, `multi4` 4-constant JSON synthesis at depths `{0.15, 0.40, 0.65, 0.85}`, and `trace4` 4-hop function call chain across depths `{0.82, 0.58, 0.34, 0.12}` + arithmetic mod `1000`):

#### `PseudoBrain-Nex-35B-A3B` (`pb27-nexc` — [`reports/pb27/nexc_pt.summary.json`](reports/pb27/nexc_pt.summary.json), [`nexc_ext.summary.json`](reports/pb27/nexc_ext.summary.json), [`nexc_fast_verify.summary.json`](reports/pb27/nexc_fast_verify.summary.json))

| Context Length ($L$) | `needle@0.1` | `needle@0.5` | `needle@0.9` | `multi4` (`4` constants) | `trace4` (`4`-hop chain) | Overall Score | Mean TTFT | Decode Speed | GPU KV Pool Used |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`8,192`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`5/5 (100%)`** | **`1.8s`** | **`33.59 tok/s`** (`25.48` @ 1M cap) | **`8.1%`** |
| **`32,768`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`5/5 (100%)`** | **`17.0s`** (`8.8s` warm) | **`19.5 tok/s`** | **`8.1%`** |
| **`65,536`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`5/5 (100%)`** | **`21.2s`** | **`17.1 tok/s`** | **`8.1%`** |
| **`120,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`5/5 (100%)`** | **`36.7s`** | **`13.82 tok/s`** (`14.47` peak) | **`8.1%`** |
| **`200,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`5/5 (100%)`** | **`81.2s`** | **`13.5 tok/s`** | **`8.1%`** |
| **`250,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`5/5 (100%)`** | **`90.3s`** | **`13.31 tok/s`** (`14.03` peak) | **`8.1%`** |
| **`500,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`5/5 (100%)`** | **`238.3s`** | **`~10 tok/s`** | **`8.1%`** |
| **`1,000,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`5/5 (100%)`** | **`615.4s`** | **`~8 tok/s`** | **`8.1%`** |

#### `PseudoBrain-27B` (`pb27c` — [`reports/pb27/pb27c_pt.summary.json`](reports/pb27/pb27c_pt.summary.json), [`pb27c_ext.summary.json`](reports/pb27/pb27c_ext.summary.json))

| Context Length ($L$) | `needle@0.1` | `needle@0.5` | `needle@0.9` | `multi4` (`4` constants) | Mean TTFT | GPU KV Pool Used |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`8,192`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`5.8s`** (`13.07 tok/s`) | **`8.1%`** |
| **`32,768`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`26.1s`** | **`8.1%`** |
| **`65,536`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`57.4s`** | **`8.1%`** |
| **`120,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`115.6s`** | **`8.1%`** |
| **`200,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`215.5s`** | **`8.1%`** |
| **`250,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`312.8s`** | **`8.1%`** |
| **`500,000`** | **`1.0`** | **`1.0`** | **`1.0`** | `0.50` (`2/4`) | **`703.2s`** | **`8.1%`** |
| **`1,000,000`** | **`1.0`** | **`1.0`** | **`1.0`** | `0.75` (`3/4`) | **`1571.6s`** | **`8.1%`** |

### 2.2 General Intelligence (`lm-eval`), Software Engineering (`PB-SWE-Live`), & Algorithmic Coding

- **Standard `lm-eval` Battery (`1000` items/task, served through vLLM — [`reports/lmeval/`](reports/lmeval/)):**
  - `PB-27` vs. Stock `27B`: **HellaSwag** `0.733` vs `0.740`, **PIQA** `0.817` vs `0.816`, **ARC-E** `0.809` vs `0.800`, **ARC-C** `0.582` vs `0.578`, **Winogrande** `0.759` vs `0.761`, **LAMBADA** `0.728` vs `0.733` ($\max |\Delta| = 0.009$, within 1 SE $\approx 0.014$).
- **Real-World Post-Cutoff Software Engineering (`PB-SWE-Live`, 59 tasks, 100-step agent loop — [`reports/swe_live/`](reports/swe_live/)):**
  - `PB-27` **`26 / 59` (`44.1%`)** vs. Stock `27B` **`28 / 59` (`47.5%`)** (paired McNemar $p \approx 0.7$, statistically indistinguishable).
- **33-Problem × 2-Sample Medium-Hard Python Algorithmic Coding Suite ([`scripts/bench/custom_code_bench.py`](scripts/bench/custom_code_bench.py)):**
  - `PseudoBrain-27B` (`pb27c`): **`60 / 66` (`90.91%`)** sample pass rate ([`reports/pb27/pb27c_med.summary.json`](reports/pb27/pb27c_med.summary.json)).
  - `PseudoBrain-Nex-35B-A3B` (`pb27-nexc`): **`26 / 33` (`78.8%`)** problems solved (`43 / 66 = 65.2%` sample pass rate, [`reports/pb27/nexc_med.summary.json`](reports/pb27/nexc_med.summary.json)).

---

## 3. Repository Layout

```text
pseudobrain/        Self-contained PyTorch conversion core (WindowedHybridLayer, GatedDeltaNetLayer, Archive)
serving/pb27_vllm/  Out-of-tree vLLM v0.31.0 plugin (PB27 & PB27MoE + 3-tier exact memory archive)
scripts/convert/    Distillation (convert_token.py), slim export (export_pb27_slim.py), NVFP4 composition (make_pb27_nvfp4.py)
scripts/eval/       lm-eval / NLL battery (evaluate.py), recall probes (recall_probes.py), vLLM parity (vllm_parity.py)
scripts/bench/      8K-1M long-context suite (custom_longctx_bench.py), coding suite (custom_code_bench.py), PB-SWE-Live
docs/               Full architecture paper (PSEUDOBRAIN_ARCHITECTURE_PAPER.md) and design specs
checkpoints/        Model cards and links to Hugging Face weights (defnotean1/...)
reports/            JSON/JSONL receipts for every benchmark table in this README and paper
```

---

## 4. Quickstart

### 4.1 Serve `PseudoBrain-Nex-35B-A3B-NVFP4` or `PseudoBrain-27B-NVFP4` with vLLM

`serving/pb27_vllm/serve.sh` automatically installs the `pb27_vllm` plugin inside `vllm/vllm-openai:v0.31.0-aarch64` with all zero-config production defaults enabled (`16K` sliding window + `4` sinks, `256K` GPU-first archive tier, `1M+` NVMe `mmap` overflow, and 2-hop exact retrieval):

```bash
# Serve PseudoBrain-Nex-35B-A3B-NVFP4 (35B-A3B MoE, 33.6 tok/s @ 8K, 13.8 tok/s @ 120K, 100% recall to 1M):
bash serving/pb27_vllm/serve.sh defnotean1/PseudoBrain-Nex-35B-A3B-NVFP4 pb27-nex 8000

# Serve PseudoBrain-27B-NVFP4 (27B Dense, 90.9% medium coding, 100% needle recall to 1M):
bash serving/pb27_vllm/serve.sh defnotean1/PseudoBrain-27B-NVFP4 pb27 8000
```

### 4.2 Run the `8K–1,000,000` Long-Context & Coding Benchmarks

```bash
# 8K, 32K, 65K, 120K, 200K exact retrieval + multi4 + trace4:
bash scripts/bench/custom_longctx_pertoken.sh nexc_pt /path/to/PseudoBrain-Nex-35B-A3B-NVFP4

# 250K, 500K, 1,000,000 extended exact retrieval + multi4 + trace4:
bash scripts/bench/custom_longctx_extended.sh nexc_ext /path/to/PseudoBrain-Nex-35B-A3B-NVFP4

# 33-problem x 2-sample medium-hard Python coding benchmark:
python3 scripts/bench/custom_code_bench.py \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --model pb27-nex --tier medium --n 2 --out reports/pb27/nexc_med
```

---

## 5. Limitations & Honest Trade-Offs vs. Stock

1. **Ultra-Short Context (`< 4K` tokens) Overhead:** Even with the short-context archive bypass (`<= 16,384` tokens), evaluating the parallel Gated DeltaNet branch on the `10` (Nex) or `16` (27B) hybrid layers adds `~5–8%` compute per token compared to a stock model below `4K` tokens. Above `16K` tokens, PseudoBrain's bounded $O(W + k \cdot 64)$ attention rapidly overtakes stock's $O(L)$ full attention.
2. **Dense `27B` vs. MoE `35B-A3B` Coding vs. Throughput Trade-Off:** Because `PseudoBrain-Nex-35B-A3B` activates only `~3B` parameters per token (`top-8` of `256` experts), it is `2.6×–3.4×` faster than `PseudoBrain-27B`, but scores `78.8%` (`26/33` problems solved) on the medium-hard algorithmic coding benchmark vs. `90.91%` for `PseudoBrain-27B`. Choose `PseudoBrain-Nex-35B-A3B` for high-speed interactive & million-token agentic workloads, and `PseudoBrain-27B` for maximum single-try dense coding accuracy.
3. **Eager Execution Required (`--enforce-eager`):** Because the exact memory archive (`pb27_vllm/memory.py`) performs dynamic block retrieval and LSE merging per request, `PB27_EAGER=1` is required when exact memory retrieval is active.

---

## 6. License & Author

- **Code License:** Apache-2.0 (see [`LICENSE`](LICENSE)). Converted model weights inherit the base model's license.
- **Author:** Ian Kurt Galeon ([@defnotean](https://github.com/defnotean)).
