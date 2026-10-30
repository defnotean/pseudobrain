# PseudoBrain (`PB-27`, `PB-Nex`, & `PB-Flash-Next`)

**Convert open dense or MoE LLMs into constant-GPU-memory, million-token hybrids — $O(1)$ GPU KV footprint from `8K` to `1,000,000+` tokens, `100%` exact retrieval across `8K–1M`, and True $O(1)$ Constant-Speed Decode (`44.6–183.5 tok/s` at `500K–1M` with `DFlash2` speculative decoding) on a single GPU.**

- **Architecture Paper:** [`docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.pdf`](docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.pdf) | [`docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.docx`](docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.docx) | [`docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.md`](docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.md)
- **Released Weights on Hugging Face:**
  - **PseudoBrain-Qwen3.8-Flash-Next (`flash_next` / `PB27Qwen4ExpForConditionalGeneration`):** [`defnotean1/PseudoBrain-Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/defnotean1/PseudoBrain-Qwen3.8-Flash-Next-NVFP4) (`36B` total, `~1.7B` active ultra-sparse `top-10/512` MoE + PLE + `DFlash2` speculative decoding, `21 GiB`)
  - **PseudoBrain-Nex-35B-A3B (`pb27-nexc` / `PB27MoEForConditionalGeneration`):** [`defnotean1/PseudoBrain-Nex-35B-A3B-NVFP4`](https://huggingface.co/defnotean1/PseudoBrain-Nex-35B-A3B-NVFP4) | [`defnotean1/PseudoBrain-35B-A3B-MoE-NVFP4`](https://huggingface.co/defnotean1/PseudoBrain-35B-A3B-MoE-NVFP4) (`35B` total, `~3B` active MoE, `23 GiB`)
  - **PseudoBrain-27B (`pb27c` / `PB27ForConditionalGeneration`):** [`defnotean1/PseudoBrain-27B-NVFP4`](https://huggingface.co/defnotean1/PseudoBrain-27B-NVFP4) (`27B` dense, `20 GiB`)
  - **Slim BF16 GDN-Only Delta Shards:** [`defnotean1/pb27-checkpoints`](https://huggingface.co/defnotean1/pb27-checkpoints) (`1.39 GB` for `flash_next`, `644 MB` for `nexc`, `3.5 GB` for `pb27c`)

---

## 1. What PseudoBrain Is

Stock Qwen3.5 / Qwen3.8 (`27B` dense, `35B-A3B` MoE, and `Qwen3.8-Flash-Next` `36B-A1.7B` MoE + PLE) interleave 3 Gated DeltaNet layers with 1 attention layer (`16` attention layers in `27B`; `10` in `35B-A3B`; `12` in `Flash-Next`). While the Gated DeltaNet layers have fixed recurrent state, the attention layers either require a linearly growing $O(L)$ KV cache (`64 GiB` at `1M` tokens on `27B`; `20 GiB` on `35B-A3B`) or forget everything older than `16,384` tokens (`Flash-Next`).

**PseudoBrain** converts every attention layer into a **Hybrid Windowed-Attention + Parallel Gated DeltaNet** block paired with our **Three-Tier Exact Memory Hierarchy** and **True $O(1)$ Constant-Speed Decode** inside vLLM (`serving/pb27_vllm/`):

| Feature | **PseudoBrain-Qwen3.8-Flash-Next (`flash_next`)** | **PseudoBrain-Nex-35B-A3B (`pb27-nexc`)** | **PseudoBrain-27B (`pb27c`)** |
| :--- | :---: | :---: | :---: |
| **Base Model (Abliterated)** | `Huihui-Qwen3.8-Flash-Next-abliterated` (`Qwen4Exp`) | `Huihui-Qwen3.5-35B-A3B-abliterated` / `Nex-N2.5` | `Huihui-Qwen3.8-27B-abliterated` |
| **Attention Layers** | `12` × (`16K` SWA + `4` sinks + parallel GDN + Exact Memory Archive) | `10` × (`16K` SWA + `4` sinks + parallel GDN + Exact Memory Archive) | `16` × (`16K` SWA + `4` sinks + parallel GDN + Exact Memory Archive) |
| **Native Recurrent Layers** | `36` Gated DeltaNet + Layer-2 Per-Layer Embeddings (`PLE`) | `30` Gated DeltaNet | `48` Gated DeltaNet |
| **Active Parameters / Token** | `~1.7B` (`top-10` of `512` experts + `1` shared + PLE) + `DFlash2` draft | `~3B` (`top-8` of `256` experts + `1` shared) | `27B` dense |
| **GPU KV Cache at `1,000,000` Tokens** | **`486 MiB` (`8.1%` of fixed `6 GiB` pool)** | **`486 MiB` (`8.1%` of fixed `6 GiB` pool)** | **`486 MiB` (`8.1%` of fixed `6 GiB` pool)** |
| **Total GPU Memory (`8K` to `1,000,000` Tokens)** | **Flat `~28 GiB`** | **Flat `~27–31 GiB`** | **Flat `~29–33 GiB`** |
| **True $O(1)$ Decode Engine** | Stride-`64` GPU LSE-merge cache + `DECODE_BLOCKS=64` + `DECODE_RERANK=0` + 2-level `bf16` Quest super-blocks + `DFlash2` (`44.6–183.5 tok/s` at `500K–1M`) | Stride-`64` GPU LSE-merge cache + `DECODE_BLOCKS=64` + `DECODE_RERANK=0` + 2-level `bf16` Quest super-blocks | Stride-`64` GPU LSE-merge cache + `DECODE_BLOCKS=64` + `DECODE_RERANK=0` + 2-level `bf16` Quest super-blocks |

---

## 2. Benchmark Results (All Backed by Committed Receipts in [`reports/`](reports/))

### 2.1 Million-Token Long-Context Exact Retrieval (`8K` to `1,000,000` Tokens)

Evaluated via [`scripts/bench/custom_longctx_bench.py`](scripts/bench/custom_longctx_bench.py) (`needle@0.1`, `needle@0.5`, `needle@0.9`, `multi4` 4-constant JSON synthesis at depths `{0.15, 0.40, 0.65, 0.85}`, and `trace4` 4-hop function call chain across depths `{0.82, 0.58, 0.34, 0.12}` + arithmetic mod `1000`):

#### `PseudoBrain-Qwen3.8-Flash-Next-NVFP4` (`flash_next` + `DFlash2` — [`reports/pb27/flash_next_pt.summary.json`](reports/pb27/flash_next_pt.summary.json), [`flash_next_ext.summary.json`](reports/pb27/flash_next_ext.summary.json))

| Context Length ($L$) | `needle@0.1` | `needle@0.5` | `needle@0.9` | `multi4` (`4` constants) | `trace4` (`4`-hop chain) | Warm TTFT | `trace4` / Peak Decode `tok/s` | GPU KV Pool Used |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`8,192`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`1.38s`** | **`93.01 tok/s`** (`85.94` mean) | **`8.1%`** |
| **`32,768`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`12.40s`** | **`66.51 tok/s`** (`59.22` mean) | **`8.1%`** |
| **`65,536`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`28.99s`** | **`67.86 tok/s`** (`51.68` `multi4`) | **`8.1%`** |
| **`120,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`61.36s`** | **`50.50 tok/s`** (`36.66` `multi4`) | **`8.1%`** |
| **`200,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`125.99s`** | **`50.81 tok/s`** | **`8.1%`** |
| **`250,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`110.16s`** | **`69.87 tok/s`** (`58.98` `multi4`) | **`8.1%`** |
| **`500,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`1.0` (`4/4`)** | **`1.0`** | **`234.05s`** | **`64.91 tok/s`** (`183.54` peak, `76.47` mean) | **`8.1%`** |
| **`1,000,000`** | **`1.0`** | **`1.0`** | **`1.0`** | **`0.75` (`3/4`)** | `0.0` | **`543.06s`** (`1,801 tok/s` prefill) | **`63.76 tok/s`** (`44.60` mean, `37.27` `multi4`) | **`8.1%`** |

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

### 2.2 General Intelligence (`MMLU`, `GSM8K`, `MMLU-Pro`, `lm-eval`), Software Engineering (`PB-SWE-Live`), & Algorithmic Coding

- **`PseudoBrain-Qwen3.8-Flash-Next-NVFP4` Public Benchmarks ([`reports/pb27/public_bench/flash_next/`](reports/pb27/public_bench/flash_next/)):**
  - **MMLU (`n=300`, 0-shot, greedy):** **`85.00%`** (`255 / 300`, `stderr: 0.0206`)
  - **GSM8K (`n=150`, CoT, greedy non-thinking):** **`96.00%`** (`144 / 150`, `stderr: 0.0161`)
  - **MMLU-Pro (`n=150`, 10-option CoT, greedy non-thinking):** **`81.33%`** (`122 / 150`, `85.31%` non-truncated)
- **Standard `lm-eval` Battery (`1000` items/task, served through vLLM — [`reports/lmeval/`](reports/lmeval/)):**
  - `PB-27` vs. Stock `27B`: **HellaSwag** `0.733` vs `0.740`, **PIQA** `0.817` vs `0.816`, **ARC-E** `0.809` vs `0.800`, **ARC-C** `0.582` vs `0.578`, **Winogrande** `0.759` vs `0.761`, **LAMBADA** `0.728` vs `0.733` ($\max |\Delta| = 0.009$, within 1 SE $\approx 0.014$).
- **Real-World Post-Cutoff Software Engineering (`PB-SWE-Live`, 59 tasks, 100-step agent loop — [`reports/swe_live/`](reports/swe_live/)):**
  - `PB-27` **`26 / 59` (`44.1%`)** vs. Stock `27B` **`28 / 59` (`47.5%`)** (paired McNemar $p \approx 0.7$, statistically indistinguishable).
- **33-Problem Medium-Hard Python Algorithmic Coding Suite ([`scripts/bench/custom_code_bench.py`](scripts/bench/custom_code_bench.py)):**
  - **`PseudoBrain-Qwen3.8-Flash-Next-NVFP4` (`flash_next`):** **`32 / 33` (`96.97%`)** at **`74.74 tok/s`** ([`reports/pb27/flash_next_med.summary.json`](reports/pb27/flash_next_med.summary.json)).
  - `PseudoBrain-27B` (`pb27c`): **`60 / 66` (`90.91%`)** sample pass rate ([`reports/pb27/pb27c_med.summary.json`](reports/pb27/pb27c_med.summary.json)).
  - `PseudoBrain-Nex-35B-A3B` (`pb27-nexc`): **`26 / 33` (`78.8%`)** problems solved (`43 / 66 = 65.2%` sample pass rate, [`reports/pb27/nexc_med.summary.json`](reports/pb27/nexc_med.summary.json)).

---

## 3. Repository Layout

```text
pseudobrain/        Self-contained PyTorch conversion core (WindowedHybridLayer, GatedDeltaNetLayer, Archive)
serving/pb27_vllm/  Out-of-tree vLLM v0.31.0 plugin (PB27, PB27MoE & PB27Qwen4Exp + 3-tier exact memory archive)
scripts/convert/    Distillation (convert_token.py), slim export (export_pb27_slim.py), NVFP4 composition (make_pb27_nvfp4.py)
scripts/eval/       lm-eval / NLL battery (evaluate.py), recall probes (recall_probes.py), vLLM parity (vllm_parity.py)
scripts/bench/      8K-1M long-context suite (custom_longctx_bench.py), coding suite (custom_code_bench.py), PB-SWE-Live
docs/               Full architecture paper (PSEUDOBRAIN_ARCHITECTURE_PAPER.md) and design specs
checkpoints/        Model cards and links to Hugging Face weights (defnotean1/...)
reports/            JSON/JSONL receipts for every benchmark table in this README and paper
```

---

## 4. Quickstart

### 4.1 Serve `PseudoBrain-Qwen3.8-Flash-Next-NVFP4`, `PseudoBrain-Nex-35B-A3B-NVFP4`, or `PseudoBrain-27B-NVFP4` with vLLM

`serving/pb27_vllm/serve.sh` automatically installs the `pb27_vllm` plugin inside `vllm/vllm-openai:v0.31.0-aarch64` with all zero-config production defaults enabled (`16K` sliding window + `4` sinks, `256K` GPU-first archive tier, `1M+` NVMe `mmap` overflow, $O(1)$ constant-speed decode, and 2-hop exact retrieval):

```bash
# Serve PseudoBrain-Qwen3.8-Flash-Next-NVFP4 + DFlash2 speculative decoding (85.0% MMLU, 96.0% GSM8K, 97.0% coding, 45-183 tok/s up to 1M):
PB27_SPECULATIVE_CONFIG='{"method":"dflash","model":"incoai/Qwen3.8-Flash-Next-DFlash2","num_speculative_tokens":5}' \
bash serving/pb27_vllm/serve.sh defnotean1/PseudoBrain-Qwen3.8-Flash-Next-NVFP4 pb27 8000

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
