# Evaluation & Benchmark Receipts (`reports/`)

Every metric reported in [`README.md`](../README.md) and [`docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.md`](../docs/PSEUDOBRAIN_ARCHITECTURE_PAPER.md) is backed by a committed JSON/JSONL receipt in this directory:

## 1. Million-Token Long-Context Exact Retrieval & Speed (`reports/pb27/`)

- **`PseudoBrain-Nex-35B-A3B` (`pb27-nexc`):**
  - [`pb27/nexc_pt.summary.json`](pb27/nexc_pt.summary.json) & [`pb27/nexc_pt.jsonl`](pb27/nexc_pt.jsonl): `8K, 32K, 65K, 120K, 200K` evaluation (`25/25 = 100.0%` across `needle@0.1`, `needle@0.5`, `needle@0.9`, `multi4`, and `trace4`).
  - [`pb27/nexc_ext.summary.json`](pb27/nexc_ext.summary.json) & [`pb27/nexc_ext.jsonl`](pb27/nexc_ext.jsonl): `250K, 500K, 1,000,000` extended-context evaluation (`15/15 = 100.0%` across `needle@0.1`, `needle@0.5`, `needle@0.9`, `multi4`, and `trace4`, `8.1%` KV pool usage of `6 GiB`).
  - [`pb27/nexc_fast_verify.summary.json`](pb27/nexc_fast_verify.summary.json) & [`pb27/nexc_fast_verify.jsonl`](pb27/nexc_fast_verify.jsonl): Post-optimization speed & accuracy verification at `8K, 120K, 250K` (`15/15 = 100.0%`, `8K`: `1.8s` TTFT / `25.48 tok/s`; `120K`: `36.7s` TTFT / `13.82 tok/s` (`14.47 tok/s` on `trace4`); `250K`: `90.3s` TTFT / `13.31 tok/s` (`14.03 tok/s` peak)).
  - [`pb27/nexc_med.summary.json`](pb27/nexc_med.summary.json) & [`pb27/nexc_med.jsonl`](pb27/nexc_med.jsonl): 33-problem × 2-sample medium-hard Python algorithmic coding suite (`26/33` problems solved = `78.8%`, `43/66` sample pass rate = `65.2%`).

- **`PseudoBrain-27B` (`pb27c`):**
  - [`pb27/pb27c_pt.summary.json`](pb27/pb27c_pt.summary.json) & [`pb27/pb27c_pt.jsonl`](pb27/pb27c_pt.jsonl): `8K, 32K, 65K, 120K, 200K` evaluation (`100.0%` `needle@0.1/0.5/0.9` and `multi4`).
  - [`pb27/pb27c_ext.summary.json`](pb27/pb27c_ext.summary.json) & [`pb27/pb27c_ext.jsonl`](pb27/pb27c_ext.jsonl): `250K, 500K, 1,000,000` extended-context evaluation (`100.0%` `needle@0.1/0.5/0.9` up to `1M`, `8.1%` KV pool usage of `6 GiB`).
  - [`pb27/pb27c_med.summary.json`](pb27/pb27c_med.summary.json) & [`pb27/pb27c_med.jsonl`](pb27/pb27c_med.jsonl): 33-problem × 2-sample medium-hard Python algorithmic coding suite (`60/66 = 90.91%` sample pass rate).

## 2. Standard `lm-eval` Battery & `PB-SWE-Live` (`reports/lmeval/`, `reports/swe_live/`)

- [`lmeval/pb27_nvfp4_dflash2.json`](lmeval/pb27_nvfp4_dflash2.json) vs. [`lmeval/stock27_nvfp4_dflash2.json`](lmeval/stock27_nvfp4_dflash2.json): Matched 1000-item `lm-eval` battery (`HellaSwag`, `PIQA`, `ARC-E`, `ARC-C`, `Winogrande`, `LAMBADA`, max $|\Delta| = 0.009$).
- [`swe_live/pb27_fp8mtp_med60.json`](swe_live/pb27_fp8mtp_med60.json) vs. [`swe_live/stock27_fp8_med60.json`](swe_live/stock27_fp8_med60.json): 59-task post-cutoff `PB-SWE-Live` agentic software engineering benchmark (`26/59` vs. `28/59`, McNemar $p \approx 0.7$).
