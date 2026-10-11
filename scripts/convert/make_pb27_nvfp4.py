"""Compose a PB-27 NVFP4 checkpoint directory for vLLM from a published NVFP4 Qwen3.8-27B checkpoint plus the PB-27
gdn shard (S7/S4.5 speed path, docs/PB27_RESEARCH_2026-10.md item 1).

PB-27's attention, GDN (stock) and MLP weights are the stock weights, so the quantized tensors of a stock NVFP4
checkpoint (e.g. unsloth/Qwen3.8-27B-NVFP4, compressed-tensors mixed precision) can be reused as-is; only the 16
parallel `gdn` branches added by the conversion are new and they stay bf16 (added to the quantization `ignore`
list). The output directory holds symlinks to the quantized shards, the gdn shard, a merged safetensors index, the
PB-27 config (architectures, pb_* fields) and the tokenizer/chat-template files from the slim bf16 export.

    python scripts/transplant/make_pb27_nvfp4.py --quant ~/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/<rev> \
        --slim ~/pb27/hf/pb27a --out ~/pb27/hf/pb27a_nvfp4
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

COPY = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja",
        "preprocessor_config.json", "video_preprocessor_config.json", "generation_config.json")
GDN_IGNORE = ["re:.*\\.gdn\\.(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)$", "re:.*\\.gdn\\.conv1d$",
              "re:.*\\.gdn\\.norm$", "re:.*\\.gdn$"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", required=True, help="stock NVFP4 checkpoint snapshot dir")
    ap.add_argument("--slim", required=True, help="PB-27 slim bf16 export dir (gdn shard, config, tokenizer)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    quant, slim, out = Path(args.quant).expanduser().resolve(), Path(args.slim).expanduser().resolve(), Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    qidx = json.loads((quant / "model.safetensors.index.json").read_text())
    sidx = json.loads((slim / "model.safetensors.index.json").read_text())
    gdn_files = sorted({v for k, v in sidx["weight_map"].items() if ".gdn." in k})
    weight_map = dict(qidx["weight_map"])
    for k, v in sidx["weight_map"].items():
        if ".gdn." in k:
            weight_map[k] = v
    for f in sorted(set(qidx["weight_map"].values())):
        link = out / f
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(quant / f, link)
    for f in gdn_files:
        link = out / f
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(slim / f, link)
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": qidx.get("metadata", {}), "weight_map": weight_map}, indent=1))

    qcfg = json.loads((quant / "config.json").read_text())
    scfg = json.loads((slim / "config.json").read_text())
    cfg = dict(qcfg)
    cfg["architectures"] = scfg["architectures"]
    for k, v in scfg.items():
        if k.startswith("pb_"):
            cfg[k] = v
    holder = cfg if "quantization_config" in cfg else cfg.get("text_config", {})
    qc = holder["quantization_config"]
    qc["ignore"] = list(qc.get("ignore", [])) + GDN_IGNORE
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    for name in COPY:
        src = slim / name
        if src.is_file():
            (out / name).write_bytes(src.read_bytes())
    n_q = sum(1 for k in weight_map if ".gdn." not in k)
    n_g = len(weight_map) - n_q
    print(json.dumps({"out": str(out), "quant_tensors": n_q, "gdn_tensors": n_g, "gdn_files": gdn_files,
                      "architectures": cfg["architectures"], "pb_hybrid_layers": cfg.get("pb_hybrid_layers"),
                      "ignore_tail": qc["ignore"][-4:], "files": sorted(p.name for p in out.iterdir())}, indent=1))


if __name__ == "__main__":
    main()
