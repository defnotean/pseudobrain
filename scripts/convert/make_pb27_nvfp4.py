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
import struct
from pathlib import Path

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None


def _extract_safetensors_subset(src_path: Path, dst_path: Path, keys: list[str]) -> None:
    with src_path.open("rb") as f:
        (hdr_len,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(hdr_len).decode("utf-8"))
        data_base = 8 + hdr_len
        new_hdr: dict[str, object] = {"__metadata__": hdr.get("__metadata__", {"format": "pt"})}
        chunks: list[bytes] = []
        offset = 0
        for k in keys:
            info = hdr[k]
            s, e = info["data_offsets"]
            f.seek(data_base + s)
            raw = f.read(e - s)
            new_hdr[k] = {
                "dtype": info["dtype"],
                "shape": info["shape"],
                "data_offsets": [offset, offset + len(raw)],
            }
            chunks.append(raw)
            offset += len(raw)
    hdr_bytes = json.dumps(new_hdr, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(hdr_bytes) % 8)) % 8
    hdr_bytes += b" " * pad
    with dst_path.open("wb") as out:
        out.write(struct.pack("<Q", len(hdr_bytes)))
        out.write(hdr_bytes)
        for c in chunks:
            out.write(c)

COPY = ("tokenizer.json", "tokenizer_config.json", "tokenizer.model", "special_tokens_map.json",
        "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja",
        "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json",
        "generation_config.json", "hf_quant_config.json")
GDN_IGNORE = ["re:.*\\.gdn\\.(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)$", "re:.*\\.gdn\\.conv1d$",
              "re:.*\\.gdn\\.norm$", "re:.*\\.gdn$"]
MODELOPT_GDN_IGNORE = ["*.gdn.*", "*.gdn"]
SPATIAL_IGNORE = ["re:.*\\.spatial_mlp\\.(gate_proj|up_proj|down_proj|gate_up_proj)$", "re:.*\\.spatial_mlp$"]
MODELOPT_SPATIAL_IGNORE = ["*.spatial_mlp.*", "*.spatial_mlp"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", required=True, help="stock NVFP4 checkpoint snapshot dir")
    ap.add_argument("--slim", required=True, help="PB-27 slim bf16 export dir (gdn shard, config, tokenizer)")
    ap.add_argument("--spatial", default=None, help="optional pb27-spatial safetensors shard to attach")
    ap.add_argument("--slim-out", default=None, help="optional output dir for composed slim bf16 export (e.g. ~/pb27/hf/pb27s)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    quant, slim, out = Path(args.quant).expanduser(), Path(args.slim).expanduser(), Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    qidx = json.loads((quant / "model.safetensors.index.json").read_text())
    sidx = json.loads((slim / "model.safetensors.index.json").read_text())
    gdn_files = sorted({v for k, v in sidx["weight_map"].items() if ".gdn." in k or ".spatial_mlp." in k})
    weight_map = dict(qidx["weight_map"])
    for k, v in sidx["weight_map"].items():
        if ".gdn." in k or ".spatial_mlp." in k:
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
        try:
            os.link(slim / f, link)
        except OSError:
            os.symlink(slim / f, link)
    spatial_layers = []
    spatial_inter = 0
    if args.spatial:
        sp_src = Path(args.spatial).expanduser().resolve()
        sp_name = "pb27-spatial-00001-of-00001.safetensors"
        link = out / sp_name
        if link.is_symlink() or link.exists():
            link.unlink()
        try:
            os.link(sp_src, link)
        except OSError:
            os.symlink(sp_src, link)
        with safe_open(str(sp_src), framework="pt", device="cpu") as f:
            for k in f.keys():
                weight_map[k] = sp_name
                if k.endswith(".spatial_mlp.gate_proj.weight"):
                    spatial_inter = int(f.get_tensor(k).shape[0])
                    for p in k.split("."):
                        if p.isdigit():
                            spatial_layers.append(int(p))
                            break
        spatial_layers = sorted(set(spatial_layers))
    scfg = json.loads((slim / "config.json").read_text())
    is_qwen4_exp = "PB27Qwen4ExpForConditionalGeneration" in scfg.get("architectures", [])
    if is_qwen4_exp and "model-00026-of-00041.safetensors" in set(weight_map.values()):
        (out / "model.safetensors.index.json.orig").write_text(
            json.dumps({"metadata": qidx.get("metadata", {}), "weight_map": weight_map}, indent=1)
        )
        ple_meta_name = "pb27-ple-meta.safetensors"
        ple_meta_keys = [
            "model.language_model.layers.1.ple.norm_conv.weight",
            "model.language_model.layers.1.ple.norm_key.weight",
            "model.language_model.layers.1.ple.norm_query.weight",
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale_2",
        ]
        _extract_safetensors_subset(
            quant / "model-00026-of-00041.safetensors",
            out / ple_meta_name,
            ple_meta_keys,
        )
        for k in ple_meta_keys:
            weight_map[k] = ple_meta_name
        weight_map = {
            k: v
            for k, v in weight_map.items()
            if not (
                ".ple_embedding.ngram_embedding.shard_" in k
                or k.startswith("mtp.")
                or ".self_attn.indexer." in k
            )
        }
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": qidx.get("metadata", {}), "weight_map": weight_map}, indent=1))

    qcfg = json.loads((quant / "config.json").read_text())
    cfg = dict(qcfg)
    cfg["architectures"] = scfg["architectures"]
    for k, v in scfg.items():
        if k.startswith("pb_"):
            cfg[k] = v
            if isinstance(cfg.get("text_config"), dict):
                cfg["text_config"][k] = v
    if spatial_layers and spatial_inter > 0:
        cfg["pb_spatial_layers"] = spatial_layers
        cfg["pb_spatial_intermediate_size"] = spatial_inter
        if isinstance(cfg.get("text_config"), dict):
            cfg["text_config"]["pb_spatial_layers"] = spatial_layers
            cfg["text_config"]["pb_spatial_intermediate_size"] = spatial_inter
    holder = cfg if "quantization_config" in cfg else cfg.get("text_config", {})
    qc = holder["quantization_config"]
    is_modelopt = str(qc.get("quant_method", "")).lower().startswith("modelopt")
    gdn_ign = MODELOPT_GDN_IGNORE if is_modelopt else GDN_IGNORE
    sp_ign = MODELOPT_SPATIAL_IGNORE if is_modelopt else SPATIAL_IGNORE
    extra_ignore = gdn_ign + (sp_ign if any(".spatial_mlp." in k for k in weight_map) else [])
    qc["ignore"] = list(qc.get("ignore", [])) + [x for x in extra_ignore if x not in qc.get("ignore", [])]
    has_input_scale = any(k.endswith(".input_global_scale") for k in weight_map)
    if not has_input_scale:
        for grp in qc.get("config_groups", {}).values():
            if isinstance(grp, dict) and grp.get("input_activations") is not None:
                grp["input_activations"] = None
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    for name in COPY:
        src = slim / name if (slim / name).is_file() else quant / name
        if src.is_file():
            (out / name).write_bytes(src.read_bytes())

    if args.slim_out:
        sout = Path(args.slim_out).expanduser()
        sout.mkdir(parents=True, exist_ok=True)
        slim_wmap = dict(sidx["weight_map"])
        for f in sorted(set(sidx["weight_map"].values())):
            sfile = slim / f
            link = sout / f
            if link.is_symlink() or link.exists():
                link.unlink()
            if sfile.is_symlink():
                os.symlink(sfile.resolve(), link)
            else:
                try:
                    os.link(sfile, link)
                except OSError:
                    os.symlink(sfile, link)
        if args.spatial:
            sp_src = Path(args.spatial).expanduser().resolve()
            sp_name = "pb27-spatial-00001-of-00001.safetensors"
            link = sout / sp_name
            if link.is_symlink() or link.exists():
                link.unlink()
            try:
                os.link(sp_src, link)
            except OSError:
                os.symlink(sp_src, link)
            with safe_open(str(sp_src), framework="pt", device="cpu") as f:
                for k in f.keys():
                    slim_wmap[k] = sp_name
        (sout / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": sidx.get("metadata", {}), "weight_map": slim_wmap}, indent=1)
        )
        sout_cfg = dict(scfg)
        if spatial_layers and spatial_inter > 0:
            sout_cfg["pb_spatial_layers"] = spatial_layers
            sout_cfg["pb_spatial_intermediate_size"] = spatial_inter
            if isinstance(sout_cfg.get("text_config"), dict):
                sout_cfg["text_config"]["pb_spatial_layers"] = spatial_layers
                sout_cfg["text_config"]["pb_spatial_intermediate_size"] = spatial_inter
        (sout / "config.json").write_text(json.dumps(sout_cfg, indent=1))
        for name in COPY:
            src = slim / name
            if src.is_file():
                (sout / name).write_bytes(src.read_bytes())

    n_q = sum(1 for k in weight_map if ".gdn." not in k and ".spatial_mlp." not in k)
    n_g = len(weight_map) - n_q
    print(json.dumps({"out": str(out), "slim_out": str(args.slim_out), "quant_tensors": n_q, "gdn_spatial_tensors": n_g, "gdn_files": gdn_files,
                      "architectures": cfg["architectures"], "pb_hybrid_layers": cfg.get("pb_hybrid_layers"),
                      "pb_spatial_layers": cfg.get("pb_spatial_layers"),
                      "ignore_tail": qc["ignore"][-4:], "files": sorted(p.name for p in out.iterdir())}, indent=1))


if __name__ == "__main__":
    main()
