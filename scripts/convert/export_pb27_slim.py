"""Slim PB-27 HF export: only the trained Gated DeltaNet branches, on top of the teacher.

A --shared conversion (convert_token.py --shared) leaves every attention, norm, MLP,
embedding and head tensor identical to the stock Qwen3.8-27B checkpoint; only the 16
`gdn` branches of the WindowedHybridLayers are new. Instead of rewriting 54 GB this
writes one shard with the gdn tensors (the exact names / conventions of
export_hf_qwen35's hybrid layers, via efficiency.transplant.gdn_hf_tensors), links the
teacher's shards next to it, and merges the index / config. CPU only: no TokenCore is
built and no teacher weight is read (only the teacher's index / shard headers).

    python -m scripts.transplant.export_pb27_slim --core runs/x/core.pt \\
        --teacher-dir ~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/<hash> \\
        --out exports/pb27-slim [--dtype bfloat16]

Output dir: pb27-gdn-00001-of-00001.safetensors, symlinks (or copies) of every teacher
shard, model.safetensors.index.json, config.json, tokenizer / processor files.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from efficiency import transplant as tp  # noqa: E402

GDN_SHARD = "pb27-gdn-00001-of-00001.safetensors"
EXTRA_FILES = ("tokenizer.json", "tokenizer_config.json", "tokenizer.model", "special_tokens_map.json",
               "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja",
               "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json",
               "generation_config.json")
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_payload(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # history holds plain Python the strict unpickler may refuse
        return torch.load(path, map_location="cpu", weights_only=False)


def hybrid_indices(payload: dict) -> list[int]:
    """The converted layers: from the spec's 'swa_gdn' mixers, cross-checked against the
    blocks that carry gdn.* tensors in mixers_only (both must agree)."""
    mixers = payload["config"]["spec"]["mixers"]
    from_spec = [i for i, m in enumerate(mixers) if m == "swa_gdn"]
    from_keys = sorted({int(k.split(".")[1]) for k in payload["mixers_only"] if k.split(".")[2] == "gdn"})
    if from_spec != from_keys:
        raise ValueError(f"spec swa_gdn layers {from_spec} != layers with gdn.* tensors {from_keys}")
    if not from_spec:
        raise ValueError("no swa_gdn layer in the checkpoint: nothing to export slim")
    return from_spec


def layer_gdn_dims(payload: dict, index: int) -> dict:
    """The gdn shape of swa_gdn layer `index`, as TokenCore builds it: the teacher's own
    linear-attention shape (Qwen3.5/3.8) or its attention heads (a Llama teacher), with the
    spec's converted_key_dim / converted_heads / head multipliers of a mixer that replaces
    a full-attention layer; conv from the teacher's linear attention, else the spec."""
    teacher, spec = payload["config"]["teacher"], payload["config"]["spec"]
    if teacher.get("linear_key_heads"):
        dims = dict(heads=teacher["linear_key_heads"], key_dim=teacher["linear_key_dim"],
                    value_heads=teacher["linear_value_heads"], value_dim=teacher["linear_value_dim"])
        conv = int(teacher.get("linear_conv") or 4)
    else:
        dims = dict(heads=teacher["heads"], key_dim=teacher["head_dim"], value_heads=teacher["heads"],
                    value_dim=teacher["head_dim"])
        conv = int(spec.get("conv") or 4)
    if spec.get("converted_key_dim"):
        dims["key_dim"] = int(spec["converted_key_dim"])
    if spec.get("converted_heads"):
        dims["heads"] = dims["value_heads"] = int(spec["converted_heads"])
    mult = dict((int(i), int(m)) for i, m in spec.get("layer_head_mults") or ()).get(
        index, int(spec.get("converted_head_mult") or 1))
    if mult > 1:
        dims["heads"], dims["value_heads"] = dims["heads"] * mult, dims["value_heads"] * mult
    return {**dims, "conv": conv}


def gdn_tensors(payload: dict, prefix: str, dtype: torch.dtype) -> tuple[dict[str, torch.Tensor], list[int], dict]:
    """HF-named gdn tensors of every hybrid layer, straight from mixers_only, and their
    common shape (vLLM's config holds one GDN shape for every layer)."""
    teacher = payload["config"]["teacher"]
    root = math.sqrt(int(teacher["layers"]))
    state = payload["mixers_only"]
    out: dict[str, torch.Tensor] = {}
    hybrid = hybrid_indices(payload)
    shapes = {i: layer_gdn_dims(payload, i) for i in hybrid}
    common = shapes[hybrid[0]]
    if any(s != common for s in shapes.values()):
        raise ValueError(f"hybrid layers have different gdn shapes: {shapes}")
    dims = {k: v for k, v in common.items() if k != "conv"}
    for i in hybrid:
        g = f"blocks.{i}.gdn."
        hf = tp.gdn_hf_tensors(f"{prefix}layers.{i}.gdn.", proj=state[g + "proj.weight"], conv=state[g + "conv_weight"],
                               A_log=state[g + "A_log"], dt_bias=state[g + "dt_bias"], o_norm=state[g + "o_norm"],
                               out=state[g + "out.weight"], root=root, **dims)
        out.update({k: v.detach().to("cpu", dtype).contiguous() for k, v in hf.items()})
    return out, hybrid, common


def teacher_index(teacher_dir: Path) -> dict:
    """The teacher's weight index ({metadata, weight_map}); built from the shard headers
    when the snapshot has a single unindexed file."""
    index = teacher_dir / "model.safetensors.index.json"
    if index.exists():
        return json.loads(index.read_text(encoding="utf-8"))
    reader = tp.SafeTensors(teacher_dir)
    weight_map = {name: reader.files[name][0].name for name in reader.keys()}
    total = sum(math.prod(reader.files[n][1]["shape"]) * torch.empty(0, dtype=tp._ST_DTYPES[reader.files[n][1]["dtype"]]).element_size()
                for n in reader.keys())
    return {"metadata": {"total_size": total}, "weight_map": weight_map}


def link_or_copy(src: Path, dst: Path) -> str:
    """A relative symlink dst -> src; a copy where symlinks are unavailable."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(os.path.relpath(src, dst.parent), dst)
        return "symlink"
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)
        return "copy"


def export_slim(core_path: Path, teacher_dir: Path, out: Path, dtype: torch.dtype = torch.bfloat16) -> dict:
    payload = load_payload(core_path)
    if "mixers_only" not in payload:
        raise ValueError(f"{core_path} is not a --shared checkpoint (no mixers_only)")
    spec = payload["config"]["spec"]
    out.mkdir(parents=True, exist_ok=True)
    index = teacher_index(teacher_dir)
    prefix = "model.language_model." if any(k.startswith("model.language_model.") for k in index["weight_map"]) else "model."
    tensors, hybrid, gdn = gdn_tensors(payload, prefix, dtype)
    clash = sorted(set(tensors) & set(index["weight_map"]))
    if clash:
        raise ValueError(f"gdn names already in the teacher index: {clash[:3]}...")
    tp.write_safetensors(out / GDN_SHARD, tensors)
    linked = {}
    for shard in sorted(set(index["weight_map"].values())):
        linked[shard] = link_or_copy(teacher_dir / shard, out / shard)
    weight_map = dict(index["weight_map"])
    weight_map.update({name: GDN_SHARD for name in tensors})
    added = sum(t.numel() * t.element_size() for t in tensors.values())
    merged = {"metadata": {**index.get("metadata", {}), "total_size": int(index.get("metadata", {}).get("total_size", 0)) + added},
              "weight_map": weight_map}
    (out / "model.safetensors.index.json").write_text(json.dumps(merged, indent=1), encoding="utf-8")
    config = json.loads((teacher_dir / "config.json").read_text(encoding="utf-8"))
    tp.pb27_config(config, hybrid, spec.get("attention_window"), spec.get("attention_sinks", 0), gdn)
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    copied = []
    for name in EXTRA_FILES:
        if (teacher_dir / name).exists():
            shutil.copy2(teacher_dir / name, out / name)
            copied.append(name)
    return {"hybrid": hybrid, "gdn_tensors": len(tensors), "gdn_bytes": added, "shards": linked, "copied": copied,
            "teacher": payload.get("teacher") or payload.get("history", {}).get("args", {}).get("teacher")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--core", required=True, help="--shared core.pt (payload with mixers_only)")
    ap.add_argument("--teacher-dir", required=True, help="HF snapshot directory of the teacher")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    args = ap.parse_args(argv)
    info = export_slim(Path(args.core).expanduser(), Path(args.teacher_dir).expanduser(), Path(args.out).expanduser(),
                       DTYPES[args.dtype])
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
