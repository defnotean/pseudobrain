"""Server-side Hugging Face publisher for PseudoBrain multi-precision editions (BF16, W4A16, FP8, NVFP4).

Uses Hugging Face Hub's cross-repository `CommitOperationCopy` so all base `.safetensors` shards and the trained
`pb27-gdn-00001-of-00001.safetensors` delta shard are linked server-side on Hugging Face with 0 GB of local disk
storage and near-instant commit speed.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from huggingface_hub import CommitOperationAdd, CommitOperationCopy, HfApi, hf_hub_download

COPY_SMALL = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "generation_config.json",
    "hf_quant_config.json",
)

GDN_IGNORE = [
    "re:.*\\.gdn\\.(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)$",
    "re:.*\\.gdn\\.conv1d$",
    "re:.*\\.gdn\\.norm$",
    "re:.*\\.gdn$",
]
MODELOPT_GDN_IGNORE = ["*.gdn.*", "*.gdn"]
GDN_SUBMODULES = ("", ".in_proj_qkv", ".in_proj_z", ".in_proj_a", ".in_proj_b", ".out_proj", ".conv1d", ".norm")

VARIANTS = {
    # 1. PseudoBrain-27B (pb27c)
    "pb27-bf16": {
        "target_repo": "defnoteatin/PseudoBrain-27B-BF16",
        "base_repo": "huihui-ai/Huihui-Qwen3.8-27B-abliterated",
        "slim_dir": "/home/defnotean/pb27/hf/pb27c",
        "gdn_ckpt_path": "pb27c/pb27-gdn-00001-of-00001.safetensors",
        "readme_path": "/home/defnotean/pb27/release/pb27c/README.md",
        "old_repo_str": "PseudoBrain-27B-NVFP4",
    },
    "pb27-w4a16": {
        "target_repo": "defnoteatin/PseudoBrain-27B-W4A16",
        "base_repo": "philbert440/Qwen3.8-27B-Uncensored-Aggressive-W4A16-AWQ",
        "slim_dir": "/home/defnotean/pb27/hf/pb27c",
        "gdn_ckpt_path": "pb27c/pb27-gdn-00001-of-00001.safetensors",
        "readme_path": "/home/defnotean/pb27/release/pb27c/README.md",
        "old_repo_str": "PseudoBrain-27B-NVFP4",
    },
    "pb27-fp8": {
        "target_repo": "defnoteatin/PseudoBrain-27B-FP8",
        "base_repo": "batsclamp/Huihui-Qwen3.8-27B-abliterated-FP8-v2",
        "slim_dir": "/home/defnotean/pb27/hf/pb27c",
        "gdn_ckpt_path": "pb27c/pb27-gdn-00001-of-00001.safetensors",
        "readme_path": "/home/defnotean/pb27/release/pb27c/README.md",
        "old_repo_str": "PseudoBrain-27B-NVFP4",
    },
    # 2. PseudoBrain-Nex-35B-A3B (nexc)
    "nex-bf16": {
        "target_repo": "defnoteatin/PseudoBrain-Nex-35B-A3B-BF16",
        "base_repo": "nex-agi/Nex-N2.5-mini",
        "slim_dir": "/home/defnotean/pb27/hf/nexc",
        "gdn_ckpt_path": "nexc/pb27-gdn-00001-of-00001.safetensors",
        "readme_path": "/home/defnotean/pb27/release/pb27_nex/README.md",
        "old_repo_str": "PseudoBrain-Nex-35B-A3B-NVFP4",
    },
    "nex-w4a16": {
        "target_repo": "defnoteatin/PseudoBrain-Nex-35B-A3B-W4A16",
        "base_repo": "Arahide/Nex-N2.5-mini-INT4-W4A16",
        "slim_dir": "/home/defnotean/pb27/hf/nexc",
        "gdn_ckpt_path": "nexc/pb27-gdn-00001-of-00001.safetensors",
        "readme_path": "/home/defnotean/pb27/release/pb27_nex/README.md",
        "old_repo_str": "PseudoBrain-Nex-35B-A3B-NVFP4",
    },
    "nex-fp8": {
        "target_repo": "defnoteatin/PseudoBrain-Nex-35B-A3B-FP8",
        "base_repo": "cbert33/Nex-N2.5-mini-FP8-Calibrated",
        "slim_dir": "/home/defnotean/pb27/hf/nexc",
        "gdn_ckpt_path": "nexc/pb27-gdn-00001-of-00001.safetensors",
        "readme_path": "/home/defnotean/pb27/release/pb27_nex/README.md",
        "old_repo_str": "PseudoBrain-Nex-35B-A3B-NVFP4",
    },
    # 3. PseudoBrain-Qwen3.8-Flash-Next (flash_next)
    "flash-bf16": {
        "target_repo": "defnoteatin/PseudoBrain-Qwen3.8-Flash-Next-BF16",
        "base_repo": "Qwen/Qwen3.8-Flash-Next",
        "slim_dir": "/home/defnotean/pb27/hf/flash_next_gdn_slim",
        "gdn_ckpt_path": "flash_next/pb27-gdn-00001-of-00001.safetensors",
        "readme_path": "/home/defnotean/pb27/release/flash_next/README.md",
        "old_repo_str": "PseudoBrain-Qwen3.8-Flash-Next-NVFP4",
    },
    "flash-w4a16": {
        "target_repo": "defnoteatin/PseudoBrain-Qwen3.8-Flash-Next-W4A16",
        "base_repo": "cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4",
        "slim_dir": "/home/defnotean/pb27/hf/flash_next_gdn_slim",
        "gdn_ckpt_path": "flash_next/pb27-gdn-00001-of-00001.safetensors",
        "readme_path": "/home/defnotean/pb27/release/flash_next/README.md",
        "old_repo_str": "PseudoBrain-Qwen3.8-Flash-Next-NVFP4",
    },
    "flash-fp8": {
        "target_repo": "defnoteatin/PseudoBrain-Qwen3.8-Flash-Next-FP8",
        "base_repo": "Qwen/Qwen3.8-Flash-Next-FP8",
        "slim_dir": "/home/defnotean/pb27/hf/flash_next_gdn_slim",
        "gdn_ckpt_path": "flash_next/pb27-gdn-00001-of-00001.safetensors",
        "readme_path": "/home/defnotean/pb27/release/flash_next/README.md",
        "old_repo_str": "PseudoBrain-Qwen3.8-Flash-Next-NVFP4",
    },
}


def publish_one(api: HfApi, key: str, spec: dict, plugin_root: Path, cache_dir: Path) -> None:
    target_repo = spec["target_repo"]
    base_repo = spec["base_repo"]
    slim_dir = Path(spec["slim_dir"])
    gdn_ckpt_path = spec["gdn_ckpt_path"]

    print(f"\n=== Publishing {key}: {target_repo} (base={base_repo}) ===")
    api.create_repo(repo_id=target_repo, repo_type="model", private=False, exist_ok=True)

    base_files = set(api.list_repo_files(base_repo))
    qcfg_p = hf_hub_download(base_repo, "config.json", cache_dir=str(cache_dir), token=api.token)
    qidx_p = hf_hub_download(base_repo, "model.safetensors.index.json", cache_dir=str(cache_dir), token=api.token)
    qcfg = json.loads(Path(qcfg_p).read_text())
    qidx = json.loads(Path(qidx_p).read_text())

    scfg = json.loads((slim_dir / "config.json").read_text())
    sidx = json.loads((slim_dir / "model.safetensors.index.json").read_text())

    weight_map = dict(qidx["weight_map"])
    for k, v in sidx["weight_map"].items():
        if ".gdn." in k or ".spatial_mlp." in k:
            weight_map[k] = "pb27-gdn-00001-of-00001.safetensors"

    is_qwen4_exp = "PB27Qwen4ExpForConditionalGeneration" in scfg.get("architectures", [])
    if is_qwen4_exp:
        ple_meta_keys = [
            "model.language_model.layers.1.ple.norm_conv.weight",
            "model.language_model.layers.1.ple.norm_key.weight",
            "model.language_model.layers.1.ple.norm_query.weight",
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale_2",
        ]
        for k in ple_meta_keys:
            if k in weight_map or k.endswith(".weight"):
                weight_map[k] = "pb27-ple-meta.safetensors"
        weight_map = {
            k: v
            for k, v in weight_map.items()
            if not (
                ".ple_embedding.ngram_embedding.shard_" in k
                or k.startswith("mtp.")
                or ".self_attn.indexer." in k
            )
        }

    kept_base_shards = sorted({
        v for v in weight_map.values()
        if v not in ("pb27-gdn-00001-of-00001.safetensors", "pb27-ple-meta.safetensors")
    })

    # Merge config.json
    cfg = dict(qcfg)
    cfg["architectures"] = scfg["architectures"]
    extra_keys = [k for k in scfg if k.startswith("pb_") or k.startswith("linear_") or k == "mamba_ssm_dtype"]
    for k in extra_keys:
        cfg[k] = scfg[k]
        if isinstance(cfg.get("text_config"), dict):
            cfg["text_config"][k] = scfg[k]
    if isinstance(scfg.get("text_config"), dict):
        for k, v in scfg["text_config"].items():
            if k.startswith("pb_") or k.startswith("linear_") or k == "mamba_ssm_dtype":
                cfg[k] = v
                if isinstance(cfg.get("text_config"), dict):
                    cfg["text_config"][k] = v

    holder = cfg if "quantization_config" in cfg else cfg.get("text_config", {})
    if isinstance(holder, dict) and "quantization_config" in holder:
        qc = holder["quantization_config"]
        qmethod = str(qc.get("quant_method", "")).lower()
        is_modelopt = qmethod.startswith("modelopt")
        gdn_ign = MODELOPT_GDN_IGNORE if is_modelopt else GDN_IGNORE
        if "ignore" in qc or qmethod in ("compressed-tensors", "modelopt"):
            qc["ignore"] = list(qc.get("ignore", [])) + [x for x in gdn_ign if x not in qc.get("ignore", [])]

        hybrid_layers = cfg.get("pb_hybrid_layers") or cfg.get("text_config", {}).get("pb_hybrid_layers") or []
        layer_prefix = "model.language_model.layers" if "text_config" in cfg else "model.layers"
        explicit_gdn = [
            f"{layer_prefix}.{l}.gdn{sub}"
            for l in hybrid_layers
            for sub in GDN_SUBMODULES
        ]
        if "modules_to_not_convert" in qc or qmethod in ("fp8", "awq", "gptq"):
            cur = list(qc.get("modules_to_not_convert") or [])
            qc["modules_to_not_convert"] = cur + [x for x in explicit_gdn if x not in cur]
        if "ignored_layers" in qc:
            cur = list(qc.get("ignored_layers") or [])
            qc["ignored_layers"] = cur + [x for x in explicit_gdn if x not in cur]

        has_input_scale = any(k.endswith(".input_global_scale") for k in weight_map)
        if not has_input_scale:
            for grp in qc.get("config_groups", {}).values():
                if isinstance(grp, dict) and grp.get("input_activations") is not None:
                    grp["input_activations"] = None

    ops = []
    # 1. Server-side copy all kept base shards from base_repo
    for shard in kept_base_shards:
        ops.append(
            CommitOperationCopy(
                src_path_in_repo=shard,
                path_in_repo=shard,
                src_repo_id=base_repo,
                src_repo_type="model",
            )
        )

    # 2. Server-side copy pb27-gdn shard (and pb27-ple-meta if Flash-Next) from defnoteatin/pb27-checkpoints
    ops.append(
        CommitOperationCopy(
            src_path_in_repo=gdn_ckpt_path,
            path_in_repo="pb27-gdn-00001-of-00001.safetensors",
            src_repo_id="defnoteatin/pb27-checkpoints",
            src_repo_type="model",
        )
    )
    if is_qwen4_exp:
        ops.append(
            CommitOperationCopy(
                src_path_in_repo="flash_next/pb27-ple-meta.safetensors",
                path_in_repo="pb27-ple-meta.safetensors",
                src_repo_id="defnoteatin/pb27-checkpoints",
                src_repo_type="model",
            )
        )

    # 3. Add merged config.json and model.safetensors.index.json
    merged_idx_bytes = json.dumps(
        {"metadata": qidx.get("metadata", {}), "weight_map": weight_map}, indent=1
    ).encode("utf-8")
    merged_cfg_bytes = json.dumps(cfg, indent=1).encode("utf-8")
    ops.append(CommitOperationAdd(path_in_repo="model.safetensors.index.json", path_or_fileobj=merged_idx_bytes))
    ops.append(CommitOperationAdd(path_in_repo="config.json", path_or_fileobj=merged_cfg_bytes))

    # 4. Copy or add small tokenizer/processor files
    for fname in COPY_SMALL:
        local_f = slim_dir / fname
        if local_f.is_file():
            ops.append(CommitOperationAdd(path_in_repo=fname, path_or_fileobj=str(local_f)))
        elif fname in base_files:
            ops.append(
                CommitOperationCopy(
                    src_path_in_repo=fname,
                    path_in_repo=fname,
                    src_repo_id=base_repo,
                    src_repo_type="model",
                )
            )

    # 5. Add pb27_vllm_plugin
    for p in sorted(plugin_root.rglob("*")):
        if p.is_file() and "__pycache__" not in str(p):
            rel = p.relative_to(plugin_root.parent).as_posix()
            ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(p)))

    # 6. Add README.md
    readme_p = Path(spec["readme_path"])
    if readme_p.is_file():
        rtext = readme_p.read_text()
        new_repo_name = target_repo.split("/")[-1]
        rtext = rtext.replace("defnotean1/", "defnoteatin/").replace(spec["old_repo_str"], new_repo_name)
        ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=rtext.encode("utf-8")))

    info = api.create_commit(
        repo_id=target_repo,
        repo_type="model",
        operations=ops,
        commit_message=f"Publish {target_repo} ({len(kept_base_shards)} base shards + PB27 GDN)",
    )
    print(f"DONE {target_repo}: {info.commit_url} ({len(kept_base_shards)} base shards)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*", default=list(VARIANTS.keys()))
    args = ap.parse_args()

    tok = Path("/home/defnotean/.cache/huggingface/token").read_text().strip()
    api = HfApi(token=tok)
    plugin_root = Path("/home/defnotean/pb27/release/PseudoBrain-27B-NVFP4/pb27_vllm_plugin")
    cache_dir = Path("/tmp/hf_meta_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    for k in args.keys:
        publish_one(api, k, VARIANTS[k], plugin_root, cache_dir)
    shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
