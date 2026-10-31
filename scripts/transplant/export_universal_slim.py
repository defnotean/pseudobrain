"""Universal slim PB-27 GDN exporter (`pb27-gdn-00001-of-00001.safetensors`,
`model.safetensors.index.json`, `config.json`) supporting:
  - `qwen3_next` (e.g. `Qwen/Qwen3-Coder-Next` -> `PB27Qwen3NextForCausalLM`)
  - `qwen3_moe`  (e.g. `Qwen/Qwen3-Coder-30B-A3B-Instruct` -> `PB27Qwen3MoeForCausalLM`)
  - `qwen3`      (e.g. `Qwen/Qwen3-32B` -> `PB27Qwen3ForCausalLM`)
  - `gemma4`     (e.g. `google/gemma-4-26B-A4B-it`, `google/gemma-4-31B-it` -> `PB27Gemma4ForConditionalGeneration`)

Uses HTTP Range requests (`RemoteSafeTensors`) to fetch only the exact BF16
`self_attn` and `linear_attn` byte ranges from Hugging Face with 0 GB of full
teacher shards downloaded to disk.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from efficiency import transplant as tp  # noqa: E402
from efficiency.gated_deltanet import GatedDeltaNetLayer  # noqa: E402
from scripts.transplant.export_pb27_slim import GDN_SHARD  # noqa: E402
from scripts.transplant.export_qwen4_exp_slim import RemoteSafeTensors  # noqa: E402


ARCH_MAP = {
    "qwen3_next": ("PB27Qwen3NextForCausalLM", "model."),
    "qwen3_moe": ("PB27Qwen3MoeForCausalLM", "model."),
    "qwen3": ("PB27Qwen3ForCausalLM", "model."),
    "gemma4": ("PB27Gemma4ForConditionalGeneration", "model.language_model."),
    "gemma4_text": ("PB27Gemma4ForConditionalGeneration", "model.language_model."),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF repo ID for the BF16 teacher")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--window", type=int, default=16384)
    ap.add_argument("--sinks", type=int, default=4)
    ap.add_argument("--gdn-key-heads", type=int, default=16)
    ap.add_argument("--gdn-key-dim", type=int, default=128)
    ap.add_argument("--gdn-value-heads", type=int, default=16)
    ap.add_argument("--gdn-value-dim", type=int, default=128)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    print(f"Fetching config.json and index from {args.repo}@{args.revision}...", flush=True)
    remote = RemoteSafeTensors(args.repo, args.revision)
    cfg = remote.config
    text_cfg = cfg.get("text_config", cfg)
    model_type = text_cfg.get("model_type") or cfg.get("model_type")
    if model_type not in ARCH_MAP:
        raise ValueError(f"Unsupported model_type={model_type!r}; expected one of {list(ARCH_MAP)}")
    pb_arch, prefix = ARCH_MAP[model_type]

    num_layers = int(text_cfg["num_hidden_layers"])
    hidden_size = int(text_cfg["hidden_size"])
    num_heads = int(text_cfg["num_attention_heads"])
    num_kv_heads = int(text_cfg.get("num_key_value_heads", num_heads))
    head_dim = int(text_cfg.get("head_dim") or (hidden_size // num_heads))
    norm_eps = float(text_cfg.get("rms_norm_eps", 1e-6))

    # Determine hybrid layers
    layer_types = text_cfg.get("layer_types")
    if not layer_types and "full_attention_interval" in text_cfg:
        interval = int(text_cfg["full_attention_interval"])
        layer_types = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(num_layers)
        ]
        text_cfg["layer_types"] = layer_types

    if layer_types and ("sliding_attention" in layer_types or "linear_attention" in layer_types):
        first_kv_shared = num_layers - int(text_cfg.get("num_kv_shared_layers", 0) or 0)
        hybrid = [
            i
            for i, lt in enumerate(layer_types[:first_kv_shared])
            if lt == "full_attention"
        ]
    else:
        # Pure full-attention model (Qwen3Moe, Qwen3): window all layers to 16K + Quest Memory,
        # and attach parallel GDN recurrent state on every 4th layer ([3, 7, 11, ...])
        hybrid = [i for i in range(num_layers) if (i + 1) % 4 == 0]

    dims = dict(
        heads=int(text_cfg.get("linear_num_key_heads", args.gdn_key_heads)),
        key_dim=int(text_cfg.get("linear_key_head_dim", args.gdn_key_dim)),
        value_heads=int(text_cfg.get("linear_num_value_heads", args.gdn_value_heads)),
        value_dim=int(text_cfg.get("linear_value_head_dim", args.gdn_value_dim)),
    )
    conv = int(text_cfg.get("linear_conv_kernel_dim", 4))
    root = math.sqrt(num_layers)

    print(
        f"Model type={model_type} -> {pb_arch}, layers={num_layers}, "
        f"hybrid layers ({len(hybrid)}): {hybrid[:8]}{'...' if len(hybrid) > 8 else ''}, GDN dims={dims}",
        flush=True,
    )

    out_tensors: dict[str, torch.Tensor] = {}
    for idx, i in enumerate(hybrid):
        sa = f"{prefix}layers.{i}.self_attn."
        print(f"[{idx + 1}/{len(hybrid)}] Initializing layer {i}.gdn from {sa}...", flush=True)
        q = remote.get_tensor(sa + "q_proj.weight").float()
        k = remote.get_tensor(sa + "k_proj.weight").float()
        if (sa + "v_proj.weight") in remote.weight_map:
            v = remote.get_tensor(sa + "v_proj.weight").float()
        else:
            # Gemma 4 full_attention with attention_k_eq_v=True uses k_proj for both K and V
            v = k.clone()
        o = remote.get_tensor(sa + "o_proj.weight").float()

        # Infer per-layer head_dim and kv_heads directly from k_proj and o_proj shapes
        # o_proj has shape [hidden_size, num_heads * layer_head_dim]
        layer_head_dim = o.shape[1] // num_heads
        layer_kv_heads = k.shape[0] // layer_head_dim
        is_q_gate = (q.shape[0] == 2 * num_heads * layer_head_dim)
        c_layer = tp.TeacherConfig(
            vocab=int(text_cfg["vocab_size"]),
            width=hidden_size,
            layers=num_layers,
            heads=num_heads,
            kv_heads=layer_kv_heads,
            head_dim=layer_head_dim,
            hidden=int(text_cfg.get("intermediate_size", hidden_size * 4)),
            rope_theta=1000000.0,
            norm_eps=norm_eps,
            tie_embeddings=bool(text_cfg.get("tie_word_embeddings", False)),
            model_type=str(model_type),
            layer_types=tuple(["full_attention"] * num_layers),
            rotary_dim=layer_head_dim,
            attn_output_gate=is_q_gate,
            qk_norm=True,
            zero_centered_norm=(model_type == "qwen3_next"),
        )

        gdn = GatedDeltaNetLayer(
            hidden_size,
            1.0 / root,
            dims["heads"],
            dims["key_dim"],
            conv,
            backend="reference",
            norm="rms",
            norm_eps=norm_eps,
            block_norm_eps=norm_eps,
            value_heads=dims["value_heads"],
            value_dim=dims["value_dim"],
            l2norm="hf",
        ).to(torch.float32)

        with torch.no_grad():
            if is_q_gate:
                both = q.reshape(num_heads, 2, layer_head_dim, -1)
                qq = both[:, 0].reshape(num_heads * layer_head_dim, -1)
                gate = both[:, 1].reshape(num_heads * layer_head_dim, -1)
            else:
                qq = q
                gate = None
            done = tp.init_gdn_from_attention(gdn, c_layer, qq, k, v, o, gate, root)

            # Copy decay/beta/norm parameters from preceding linear_attn layer if present
            prev_la = f"{prefix}layers.{i - 1}.linear_attn." if i > 0 else ""
            if prev_la and (prev_la + "A_log") in remote.weight_map:
                hk = 2 * dims["heads"] * dims["key_dim"]
                vw = dims["value_heads"] * dims["value_dim"]
                hv = dims["value_heads"]
                if (prev_la + "in_proj_a.weight") in remote.weight_map:
                    in_a = remote.get_tensor(prev_la + "in_proj_a.weight").float()
                    in_b = remote.get_tensor(prev_la + "in_proj_b.weight").float()
                elif (prev_la + "in_proj_ba.weight") in remote.weight_map:
                    ba = remote.get_tensor(prev_la + "in_proj_ba.weight").float()
                    ba = ba.reshape(dims["heads"], 2, hv // dims["heads"], -1)
                    in_b = ba[:, 0].reshape(hv, -1)
                    in_a = ba[:, 1].reshape(hv, -1)
                else:
                    in_a = in_b = None
                if in_a is not None and in_b is not None:
                    gdn.proj.weight[hk + 2 * vw : hk + 2 * vw + hv].copy_(in_a)
                    gdn.proj.weight[hk + 2 * vw + hv :].copy_(in_b)
                gdn.A_log.copy_(remote.get_tensor(prev_la + "A_log").float())
                gdn.dt_bias.copy_(remote.get_tensor(prev_la + "dt_bias").float())
                gdn.o_norm.copy_(remote.get_tensor(prev_la + "norm.weight").float())
                done.append(f"decay/norm=layers.{i - 1}")

            # Zero out_proj at initialization so short-context KL drift is 0.0
            gdn.out.weight.zero_()
            done.append("out=0")

        hf = tp.gdn_hf_tensors(
            f"{prefix}layers.{i}.gdn.",
            proj=gdn.proj.weight,
            conv=gdn.conv_weight,
            A_log=gdn.A_log,
            dt_bias=gdn.dt_bias,
            o_norm=gdn.o_norm,
            out=gdn.out.weight,
            root=root,
            **dims,
        )
        for name, tensor in hf.items():
            out_tensors[name] = tensor.detach().to("cpu", torch.bfloat16).contiguous()
        print(f"    -> {','.join(done)}", flush=True)

    shard_path = out / GDN_SHARD
    tp.write_safetensors(shard_path, out_tensors)
    added_bytes = sum(t.numel() * t.element_size() for t in out_tensors.values())
    print(f"Wrote {shard_path} ({len(out_tensors)} tensors, {added_bytes / (1 << 20):.1f} MiB)", flush=True)

    weight_map = {name: GDN_SHARD for name in out_tensors}
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": added_bytes}, "weight_map": weight_map}, indent=1),
        encoding="utf-8",
    )

    tp.pb27_config(
        cfg,
        hybrid,
        args.window,
        args.sinks,
        {**dims, "conv": conv},
    )
    cfg["architectures"] = [pb_arch]
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
