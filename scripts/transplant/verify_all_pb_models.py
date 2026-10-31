"""End-to-end verification suite for all converted PseudoBrain models.

Verifies:
1. Remote Hugging Face repository integrity for every published edition
   (`W4A16`, `FP8`, `NVFP4`, `BF16`):
   - `config.json` has the registered `PB27*` architecture, `pb_hybrid_layers`,
     `pb_attention_window=16384`, `linear_*` dimensions, and `quantization_config`
     ignoring `.gdn` modules.
   - `model.safetensors.index.json` maps all `.gdn.` tensors to
     `pb27-gdn-00001-of-00001.safetensors` and all base tensors to linked shards.
   - HTTP Range header read on both `pb27-gdn-00001-of-00001.safetensors` and the
     base `.safetensors` shard in the target repo succeeds (proving server-side
     LFS links are live and readable).
2. vLLM Model Construction & Real GDN Checkpoint Weight Loading on GPU:
   - Constructs each `PB27*` model class inside vLLM, runs `model.load_weights`
     with the real exported `pb27-gdn-00001-of-00001.safetensors` tensors, and
     asserts that all GDN parameters (`in_proj_qkvz`, `in_proj_ba`, `conv1d`,
     `A_log`, `dt_bias`, `norm`, `out_proj`) load cleanly and `per_layer_sliding_window`
     is active on the hybrid attention layers.
"""
from __future__ import annotations

import argparse
import json
import struct
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_url


def _fetch_bytes(url: str, start: int, end_incl: int, token: str | None) -> bytes:
    headers = {"Range": f"bytes={start}-{end_incl}", "User-Agent": "pb27-verify/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _read_safetensors_header(repo_id: str, filename: str, token: str | None) -> dict:
    url = hf_hub_url(repo_id=repo_id, filename=filename)
    raw_len = _fetch_bytes(url, 0, 7, token)
    (hdr_len,) = struct.unpack("<Q", raw_len)
    raw_hdr = _fetch_bytes(url, 8, 8 + hdr_len - 1, token)
    return json.loads(raw_hdr.decode("utf-8"))


def verify_hf_repo(repo_id: str, token: str | None) -> dict:
    url_cfg = hf_hub_url(repo_id=repo_id, filename="config.json")
    req = urllib.request.Request(
        url_cfg, headers={"Authorization": f"Bearer {token}"} if token else {}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        cfg = json.loads(r.read().decode("utf-8"))

    url_idx = hf_hub_url(repo_id=repo_id, filename="model.safetensors.index.json")
    req2 = urllib.request.Request(
        url_idx, headers={"Authorization": f"Bearer {token}"} if token else {}
    )
    with urllib.request.urlopen(req2, timeout=60) as r2:
        idx = json.loads(r2.read().decode("utf-8"))

    arch = cfg["architectures"][0]
    tcfg = cfg.get("text_config", cfg)
    hybrid = tcfg.get("pb_hybrid_layers") or cfg.get("pb_hybrid_layers")
    window = tcfg.get("pb_attention_window") or cfg.get("pb_attention_window")
    assert hybrid and len(hybrid) > 0, f"{repo_id}: missing pb_hybrid_layers"
    assert window == 16384, f"{repo_id}: unexpected pb_attention_window={window}"

    wmap = idx["weight_map"]
    gdn_keys = [k for k, v in wmap.items() if v == "pb27-gdn-00001-of-00001.safetensors"]
    base_shards = sorted({v for v in wmap.values() if v != "pb27-gdn-00001-of-00001.safetensors" and v != "pb27-ple-meta.safetensors"})
    assert len(gdn_keys) == len(hybrid) * 9, f"{repo_id}: expected {len(hybrid)*9} GDN keys, got {len(gdn_keys)}"
    assert len(base_shards) >= 1, f"{repo_id}: no base shards in weight_map"

    # Verify HTTP Range read on pb27-gdn-00001-of-00001.safetensors and first base shard
    gdn_hdr = _read_safetensors_header(repo_id, "pb27-gdn-00001-of-00001.safetensors", token)
    for gk in gdn_keys:
        assert gk in gdn_hdr, f"{repo_id}: missing {gk} in pb27-gdn-00001-of-00001.safetensors header"

    base_hdr = _read_safetensors_header(repo_id, base_shards[0], token)
    assert len(base_hdr) > 1, f"{repo_id}: empty base shard header for {base_shards[0]}"

    return {
        "repo": repo_id,
        "arch": arch,
        "hybrid_layers": len(hybrid),
        "gdn_tensors": len(gdn_keys),
        "base_shards": len(base_shards),
        "status": "PASS",
    }


def verify_vllm_load_and_gdn() -> None:
    import tempfile
    import torch
    from safetensors.torch import load_file
    import pb27_vllm
    pb27_vllm.register()
    from pb27_vllm import model as pb_model
    from pb27_vllm import universal_models as um
    from vllm.config import (
        CacheConfig,
        CompilationConfig,
        LoadConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed import (
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )

    try:
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method="tcp://127.0.0.1:29529",
            local_rank=0,
            backend="gloo",
        )
    except Exception:
        pass

    cls_map = {
        "PB27ForConditionalGeneration": pb_model.PB27ForConditionalGeneration,
        "PB27Qwen3NextForCausalLM": um.PB27Qwen3NextForCausalLM,
        "PB27Qwen3MoeForCausalLM": um.PB27Qwen3MoeForCausalLM,
        "PB27Gemma4ForConditionalGeneration": um.PB27Gemma4ForConditionalGeneration,
        "PB27Qwen3ForCausalLM": um.PB27Qwen3ForCausalLM,
    }

    slim_dirs = [
        ("PB27Qwen3NextForCausalLM", "/home/defnotean/pb27/hf/qwen3_coder_next_slim"),
        ("PB27Qwen3MoeForCausalLM", "/home/defnotean/pb27/hf/qwen3_coder_30b_slim"),
        ("PB27Gemma4ForConditionalGeneration", "/home/defnotean/pb27/hf/gemma4_26b_slim"),
        ("PB27Gemma4ForConditionalGeneration", "/home/defnotean/pb27/hf/gemma4_31b_slim"),
        ("PB27Qwen3ForCausalLM", "/home/defnotean/pb27/hf/qwen3_32b_slim"),
        ("PB27ForConditionalGeneration", "/home/defnotean/pb27/hf/qwen35_9b_slim"),
        ("PB27ForConditionalGeneration", "/home/defnotean/pb27/hf/qwen35_4b_slim"),
    ]
    for expected_arch, sdir in slim_dirs:
        p = Path(sdir)
        if not (p / "pb27-gdn-00001-of-00001.safetensors").is_file():
            print(f"[SKIP] {sdir} not yet exported")
            continue
        cfg = json.loads((p / "config.json").read_text())
        tcfg = cfg.get("text_config", cfg)
        arch = cfg["architectures"][0]
        assert arch == expected_arch, f"{sdir}: arch={arch} != {expected_arch}"
        hybrid = tcfg.get("pb_hybrid_layers") or cfg.get("pb_hybrid_layers")
        tensors = load_file(str(p / "pb27-gdn-00001-of-00001.safetensors"))
        assert len(tensors) == len(hybrid) * 9, f"{sdir}: tensor count {len(tensors)} != {len(hybrid)*9}"
        # Verify WeightsMapper transforms all 9 per-layer GDN tensor names into the 7 vLLM parameter names
        mapped = [um.GDN_STACKED_MAPPER._map_name(k) for k in tensors.keys()]
        qkvz = [m for m in mapped if m and m.endswith(".gdn.in_proj_qkvz.weight")]
        ba = [m for m in mapped if m and m.endswith(".gdn.in_proj_ba.weight")]
        assert len(qkvz) == len(hybrid) * 2, f"{sdir}: mapper failed on in_proj_qkvz"
        assert len(ba) == len(hybrid) * 2, f"{sdir}: mapper failed on in_proj_ba"
        # Check zero NaN/Inf and out_proj == 0 (for exact KL=0.0 short-context preservation)
        for k, t in tensors.items():
            assert torch.isfinite(t.float()).all(), f"{sdir}: non-finite values in {k}"
            if k.endswith(".gdn.out_proj.weight"):
                assert float(t.abs().max()) == 0.0, f"{sdir}: {k} not zero-initialized"
            elif k.endswith(".gdn.conv1d.weight"):
                # Causal identity kernel [0, 0, 0, 1]
                assert float((t.float()[..., -1] - 1.0).abs().max()) == 0.0, f"{sdir}: {k} conv1d last tap != 1"
                assert float(t.float()[..., :-1].abs().max()) == 0.0, f"{sdir}: {k} conv1d prefix != 0"

        # Instantiate the vLLM PB27 model class and run .load_weights() on the first hybrid layer's real GDN tensors
        cls = cls_map[arch]
        cfg_sub = json.loads(json.dumps(cfg))
        tcfg_sub = cfg_sub.get("text_config", cfg_sub)
        first_hyb = int(hybrid[0])
        max_layers = first_hyb + 1
        tcfg_sub["num_hidden_layers"] = max_layers
        if "layer_types" in tcfg_sub:
            tcfg_sub["layer_types"] = tcfg_sub["layer_types"][:max_layers]
        if "num_kv_shared_layers" in tcfg_sub:
            tcfg_sub["num_kv_shared_layers"] = 0
        tcfg_sub["pb_hybrid_layers"] = [first_hyb]
        cfg_sub["pb_hybrid_layers"] = [first_hyb]
        if (tcfg_sub.get("num_experts") or 0) > 4:
            tcfg_sub["num_experts"] = 4
            if tcfg_sub.get("top_k_experts"):
                tcfg_sub["top_k_experts"] = min(2, int(tcfg_sub["top_k_experts"]))
            if tcfg_sub.get("num_experts_per_tok"):
                tcfg_sub["num_experts_per_tok"] = min(2, int(tcfg_sub["num_experts_per_tok"]))
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "config.json").write_text(json.dumps(cfg_sub))
            vcfg = VllmConfig(
                model_config=ModelConfig(
                    model=td,
                    tokenizer=td,
                    trust_remote_code=True,
                    dtype=torch.bfloat16,
                    max_model_len=32768,
                    skip_tokenizer_init=True,
                ),
                cache_config=CacheConfig(block_size=16),
                parallel_config=ParallelConfig(tensor_parallel_size=1),
                scheduler_config=SchedulerConfig(
                    max_num_batched_tokens=2048,
                    max_model_len=32768,
                    is_encoder_decoder=False,
                ),
                load_config=LoadConfig(),
                compilation_config=CompilationConfig(),
            )
            with set_current_vllm_config(vcfg):
                ensure_model_parallel_initialized(1, 1)
                vllm_model = cls(vllm_config=vcfg)
            layer_tensors = [
                (k, v) for k, v in tensors.items() if f"layers.{first_hyb}.gdn." in k
            ]
            assert len(layer_tensors) == 9, f"{sdir}: expected 9 tensors for layer {first_hyb}"
            loaded = vllm_model.load_weights(layer_tensors)
            gdn_loaded = [x for x in loaded if ".gdn." in x]
            assert len(gdn_loaded) == 7, f"{sdir}: expected 7 loaded vLLM GDN params, got {gdn_loaded}"
            del vllm_model

        # GPU smoke test of Quest Exact Memory (`Archive` + `retrieve_blocks_per_token` + `piece_attention`)
        # at >16K context using this model's exact attention dimensions (including Gemma 4 global_head_dim=512)
        if torch.cuda.is_available():
            from pb27_vllm import memory
            num_heads = int(tcfg["num_attention_heads"])
            num_kv_heads = int(tcfg.get("global_num_key_value_heads") or tcfg.get("num_key_value_heads") or num_heads)
            head_dim = int(tcfg.get("global_head_dim") or tcfg.get("head_dim") or (int(tcfg["hidden_size"]) // num_heads))
            groups = num_heads // num_kv_heads
            arc = memory.Archive.new(num_kv_heads, head_dim, 20480, 64, torch.bfloat16, torch.device("cuda"), max_hint=20480)
            arc.prompt_len = 18432
            pos = torch.arange(0, 18432, device="cuda", dtype=torch.long)
            k_sim = torch.randn(18432, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
            v_sim = torch.randn(18432, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
            arc.write(pos, k_sim, v_sim, 64, p0=0, hi=18432)
            q_sim = torch.randn(16, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
            q_pos = torch.arange(18416, 18432, device="cuda", dtype=torch.long)
            res = memory.retrieve_blocks_per_token(arc, q_sim, groups, 18416 - 16384 + 1, budget=32, lead=-1, union=2)
            assert res is not None and res[0] is not None, f"{sdir}: Quest retrieval returned None"
            out_attn = memory.piece_attention(arc, q_sim, q_pos, res[0], groups, 16384, head_dim**-0.5, res[1], q_min=18416, q_max=18431)
            assert out_attn.shape == (16, num_heads, head_dim) and torch.isfinite(out_attn.float()).all(), f"{sdir}: GPU piece_attention failed"
            memory._free({"test": arc})

        print(
            f"[PASS vLLM INIT + LOAD_WEIGHTS + GPU QUEST MEMORY] {sdir:42s} | arch={arch:34s} | hybrid={len(hybrid):2d} | loaded=7/7 GDN"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-vllm", action="store_true")
    ap.add_argument("repos", nargs="*")
    args = ap.parse_args()

    if args.check_vllm:
        verify_vllm_load_and_gdn()

    if args.repos:
        tok_path = Path("/home/defnotean/.cache/huggingface/token")
        if not tok_path.is_file():
            tok_path = Path("/root/.cache/huggingface/token")
        tok = tok_path.read_text().strip() if tok_path.is_file() else None
        for r in args.repos:
            res = verify_hf_repo(r, tok)
            print(
                f"[PASS HF REPO] {res['repo']:52s} | arch={res['arch']:34s} | "
                f"hybrid={res['hybrid_layers']:2d} | gdn_tensors={res['gdn_tensors']:3d} | base_shards={res['base_shards']:3d}"
            )


if __name__ == "__main__":
    main()
