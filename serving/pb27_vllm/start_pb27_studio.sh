#!/usr/bin/env bash
# Install at ~/pb27/start_pb27_studio.sh on the Spark.
# Keep the standing memory profile explicit; experimental profiles can override it.
set -euo pipefail
ROOT=$HOME/pb27
export PB27_MEMORY=${PB27_MEMORY-256,64,256,0,512}
export PB27_MEMORY_SCALE=${PB27_MEMORY_SCALE-2048:384:4096:512:8192:1536}
export PB27_MEMORY_PREFILL=${PB27_MEMORY_PREFILL:-pt}
export PB27_MEMORY_SINKS=${PB27_MEMORY_SINKS:-4}
export PB27_MEMORY_OFFLOAD=${PB27_MEMORY_OFFLOAD:-nvme}
export PB27_MEMORY_NVME_DTYPE=${PB27_MEMORY_NVME_DTYPE:-bf16}
export PB27_MEMORY_FAITHFUL=${PB27_MEMORY_FAITHFUL:-1}
export PB27_MEMORY_PERTOKEN=${PB27_MEMORY_PERTOKEN:-1}
export PB27_MEMORY_PERTOKEN_CAND=${PB27_MEMORY_PERTOKEN_CAND:-4}
export PB27_MEMORY_PERTOKEN_UNION=${PB27_MEMORY_PERTOKEN_UNION:-2}
export PB27_MEMORY_PERTOKEN_LEAD=${PB27_MEMORY_PERTOKEN_LEAD:-1}
export PB27_MEMORY_PERTOKEN_OPEN=${PB27_MEMORY_PERTOKEN_OPEN:-1}
export PB27_MEMORY_PERTOKEN_PREFILL_UNION=${PB27_MEMORY_PERTOKEN_PREFILL_UNION:-2}
export PB27_MEMORY_DECODE_STRIDE=${PB27_MEMORY_DECODE_STRIDE:-4}
export PB27_EAGER=1  # the archive/retrieval path is not CUDA-graph safe yet
export GPU_MEM=${GPU_MEM:-0.75}
export PB27_KV_CACHE_MEMORY=${PB27_KV_CACHE_MEMORY:-8G}
export MAXLEN=${MAXLEN:-131072}
export QUANT=
# An explicit empty SPEC permits the non-speculative diagnostic; unset preserves the default.
export SPEC=${SPEC-'{"method":"dflash","model":"z-lab/Qwen3.8-27B-DFlash2","num_speculative_tokens":16}'}
# Checkpoint selection; the defaults are the standing pb27c server (rollback: pb27b at $ROOT/hf/pb27b_nvfp4, sha f84d2ab72bf88529301fec1363c84c507a79f497a35c05dbc0fa9513f5226d19).
MODEL_DIR=${PB27_MODEL_DIR:-$ROOT/hf/pb27c_nvfp4}
CHECKPOINT=${PB27_CHECKPOINT:-pb27c}
SHA=${PB27_SHA:-45c66437693cc96eaf23c6890c04559830e2c6aaca6aee213dc89223fde52a90}

if ! docker ps --format '{{.Names}}' | grep -qx pb27-vllm; then
  (
    cd "$ROOT"
    nohup bash "$ROOT/serving/pb27_vllm/serve.sh" "$MODEL_DIR" pb27 8000 \
      > "$ROOT/vllm_pb27_dflash.log" 2>&1 < /dev/null &
  )
  echo "vLLM launched: $CHECKPOINT ($MODEL_DIR) memory=$PB27_MEMORY scale=$PB27_MEMORY_SCALE prefill=$PB27_MEMORY_PREFILL (startup takes several minutes)"
fi
if ss -ltn 2>/dev/null | grep -q ':18080 ' && ! curl -s -m 2 http://127.0.0.1:18080/health | grep -q "$SHA"; then
  p=$(ss -ltnp 2>/dev/null | awk '/:18080 / && match($0, /pid=[0-9]+/) { print substr($0, RSTART+4, RLENGTH-4); exit }')
  if [ -n "$p" ] && ps -fp "$p" | grep -q "studio_proxy.py"; then
    kill "$p" && sleep 1
  fi
fi
if ! ss -ltn 2>/dev/null | grep -q ':18080 '; then
  nohup python3 "$ROOT/serving/pb27_vllm/studio_proxy.py" --listen 18080 \
    --upstream http://127.0.0.1:8000 --model pb27 --public-model pb27 \
    --sha "$SHA" --checkpoint "$CHECKPOINT" \
    --context 131072 --decoder "vLLM NVFP4 + DFlash2" \
    > "$ROOT/studio_front.log" 2>&1 < /dev/null &
  echo "Studio front launched on :18080"
fi
echo "ready-check: $(curl -s -m 5 http://127.0.0.1:18080/health | head -c 160)"
