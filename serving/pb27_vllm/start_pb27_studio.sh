#!/usr/bin/env bash
# Install at ~/pb27/start_pb27_studio.sh on the Spark.
# Keep the standing memory profile explicit; experimental profiles can override it.
set -euo pipefail
ROOT=$HOME/pb27
export PB27_MEMORY=${PB27_MEMORY-128,64,128,0,256}
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
export PB27_MEMORY_DECODE_STRIDE=${PB27_MEMORY_DECODE_STRIDE:-64}
export PB27_MEMORY_DECODE_BLOCKS=${PB27_MEMORY_DECODE_BLOCKS:-64}
export PB27_MEMORY_DECODE_RERANK=${PB27_MEMORY_DECODE_RERANK:-0}
export PB27_MEMORY_SUPERBLOCK=${PB27_MEMORY_SUPERBLOCK:-64}
export PB27_EAGER=1  # the archive/retrieval path is not CUDA-graph safe yet
export GPU_MEM=${GPU_MEM:-0.75}
export PB27_KV_CACHE_MEMORY=${PB27_KV_CACHE_MEMORY:-6G}
export MAXLEN=${MAXLEN:-1056768}
export MAX_BATCH_TOKENS=${MAX_BATCH_TOKENS:-8192}
export QUANT=
# An explicit empty SPEC permits the non-speculative diagnostic; unset preserves the default.
export SPEC=${SPEC-'{"method":"dflash","model":"incoai/Qwen3.8-Flash-Next-DFlash2","num_speculative_tokens":8}'}
# Checkpoint selection; default is flash_next (rollback: pb27c at $ROOT/hf/pb27c_nvfp4, pb27b at $ROOT/hf/pb27b_nvfp4).
MODEL_DIR=${PB27_MODEL_DIR:-$ROOT/hf/flash_next_nvfp4}
CHECKPOINT=${PB27_CHECKPOINT:-flash_next}
SHA=${PB27_SHA:-eb604fc3920c510e4cb8b35f04ba35682bcae148a085c3c61ed3337894f3bf7d}

if ! docker ps --format '{{.Names}}' | grep -qx pb27-vllm; then
  (
    cd "$ROOT"
    nohup bash "$ROOT/serving/pb27_vllm/serve.sh" "$MODEL_DIR" pb27 8000 \
      > "$ROOT/vllm_pb27_dflash.log" 2>&1 < /dev/null &
  )
  echo "vLLM launched: $CHECKPOINT ($MODEL_DIR) memory=$PB27_MEMORY scale=$PB27_MEMORY_SCALE prefill=$PB27_MEMORY_PREFILL (startup takes several minutes)"
fi
if ss -ltn 2>/dev/null | grep -q ':18080 ' && ( ! curl -s -m 2 http://127.0.0.1:18080/health | grep -q "$SHA" || ! curl -s -m 2 http://127.0.0.1:18080/health | grep -q "\"context\": $MAXLEN" ); then
  p=$(ss -ltnp 2>/dev/null | awk '/:18080 / && match($0, /pid=[0-9]+/) { print substr($0, RSTART+4, RLENGTH-4); exit }')
  if [ -n "$p" ] && ps -fp "$p" | grep -q "studio_proxy.py"; then
    kill "$p" && sleep 1
  fi
fi
if ! ss -ltn 2>/dev/null | grep -q ':18080 '; then
  nohup python3 "$ROOT/serving/pb27_vllm/studio_proxy.py" --listen 18080 \
    --upstream http://127.0.0.1:8000 --model pb27 --public-model pb27 \
    --sha "$SHA" --checkpoint "$CHECKPOINT" \
    --context "$MAXLEN" --decoder "vLLM NVFP4 + DFlash2" \
    > "$ROOT/studio_front.log" 2>&1 < /dev/null &
  echo "Studio front launched on :18080"
fi
echo "ready-check: $(curl -s -m 5 http://127.0.0.1:18080/health | head -c 160)"
