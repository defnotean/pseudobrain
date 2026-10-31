#!/usr/bin/env bash
# Switch the standing Spark OpenCode server (:8000 / :18080) back and forth between:
#   before | pb27c            -> ~/pb27/hf/pb27c_nvfp4 (28.86B pre-spatial PB-27C)
#   after  | spatial | pb27s  -> ~/pb27/hf/pb27s_nvfp4 (30.71B PB-27S with +1.856B spatial_mlp)
#   pb27b                     -> ~/pb27/hf/pb27b_nvfp4 (28.86B PB-27B rollback)
#   status                    -> show active server checkpoint and saved model directories
set -euo pipefail

ROOT=$HOME/pb27
TARGET=${1:-status}

show_status() {
  echo "=== Active Server (:18080) ==="
  curl -s -m 3 http://127.0.0.1:18080/health || echo "(server not responding on :18080)"
  echo ""
  echo "=== Saved Checkpoints in $ROOT/hf ==="
  for tag in pb27c pb27c_nvfp4 pb27s pb27s_nvfp4 pb27b pb27b_nvfp4 nexc nexc_nvfp4; do
    if [ -d "$ROOT/hf/$tag" ]; then
      echo "  [READY] $ROOT/hf/$tag"
    else
      echo "  [PENDING] $ROOT/hf/$tag"
    fi
  done
}

case "$TARGET" in
  status|list)
    show_status
    exit 0
    ;;
  before|pb27c)
    CHECKPOINT="pb27c"
    MODEL_DIR="$ROOT/hf/pb27c_nvfp4"
    SHA=$(sha256sum "$MODEL_DIR/pb27-gdn-00001-of-00001.safetensors" | cut -c1-64)
    ;;
  after|spatial|pb27s)
    CHECKPOINT="pb27s"
    MODEL_DIR="$ROOT/hf/pb27s_nvfp4"
    if [ ! -f "$MODEL_DIR/pb27-spatial-00001-of-00001.safetensors" ]; then
      echo "Error: $MODEL_DIR/pb27-spatial-00001-of-00001.safetensors is not ready yet (Colab G4 training still in progress)." >&2
      exit 1
    fi
    SHA=$(sha256sum "$MODEL_DIR/pb27-spatial-00001-of-00001.safetensors" | cut -c1-64)
    ;;
  pb27b)
    CHECKPOINT="pb27b"
    MODEL_DIR="$ROOT/hf/pb27b_nvfp4"
    SHA=$(sha256sum "$MODEL_DIR/pb27-gdn-00001-of-00001.safetensors" | cut -c1-64)
    ;;
  *)
    echo "Usage: bash $0 {status|before|pb27c|after|spatial|pb27s|pb27b}" >&2
    exit 1
    ;;
esac

if [ ! -d "$MODEL_DIR" ]; then
  echo "Error: model directory $MODEL_DIR does not exist." >&2
  exit 1
fi

echo "Switching standing server to $CHECKPOINT ($MODEL_DIR, sha=${SHA:0:12}...)..."
docker rm -f pb27-vllm >/dev/null 2>&1 || true

PB27_MODEL_DIR="$MODEL_DIR" \
PB27_CHECKPOINT="$CHECKPOINT" \
PB27_SHA="$SHA" \
bash "$ROOT/start_pb27_studio.sh"

echo "Waiting for vLLM + DFlash2 warmup on :18080..."
for i in $(seq 1 90); do
  if curl -s -m 3 http://127.0.0.1:18080/health | grep -q '"ready": true'; then
    echo "Switch complete!"
    curl -s -m 3 http://127.0.0.1:18080/health
    echo ""
    exit 0
  fi
  sleep 2
done

echo "Server is still warming up; check logs with: tail -f $ROOT/vllm_pb27_dflash.log"
