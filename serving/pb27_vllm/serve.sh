#!/usr/bin/env bash
# Serve PB-27 with vLLM on the Spark (GB10) through the out-of-tree plug-in in this directory.
#   serving/pb27_vllm/serve.sh [model-dir] [served-name] [port] [extra vllm args...]
# Defaults: model-dir $HOME/models/pb27a (mounted at /models/pb27a), name pb27, port 8000.
# Runs in the foreground under ~/gpu_run.sh so it holds the GPU FIFO lease for its lifetime
# (same docker flags as scripts/bench/serve_vllm.sh). Stop it with: docker stop pb27-vllm
#
# The package is bind-mounted at /pb27_vllm and `pip install -e` is run inside the container
# at start-up, which writes the `vllm.general_plugins` entry point that vLLM reads in every
# process (vllm/plugins/__init__.py: load_general_plugins). Set PB27_INSTALL=shim to skip pip
# and use PYTHONPATH plus a sitecustomize that registers the model instead (no network, no
# writable site-packages needed).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_DIR=${1:-$HOME/models/pb27a}; NAME=${2:-pb27}; PORT=${3:-8000}; shift $(( $# < 3 ? $# : 3 ))
IMAGE=${VLLM_IMAGE:-vllm/vllm-openai:v0.31.0-aarch64}
MAXLEN=${MAXLEN:-1048576}
MAX_BATCH_TOKENS=${MAX_BATCH_TOKENS:-8192}
MEM=${GPU_MEM:-0.80}
INSTALL=${PB27_INSTALL:-pip}
# Speed knobs (lossless in principle; each is gated on the battery before it becomes the default):
#   QUANT=fp8  -> --quantization fp8 (online FP8 weights)
#   SPEC='{"method":"qwen3_5_mtp","num_speculative_tokens":3}' -> --speculative-config (MTP drafts)
#   PB27_EAGER=0 -> drop --enforce-eager (CUDA graphs)
#   PB27_MEMORY=blocks,block_size,decode_blocks,neighbors,rerank -> exact memory tier (pb27_vllm/memory.py); eager only in v1
#   PB27_PREFIX_CACHE=1 -> vLLM prefix caching; the memory tier retains PB27_MEMORY_RETAIN finished archives for reuse
#                          (experimental, not yet qualified)
QUANT=${QUANT:-}
SPEC=${SPEC:-}
EAGER=${PB27_EAGER:-1}
KV_MEM=${PB27_KV_CACHE_MEMORY:-}
extra=""
[ -n "$QUANT" ] && extra="$extra --quantization $QUANT"
[ "$EAGER" = 1 ] && extra="$extra --enforce-eager"
[ -n "$KV_MEM" ] && extra="$extra --kv-cache-memory-bytes $KV_MEM"
RETAIN=0
if [ "${PB27_PREFIX_CACHE:-0}" = 1 ]; then extra="$extra --enable-prefix-caching"; RETAIN=${PB27_MEMORY_RETAIN:-2}; else extra="$extra --no-enable-prefix-caching"; fi
spec_arg=""
[ -n "$SPEC" ] && spec_arg="--speculative-config '$SPEC'"
MODEL_DIR=$(cd "$MODEL_DIR" && pwd)
# Mount the model at its host path: the slim export links its stock shards by relative path into
# the host HF cache, so both must sit at the same paths inside the container.
MOUNT=$MODEL_DIR

case "$INSTALL" in
  pip)
    setup='pip install --no-deps --no-build-isolation -q -e /pb27_vllm' ;;
  shim)
    # Register without entry points: vLLM also honours plugins that are simply importable
    # if we call register() ourselves before the engine starts. sitecustomize runs in every
    # python process (API server, engine core, workers), matching the entry-point behaviour.
    setup='mkdir -p /pb27_shim && printf "import pb27_vllm\npb27_vllm.register()\n" > /pb27_shim/sitecustomize.py && export PYTHONPATH=/pb27_vllm:/pb27_shim${PYTHONPATH:+:$PYTHONPATH}' ;;
  *) echo "PB27_INSTALL must be pip or shim" >&2; exit 2 ;;
esac

# --enforce-eager and --no-enable-prefix-caching: see docs/PB27_VLLM_PLUGIN.md sections 3 and 5
# (two cache-bearing modules in one decoder layer; cudagraphs/compile and prefix caching are
# later steps). No speculative config: MTP for the hybrid model is step (e).
mkdir -p "$HOME/.cache/vllm" "$HOME/.cache/flashinfer"
exec ~/gpu_run.sh docker run --rm --name pb27-vllm --gpus all --ipc host --network host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/vllm:/root/.cache/vllm" -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
  -e HF_HUB_OFFLINE=1 \
  -e PB27_MEMORY="${PB27_MEMORY:-256,64,256,0,512}" -e PB27_MEMORY_DEBUG="${PB27_MEMORY_DEBUG:-}" -e PB27_MEMORY_NORETRIEVE="${PB27_MEMORY_NORETRIEVE:-}" -e PB27_MEMORY_SINKS="${PB27_MEMORY_SINKS:-4}" -e PB27_MEMORY_PREFILL="${PB27_MEMORY_PREFILL:-pt}" -e PB27_MEMORY_RERANK_ALL="${PB27_MEMORY_RERANK_ALL:-1}" -e PB27_MEMORY_PREFILL_BLOCKS="${PB27_MEMORY_PREFILL_BLOCKS:-0}" -e PB27_MEMORY_ORACLE="${PB27_MEMORY_ORACLE:-}" -e PB27_MEMORY_FAITHFUL="${PB27_MEMORY_FAITHFUL:-1}" -e PB27_MEMORY_QWIN="${PB27_MEMORY_QWIN:-}" -e PB27_MEMORY_DSTEP="${PB27_MEMORY_DSTEP:-}" -e PB27_MEMORY_SCALE="${PB27_MEMORY_SCALE:-2048:384:4096:512:8192:512}" -e PB27_MEMORY_ROPE_HORIZON="${PB27_MEMORY_ROPE_HORIZON:-262144}" -e PB27_MEMORY_ROPE_ARC_SPAN="${PB27_MEMORY_ROPE_ARC_SPAN:-229376}" -e PB27_MEMORY_ROPE_THETA="${PB27_MEMORY_ROPE_THETA:-10000000.0}" -e PB27_MEMORY_OFFLOAD="${PB27_MEMORY_OFFLOAD:-nvme}" -e PB27_MEMORY_RETAIN="$RETAIN" -e VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}" -e PB27_MEMORY_NVME_DIR="${PB27_MEMORY_NVME_DIR:-/root/.cache/vllm}" -e PB27_MEMORY_NVME_CAP="${PB27_MEMORY_NVME_CAP:-1114112}" -e PB27_MEMORY_NVME_DTYPE="${PB27_MEMORY_NVME_DTYPE:-bf16}" -e PB27_MEMORY_PERTOKEN="${PB27_MEMORY_PERTOKEN:-1}" -e PB27_MEMORY_PERTOKEN_CAND="${PB27_MEMORY_PERTOKEN_CAND:-4}" -e PB27_MEMORY_PERTOKEN_UNION="${PB27_MEMORY_PERTOKEN_UNION:-2}" -e PB27_MEMORY_PERTOKEN_LEAD="${PB27_MEMORY_PERTOKEN_LEAD:-1}" -e PB27_MEMORY_PERTOKEN_OPEN="${PB27_MEMORY_PERTOKEN_OPEN:-1}" -e PB27_MEMORY_PERTOKEN_PREFILL_UNION="${PB27_MEMORY_PERTOKEN_PREFILL_UNION:-2}" -e PB27_MEMORY_DECODE_STRIDE="${PB27_MEMORY_DECODE_STRIDE:-64}" -e PB27_MEMORY_DECODE_BLOCKS="${PB27_MEMORY_DECODE_BLOCKS:-64}" -e PB27_MEMORY_DECODE_RERANK="${PB27_MEMORY_DECODE_RERANK:-0}" -e PB27_MEMORY_SUPERBLOCK="${PB27_MEMORY_SUPERBLOCK:-64}" -e PB27_MEMORY_LAYER_STRIDE="${PB27_MEMORY_LAYER_STRIDE:-1}" -e PB27_MEMORY_GPU_ARC_CAP="${PB27_MEMORY_GPU_ARC_CAP:-262144}" \
  -v "$HOME/.cache/huggingface:$HOME/.cache/huggingface:ro" \
  -v "$MODEL_DIR:$MOUNT:ro" -v "$HERE:/pb27_vllm" \
  --entrypoint bash "$IMAGE" -c "$setup && exec vllm serve $MOUNT \
    --served-model-name $NAME --port $PORT --host 127.0.0.1 \
    --max-model-len $MAXLEN --max-num-batched-tokens $MAX_BATCH_TOKENS --max-num-seqs 16 --gpu-memory-utilization $MEM \
    $extra $spec_arg \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder $*"
