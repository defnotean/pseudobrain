#!/usr/bin/env bash
# Extended long-context phase (256K..1M prompt tokens) for custom_longctx_bench.py on Spark:
#   custom_longctx_extended.sh <name> <lengths> <model-dir> <checkpoint> <sha>
# Waits for ~/pb27/memory_gates.lock, stops the standing server and serves the checkpoint from a COPY of the
# plug-in (~/pb27/serving_nvme_<ts>/pb27_vllm) whose memory.py is ~/pb27/prepared/memory.pertoken_lead.py (exact
# archive offloaded to bf16 file-backed mmap, PB27_MEMORY_OFFLOAD=nvme, PB27_MEMORY_NVME_DTYPE=bf16, plus
# per-token final-prefill & decode retrieval) and whose serve.sh also passes VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# and PB27_MEMORY_NVME_*; no speculative decoding. The standing tree is not touched.
# Afterwards the standing server is restored with start_pb27_studio.sh for the same checkpoint and its health is checked.
set -u
NAME=$1; LENGTHS=$2; CAND=$3; TAG=$4; SHA=$5
ROOT=/home/defnotean/pb27
OUT="$ROOT/reports/custom_longctx"
LOG="$OUT/$NAME.run.log"
DONE="CUSTOM LONGCTX $NAME FINISHED"
COPY="$ROOT/serving_nvme_$(date +%Y%m%dT%H%M%S)/pb27_vllm"
MAXLEN=${EXT_MAXLEN:-1056768}   # 1M + 8K
mkdir -p "$OUT"
log() { echo "[$(date -Is)] $*" >> "$LOG"; }
log "queued (pid=$$) lengths=$LENGTHS cand=$CAND; waiting for memory_gates.lock"
exec 9>"$ROOT/memory_gates.lock"
flock 9
log "lock acquired"
bash "$ROOT/mem_watchdog.sh" "$LOG" "$DONE" >> "$LOG.watchdog" 2>&1 9>&- &

stop_server() {
  docker stop -t 30 pb27-vllm >/dev/null 2>&1 || true
  docker rm -f pb27-vllm >/dev/null 2>&1 || true
  sleep 5
}

wait_healthy() {  # wait_healthy <log>
  for _ in $(seq 1 120); do
    curl -fsS --max-time 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && { log "server healthy"; return 0; }
    grep -a -q "Traceback" "$1" 2>/dev/null && { log "server failed: see $1"; return 1; }
    sleep 10
  done
  log "server not healthy after 20 min"; return 1
}

restore() {
  stop_server
  PB27_MODEL_DIR="$CAND" PB27_CHECKPOINT="$TAG" PB27_SHA="$SHA" bash "$ROOT/start_pb27_studio.sh" >> "$LOG" 2>&1 9>&- || true
  wait_healthy "$ROOT/vllm_pb27_dflash.log" && log "standing server restored: $(curl -s -m 5 http://127.0.0.1:18080/health | head -c 300)"
}

mkdir -p "$(dirname "$COPY")"
cp -a "$ROOT/serving/pb27_vllm" "$COPY"
rm -rf "$COPY"/*.egg-info "$COPY"/pb27_vllm/__pycache__
cp "$ROOT/prepared/memory.pertoken_lead.py" "$COPY/pb27_vllm/memory.py"
if ! grep -q VLLM_ALLOW_LONG_MAX_MODEL_LEN "$COPY/serve.sh"; then
  sed -i 's|-e PB27_MEMORY_RETAIN="$RETAIN"|-e PB27_MEMORY_RETAIN="$RETAIN" -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e PB27_MEMORY_NVME_DIR="${PB27_MEMORY_NVME_DIR:-}" -e PB27_MEMORY_NVME_CAP="${PB27_MEMORY_NVME_CAP:-}" -e PB27_MEMORY_NVME_DTYPE="${PB27_MEMORY_NVME_DTYPE:-bf16}" -e PB27_MEMORY_PERTOKEN="${PB27_MEMORY_PERTOKEN:-}" -e PB27_MEMORY_PERTOKEN_CAND="${PB27_MEMORY_PERTOKEN_CAND:-}" -e PB27_MEMORY_PERTOKEN_UNION="${PB27_MEMORY_PERTOKEN_UNION:-}" -e PB27_MEMORY_PERTOKEN_LEAD="${PB27_MEMORY_PERTOKEN_LEAD:-}" -e PB27_MEMORY_PERTOKEN_OPEN="${PB27_MEMORY_PERTOKEN_OPEN:-}" -e PB27_MEMORY_PERTOKEN_PREFILL_UNION="${PB27_MEMORY_PERTOKEN_PREFILL_UNION:-}"|' "$COPY/serve.sh"
fi
grep -q VLLM_ALLOW_LONG_MAX_MODEL_LEN "$COPY/serve.sh" || { log "serve.sh patch failed"; echo "$DONE" >> "$LOG"; exit 1; }
log "nvme copy ready: memory.py sha $(sha256sum "$COPY/pb27_vllm/memory.py" | cut -c1-16)"

stop_server
(
  cd "$ROOT"
  PB27_MEMORY=64,64,0,0,256 PB27_MEMORY_SCALE=${PB27_MEMORY_SCALE:-256:128:2048:256} PB27_MEMORY_PREFILL=pt PB27_MEMORY_SINKS=4 PB27_EAGER=1 \
  PB27_MEMORY_OFFLOAD=nvme PB27_MEMORY_NVME_DTYPE=${PB27_MEMORY_NVME_DTYPE:-bf16} PB27_MEMORY_NVME_CAP=$MAXLEN \
  PB27_MEMORY_PERTOKEN=${PB27_MEMORY_PERTOKEN:-1} PB27_MEMORY_PERTOKEN_CAND=${PB27_MEMORY_PERTOKEN_CAND:-2} \
  PB27_MEMORY_PERTOKEN_UNION=${PB27_MEMORY_PERTOKEN_UNION:-4} PB27_MEMORY_PERTOKEN_LEAD=${PB27_MEMORY_PERTOKEN_LEAD:-1} \
  PB27_MEMORY_PERTOKEN_OPEN=${PB27_MEMORY_PERTOKEN_OPEN:-1} PB27_MEMORY_PERTOKEN_PREFILL_UNION=${PB27_MEMORY_PERTOKEN_PREFILL_UNION:-4} \
  GPU_MEM=${GPU_MEM:-0.70} PB27_KV_CACHE_MEMORY=${PB27_KV_CACHE_MEMORY:-6G} MAXLEN=$MAXLEN QUANT= SPEC= \
  bash "$COPY/serve.sh" "$CAND" pb27 8000 > "$OUT/$NAME.vllm.log" 2>&1 < /dev/null 9>&- &
)
if wait_healthy "$OUT/$NAME.vllm.log"; then
  python3 "$ROOT/scripts/bench/custom_longctx_bench.py" --name "$NAME" --lengths "$LENGTHS" --request-timeout 21600 --opencode-yield 120 >> "$LOG" 2>&1 9>&-
  log "bench exit $?"
else
  log "extended server did not come up; see $NAME.vllm.log"
fi
restore
echo "$DONE" >> "$LOG"
