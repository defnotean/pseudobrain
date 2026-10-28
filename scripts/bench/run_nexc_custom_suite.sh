#!/usr/bin/env bash
# Non-intrusive custom benchmark suite for MoE nexc (~/pb27/hf/nexc_nvfp4, 35B-A3B qwen3_5_moe):
#   bash scripts/bench/run_nexc_custom_suite.sh
# Runs custom_code_bench.py (nexc_med) and custom_longctx_bench.py (nexc_pt, 8K..120K) under
# ~/pb27/memory_gates.lock with SPEC="" (no 27B DFlash2 drafter), then restores standing pb27c.
set -u
ROOT=/home/defnotean/pb27
CAND=${NEXC_MODEL_DIR:-$ROOT/hf/nexc_nvfp4}
OUT="$ROOT/reports/custom_longctx"
LOG="$OUT/nexc_suite.run.log"
DONE="NEXC CUSTOM SUITE FINISHED"
mkdir -p "$OUT" "$ROOT/reports/custom_code"
log() { echo "[$(date -Is)] $*" >> "$LOG"; }

log "queued (pid=$$) cand=$CAND; waiting for memory_gates.lock"
exec 9>"$ROOT/memory_gates.lock"
flock 9
log "lock acquired"
bash "$ROOT/mem_watchdog.sh" "$LOG" "$DONE" >> "$LOG.watchdog" 2>&1 9>&- &

stop_server() {
  docker stop -t 30 pb27-vllm >/dev/null 2>&1 || true
  docker rm -f pb27-vllm >/dev/null 2>&1 || true
  sleep 5
}

wait_healthy() {
  for _ in $(seq 1 120); do
    curl -fsS --max-time 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && { log "server healthy"; return 0; }
    grep -a -q "Traceback" "$1" 2>/dev/null && { log "server failed: see $1"; return 1; }
    sleep 10
  done
  log "server not healthy after 20 min"; return 1
}

restore() {
  stop_server
  bash "$ROOT/start_pb27_studio.sh" >> "$LOG" 2>&1 9>&- || true
  wait_healthy "$ROOT/vllm_pb27_dflash.log" && log "standing server restored: $(curl -s -m 5 http://127.0.0.1:18080/health | head -c 300)"
  echo "$DONE" >> "$LOG"
}
trap restore EXIT

stop_server
(
  cd "$ROOT"
  PB27_MEMORY=64,64,0,0,256 PB27_MEMORY_SCALE=256:128:2048:256 PB27_MEMORY_PREFILL=pt PB27_MEMORY_SINKS=4 PB27_EAGER=1 \
  PB27_MEMORY_OFFLOAD=nvme PB27_MEMORY_NVME_DTYPE=bf16 PB27_MEMORY_NVME_CAP=1056768 \
  PB27_MEMORY_PERTOKEN=1 PB27_MEMORY_PERTOKEN_CAND=2 PB27_MEMORY_PERTOKEN_UNION=4 \
  PB27_MEMORY_PERTOKEN_LEAD=1 PB27_MEMORY_PERTOKEN_OPEN=1 PB27_MEMORY_PERTOKEN_PREFILL_UNION=4 \
  GPU_MEM=0.70 PB27_KV_CACHE_MEMORY=6G MAXLEN=1056768 QUANT= SPEC= \
  bash "$ROOT/serving/pb27_vllm/serve.sh" "$CAND" pb27 8000 > "$OUT/nexc_suite.vllm.log" 2>&1 < /dev/null 9>&- &
)

if wait_healthy "$OUT/nexc_suite.vllm.log"; then
  log "running custom_longctx_bench.py (nexc_pt 8K..120K)"
  python3 "$ROOT/scripts/bench/custom_longctx_bench.py" --name nexc_pt --lengths 8192,32768,65536,120000 --request-timeout 7200 --opencode-yield 120 >> "$LOG" 2>&1 9>&-
  log "nexc_pt exit $?"
  log "running custom_longctx_bench.py (nexc_ext 200K..1M)"
  python3 "$ROOT/scripts/bench/custom_longctx_bench.py" --name nexc_ext --lengths 200000,250000,500000,1000000 --request-timeout 21600 --opencode-yield 120 >> "$LOG" 2>&1 9>&-
  log "nexc_ext exit $?"
  log "running custom_code_bench.py (nexc_med)"
  python3 "$ROOT/scripts/bench/custom_code_bench.py" --name nexc_med --samples 2 --opencode-yield 120 >> "$LOG" 2>&1 9>&-
  log "nexc_med exit $?"
else
  log "nexc server did not come up; see nexc_suite.vllm.log"
fi
