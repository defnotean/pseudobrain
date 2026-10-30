#!/usr/bin/env bash
# Extended 200K..1M long-context needle/multi4/trace4 suite for flash_next (Qwen3.8-Flash-Next):
#   bash scripts/bench/run_flash_next_ext.sh
# Waits for ~/pb27/memory_gates.lock, runs 200K and 250K against the live server, then refreshes pb27-vllm
# with MAXLEN=1056768 and DFlash2 max_position_embeddings=1114112 to run 500K and 1M.
set -u
ROOT=/home/defnotean/pb27
OUT="$ROOT/reports/custom_longctx"
LOG="$OUT/flash_next_ext.run.log"
DONE="FLASH_NEXT EXT SUITE FINISHED"
mkdir -p "$OUT"
log() { echo "[$(date -Is)] $*" >> "$LOG"; }

log "queued (pid=$$); waiting for memory_gates.lock"
exec 9>"$ROOT/memory_gates.lock"
flock 9
log "lock acquired (pid=$$)"
bash "$ROOT/mem_watchdog.sh" "$LOG" "$DONE" >> "$LOG.watchdog" 2>&1 9>&- &

wait_healthy() {
  for _ in $(seq 1 120); do
    curl -fsS --max-time 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && { log "server healthy"; return 0; }
    grep -a -q "Traceback" "$1" 2>/dev/null && { log "server failed: see $1"; return 1; }
    sleep 10
  done
  log "server not healthy after 20 min"; return 1
}

log "starting pb27-vllm with MAXLEN=1056768 and DFlash2 max_position_embeddings=1114112 for 200K..1M"
if ! docker ps --format '{{.Names}}' | grep -qx pb27-vllm; then
  MAXLEN=1056768 bash "$ROOT/start_pb27_studio.sh" >> "$LOG" 2>&1 9>&- || true
fi

if wait_healthy "$ROOT/vllm_pb27_dflash.log"; then
  log "running custom_longctx_bench.py (flash_next_ext 200K, 250K, 500K, 1M)"
  python3 "$ROOT/scripts/bench/custom_longctx_bench.py" --name flash_next_ext --lengths 200000,250000,500000,1000000 --request-timeout 21600 --opencode-yield 10 --retry-failed >> "$LOG" 2>&1 9>&-
  log "200K..1M exit $?"
else
  log "server failed to come up for 200K..1M"
fi

echo "$DONE" >> "$LOG"
