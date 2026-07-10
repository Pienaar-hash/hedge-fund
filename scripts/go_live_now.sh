#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Export environment from .env without leaking secrets
set -a
source ./.env
set +a
export PYTHONPATH="$(pwd)"
export BINANCE_TESTNET="${BINANCE_TESTNET:-0}"
export ENV="${ENV:-prod}"
export DRY_RUN="${DRY_RUN:-0}"
export EVENT_GUARD="${EVENT_GUARD:-1}"
export FIRESTORE_ENABLED="${FIRESTORE_ENABLED:-0}"
export PYTHONUNBUFFERED=1

# Basic exchange reachability
echo "== Exchange reachability =="
curl -fsS https://fapi.binance.com/fapi/v1/ping >/dev/null \
  && echo "exchange ping OK" \
  || echo "exchange ping failed"

# Optional margin mode helper
if [[ -x "scripts/margin_mode_once.py" ]]; then
  echo "== Margin mode (CROSS) =="
  BINANCE_TESTNET=0 ./venv/bin/python scripts/margin_mode_once.py || true
fi

# One-shot warmup with timeout safeguard
if [[ -x "scripts/exec_once_timeout.sh" ]]; then
  echo "== Warmup ONE_SHOT with timeout =="
  EXECUTOR_MAX_SEC=45 bash scripts/exec_once_timeout.sh || true
fi

# Continuous executor with log rollover
mkdir -p logs
LOGFILE="logs/executor_live.$(date -u +%Y%m%dT%H%M%SZ).log"
echo "== Starting continuous live executor (logging to $LOGFILE) =="
while true; do
  date -u +"[start %FT%TZ]" | tee -a "$LOGFILE"
  ONE_SHOT=0 ./venv/bin/python -m execution.executor_live 2>&1 | tee -a "$LOGFILE"
  code=$?
  date -u +"[exit %FT%TZ rc=$code]" | tee -a "$LOGFILE"
  sleep 3
done
