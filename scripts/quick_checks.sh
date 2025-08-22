#!/usr/bin/env bash
# Idempotent one-liners for routine checks.
set -Eeuo pipefail

ROOT="/root/hedge-fund"
ENV_FILE="$ROOT/.env"

echo "== Supervisor status =="
supervisorctl status || true
echo

echo "== NGINX reachability (expect 401/200 with auth) =="
curl -s -o /dev/null -w "HTTP %{http_code}\n" http://167.235.205.126/ || true
echo

echo "== Executor environment (Firestore/Binance) =="
pid=$(supervisorctl pid hedge-executor 2>/dev/null || true)
if [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] && [[ -r /proc/$pid/environ ]]; then
  tr '\0' '\n' < /proc/$pid/environ | egrep 'ENV=|USE_FUTURES|BINANCE_|GOOGLE|FIREBASE|PYTHONPATH' | sort || true
else
  echo "executor pid not found"
fi
echo

echo "== Hedge vs One-way (dualSide) =="
PYTHONPATH="$ROOT" "$ROOT/venv/bin/python" - <<'PY' || true
from execution.exchange_utils import _is_dual_side
print("dualSide:", _is_dual_side())
PY
echo

echo "== Klines sanity (BTCUSDT 15m last 3 closes) =="
PYTHONPATH="$ROOT" "$ROOT/venv/bin/python" - <<'PY' || true
from execution.exchange_utils import get_klines
k = get_klines("BTCUSDT","15m",limit=3)
print(k)
PY
echo

echo "== Log tail (pipeline) =="
grep -E '\[screener\]|\[decision\]|\[screener->executor\]|\[executor\]' /var/log/hedge-executor.out.log | tail -n 40 || true
