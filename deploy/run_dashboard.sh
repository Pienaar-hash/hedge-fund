#!/usr/bin/env bash
set -euo pipefail

DBG=/var/log/hedge/hedge-dashboard.debug.log
{
  echo "==== $(date -Is) starting ===="
  echo "pwd=$(pwd)"
  echo "whoami=$(whoami)"
  echo "python=$(/root/hedge-fund/venv/bin/python -V 2>&1)"
  echo "streamlit import test..."
  /root/hedge-fund/venv/bin/python - <<'PY'
import sys
print(" OK importing streamlit...")
import streamlit
print(" streamlit version:", streamlit.__version__)
sys.exit(0)
PY
  echo "env (redacted):"
  env | sed -E 's/(BINANCE_API_(KEY|SECRET)|TELEGRAM_TOKEN)=.*/\1=****/g' | sort
  echo "launching streamlit…"
} >>"$DBG" 2>&1

PORT=""
for candidate in 8501 8502 8503; do
  if /root/hedge-fund/venv/bin/python - <<'PY' "$candidate"; then
import socket
import sys
p = int(sys.argv[1])
s = socket.socket()
try:
    s.bind(("0.0.0.0", p))
    s.close()
    sys.exit(0)
except OSError:
    sys.exit(1)
PY
  then
    PORT="$candidate"
    break
  fi
done

if [[ -z "$PORT" ]]; then
  echo "[dashboard] No available port found in fallback list (8501-8503)" >>"$DBG"
  PORT=8501
fi

if [[ "$PORT" != "8501" ]]; then
  echo "[dashboard] Using fallback port: $PORT" >>"$DBG"
fi

export DASHBOARD_PORT="$PORT"

exec /root/hedge-fund/venv/bin/python -m streamlit run dashboard/app.py \
  --server.port="$PORT" --server.address=0.0.0.0 --server.headless true --logger.level=info
