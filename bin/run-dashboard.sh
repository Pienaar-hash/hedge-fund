#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/hedge-fund
set -a; . /root/hedge-fund/.env; set +a

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
  echo "[dashboard] No available port found in fallback list (8501-8503)"
  PORT=8501
fi

if [[ "$PORT" != "8501" ]]; then
  echo "[dashboard] Using fallback port: $PORT"
fi

export DASHBOARD_PORT="$PORT"

exec /root/hedge-fund/venv/bin/streamlit run dashboard/main.py --server.headless=true --server.port="$PORT"
