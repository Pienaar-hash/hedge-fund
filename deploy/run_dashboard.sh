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

exec /root/hedge-fund/venv/bin/python -m streamlit run dashboard/app.py \
  --server.port=8501 --server.address=0.0.0.0 --server.headless true --logger.level=info
