#!/usr/bin/env bash
set -Eeuo pipefail

# Always run from repo root
cd /root/hedge-fund

# Export repo into PYTHONPATH explicitly
export PYTHONPATH=/root/hedge-fund:${PYTHONPATH:-}

# Load .env (exported)
set -a
. /root/hedge-fund/.env
set +a

# Exec sync_state using venv python
exec /root/hedge-fund/venv/bin/python -m execution.sync_state
