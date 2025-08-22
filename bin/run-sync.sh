#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/hedge-fund
set -a; . /root/hedge-fund/.env; set +a
exec /root/hedge-fund/venv/bin/python -m execution.sync_state
