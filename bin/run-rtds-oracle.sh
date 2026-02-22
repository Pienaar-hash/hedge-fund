#!/bin/bash
# Run the RTDS Oracle Client standalone.
# Layer 1 of the Binary Sleeve data plane.
#
# Usage:
#   ./bin/run-rtds-oracle.sh          # foreground (Ctrl-C to stop)
#   ./bin/run-rtds-oracle.sh &        # background
#
# Logs:
#   Ticks:     logs/prediction/rtds_oracle.jsonl
#   Health:    logs/prediction/rtds_oracle_health.jsonl
#   Anomalies: logs/execution/environment_events.jsonl

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="${PWD}"
export PYTHONUNBUFFERED=1

exec ./venv/bin/python -m prediction.rtds_oracle_client
