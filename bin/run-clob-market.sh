#!/usr/bin/env bash
# Run the CLOB Market Client (Layer 2 — Binary Sleeve data plane)
#
# Static mode (default):
#   bin/run-clob-market.sh
#
# Discovery mode (15m BTC Up/Down auto-rotation):
#   CLOB_DISCOVERY_MODE=1 bin/run-clob-market.sh
#   CLOB_DISCOVERY_MODE=1 CLOB_DISCOVERY_TIMEFRAME=5m bin/run-clob-market.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
exec python -m prediction.clob_market_client "$@"
