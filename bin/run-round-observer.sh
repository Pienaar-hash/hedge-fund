#!/usr/bin/env bash
# Run the Round Observer (Layer 3 — Binary Sleeve data plane)
#
# Observes 15-minute BTC Up/Down rounds, captures oracle + CLOB snapshots,
# and logs completed round records.
#
# Prerequisites:
#   - RTDS Oracle Client (Layer 1) running:  bin/run-rtds-oracle.sh
#   - CLOB Market Client (Layer 2) running:  CLOB_DISCOVERY_MODE=1 bin/run-clob-market.sh
#
# Usage:
#   bin/run-round-observer.sh                    # observe indefinitely
#   OBSERVER_MAX_ROUNDS=10 bin/run-round-observer.sh  # observe 10 rounds then exit
#   OBSERVER_TIMEFRAME=5m bin/run-round-observer.sh   # observe 5m rounds
#
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
exec python -m prediction.round_observer "$@"
