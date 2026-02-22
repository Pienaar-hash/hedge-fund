#!/usr/bin/env bash
# Run the CLOB Market Client (Layer 2 — Binary Sleeve data plane)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
exec python -m prediction.clob_market_client "$@"
