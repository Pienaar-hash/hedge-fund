#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec /bin/bash scripts/ml_retrain_cron.sh
