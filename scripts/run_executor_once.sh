#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
set -a
source ./.env
set +a
export PYTHONPATH="$(pwd)"
ONE_SHOT=1 ./venv/bin/python -m execution.executor_live
