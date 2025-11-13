#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MAX_BYTES="${MAX_ROTATE_BYTES:-5000000}"
KEEP_ARCHIVES="${MAX_ROTATE_ARCHIVES:-10}"

python scripts/ops_cleanup.py --rotate --max-bytes "$MAX_BYTES" --keep "$KEEP_ARCHIVES"
