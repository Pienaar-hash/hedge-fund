#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== ML Health =="
for f in models/registry.json models/signal_eval.json models/last_train_report.json; do
  if [[ -f "$f" ]]; then
    printf "[OK] %s (updated: %s)\n" "$f" "$(date -u -r "$f" +%FT%TZ)"
  else
    echo "[MISS] $f"
  fi
done

echo "---- last_train_report.json (tail) ----"
if [[ -f models/last_train_report.json ]]; then
  tail -n 50 models/last_train_report.json
fi
