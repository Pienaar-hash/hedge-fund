#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

LOG1="deploy/supervisor-user/logs/hedge-executor.out.log"
LOG2="/var/log/hedge-executor.out.log"
LOG3=$(ls -1t logs/executor_live.*.log 2>/dev/null | head -n1 || true)
log=""
for candidate in "$LOG3" "$LOG1" "$LOG2"; do
  if [[ -f "$candidate" ]]; then
    log="$candidate"
    break
  fi
done

if [[ -z "$log" ]]; then
  echo "No log found"
  exit 1
fi

echo "Tailing $log"
tail -n 200 -f "$log" | grep -E --line-buffered 'ORDER_REQ|veto|ml_p='
