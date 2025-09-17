#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

LOG1="deploy/supervisor-user/logs/hedge-executor.out.log"
LOG2="/var/log/hedge-executor.out.log"

log=""
if [[ -f "$LOG1" ]]; then log="$LOG1"; fi
if [[ -z "$log" && -f "$LOG2" ]]; then log="$LOG2"; fi
if [[ -z "$log" ]]; then
  echo "No known executor log file found."
  exit 2
fi

echo "Scanning: $log"
# Count successful order requests and show last few
grep -E 'ORDER_REQ.* 200' -n "$log" | tail -n 20 || true
COUNT=$(grep -E 'ORDER_REQ.* 200' -c "$log" || true)
echo "ORDER_REQ_200_COUNT=$COUNT"

# If zero, surface top veto reasons (if present)
if [[ "${COUNT:-0}" -eq 0 ]]; then
  echo "No ORDER_REQ 200 found. Top veto markers (last 200 lines):"
  tail -n 200 "$log" | grep -E 'veto|portfolio_cap|symbol_cap|tier_cap|cooldown|daily_loss|below_min_notional|trade_rate_limit|ob_adverse' || true
fi
