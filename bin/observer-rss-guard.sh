#!/usr/bin/env bash
# observer-rss-guard.sh — RSS watchdog for round_observer
#
# Restarts round_observer if its RSS exceeds the threshold.
# Designed to run via cron every 5 minutes as a safety fuse.
#
# This should never fire after the _tail_lines fix, but it
# prevents any future regression from taking down the box.

set -euo pipefail

RSS_LIMIT_KB=716800  # 700 MB in KB

PID=$(supervisorctl pid hedge:hedge-round_observer 2>/dev/null || echo "0")

if [[ "$PID" == "0" ]] || [[ ! -d "/proc/$PID" ]]; then
    exit 0  # not running, nothing to guard
fi

RSS_KB=$(awk '{print $2}' /proc/"$PID"/statm 2>/dev/null || echo "0")
# statm reports pages; convert to KB (4KB pages on x86_64)
RSS_KB=$(( RSS_KB * 4 ))

if (( RSS_KB > RSS_LIMIT_KB )); then
    logger -t observer-rss-guard "ALERT: round_observer PID=$PID RSS=${RSS_KB}KB exceeds ${RSS_LIMIT_KB}KB — restarting"
    supervisorctl restart hedge:hedge-round_observer
fi
