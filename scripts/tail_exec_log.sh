#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f "deploy/supervisor-user/logs/hedge-executor.out.log" ]]; then
  tail -n 200 -f deploy/supervisor-user/logs/hedge-executor.out.log
elif [[ -f "/var/log/hedge-executor.out.log" ]]; then
  tail -n 200 -f /var/log/hedge-executor.out.log
else
  echo "No known executor log file found."
  exit 1
fi
