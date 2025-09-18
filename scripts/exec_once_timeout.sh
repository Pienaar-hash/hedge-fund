#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Hard time limit for a single ONE_SHOT run (default 45s)
MAX_SEC="${EXECUTOR_MAX_SEC:-45}"

set -a
source ./.env
set +a
export PYTHONPATH="$(pwd)"
export FIRESTORE_ENABLED=0
export ONE_SHOT=1
export PYTHONUNBUFFERED=1

# Use coreutils timeout to avoid hangs; if unavailable, fallback to python -c alarm
if command -v timeout >/dev/null 2>&1; then
  timeout -s SIGINT "${MAX_SEC}s" ./venv/bin/python -m execution.executor_live
else
  python3 - <<'PY'
import os, signal, sys, time, threading, subprocess
max_sec = int(os.environ.get("EXECUTOR_MAX_SEC","45"))
def killer(p):
    time.sleep(max_sec)
    try:
        p.send_signal(signal.SIGINT)
    except Exception:
        pass
cmd = [os.path.join("venv","bin","python"), "-m", "execution.executor_live"]
p = subprocess.Popen(cmd, env=os.environ.copy())
threading.Thread(target=killer, args=(p,), daemon=True).start()
p.wait()
sys.exit(p.returncode)
PY
fi
