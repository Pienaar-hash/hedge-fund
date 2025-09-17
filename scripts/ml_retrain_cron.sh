#!/usr/bin/env bash
# Nightly retrain + evaluation for cron (preferred path, no supervisor needed).
# Example crontab entry (UTC):
#   10 2 * * * cd /path/to/hedge-fund && /bin/bash scripts/ml_retrain_cron.sh >> models/cron.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."
set -a
source ./.env 2>/dev/null || true
set +a
export PYTHONPATH="$(pwd)"
mkdir -p models

ts() { date -u +%FT%TZ; }

START_TS="$(ts)"
FIT_RC=0
EVAL_RC=0

FIT_OUT="$('./venv/bin/python' scripts/ml_fit.py 2>&1)" || FIT_RC=$?
EVAL_OUT="$('./venv/bin/python' scripts/signal_eval.py 2>&1)" || EVAL_RC=$?

python3 - "$START_TS" "$FIT_RC" "$EVAL_RC" "$FIT_OUT" "$EVAL_OUT" > models/last_train_report.json <<'PY'
import json, sys, datetime
start_ts, fit_rc, eval_rc, fit_out, eval_out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
def parse_raw(raw: str):
    try:
        return json.loads(raw)
    except Exception:
        return {"stdout_tail": raw[-2000:]}
report = {
    "started_at_utc": start_ts,
    "finished_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "fit_rc": fit_rc,
    "fit_result": parse_raw(fit_out),
    "eval_rc": eval_rc,
    "eval_result": parse_raw(eval_out),
}
print(json.dumps(report, indent=2))
PY

echo "[cron] retrain+eval done at $(ts) (fit_rc=$FIT_RC, eval_rc=$EVAL_RC)"
