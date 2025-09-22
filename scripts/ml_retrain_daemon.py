#!/usr/bin/env python3
"""
Nightly retrain + evaluation daemon (no sudo). Sleeps until next 02:10 UTC,
then runs:
  - scripts/ml_fit.py
  - scripts/signal_eval.py
Also writes models/last_train_report.json (summary).
Exit only on fatal error; otherwise loop forever.
"""
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"
MODELS.mkdir(exist_ok=True)


def utc_now() -> dt.datetime:
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)


def seconds_until(hour: int = 2, minute: int = 10, second: int = 0) -> float:
    now = utc_now()
    target = now.replace(hour=hour, minute=minute, second=second, microsecond=0)
    if target <= now:
        target += dt.timedelta(days=1)
    return (target - now).total_seconds()


def run_cmd(cmd, env=None, timeout=3600):
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env or os.environ.copy(),
    )
    return proc.returncode, proc.stdout, proc.stderr


def main() -> None:
    os.chdir(REPO)
    while True:
        wait_seconds = int(seconds_until())
        print(f"[ml-retrain] sleeping {wait_seconds}s until next 02:10 UTC", flush=True)
        try:
            time.sleep(wait_seconds)
        except KeyboardInterrupt:
            print("[ml-retrain] interrupted; exiting", flush=True)
            return

        iteration = {
            "started_at_utc": utc_now().isoformat(),
        }
        t0 = time.time()

        rc_fit, out_fit, err_fit = run_cmd([sys.executable, "scripts/ml_fit.py"], timeout=5400)
        iteration["fit_rc"] = rc_fit
        try:
            iteration["fit_result"] = json.loads(out_fit)
        except Exception:
            iteration["fit_stdout_tail"] = out_fit[-2000:]
        if err_fit:
            iteration["fit_stderr_tail"] = err_fit[-2000:]

        rc_eval, out_eval, err_eval = run_cmd([sys.executable, "scripts/signal_eval.py"], timeout=1800)
        iteration["eval_rc"] = rc_eval
        try:
            iteration["eval_result"] = json.loads(out_eval)
        except Exception:
            iteration["eval_stdout_tail"] = out_eval[-2000:]
        if err_eval:
            iteration["eval_stderr_tail"] = err_eval[-2000:]

        iteration["finished_at_utc"] = utc_now().isoformat()
        iteration["elapsed_sec"] = round(time.time() - t0, 2)

        with open(MODELS / "last_train_report.json", "w", encoding="utf-8") as handle:
            json.dump(iteration, handle, indent=2)

        print("[ml-retrain] cycle complete", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"[ml-retrain] fatal error: {exc}", file=sys.stderr)
        raise
