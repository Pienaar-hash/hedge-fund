#!/usr/bin/env python3
"""
Binary Lab Gate — Daily Cron Evaluator
======================================

Runs the hybrid variance audit and gate evaluation, appending a
single-line ops log entry.

Intended to be called via cron once per day::

    0 6 * * * cd /root/hedge-fund && PYTHONPATH=. python scripts/binary_lab_gate_cron.py >> logs/audit/binary_lab_gate.log 2>&1

Output format (one line per run)::

    2026-02-19T06:00:00Z BINARY_LAB_GATE=GO  gates=9/9 window=24h records=4820 σH=0.0150 router=21.9% signal=78.1% conv=low:42,very_low:2
    2026-02-19T06:00:00Z BINARY_LAB_GATE=NO_GO gates=4/9 window=24h records=4820 σH=0.0083 router=55.1% signal=44.9% conv=low:200 failed=condition_a_router,condition_b_signal insufficient_data=condition_c_weight_consistency

Manifest impact: None (ops log only, not a state surface).
Doctrine impact: None.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from hybrid_variance_audit import (  # noqa: E402
    SCORE_DECOMP_PATH,
    load_records,
    parse_window,
    run_audit,
)
from evaluate_binary_lab_gate import evaluate  # noqa: E402

LOG_DIR = ROOT / "logs" / "audit"
LOG_FILE = LOG_DIR / "binary_lab_gate.log"


def main() -> None:
    window_str = os.environ.get("BINARY_LAB_WINDOW", "24h")
    window = parse_window(window_str)

    jsonl_path = Path(os.environ.get("SCORE_DECOMP_PATH", str(SCORE_DECOMP_PATH)))
    records = load_records(jsonl_path, window)

    result = run_audit(
        records,
        window_label=window_str,
        output_json=False,
        verbose=False,
    )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    n_records = result.get("n_records", 0)

    if not result.get("sufficient_data", False):
        line = f"{now} BINARY_LAB_GATE=NO_DATA window={window_str} records={n_records}"
        print(line)
        return

    sheet = result["decision_sheet"]
    verdict = evaluate(sheet)

    activation = verdict["binary_lab_activation"]
    gates_passed = verdict["gates_passed"]
    gates_total = verdict["gates_evaluated"]

    # Extract ops memo fields from decision sheet
    contrib = sheet.get("contribution", {})
    router_pct = contrib.get("router", 0.0) * 100
    signal_pct = contrib.get("signal_share", 0.0) * 100
    sigma_h = sheet.get("stddev", {}).get("hybrid", 0.0)
    conv_dist = sheet.get("conviction", {}).get("distribution", {})
    conv_summary = ",".join(f"{k}:{v*100:.1f}%" for k, v in sorted(conv_dist.items()) if v > 0)

    line = (
        f"{now} BINARY_LAB_GATE={activation}"
        f" gates={gates_passed}/{gates_total}"
        f" window={window_str}"
        f" records={n_records}"
        f" σH={sigma_h:.4f}"
        f" router={router_pct:.1f}%"
        f" signal={signal_pct:.1f}%"
        f" conv={conv_summary or 'none'}"
    )

    if verdict["failed_conditions"]:
        line += f" failed={','.join(verdict['failed_conditions'])}"

    if verdict.get("insufficient_data"):
        line += f" insufficient_data={','.join(verdict['insufficient_data'])}"

    # Ensure log directory exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Append to ops log
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

    # Also print to stdout (for cron capture)
    print(line)


if __name__ == "__main__":
    main()
