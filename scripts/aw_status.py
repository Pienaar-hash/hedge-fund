#!/usr/bin/env python3
"""
Activation Window — Daily Quick Status (60-second check)
=========================================================

One-line ops command for the 14-day certification window.
Run daily to confirm structural integrity holds.

Usage:
    PYTHONPATH=. python scripts/aw_status.py
    make aw-status                # if Makefile alias added

Output is compact, color-coded, and terminal-friendly.
An exit code of 0 means all green; 1 means at least one alert.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# State paths
# ---------------------------------------------------------------------------
AW_STATE = Path("logs/state/activation_window_state.json")
NAV_STATE = Path("logs/state/nav_state.json")
VERDICT_PATH = Path("logs/state/activation_verification_verdict.json")

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

NO_COLOR = os.environ.get("NO_COLOR", "") != ""
if NO_COLOR:
    GREEN = RED = YELLOW = CYAN = DIM = BOLD = RESET = ""


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return None


def _ok(label: str) -> str:
    return f"  {GREEN}✓{RESET} {label}"


def _fail(label: str) -> str:
    return f"  {RED}✗{RESET} {label}"


def _warn(label: str) -> str:
    return f"  {YELLOW}⚠{RESET} {label}"


def _check(ok: bool, label: str) -> str:
    return _ok(label) if ok else _fail(label)


def main() -> None:
    alerts = 0

    # --- Header ---
    print(f"\n{BOLD}{'─' * 56}{RESET}")
    print(f"{BOLD}  ACTIVATION WINDOW — DAILY STATUS{RESET}")
    print(f"{BOLD}{'─' * 56}{RESET}")

    now = datetime.now(timezone.utc)
    print(f"  {DIM}checked: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}{RESET}")
    print()

    # --- Load state ---
    state = _read_json(AW_STATE)
    nav_data = _read_json(NAV_STATE)

    if state is None or not state.get("active"):
        # Window not active — show preflight summary
        print(f"  status:  {YELLOW}INACTIVE{RESET}  (window not started)")
        print()

        # Show NAV even if inactive
        nav = 0.0
        if nav_data:
            nav = float(nav_data.get("total_equity") or nav_data.get("nav_usd") or 0.0)
        if nav > 0:
            print(_ok(f"NAV online: ${nav:,.2f}"))
        else:
            print(_fail("NAV offline"))
            alerts += 1

        # Show verdict if exists
        verdict = _read_json(VERDICT_PATH)
        if verdict:
            v = verdict.get("verdict", "?")
            p = verdict.get("passed", 0)
            t = verdict.get("total_gates", 7)
            ts = verdict.get("recorded_at", "?")[:19]
            color = GREEN if v == "GO" else RED
            print(f"  {color}verdict: {v} ({p}/{t}){RESET}  {DIM}recorded {ts}{RESET}")
        else:
            print(f"  {DIM}no verdict recorded yet{RESET}")

        print(f"\n{BOLD}{'─' * 56}{RESET}\n")
        sys.exit(0)

    # --- Active window ---
    elapsed_days = float(state.get("elapsed_days", 0))
    remaining_days = float(state.get("remaining_days", 0))
    duration_days = int(state.get("duration_days", 14))
    halted = state.get("halted", False)
    halt_reason = state.get("halt_reason", "")
    window_expired = state.get("window_expired", False)

    # Status line
    if halted and window_expired:
        status_color = GREEN
        status_text = "COMPLETED"
    elif halted:
        status_color = RED
        status_text = "HALTED"
        alerts += 1
    else:
        status_color = CYAN
        status_text = "ACTIVE"

    print(f"  status:  {status_color}{BOLD}{status_text}{RESET}")
    print(f"  day:     {elapsed_days:.1f} / {duration_days}  ({remaining_days:.1f} remaining)")
    print()

    if halted and not window_expired:
        print(f"  {RED}{BOLD}HALT REASON: {halt_reason}{RESET}")
        print()

    # --- Integrity checks ---
    manifest_ok = state.get("manifest_intact", False)
    config_ok = state.get("config_intact", False)
    bl_freeze_ok = state.get("binary_lab_freeze_ok", True)

    if not manifest_ok:
        alerts += 1
    if not config_ok:
        alerts += 1
    if not bl_freeze_ok:
        alerts += 1

    print(_check(manifest_ok, f"manifest intact   {DIM}(hash: {state.get('current_manifest_hash', '?')[:8]}){RESET}"))
    print(_check(config_ok, f"config intact     {DIM}(hash: {state.get('current_config_hash', '?')[:8]}){RESET}"))
    print(_check(bl_freeze_ok, "binary lab freeze"))
    print()

    # --- Risk metrics ---
    dd_pct = float(state.get("drawdown_pct", 0)) * 100
    dd_kill = float(state.get("drawdown_kill_pct", 0.05)) * 100
    dd_ok = dd_pct < dd_kill
    if not dd_ok:
        alerts += 1

    nav_usd = float(state.get("nav_usd", 0))
    dle_mm = int(state.get("dle_mismatches", 0))
    vetoes = int(state.get("risk_veto_count", 0))
    episodes = int(state.get("episodes_completed", 0))

    dle_ok = dle_mm < 50
    veto_ok = vetoes < 500
    if not dle_ok:
        alerts += 1
    if not veto_ok:
        alerts += 1

    print(_check(dd_ok, f"drawdown: {dd_pct:.2f}% / {dd_kill:.1f}% kill"))
    print(_check(nav_usd > 0, f"NAV: ${nav_usd:,.2f}"))
    print(_check(dle_ok, f"DLE mismatches: {dle_mm}"))
    print(_check(veto_ok, f"risk vetoes: {vetoes}"))
    print(f"  {DIM}episodes: {episodes}{RESET}")
    print()

    # --- Scale gate ---
    verdict = _read_json(VERDICT_PATH)
    if verdict:
        v = verdict.get("verdict", "?")
        ts = verdict.get("recorded_at", "?")[:19]
        color = GREEN if v == "GO" else YELLOW
        print(f"  {DIM}last verdict:{RESET} {color}{v}{RESET}  {DIM}({ts}){RESET}")
    else:
        print(f"  {DIM}no verdict recorded{RESET}")

    # --- Summary ---
    print()
    if alerts == 0:
        print(f"  {GREEN}{BOLD}ALL GREEN{RESET} — no structural incidents")
    else:
        print(f"  {RED}{BOLD}{alerts} ALERT(S){RESET} — investigate before continuing")
    print(f"\n{BOLD}{'─' * 56}{RESET}\n")

    sys.exit(1 if alerts > 0 else 0)


if __name__ == "__main__":
    main()
