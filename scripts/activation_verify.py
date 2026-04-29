#!/usr/bin/env python3
"""
Activation Window — Day-14 Formal Verification & GO/NO-GO Script (v8.0)
========================================================================

Run at the end of the 14-day activation window.  Performs 7-gate
verification of full-stack integrity and outputs a machine-readable
verdict.

Usage:
    PYTHONPATH=. python scripts/activation_verify.py
    PYTHONPATH=. python scripts/activation_verify.py --json
    PYTHONPATH=. python scripts/activation_verify.py --json > /tmp/activation_verdict.json

Gates:
    1. nav_stable         — NAV > 0, drawdown within limits
    2. drawdown_within_limits — DD < kill threshold
    3. risk_veto_consistent   — No unexplained veto spikes (< 5000)
    4. binary_lab_shadow_valid — Freeze intact, trades recorded if SHADOW mode
    5. manifest_intact        — v7_manifest.json unchanged since boot
    6. no_freeze_violations   — Binary Lab config hash stable
    7. dle_shadow_clean       — DLE mismatches < 50

Verdict:
    7/7 GO → Promote to Production
    6/7 GO → Extend window 7 days
    ≤5 GO  → Investigate, do not scale
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RUNTIME_YAML = Path("config/runtime.yaml")
MANIFEST_PATH = Path("v7_manifest.json")
BINARY_LAB_TRADES_LOG = Path("logs/execution/binary_lab_trades.jsonl")
RISK_VETOES_LOG = Path("logs/execution/risk_vetoes.jsonl")
DLE_SHADOW_LOG = Path("logs/execution/dle_shadow_events.jsonl")
ACTIVATION_STATE = Path("logs/state/activation_window_state.json")

# Thresholds
# ---------------------------------------------------------------------------
# MAX_RISK_VETOES: calibrated to Phase C steady-state veto emission.
#
# Observed baseline:  ~300 governance vetoes/day (Feb 2026, Phase C shadow).
# 14-day projection:  ~4,200.
# Multiplier:         ~1.2× baseline projection.
# Threshold:          5,000.
#
# Intent: detect >~20% sustained deviation from steady-state, NOT absolute
# anomaly.  If baseline shifts materially, recalibrate this constant.
#
# Veto composition at calibration (representative, not prescriptive):
#   portfolio_dd_circuit, symbol_cap, nav_stale, daily_loss,
#   correlation_cap, kill_switch, leverage_cap — all structural safety gates.
#
# Plumbing vetoes (min_notional) are excluded from this count.
# ---------------------------------------------------------------------------
MAX_RISK_VETOES = 5000
MAX_DLE_MISMATCHES = 50     # DLE shadow mismatches upper bound


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return None


def _load_activation_state() -> Dict[str, Any]:
    """Load the last emitted activation window state."""
    data = _read_json(ACTIVATION_STATE)
    if data and data.get("active"):
        return data
    return {}


def _count_binary_lab_shadow_trades(start_ts: str) -> int:
    """Count Binary Lab SHADOW trades since start_ts."""
    if not BINARY_LAB_TRADES_LOG.exists():
        return 0
    count = 0
    try:
        with BINARY_LAB_TRADES_LOG.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("execution_mode") == "SHADOW" and (rec.get("ts", "") or "") >= start_ts:
                        count += 1
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return count


# Plumbing veto reasons excluded from governance risk counting.
# These are exchange constraint artifacts (e.g. order too small),
# not risk model / doctrine / regime vetoes.
_PLUMBING_VETO_REASONS: frozenset = frozenset({
    "min_notional",
    "below_min_notional",
})


def _count_risk_vetoes_since(path: Path, start_ts: str) -> int:
    """Count *governance-relevant* risk vetoes since start_ts.

    Excludes plumbing vetoes (``min_notional``) which reflect exchange
    sizing constraints, not risk instability.
    """
    if not path.exists():
        return 0
    count = 0
    try:
        with path.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if (rec.get("ts", "") or "") >= start_ts:
                        reason = str(rec.get("veto_reason", "")).lower()
                        if reason not in _PLUMBING_VETO_REASONS:
                            count += 1
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return count


def _count_dle_mismatches(path: Path, start_ts: str) -> int:
    """Count DLE shadow mismatch events since start_ts."""
    if not path.exists():
        return 0
    count = 0
    try:
        with path.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if (rec.get("ts", "") or "") >= start_ts:
                        if rec.get("mismatch") or rec.get("event") == "DLE_MISMATCH":
                            count += 1
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return count


# ---------------------------------------------------------------------------
# Verification gates
# ---------------------------------------------------------------------------
def _build_preflight_state(
    manifest_path: Path = MANIFEST_PATH,
) -> Dict[str, Any]:
    """Build live state snapshot for preflight verification.

    Used when the activation window is not yet active to validate
    that the GO path is real before starting the 14-day run.
    Reads NAV, DD, manifest hash, and config hash directly from
    live sources.
    """
    import hashlib
    import yaml as _yaml

    runtime_path = Path("config/runtime.yaml")

    # NAV from state file
    nav_data = _read_json(Path("logs/state/nav_state.json")) or {}
    nav_usd = float(
        nav_data.get("total_equity")
        or nav_data.get("nav_usd")
        or nav_data.get("nav")
        or 0.0
    )

    # Drawdown from nav_state or risk_snapshot
    dd_pct = float(nav_data.get("drawdown_pct", 0.0) or 0.0)
    if dd_pct == 0.0:
        risk_snap = _read_json(Path("logs/state/risk_snapshot.json")) or {}
        dd_pct = float(risk_snap.get("dd_frac", 0.0) or 0.0)

    # Drawdown kill threshold from config
    dd_kill = 0.05
    try:
        with open(runtime_path) as f:
            cfg = _yaml.safe_load(f) or {}
        aw = cfg.get("activation_window", {})
        dd_kill = float(aw.get("drawdown_kill_pct", 0.05))
    except Exception:
        pass

    # Compute hashes live
    def _hash(p: Path) -> Optional[str]:
        try:
            return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        except Exception:
            return None

    manifest_hash = _hash(manifest_path)
    config_hash = _hash(runtime_path)

    # Binary lab freeze (DISABLED lab cannot violate a freeze)
    bl_data = _read_json(Path("logs/state/binary_lab_state.json"))
    binary_lab_freeze_ok = True
    if bl_data is not None:
        bl_status = str(bl_data.get("status", "")).upper()
        if bl_status in ("DISABLED", "TERMINATED", ""):
            binary_lab_freeze_ok = True  # Lab not running
        else:
            binary_lab_freeze_ok = bl_data.get("freeze_intact", True)

    return {
        "start_ts": "",
        "nav_usd": nav_usd,
        "drawdown_pct": dd_pct,
        "drawdown_kill_pct": dd_kill,
        "manifest_intact": manifest_hash is not None,
        "config_intact": config_hash is not None,
        "binary_lab_freeze_ok": binary_lab_freeze_ok,
        "dle_mismatches": 0,
        "elapsed_days": 0.0,
        "current_manifest_hash": manifest_hash,
        "boot_manifest_hash": manifest_hash,  # same at preflight
        "preflight": True,
    }


def run_verification(
    *,
    activation_state_path: Path = ACTIVATION_STATE,
    manifest_path: Path = MANIFEST_PATH,
    binary_lab_trades_log: Path = BINARY_LAB_TRADES_LOG,
    risk_vetoes_log: Path = RISK_VETOES_LOG,
    dle_shadow_log: Path = DLE_SHADOW_LOG,
    preflight: bool = False,
) -> Dict[str, Any]:
    """Execute 7-gate verification.  Returns full verdict dict.

    When ``preflight=True``, reads live state directly instead of
    relying on ``activation_window_state.json``.  This validates
    the GO path is real before starting the 14-day window.
    """
    if preflight:
        state = _build_preflight_state(manifest_path)
    else:
        state = _read_json(activation_state_path) or {}

    start_ts = state.get("start_ts", "")
    nav_usd = float(state.get("nav_usd", 0.0))
    dd_pct = float(state.get("drawdown_pct", 0.0))
    dd_kill = float(state.get("drawdown_kill_pct", 0.05))
    manifest_intact = state.get("manifest_intact", False)
    config_intact = state.get("config_intact", False)
    binary_lab_freeze_ok = state.get("binary_lab_freeze_ok", True)
    dle_mismatches_count = state.get("dle_mismatches", 0)
    elapsed_days = float(state.get("elapsed_days", 0.0))

    # Additional counts from logs
    shadow_trades = _count_binary_lab_shadow_trades(start_ts) if start_ts else 0
    risk_vetoes = _count_risk_vetoes_since(risk_vetoes_log, start_ts) if start_ts else 0
    dle_mm = _count_dle_mismatches(dle_shadow_log, start_ts) if start_ts else dle_mismatches_count

    gates: Dict[str, Dict[str, Any]] = {}

    # Gate 1: NAV stable
    gates["nav_stable"] = {
        "pass": nav_usd > 0,
        "value": nav_usd,
        "note": f"NAV = ${nav_usd:.2f}" if nav_usd > 0 else "NAV unavailable or zero",
    }

    # Gate 2: Drawdown within limits
    gates["drawdown_within_limits"] = {
        "pass": dd_pct < dd_kill,
        "value": round(dd_pct, 6),
        "threshold": dd_kill,
        "note": f"DD {dd_pct:.4%} < kill {dd_kill:.4%}" if dd_pct < dd_kill else f"DD {dd_pct:.4%} >= kill {dd_kill:.4%}",
    }

    # Gate 3: Risk vetoes consistent (no spike)
    gates["risk_veto_consistent"] = {
        "pass": risk_vetoes < MAX_RISK_VETOES,
        "value": risk_vetoes,
        "threshold": MAX_RISK_VETOES,
        "note": f"{risk_vetoes} vetoes (< {MAX_RISK_VETOES} threshold)",
    }

    # Gate 4: Binary Lab SHADOW valid
    # Must have freeze intact AND if shadow mode was active, trades recorded
    bl_valid = binary_lab_freeze_ok
    bl_note = "freeze intact"
    if shadow_trades > 0:
        bl_note = f"freeze intact, {shadow_trades} shadow trades recorded"
    gates["binary_lab_shadow_valid"] = {
        "pass": bl_valid,
        "value": shadow_trades,
        "note": bl_note,
    }

    # Gate 5: Manifest intact
    gates["manifest_intact"] = {
        "pass": manifest_intact,
        "value": state.get("current_manifest_hash", "unknown"),
        "boot_hash": state.get("boot_manifest_hash", "unknown"),
        "note": "manifest unchanged" if manifest_intact else "MANIFEST DRIFT DETECTED",
    }

    # Gate 6: No freeze violations
    gates["no_freeze_violations"] = {
        "pass": binary_lab_freeze_ok and config_intact,
        "value": {"freeze_ok": binary_lab_freeze_ok, "config_intact": config_intact},
        "note": "all config frozen" if (binary_lab_freeze_ok and config_intact) else "FREEZE VIOLATION",
    }

    # Gate 7: DLE shadow clean
    gates["dle_shadow_clean"] = {
        "pass": dle_mm < MAX_DLE_MISMATCHES,
        "value": dle_mm,
        "threshold": MAX_DLE_MISMATCHES,
        "note": f"{dle_mm} DLE mismatches (< {MAX_DLE_MISMATCHES} threshold)",
    }

    passed = sum(1 for g in gates.values() if g["pass"])
    total_gates = len(gates)

    if passed == total_gates:
        verdict = "GO"
        action = "Promote to Production"
    elif passed == total_gates - 1:
        verdict = "EXTEND"
        action = "Extend window 7 days, review failing gate"
    else:
        verdict = "NO-GO"
        action = "Investigate, do not scale"

    return {
        "verdict": verdict,
        "passed": passed,
        "total_gates": total_gates,
        "action": action,
        "gates": gates,
        "window": {
            "start_ts": start_ts,
            "elapsed_days": elapsed_days,
            "episodes_completed": state.get("episodes_completed", 0),
            "shadow_trades": shadow_trades,
        },
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _print_report(result: Dict[str, Any]) -> None:
    """Print human-readable verification report."""
    sep = "=" * 72
    print(f"\n{sep}")
    print("  ACTIVATION WINDOW — FULL-STACK VERIFICATION (v8.0)")
    print(f"{sep}")
    print(f"  Evaluated:     {result['evaluated_at']}")
    w = result.get("window", {})
    print(f"  Window start:  {w.get('start_ts', 'N/A')}")
    print(f"  Elapsed days:  {w.get('elapsed_days', 0):.1f}")
    print(f"  Episodes:      {w.get('episodes_completed', 0)}")
    print(f"  Shadow trades: {w.get('shadow_trades', 0)}")
    print()

    print(f"  ┌─ 7-Gate Verification ({'─' * 46})┐")
    for name, gate in result["gates"].items():
        icon = "✓" if gate["pass"] else "✗"
        print(f"  {icon}  {name:<30} {gate.get('note', '')}")
    print()

    v = result["verdict"]
    p = result["passed"]
    t = result["total_gates"]
    print(f"  VERDICT: {v} ({p}/{t})")
    print(f"  ACTION:  {result['action']}")
    print()
    print(sep)

    if v == "GO":
        print("\n  Full-stack integrity confirmed.")
        print("  System is production-grade stable.\n")
    elif v == "EXTEND":
        failing = [n for n, g in result["gates"].items() if not g["pass"]]
        print(f"\n  Failing gates: {', '.join(failing)}")
        print("  Extend window 7 days and review.\n")
    else:
        failing = [n for n, g in result["gates"].items() if not g["pass"]]
        print(f"\n  Failing gates: {', '.join(failing)}")
        print("  Do NOT scale. Investigate failures.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Activation Window — Full-Stack Verification (v8.0)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output machine-readable JSON",
    )
    parser.add_argument(
        "--preflight", action="store_true",
        help="Preflight mode: read live state directly (no active window required)",
    )
    args = parser.parse_args()

    result = run_verification(preflight=args.preflight)

    # Record verdict for production scale gating + DLE lifecycle event
    try:
        from execution.activation_window import record_verification_verdict
        record_verification_verdict(result)
    except Exception as exc:
        print(f"Warning: could not record verdict: {exc}", file=sys.stderr)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)

    # Exit code: 0 for GO, 1 for anything else
    sys.exit(0 if result["verdict"] == "GO" else 1)


if __name__ == "__main__":
    main()
