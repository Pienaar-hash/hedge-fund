# execution/enforcement_rehearsal.py
"""
B.5 — DLE Enforcement Rehearsal (Shadow-Only)

Measures what would happen if we required a valid PERMIT for every executed order.
NO behavior change. Pure instrumentation. Pure metrics.

Counterfactual rule:
  An order is valid only if:
    - It has a matching PERMIT
    - PERMIT.state == ISSUED
    - ts <= expires_ts  (not expired)
    - permit.symbol matches order.symbol
    - permit.direction matches order.direction

This module:
  1. Builds a permit index from the DLE shadow log
  2. Evaluates each order against that index
  3. Logs rehearsal outcomes (append-only)
  4. Tracks runtime metrics (reset on restart)

CRITICAL SAFEGUARDS:
  - Never throws (fail-open everywhere)
  - Never prevents order send
  - If shadow index unavailable → log REHEARSAL_DISABLED
  - If parsing fails → fail open
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dateutil import parser as dateparser

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REHEARSAL_LOG_PATH = Path("logs/execution/dle_enforcement_rehearsal.jsonl")
DLE_SHADOW_LOG_PATH = Path("logs/execution/dle_shadow_events.jsonl")

# Rehearsal denial reasons (canonical)
REASON_OK = "OK"
REASON_NO_PERMIT = "NO_PERMIT"
REASON_EXPIRED = "EXPIRED"
REASON_MISMATCH_SYMBOL = "MISMATCH_SYMBOL"
REASON_MISMATCH_DIRECTION = "MISMATCH_DIRECTION"
REASON_DISABLED = "REHEARSAL_DISABLED"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class _PermitRecord:
    """Parsed PERMIT event for the enforcement index."""
    permit_id: str
    ts_unix: float
    ts_iso: str
    expires_ts_unix: Optional[float]
    expires_ts_iso: str
    symbol: str
    direction: str  # BUY | SELL | LONG | SHORT
    state: str      # ISSUED
    decision_id: str
    request_id: str


@dataclass
class RehearsalResult:
    """Outcome of a single enforcement rehearsal evaluation."""
    ts: str
    order_id: str
    symbol: str
    direction: str
    matched_permit_id: Optional[str]
    permit_valid: bool
    reason: str  # OK | NO_PERMIT | EXPIRED | MISMATCH_SYMBOL | MISMATCH_DIRECTION
    would_block: bool
    phase_id: str
    engine_version: str
    git_sha: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RehearsalMetrics:
    """Runtime metrics — reset on restart."""
    total_orders: int = 0
    would_block_count: int = 0
    would_block_pct: float = 0.0
    expired_permit_count: int = 0
    missing_permit_count: int = 0
    mismatch_count: int = 0
    ok_count: int = 0
    last_evaluation_ts: str = ""
    enabled: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def _update_pct(self) -> None:
        if self.total_orders > 0:
            self.would_block_pct = round(
                self.would_block_count / self.total_orders * 100, 2
            )
        else:
            self.would_block_pct = 0.0


# ---------------------------------------------------------------------------
# Permit index (built from shadow log)
# ---------------------------------------------------------------------------

def _iso_to_unix(ts_iso: str) -> Optional[float]:
    """Convert ISO timestamp to Unix seconds. Returns None on failure."""
    try:
        dt = dateparser.parse(ts_iso)
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _normalize_direction(raw: str) -> str:
    """Normalize direction to comparable form (uppercase)."""
    return str(raw).upper().strip()


def build_permit_index(
    shadow_log_path: Optional[Path] = None,
) -> Dict[str, List[_PermitRecord]]:
    """
    Build an index of PERMIT events from the DLE shadow log.

    Returns: {symbol: [_PermitRecord sorted by ts_unix descending (newest first)]}

    Fail-open: returns empty dict on any error.
    """
    log_path = shadow_log_path or DLE_SHADOW_LOG_PATH
    index: Dict[str, List[_PermitRecord]] = {}

    if not log_path.exists():
        logger.info("B.5: No shadow log at %s — permit index empty", log_path)
        return index

    try:
        with open(log_path, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    evt = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                if evt.get("event_type") != "PERMIT":
                    continue

                payload = evt.get("payload", {})
                action = payload.get("action", {})
                if not isinstance(action, dict):
                    continue  # v1 PERMIT without action — skip

                ts_iso = evt.get("ts", "")
                ts_unix = _iso_to_unix(ts_iso)
                if ts_unix is None:
                    continue

                expires_iso = payload.get("expires_ts", "")
                expires_unix = _iso_to_unix(expires_iso) if expires_iso else None

                symbol = str(action.get("symbol", "")).upper()
                direction = _normalize_direction(action.get("direction", ""))
                state = payload.get("state", "")

                if not symbol or not direction:
                    continue

                rec = _PermitRecord(
                    permit_id=payload.get("permit_id", ""),
                    ts_unix=ts_unix,
                    ts_iso=ts_iso,
                    expires_ts_unix=expires_unix,
                    expires_ts_iso=expires_iso,
                    symbol=symbol,
                    direction=direction,
                    state=state,
                    decision_id=payload.get("decision_id", ""),
                    request_id=payload.get("request_id", ""),
                )
                index.setdefault(symbol, []).append(rec)
    except Exception as e:
        logger.warning("B.5: Failed to load shadow log for permit index: %s", e)
        return {}

    # Sort each bucket newest-first for efficient lookup
    for bucket in index.values():
        bucket.sort(key=lambda r: r.ts_unix, reverse=True)

    logger.info("B.5: Permit index built — %d symbols, %d total permits",
                len(index), sum(len(v) for v in index.values()))
    return index


# ---------------------------------------------------------------------------
# Enforcement evaluation
# ---------------------------------------------------------------------------

def evaluate_order(
    *,
    symbol: str,
    direction: str,
    order_ts_unix: float,
    permit_index: Dict[str, List[_PermitRecord]],
) -> Tuple[bool, str, Optional[str]]:
    """
    Evaluate whether an order would be blocked under enforcement.

    Returns: (would_block, reason, matched_permit_id)
    """
    symbol = symbol.upper()
    direction = _normalize_direction(direction)
    candidates = permit_index.get(symbol)

    if not candidates:
        return True, REASON_NO_PERMIT, None

    # Find the best matching permit:
    # Walk newest-first, find one that covers this order's timestamp
    for permit in candidates:
        # Symbol already matched by index key
        # Check direction
        if permit.direction != direction:
            continue  # keep looking for a direction match

        # Check state
        if permit.state != "ISSUED":
            continue

        # Check timing: order_ts must be >= permit.ts_unix
        if order_ts_unix < permit.ts_unix:
            continue  # permit issued after order — not valid

        # Check expiry
        if permit.expires_ts_unix is not None and order_ts_unix > permit.expires_ts_unix:
            # This permit existed but expired — record it specifically
            return True, REASON_EXPIRED, permit.permit_id

        # Valid permit found
        return False, REASON_OK, permit.permit_id

    # Check if we had permits for this symbol but all mismatched on direction
    any_symbol_match = bool(candidates)
    if any_symbol_match:
        # Had permits for symbol, but none matched direction
        return True, REASON_MISMATCH_DIRECTION, None

    return True, REASON_NO_PERMIT, None


# ---------------------------------------------------------------------------
# Rehearsal writer (append-only, fail-open)
# ---------------------------------------------------------------------------

class RehearsalWriter:
    """Append-only JSONL writer for enforcement rehearsal outcomes."""

    def __init__(self, log_path: Optional[str] = None) -> None:
        self.log_path = log_path or str(REHEARSAL_LOG_PATH)
        self._write_failures = 0
        self._lock = threading.Lock()

    def write(self, result: RehearsalResult) -> None:
        """Append result to log. Fail-open: never raises."""
        try:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            line = json.dumps(result.to_dict(), sort_keys=True, separators=(",", ":"))
            with self._lock:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception as e:
            self._write_failures += 1
            print(
                f"[B.5_REHEARSAL] Warning: log write failed ({self._write_failures}): {e}",
                file=sys.stderr,
            )

    @property
    def write_failure_count(self) -> int:
        return self._write_failures


# ---------------------------------------------------------------------------
# Global state (singleton, thread-safe)
# ---------------------------------------------------------------------------

_rehearsal_writer: Optional[RehearsalWriter] = None
_permit_index: Dict[str, List[_PermitRecord]] = {}
_metrics = RehearsalMetrics()
_metrics_lock = threading.Lock()
_index_loaded = False
_enabled = False


def init_rehearsal(
    *,
    shadow_log_path: Optional[Path] = None,
    rehearsal_log_path: Optional[str] = None,
    force: bool = False,
) -> bool:
    """
    Initialize enforcement rehearsal. Called once at startup.

    Returns True if rehearsal is enabled and ready.
    Fail-open: returns False on any error.
    """
    global _rehearsal_writer, _permit_index, _metrics, _index_loaded, _enabled

    try:
        # Check feature flag
        from execution.v6_flags import get_flags
        flags = get_flags()
        if not flags.shadow_dle_enabled:
            logger.info("B.5: Rehearsal disabled — SHADOW_DLE_ENABLED=0")
            _enabled = False
            return False
    except Exception:
        logger.info("B.5: Rehearsal disabled — cannot read flags")
        _enabled = False
        return False

    try:
        _permit_index = build_permit_index(shadow_log_path)
        _index_loaded = True
        _rehearsal_writer = RehearsalWriter(rehearsal_log_path)
        _metrics = RehearsalMetrics(enabled=True)
        _enabled = True
        logger.info("B.5: Enforcement rehearsal initialized — %d symbols in permit index",
                     len(_permit_index))
        return True
    except Exception as e:
        logger.warning("B.5: Rehearsal init failed (shadow-safe): %s", e)
        _enabled = False
        return False


def refresh_permit_index(shadow_log_path: Optional[Path] = None) -> int:
    """
    Reload the permit index from the shadow log.

    Returns the number of permits loaded.
    Called periodically (e.g., once per cycle) to pick up new permits.
    """
    global _permit_index, _index_loaded
    try:
        _permit_index = build_permit_index(shadow_log_path)
        _index_loaded = True
        return sum(len(v) for v in _permit_index.values())
    except Exception as e:
        logger.warning("B.5: Permit index refresh failed: %s", e)
        return 0


def rehearse_order(
    *,
    symbol: str,
    direction: str,
    order_id: str = "",
    phase_id: str = "",
    engine_version: str = "",
    git_sha: str = "",
    order_ts: Optional[float] = None,
) -> Optional[RehearsalResult]:
    """
    Run enforcement rehearsal for a single order.

    This is the main entry point called from order_router.
    Fail-open: returns None on any error (order proceeds regardless).
    """
    global _metrics

    if not _enabled:
        return None

    try:
        ts_unix = order_ts or time.time()
        ts_iso = datetime.fromtimestamp(ts_unix, tz=timezone.utc).isoformat()

        would_block, reason, matched_permit_id = evaluate_order(
            symbol=symbol,
            direction=direction,
            order_ts_unix=ts_unix,
            permit_index=_permit_index,
        )

        result = RehearsalResult(
            ts=ts_iso,
            order_id=order_id,
            symbol=symbol.upper(),
            direction=_normalize_direction(direction),
            matched_permit_id=matched_permit_id,
            permit_valid=not would_block,
            reason=reason,
            would_block=would_block,
            phase_id=phase_id,
            engine_version=engine_version,
            git_sha=git_sha,
        )

        # Update metrics (thread-safe)
        with _metrics_lock:
            _metrics.total_orders += 1
            if would_block:
                _metrics.would_block_count += 1
                if reason == REASON_EXPIRED:
                    _metrics.expired_permit_count += 1
                elif reason == REASON_NO_PERMIT:
                    _metrics.missing_permit_count += 1
                elif reason in (REASON_MISMATCH_SYMBOL, REASON_MISMATCH_DIRECTION):
                    _metrics.mismatch_count += 1
            else:
                _metrics.ok_count += 1
            _metrics.last_evaluation_ts = ts_iso
            _metrics._update_pct()

        # Write to log (fail-open)
        if _rehearsal_writer is not None:
            _rehearsal_writer.write(result)

        return result

    except Exception as e:
        logger.debug("B.5: Rehearsal evaluation failed (shadow-safe): %s", e)
        return None


def get_rehearsal_metrics() -> Dict[str, Any]:
    """Return current rehearsal metrics snapshot (for state surface)."""
    with _metrics_lock:
        return _metrics.to_dict()


# ---------------------------------------------------------------------------
# Phase C readiness surface
# ---------------------------------------------------------------------------

_READINESS_STATE_PATH = Path("logs/state/phase_c_readiness.json")
_WINDOW_DAYS_REQUIRED = 14
_WOULD_BLOCK_PCT_THRESHOLD = 0.1  # 0.1%
_EXPIRED_THRESHOLD = 0
_MISSING_THRESHOLD = 0

# Breach tracking (in-memory, reset on restart — actual persistence is in state file)
_last_breach_ts: Optional[str] = None
_last_breach_reason: Optional[str] = None
_window_start_ts: Optional[str] = None


def compute_phase_c_readiness() -> Dict[str, Any]:
    """
    Build the Phase C readiness payload from current rehearsal metrics.

    This is called from the executor's state-publish cycle and written to
    logs/state/phase_c_readiness.json.

    Readiness is True when ALL criteria are met:
      - would_block_pct < 0.1%
      - expired_permit_count == 0
      - missing_permit_count == 0
      - 14 consecutive days sustained (tracked via window_start)

    Fail-open: returns a safe payload on any error.
    """
    global _last_breach_ts, _last_breach_reason, _window_start_ts

    now_iso = datetime.now(timezone.utc).isoformat()
    metrics_snap = get_rehearsal_metrics()

    # Evaluate gate criteria
    would_block_pct = metrics_snap.get("would_block_pct", 0.0)
    expired = metrics_snap.get("expired_permit_count", 0)
    missing = metrics_snap.get("missing_permit_count", 0)
    total = metrics_snap.get("total_orders", 0)
    rehearsal_enabled = metrics_snap.get("enabled", False)

    breaches: list[str] = []
    if would_block_pct >= _WOULD_BLOCK_PCT_THRESHOLD:
        breaches.append(f"would_block_pct={would_block_pct:.2f}% >= {_WOULD_BLOCK_PCT_THRESHOLD}%")
    if expired > _EXPIRED_THRESHOLD:
        breaches.append(f"expired_permit_count={expired} > {_EXPIRED_THRESHOLD}")
    if missing > _MISSING_THRESHOLD:
        breaches.append(f"missing_permit_count={missing} > {_MISSING_THRESHOLD}")
    if not rehearsal_enabled:
        breaches.append("rehearsal_disabled")
    if total == 0:
        breaches.append("no_orders_evaluated")

    criteria_met_now = len(breaches) == 0

    if not criteria_met_now:
        _last_breach_ts = now_iso
        _last_breach_reason = "; ".join(breaches)
        _window_start_ts = None  # reset window
    elif _window_start_ts is None:
        # First clean tick — start the window
        _window_start_ts = now_iso

    # Calculate window days
    window_days_met = 0
    if _window_start_ts is not None:
        try:
            from dateutil import parser as dp
            start = dp.parse(_window_start_ts)
            now = dp.parse(now_iso)
            window_days_met = max(0, (now - start).days)
        except Exception:
            window_days_met = 0

    gate_satisfied = criteria_met_now and window_days_met >= _WINDOW_DAYS_REQUIRED

    return {
        "window_days_required": _WINDOW_DAYS_REQUIRED,
        "window_days_met": window_days_met,
        "window_start_ts": _window_start_ts,
        "criteria_met": criteria_met_now,
        "gate_satisfied": gate_satisfied,
        "last_breach_ts": _last_breach_ts,
        "breach_reason": _last_breach_reason,
        "current_metrics": {
            "would_block_pct": would_block_pct,
            "expired_permit_count": expired,
            "missing_permit_count": missing,
            "total_orders": total,
            "ok_count": metrics_snap.get("ok_count", 0),
        },
        "thresholds": {
            "would_block_pct": _WOULD_BLOCK_PCT_THRESHOLD,
            "expired_permit_count": _EXPIRED_THRESHOLD,
            "missing_permit_count": _MISSING_THRESHOLD,
        },
        "rehearsal_enabled": rehearsal_enabled,
        "updated_ts": now_iso,
    }


def reset_rehearsal() -> None:
    """Reset global state (for testing only)."""
    global _rehearsal_writer, _permit_index, _metrics, _index_loaded, _enabled
    global _last_breach_ts, _last_breach_reason, _window_start_ts
    _rehearsal_writer = None
    _permit_index = {}
    _metrics = RehearsalMetrics()
    _index_loaded = False
    _enabled = False
    _last_breach_ts = None
    _last_breach_reason = None
    _window_start_ts = None
