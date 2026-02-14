"""
NAV-Delta PnL — Compute real portfolio PnL from NAV history.

24h PnL = NAV(now) - NAV(24h ago), NOT closed-episode PnL.

Episode PnL only counts realised closed trades. It systematically
understates (or overstates) actual portfolio performance because it
ignores unrealised gains, mark-to-market on holdings, funding, etc.

Data source: logs/nav_log.json — array of {nav, t, unrealized_pnl}
where t is Unix epoch seconds.

--- Span Authority ---

Any windowed NAV metric must prove sufficient log span before being
displayed.  compute_nav_deltas() returns ``span_ok`` flags for each
window.  Consumers MUST check ``span_ok[window]`` — silent fallback
is a structural violation.

Use ``nav_window_valid(window)`` as the canonical single-point guard.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

# ── Span thresholds (days) ─────────────────────────────────────────
# A window is only valid if the NAV log spans at least this fraction
# of the window.  Keeps thresholds in ONE place.
_SPAN_THRESHOLDS: Dict[str, float] = {
    "24h": 0.9,       # need ~21.6 h of data for 24h window
    "7d": 6.0,        # need ~6 d
    "30d": 25.0,      # need ~25 d
    "all_time": 7.0,  # need >=7 d before all-time is meaningful
}

_NAV_LOG_PATH = Path("logs/nav_log.json")

# Module-level cache: (mtime, data)
_cache: Dict[str, Any] = {"mtime": 0.0, "entries": []}


def _load_nav_log() -> list:
    """Load nav_log.json with simple mtime-based caching."""
    try:
        mtime = _NAV_LOG_PATH.stat().st_mtime
    except FileNotFoundError:
        return []

    if mtime == _cache["mtime"] and _cache["entries"]:
        return _cache["entries"]

    try:
        with open(_NAV_LOG_PATH) as f:
            entries = json.load(f)
        if not isinstance(entries, list):
            return []
        _cache["mtime"] = mtime
        _cache["entries"] = entries
        return entries
    except Exception:
        return []


def _nav_at_offset(entries: list, current_t: float, seconds_ago: float) -> Optional[float]:
    """Find the NAV value closest to (current_t - seconds_ago).

    Walks the list once and picks the entry whose timestamp is
    closest to the target without going past it. Falls back to
    the earliest entry if the log doesn't go back far enough.
    """
    target = current_t - seconds_ago
    best: Optional[dict] = None
    best_dist = float("inf")

    for e in entries:
        t = e.get("t")
        if t is None:
            continue
        dist = abs(t - target)
        if dist < best_dist:
            best = e
            best_dist = dist

    if best is None:
        return None
    return float(best.get("nav", 0))


def compute_nav_deltas() -> Dict[str, Any]:
    """Return NAV-delta PnL for standard windows **with span authority**.

    Returns dict with keys:
        pnl_24h, pnl_7d, pnl_30d, pnl_all_time  — raw deltas (float)
        nav_current  — latest NAV
        log_span_days  — actual log span
        span_ok  — dict mapping window → bool (canonical authority)

    Consumers MUST check ``span_ok[window]`` before displaying the
    corresponding pnl value.  If ``span_ok`` is ``False``, the raw
    delta is still present for diagnostic purposes but MUST NOT be
    shown to operators as a valid metric.
    """
    _empty: Dict[str, Any] = {
        "pnl_24h": 0.0, "pnl_7d": 0.0, "pnl_30d": 0.0,
        "pnl_all_time": 0.0, "nav_current": 0.0,
        "log_span_days": 0.0,
        "span_ok": {"24h": False, "7d": False, "30d": False, "all_time": False},
    }
    entries = _load_nav_log()
    if not entries:
        return _empty

    # Current NAV = last entry
    latest = entries[-1]
    nav_now = float(latest.get("nav", 0))
    t_now = float(latest.get("t", time.time()))

    # Earliest entry for all-time
    earliest = entries[0]
    nav_first = float(earliest.get("nav", nav_now))
    t_first = float(earliest.get("t", t_now))
    log_span_days = (t_now - t_first) / 86400

    # Window deltas
    nav_24h = _nav_at_offset(entries, t_now, 86400)  # 24 hours
    nav_7d = _nav_at_offset(entries, t_now, 604800)   # 7 days
    nav_30d = _nav_at_offset(entries, t_now, 2592000)  # 30 days

    # Only report all-time PnL if nav_log spans >= 7 days.
    # Short logs (e.g. 12 hours after restart) would silently
    # overwrite the episode ledger's true historical PnL.
    pnl_all_time = round(nav_now - nav_first, 2) if log_span_days >= _SPAN_THRESHOLDS["all_time"] else 0.0

    # ── Span authority ──────────────────────────────────────────
    span_ok = {
        "24h": log_span_days >= _SPAN_THRESHOLDS["24h"],
        "7d": log_span_days >= _SPAN_THRESHOLDS["7d"],
        "30d": log_span_days >= _SPAN_THRESHOLDS["30d"],
        "all_time": log_span_days >= _SPAN_THRESHOLDS["all_time"],
    }

    return {
        "pnl_24h": round(nav_now - nav_24h, 2) if nav_24h is not None else 0.0,
        "pnl_7d": round(nav_now - nav_7d, 2) if nav_7d is not None else 0.0,
        "pnl_30d": round(nav_now - nav_30d, 2) if nav_30d is not None else 0.0,
        "pnl_all_time": pnl_all_time,
        "nav_current": nav_now,
        "log_span_days": round(log_span_days, 2),
        "span_ok": span_ok,
    }


def nav_window_valid(window: str) -> bool:
    """Single-point span guard for any dashboard panel.

    >>> nav_window_valid("24h")   # True only if log spans >= 0.9d
    >>> nav_window_valid("7d")    # True only if log spans >= 6d

    This prevents recurrence of the authority-ordering violation where
    different panels applied (or forgot) their own span logic.
    """
    deltas = compute_nav_deltas()
    span_ok = deltas.get("span_ok", {})
    return bool(span_ok.get(window, False))
