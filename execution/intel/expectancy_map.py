from __future__ import annotations

"""
Time-of-day expectancy and slippage analytics.

This module reads router/trade telemetry and buckets it by hour-of-day so we
can see when a symbol tends to trade well (positive expectancy, low slippage)
or poorly (negative expectancy, high slippage).

v5.10.0 adds a minimal, read-only skeleton:
- No behavior changes.
- API designed for later integration with dashboard and router tuning.
"""

from dataclasses import dataclass
import datetime as dt
from typing import Dict, Iterable, List, Optional

from execution.router_metrics import get_recent_router_events


@dataclass
class HourlyStats:
    """Aggregate stats for a single hour-of-day bucket."""

    count: int = 0
    pnl_sum: float = 0.0  # aggregate realized PnL (if available)
    notional_sum: float = 0.0  # aggregate notional traded
    slip_bps_sum: float = 0.0  # sum of slippage in bps
    slip_bps_count: int = 0  # count of events with slippage


def _to_hour(ts: float) -> Optional[int]:
    """Convert epoch seconds to 0–23 hour-of-day (UTC for now)."""
    try:
        ts_val = float(ts)
    except (TypeError, ValueError):
        return None
    if ts_val > 1e12:
        ts_val /= 1000.0
    try:
        return dt.datetime.utcfromtimestamp(ts_val).hour
    except (OverflowError, OSError, ValueError):
        return None


def _iter_events(symbol: Optional[str] = None) -> Iterable[dict]:
    """
    Thin wrapper around router_metrics.get_recent_router_events.

    For v5.10.0, we default to a 7d window and leave symbol=None as
    aggregate mode.
    """
    # NOTE: .get_recent_router_events is already used by router_effectiveness_7d
    return get_recent_router_events(symbol=symbol, window_days=7)


def _update_stats(stats: HourlyStats, event: dict) -> None:
    """
    Update a single HourlyStats bucket with one router/trade event.

    v5.10.0 skeleton:
    - Treat `pnl` and `notional` as optional fields.
    - Treat `slippage_bps` as optional.
    """
    stats.count += 1

    pnl = event.get("realized_pnl", 0.0) or 0.0
    notional = event.get("notional", 0.0) or 0.0
    slip = event.get("slippage_bps")

    try:
        stats.pnl_sum += float(pnl)
    except (TypeError, ValueError):
        pass
    try:
        stats.notional_sum += float(notional)
    except (TypeError, ValueError):
        pass

    if slip is not None:
        try:
            stats.slip_bps_sum += float(slip)
            stats.slip_bps_count += 1
        except (TypeError, ValueError):
            pass


def hourly_expectancy(symbol: Optional[str] = None) -> Dict[int, dict]:
    """
    Compute simple per-hour aggregates for the last N days.

    Returns:
        {hour: {"count": int,
                "exp_per_notional": float | None,
                "slip_bps_avg": float | None}}
    """
    buckets: Dict[int, HourlyStats] = {}

    for ev in _iter_events(symbol=symbol):
        hour = _to_hour(ev.get("ts"))
        if hour is None:
            continue

        stats = buckets.get(hour)
        if stats is None:
            stats = HourlyStats()
            buckets[hour] = stats

        _update_stats(stats, ev)

    result: Dict[int, dict] = {}
    for hour, s in buckets.items():
        exp_per_notional: Optional[float]
        if s.notional_sum > 0:
            exp_per_notional = s.pnl_sum / s.notional_sum
        else:
            exp_per_notional = None

        slip_avg: Optional[float]
        if s.slip_bps_count > 0:
            slip_avg = s.slip_bps_sum / s.slip_bps_count
        else:
            slip_avg = None

        result[hour] = {
            "count": s.count,
            "exp_per_notional": exp_per_notional,
            "slip_bps_avg": slip_avg,
        }

    return result


def best_hours(
    symbol: str,
    min_trades: int = 20,
    min_expectancy: float = 0.0,
) -> List[int]:
    """
    Return hours-of-day where expectancy looks favorable.

    A "best" hour is any bucket where:
      - trade count >= min_trades
      - exp_per_notional >= min_expectancy

    v5.10.0 keeps the policy intentionally simple; future versions can add
    slippage filters, volatility filters, etc.
    """
    stats = hourly_expectancy(symbol)
    good: List[int] = []
    for hour, row in stats.items():
        if row["count"] < min_trades:
            continue
        exp_val = row["exp_per_notional"]
        if exp_val is None:
            continue
        if exp_val >= min_expectancy:
            good.append(hour)
    return sorted(good)


__all__ = [
    "HourlyStats",
    "hourly_expectancy",
    "best_hours",
]
