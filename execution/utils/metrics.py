"""
Metrics helpers for execution hardening.

These functions are thin wrappers around canonical telemetry sources
such as fills logs, PnL trackers, and drawdown trackers. Swap imports
to your concrete modules as needed.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from execution import drawdown_tracker
from execution import fills
# Optional pnl_tracker helpers may be absent in stripped-down runtimes.
try:
    from execution import pnl_tracker  # type: ignore
except Exception:  # pragma: no cover - fallback for tests/CLI
    pnl_tracker = None  # type: ignore
from execution import router_metrics


ALLOWED_SUFFIXES = ("USDC", "USDT")


def is_in_asset_universe(symbol: str) -> bool:
    """
    Limit trading to USDC-quoted pairs.
    """
    if not symbol:
        return False
    sym = symbol.upper()
    return any(sym.endswith(suffix) for suffix in ALLOWED_SUFFIXES)


# ---------- Notional helpers ----------

def _sum_notional(events: List[Dict[str, Any]]) -> float:
    return float(sum(abs(float(e.get("notional", 0.0))) for e in events))


def notional_7d_by_symbol(symbol: str) -> float:
    """Total absolute notional traded for `symbol` over the last 7 days."""
    events = fills.get_recent_fills(symbol=symbol, window_days=7)
    return _sum_notional(events)


def total_notional_7d() -> float:
    """Total absolute notional across all symbols over the last 7 days."""
    events = fills.get_recent_fills(symbol=None, window_days=7)
    return _sum_notional(events)


# ---------- Drawdown, Sharpe & PnL ----------

def dd_today_pct(symbol: str) -> float:
    """Per-symbol drawdown for the current day, as a percentage of equity."""
    return float(drawdown_tracker.get_symbol_dd_today_pct(symbol))


def rolling_sharpe_7d(symbol: str) -> float:
    """Rolling Sharpe over ~7d for the given symbol."""
    if pnl_tracker is None:
        return 0.0
    stats = pnl_tracker.get_symbol_stats(symbol, window_days=7)
    mean = float(stats.get("mean", 0.0))
    std = float(stats.get("std", 0.0))
    if std <= 0:
        return 0.0
    return mean / std


# ---------- Fee, PnL, slippage, fill/submitted ----------

def fill_notional_7d(symbol: Optional[str] = None) -> float:
    events = fills.get_recent_fills(symbol=symbol, window_days=7)
    return _sum_notional(events)


def submitted_notional_7d(symbol: Optional[str] = None) -> float:
    """
    Returns total submitted notional over ~7d. Falls back to fill notional
    if order submissions are not logged separately.
    """
    events = fills.get_recent_orders(symbol=symbol, window_days=7)
    if not events:
        return fill_notional_7d(symbol)
    return _sum_notional(events)


def gross_realized_7d(symbol: Optional[str] = None) -> float:
    """Sum of realized PnL over ~7d (before fees)."""
    if pnl_tracker is None:
        return 0.0
    return float(pnl_tracker.get_gross_realized(symbol=symbol, window_days=7))


def fees_7d(symbol: Optional[str] = None) -> float:
    """Sum of trading fees over ~7d."""
    if pnl_tracker is None:
        return 0.0
    return float(pnl_tracker.get_fees(symbol=symbol, window_days=7))


LOG_DIR = Path(os.getenv("EXEC_LOG_DIR") or "logs/execution")
ORDER_METRICS_PATH = Path(os.getenv("ORDER_METRICS_PATH") or (LOG_DIR / "order_metrics.jsonl"))
_READ_LIMIT = int(os.getenv("EXEC_LOG_MAX_ROWS", "5000") or 5000)


def _to_float(value: Any) -> Optional[float]:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num


def _read_jsonl(path: Path, limit: int) -> List[Dict[str, Any]]:
    if not path.exists() or limit <= 0:
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except Exception:
        return []
    if len(rows) > limit:
        rows = rows[-limit:]
    return rows


def _recent_slippage_from_metrics(symbol: Optional[str], window_days: int) -> List[float]:
    records = _read_jsonl(ORDER_METRICS_PATH, _READ_LIMIT)
    cutoff = time.time() - max(window_days, 0) * 86400.0
    values: List[float] = []
    sym_filter = str(symbol or "").upper()
    for record in records:
        sym = str(record.get("symbol") or "").upper()
        if sym_filter and sym != sym_filter:
            continue
        ts = record.get("ts")
        if isinstance(ts, (int, float)) and ts < cutoff:
            continue
        slip = _to_float(record.get("slippage_bps"))
        if slip is not None:
            values.append(slip)
    return values


def realized_slippage_bps_7d(symbol: Optional[str] = None) -> float:
    """Average realized slippage over ~7d, in bps."""
    events = fills.get_recent_fills(symbol=symbol, window_days=7)
    samples: List[float] = []
    for event in events:
        slip = _to_float(event.get("slippage_bps"))
        if slip is not None:
            samples.append(slip)
            continue
        mid = _to_float(event.get("mid_before"))
        px = _to_float(event.get("price"))
        if mid and px and mid != 0:
            diff = (px - mid) / mid * 10_000.0
            side = str(event.get("side") or "").upper()
            if side == "SELL":
                diff *= -1.0
            samples.append(diff)
    if not samples:
        samples = _recent_slippage_from_metrics(symbol, 7)
    if not samples:
        return 0.0
    return float(sum(samples) / len(samples))


def hourly_expectancy(symbol: str) -> Dict[int, float]:
    """
    Returns {hour: expectancy_in_USDT} for the last 7 days for a symbol.
    Delegates to pnl_tracker.
    """
    return pnl_tracker.get_hourly_expectancy(symbol=symbol, window_days=7)


def router_effectiveness_7d(symbol: Optional[str] = None) -> Dict[str, Optional[float]]:
    """
    Compute router quality stats over the last ~7 days.
    """
    events = router_metrics.get_recent_router_events(symbol=symbol, window_days=7)
    if not events:
        return {
            "maker_fill_ratio": None,
            "fallback_ratio": None,
            "slip_q25": None,
            "slip_q50": None,
            "slip_q75": None,
        }

    maker_flags: List[float] = []
    fallback_flags: List[float] = []
    slippages: List[float] = []

    for event in events:
        is_maker_final = bool(event.get("is_maker_final"))
        started_maker = bool(event.get("started_maker"))
        used_fallback = bool(event.get("used_fallback"))
        slip_bps = event.get("slippage_bps")

        if slip_bps is not None:
            try:
                slippages.append(float(slip_bps))
            except (TypeError, ValueError):
                pass

        if "is_maker_final" in event:
            maker_flags.append(1.0 if is_maker_final else 0.0)

        if started_maker:
            fallback_flags.append(1.0 if used_fallback else 0.0)

    def _ratio(values: List[float]) -> Optional[float]:
        if not values:
            return None
        return float(sum(values) / len(values))

    def _quartiles(values: List[float]) -> tuple[Optional[float], Optional[float], Optional[float]]:
        if not values:
            return (None, None, None)
        arr = np.array(values, dtype=float)
        try:
            q25 = float(np.percentile(arr, 25))
            q50 = float(np.percentile(arr, 50))
            q75 = float(np.percentile(arr, 75))
            return (q25, q50, q75)
        except Exception:
            return (None, None, None)

    q25, q50, q75 = _quartiles(slippages)

    return {
        "maker_fill_ratio": _ratio(maker_flags),
        "fallback_ratio": _ratio(fallback_flags),
        "slip_q25": q25,
        "slip_q50": q50,
        "slip_q75": q75,
    }
