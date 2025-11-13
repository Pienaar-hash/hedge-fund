"""
Volatility helpers for execution sizing.

Backed by pnl_tracker or other utilities that compute realized volatility.
"""

from __future__ import annotations

from typing import Any, Mapping

from execution import pnl_tracker


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def rolling_sigma(symbol: str, lookback: int = 50) -> float:
    """
    Realized std of per-trade or per-bar returns for `symbol` over `lookback`.
    Delegates to pnl_tracker; normalize as needed downstream.
    """
    stats = pnl_tracker.get_symbol_stats(symbol, window_trades=lookback)
    return float(stats.get("std", 0.0) or 0.0)


def atr_pct(symbol: str, lookback_bars: int = 50, *, median_only: bool = False) -> float:
    """
    Percent ATR over `lookback_bars`. Falls back to medians when requested.
    """
    kwargs = {"lookback_bars": lookback_bars}
    if median_only:
        kwargs["median_only"] = True
    payload: Mapping[str, Any] | None = None
    try:
        payload = pnl_tracker.get_symbol_atr(symbol, **kwargs)  # type: ignore[arg-type]
    except TypeError:
        # Legacy signatures may not accept kwargs.
        try:
            payload = pnl_tracker.get_symbol_atr(symbol, lookback_bars)  # type: ignore[misc]
        except Exception:
            payload = None
    except Exception:
        payload = None
    if not isinstance(payload, Mapping):
        return 0.0
    key = "atr_pct_median" if median_only else "atr_pct"
    value = payload.get(key)
    if value is None and median_only:
        value = payload.get("atr_pct")
    return _as_float(value)
