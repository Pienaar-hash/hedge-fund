"""
Execution health engine.

Combine router effectiveness, Sharpe, ATR regime, DD state, toggle status,
and sizing multipliers into a deterministic health payload.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from execution.utils.metrics import (
    router_effectiveness_7d,
    rolling_sharpe_7d,
    dd_today_pct,
)
from execution.utils.vol import atr_pct
from execution.utils.toggle import is_symbol_disabled, get_symbol_disable_meta
from execution.risk_autotune import size_multiplier
from execution.position_sizing import volatility_regime_scale


FALLBACK_WARN_THRESHOLD = 0.50
SLIP_MEDIAN_WARN_BPS = 4.0
SHARPE_BAD = -1.0
SHARPE_GOOD = 1.5
DD_WARN_PCT = -1.5
DD_KILL_PCT = -3.0
ATR_QUIET = 0.7
ATR_HOT = 1.5
ATR_PANIC = 2.5


def classify_atr_regime(symbol: str) -> Dict[str, Any]:
    atr_now = atr_pct(symbol, lookback_bars=50)
    atr_med = atr_pct(symbol, lookback_bars=500)
    if atr_med <= 0:
        return {"atr_now": atr_now, "atr_med": atr_med, "atr_ratio": None, "atr_regime": "unknown"}
    ratio = atr_now / atr_med
    if ratio < ATR_QUIET:
        regime = "quiet"
    elif ratio < ATR_HOT:
        regime = "normal"
    elif ratio < ATR_PANIC:
        regime = "hot"
    else:
        regime = "panic"
    return {"atr_now": atr_now, "atr_med": atr_med, "atr_ratio": ratio, "atr_regime": regime}


def classify_router_health(router_stats: Dict[str, Any]) -> Dict[str, Any]:
    maker_fill_ratio = router_stats.get("maker_fill_ratio")
    fallback_ratio = router_stats.get("fallback_ratio")
    slip_q50 = router_stats.get("slip_q50")
    warnings = []
    if fallback_ratio is not None and fallback_ratio > FALLBACK_WARN_THRESHOLD:
        warnings.append("high_fallback_ratio")
    if slip_q50 is not None and slip_q50 > SLIP_MEDIAN_WARN_BPS:
        warnings.append("elevated_median_slippage")
    return {
        "router_warnings": warnings,
        "maker_fill_ratio": maker_fill_ratio,
        "fallback_ratio": fallback_ratio,
        "slip_q25": router_stats.get("slip_q25"),
        "slip_q50": slip_q50,
        "slip_q75": router_stats.get("slip_q75"),
    }


def classify_risk_health(symbol: str, sharpe: Optional[float]) -> Dict[str, Any]:
    dd = dd_today_pct(symbol) or 0.0
    disabled = is_symbol_disabled(symbol)
    toggle_meta = get_symbol_disable_meta(symbol) if disabled else None
    risk_flags = []
    if dd <= DD_WARN_PCT:
        risk_flags.append("dd_warning")
    if dd <= DD_KILL_PCT:
        risk_flags.append("dd_kill_threshold")
    if disabled:
        risk_flags.append("symbol_disabled")
    if sharpe is None:
        sharpe_state = "unknown"
    elif sharpe <= SHARPE_BAD:
        sharpe_state = "poor"
    elif sharpe >= SHARPE_GOOD:
        sharpe_state = "strong"
    else:
        sharpe_state = "neutral"
    return {
        "dd_today_pct": dd,
        "risk_flags": risk_flags,
        "sharpe_state": sharpe_state,
        "toggle_active": disabled,
        "toggle_meta": toggle_meta,
    }


def compute_execution_health(symbol: Optional[str] = None) -> Dict[str, Any]:
    router_stats = router_effectiveness_7d(symbol)
    router_part = classify_router_health(router_stats)
    if symbol is None:
        return {"symbol": None, "router": router_part, "risk": None, "vol": None, "sizing": None}
    sharpe = rolling_sharpe_7d(symbol)
    risk_part = classify_risk_health(symbol, sharpe)
    vol_part = classify_atr_regime(symbol)
    size_mult = size_multiplier(symbol)
    regime_mult = volatility_regime_scale(symbol)
    sizing_part = {
        "sharpe_7d": sharpe,
        "size_mult_sharpe": size_mult,
        "size_mult_regime": regime_mult,
        "size_mult_combined": size_mult * regime_mult,
    }
    return {
        "symbol": symbol,
        "router": router_part,
        "risk": risk_part,
        "vol": vol_part,
        "sizing": sizing_part,
    }


__all__ = [
    "compute_execution_health",
    "classify_atr_regime",
    "classify_router_health",
    "classify_risk_health",
]
