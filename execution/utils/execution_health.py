"""
Execution health engine.

Combine router effectiveness, Sharpe, ATR regime, DD state, toggle status,
and sizing multipliers into a deterministic health payload.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Sequence

from execution.utils.metrics import (
    router_effectiveness_7d,
    rolling_sharpe_7d,
    dd_today_pct,
)
from execution.utils.vol import atr_pct
from execution.utils.toggle import is_symbol_disabled, get_symbol_disable_meta
from execution.position_sizing import volatility_regime_scale
from execution.intel.symbol_score import symbol_size_factor
from execution.intel.maker_offset import suggest_maker_offset_bps
from execution.intel.router_policy import router_policy
from execution.risk_limits import classify_drawdown_state


def size_multiplier(_symbol: str) -> float:
    return 1.0


FALLBACK_WARN_THRESHOLD = 0.50
SLIP_MEDIAN_WARN_BPS = 4.0
SHARPE_BAD = -1.0
SHARPE_GOOD = 1.5
DD_WARN_PCT = -1.5
DD_KILL_PCT = -3.0
ATR_QUIET = 0.7
ATR_HOT = 1.5
ATR_PANIC = 2.5
ERROR_SCHEMA = "execution_health_v1"

# component -> symbol -> {"count": int, "last": {...}}
_ERROR_REGISTRY: dict[str, dict[str, dict[str, Any]]] = {}
_RISK_GATES: list[dict[str, Any]] = []
_RISK_GATE_LIMIT = 256


def reset_error_registry() -> None:
    _ERROR_REGISTRY.clear()
    _RISK_GATES.clear()


def atr_regime_from_ratio(ratio: float) -> str:
    if ratio < ATR_QUIET:
        return "quiet"
    if ratio < ATR_HOT:
        return "normal"
    if ratio < ATR_PANIC:
        return "hot"
    return "panic"


def record_execution_error(
    component: str,
    *,
    symbol: Optional[str] = None,
    message: Optional[str] = None,
    classification: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Track the latest error per component + symbol for health snapshots."""
    comp = _ERROR_REGISTRY.setdefault(component, {})
    sym_key = (symbol or "__global__").upper()
    entry = comp.setdefault(sym_key, {"count": 0, "last": {}})
    entry["count"] = int(entry.get("count", 0) or 0) + 1
    detail: Dict[str, Any] = {
        "ts": time.time(),
        "message": message,
    }
    if classification:
        detail["classification"] = dict(classification)
    if context:
        detail["context"] = dict(context)
    entry["last"] = detail
    comp[sym_key] = entry


def record_risk_gate_triggered(
    gate: str,
    *,
    symbol: Optional[str] = None,
    reason: Optional[str] = None,
    thresholds: Optional[Dict[str, Any]] = None,
    observations: Optional[Dict[str, Any]] = None,
) -> None:
    """Optional hook to track risk gate events for execution health displays."""
    event: Dict[str, Any] = {
        "ts": time.time(),
        "gate": gate,
        "symbol": symbol,
        "reason": reason,
    }
    if thresholds:
        event["thresholds"] = dict(thresholds)
    if observations:
        event["observations"] = dict(observations)
    _RISK_GATES.append(event)
    if len(_RISK_GATES) > _RISK_GATE_LIMIT:
        del _RISK_GATES[0 : len(_RISK_GATES) - _RISK_GATE_LIMIT]


def _error_view_for_symbol(symbol: Optional[str]) -> Dict[str, Any]:
    view: Dict[str, Any] = {}
    sym_key = symbol.upper() if symbol else None
    for component, by_symbol in _ERROR_REGISTRY.items():
        count = 0
        last: Dict[str, Any] | None = None
        for key, payload in by_symbol.items():
            if sym_key and key not in {sym_key, "__global__"}:
                continue
            count += int(payload.get("count", 0) or 0)
            latest = payload.get("last") or {}
            if not last or (latest.get("ts") or 0) > (last.get("ts") or 0):
                last = latest
        view[component] = {
            "count": count,
            "last_error": last,
        }
    return view


def _recent_risk_gates(symbol: Optional[str]) -> list[dict[str, Any]]:
    if not _RISK_GATES:
        return []
    sym_key = symbol.upper() if symbol else None
    events = []
    for event in _RISK_GATES[-_RISK_GATE_LIMIT:]:
        if sym_key and str(event.get("symbol") or "").upper() not in {sym_key, ""}:
            continue
        events.append(event)
    return events


def classify_atr_regime(symbol: str) -> Dict[str, Any]:
    atr_now = atr_pct(symbol, lookback_bars=50)
    atr_med = atr_pct(symbol, lookback_bars=500)
    if atr_med <= 0:
        return {"atr_now": atr_now, "atr_med": atr_med, "atr_ratio": None, "atr_regime": "unknown"}
    ratio = atr_now / atr_med
    regime = atr_regime_from_ratio(ratio)
    return {"atr_now": atr_now, "atr_med": atr_med, "atr_ratio": ratio, "atr_regime": regime}


def summarize_atr_regimes(symbols: Sequence[str]) -> Dict[str, Any]:
    entries: list[Dict[str, Any]] = []
    ratios: list[float] = []
    for sym in symbols:
        entry = classify_atr_regime(sym)
        entry["symbol"] = sym
        entries.append(entry)
        ratio_val = entry.get("atr_ratio")
        try:
            if ratio_val is not None:
                ratios.append(float(ratio_val))
        except Exception:
            continue
    median_ratio = None
    if ratios:
        try:
            ratios_sorted = sorted(ratios)
            mid = len(ratios_sorted) // 2
            if len(ratios_sorted) % 2 == 1:
                median_ratio = ratios_sorted[mid]
            else:
                median_ratio = (ratios_sorted[mid - 1] + ratios_sorted[mid]) / 2.0
        except Exception:
            median_ratio = None
    regime = "unknown"
    if median_ratio is not None:
        regime = atr_regime_from_ratio(median_ratio)
    return {"atr_regime": regime, "median_ratio": median_ratio, "symbols": entries}


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
    dd_state = classify_drawdown_state(dd_pct=dd, alert_pct=abs(DD_WARN_PCT), kill_pct=abs(DD_KILL_PCT))
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
        "dd_state": dd_state,
        "risk_flags": risk_flags,
        "sharpe_state": sharpe_state,
        "toggle_active": disabled,
        "toggle_meta": toggle_meta,
    }


def compute_execution_health(symbol: Optional[str] = None) -> Dict[str, Any]:
    router_stats = router_effectiveness_7d(symbol)
    router_part = classify_router_health(router_stats)
    if symbol is None:
        return {
            "schema": ERROR_SCHEMA,
            "symbol": None,
            "router": router_part,
            "risk": None,
            "vol": None,
            "sizing": None,
            "errors": _error_view_for_symbol(None),
            "components": {"router": router_part, "risk": None, "vol": None, "sizing": None},
        }
    sharpe = rolling_sharpe_7d(symbol)
    risk_part = classify_risk_health(symbol, sharpe)
    vol_part = classify_atr_regime(symbol)
    size_mult = size_multiplier(symbol)
    regime_mult = volatility_regime_scale(symbol)
    intel_factor = 1.0
    try:
        intel_payload = symbol_size_factor(symbol)
        intel_factor = float(intel_payload.get("size_factor") or 1.0)
    except Exception:
        intel_factor = 1.0
    sizing_part = {
        "sharpe_7d": sharpe,
        "size_mult_sharpe": size_mult,
        "size_mult_regime": regime_mult,
        "intel_size_factor": intel_factor,
        "size_mult_combined": size_mult * regime_mult,
        "final_size_factor": size_mult * regime_mult * intel_factor,
    }
    router_part["last_route_decision"] = (router_stats or {}).get("last_route_decision") if isinstance(router_stats, dict) else None
    policy_obj = None
    try:
        policy_obj = router_policy(symbol)
    except Exception:
        policy_obj = None
    if policy_obj is not None:
        router_part["policy_quality"] = policy_obj.quality
        router_part["policy_maker_first"] = policy_obj.maker_first
        router_part["policy_taker_bias"] = policy_obj.taker_bias
        router_part["policy_reason"] = policy_obj.reason
        router_part["maker_offset_bps"] = router_part.get("maker_offset_bps") or policy_obj.offset_bps
        router_part["policy"] = {
            "maker_first": policy_obj.maker_first,
            "taker_bias": policy_obj.taker_bias,
            "quality": policy_obj.quality,
            "reason": policy_obj.reason,
            "offset_bps": policy_obj.offset_bps,
        }
    else:
        router_part.setdefault("policy_quality", None)
        router_part.setdefault("policy_maker_first", None)
        router_part.setdefault("policy_taker_bias", None)
        router_part.setdefault("policy_reason", None)
        router_part.setdefault("policy", None)
    if router_part.get("maker_offset_bps") is None:
        try:
            router_part["maker_offset_bps"] = suggest_maker_offset_bps(symbol)
        except Exception:
            router_part.setdefault("maker_offset_bps", None)
    errors = _error_view_for_symbol(symbol)
    components = {
        "router": router_part,
        "risk": risk_part,
        "vol": vol_part,
        "sizing": sizing_part,
    }
    return {
        "schema": ERROR_SCHEMA,
        "symbol": symbol,
        "router": router_part,
        "risk": risk_part,
        "vol": vol_part,
        "sizing": sizing_part,
        "errors": errors,
        "components": components,
        "events": {"risk_gates": _recent_risk_gates(symbol)},
    }


__all__ = [
    "compute_execution_health",
    "classify_atr_regime",
    "classify_router_health",
    "classify_risk_health",
    "record_execution_error",
    "record_risk_gate_triggered",
    "reset_error_registry",
    "summarize_atr_regimes",
]
