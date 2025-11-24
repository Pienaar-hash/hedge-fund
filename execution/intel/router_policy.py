from __future__ import annotations

"""
Router policy engine (v5.10.4).

Decides whether to run maker-first and what taker bias to apply based on
router effectiveness and volatility regimes.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

from execution.utils.metrics import router_effectiveness_7d
from execution.intel import maker_offset
from execution.intel.maker_offset import classify_atr_regime


@dataclass
class RouterPolicy:
    maker_first: bool
    taker_bias: str
    quality: str
    reason: str
    offset_bps: float | None = None


def _to_float(value: Any) -> Optional[float]:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num


def _clamp01(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return max(0.0, min(1.0, value))


def classify_router_quality(
    eff: Mapping[str, Any] | float | None,
    slip_p95_bps: Optional[float] = None,
    fallback_rate: Optional[float] = None,
    latency_ms: Optional[float] = None,
) -> str:
    """
    Classify router quality into: "good", "ok", "degraded", "broken".

    Backwards compatible with the legacy dict signature while also supporting
    direct numeric inputs for the v6 intel helpers.
    """
    maker = None
    if isinstance(eff, Mapping):
        maker = _to_float(eff.get("maker_fill_ratio") or eff.get("maker_fill_rate"))
        if slip_p95_bps is None:
            slip_p95_bps = _to_float(eff.get("slip_q95") or eff.get("slippage_p95") or eff.get("slip_q50"))
        if fallback_rate is None:
            fallback_rate = _to_float(eff.get("fallback_ratio") or eff.get("fallback_rate"))
        if latency_ms is None:
            latency_ms = _to_float(eff.get("ack_latency_ms") or eff.get("latency_ms") or eff.get("latency_p50_ms"))
    elif eff is not None:
        maker = _to_float(eff)

    maker = _clamp01(maker) or 0.0
    fallback = _clamp01(_to_float(fallback_rate)) or 0.0
    slip_val = _to_float(slip_p95_bps)
    if slip_val is None:
        slip_val = 0.0
    latency_val = _to_float(latency_ms) or 0.0

    if fallback >= 0.9 or slip_val >= 20.0 or latency_val >= 2500:
        return "broken"
    if fallback >= 0.6 or slip_val >= 8.0 or latency_val >= 1500:
        return "degraded"
    if maker >= 0.7 and fallback <= 0.4 and slip_val <= 4.0 and (latency_val == 0.0 or latency_val <= 800):
        return "good"
    return "ok"


def classify_router_regime(metrics: Mapping[str, Any]) -> str:
    """
    Determine a coarse router regime as a helper for auto-tune intel.
    """
    maker = _clamp01(_to_float(metrics.get("maker_fill_rate") or metrics.get("maker_fill_ratio"))) or 0.0
    fallback = _clamp01(_to_float(metrics.get("fallback_rate") or metrics.get("fallback_ratio"))) or 0.0
    slip = _to_float(metrics.get("slippage_p95") or metrics.get("slip_q95") or metrics.get("slip_q50"))
    latency = _to_float(metrics.get("ack_latency_ms") or metrics.get("latency_ms") or metrics.get("latency_p50_ms"))

    if fallback >= 0.85 or maker <= 0.2 or (slip is not None and slip >= 25.0) or (latency is not None and latency >= 3000):
        return "broken"
    if fallback >= 0.55:
        return "fallback_heavy"
    if slip is not None and slip >= 12.0:
        return "slippage_hot"
    if maker >= 0.7 and fallback <= 0.3 and (slip is None or slip <= 5.0):
        return "maker_strong"
    return "balanced"


def compute_router_summary(metrics: Iterable[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    """
    Aggregate raw router telemetry rows into the canonical summary fields
    used across router policy, dashboards, and auto-tune logic.
    """
    maker_samples = 0
    maker_success = 0
    fallback_samples = 0
    fallback_used = 0
    slippage_values: list[float] = []
    latency_values: list[float] = []

    for row in metrics:
        is_maker_final = row.get("is_maker_final")
        maker_started = row.get("maker_start") or row.get("started_maker")
        used_fallback = row.get("used_fallback")
        if is_maker_final is not None:
            maker_samples += 1
            if bool(is_maker_final):
                maker_success += 1
        if maker_started:
            fallback_samples += 1
            if bool(used_fallback):
                fallback_used += 1

        slip = row.get("slippage_bps") or row.get("slippage")
        slip_val = _to_float(slip)
        if slip_val is not None:
            slippage_values.append(slip_val)

        latency = row.get("ack_latency_ms") or row.get("latency_ms")
        lat_val = _to_float(latency)
        if lat_val is not None:
            latency_values.append(lat_val)

    def _ratio(success: int, total: int) -> Optional[float]:
        if total <= 0:
            return None
        return float(success) / float(total)

    def _percentile(values: list[float], pct: float) -> Optional[float]:
        if not values:
            return None
        ordered = sorted(values)
        if not ordered:
            return None
        idx = (len(ordered) - 1) * max(0.0, min(1.0, pct / 100.0))
        lower = int(idx)
        upper = min(lower + 1, len(ordered) - 1)
        weight = idx - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "maker_fill_rate": _ratio(maker_success, maker_samples),
        "fallback_rate": _ratio(fallback_used, fallback_samples),
        "slippage_p50": _percentile(slippage_values, 50.0),
        "slippage_p95": _percentile(slippage_values, 95.0),
        "ack_latency_p50_ms": _percentile(latency_values, 50.0),
        "sample_count": float(maker_samples) if maker_samples else None,
    }


def router_policy(symbol: str) -> RouterPolicy:
    eff = router_effectiveness_7d(symbol) or {}
    quality = classify_router_quality(eff)
    regime = classify_atr_regime(symbol)
    try:
        base_offset = float(maker_offset.suggest_maker_offset_bps(symbol))
    except Exception:
        base_offset = None

    maker_first = True
    taker_bias = "balanced"
    reason_parts = [f"quality={quality}", f"regime={regime}"]

    if quality == "broken":
        maker_first = False
        taker_bias = "prefer_taker"
        reason_parts.append("fallback/slippage too high")
    elif quality == "degraded":
        maker_first = True
        taker_bias = "prefer_taker"
        reason_parts.append("router degraded")
    elif quality == "good":
        taker_bias = "prefer_maker"
        reason_parts.append("router good")

    if regime == "panic":
        taker_bias = "prefer_taker"
        if quality in ("degraded", "broken"):
            maker_first = False
            reason_parts.append("panic regime + weak router")

    return RouterPolicy(
        maker_first=maker_first,
        taker_bias=taker_bias,
        quality=quality,
        reason="; ".join(reason_parts),
        offset_bps=base_offset,
    )


__all__ = [
    "RouterPolicy",
    "router_policy",
    "classify_router_quality",
    "classify_router_regime",
    "compute_router_summary",
]
