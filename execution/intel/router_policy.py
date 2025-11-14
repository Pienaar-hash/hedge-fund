from __future__ import annotations

"""
Router policy engine (v5.10.4).

Decides whether to run maker-first and what taker bias to apply based on
router effectiveness and volatility regimes.
"""

from dataclasses import dataclass
from typing import Any, Dict

from execution.utils.metrics import router_effectiveness_7d
from execution.intel.maker_offset import classify_atr_regime


@dataclass
class RouterPolicy:
    maker_first: bool
    taker_bias: str
    quality: str
    reason: str


def classify_router_quality(eff: Dict[str, Any]) -> str:
    """
    Classify router quality into: "good", "ok", "degraded", "broken".
    """
    maker = float(eff.get("maker_fill_ratio") or 0.0)
    fallback = float(eff.get("fallback_ratio") or 0.0)
    slip_med = eff.get("slip_q50")
    try:
        slip_med_val = float(slip_med)
    except (TypeError, ValueError):
        slip_med_val = 0.0

    if fallback >= 0.9 or slip_med_val >= 20.0:
        return "broken"
    if fallback >= 0.6 or slip_med_val >= 8.0:
        return "degraded"
    if maker >= 0.7 and fallback <= 0.4 and slip_med_val <= 4.0:
        return "good"
    return "ok"


def router_policy(symbol: str) -> RouterPolicy:
    eff = router_effectiveness_7d(symbol) or {}
    quality = classify_router_quality(eff)
    regime = classify_atr_regime(symbol)

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
    )


__all__ = ["RouterPolicy", "router_policy", "classify_router_quality"]
