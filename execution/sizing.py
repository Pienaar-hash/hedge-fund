"""Position sizing helpers extracted from executor_live.py.

All functions are pure — they take explicit arguments and return values
without reading module-level mutable state.

Part of v7.9 architecture repair sprint — Phase 1 extraction.
"""

from __future__ import annotations

from typing import Any, Mapping


def nav_pct_fraction(value: Any) -> float:
    """Interpret numeric percent inputs as fractions; 10 -> 0.10, 0.02 -> 0.02."""
    try:
        pct = float(value)
    except (TypeError, ValueError):
        return 0.0
    if pct <= 0.0:
        return 0.0
    if pct > 1.0:
        return pct / 100.0
    return pct


def size_from_nav(symbol: str, nav_usd: float, pct: float) -> float:
    """Compute gross notional from NAV * pct fraction."""
    try:
        return float(nav_usd) * float(pct)
    except Exception:
        return 0.0


def estimate_intent_qty(intent: Mapping[str, Any], gross_target: float, price_hint: float) -> float:
    """Resolve order quantity from intent fields, falling back to gross/price."""
    for key in ("quantity", "qty", "order_qty", "orderQty", "size", "units"):
        if key in intent:
            try:
                return float(intent[key])
            except Exception:
                continue
    try:
        normalized = intent.get("normalized")
        if isinstance(normalized, Mapping) and "qty" in normalized:
            return float(normalized.get("qty") or 0.0)
    except Exception:
        pass
    try:
        if price_hint and price_hint > 0:
            return float(gross_target) / float(price_hint)
    except Exception:
        pass
    return float(intent.get("qty_estimate", 0.0) or 0.0)
