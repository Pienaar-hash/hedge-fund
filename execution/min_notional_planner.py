"""Min-notional planning decisions for order intents.

This module is policy-only. It never bypasses exchange minimums and does not
place or mutate orders directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MinNotionalAction(str, Enum):
    PASS = "PASS"
    UPSIZE_TO_MIN_NOTIONAL = "UPSIZE_TO_MIN_NOTIONAL"
    ABSTAIN_MIN_NOTIONAL = "ABSTAIN_MIN_NOTIONAL"
    REJECT_MIN_NOTIONAL_UNECONOMIC = "REJECT_MIN_NOTIONAL_UNECONOMIC"


@dataclass(frozen=True)
class MinNotionalPlan:
    action: MinNotionalAction
    min_notional_required: float
    intended_notional: float
    adjusted_notional: float
    reason: str

    def to_log_fields(self) -> dict[str, float | str]:
        return {
            "min_notional_action": self.action.value,
            "min_notional_required": float(self.min_notional_required),
            "intended_notional": float(self.intended_notional),
            "adjusted_notional": float(self.adjusted_notional),
            "reason": self.reason,
        }


def _f(value: float | int | str | None) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def plan_min_notional_action(
    *,
    symbol: str,
    intended_qty: float,
    mark_price: float,
    intended_notional: float,
    min_notional: float,
    nav_usd: float,
    max_nav_pct: float,
    leverage: float,
    fee_rate: float,
) -> MinNotionalPlan:
    """Return a min-notional planning decision for the provided intent."""
    _ = symbol  # reserved for future symbol-aware economics

    qty = _f(intended_qty)
    price = _f(mark_price)
    intended = max(0.0, _f(intended_notional))
    required = max(0.0, _f(min_notional))
    nav = max(0.0, _f(nav_usd))
    max_pct = max(0.0, _f(max_nav_pct))
    lev = max(0.0, _f(leverage))
    fee = max(0.0, _f(fee_rate))

    if required <= 0.0 or intended >= required:
        return MinNotionalPlan(
            action=MinNotionalAction.PASS,
            min_notional_required=required,
            intended_notional=intended,
            adjusted_notional=intended,
            reason="meets_min_notional",
        )

    if qty <= 0.0 or price <= 0.0:
        return MinNotionalPlan(
            action=MinNotionalAction.ABSTAIN_MIN_NOTIONAL,
            min_notional_required=required,
            intended_notional=intended,
            adjusted_notional=intended,
            reason="invalid_qty_or_price",
        )

    adjusted = required
    implied_cap = 0.0
    if nav > 0.0 and max_pct > 0.0:
        implied_cap = nav * max_pct
    if implied_cap > 0.0 and adjusted > implied_cap:
        return MinNotionalPlan(
            action=MinNotionalAction.ABSTAIN_MIN_NOTIONAL,
            min_notional_required=required,
            intended_notional=intended,
            adjusted_notional=adjusted,
            reason="upsize_breaches_nav_cap",
        )

    uplift_ratio = adjusted / max(intended, 1e-12)
    est_round_trip_fee = adjusted * fee * 2.0
    if uplift_ratio >= 2.0 and est_round_trip_fee >= max(0.01, intended * 0.01):
        return MinNotionalPlan(
            action=MinNotionalAction.REJECT_MIN_NOTIONAL_UNECONOMIC,
            min_notional_required=required,
            intended_notional=intended,
            adjusted_notional=adjusted,
            reason="upsize_uneconomic",
        )

    return MinNotionalPlan(
        action=MinNotionalAction.UPSIZE_TO_MIN_NOTIONAL,
        min_notional_required=required,
        intended_notional=intended,
        adjusted_notional=adjusted,
        reason=("upsize_to_exchange_min_with_leverage" if lev > 1.0 else "upsize_to_exchange_min"),
    )
