from __future__ import annotations

from typing import Any, Mapping


def classify_weekly_risk_mode(
    *,
    weekly_loss_pct: float,
    drawdown_pct: float,
    data_valid: bool,
    audit_state_valid: bool,
    maximum_weekly_loss_pct: float = 0.08,
    maximum_drawdown_pct: float = 0.20,
) -> str:
    if (
        weekly_loss_pct >= maximum_weekly_loss_pct
        or drawdown_pct >= maximum_drawdown_pct
        or not data_valid
        or not audit_state_valid
    ):
        return "HALTED"
    return "ACTIVE"


def validate_entry(
    candidate: Mapping[str, Any],
    *,
    risk_mode: str,
    selected_symbols: set[str],
    existing_groups: set[str],
    unresolved_orders: bool,
    audit_chain_ok: bool,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if risk_mode != "ACTIVE":
        reasons.append("risk_mode")
    if candidate.get("conviction") != "HIGH":
        reasons.append("conviction")
    if candidate.get("symbol") not in selected_symbols:
        reasons.append("selection")
    if candidate.get("correlation_group") in existing_groups:
        reasons.append("correlation_group")
    if unresolved_orders:
        reasons.append("unresolved_orders")
    if not audit_chain_ok:
        reasons.append("audit_chain")
    return (not reasons, reasons)


def validate_portfolio_targets(
    target_positions: list[Mapping[str, Any]],
    *,
    maximum_gross_exposure_pct: float = 1.0,
    maximum_concurrent_position_risk_pct: float = 0.10,
) -> tuple[bool, list[str]]:
    gross = sum(float(item["target_pct"]) for item in target_positions)
    concurrent_risk = sum(
        float(item["target_pct"]) * float(item.get("hard_stop_pct", 0.10))
        for item in target_positions
    )
    reasons: list[str] = []
    if gross > maximum_gross_exposure_pct:
        reasons.append("gross_exposure")
    if concurrent_risk > maximum_concurrent_position_risk_pct:
        reasons.append("concurrent_position_risk")
    return (not reasons, reasons)


def evaluate_hard_stop(
    *,
    entry_price: float,
    current_price: float,
    hard_stop_pct: float = 0.10,
) -> bool:
    stop_price = entry_price * (1.0 - hard_stop_pct)
    return current_price <= stop_price
