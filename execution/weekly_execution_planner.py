from __future__ import annotations

from typing import Any, Mapping


def diff_current_vs_target(
    current_positions: list[Mapping[str, Any]],
    target_positions: list[Mapping[str, Any]],
    *,
    rebalance_threshold_pct: float = 0.05,
) -> list[dict[str, Any]]:
    current_by_symbol = {item["symbol"]: dict(item) for item in current_positions}
    target_by_symbol = {item["symbol"]: dict(item) for item in target_positions}
    symbols = sorted(set(current_by_symbol) | set(target_by_symbol))
    diffs: list[dict[str, Any]] = []
    for symbol in symbols:
        current = current_by_symbol.get(symbol, {})
        target = target_by_symbol.get(symbol, {})
        current_pct = float(current.get("current_pct", current.get("target_pct", 0.0)) or 0.0)
        target_pct = float(target.get("target_pct", 0.0) or 0.0)
        delta_pct = target_pct - current_pct
        if current and not target:
            action = "EXIT"
        elif target and not current:
            action = "ENTER"
        elif abs(delta_pct) >= rebalance_threshold_pct:
            action = "REBALANCE"
        else:
            action = "HOLD"
        diffs.append(
            {
                "symbol": symbol,
                "action": action,
                "current_pct": current_pct,
                "target_pct": target_pct,
                "delta_pct": delta_pct,
                "current_position": current,
                "target_position": target,
            }
        )
    return diffs


def build_order_intents(
    diffs: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    orders: list[dict[str, Any]] = []
    for diff in diffs:
        if diff["action"] == "HOLD":
            continue
        side = "BUY"
        if diff["action"] == "EXIT" or float(diff["delta_pct"]) < 0:
            side = "SELL"
        orders.append(
            {
                "symbol": diff["symbol"],
                "action": diff["action"],
                "side": side,
                "target_pct": diff["target_pct"],
                "delta_pct": diff["delta_pct"],
                "target_notional_usd": float(
                    diff.get("target_position", {}).get("target_notional_usd", 0.0)
                ),
            }
        )
    return orders


def build_weekly_plan(
    *,
    cycle_id: str,
    risk_mode: str,
    current_positions: list[Mapping[str, Any]],
    target_positions: list[Mapping[str, Any]],
    rebalance_threshold_pct: float = 0.05,
) -> dict[str, Any]:
    diffs = diff_current_vs_target(
        current_positions,
        target_positions,
        rebalance_threshold_pct=rebalance_threshold_pct,
    )
    orders = build_order_intents(diffs)
    return {
        "cycle_id": cycle_id,
        "risk_mode": risk_mode,
        "current_positions": [dict(item) for item in current_positions],
        "target_positions": [dict(item) for item in target_positions],
        "diffs": diffs,
        "orders": orders,
    }
