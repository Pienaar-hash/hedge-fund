from __future__ import annotations

import math
from typing import Any, Mapping

from execution.exchange_precision import get_min_notional


def base_allocation_for_count(count: int) -> float:
    if count <= 0:
        return 0.0
    if count == 1:
        return 0.60
    if count == 2:
        return 0.40
    return 0.30


def conviction_multiplier(momentum_score: float) -> float:
    if momentum_score >= 0.90:
        return 1.10
    if momentum_score >= 0.85:
        return 1.00
    return 0.90


def risk_based_position_limit(
    *,
    max_loss_pct_per_position: float = 0.04,
    hard_stop_pct: float = 0.10,
) -> float:
    return round(max_loss_pct_per_position / hard_stop_pct, 10)


def _floor_notional(value: float, decimals: int) -> float:
    scale = 10 ** decimals
    return math.floor(value * scale) / scale


def compute_target_positions(
    selected_candidates: list[Mapping[str, Any]],
    *,
    nav_usd: float,
    minimum_position_pct: float = 0.20,
    absolute_position_cap_pct: float = 0.45,
    maximum_gross_exposure_pct: float = 1.0,
    max_loss_pct_per_position: float = 0.04,
    hard_stop_pct: float = 0.10,
    symbol_precision: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    precision_table = symbol_precision or {}
    base_pct = base_allocation_for_count(len(selected_candidates))
    raw_targets: list[dict[str, Any]] = []
    for item in selected_candidates:
        raw_pct = base_pct * conviction_multiplier(float(item["momentum_score"]))
        raw_targets.append({"symbol": item["symbol"], "candidate": dict(item), "raw_target_pct": raw_pct})
    total_raw = sum(item["raw_target_pct"] for item in raw_targets)
    if total_raw > maximum_gross_exposure_pct and total_raw > 0:
        for item in raw_targets:
            item["target_pct"] = item["raw_target_pct"] / total_raw
    else:
        for item in raw_targets:
            item["target_pct"] = item["raw_target_pct"]
    risk_limit = risk_based_position_limit(
        max_loss_pct_per_position=max_loss_pct_per_position,
        hard_stop_pct=hard_stop_pct,
    )
    targets: list[dict[str, Any]] = []
    for item in raw_targets:
        target_pct = min(item["target_pct"], risk_limit, absolute_position_cap_pct)
        if target_pct < minimum_position_pct or target_pct > absolute_position_cap_pct:
            continue
        precision_cfg = precision_table.get(item["symbol"], {})
        notional_precision = int(precision_cfg.get("notional_precision", 2))
        min_notional = float(precision_cfg.get("min_notional", get_min_notional(item["symbol"])))
        target_notional_usd = _floor_notional(nav_usd * target_pct, notional_precision)
        if target_notional_usd < min_notional:
            continue
        targets.append(
            {
                "symbol": item["symbol"],
                "target_pct": round(target_pct, 10),
                "target_notional_usd": target_notional_usd,
                "hard_stop_pct": hard_stop_pct,
                "max_loss_pct_per_position": max_loss_pct_per_position,
                "momentum_score": item["candidate"]["momentum_score"],
                "conviction": item["candidate"]["conviction"],
                "correlation_group": item["candidate"].get("correlation_group"),
            }
        )
    return targets
