from __future__ import annotations

from execution.weekly_position_sizer import (
    base_allocation_for_count,
    compute_target_positions,
    conviction_multiplier,
    risk_based_position_limit,
)


def test_base_allocations_are_fixed():
    assert base_allocation_for_count(1) == 0.60
    assert base_allocation_for_count(2) == 0.40
    assert base_allocation_for_count(3) == 0.30


def test_conviction_multiplier_bands_are_fixed():
    assert conviction_multiplier(0.91) == 1.10
    assert conviction_multiplier(0.87) == 1.00
    assert conviction_multiplier(0.82) == 0.90


def test_position_risk_cap_binds_correctly():
    selected = [
        {"symbol": "BTCUSDT", "momentum_score": 0.95, "conviction": "HIGH"},
        {"symbol": "SOLUSDT", "momentum_score": 0.81, "conviction": "HIGH"},
    ]
    targets = compute_target_positions(
        selected,
        nav_usd=1000.0,
        symbol_precision={
            "BTCUSDT": {"notional_precision": 2, "min_notional": 25.0},
            "SOLUSDT": {"notional_precision": 2, "min_notional": 25.0},
        },
    )
    assert risk_based_position_limit() == 0.4
    assert all(target["target_pct"] <= 0.4 for target in targets)


def test_fewer_than_three_candidates_leaves_cash():
    selected = [{"symbol": "BTCUSDT", "momentum_score": 0.85, "conviction": "HIGH"}]
    targets = compute_target_positions(
        selected,
        nav_usd=10000.0,
        symbol_precision={"BTCUSDT": {"notional_precision": 2, "min_notional": 25.0}},
    )
    assert len(targets) == 1
    assert targets[0]["target_pct"] == 0.4
