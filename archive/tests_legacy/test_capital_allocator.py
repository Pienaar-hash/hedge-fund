from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from execution.capital_allocator import (
    AllocationConfig,
    CapitalAllocation,
    StrategyStats,
    compute_allocation,
    compute_strategy_stats,
    from_returns_frame,
    persist_allocation,
)


def test_compute_strategy_stats_basic() -> None:
    returns = [0.01, -0.005, 0.02, 0.015, 0.0]
    stats = compute_strategy_stats(returns, annualization=252, min_periods=3)
    assert stats.sample_size == len(returns)
    assert stats.volatility > 0
    assert stats.sharpe > 0


def test_allocation_from_returns_frame(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    returns = pd.DataFrame(
        {
            "momentum": rng.normal(0.001, 0.02, len(dates)),
            "carry": rng.normal(0.0005, 0.015, len(dates)),
            "vol_arb": rng.normal(0.0008, 0.017, len(dates)),
        },
        index=dates,
    )
    allocation = from_returns_frame(returns, window=20, method="pearson")
    assert isinstance(allocation, CapitalAllocation)
    assert abs(sum(allocation.weights.values()) - 1.0) < 1e-9
    assert all(weight >= 0 for weight in allocation.weights.values())

    output_path = tmp_path / "capital_alloc.json"
    persist_allocation(allocation, path=output_path)
    payload = output_path.read_text()
    assert '"weights"' in payload


def test_compute_allocation_penalizes_correlation() -> None:
    stats = {
        "A": StrategyStats(sharpe=1.6, volatility=0.02, drawdown=-0.1, sample_size=120),
        "B": StrategyStats(sharpe=1.6, volatility=0.02, drawdown=-0.12, sample_size=120),
        "C": StrategyStats(sharpe=1.0, volatility=0.03, drawdown=-0.08, sample_size=120),
    }
    matrix = pd.DataFrame(
        [
            [1.0, 0.8, 0.1],
            [0.8, 1.0, 0.8],
            [0.1, 0.8, 1.0],
        ],
        index=list(stats.keys()),
        columns=list(stats.keys()),
    )
    allocation = compute_allocation(stats, matrix, config=AllocationConfig(volatility_floor=1e-4))
    weights = allocation.weights
    assert weights["A"] > weights["B"]
    assert weights["A"] > weights["C"]
