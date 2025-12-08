from __future__ import annotations

import math

import numpy as np

from execution.metrics_normalizer import (
    confidence_weighted_cumsum,
    compute_normalized_metrics,
    rolling_sharpe,
)


def test_compute_normalized_metrics_basic() -> None:
    returns = [0.01, -0.005, 0.02, 0.015, 0.0]
    metrics = compute_normalized_metrics(returns, target_vol=0.02, annualization=252, min_observations=3)
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    raw_expected = (mean / std) * math.sqrt(252)
    norm_expected = raw_expected * (std / 0.02)

    assert metrics.sample_size == len(returns)
    assert math.isclose(metrics.raw_sharpe, raw_expected, rel_tol=1e-6)
    assert math.isclose(metrics.normalized_sharpe, norm_expected, rel_tol=1e-6)
    assert math.isclose(metrics.volatility_scale, max(0.1, min(3.0, 0.02 / std)), rel_tol=1e-6)


def test_confidence_weighted_cumsum() -> None:
    values = [10.0, -5.0, 2.0]
    confidences = [1.0, None, 0.2]
    result = confidence_weighted_cumsum(values, confidences, default_conf=0.5)
    expected = np.array([10.0, 10.0 - 5.0 * 0.5, 10.0 - 5.0 * 0.5 + 2.0 * 0.2])
    np.testing.assert_allclose(result, expected)


def test_rolling_sharpe_window() -> None:
    pnl = np.linspace(-0.01, 0.02, num=30)
    series = rolling_sharpe(pnl, window=10)
    assert len(series) == len(pnl)
    assert np.isfinite(series.iloc[-1])
    assert series.iloc[-1] != 0.0  # expect populated sharpe for final window
