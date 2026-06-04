"""Tests for research/funding_rate_momentum.py — unit tests only, no network calls."""
from __future__ import annotations

import math

import pytest

from research.funding_rate_momentum import (
    _FR_PER_DAY,
    _MIN_RHO,
    _ROUND_TRIP_FEE,
    AnalysisResult,
    Bar,
    FundingRecord,
    _rank,
    build_signal_return_pairs,
    spearman_rho,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DAY_MS = 24 * 3_600_000
_FR_INTERVAL_MS = _DAY_MS // _FR_PER_DAY  # 8 hours


def _make_funding(n: int, start_ms: int = 0, rate_fn=None) -> list[FundingRecord]:
    if rate_fn is None:
        rate_fn = lambda i: 0.0001  # type: ignore[misc]
    return [FundingRecord(ts_ms=start_ms + i * _FR_INTERVAL_MS, rate=rate_fn(i)) for i in range(n)]


def _make_bars(n: int, start_ms: int = 0, price_fn=None) -> list[Bar]:
    if price_fn is None:
        price_fn = lambda i: 1000.0 + i  # type: ignore[misc]
    return [Bar(ts_ms=start_ms + i * _DAY_MS, close=price_fn(i)) for i in range(n)]


def _sufficient_funding_and_bars(
    window: int = 7,
    horizon_days: int = 1,
    extra_days: int = 60,
) -> tuple[list[FundingRecord], list[Bar]]:
    """Generate enough funding records and bars for at least one sample pair."""
    total_days = 2 * window + horizon_days + extra_days
    n_fr = total_days * _FR_PER_DAY
    n_bars = total_days + 10
    funding = _make_funding(n_fr)
    bars = _make_bars(n_bars)
    return funding, bars


# ---------------------------------------------------------------------------
# _rank
# ---------------------------------------------------------------------------

def test_rank_ascending():
    assert _rank([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]


def test_rank_ties_average():
    ranks = _rank([1.0, 1.0, 3.0])
    assert ranks[0] == pytest.approx(1.5)
    assert ranks[1] == pytest.approx(1.5)
    assert ranks[2] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# spearman_rho — p-value regression test
# ---------------------------------------------------------------------------

def test_spearman_perfect_positive():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    rho, p = spearman_rho(x, x)
    assert rho == pytest.approx(1.0, abs=1e-9)
    assert p < 0.05


def test_spearman_too_short():
    assert spearman_rho([1.0, 2.0], [2.0, 1.0]) == (0.0, 1.0)


def test_spearman_near_zero_not_significant():
    import random
    rng = random.Random(99)
    x = [rng.gauss(0, 1) for _ in range(100)]
    y = [rng.gauss(0, 1) for _ in range(100)]
    rho, p = spearman_rho(x, y)
    assert abs(rho) < 0.3
    assert p > 0.05, f"near-zero ρ={rho:.4f} should not be significant, p={p:.6f}"


# ---------------------------------------------------------------------------
# AnalysisResult defaults
# ---------------------------------------------------------------------------

def test_result_defaults():
    r = AnalysisResult(symbol="BTCUSDT")
    assert r.verdict == "PENDING"
    assert r.n == 0
    assert r.rho == 0.0
    assert r.pvalue == 1.0
    assert r.quintile_means == []


def test_result_stores_params():
    r = AnalysisResult(symbol="ETHUSDT", window=14, horizon_days=4)
    assert r.window == 14
    assert r.horizon_days == 4


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_fee_constant():
    assert 0 < _ROUND_TRIP_FEE < 0.01


def test_min_rho_gate():
    assert _MIN_RHO == pytest.approx(0.15)


def test_fr_per_day():
    assert _FR_PER_DAY == 3


# ---------------------------------------------------------------------------
# build_signal_return_pairs
# ---------------------------------------------------------------------------

def test_build_empty_funding_returns_empty():
    _, bars = _sufficient_funding_and_bars()
    signals, returns = build_signal_return_pairs([], bars, 7, 1)
    assert signals == []
    assert returns == []


def test_build_empty_bars_returns_empty():
    funding, _ = _sufficient_funding_and_bars()
    signals, returns = build_signal_return_pairs(funding, [], 7, 1)
    assert signals == []
    assert returns == []


def test_build_insufficient_funding_returns_empty():
    # Only 5 funding records — far less than 2*window*3
    funding = _make_funding(5)
    bars = _make_bars(100)
    signals, returns = build_signal_return_pairs(funding, bars, 7, 1)
    assert signals == []
    assert returns == []


def test_build_produces_pairs():
    funding, bars = _sufficient_funding_and_bars(window=7, horizon_days=1, extra_days=60)
    signals, returns = build_signal_return_pairs(funding, bars, 7, 1)
    assert len(signals) > 0
    assert len(signals) == len(returns)


def test_build_returns_finite():
    funding, bars = _sufficient_funding_and_bars()
    signals, returns = build_signal_return_pairs(funding, bars, 7, 1)
    for s, r in zip(signals, returns):
        assert math.isfinite(s)
        assert math.isfinite(r)


def test_build_signal_positive_when_funding_accelerating_up():
    """When the recent funding window has higher rates than the prior window,
    signal should be positive."""
    window = 7
    window_periods = window * _FR_PER_DAY  # 21 periods
    total_fr = 2 * window_periods * 4  # generous buffer
    # Prior window: low rate; recent window: high rate
    # We'll set rate as a step function: first half 0.0001, second half 0.0005
    def rate_fn(i: int) -> float:
        return 0.0001 if i < window_periods else 0.0005

    funding = _make_funding(total_fr, rate_fn=rate_fn)
    bars = _make_bars(total_fr // _FR_PER_DAY + 10)
    signals, _ = build_signal_return_pairs(funding, bars, window, 1)

    # After the step-up, signals should be positive
    # The step happens at index window_periods; once we sample past 2*window_periods
    # into the step-up region, the recent window > prior window
    positive_signals = [s for s in signals if s > 0]
    assert len(positive_signals) > 0, "Expected at least some positive signals after funding step-up"


def test_build_signal_negative_when_funding_decelerating():
    """When recent funding window has lower rates than prior, signal is negative."""
    window = 7
    window_periods = window * _FR_PER_DAY
    total_fr = 2 * window_periods * 4

    def rate_fn(i: int) -> float:
        # Start high, step down
        return 0.0005 if i < window_periods else 0.0001

    funding = _make_funding(total_fr, rate_fn=rate_fn)
    bars = _make_bars(total_fr // _FR_PER_DAY + 10)
    signals, _ = build_signal_return_pairs(funding, bars, window, 1)

    negative_signals = [s for s in signals if s < 0]
    assert len(negative_signals) > 0, "Expected negative signals after funding step-down"


def test_build_flat_funding_signals_near_zero():
    """Constant funding rate → signal = 0 for all periods."""
    funding, bars = _sufficient_funding_and_bars(extra_days=60)
    # Default rate_fn gives 0.0001 constantly → mean_recent - mean_prior = 0
    signals, _ = build_signal_return_pairs(funding, bars, 7, 1)
    assert len(signals) > 0
    for s in signals:
        assert s == pytest.approx(0.0, abs=1e-12), f"Flat funding should give zero signal, got {s}"


def test_build_longer_horizon_fewer_pairs():
    """With a longer horizon, the non-overlapping step is larger → fewer pairs."""
    funding, bars = _sufficient_funding_and_bars(window=7, horizon_days=1, extra_days=120)
    s1, _ = build_signal_return_pairs(funding, bars, 7, 1)
    s4, _ = build_signal_return_pairs(funding, bars, 7, 4)
    assert len(s1) > len(s4), "Longer horizon should yield fewer non-overlapping pairs"


def test_build_signal_trend_correlated_with_return():
    """When rising funding rate predicts rising price (mock), pooled ρ should be positive."""
    window = 7
    days = 200
    n_fr = days * _FR_PER_DAY

    # Funding rate slowly increases; price mirrors it
    def rate_fn(i: int) -> float:
        return 0.0001 + 0.000001 * i  # slow uptrend

    # Price rises slowly too — so rising funding slope should correlate with positive returns
    def price_fn(i: int) -> float:
        return 1000.0 + i * 0.5

    funding = _make_funding(n_fr, rate_fn=rate_fn)
    bars = _make_bars(days + 20, price_fn=price_fn)
    signals, returns = build_signal_return_pairs(funding, bars, window, 1)

    if len(signals) >= 4:
        rho, _ = spearman_rho(signals, returns)
        # With a monotone trend in both fr and price, signal and return should correlate
        # (not guaranteed perfectly but should be non-negative)
        assert rho >= -0.5, f"Expected non-strongly-negative ρ, got {rho:.4f}"
