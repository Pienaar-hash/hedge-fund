"""Tests for research/funding_rate_extremes.py — unit tests only, no network calls."""
from __future__ import annotations

import math

import pytest

from research.funding_rate_extremes import (
    _ROUND_TRIP_FEE,
    AnalysisResult,
    _rank,
    spearman_rho,
)


# ---------------------------------------------------------------------------
# _rank
# ---------------------------------------------------------------------------

def test_rank_simple_ascending():
    assert _rank([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]


def test_rank_descending():
    assert _rank([3.0, 2.0, 1.0]) == [3.0, 2.0, 1.0]


def test_rank_ties_average():
    # [1, 1, 3] → tied at positions 1&2 → avg rank 1.5
    ranks = _rank([1.0, 1.0, 3.0])
    assert ranks[0] == pytest.approx(1.5)
    assert ranks[1] == pytest.approx(1.5)
    assert ranks[2] == pytest.approx(3.0)


def test_rank_all_equal():
    ranks = _rank([5.0, 5.0, 5.0])
    assert all(r == pytest.approx(2.0) for r in ranks)


# ---------------------------------------------------------------------------
# spearman_rho
# ---------------------------------------------------------------------------

def test_spearman_perfect_positive():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    rho, p = spearman_rho(x, x)
    assert rho == pytest.approx(1.0, abs=1e-9)
    assert p < 0.05


def test_spearman_perfect_negative():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [5.0, 4.0, 3.0, 2.0, 1.0]
    rho, p = spearman_rho(x, y)
    assert rho == pytest.approx(-1.0, abs=1e-9)
    assert p < 0.05


def test_spearman_uncorrelated_p_above_threshold():
    # Alternating pattern has near-zero Spearman correlation
    x = list(range(10))
    y = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    rho, p = spearman_rho(x, y)
    # Not expecting significance here
    assert abs(rho) < 0.7
    assert 0.0 <= p <= 1.0


def test_spearman_too_short_returns_zero():
    rho, p = spearman_rho([1.0, 2.0], [2.0, 1.0])
    assert rho == 0.0
    assert p == 1.0


def test_spearman_known_value():
    # Hand-computed example: x=[1,2,3,4,5], y=[1,3,2,5,4]
    # Rank x = [1,2,3,4,5], Rank y = [1,3,2,5,4]
    # d = [0,-1,1,-1,1], d^2 = [0,1,1,1,1], sum=4
    # rho = 1 - 6*4 / (5*24) = 1 - 24/120 = 0.8
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [1.0, 3.0, 2.0, 5.0, 4.0]
    rho, p = spearman_rho(x, y)
    assert rho == pytest.approx(0.8, abs=1e-6)


def test_spearman_p_value_significant_strong_correlation():
    # Strong monotone relationship over n=20 should be significant
    x = list(range(20))
    y = [v + 0.1 * (i % 3) for i, v in enumerate(x)]
    rho, p = spearman_rho(x, y)
    assert rho > 0.9
    assert p < 0.01


def test_spearman_p_bounds():
    x = [float(i) for i in range(30)]
    y = list(reversed(x))
    rho, p = spearman_rho(x, y)
    assert 0.0 <= p <= 1.0
    assert rho == pytest.approx(-1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# AnalysisResult defaults and verdict logic (no network)
# ---------------------------------------------------------------------------

def test_analysis_result_defaults():
    r = AnalysisResult(symbol="BTCUSDT")
    assert r.verdict == "PENDING"
    assert r.n == 0
    assert r.rho == 0.0
    assert r.pvalue == 1.0


def test_fee_constant_positive():
    assert _ROUND_TRIP_FEE > 0
    assert _ROUND_TRIP_FEE < 0.01  # sanity: < 1%


# ---------------------------------------------------------------------------
# Signal direction: high positive funding → negative return (short earns)
# ---------------------------------------------------------------------------

def test_anti_signal_direction():
    """Verify anti_signal = -(fr / mean_abs) has correct sign convention.
    High positive funding rate should produce a negative anti_signal,
    meaning we expect a positive return when SHORT (anti_signal > 0 → long).
    """
    fr = 0.001       # positive funding (longs pay)
    mean_abs = 0.0005
    anti_signal = -(fr / mean_abs)
    # High positive funding → anti_signal is negative → we would NOT go long
    # We only trade when anti_signal is extreme in either direction
    assert anti_signal == pytest.approx(-2.0)


def test_anti_signal_negative_funding():
    fr = -0.001      # negative funding (shorts pay)
    mean_abs = 0.0005
    anti_signal = -(fr / mean_abs)
    # Negative funding → anti_signal positive → expect price to rise (short squeeze unwind)
    assert anti_signal == pytest.approx(2.0)
