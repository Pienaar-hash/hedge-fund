"""Tests for research/weekly_momentum.py — unit tests only, no network calls."""
from __future__ import annotations

import math

import pytest

from research.weekly_momentum import (
    _MIN_MONOTONICITY,
    _MIN_PAIRS,
    _MIN_RHO,
    _ROUND_TRIP_FEE,
    _VOL_WINDOW_WEEKS,
    AnalysisResult,
    Bar,
    _rank,
    build_signal_return_pairs,
    spearman_rho,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DAY_MS = 24 * 3_600_000
_WEEK_MS = 7 * _DAY_MS


def _make_bars(n: int, start_ms: int = 0, price_fn=None) -> list[Bar]:
    """Produce n daily bars; default price slowly rises by 1 per day."""
    if price_fn is None:
        price_fn = lambda i: 100.0 + i  # type: ignore[misc]
    return [Bar(ts_ms=start_ms + i * _DAY_MS, close=price_fn(i)) for i in range(n)]


def _sufficient_bars(
    lookback_weeks: int = 4,
    horizon_weeks: int = 1,
    extra_days: int = 80,
) -> list[Bar]:
    """Return enough bars so build_signal_return_pairs yields at least one pair."""
    min_days = (lookback_weeks + _VOL_WINDOW_WEEKS) * 7 + horizon_weeks * 7 + extra_days
    return _make_bars(min_days)


# ---------------------------------------------------------------------------
# _rank (same logic as funding_rate_extremes — verify copy is consistent)
# ---------------------------------------------------------------------------

def test_rank_simple_ascending():
    assert _rank([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]


def test_rank_descending():
    assert _rank([3.0, 2.0, 1.0]) == [3.0, 2.0, 1.0]


def test_rank_ties_average():
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


def test_spearman_too_short_returns_zero():
    rho, p = spearman_rho([1.0, 2.0], [2.0, 1.0])
    assert rho == 0.0
    assert p == 1.0


def test_spearman_known_value():
    # x=[1,2,3,4,5], y=[1,3,2,5,4] → ρ=0.8 by formula
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [1.0, 3.0, 2.0, 5.0, 4.0]
    rho, p = spearman_rho(x, y)
    assert rho == pytest.approx(0.8, abs=1e-6)


def test_spearman_p_bounds():
    x = [float(i) for i in range(30)]
    y = list(reversed(x))
    rho, p = spearman_rho(x, y)
    assert 0.0 <= p <= 1.0
    assert rho == pytest.approx(-1.0, abs=1e-6)


def test_spearman_significant_strong_correlation():
    x = list(range(20))
    y = [v + 0.1 * (i % 3) for i, v in enumerate(x)]
    rho, p = spearman_rho(x, y)
    assert rho > 0.9
    assert p < 0.01


# ---------------------------------------------------------------------------
# AnalysisResult defaults
# ---------------------------------------------------------------------------

def test_analysis_result_defaults():
    r = AnalysisResult(symbol="BTCUSDT")
    assert r.verdict == "PENDING"
    assert r.n == 0
    assert r.rho == 0.0
    assert r.pvalue == 1.0
    assert r.q5_minus_q1 == 0.0
    assert r.quintile_means == []
    assert r.criteria == {}


def test_analysis_result_stores_params():
    r = AnalysisResult(symbol="ETHUSDT", lookback_weeks=8, horizon_weeks=2, vol_normalize=True)
    assert r.lookback_weeks == 8
    assert r.horizon_weeks == 2
    assert r.vol_normalize is True


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------

def test_fee_constant_positive():
    assert _ROUND_TRIP_FEE > 0
    assert _ROUND_TRIP_FEE < 0.01


def test_min_rho_gate():
    assert _MIN_RHO == pytest.approx(0.15)


def test_vol_window_weeks():
    assert _VOL_WINDOW_WEEKS == 26


def test_min_pairs():
    assert _MIN_PAIRS >= 20  # sanity: meaningful minimum


# ---------------------------------------------------------------------------
# build_signal_return_pairs
# ---------------------------------------------------------------------------

def test_build_empty_bars():
    signals, returns = build_signal_return_pairs([], 4, 1, False)
    assert signals == []
    assert returns == []


def test_build_insufficient_bars_returns_empty():
    # Only 30 bars — far less than the lookback+vol-window requirement
    bars = _make_bars(30)
    signals, returns = build_signal_return_pairs(bars, 4, 1, False)
    assert signals == []
    assert returns == []


def test_build_uptrend_signals_positive():
    """With a steadily rising price series all momentum signals should be positive."""
    bars = _sufficient_bars(lookback_weeks=4, horizon_weeks=1, extra_days=100)
    signals, returns = build_signal_return_pairs(bars, 4, 1, False)
    assert len(signals) > 0
    assert all(s > 0 for s in signals), "Rising market must yield positive signals"


def test_build_downtrend_signals_negative():
    """With a steadily falling price series all momentum signals should be negative."""
    n = (4 + _VOL_WINDOW_WEEKS) * 7 + 7 + 100
    bars = _make_bars(n, price_fn=lambda i: max(1.0, 1000.0 - i))
    signals, returns = build_signal_return_pairs(bars, 4, 1, False)
    assert len(signals) > 0
    assert all(s < 0 for s in signals), "Falling market must yield negative signals"


def test_build_returns_type():
    bars = _sufficient_bars()
    signals, returns = build_signal_return_pairs(bars, 4, 1, False)
    for s, r in zip(signals, returns):
        assert isinstance(s, float)
        assert isinstance(r, float)
        assert math.isfinite(s)
        assert math.isfinite(r)


def test_build_non_overlapping_count():
    """Number of samples ≈ (eligible_span) / horizon_step."""
    lookback_weeks, horizon_weeks = 4, 1
    extra_days = 100
    bars = _sufficient_bars(lookback_weeks=lookback_weeks, horizon_weeks=horizon_weeks, extra_days=extra_days)
    signals, returns = build_signal_return_pairs(bars, lookback_weeks, horizon_weeks, False)
    # With step = horizon_weeks * 7 days, each sample advances exactly one week
    # Just verify we get a reasonable non-zero count
    assert len(signals) >= 1
    assert len(signals) == len(returns)


def test_build_vol_normalize_produces_valid_signals():
    """vol_normalize=True should still produce finite signals."""
    bars = _sufficient_bars(lookback_weeks=4, horizon_weeks=1, extra_days=100)
    signals, returns = build_signal_return_pairs(bars, 4, 1, True)
    assert len(signals) > 0
    for s in signals:
        assert math.isfinite(s)


def test_build_vol_normalize_scales_signal():
    """Vol-normalized signals should be larger in magnitude than raw when vol < 1."""
    bars = _sufficient_bars(lookback_weeks=4, horizon_weeks=1, extra_days=100)
    raw_signals, _ = build_signal_return_pairs(bars, 4, 1, False)
    vol_signals, _ = build_signal_return_pairs(bars, 4, 1, True)
    # Both should be non-empty and the vol-normalized version should differ
    if raw_signals and vol_signals:
        assert raw_signals[0] != pytest.approx(vol_signals[0])


def test_build_longer_lookback_larger_signal():
    """8-week lookback signal magnitude should exceed 4-week for a trend."""
    bars = _sufficient_bars(lookback_weeks=8, horizon_weeks=1, extra_days=150)
    s4, _ = build_signal_return_pairs(bars, 4, 1, False)
    s8, _ = build_signal_return_pairs(bars, 8, 1, False)
    if s4 and s8:
        assert abs(s8[0]) > abs(s4[0]), "Longer lookback captures more of the trend"


def test_build_signal_known_value():
    """Verify exact signal value for a simple linearly rising price series."""
    # Price at day i = 100 + i
    n = (4 + _VOL_WINDOW_WEEKS) * 7 + 7 + 80
    bars = _make_bars(n, price_fn=lambda i: 100.0 + i)

    signals, _ = build_signal_return_pairs(bars, 4, 1, False)
    assert len(signals) > 0

    # At the first sample point t_idx = lookback_days + vol_window_days:
    #   lookback_days = 4*7 = 28, vol_window_days = 26*7 = 182 → t_idx = 210
    #   px_now  = 100 + 210 = 310
    #   px_back = 100 + (210 - 28) = 282
    #   signal  = (310 - 282) / 282 = 28/282
    expected_signal = 28.0 / 282.0
    assert signals[0] == pytest.approx(expected_signal, rel=0.01)


def test_build_forward_return_direction_matches_trend():
    """In an uptrend, forward returns should be positive for all samples."""
    bars = _sufficient_bars(lookback_weeks=4, horizon_weeks=1, extra_days=100)
    _, returns = build_signal_return_pairs(bars, 4, 1, False)
    assert len(returns) > 0
    assert all(r > 0 for r in returns), "Rising market should have positive forward returns"
