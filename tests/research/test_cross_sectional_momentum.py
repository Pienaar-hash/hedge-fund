"""Tests for research/cross_sectional_momentum.py — unit tests only, no network calls."""
from __future__ import annotations

import math

import pytest

from research.cross_sectional_momentum import (
    _MIN_RHO,
    _ROUND_TRIP_FEE,
    _VOL_WINDOW_WEEKS,
    AnalysisResult,
    Bar,
    _rank,
    build_cross_sectional_pairs,
    spearman_rho,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DAY_MS = 24 * 3_600_000
_WEEK_MS = 7 * _DAY_MS


def _make_bars(n: int, start_ms: int = 0, price_fn=None) -> list[Bar]:
    if price_fn is None:
        def price_fn(i: int) -> float:
            return 100.0 + i
    return [Bar(ts_ms=start_ms + i * _DAY_MS, close=price_fn(i)) for i in range(n)]


def _sufficient_n(lookback_weeks: int = 4, horizon_weeks: int = 1, extra_days: int = 80) -> int:
    return (lookback_weeks + _VOL_WINDOW_WEEKS) * 7 + horizon_weeks * 7 + extra_days


def _make_universe(
    n: int,
    start_ms: int = 0,
    price_fns: dict[str, object] | None = None,
) -> dict[str, list[Bar]]:
    """Build a multi-asset bar dict with default slowly-rising prices."""
    if price_fns is None:
        price_fns = {
            "A": lambda i: 100.0 + i * 1.0,
            "B": lambda i: 100.0 + i * 0.5,
            "C": lambda i: 100.0 + i * 0.1,
        }
    return {sym: _make_bars(n, start_ms, fn) for sym, fn in price_fns.items()}


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
# spearman_rho
# ---------------------------------------------------------------------------

def test_spearman_perfect_positive():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    rho, p = spearman_rho(x, x)
    assert rho == pytest.approx(1.0, abs=1e-9)
    assert p < 0.05


def test_spearman_perfect_negative():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    rho, p = spearman_rho(x, [5.0, 4.0, 3.0, 2.0, 1.0])
    assert rho == pytest.approx(-1.0, abs=1e-9)
    assert p < 0.05


def test_spearman_too_short():
    assert spearman_rho([1.0, 2.0], [2.0, 1.0]) == (0.0, 1.0)


def test_spearman_near_zero_not_significant():
    import random
    rng = random.Random(42)
    x = [rng.gauss(0, 1) for _ in range(100)]
    y = [rng.gauss(0, 1) for _ in range(100)]
    rho, p = spearman_rho(x, y)
    assert abs(rho) < 0.3
    assert p > 0.05, f"near-zero ρ={rho:.4f} should not be significant, p={p:.6f}"


def test_spearman_known_value():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [1.0, 3.0, 2.0, 5.0, 4.0]
    rho, _ = spearman_rho(x, y)
    assert rho == pytest.approx(0.8, abs=1e-6)


# ---------------------------------------------------------------------------
# AnalysisResult defaults
# ---------------------------------------------------------------------------

def test_result_defaults():
    r = AnalysisResult()
    assert r.verdict == "PENDING"
    assert r.n_total == 0
    assert r.rho == 0.0
    assert r.pvalue == 1.0
    assert r.tercile_means == []
    assert r.criteria == {}


def test_result_stores_params():
    r = AnalysisResult(symbols=["BTC", "ETH"], lookback_weeks=8, horizon_weeks=2)
    assert r.lookback_weeks == 8
    assert r.horizon_weeks == 2


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_fee_constant():
    assert 0 < _ROUND_TRIP_FEE < 0.01


def test_min_rho_gate():
    assert _MIN_RHO == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# build_cross_sectional_pairs
# ---------------------------------------------------------------------------

def test_build_empty_universe():
    signals, returns, labels = build_cross_sectional_pairs({}, 4, 1)
    assert signals == []
    assert returns == []
    assert labels == []


def test_build_single_asset_returns_empty():
    n = _sufficient_n()
    universe = {"A": _make_bars(n)}
    signals, returns, labels = build_cross_sectional_pairs(universe, 4, 1)
    assert signals == []
    assert returns == []


def test_build_insufficient_bars_returns_empty():
    universe = {
        "A": _make_bars(30),
        "B": _make_bars(30),
    }
    signals, returns, labels = build_cross_sectional_pairs(universe, 4, 1)
    assert signals == []
    assert returns == []


def test_build_produces_pairs_for_multi_asset():
    n = _sufficient_n(extra_days=100)
    universe = _make_universe(n)
    signals, returns, labels = build_cross_sectional_pairs(universe, 4, 1)
    assert len(signals) > 0
    assert len(signals) == len(returns) == len(labels)


def test_build_pairs_multiple_of_n_assets():
    """Total pairs must be a multiple of the number of assets (balanced cross-section)."""
    n = _sufficient_n(extra_days=100)
    universe = _make_universe(n)
    signals, returns, labels = build_cross_sectional_pairs(universe, 4, 1)
    n_assets = len(universe)
    assert len(signals) % n_assets == 0


def test_build_signal_range():
    """Signal values must be in [0, 1] — they are normalised ranks."""
    n = _sufficient_n(extra_days=100)
    universe = _make_universe(n)
    signals, _, _ = build_cross_sectional_pairs(universe, 4, 1)
    assert all(0.0 <= s <= 1.0 for s in signals)


def test_build_all_labels_present():
    """Every asset must appear in labels (balanced cross-section)."""
    n = _sufficient_n(extra_days=100)
    universe = _make_universe(n)
    _, _, labels = build_cross_sectional_pairs(universe, 4, 1)
    assert set(labels) == set(universe.keys())


def test_build_cross_sectional_ranks_sum_to_constant():
    """At each sample time, the n signals sum to (n_assets − 1)/2 * n_assets / n_assets = 0.5*(n-1)/n.
    For 3 assets with signals 0, 0.5, 1: sum = 1.5, mean = 0.5."""
    n = _sufficient_n(extra_days=100)
    universe = _make_universe(n, price_fns={
        "A": lambda i: 100.0 + i * 2.0,
        "B": lambda i: 100.0 + i * 1.0,
        "C": lambda i: 100.0 + i * 0.1,
    })
    signals, _, _ = build_cross_sectional_pairs(universe, 4, 1)
    n_assets = len(universe)
    n_periods = len(signals) // n_assets
    for period in range(n_periods):
        period_signals = signals[period * n_assets:(period + 1) * n_assets]
        assert sum(period_signals) == pytest.approx(1.5, abs=1e-9)  # 0 + 0.5 + 1 = 1.5


def test_build_top_signal_is_best_performer():
    """The asset with the highest past return must get signal=1.0."""
    n = _sufficient_n(extra_days=100)
    # A rises fastest → should always be ranked top (signal=1.0)
    universe = _make_universe(n, price_fns={
        "A": lambda i: 100.0 + i * 3.0,
        "B": lambda i: 100.0 + i * 1.0,
        "C": lambda i: 100.0 + i * 0.1,
    })
    signals, _, labels = build_cross_sectional_pairs(universe, 4, 1)
    n_assets = len(universe)
    for idx, (s, sym) in enumerate(zip(signals, labels)):
        if sym == "A":
            assert s == pytest.approx(1.0), f"A should be top-ranked at period {idx // n_assets}"


def test_build_returns_finite():
    n = _sufficient_n(extra_days=100)
    universe = _make_universe(n)
    signals, returns, _ = build_cross_sectional_pairs(universe, 4, 1)
    for s, r in zip(signals, returns):
        assert math.isfinite(s)
        assert math.isfinite(r)


def test_build_stable_ranking_produces_monotone_returns():
    """When asset A always beats B beats C in past returns, and the same
    ordering holds for forward returns, pooled ρ should be positive."""
    n = _sufficient_n(extra_days=150)
    universe = _make_universe(n, price_fns={
        "A": lambda i: 100.0 * (1.001 ** i),  # fastest compound growth
        "B": lambda i: 100.0 * (1.0005 ** i),
        "C": lambda i: 100.0 * (1.0001 ** i),
    })
    signals, returns, _ = build_cross_sectional_pairs(universe, 4, 1)
    rho, _ = spearman_rho(signals, returns)
    assert rho > 0, "Consistently faster-growing asset should yield positive pooled ρ"
