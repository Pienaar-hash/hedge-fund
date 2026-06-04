"""Tests for research/oi_momentum.py — unit tests only, no network calls."""
from __future__ import annotations

import math

import pytest

from research.oi_momentum import (
    _MIN_RHO,
    _ROUND_TRIP_FEE,
    AnalysisResult,
    Bar,
    OIRecord,
    _rank,
    build_signal_return_pairs,
    spearman_rho,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DAY_MS = 24 * 3_600_000


def _make_oi(n: int, start_ms: int = 0, oi_fn=None) -> list[OIRecord]:
    if oi_fn is None:
        oi_fn = lambda i: 1_000_000.0 + i * 1000  # type: ignore[misc]
    return [OIRecord(ts_ms=start_ms + i * _DAY_MS, oi=oi_fn(i)) for i in range(n)]


def _make_bars(n: int, start_ms: int = 0, price_fn=None) -> list[Bar]:
    if price_fn is None:
        price_fn = lambda i: 1000.0 + i  # type: ignore[misc]
    return [Bar(ts_ms=start_ms + i * _DAY_MS, close=price_fn(i)) for i in range(n)]


def _sufficient(window: int = 7, horizon_days: int = 1, extra_days: int = 60):
    total = window + horizon_days + extra_days
    return _make_oi(total), _make_bars(total)


# ---------------------------------------------------------------------------
# _rank
# ---------------------------------------------------------------------------

def test_rank_ascending():
    assert _rank([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]


def test_rank_ties_average():
    ranks = _rank([5.0, 5.0, 9.0])
    assert ranks[0] == pytest.approx(1.5)
    assert ranks[1] == pytest.approx(1.5)
    assert ranks[2] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# spearman_rho — p-value regression
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
    rng = random.Random(7)
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


# ---------------------------------------------------------------------------
# build_signal_return_pairs
# ---------------------------------------------------------------------------

def test_build_empty_oi_returns_empty():
    _, bars = _sufficient()
    signals, returns = build_signal_return_pairs([], bars, 7, 1)
    assert signals == []
    assert returns == []


def test_build_empty_bars_returns_empty():
    oi, _ = _sufficient()
    signals, returns = build_signal_return_pairs(oi, [], 7, 1)
    assert signals == []
    assert returns == []


def test_build_insufficient_oi_returns_empty():
    # Only 3 OI records — less than window=7 days of history
    oi = _make_oi(3)
    bars = _make_bars(100)
    signals, returns = build_signal_return_pairs(oi, bars, 7, 1)
    assert signals == []
    assert returns == []


def test_build_produces_pairs():
    oi, bars = _sufficient(window=7, horizon_days=1, extra_days=60)
    signals, returns = build_signal_return_pairs(oi, bars, 7, 1)
    assert len(signals) > 0
    assert len(signals) == len(returns)


def test_build_returns_finite():
    oi, bars = _sufficient()
    signals, returns = build_signal_return_pairs(oi, bars, 7, 1)
    for s, r in zip(signals, returns):
        assert math.isfinite(s)
        assert math.isfinite(r)


def test_signal_positive_when_oi_grows_with_uptrend():
    """OI growing + price rising → signal > 0."""
    n = 100
    # OI growing, price rising
    oi = _make_oi(n, oi_fn=lambda i: 1_000_000 + i * 1000)
    bars = _make_bars(n, price_fn=lambda i: 1000.0 + i)
    signals, _ = build_signal_return_pairs(oi, bars, 7, 1)
    assert len(signals) > 0
    assert all(s > 0 for s in signals), "Growing OI + rising price → positive signal"


def test_signal_negative_when_oi_grows_with_downtrend():
    """OI growing + price falling → signal < 0."""
    n = 100
    oi = _make_oi(n, oi_fn=lambda i: 1_000_000 + i * 1000)
    bars = _make_bars(n, price_fn=lambda i: max(1.0, 1000.0 - i))
    signals, _ = build_signal_return_pairs(oi, bars, 7, 1)
    assert len(signals) > 0
    assert all(s < 0 for s in signals), "Growing OI + falling price → negative signal"


def test_signal_negative_when_oi_shrinks_with_uptrend():
    """OI shrinking + price rising → negative signal (exhaustion)."""
    n = 100
    # OI shrinking (positions closing), price still rising
    oi = _make_oi(n, oi_fn=lambda i: max(1.0, 1_000_000 - i * 1000))
    bars = _make_bars(n, price_fn=lambda i: 1000.0 + i)
    signals, _ = build_signal_return_pairs(oi, bars, 7, 1)
    assert len(signals) > 0
    assert all(s < 0 for s in signals), "Shrinking OI + rising price → negative signal"


def test_signal_positive_when_oi_shrinks_with_downtrend():
    """OI shrinking + price falling → positive signal (shorts covering)."""
    n = 100
    oi = _make_oi(n, oi_fn=lambda i: max(1.0, 1_000_000 - i * 1000))
    bars = _make_bars(n, price_fn=lambda i: max(1.0, 1000.0 - i))
    signals, _ = build_signal_return_pairs(oi, bars, 7, 1)
    assert len(signals) > 0
    assert all(s > 0 for s in signals), "Shrinking OI + falling price → positive signal"


def test_signal_zero_when_price_unchanged():
    """price_direction = 0 when close_t == close_{t-window} → signal = 0."""
    n = 100
    oi = _make_oi(n, oi_fn=lambda i: 1_000_000 + i * 1000)
    bars = _make_bars(n, price_fn=lambda i: 1000.0)  # flat price
    signals, _ = build_signal_return_pairs(oi, bars, 7, 1)
    assert len(signals) > 0
    assert all(s == pytest.approx(0.0) for s in signals), "Flat price → zero signal"


def test_longer_horizon_fewer_pairs():
    oi, bars = _sufficient(window=7, horizon_days=1, extra_days=120)
    s1, _ = build_signal_return_pairs(oi, bars, 7, 1)
    s4, _ = build_signal_return_pairs(oi, bars, 7, 4)
    assert len(s1) > len(s4)
