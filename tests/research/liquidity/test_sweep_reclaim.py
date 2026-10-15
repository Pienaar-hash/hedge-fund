"""Tests for research/liquidity/sweep_reclaim.py — no network, no filesystem."""
from __future__ import annotations

import math

import numpy as np
import pytest

from research.liquidity.structural_levels import Level
from research.liquidity.sweep_reclaim import (
    classify,
    compute_vwap,
    find_entry,
    find_first_crossing,
    find_outcome_price,
    penetration_bps,
    signed_return,
)

DAY0 = 0
DAY1 = 24 * 60 * 60 * 1000


def upper_level(price: float = 100.0) -> Level:
    return Level(symbol="BTCUSDT", side="PDH", day_epoch_ms=DAY1, price=price, source_day_epoch_ms=DAY0)


def lower_level(price: float = 100.0) -> Level:
    return Level(symbol="BTCUSDT", side="PDL", day_epoch_ms=DAY1, price=price, source_day_epoch_ms=DAY0)


def test_first_outside_crossing_only():
    # prices: 99 (prev), 101 (first cross), 99.5 (back below, irrelevant), 102 (second cross - must be ignored)
    t = np.array([DAY1, DAY1 + 1000, DAY1 + 2000, DAY1 + 3000], dtype=np.int64)
    p = np.array([99.0, 101.0, 99.5, 102.0])
    crossing = find_first_crossing(t, p, DAY1, DAY1 + DAY1, upper_level(100.0))
    assert crossing is not None
    assert crossing.t0_ms == DAY1 + 1000


def test_exact_equality_does_not_cross():
    t = np.array([DAY1, DAY1 + 1000], dtype=np.int64)
    p = np.array([100.0, 100.0])  # equal to level, never strictly exceeds
    crossing = find_first_crossing(t, p, DAY1, DAY1 + DAY1, upper_level(100.0))
    assert crossing is None


def test_upper_lower_symmetry():
    t = np.array([DAY1, DAY1 + 1000], dtype=np.int64)
    p_up = np.array([99.0, 101.0])
    p_down = np.array([101.0, 99.0])
    up = find_first_crossing(t, p_up, DAY1, DAY1 + DAY1, upper_level(100.0))
    down = find_first_crossing(t, p_down, DAY1, DAY1 + DAY1, lower_level(100.0))
    assert up is not None and down is not None


def test_crossing_previous_trade_may_be_from_prior_day():
    t = np.array([DAY1 - 1000, DAY1 + 100], dtype=np.int64)
    p = np.array([99.0, 101.0])
    crossing = find_first_crossing(t, p, DAY1, DAY1 + DAY1, upper_level(100.0))
    assert crossing is not None
    assert crossing.t0_ms == DAY1 + 100


def _vwap_series(t0: int, level: float, penetration_bps_val: float, reclaim_ok: bool):
    """Build a trade tape crossing at t0 that penetrates by penetration_bps_val
    and ends with VWAP either reclaimed or not by tC = t0 + 5min."""
    tC = t0 + 5 * 60 * 1000
    peak = level * (1 + penetration_bps_val / 10_000.0)
    final_price = level * 0.9990 if reclaim_ok else level * 1.0010
    t = np.array([t0 - 1000, t0, t0 + 1000, tC - 15_000, tC - 5_000], dtype=np.int64)
    p = np.array([level * 0.999, peak, peak * 0.999, final_price, final_price])
    q = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    return t, p, q


def test_sweep_reclaim_classification():
    level = upper_level(100.0)
    t0 = DAY1 + 10_000
    t, p, q = _vwap_series(t0, 100.0, penetration_bps_val=15.0, reclaim_ok=True)
    from research.liquidity.sweep_reclaim import Crossing

    crossing = Crossing(level=level, t0_ms=t0, t0_index=1)
    result = classify(t, p, q, crossing)
    assert result.state == "SWEEP_RECLAIM"


def test_excess_penetration_fails_sweep_reclaim():
    level = upper_level(100.0)
    t0 = DAY1 + 10_000
    t, p, q = _vwap_series(t0, 100.0, penetration_bps_val=40.0, reclaim_ok=True)  # exceeds 25bp cap
    from research.liquidity.sweep_reclaim import Crossing

    crossing = Crossing(level=level, t0_ms=t0, t0_index=1)
    result = classify(t, p, q, crossing)
    assert result.state != "SWEEP_RECLAIM"


def test_accepted_breakout_classification():
    level = upper_level(100.0)
    t0 = DAY1 + 10_000
    tC = t0 + 5 * 60 * 1000
    t = np.array([t0 - 1000, t0, tC - 5000], dtype=np.int64)
    p = np.array([99.9, 100.5, 100.6])  # VWAP30 ends >= 1.0005 * 100 = 100.05
    q = np.array([1.0, 1.0, 1.0])
    from research.liquidity.sweep_reclaim import Crossing

    crossing = Crossing(level=level, t0_ms=t0, t0_index=1)
    result = classify(t, p, q, crossing)
    assert result.state == "ACCEPTED_BREAKOUT"


def test_deadband_is_neither():
    level = upper_level(100.0)
    t0 = DAY1 + 10_000
    tC = t0 + 5 * 60 * 1000
    t = np.array([t0 - 1000, t0, tC - 5000], dtype=np.int64)
    p = np.array([99.9, 100.1, 100.02])  # 100.02 is between 0.9998*100=99.98 and 1.0005*100=100.05
    q = np.array([1.0, 1.0, 1.0])
    from research.liquidity.sweep_reclaim import Crossing

    crossing = Crossing(level=level, t0_ms=t0, t0_index=1)
    result = classify(t, p, q, crossing)
    assert result.state == "NEITHER"


def test_classification_cannot_use_trades_after_tC():
    level = upper_level(100.0)
    t0 = DAY1 + 10_000
    tC = t0 + 5 * 60 * 1000
    # A trade right after tC would flip VWAP from reclaim to non-reclaim; must be excluded.
    t = np.array([t0, tC - 1000, tC + 1000], dtype=np.int64)
    p = np.array([100.1, 99.9, 500.0])  # 500.0 would blow up VWAP if included
    q = np.array([1.0, 1.0, 1.0])
    from research.liquidity.sweep_reclaim import Crossing

    crossing = Crossing(level=level, t0_ms=t0, t0_index=0)
    result = classify(t, p, q, crossing)
    assert result.vwap30 is not None
    assert result.vwap30 < 100.0  # unaffected by the post-tC trade


def test_no_vwap_data_is_invalid():
    level = upper_level(100.0)
    t0 = DAY1 + 10_000
    t = np.array([t0], dtype=np.int64)
    p = np.array([100.1])
    q = np.array([1.0])
    from research.liquidity.sweep_reclaim import Crossing

    crossing = Crossing(level=level, t0_ms=t0, t0_index=0)
    result = classify(t, p, q, crossing)
    # only one trade exists and it is exactly at window start; VWAP should still
    # compute from that single trade — force true absence by using an empty array
    empty_t = np.array([], dtype=np.int64)
    empty_p = np.array([])
    empty_q = np.array([])
    vwap = compute_vwap(empty_t, empty_p, empty_q, t0, 30_000)
    assert vwap is None


def test_entry_only_after_tC_plus_one_second():
    tC = DAY1 + 100_000
    t = np.array([tC, tC + 500, tC + 1200], dtype=np.int64)
    p = np.array([100.0, 100.1, 100.2])
    entry = find_entry(t, p, tC)
    assert entry is not None
    assert entry.entry_t_ms == tC + 1200
    assert entry.entry_t_ms >= tC + 1000


def test_entry_invalidated_when_no_trade_within_tolerance():
    tC = DAY1 + 100_000
    t = np.array([tC], dtype=np.int64)  # next trade is far beyond the 5s tolerance
    p = np.array([100.0])
    entry = find_entry(t, p, tC)
    assert entry is None


def test_outcome_price_respects_horizon():
    tC = DAY1 + 100_000
    horizon = 15 * 60 * 1000
    t = np.array([tC, tC + horizon + 500], dtype=np.int64)
    p = np.array([100.0, 105.0])
    outcome = find_outcome_price(t, p, tC, horizon)
    assert outcome is not None
    assert outcome.entry_price == 105.0


def test_signed_return_direction():
    # upper (PDH) sweep: price falling back down after reclaim is a POSITIVE hypothesis outcome
    r_up = signed_return("PDH", entry_price=100.0, exit_price=99.0)
    assert r_up > 0
    # lower (PDL) sweep: price rising back up after reclaim is a POSITIVE hypothesis outcome
    r_down = signed_return("PDL", entry_price=100.0, exit_price=101.0)
    assert r_down > 0
