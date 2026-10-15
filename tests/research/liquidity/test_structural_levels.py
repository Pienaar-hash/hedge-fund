"""Tests for research/liquidity/structural_levels.py — no network, no filesystem."""
from __future__ import annotations

import numpy as np

from research.liquidity.structural_levels import compute_daily_levels, levels_by_usable_day

MS_PER_DAY = 24 * 60 * 60 * 1000


def _day_ms(day_index: int) -> int:
    return day_index * MS_PER_DAY


def test_pdh_pdl_uses_only_completed_prior_day():
    # day 0: high=105, low=95 ; day 1: high=110, low=90 (must not leak into day 1's own level)
    open_time = np.array([
        _day_ms(0) + 0, _day_ms(0) + 60_000,
        _day_ms(1) + 0, _day_ms(1) + 60_000,
    ], dtype=np.int64)
    high = np.array([100.0, 105.0, 110.0, 108.0])
    low = np.array([98.0, 95.0, 90.0, 92.0])

    levels = compute_daily_levels("BTCUSDT", open_time, high, low)
    by_day = levels_by_usable_day(levels)

    day1_levels = by_day[_day_ms(1)]
    assert day1_levels["PDH"].price == 105.0
    assert day1_levels["PDL"].price == 95.0
    # day 1's level must not equal anything derived from day 1's own bars (110/90)
    assert day1_levels["PDH"].price != 110.0
    assert day1_levels["PDL"].price != 90.0


def test_level_usable_day_is_day_after_source_day():
    open_time = np.array([_day_ms(5) + 0], dtype=np.int64)
    high = np.array([200.0])
    low = np.array([190.0])
    levels = compute_daily_levels("ETHUSDT", open_time, high, low)
    assert len(levels) == 2
    for lvl in levels:
        assert lvl.source_day_epoch_ms == _day_ms(5)
        assert lvl.day_epoch_ms == _day_ms(6)


def test_empty_input_produces_no_levels():
    assert compute_daily_levels("SOLUSDT", np.array([], dtype=np.int64), np.array([]), np.array([])) == []


def test_unsorted_input_is_sorted_before_grouping():
    open_time = np.array([_day_ms(1) + 0, _day_ms(0) + 0], dtype=np.int64)
    high = np.array([110.0, 105.0])
    low = np.array([90.0, 95.0])
    levels = compute_daily_levels("BTCUSDT", open_time, high, low)
    by_day = levels_by_usable_day(levels)
    assert by_day[_day_ms(1)]["PDH"].price == 105.0
