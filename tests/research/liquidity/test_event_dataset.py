"""Tests for research/liquidity/event_dataset.py — no network, no filesystem."""
from __future__ import annotations

import numpy as np
import pytest

from research.liquidity.event_dataset import (
    Event,
    MatchFeatures,
    _event_id,
    build_events_for_symbol,
    match_events,
)
from research.liquidity.structural_levels import Level

DAY = 24 * 60 * 60 * 1000


def _level(side: str, day_epoch_ms: int, price: float) -> Level:
    return Level(symbol="BTCUSDT", side=side, day_epoch_ms=day_epoch_ms, price=price, source_day_epoch_ms=day_epoch_ms - DAY)


def test_event_id_is_deterministic_and_distinct():
    a1 = _event_id("BTCUSDT", "PDH", 1234)
    a2 = _event_id("BTCUSDT", "PDH", 1234)
    b = _event_id("BTCUSDT", "PDH", 1235)
    c = _event_id("BTCUSDT", "PDL", 1234)
    assert a1 == a2
    assert a1 != b
    assert a1 != c


def test_overlap_embargo_rejects_second_candidate_within_60min():
    day1 = DAY
    level_h = _level("PDH", day1, 100.0)
    level_l = _level("PDL", day1, 90.0)
    levels = [level_h, level_l]

    t0_a = day1 + 10_000
    tC_a = t0_a + 5 * 60 * 1000
    # second crossing (PDL) happens only 10 minutes after the first event's tC -> must be embargoed
    t0_b = tC_a + 10 * 60 * 1000

    # Build a trade tape: cross PDH at t0_a (sweep+reclaim), then cross PDL at t0_b.
    times = [day1 - 1000, t0_a]
    prices = [99.0, 101.0]
    # fill VWAP window with a reclaim back under 100
    times += [tC_a - 5000]
    prices += [99.8]
    # now the PDL candidate crossing, still within 60 minutes of tC_a
    times += [t0_b - 1000, t0_b]
    prices += [90.5, 89.5]
    times += [t0_b + 5 * 60 * 1000 - 5000]
    prices += [89.0]

    t = np.array(times, dtype=np.int64)
    p = np.array(prices, dtype=np.float64)
    q = np.ones_like(p)
    kt = np.array([day1 - DAY], dtype=np.int64)
    kh = np.array([100.0])
    kl = np.array([90.0])
    kc = np.array([95.0])

    events = build_events_for_symbol("BTCUSDT", t, p, q, kt, kh, kl, kc, levels)
    reasons = {e.event_id: e.reject_reason for e in events if e.state == "REJECTED"}
    assert "overlap_embargo" in reasons.values()


def test_no_embargo_when_gap_exceeds_60min():
    day1 = DAY
    level_h = _level("PDH", day1, 100.0)
    level_l = _level("PDL", day1, 90.0)
    levels = [level_h, level_l]

    t0_a = day1 + 10_000
    tC_a = t0_a + 5 * 60 * 1000
    t0_b = tC_a + 61 * 60 * 1000  # safely beyond 60-minute embargo

    times = [day1 - 1000, t0_a, tC_a - 5000, t0_b - 1000, t0_b, t0_b + 5 * 60 * 1000 - 5000]
    prices = [99.0, 101.0, 99.8, 90.5, 89.5, 89.0]

    t = np.array(times, dtype=np.int64)
    p = np.array(prices, dtype=np.float64)
    q = np.ones_like(p)
    kt = np.array([day1 - DAY], dtype=np.int64)
    kh = np.array([100.0])
    kl = np.array([90.0])
    kc = np.array([95.0])

    events = build_events_for_symbol("BTCUSDT", t, p, q, kt, kh, kl, kc, levels)
    reasons = [e.reject_reason for e in events if e.state == "REJECTED"]
    assert "overlap_embargo" not in reasons


def _mk_event(event_id: str, symbol: str, side: str, t0_ms: int, state: str, rv: float, vol: float, quarter: str, block: int) -> Event:
    return Event(
        event_id=event_id, symbol=symbol, side=side, level_price=100.0, t0_ms=t0_ms, tC_ms=t0_ms + 300_000,
        state=state, penetration_bps=10.0, vwap30=99.9,
        features=MatchFeatures(realized_vol_30m=rv, quote_notional_30m=vol, utc_4h_block=block, quarter=quarter, symbol=symbol, side=side),
    )


def test_matching_picks_nearest_control_no_replacement():
    sweep = _mk_event("s1", "BTCUSDT", "PDH", 1000, "SWEEP_RECLAIM", rv=1.0, vol=1000.0, quarter="2023Q1", block=0)
    close_control = _mk_event("c_close", "BTCUSDT", "PDH", 500, "ACCEPTED_BREAKOUT", rv=1.05, vol=1050.0, quarter="2023Q1", block=0)
    far_control = _mk_event("c_far", "BTCUSDT", "PDH", 600, "ACCEPTED_BREAKOUT", rv=1.8, vol=1900.0, quarter="2023Q1", block=0)

    pairs = match_events([sweep, close_control, far_control])
    assert len(pairs) == 1
    assert pairs[0].sweep_event_id == "s1"
    assert pairs[0].control_event_id == "c_close"


def test_matching_ineligible_control_rejected_by_symbol_side_quarter_block():
    sweep = _mk_event("s1", "BTCUSDT", "PDH", 1000, "SWEEP_RECLAIM", rv=1.0, vol=1000.0, quarter="2023Q1", block=0)
    wrong_symbol = _mk_event("c1", "ETHUSDT", "PDH", 500, "ACCEPTED_BREAKOUT", rv=1.0, vol=1000.0, quarter="2023Q1", block=0)
    wrong_side = _mk_event("c2", "BTCUSDT", "PDL", 500, "ACCEPTED_BREAKOUT", rv=1.0, vol=1000.0, quarter="2023Q1", block=0)
    wrong_quarter = _mk_event("c3", "BTCUSDT", "PDH", 500, "ACCEPTED_BREAKOUT", rv=1.0, vol=1000.0, quarter="2023Q2", block=0)
    wrong_block = _mk_event("c4", "BTCUSDT", "PDH", 500, "ACCEPTED_BREAKOUT", rv=1.0, vol=1000.0, quarter="2023Q1", block=3)

    pairs = match_events([sweep, wrong_symbol, wrong_side, wrong_quarter, wrong_block])
    assert len(pairs) == 0


def test_matching_no_replacement_second_sweep_gets_next_best():
    sweep1 = _mk_event("s1", "BTCUSDT", "PDH", 1000, "SWEEP_RECLAIM", rv=1.0, vol=1000.0, quarter="2023Q1", block=0)
    sweep2 = _mk_event("s2", "BTCUSDT", "PDH", 2000, "SWEEP_RECLAIM", rv=1.0, vol=1000.0, quarter="2023Q1", block=0)
    control_a = _mk_event("cA", "BTCUSDT", "PDH", 500, "ACCEPTED_BREAKOUT", rv=1.0, vol=1000.0, quarter="2023Q1", block=0)
    control_b = _mk_event("cB", "BTCUSDT", "PDH", 600, "ACCEPTED_BREAKOUT", rv=1.1, vol=1100.0, quarter="2023Q1", block=0)

    pairs = match_events([sweep1, sweep2, control_a, control_b])
    assigned = {p.sweep_event_id: p.control_event_id for p in pairs}
    assert assigned["s1"] == "cA"  # earlier sweep gets the exact match first (processed chronologically)
    assert assigned["s2"] == "cB"  # second sweep must not reuse cA


def test_matching_outside_factor_two_band_is_ineligible():
    sweep = _mk_event("s1", "BTCUSDT", "PDH", 1000, "SWEEP_RECLAIM", rv=1.0, vol=1000.0, quarter="2023Q1", block=0)
    too_far = _mk_event("c1", "BTCUSDT", "PDH", 500, "ACCEPTED_BREAKOUT", rv=3.0, vol=1000.0, quarter="2023Q1", block=0)
    pairs = match_events([sweep, too_far])
    assert len(pairs) == 0
