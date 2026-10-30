"""Offline checks for candle timing, boundaries and the fixed screening decision."""
import csv
import json
import math

import pytest

from research.liquidity import candle_screen as screen


def day(start=screen.START):
    return [(start + i * screen.MINUTE, 100., 101., 99., 100.) for i in range(1440)]


def rejection(start=screen.START, index=10, lower=False):
    rows = day(start)
    t = rows[index][0]
    rows[index] = (t, 100., 101. if lower else 106., 94. if lower else 99., 100.)
    return rows


def test_next_minute_entry_and_fifteen_minute_exit_cost_once():
    rows = rejection()
    rows[15] = (rows[15][0], 104., 104., 103., 103.)
    rows[30] = (rows[30][0], 103., 104., 102., 103.)
    events, _ = screen.screen_day("BTCUSDT", rows, (105., 95.))
    event = events[0]
    assert event["state"] == "rejection"
    assert event["cross_bar_ms"] == rows[10][0]
    assert event["classification_ms"] == rows[15][0] == event["entry_ms"]
    assert event["exit_ms"] - event["entry_ms"] == 15 * screen.MINUTE
    assert event["gross_bps"] == pytest.approx(-10000 * math.log(103 / 104))
    assert event["net_bps"] == pytest.approx(event["gross_bps"] - 15)


@pytest.mark.parametrize("lower,close,state", [(False, 104., "rejection"), (False, 106., "acceptance"),
                                               (True, 96., "rejection"), (True, 94., "acceptance")])
def test_symmetric_classification_uses_fifth_close(lower, close, state):
    rows = rejection(lower=lower)
    rows[14] = (*rows[14][:4], close)
    # A later close must not change the fixed-time class.
    rows[15] = (*rows[15][:4], 1000.)
    events, _ = screen.screen_day("BTCUSDT", rows, (105., 95.))
    assert events[0]["state"] == state


def test_first_penetration_only_and_equality_excluded():
    rows = rejection()
    rows[14] = (*rows[14][:4], 105.)
    rows[100] = (rows[100][0], 100., 106., 99., 100.)
    events, skips = screen.screen_day("BTCUSDT", rows, (105., 95.))
    assert not events and skips["equality_deadband"] == 1


def test_same_bar_both_levels_is_ambiguous():
    rows = rejection()
    rows[10] = (rows[10][0], 100., 106., 94., 100.)
    events, skips = screen.screen_day("BTCUSDT", rows, (105., 95.))
    assert not events and skips["ambiguous_double_penetration"] == 2


def test_incomplete_days_and_unavailable_levels_excluded():
    assert screen.screen_day("BTCUSDT", rejection()[:-1], (105., 95.))[1]["incomplete_crossing_day"] == 1
    assert screen.screen_day("BTCUSDT", rejection(), None)[1]["unavailable_previous_day"] == 1


def test_cross_midnight_and_no_2025_context():
    rows = rejection(index=1438)
    events, _ = screen.screen_day("BTCUSDT", rows, (105., 95.), day(screen.START + screen.DAY))
    assert events[0]["entry_ms"] == screen.START + screen.DAY + 3 * screen.MINUTE
    last = rejection(screen.END - screen.DAY, index=1438)
    events, skips = screen.screen_day("BTCUSDT", last, (105., 95.), day(screen.END))
    assert not events and skips["unavailable_exit_context"] == 1


def test_missing_next_day_minute_invalidates_event_window():
    rows = rejection(index=1438)
    following = day(screen.START + screen.DAY)
    del following[5]
    events, skips = screen.screen_day("BTCUSDT", rows, (105., 95.), following)
    assert not events and skips["gap_in_event_window"] == 1


def test_previous_completed_day_levels_in_run(tmp_path, monkeypatch):
    prior = day(screen.START - screen.DAY)
    prior[0] = (prior[0][0], 100., 105., 95., 100.)
    current = rejection()

    def offline_rows(*args):
        yield from prior
        yield from current

    monkeypatch.setattr(screen, "candle_rows", offline_rows)
    output = tmp_path / "screen"
    screen.run(output)
    with (output / "events.csv").open() as handle:
        events = list(csv.DictReader(handle))
    assert len(events) == 3  # once per symbol, upper only
    assert all(float(e["level"]) == 105. for e in events)
    results = json.loads((output / "results.json").read_text())
    assert results["held_out_2025"] == "NOT_DOWNLOADED_OR_INSPECTED"
    assert results["decision"] == "REJECT_SCREEN_MECHANISM"


def test_fixed_decision_and_difference_cost_cancellation():
    events = [{"symbol": s, "year": y, "state": state, "gross_bps": value, "net_bps": value - 15}
              for s in screen.SYMBOLS for y in (2022, 2023, 2024)
              for state, value in (("rejection", 25.), ("acceptance", 5.)) for _ in range(50)]
    summary = screen.summarize(events)
    assert summary["pooled"]["median_difference_bps"] == 20.
    assert summary["pooled"]["rejection"]["median_net_bps"] == 10.
    assert screen.decide(summary)["decision"] == "JUSTIFY_MORE_PRECISE_VALIDATION"
    for e in events:
        if e["state"] == "rejection":
            e["gross_bps"], e["net_bps"] = 14., -1.
    assert screen.decide(screen.summarize(events))["decision"] == "REJECT_SCREEN_MECHANISM"
