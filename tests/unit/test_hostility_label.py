"""Tests for regime_pressure._compute_hostility_label() MODERATE tier."""
import pytest

from execution.regime_pressure import RegimePressureState


def _make_state(changes_7d: int, avg_dwell_h: float) -> RegimePressureState:
    return RegimePressureState(
        regime_changes_7d=changes_7d,
        avg_dwell_time_hours_7d=avg_dwell_h,
    )


class TestHostilityLabel:
    def test_calm(self):
        s = _make_state(changes_7d=7, avg_dwell_h=15.0)  # 1/day, >12h
        assert s._compute_hostility_label() == "CALM"

    def test_moderate(self):
        # 3.14/day, 7.63h → was HOSTILE, now MODERATE
        s = _make_state(changes_7d=22, avg_dwell_h=7.63)
        assert s._compute_hostility_label() == "MODERATE"

    def test_moderate_boundary_low(self):
        # 2/day, 12h → just outside CALM, should be MODERATE
        s = _make_state(changes_7d=14, avg_dwell_h=12.0)
        assert s._compute_hostility_label() == "MODERATE"

    def test_hostile(self):
        # 4.5/day, 5h dwell → HOSTILE
        s = _make_state(changes_7d=32, avg_dwell_h=5.0)
        assert s._compute_hostility_label() == "HOSTILE"

    def test_hostile_short_dwell(self):
        # 3/day but short dwell → HOSTILE from dwell check
        s = _make_state(changes_7d=21, avg_dwell_h=5.5)
        assert s._compute_hostility_label() == "HOSTILE"

    def test_extreme_high_churn(self):
        s = _make_state(changes_7d=42, avg_dwell_h=3.0)  # 6/day, 3h
        assert s._compute_hostility_label() == "EXTREME"

    def test_extreme_short_dwell(self):
        s = _make_state(changes_7d=14, avg_dwell_h=3.5)  # 2/day, 3.5h
        assert s._compute_hostility_label() == "EXTREME"

    def test_calm_zero_changes(self):
        s = _make_state(changes_7d=0, avg_dwell_h=24.0)
        assert s._compute_hostility_label() == "CALM"
