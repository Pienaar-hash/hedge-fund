"""Tests for dashboard/components/equity_curve.py."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from dashboard.components.equity_curve import (
    _build_equity_svg,
    _downsample,
    _load_nav_series,
)


@pytest.fixture()
def nav_log(tmp_path):
    """Create a minimal nav_log.json with 50 entries over 24h."""
    now = time.time()
    entries = []
    for i in range(50):
        t = now - 86400 + (i * 86400 / 49)
        nav = 10000 + i * 2  # steady uptrend
        entries.append({"nav": nav, "t": t, "unrealized_pnl": 0})
    path = tmp_path / "nav_log.json"
    path.write_text(json.dumps(entries))
    return path, entries


class TestLoadNavSeries:

    def test_loads_valid(self, nav_log):
        path, expected = nav_log
        result = _load_nav_series(path)
        assert len(result) == 50
        assert result[0]["nav"] == 10000

    def test_missing_file(self, tmp_path):
        result = _load_nav_series(tmp_path / "nonexistent.json")
        assert result == []

    def test_filters_incomplete(self, tmp_path):
        path = tmp_path / "nav_log.json"
        path.write_text(json.dumps([
            {"nav": 100, "t": 1000},
            {"nav": None, "t": 2000},
            {"t": 3000},
            {"nav": 200, "t": 4000},
        ]))
        result = _load_nav_series(path)
        assert len(result) == 2


class TestDownsample:

    def test_noop_if_small(self):
        entries = [{"nav": i, "t": i} for i in range(10)]
        result = _downsample(entries, max_points=100)
        assert len(result) == 10

    def test_downsamples_large(self):
        entries = [{"nav": i, "t": i} for i in range(1000)]
        result = _downsample(entries, max_points=100)
        assert len(result) <= 102  # max_points + possible last point
        assert result[-1] is entries[-1]  # last point always included

    def test_preserves_first(self):
        entries = [{"nav": i, "t": i} for i in range(500)]
        result = _downsample(entries, max_points=50)
        assert result[0] is entries[0]


class TestBuildEquitySvg:

    def test_returns_svg(self, nav_log):
        _, entries = nav_log
        result = _build_equity_svg(entries)
        assert result is not None
        svg, span_text, current_nav, delta, delta_pct, delta_color, delta_sign = result
        assert "<svg" in svg
        assert "<polyline" in svg
        assert current_nav == 10098  # last entry
        assert delta == 98  # 10098 - 10000
        assert delta_sign == "+"

    def test_insufficient_data(self):
        result = _build_equity_svg([{"nav": 100, "t": 1000}])
        assert result == ""

    def test_empty_data(self):
        result = _build_equity_svg([])
        assert result == ""

    def test_downtrend_red(self):
        entries = [
            {"nav": 10000, "t": 1000},
            {"nav": 9500, "t": 2000},
        ]
        result = _build_equity_svg(entries)
        assert result is not None
        svg, _, _, delta, _, delta_color, delta_sign = result
        assert delta < 0
        assert "#ef4444" in svg  # red line
        assert delta_color == "#ef4444"

    def test_flat_no_crash(self):
        entries = [
            {"nav": 10000, "t": 1000},
            {"nav": 10000, "t": 2000},
        ]
        result = _build_equity_svg(entries)
        assert result is not None
        svg = result[0]
        assert "<svg" in svg

    def test_y_axis_labels(self, nav_log):
        _, entries = nav_log
        result = _build_equity_svg(entries)
        svg = result[0]
        assert "$10,0" in svg  # y-axis label should show dollar amounts

    def test_x_axis_labels(self, nav_log):
        _, entries = nav_log
        result = _build_equity_svg(entries)
        svg = result[0]
        # Should have time labels (HH:MM format for <48h span)
        assert ":" in svg  # time format contains colon
