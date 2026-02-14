"""
Tests for NAV PnL span authority.

Validates that compute_nav_deltas() returns span_ok flags and that
nav_window_valid() is the single-point guard for all dashboard panels.

Bug class: authority-ordering violation — windowed metrics displayed
without proving sufficient log span.
"""
import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _clear_cache():
    """Reset module-level cache between tests."""
    from dashboard.components.nav_pnl import _cache
    _cache["mtime"] = 0.0
    _cache["entries"] = []
    yield
    _cache["mtime"] = 0.0
    _cache["entries"] = []


def _make_nav_log(span_hours: float, n_entries: int = 100) -> list:
    """Generate a synthetic nav_log with given span."""
    now = time.time()
    start = now - (span_hours * 3600)
    interval = (span_hours * 3600) / max(n_entries - 1, 1)
    entries = []
    for i in range(n_entries):
        entries.append({
            "nav": 10000.0 + i * 0.5,
            "t": start + i * interval,
            "unrealized_pnl": 0.0,
        })
    return entries


class TestSpanOkFlags:
    """compute_nav_deltas() must return span_ok for every window."""

    def test_empty_log_all_false(self):
        from dashboard.components.nav_pnl import compute_nav_deltas
        with patch("dashboard.components.nav_pnl._load_nav_log", return_value=[]):
            result = compute_nav_deltas()
        assert "span_ok" in result
        for window in ("24h", "7d", "30d", "all_time"):
            assert result["span_ok"][window] is False

    def test_short_span_24h_false(self):
        """12h log → 24h span_ok must be False."""
        from dashboard.components.nav_pnl import compute_nav_deltas
        entries = _make_nav_log(span_hours=12)
        with patch("dashboard.components.nav_pnl._load_nav_log", return_value=entries):
            result = compute_nav_deltas()
        assert result["span_ok"]["24h"] is False
        assert result["span_ok"]["7d"] is False
        assert result["span_ok"]["30d"] is False
        assert result["span_ok"]["all_time"] is False

    def test_sufficient_span_24h_true(self):
        """25h log → 24h span_ok must be True (≥0.9d ≈ 21.6h)."""
        from dashboard.components.nav_pnl import compute_nav_deltas
        entries = _make_nav_log(span_hours=25)
        with patch("dashboard.components.nav_pnl._load_nav_log", return_value=entries):
            result = compute_nav_deltas()
        assert result["span_ok"]["24h"] is True
        assert result["span_ok"]["7d"] is False  # 25h < 6d

    def test_7d_span(self):
        """8-day log → 24h and 7d valid, 30d not."""
        from dashboard.components.nav_pnl import compute_nav_deltas
        entries = _make_nav_log(span_hours=8 * 24, n_entries=200)
        with patch("dashboard.components.nav_pnl._load_nav_log", return_value=entries):
            result = compute_nav_deltas()
        assert result["span_ok"]["24h"] is True
        assert result["span_ok"]["7d"] is True
        assert result["span_ok"]["all_time"] is True  # 8d >= 7d threshold
        assert result["span_ok"]["30d"] is False

    def test_30d_span(self):
        """35-day log → all windows valid."""
        from dashboard.components.nav_pnl import compute_nav_deltas
        entries = _make_nav_log(span_hours=35 * 24, n_entries=500)
        with patch("dashboard.components.nav_pnl._load_nav_log", return_value=entries):
            result = compute_nav_deltas()
        for window in ("24h", "7d", "30d", "all_time"):
            assert result["span_ok"][window] is True

    def test_pnl_values_still_computed(self):
        """Raw deltas are present even when span_ok is False (for diagnostics)."""
        from dashboard.components.nav_pnl import compute_nav_deltas
        entries = _make_nav_log(span_hours=12)
        with patch("dashboard.components.nav_pnl._load_nav_log", return_value=entries):
            result = compute_nav_deltas()
        # pnl_24h should have a non-zero value (diagnostic), but span_ok is False
        assert result["span_ok"]["24h"] is False
        assert "pnl_24h" in result  # raw value still present

    def test_all_time_suppressed_below_7d(self):
        """all_time PnL must be 0.0 when span < 7d."""
        from dashboard.components.nav_pnl import compute_nav_deltas
        entries = _make_nav_log(span_hours=5 * 24)  # 5 days
        with patch("dashboard.components.nav_pnl._load_nav_log", return_value=entries):
            result = compute_nav_deltas()
        assert result["pnl_all_time"] == 0.0
        assert result["span_ok"]["all_time"] is False


class TestNavWindowValid:
    """nav_window_valid() — single-point guard for any panel."""

    def test_valid_when_sufficient(self):
        from dashboard.components.nav_pnl import nav_window_valid
        entries = _make_nav_log(span_hours=25)
        with patch("dashboard.components.nav_pnl._load_nav_log", return_value=entries):
            assert nav_window_valid("24h") is True

    def test_invalid_when_insufficient(self):
        from dashboard.components.nav_pnl import nav_window_valid
        entries = _make_nav_log(span_hours=12)
        with patch("dashboard.components.nav_pnl._load_nav_log", return_value=entries):
            assert nav_window_valid("24h") is False

    def test_unknown_window_returns_false(self):
        from dashboard.components.nav_pnl import nav_window_valid
        entries = _make_nav_log(span_hours=25)
        with patch("dashboard.components.nav_pnl._load_nav_log", return_value=entries):
            assert nav_window_valid("1y") is False

    def test_empty_log_returns_false(self):
        from dashboard.components.nav_pnl import nav_window_valid
        with patch("dashboard.components.nav_pnl._load_nav_log", return_value=[]):
            assert nav_window_valid("24h") is False


class TestSpanThresholdConstants:
    """Thresholds are defined once in nav_pnl — verify they exist."""

    def test_thresholds_exist(self):
        from dashboard.components.nav_pnl import _SPAN_THRESHOLDS
        assert _SPAN_THRESHOLDS["24h"] == 0.9
        assert _SPAN_THRESHOLDS["7d"] == 6.0
        assert _SPAN_THRESHOLDS["30d"] == 25.0
        assert _SPAN_THRESHOLDS["all_time"] == 7.0


class TestKpiStripSpanAuthority:
    """KPI strip must respect span_ok — never show misleading delta."""

    def test_kpi_strip_suppresses_when_span_insufficient(self):
        """With 12h span, KPI strip 24h PnL must be 0, not the raw delta."""
        from dashboard.components.kpi_strip import build_kpi_cards
        entries = _make_nav_log(span_hours=12)
        with patch("dashboard.components.nav_pnl._load_nav_log", return_value=entries):
            cards = build_kpi_cards(
                nav_state={"nav_usd": 10000, "unrealized_pnl": 0, "gross_exposure": 0, "drawdown_pct": 0},
                aum_data={},
                kpis={},
                risk_snapshot={},
            )
        pnl_card = next(c for c in cards if c["label"] == "24h PnL")
        # Should show +$0 or $0, not a misleading delta
        assert "$0" in pnl_card["value_html"]

    def test_kpi_strip_shows_pnl_when_span_sufficient(self):
        """With 25h span, KPI strip 24h PnL should show the delta."""
        from dashboard.components.kpi_strip import build_kpi_cards
        entries = _make_nav_log(span_hours=25)
        with patch("dashboard.components.nav_pnl._load_nav_log", return_value=entries):
            cards = build_kpi_cards(
                nav_state={"nav_usd": 10000, "unrealized_pnl": 0, "gross_exposure": 0, "drawdown_pct": 0},
                aum_data={},
                kpis={},
                risk_snapshot={},
            )
        pnl_card = next(c for c in cards if c["label"] == "24h PnL")
        # Should show actual value (non-zero since we have NAV drift in synthetic data)
        assert pnl_card["value_html"] != "$0"
