"""Unit tests for multi-engine dashboard panel helpers."""

import pytest

from dashboard.multi_engine_panel import _sparkline


@pytest.mark.unit
class TestSparkline:

    def test_basic_ascending(self):
        """Ascending values produce ascending blocks."""
        result = _sparkline([0.0, 0.25, 0.5, 0.75, 1.0])
        assert len(result) == 5
        # Each character should be >= previous
        assert list(result) == sorted(result)

    def test_constant_values(self):
        """Identical values produce identical blocks."""
        result = _sparkline([0.5, 0.5, 0.5])
        assert len(set(result)) == 1

    def test_empty(self):
        """Empty list returns empty string."""
        assert _sparkline([]) == ""

    def test_single_value(self):
        """Single value returns single block."""
        result = _sparkline([0.5])
        assert len(result) == 1

    def test_full_range(self):
        """0.0 maps to lowest block, 1.0 to highest."""
        result = _sparkline([0.0, 1.0])
        assert result[0] == " "
        assert result[-1] == "\u2588"

    def test_custom_range(self):
        """Custom lo/hi range is respected."""
        result = _sparkline([10, 20, 30], lo=10, hi=30)
        assert len(result) == 3
        assert result[0] < result[-1]
