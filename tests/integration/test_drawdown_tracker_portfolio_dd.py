"""
Tests for portfolio drawdown state computation in drawdown_tracker.

Tests the get_portfolio_dd_state function used by the portfolio_dd_circuit breaker.
"""
from __future__ import annotations

import pytest

from execution.drawdown_tracker import get_portfolio_dd_state, PortfolioDDState


class TestGetPortfolioDDState:
    """Test suite for get_portfolio_dd_state function."""

    def test_monotonic_increasing_nav_zero_drawdown(self) -> None:
        """Monotonically increasing NAV should result in zero drawdown."""
        nav_history = [1000.0, 1050.0, 1100.0, 1150.0, 1200.0]
        result = get_portfolio_dd_state(nav_history)

        assert result is not None
        assert isinstance(result, PortfolioDDState)
        assert result.current_dd_pct == 0.0
        assert result.peak_nav_usd == 1200.0
        assert result.latest_nav_usd == 1200.0

    def test_nav_off_peak_ten_percent(self) -> None:
        """NAV 10% off peak should result in 0.10 drawdown."""
        nav_history = [1000.0, 1100.0, 1200.0, 1080.0]
        result = get_portfolio_dd_state(nav_history)

        assert result is not None
        # DD = (1200 - 1080) / 1200 = 120 / 1200 = 0.10
        assert abs(result.current_dd_pct - 0.10) < 0.001
        assert result.peak_nav_usd == 1200.0
        assert result.latest_nav_usd == 1080.0

    def test_nav_off_peak_fifteen_percent(self) -> None:
        """NAV 15% off peak should result in 0.15 drawdown."""
        nav_history = [1000.0, 1200.0, 1020.0]
        result = get_portfolio_dd_state(nav_history)

        assert result is not None
        # DD = (1200 - 1020) / 1200 = 180 / 1200 = 0.15
        assert abs(result.current_dd_pct - 0.15) < 0.001
        assert result.peak_nav_usd == 1200.0
        assert result.latest_nav_usd == 1020.0

    def test_empty_nav_history_returns_none(self) -> None:
        """Empty NAV history should return None."""
        result = get_portfolio_dd_state([])
        assert result is None

    def test_all_non_positive_values_returns_none(self) -> None:
        """All non-positive values should return None."""
        result = get_portfolio_dd_state([0.0, -100.0, 0.0])
        assert result is None

    def test_single_positive_value(self) -> None:
        """Single positive value should result in zero drawdown."""
        nav_history = [1000.0]
        result = get_portfolio_dd_state(nav_history)

        assert result is not None
        assert result.current_dd_pct == 0.0
        assert result.peak_nav_usd == 1000.0
        assert result.latest_nav_usd == 1000.0

    def test_nav_with_mixed_values(self) -> None:
        """NAV with some non-positive values should filter them out."""
        nav_history = [1000.0, 0.0, 1200.0, -50.0, 1100.0]
        result = get_portfolio_dd_state(nav_history)

        assert result is not None
        # Valid values are [1000, 1200, 1100], peak=1200, latest=1100
        # DD = (1200 - 1100) / 1200 = 100 / 1200 ≈ 0.0833
        assert abs(result.current_dd_pct - 0.0833) < 0.001
        assert result.peak_nav_usd == 1200.0
        assert result.latest_nav_usd == 1100.0

    def test_peak_at_start_drawdown_throughout(self) -> None:
        """Peak at start with declining NAV should show correct drawdown."""
        nav_history = [1500.0, 1400.0, 1300.0, 1200.0]
        result = get_portfolio_dd_state(nav_history)

        assert result is not None
        # DD = (1500 - 1200) / 1500 = 300 / 1500 = 0.20
        assert abs(result.current_dd_pct - 0.20) < 0.001
        assert result.peak_nav_usd == 1500.0
        assert result.latest_nav_usd == 1200.0

    def test_recovery_from_drawdown(self) -> None:
        """NAV that recovers to new peak should show zero drawdown."""
        nav_history = [1000.0, 1200.0, 1100.0, 1250.0]
        result = get_portfolio_dd_state(nav_history)

        assert result is not None
        assert result.current_dd_pct == 0.0
        assert result.peak_nav_usd == 1250.0
        assert result.latest_nav_usd == 1250.0

    def test_type_coercion_with_integers(self) -> None:
        """Should handle integer inputs correctly."""
        nav_history = [1000, 1200, 1080]
        result = get_portfolio_dd_state(nav_history)

        assert result is not None
        assert abs(result.current_dd_pct - 0.10) < 0.001

    def test_extreme_drawdown(self) -> None:
        """Test extreme drawdown scenario (50% DD)."""
        nav_history = [2000.0, 1000.0]
        result = get_portfolio_dd_state(nav_history)

        assert result is not None
        assert abs(result.current_dd_pct - 0.50) < 0.001
        assert result.peak_nav_usd == 2000.0
        assert result.latest_nav_usd == 1000.0

    def test_dataclass_fields(self) -> None:
        """Verify PortfolioDDState dataclass has expected fields."""
        nav_history = [1000.0, 1100.0, 1050.0]
        result = get_portfolio_dd_state(nav_history)

        assert result is not None
        assert hasattr(result, 'current_dd_pct')
        assert hasattr(result, 'peak_nav_usd')
        assert hasattr(result, 'latest_nav_usd')
        assert isinstance(result.current_dd_pct, float)
        assert isinstance(result.peak_nav_usd, float)
        assert isinstance(result.latest_nav_usd, float)
