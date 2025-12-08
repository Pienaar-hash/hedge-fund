"""
Tests for portfolio drawdown circuit breaker in risk_limits.

Tests the portfolio_dd_circuit veto reason and circuit breaker behavior.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict
from unittest import mock

import pytest

from execution import risk_limits as risk_limits_module
from execution.risk_limits import check_order, RiskState


def _base_cfg() -> Dict[str, Any]:
    """Base config for circuit breaker tests."""
    return {
        "global": {
            "min_notional_usdt": 25.0,
            "max_trade_nav_pct": 0.2,
            "trade_equity_nav_pct": 0.15,
            "nav_freshness_seconds": 1_000_000,
            "fail_closed_on_nav_stale": False,
        },
        "per_symbol": {
            "BTCUSDT": {
                "min_notional": 25.0,
                "max_order_notional": 50000.0,
                "max_nav_pct": 0.25,
                "max_leverage": 4,
            },
        },
    }


def _mock_nav_health(fresh: bool = True, nav_total: float = 10000.0):
    """Create a mock nav_health_snapshot function."""
    def _mock(threshold_s: int = 90) -> Dict[str, Any]:
        return {
            "fresh": fresh,
            "age_s": 10.0 if fresh else 200.0,
            "sources_ok": True,
            "nav_total": nav_total,
        }
    return _mock


class TestPortfolioDDCircuitNoConfig:
    """Tests when circuit breaker is not configured."""

    def test_no_circuit_config_does_not_veto(self, monkeypatch, tmp_path):
        """When max_portfolio_dd_nav_pct is None, no circuit veto should occur."""
        cfg = _base_cfg()
        # No circuit_breakers config
        
        st = RiskState()
        
        # Create a nav_log.json with high drawdown
        nav_log_path = tmp_path / "logs" / "nav_log.json"
        nav_log_path.parent.mkdir(parents=True, exist_ok=True)
        nav_log_path.write_text(json.dumps([
            {"nav": 10000.0},
            {"nav": 12000.0},  # peak
            {"nav": 8000.0},   # 33% drawdown
        ]))
        
        monkeypatch.setattr(
            risk_limits_module,
            "nav_health_snapshot",
            _mock_nav_health(fresh=True, nav_total=8000.0),
        )
        monkeypatch.setattr(
            risk_limits_module,
            "_nav_history_from_log",
            lambda limit=200: [10000.0, 12000.0, 8000.0],
        )
        monkeypatch.setattr(
            risk_limits_module,
            "get_nav_freshness_snapshot",
            lambda: (10.0, True),
        )
        monkeypatch.setattr(
            risk_limits_module,
            "universe_by_symbol",
            lambda: {"BTCUSDT": {}},
        )
        
        veto, detail = check_order(
            symbol="BTCUSDT",
            side="BUY",
            requested_notional=100.0,
            price=50000.0,
            nav=8000.0,
            open_qty=0.0,
            now=1700000000.0,
            cfg=cfg,
            state=st,
        )
        
        # Should not veto with portfolio_dd_circuit
        reasons = detail.get("reasons", [])
        assert "portfolio_dd_circuit" not in reasons


class TestPortfolioDDCircuitBelowThreshold:
    """Tests when drawdown is below threshold."""

    def test_dd_below_threshold_no_veto(self, monkeypatch):
        """When DD is below threshold, no circuit veto should occur."""
        cfg = _base_cfg()
        cfg["circuit_breakers"] = {
            "max_portfolio_dd_nav_pct": 0.10,  # 10% threshold
        }
        
        st = RiskState()
        
        # Mock nav history with 5% drawdown (below 10% threshold)
        # Peak = 10000, current = 9500 -> DD = 5%
        monkeypatch.setattr(
            risk_limits_module,
            "nav_health_snapshot",
            _mock_nav_health(fresh=True, nav_total=9500.0),
        )
        monkeypatch.setattr(
            risk_limits_module,
            "_nav_history_from_log",
            lambda limit=200: [9000.0, 10000.0, 9500.0],
        )
        monkeypatch.setattr(
            risk_limits_module,
            "get_nav_freshness_snapshot",
            lambda: (10.0, True),
        )
        monkeypatch.setattr(
            risk_limits_module,
            "universe_by_symbol",
            lambda: {"BTCUSDT": {}},
        )
        
        veto, detail = check_order(
            symbol="BTCUSDT",
            side="BUY",
            requested_notional=100.0,
            price=50000.0,
            nav=9500.0,
            open_qty=0.0,
            now=1700000000.0,
            cfg=cfg,
            state=st,
        )
        
        # Should not veto with portfolio_dd_circuit
        reasons = detail.get("reasons", [])
        assert "portfolio_dd_circuit" not in reasons
        
        # Circuit breaker info should be present but not active
        # Check in observations (where extra details are stored)
        observations = detail.get("observations", {})
        cb_info = observations.get("circuit_breaker", {})
        assert cb_info.get("active") is False
        assert cb_info.get("max_portfolio_dd_nav_pct") == 0.10


class TestPortfolioDDCircuitAboveThreshold:
    """Tests when drawdown exceeds threshold (circuit tripped)."""

    def test_dd_above_threshold_triggers_veto(self, monkeypatch):
        """When DD exceeds threshold, portfolio_dd_circuit veto should trigger."""
        cfg = _base_cfg()
        cfg["circuit_breakers"] = {
            "max_portfolio_dd_nav_pct": 0.10,  # 10% threshold
        }
        
        st = RiskState()
        
        # Mock nav history with 15% drawdown (above 10% threshold)
        # Peak = 10000, current = 8500 -> DD = 15%
        monkeypatch.setattr(
            risk_limits_module,
            "nav_health_snapshot",
            _mock_nav_health(fresh=True, nav_total=8500.0),
        )
        monkeypatch.setattr(
            risk_limits_module,
            "_nav_history_from_log",
            lambda limit=200: [9000.0, 10000.0, 8500.0],
        )
        monkeypatch.setattr(
            risk_limits_module,
            "get_nav_freshness_snapshot",
            lambda: (10.0, True),
        )
        monkeypatch.setattr(
            risk_limits_module,
            "universe_by_symbol",
            lambda: {"BTCUSDT": {}},
        )
        
        veto, detail = check_order(
            symbol="BTCUSDT",
            side="BUY",
            requested_notional=100.0,
            price=50000.0,
            nav=8500.0,
            open_qty=0.0,
            now=1700000000.0,
            cfg=cfg,
            state=st,
        )
        
        # Should veto with portfolio_dd_circuit
        reasons = detail.get("reasons", [])
        assert "portfolio_dd_circuit" in reasons
        
        # Circuit breaker info should be present and active
        # Check in observations (where extra details are stored)
        observations = detail.get("observations", {})
        cb_info = observations.get("circuit_breaker", {})
        assert cb_info.get("active") is True
        assert cb_info.get("max_portfolio_dd_nav_pct") == 0.10
        
        # Check portfolio_dd_circuit detail in observations
        dd_detail = observations.get("portfolio_dd_circuit", {})
        assert dd_detail.get("max_portfolio_dd_nav_pct") == 0.10
        assert dd_detail.get("peak_nav_usd") == 10000.0
        assert dd_detail.get("latest_nav_usd") == 8500.0
        # DD = (10000 - 8500) / 10000 = 0.15
        assert abs(dd_detail.get("current_dd_pct", 0) - 0.15) < 0.001

    def test_dd_exactly_at_threshold_triggers_veto(self, monkeypatch):
        """When DD equals threshold exactly, circuit should trigger."""
        cfg = _base_cfg()
        cfg["circuit_breakers"] = {
            "max_portfolio_dd_nav_pct": 0.10,  # 10% threshold
        }
        
        st = RiskState()
        
        # Mock nav history with exactly 10% drawdown
        # Peak = 10000, current = 9000 -> DD = 10%
        monkeypatch.setattr(
            risk_limits_module,
            "nav_health_snapshot",
            _mock_nav_health(fresh=True, nav_total=9000.0),
        )
        monkeypatch.setattr(
            risk_limits_module,
            "_nav_history_from_log",
            lambda limit=200: [9500.0, 10000.0, 9000.0],
        )
        monkeypatch.setattr(
            risk_limits_module,
            "get_nav_freshness_snapshot",
            lambda: (10.0, True),
        )
        monkeypatch.setattr(
            risk_limits_module,
            "universe_by_symbol",
            lambda: {"BTCUSDT": {}},
        )
        
        veto, detail = check_order(
            symbol="BTCUSDT",
            side="BUY",
            requested_notional=100.0,
            price=50000.0,
            nav=9000.0,
            open_qty=0.0,
            now=1700000000.0,
            cfg=cfg,
            state=st,
        )
        
        # Should veto with portfolio_dd_circuit (>= threshold)
        reasons = detail.get("reasons", [])
        assert "portfolio_dd_circuit" in reasons


class TestPortfolioDDCircuitEmptyNavHistory:
    """Tests when NAV history is empty or unavailable."""

    def test_empty_nav_history_no_veto(self, monkeypatch):
        """When NAV history is empty, no circuit veto should occur (fail-open)."""
        cfg = _base_cfg()
        cfg["circuit_breakers"] = {
            "max_portfolio_dd_nav_pct": 0.10,
        }
        
        st = RiskState()
        
        # Mock empty nav history
        monkeypatch.setattr(
            risk_limits_module,
            "nav_health_snapshot",
            _mock_nav_health(fresh=True, nav_total=10000.0),
        )
        monkeypatch.setattr(
            risk_limits_module,
            "_nav_history_from_log",
            lambda limit=200: [],
        )
        monkeypatch.setattr(
            risk_limits_module,
            "get_nav_freshness_snapshot",
            lambda: (10.0, True),
        )
        monkeypatch.setattr(
            risk_limits_module,
            "universe_by_symbol",
            lambda: {"BTCUSDT": {}},
        )
        
        veto, detail = check_order(
            symbol="BTCUSDT",
            side="BUY",
            requested_notional=100.0,
            price=50000.0,
            nav=10000.0,
            open_qty=0.0,
            now=1700000000.0,
            cfg=cfg,
            state=st,
        )
        
        # Should not veto with portfolio_dd_circuit when NAV history is unavailable
        reasons = detail.get("reasons", [])
        assert "portfolio_dd_circuit" not in reasons
        
        # Circuit breaker info should show not active
        observations = detail.get("observations", {})
        cb_info = observations.get("circuit_breaker", {})
        assert cb_info.get("active") is False


class TestPortfolioDDCircuitThresholdValues:
    """Tests for various threshold values."""

    @pytest.mark.parametrize("threshold,dd_pct,should_veto", [
        (0.05, 0.03, False),   # 5% threshold, 3% DD - no veto
        (0.05, 0.05, True),    # 5% threshold, 5% DD - veto (equal)
        (0.05, 0.08, True),    # 5% threshold, 8% DD - veto
        (0.20, 0.15, False),   # 20% threshold, 15% DD - no veto
        (0.20, 0.25, True),    # 20% threshold, 25% DD - veto
        (0.01, 0.02, True),    # 1% threshold, 2% DD - veto
    ])
    def test_various_threshold_values(self, monkeypatch, threshold, dd_pct, should_veto):
        """Test circuit breaker with various threshold and DD combinations."""
        cfg = _base_cfg()
        cfg["circuit_breakers"] = {
            "max_portfolio_dd_nav_pct": threshold,
        }
        
        st = RiskState()
        
        # Create nav history to achieve desired DD
        peak = 10000.0
        current = peak * (1 - dd_pct)
        
        monkeypatch.setattr(
            risk_limits_module,
            "nav_health_snapshot",
            _mock_nav_health(fresh=True, nav_total=current),
        )
        monkeypatch.setattr(
            risk_limits_module,
            "_nav_history_from_log",
            lambda limit=200: [8000.0, peak, current],
        )
        monkeypatch.setattr(
            risk_limits_module,
            "get_nav_freshness_snapshot",
            lambda: (10.0, True),
        )
        monkeypatch.setattr(
            risk_limits_module,
            "universe_by_symbol",
            lambda: {"BTCUSDT": {}},
        )
        
        veto, detail = check_order(
            symbol="BTCUSDT",
            side="BUY",
            requested_notional=100.0,
            price=50000.0,
            nav=current,
            open_qty=0.0,
            now=1700000000.0,
            cfg=cfg,
            state=st,
        )
        
        reasons = detail.get("reasons", [])
        if should_veto:
            assert "portfolio_dd_circuit" in reasons, f"Expected veto with threshold={threshold}, dd={dd_pct}"
        else:
            assert "portfolio_dd_circuit" not in reasons, f"Unexpected veto with threshold={threshold}, dd={dd_pct}"
