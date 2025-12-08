"""
Test suite for correlation_cap veto in execution/risk_limits.py.

Tests that orders are vetoed when correlation group exposure would exceed caps.
"""

import pytest
from decimal import Decimal
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from execution import risk_limits as risk_limits_module
from execution.risk_limits import check_order, RiskState
from execution.risk_loader import CorrelationGroupConfig, CorrelationGroupsConfig


def _base_cfg() -> Dict[str, Any]:
    """Return base config for tests."""
    return {
        "global": {
            "max_portfolio_nav_pct": 1.0,
            "max_trade_nav_pct": 0.20,
            "nav_freshness_seconds": 90,
            "fail_closed_on_nav_stale": True,
            "leverage_cap": 5.0,
            "min_notional": 10.0,
        },
        "per_symbol": {
            "BTCUSDT": {"max_order_notional": 10000.0},
            "ETHUSDT": {"max_order_notional": 10000.0},
        },
        "whitelisted_symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"],
    }


def _fresh_nav(monkeypatch, nav_value: float):
    """Mock nav to be fresh with given value."""
    nav_snapshot = {"age_s": 0.0, "sources_ok": True, "fresh": True, "nav_total": nav_value}
    monkeypatch.setattr(risk_limits_module, "nav_health_snapshot", lambda threshold_s=None: dict(nav_snapshot))
    monkeypatch.setattr(risk_limits_module, "get_nav_freshness_snapshot", lambda: (0.0, True))
    monkeypatch.setattr(risk_limits_module, "_nav_history_from_log", lambda limit=200: [])


def _clean_drawdown(monkeypatch):
    """Mock drawdown to return clean state."""
    clean_snapshot = {
        "drawdown": {"pct": 0.0, "peak_nav": 10000.0, "nav": 10000.0, "abs": 0.0},
        "daily_loss": {"pct": 0.0},
        "dd_pct": 0.0,
        "peak": 10000.0,
        "nav": 10000.0,
        "usable": True,
        "stale_flags": {},
        "nav_health": {"fresh": True, "sources_ok": True},
        "peak_state": {},
        "assets": {},
    }
    monkeypatch.setattr(risk_limits_module, "_drawdown_snapshot", lambda g_cfg=None: clean_snapshot)


def _no_portfolio_dd(monkeypatch):
    """Mock portfolio DD to return no drawdown."""
    mock_dd_state = MagicMock()
    mock_dd_state.current_dd_pct = 0.0
    monkeypatch.setattr(risk_limits_module, "get_portfolio_dd_state", lambda: mock_dd_state)


class TestCorrelationCapVeto:
    """Tests for correlation_cap veto behavior."""
    
    def test_order_vetoed_when_group_cap_exceeded(self, monkeypatch) -> None:
        """Order should be vetoed if it would push group exposure over cap."""
        _fresh_nav(monkeypatch, 50000.0)
        _clean_drawdown(monkeypatch)
        _no_portfolio_dd(monkeypatch)
        
        correlation_config = CorrelationGroupsConfig(
            groups={
                "L1_bluechips": CorrelationGroupConfig(
                    symbols=["BTCUSDT", "ETHUSDT"],
                    max_group_nav_pct=0.35,  # 35% cap
                ),
            }
        )
        
        # Mock correlation config loader
        monkeypatch.setattr(
            risk_limits_module,
            "load_correlation_groups_config",
            lambda: correlation_config,
        )
        
        # Mock exposure calculations
        def mock_current(positions, nav_total_usd, corr_cfg):
            return {"L1_bluechips": 0.30}  # 30% current
        
        def mock_hypothetical(positions, nav_total_usd, corr_cfg, order_symbol, order_notional_usd):
            return {"L1_bluechips": 0.40}  # Would be 40% (exceeds 35%)
        
        monkeypatch.setattr(risk_limits_module, "compute_group_exposure_nav_pct", mock_current)
        monkeypatch.setattr(risk_limits_module, "compute_hypothetical_group_exposure_nav_pct", mock_hypothetical)
        
        st = RiskState()
        cfg = _base_cfg()
        
        veto, detail = check_order(
            symbol="ETHUSDT",
            side="BUY",
            requested_notional=5000.0,
            price=3000.0,
            nav=50000.0,
            open_qty=0.0,
            now=0.0,
            cfg=cfg,
            state=st,
            current_gross_notional=15000.0,
        )
        
        assert veto is True
        reasons = detail.get("reasons") or []
        assert "correlation_cap" in reasons
    
    def test_order_approved_when_under_group_cap(self, monkeypatch) -> None:
        """Order should pass if group exposure stays under cap."""
        _fresh_nav(monkeypatch, 50000.0)
        _clean_drawdown(monkeypatch)
        _no_portfolio_dd(monkeypatch)
        
        correlation_config = CorrelationGroupsConfig(
            groups={
                "L1_bluechips": CorrelationGroupConfig(
                    symbols=["BTCUSDT", "ETHUSDT"],
                    max_group_nav_pct=0.35,  # 35% cap
                ),
            }
        )
        
        monkeypatch.setattr(
            risk_limits_module,
            "load_correlation_groups_config",
            lambda: correlation_config,
        )
        
        def mock_current(positions, nav_total_usd, corr_cfg):
            return {"L1_bluechips": 0.20}  # 20% current
        
        def mock_hypothetical(positions, nav_total_usd, corr_cfg, order_symbol, order_notional_usd):
            return {"L1_bluechips": 0.30}  # Would be 30% (under 35%)
        
        monkeypatch.setattr(risk_limits_module, "compute_group_exposure_nav_pct", mock_current)
        monkeypatch.setattr(risk_limits_module, "compute_hypothetical_group_exposure_nav_pct", mock_hypothetical)
        
        st = RiskState()
        cfg = _base_cfg()
        
        veto, detail = check_order(
            symbol="ETHUSDT",
            side="BUY",
            requested_notional=5000.0,
            price=3000.0,
            nav=50000.0,
            open_qty=0.0,
            now=0.0,
            cfg=cfg,
            state=st,
            current_gross_notional=10000.0,
        )
        
        assert veto is False
        reasons = detail.get("reasons") or []
        assert "correlation_cap" not in reasons
    
    def test_no_veto_when_no_correlation_groups_configured(self, monkeypatch) -> None:
        """Order should pass if no correlation groups are configured."""
        _fresh_nav(monkeypatch, 50000.0)
        _clean_drawdown(monkeypatch)
        _no_portfolio_dd(monkeypatch)
        
        # Empty groups config
        correlation_config = CorrelationGroupsConfig(groups={})
        monkeypatch.setattr(
            risk_limits_module,
            "load_correlation_groups_config",
            lambda: correlation_config,
        )
        
        st = RiskState()
        cfg = _base_cfg()
        
        veto, detail = check_order(
            symbol="BTCUSDT",
            side="BUY",
            requested_notional=5000.0,
            price=100000.0,
            nav=50000.0,
            open_qty=0.0,
            now=0.0,
            cfg=cfg,
            state=st,
            current_gross_notional=0.0,
        )
        
        assert veto is False
        reasons = detail.get("reasons") or []
        assert "correlation_cap" not in reasons
    
    def test_veto_detail_includes_group_info(self, monkeypatch) -> None:
        """Veto detail should include group name, hypothetical exposure, and cap."""
        _fresh_nav(monkeypatch, 50000.0)
        _clean_drawdown(monkeypatch)
        _no_portfolio_dd(monkeypatch)
        
        correlation_config = CorrelationGroupsConfig(
            groups={
                "L1_bluechips": CorrelationGroupConfig(
                    symbols=["BTCUSDT", "ETHUSDT"],
                    max_group_nav_pct=0.35,
                ),
            }
        )
        
        monkeypatch.setattr(
            risk_limits_module,
            "load_correlation_groups_config",
            lambda: correlation_config,
        )
        
        def mock_current(positions, nav_total_usd, corr_cfg):
            return {"L1_bluechips": 0.30}
        
        def mock_hypothetical(positions, nav_total_usd, corr_cfg, order_symbol, order_notional_usd):
            return {"L1_bluechips": 0.42}  # Exceeds cap
        
        monkeypatch.setattr(risk_limits_module, "compute_group_exposure_nav_pct", mock_current)
        monkeypatch.setattr(risk_limits_module, "compute_hypothetical_group_exposure_nav_pct", mock_hypothetical)
        
        st = RiskState()
        cfg = _base_cfg()
        
        veto, detail = check_order(
            symbol="ETHUSDT",
            side="BUY",
            requested_notional=10000.0,
            price=3000.0,
            nav=50000.0,
            open_qty=0.0,
            now=0.0,
            cfg=cfg,
            state=st,
            current_gross_notional=15000.0,
        )
        
        assert veto is True
        
        # Check observations for correlation_cap details
        observations = detail.get("observations") or {}
        corr_detail = observations.get("correlation_cap") or {}
        
        assert corr_detail.get("group_name") == "L1_bluechips"
        assert corr_detail.get("max_group_nav_pct") == pytest.approx(0.35)
        assert corr_detail.get("group_nav_pct_after") == pytest.approx(0.42)
