"""
Tests for VaR/CVaR integration with risk_limits.py (v7.5_A1)
"""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
from typing import Dict, Any, Optional


# ===========================================================================
# Mock RiskState for testing
# ===========================================================================

@dataclass
class MockRiskState:
    """Mock risk state for testing."""
    live_positions: Dict[str, Any] = None
    client: Any = None
    daily_pnl_pct: float = 0.0
    
    def __post_init__(self):
        if self.live_positions is None:
            self.live_positions = {}


# ===========================================================================
# Tests: VaR/CVaR Veto Integration
# ===========================================================================

class TestVarCvarVetoIntegration:
    """Test VaR/CVaR vetoes are correctly integrated into check_order."""
    
    @patch("execution.risk_limits.nav_health_snapshot")
    @patch("execution.risk_limits.load_risk_config")
    @patch("execution.risk_limits.load_correlation_groups_config")
    def test_var_veto_details_in_check_order_response(
        self,
        mock_corr_cfg,
        mock_risk_cfg,
        mock_nav_health,
    ):
        """check_order should include VaR details when computed."""
        # Setup mocks
        mock_nav_health.return_value = {
            "age_s": 10.0,
            "sources_ok": True,
            "fresh": True,
            "nav_total": 100000.0,
        }
        mock_risk_cfg.return_value = {
            "global": {
                "nav_freshness_seconds": 90,
            },
            "per_symbol": {},
        }
        mock_corr_cfg.return_value = MagicMock(groups={})
        
        # Mock vol_risk module
        with patch("execution.risk_limits.json") as mock_json:
            mock_json.load.return_value = {
                "risk_advanced": {
                    "var": {"enabled": True, "max_portfolio_var_nav_pct": 0.12},
                    "cvar": {"enabled": True, "max_position_cvar_nav_pct": 0.04},
                }
            }
            
            # Import after patching
            from execution.risk_limits import check_order
            
            state = MockRiskState(
                live_positions={
                    "BTCUSDT": {"positionAmt": 0.1, "markPrice": 50000.0, "notional": 5000.0}
                }
            )
            
            # This test verifies the integration point exists
            # Full functional testing requires returns data
            veto, details = check_order(
                symbol="ETHUSDT",
                side="BUY",
                requested_notional=5000.0,
                price=2000.0,
                nav=100000.0,
                open_qty=0.0,
                now=1700000000.0,
                cfg={
                    "global": {"nav_freshness_seconds": 90},
                    "per_symbol": {},
                },
                state=state,
            )
            
            # The response should be a valid tuple
            assert isinstance(veto, bool)
            assert isinstance(details, dict)


class TestVarVetoReason:
    """Test VaR veto reason formatting."""
    
    def test_portfolio_var_limit_veto_format(self):
        """portfolio_var_limit veto should have correct format."""
        from execution.vol_risk import VaRResult, VaRConfig, check_portfolio_var_limit
        
        var_result = VaRResult(
            var_usd=15000.0,
            var_nav_pct=0.15,  # 15% VaR exceeds 12% limit
            portfolio_volatility=0.25,
            n_assets=3,
            lookback_used=200,
        )
        var_config = VaRConfig(
            enabled=True,
            max_portfolio_var_nav_pct=0.12,
        )
        
        should_veto, details = check_portfolio_var_limit(var_result, var_config)
        
        assert should_veto == True
        assert details["reason"] == "portfolio_var_limit"
        assert "observed" in details
        assert "limits" in details
        assert details["observed"]["portfolio_var_nav_pct"] == 0.15
        assert details["limits"]["max_portfolio_var_nav_pct"] == 0.12


class TestCvarVetoReason:
    """Test CVaR veto reason formatting."""
    
    def test_position_cvar_limit_veto_format(self):
        """position_cvar_limit veto should have correct format."""
        from execution.vol_risk import CVaRResult, CVaRConfig, check_position_cvar_limit
        
        cvar_result = CVaRResult(
            symbol="BTCUSDT",
            cvar_usd=5000.0,
            cvar_nav_pct=0.05,  # 5% CVaR exceeds 4% limit
            var_usd=4000.0,
            var_nav_pct=0.04,
            position_notional_usd=10000.0,
        )
        cvar_config = CVaRConfig(
            enabled=True,
            max_position_cvar_nav_pct=0.04,
        )
        
        should_veto, details = check_position_cvar_limit(cvar_result, cvar_config)
        
        assert should_veto == True
        assert details["reason"] == "position_cvar_limit"
        assert details["observed"]["symbol"] == "BTCUSDT"
        assert details["observed"]["position_cvar_nav_pct"] == 0.05


# ===========================================================================
# Tests: Deterministic Veto Scenarios
# ===========================================================================

class TestDeterministicVetoScenarios:
    """Test deterministic veto scenarios with fake data."""
    
    def test_var_within_limit_no_veto(self):
        """VaR within limit should not trigger veto."""
        from execution.vol_risk import VaRResult, VaRConfig, check_portfolio_var_limit
        
        var_result = VaRResult(
            var_usd=8000.0,
            var_nav_pct=0.08,  # 8% VaR below 12% limit
            portfolio_volatility=0.18,
        )
        var_config = VaRConfig(enabled=True, max_portfolio_var_nav_pct=0.12)
        
        should_veto, _ = check_portfolio_var_limit(var_result, var_config)
        
        assert should_veto == False

    def test_cvar_within_limit_no_veto(self):
        """CVaR within limit should not trigger veto."""
        from execution.vol_risk import CVaRResult, CVaRConfig, check_position_cvar_limit
        
        cvar_result = CVaRResult(
            symbol="ETHUSDT",
            cvar_usd=3000.0,
            cvar_nav_pct=0.03,  # 3% CVaR below 4% limit
            var_usd=2500.0,
            var_nav_pct=0.025,
            position_notional_usd=8000.0,
        )
        cvar_config = CVaRConfig(enabled=True, max_position_cvar_nav_pct=0.04)
        
        should_veto, _ = check_position_cvar_limit(cvar_result, cvar_config)
        
        assert should_veto == False

    def test_both_var_and_cvar_can_trigger(self):
        """Both VaR and CVaR can independently trigger vetoes."""
        from execution.vol_risk import (
            VaRResult, CVaRResult, VaRConfig, CVaRConfig,
            check_portfolio_var_limit, check_position_cvar_limit,
        )
        
        # VaR breaches
        var_result = VaRResult(var_usd=15000, var_nav_pct=0.15, portfolio_volatility=0.3)
        var_config = VaRConfig(enabled=True, max_portfolio_var_nav_pct=0.12)
        var_veto, _ = check_portfolio_var_limit(var_result, var_config)
        
        # CVaR breaches
        cvar_result = CVaRResult(
            symbol="SOLUSDT", cvar_usd=6000, cvar_nav_pct=0.06,
            var_usd=5000, var_nav_pct=0.05, position_notional_usd=15000,
        )
        cvar_config = CVaRConfig(enabled=True, max_position_cvar_nav_pct=0.04)
        cvar_veto, _ = check_position_cvar_limit(cvar_result, cvar_config)
        
        assert var_veto == True
        assert cvar_veto == True


# ===========================================================================
# Tests: Config Integration
# ===========================================================================

class TestVarCvarConfigIntegration:
    """Test VaR/CVaR configuration from strategy_config.json."""
    
    def test_loads_var_config_from_strategy_config(self):
        """Should load VaR config from strategy_config.json structure."""
        from execution.vol_risk import load_var_config
        
        strategy_cfg = {
            "risk_advanced": {
                "var": {
                    "enabled": True,
                    "confidence": 0.99,
                    "lookback_bars": 500,
                    "halflife_bars": 100,
                    "max_portfolio_var_nav_pct": 0.12,
                }
            }
        }
        
        config = load_var_config(strategy_cfg)
        
        assert config.enabled == True
        assert config.confidence == 0.99
        assert config.lookback_bars == 500
        assert config.halflife_bars == 100
        assert config.max_portfolio_var_nav_pct == 0.12

    def test_loads_cvar_config_from_strategy_config(self):
        """Should load CVaR config from strategy_config.json structure."""
        from execution.vol_risk import load_cvar_config
        
        strategy_cfg = {
            "risk_advanced": {
                "cvar": {
                    "enabled": True,
                    "confidence": 0.95,
                    "lookback_bars": 400,
                    "max_position_cvar_nav_pct": 0.04,
                }
            }
        }
        
        config = load_cvar_config(strategy_cfg)
        
        assert config.enabled == True
        assert config.confidence == 0.95
        assert config.lookback_bars == 400
        assert config.max_position_cvar_nav_pct == 0.04

    def test_disabled_config_prevents_veto(self):
        """Disabled VaR/CVaR should never veto."""
        from execution.vol_risk import (
            VaRResult, CVaRResult, VaRConfig, CVaRConfig,
            check_portfolio_var_limit, check_position_cvar_limit,
        )
        
        # High VaR that would normally veto
        var_result = VaRResult(var_usd=50000, var_nav_pct=0.50, portfolio_volatility=0.5)
        var_config = VaRConfig(enabled=False, max_portfolio_var_nav_pct=0.12)
        var_veto, _ = check_portfolio_var_limit(var_result, var_config)
        
        # High CVaR that would normally veto
        cvar_result = CVaRResult(
            symbol="BTCUSDT", cvar_usd=20000, cvar_nav_pct=0.20,
            var_usd=15000, var_nav_pct=0.15, position_notional_usd=50000,
        )
        cvar_config = CVaRConfig(enabled=False, max_position_cvar_nav_pct=0.04)
        cvar_veto, _ = check_position_cvar_limit(cvar_result, cvar_config)
        
        assert var_veto == False
        assert cvar_veto == False
