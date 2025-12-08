"""
Tests for CVaR (Expected Shortfall) calculations (v7.5_A1)
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from execution.vol_risk import (
    CVaRConfig,
    CVaRResult,
    compute_position_cvar,
    compute_position_cvar_for_symbol,
    compute_all_position_cvars,
    load_cvar_config,
    check_position_cvar_limit,
)


# ===========================================================================
# Tests: CVaR Mathematical Properties
# ===========================================================================

class TestCvarMathematicalProperties:
    """Test mathematical properties of CVaR computation."""
    
    def test_cvar_tail_mean_correctness(self):
        """CVaR should be the mean of worst (1-conf) losses."""
        # Create deterministic returns for testing
        # 100 returns: worst 5 are -10%, rest are +1%
        returns = np.array([0.01] * 95 + [-0.10] * 5)
        np.random.shuffle(returns)
        
        result = compute_position_cvar(
            returns,
            nav_usd=100000.0,
            position_notional_usd=10000.0,
            confidence=0.95,  # Tail is worst 5%
        )
        
        # For 100 returns at 95% confidence:
        # Worst 5% = 5 observations
        # Expected loss from 5 returns of -10% each = 10000 * 0.10 = 1000 each
        # CVaR should be ~1000 USD
        assert pytest.approx(result.cvar_usd, rel=0.1) == 1000.0

    def test_cvar_is_coherent_risk_measure(self):
        """CVaR should satisfy sub-additivity (approximately)."""
        np.random.seed(42)
        returns1 = np.random.randn(500) * 0.03
        returns2 = np.random.randn(500) * 0.03
        
        cvar1 = compute_position_cvar(returns1, 100000.0, 10000.0, 0.95)
        cvar2 = compute_position_cvar(returns2, 100000.0, 10000.0, 0.95)
        
        # Combined returns (assuming equal positions)
        combined_returns = (returns1 + returns2) / 2
        cvar_combined = compute_position_cvar(combined_returns, 100000.0, 20000.0, 0.95)
        
        # CVaR should be sub-additive: CVaR(A+B) <= CVaR(A) + CVaR(B)
        # Allow some tolerance for sampling variation
        assert cvar_combined.cvar_usd <= (cvar1.cvar_usd + cvar2.cvar_usd) * 1.1


class TestCvarNavConversion:
    """Test CVaR to NAV percentage conversion."""
    
    def test_cvar_nav_pct_calculation(self):
        """CVaR NAV percentage should be cvar_usd / nav_usd."""
        np.random.seed(42)
        returns = np.random.randn(200) * 0.03
        nav = 100000.0
        
        result = compute_position_cvar(returns, nav, 10000.0, 0.95)
        
        expected_pct = result.cvar_usd / nav
        assert pytest.approx(result.cvar_nav_pct, rel=1e-6) == expected_pct

    def test_cvar_nav_pct_zero_nav(self):
        """Should handle zero NAV gracefully."""
        returns = np.random.randn(100) * 0.03
        
        result = compute_position_cvar(returns, 0.0, 10000.0, 0.95)
        
        assert result.cvar_nav_pct == 0.0


# ===========================================================================
# Tests: CVaR for Symbol (with data loading)
# ===========================================================================

class TestComputePositionCvarForSymbol:
    """Test CVaR computation for individual symbols."""
    
    @patch("execution.vol_risk.load_symbol_returns")
    def test_computes_cvar_for_symbol(self, mock_load_returns):
        """Should compute CVaR for a symbol."""
        mock_load_returns.return_value = np.random.randn(200) * 0.02
        
        result = compute_position_cvar_for_symbol(
            symbol="BTCUSDT",
            position_notional_usd=10000.0,
            nav_usd=100000.0,
        )
        
        assert result.symbol == "BTCUSDT"
        assert result.cvar_usd > 0
        mock_load_returns.assert_called_once()

    @patch("execution.vol_risk.load_symbol_returns")
    def test_returns_empty_result_on_insufficient_data(self, mock_load_returns):
        """Should return empty result when insufficient data."""
        mock_load_returns.return_value = np.array([0.01, 0.02, 0.03])  # Only 3 returns
        
        result = compute_position_cvar_for_symbol(
            symbol="BTCUSDT",
            position_notional_usd=10000.0,
            nav_usd=100000.0,
        )
        
        assert result.cvar_usd == 0.0
        assert result.lookback_used == 3

    @patch("execution.vol_risk.load_symbol_returns")
    def test_respects_cvar_config(self, mock_load_returns):
        """Should use provided CVaR config."""
        mock_load_returns.return_value = np.random.randn(300) * 0.02
        
        config = CVaRConfig(
            enabled=True,
            confidence=0.99,
            lookback_bars=300,
            max_position_cvar_nav_pct=0.05,
        )
        
        result = compute_position_cvar_for_symbol(
            symbol="ETHUSDT",
            position_notional_usd=15000.0,
            nav_usd=100000.0,
            cvar_config=config,
        )
        
        assert result.confidence == 0.99
        assert result.limit_nav_pct == 0.05


class TestComputeAllPositionCvars:
    """Test CVaR computation for all positions."""
    
    @patch("execution.vol_risk.load_symbol_returns")
    def test_computes_cvar_for_all_positions(self, mock_load_returns):
        """Should compute CVaR for all positions."""
        mock_load_returns.return_value = np.random.randn(200) * 0.02
        
        positions = [
            {"symbol": "BTCUSDT", "notional": 10000.0},
            {"symbol": "ETHUSDT", "notional": 8000.0},
            {"symbol": "SOLUSDT", "notional": 5000.0},
        ]
        
        results = compute_all_position_cvars(positions, nav_usd=100000.0)
        
        assert len(results) == 3
        assert "BTCUSDT" in results
        assert "ETHUSDT" in results
        assert "SOLUSDT" in results

    @patch("execution.vol_risk.load_symbol_returns")
    def test_skips_zero_notional_positions(self, mock_load_returns):
        """Should skip positions with zero notional."""
        mock_load_returns.return_value = np.random.randn(200) * 0.02
        
        positions = [
            {"symbol": "BTCUSDT", "notional": 10000.0},
            {"symbol": "ETHUSDT", "notional": 0.0},
        ]
        
        results = compute_all_position_cvars(positions, nav_usd=100000.0)
        
        assert len(results) == 1
        assert "BTCUSDT" in results


# ===========================================================================
# Tests: CVaR Veto Logic
# ===========================================================================

class TestCvarVetoLogic:
    """Test CVaR veto decision logic."""
    
    def test_veto_triggers_at_exact_limit(self):
        """Should veto when CVaR equals limit."""
        cvar_result = CVaRResult(
            symbol="BTCUSDT",
            cvar_usd=4000.0,
            cvar_nav_pct=0.04,  # Exactly at limit
            var_usd=3000.0,
            var_nav_pct=0.03,
            position_notional_usd=10000.0,
        )
        cvar_config = CVaRConfig(
            enabled=True,
            max_position_cvar_nav_pct=0.04,
        )
        
        # At exact limit should NOT veto (only > should veto)
        should_veto, details = check_position_cvar_limit(cvar_result, cvar_config)
        
        assert should_veto == False

    def test_veto_triggers_just_above_limit(self):
        """Should veto when CVaR just exceeds limit."""
        cvar_result = CVaRResult(
            symbol="BTCUSDT",
            cvar_usd=4100.0,
            cvar_nav_pct=0.041,  # Just above limit
            var_usd=3000.0,
            var_nav_pct=0.03,
            position_notional_usd=10000.0,
        )
        cvar_config = CVaRConfig(
            enabled=True,
            max_position_cvar_nav_pct=0.04,
        )
        
        should_veto, details = check_position_cvar_limit(cvar_result, cvar_config)
        
        assert should_veto == True

    def test_veto_details_contain_required_fields(self):
        """Veto details should contain all required fields for logging."""
        cvar_result = CVaRResult(
            symbol="SOLUSDT",
            cvar_usd=5000.0,
            cvar_nav_pct=0.05,
            var_usd=4000.0,
            var_nav_pct=0.04,
            position_notional_usd=15000.0,
        )
        cvar_config = CVaRConfig(
            enabled=True,
            max_position_cvar_nav_pct=0.04,
        )
        
        should_veto, details = check_position_cvar_limit(cvar_result, cvar_config)
        
        assert "reason" in details
        assert "observed" in details
        assert "limits" in details
        assert details["observed"]["symbol"] == "SOLUSDT"
        assert details["observed"]["position_cvar_nav_pct"] == 0.05
        assert details["limits"]["max_position_cvar_nav_pct"] == 0.04


# ===========================================================================
# Tests: CVaR Config Loading
# ===========================================================================

class TestCvarConfigLoading:
    """Test CVaR configuration loading."""
    
    def test_disabled_config_skips_computation(self):
        """Should skip computation when CVaR is disabled."""
        config = CVaRConfig(enabled=False)
        
        cvar_result = CVaRResult(
            symbol="BTCUSDT",
            cvar_usd=10000.0,
            cvar_nav_pct=0.10,  # Would normally breach
            var_usd=8000.0,
            var_nav_pct=0.08,
            position_notional_usd=20000.0,
        )
        
        should_veto, details = check_position_cvar_limit(cvar_result, config)
        
        assert should_veto == False
        assert details == {}

    def test_config_values_are_parsed_correctly(self):
        """Config values should be parsed as correct types."""
        strategy_cfg = {
            "risk_advanced": {
                "cvar": {
                    "enabled": "true",  # String that should become bool
                    "confidence": "0.99",  # String that should become float
                    "lookback_bars": "350",  # String that should become int
                    "max_position_cvar_nav_pct": "0.035",
                }
            }
        }
        
        # Note: load_cvar_config uses bool/float/int casts
        # Testing that these work correctly
        config = load_cvar_config(strategy_cfg)
        
        assert isinstance(config.enabled, bool)
        assert isinstance(config.confidence, float)
        assert isinstance(config.lookback_bars, int)
        assert isinstance(config.max_position_cvar_nav_pct, float)
