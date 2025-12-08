"""
Tests for vol_risk.py — VaR and CVaR calculations (v7.5_A1)
"""

import math
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from execution.vol_risk import (
    VaRConfig,
    CVaRConfig,
    VaRResult,
    CVaRResult,
    compute_ewma_weights,
    compute_ewma_covariance,
    compute_portfolio_var,
    compute_position_cvar,
    load_var_config,
    load_cvar_config,
    check_portfolio_var_limit,
    check_position_cvar_limit,
)


# ===========================================================================
# Tests: EWMA Weights
# ===========================================================================

class TestComputeEwmaWeights:
    def test_weights_sum_to_one(self):
        """EWMA weights should sum to 1.0."""
        weights = compute_ewma_weights(100, halflife=20)
        assert pytest.approx(weights.sum(), rel=1e-6) == 1.0

    def test_recent_observations_have_higher_weight(self):
        """Most recent observations should have higher weight."""
        weights = compute_ewma_weights(50, halflife=10)
        # Last element (most recent) should be larger than first
        assert weights[-1] > weights[0]

    def test_halflife_zero_returns_equal_weights(self):
        """Zero halflife should return equal weights."""
        weights = compute_ewma_weights(10, halflife=0)
        assert pytest.approx(weights[0], rel=1e-6) == 0.1

    def test_weights_length_matches_n(self):
        """Weights array length should match n."""
        weights = compute_ewma_weights(25, halflife=5)
        assert len(weights) == 25


# ===========================================================================
# Tests: EWMA Covariance
# ===========================================================================

class TestComputeEwmaCovariance:
    def test_covariance_matrix_is_square(self):
        """Covariance matrix should be square."""
        returns = np.random.randn(3, 100)  # 3 assets, 100 observations
        cov = compute_ewma_covariance(returns, halflife_bars=20)
        assert cov.shape == (3, 3)

    def test_covariance_matrix_is_symmetric(self):
        """Covariance matrix should be symmetric."""
        returns = np.random.randn(4, 50)
        cov = compute_ewma_covariance(returns, halflife_bars=10)
        assert np.allclose(cov, cov.T)

    def test_diagonal_is_positive(self):
        """Diagonal elements (variances) should be positive."""
        returns = np.random.randn(3, 100) * 0.02  # 2% daily vol
        cov = compute_ewma_covariance(returns, halflife_bars=30)
        assert all(cov[i, i] > 0 for i in range(3))

    def test_insufficient_observations(self):
        """Should handle insufficient observations gracefully."""
        returns = np.random.randn(2, 1)  # Only 1 observation
        cov = compute_ewma_covariance(returns, halflife_bars=20)
        assert cov.shape == (2, 2)
        # Should return default identity-like matrix


# ===========================================================================
# Tests: Portfolio VaR
# ===========================================================================

class TestComputePortfolioVar:
    def test_var_is_positive(self):
        """VaR should be positive (representing potential loss)."""
        np.random.seed(42)
        returns = np.random.randn(3, 200) * 0.02  # 2% daily vol
        weights = np.array([0.4, 0.3, 0.3])
        nav = 100000.0
        
        result = compute_portfolio_var(returns, weights, nav, confidence=0.99, halflife_bars=50)
        
        assert result.var_usd > 0
        assert result.var_nav_pct > 0

    def test_higher_confidence_means_higher_var(self):
        """99% VaR should be higher than 95% VaR."""
        np.random.seed(42)
        returns = np.random.randn(2, 200) * 0.02
        weights = np.array([0.5, 0.5])
        nav = 100000.0
        
        var_99 = compute_portfolio_var(returns, weights, nav, confidence=0.99, halflife_bars=50)
        var_95 = compute_portfolio_var(returns, weights, nav, confidence=0.95, halflife_bars=50)
        
        assert var_99.var_usd > var_95.var_usd

    def test_var_scales_with_nav(self):
        """VaR USD should scale with NAV."""
        np.random.seed(42)
        returns = np.random.randn(2, 100) * 0.02
        weights = np.array([0.5, 0.5])
        
        var_small = compute_portfolio_var(returns, weights, 10000.0, confidence=0.99, halflife_bars=30)
        var_large = compute_portfolio_var(returns, weights, 100000.0, confidence=0.99, halflife_bars=30)
        
        assert pytest.approx(var_large.var_usd / var_small.var_usd, rel=0.01) == 10.0

    def test_var_nav_pct_independent_of_nav(self):
        """VaR as % of NAV should be independent of NAV size."""
        np.random.seed(42)
        returns = np.random.randn(2, 100) * 0.02
        weights = np.array([0.5, 0.5])
        
        var_small = compute_portfolio_var(returns, weights, 10000.0, confidence=0.99, halflife_bars=30)
        var_large = compute_portfolio_var(returns, weights, 100000.0, confidence=0.99, halflife_bars=30)
        
        assert pytest.approx(var_small.var_nav_pct, rel=0.01) == var_large.var_nav_pct

    def test_returns_correct_metadata(self):
        """Result should contain correct metadata."""
        np.random.seed(42)
        returns = np.random.randn(3, 150) * 0.02
        weights = np.array([0.4, 0.3, 0.3])
        
        result = compute_portfolio_var(returns, weights, 100000.0, confidence=0.99, halflife_bars=50)
        
        assert result.n_assets == 3
        assert result.lookback_used == 150
        assert result.confidence == 0.99


# ===========================================================================
# Tests: Position CVaR
# ===========================================================================

class TestComputePositionCvar:
    def test_cvar_is_positive(self):
        """CVaR should be positive (representing expected shortfall)."""
        np.random.seed(42)
        returns = np.random.randn(200) * 0.03  # 3% daily vol
        
        result = compute_position_cvar(returns, nav_usd=100000.0, position_notional_usd=10000.0, confidence=0.95)
        
        assert result.cvar_usd > 0
        assert result.cvar_nav_pct > 0

    def test_cvar_greater_than_var(self):
        """CVaR (Expected Shortfall) should be >= VaR."""
        np.random.seed(42)
        returns = np.random.randn(500) * 0.03
        
        result = compute_position_cvar(returns, nav_usd=100000.0, position_notional_usd=10000.0, confidence=0.95)
        
        assert result.cvar_usd >= result.var_usd

    def test_cvar_scales_with_position_size(self):
        """CVaR should scale with position notional."""
        np.random.seed(42)
        returns = np.random.randn(200) * 0.03
        
        cvar_small = compute_position_cvar(returns, 100000.0, 5000.0, 0.95)
        cvar_large = compute_position_cvar(returns, 100000.0, 10000.0, 0.95)
        
        assert pytest.approx(cvar_large.cvar_usd / cvar_small.cvar_usd, rel=0.01) == 2.0

    def test_insufficient_returns(self):
        """Should handle insufficient returns gracefully."""
        returns = np.random.randn(5) * 0.03  # Only 5 observations
        
        result = compute_position_cvar(returns, 100000.0, 10000.0, 0.95)
        
        assert result.cvar_usd == 0.0
        assert result.lookback_used == 5


# ===========================================================================
# Tests: Config Loaders
# ===========================================================================

class TestLoadVarConfig:
    def test_loads_defaults_without_config(self):
        """Should load default config when file doesn't exist."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            config = load_var_config()
        
        assert config.enabled == True
        assert config.confidence == 0.99
        assert config.lookback_bars == 500

    def test_loads_from_strategy_config(self):
        """Should load from provided strategy config."""
        strategy_cfg = {
            "risk_advanced": {
                "var": {
                    "enabled": True,
                    "confidence": 0.95,
                    "lookback_bars": 300,
                    "halflife_bars": 50,
                    "max_portfolio_var_nav_pct": 0.15,
                }
            }
        }
        
        config = load_var_config(strategy_cfg)
        
        assert config.confidence == 0.95
        assert config.lookback_bars == 300
        assert config.max_portfolio_var_nav_pct == 0.15


class TestLoadCvarConfig:
    def test_loads_defaults_without_config(self):
        """Should load default config when file doesn't exist."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            config = load_cvar_config()
        
        assert config.enabled == True
        assert config.confidence == 0.95
        assert config.max_position_cvar_nav_pct == 0.04

    def test_loads_from_strategy_config(self):
        """Should load from provided strategy config."""
        strategy_cfg = {
            "risk_advanced": {
                "cvar": {
                    "enabled": False,
                    "confidence": 0.99,
                    "lookback_bars": 200,
                    "max_position_cvar_nav_pct": 0.05,
                }
            }
        }
        
        config = load_cvar_config(strategy_cfg)
        
        assert config.enabled == False
        assert config.confidence == 0.99
        assert config.max_position_cvar_nav_pct == 0.05


# ===========================================================================
# Tests: Veto Checks
# ===========================================================================

class TestCheckPortfolioVarLimit:
    def test_no_veto_when_within_limit(self):
        """Should not veto when VaR is within limit."""
        var_result = VaRResult(
            var_usd=5000.0,
            var_nav_pct=0.05,
            portfolio_volatility=0.15,
        )
        var_config = VaRConfig(
            enabled=True,
            max_portfolio_var_nav_pct=0.12,
        )
        
        should_veto, details = check_portfolio_var_limit(var_result, var_config)
        
        assert should_veto == False
        assert details == {}

    def test_veto_when_exceeds_limit(self):
        """Should veto when VaR exceeds limit."""
        var_result = VaRResult(
            var_usd=15000.0,
            var_nav_pct=0.15,
            portfolio_volatility=0.25,
        )
        var_config = VaRConfig(
            enabled=True,
            max_portfolio_var_nav_pct=0.12,
        )
        
        should_veto, details = check_portfolio_var_limit(var_result, var_config)
        
        assert should_veto == True
        assert details["reason"] == "portfolio_var_limit"
        assert details["observed"]["portfolio_var_nav_pct"] == 0.15

    def test_no_veto_when_disabled(self):
        """Should not veto when VaR checking is disabled."""
        var_result = VaRResult(
            var_usd=20000.0,
            var_nav_pct=0.20,
            portfolio_volatility=0.30,
        )
        var_config = VaRConfig(enabled=False)
        
        should_veto, details = check_portfolio_var_limit(var_result, var_config)
        
        assert should_veto == False


class TestCheckPositionCvarLimit:
    def test_no_veto_when_within_limit(self):
        """Should not veto when CVaR is within limit."""
        cvar_result = CVaRResult(
            symbol="BTCUSDT",
            cvar_usd=2000.0,
            cvar_nav_pct=0.02,
            var_usd=1500.0,
            var_nav_pct=0.015,
            position_notional_usd=10000.0,
        )
        cvar_config = CVaRConfig(
            enabled=True,
            max_position_cvar_nav_pct=0.04,
        )
        
        should_veto, details = check_position_cvar_limit(cvar_result, cvar_config)
        
        assert should_veto == False

    def test_veto_when_exceeds_limit(self):
        """Should veto when CVaR exceeds limit."""
        cvar_result = CVaRResult(
            symbol="ETHUSDT",
            cvar_usd=6000.0,
            cvar_nav_pct=0.06,
            var_usd=5000.0,
            var_nav_pct=0.05,
            position_notional_usd=20000.0,
        )
        cvar_config = CVaRConfig(
            enabled=True,
            max_position_cvar_nav_pct=0.04,
        )
        
        should_veto, details = check_position_cvar_limit(cvar_result, cvar_config)
        
        assert should_veto == True
        assert details["reason"] == "position_cvar_limit"
        assert details["observed"]["symbol"] == "ETHUSDT"
