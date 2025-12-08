"""
Tests for v7.5_B2 — Router Quality integration with Hybrid Scoring.

Tests:
- With router_quality disabled → hybrid unchanged
- With enabled: low rq_score → hybrid scaled by low_quality_multiplier
- With enabled: high rq_score → hybrid scaled by high_quality_multiplier
- Combined with decay/regime logic still results in clamped score
"""
from __future__ import annotations

import pytest
from unittest.mock import patch

from execution.intel.symbol_score_v6 import (
    hybrid_score,
    HybridScoreConfig,
)


class TestHybridScoreRouterQualityIntegration:
    """Test router quality integration in hybrid scoring."""

    @pytest.fixture
    def base_snapshots(self):
        """Base snapshots for testing."""
        return {
            "expectancy": {"symbols": {"BTCUSDT": {"expectancy": 10.0, "hit_rate": 0.55}}},
            "router_health": {"symbols": []},
            "funding": {"symbols": {"BTCUSDT": {"rate": 0.0001}}},
            "basis": {"symbols": {"BTCUSDT": {"basis_pct": 0.005}}},
        }

    @pytest.fixture
    def hybrid_config(self):
        """Hybrid scoring config."""
        return HybridScoreConfig(
            trend_weight=0.40,
            carry_weight=0.25,
            expectancy_weight=0.20,
            router_weight=0.15,
        )

    def test_router_quality_disabled_no_change(self, base_snapshots, hybrid_config):
        """When router_quality is disabled, hybrid score is not modified."""
        strategy_config = {
            "router_quality": {"enabled": False},
            "alpha_decay": {"enabled": False},
        }
        
        result = hybrid_score(
            symbol="BTCUSDT",
            direction="LONG",
            trend_score=0.7,
            expectancy_snapshot=base_snapshots["expectancy"],
            router_health_snapshot=base_snapshots["router_health"],
            funding_snapshot=base_snapshots["funding"],
            basis_snapshot=base_snapshots["basis"],
            regime="normal",
            config=hybrid_config,
            strategy_config=strategy_config,
        )
        
        assert "router_quality" in result
        assert result["router_quality"]["enabled"] is False
        assert result["router_quality"]["multiplier"] == 1.0

    def test_low_quality_applies_multiplier(self, base_snapshots, hybrid_config):
        """Low router quality score applies low_quality_multiplier."""
        strategy_config = {
            "router_quality": {
                "enabled": True,
                "base_score": 0.8,
                "min_score": 0.2,
                "max_score": 1.0,
                "low_quality_threshold": 0.5,
                "high_quality_threshold": 0.9,
                "low_quality_hybrid_multiplier": 0.5,
                "high_quality_hybrid_multiplier": 1.05,
                "slippage_drift_bps_thresholds": {"green": 2.0, "yellow": 6.0},
                "bucket_penalties": {"A_HIGH": 0.0, "B_MEDIUM": -0.05, "C_LOW": -0.15},
                "twap_skip_penalty": 0.10,
            },
            "alpha_decay": {"enabled": False},
        }
        
        # Mock get_router_quality_score to return low score
        with patch("execution.router_metrics.get_router_quality_score") as mock_rq:
            mock_rq.return_value = 0.3  # Below low_quality_threshold
            
            result = hybrid_score(
                symbol="BTCUSDT",
                direction="LONG",
                trend_score=0.7,
                expectancy_snapshot=base_snapshots["expectancy"],
                router_health_snapshot=base_snapshots["router_health"],
                funding_snapshot=base_snapshots["funding"],
                basis_snapshot=base_snapshots["basis"],
                regime="normal",
                config=hybrid_config,
                strategy_config=strategy_config,
            )
            
            assert result["router_quality"]["enabled"] is True
            assert result["router_quality"]["score"] == 0.3
            assert result["router_quality"]["multiplier"] == 0.5

    def test_high_quality_applies_multiplier(self, base_snapshots, hybrid_config):
        """High router quality score applies high_quality_multiplier."""
        strategy_config = {
            "router_quality": {
                "enabled": True,
                "base_score": 0.8,
                "min_score": 0.2,
                "max_score": 1.0,
                "low_quality_threshold": 0.5,
                "high_quality_threshold": 0.9,
                "low_quality_hybrid_multiplier": 0.5,
                "high_quality_hybrid_multiplier": 1.05,
                "slippage_drift_bps_thresholds": {"green": 2.0, "yellow": 6.0},
                "bucket_penalties": {"A_HIGH": 0.0, "B_MEDIUM": -0.05, "C_LOW": -0.15},
                "twap_skip_penalty": 0.10,
            },
            "alpha_decay": {"enabled": False},
        }
        
        # Mock get_router_quality_score to return high score
        with patch("execution.router_metrics.get_router_quality_score") as mock_rq:
            mock_rq.return_value = 0.95  # Above high_quality_threshold
            
            result = hybrid_score(
                symbol="BTCUSDT",
                direction="LONG",
                trend_score=0.7,
                expectancy_snapshot=base_snapshots["expectancy"],
                router_health_snapshot=base_snapshots["router_health"],
                funding_snapshot=base_snapshots["funding"],
                basis_snapshot=base_snapshots["basis"],
                regime="normal",
                config=hybrid_config,
                strategy_config=strategy_config,
            )
            
            assert result["router_quality"]["enabled"] is True
            assert result["router_quality"]["score"] == 0.95
            assert result["router_quality"]["multiplier"] == 1.05

    def test_mid_quality_no_multiplier(self, base_snapshots, hybrid_config):
        """Mid-range router quality score leaves hybrid unchanged."""
        strategy_config = {
            "router_quality": {
                "enabled": True,
                "base_score": 0.8,
                "min_score": 0.2,
                "max_score": 1.0,
                "low_quality_threshold": 0.5,
                "high_quality_threshold": 0.9,
                "low_quality_hybrid_multiplier": 0.5,
                "high_quality_hybrid_multiplier": 1.05,
                "slippage_drift_bps_thresholds": {"green": 2.0, "yellow": 6.0},
                "bucket_penalties": {"A_HIGH": 0.0, "B_MEDIUM": -0.05, "C_LOW": -0.15},
                "twap_skip_penalty": 0.10,
            },
            "alpha_decay": {"enabled": False},
        }
        
        # Mock get_router_quality_score to return mid-range score
        with patch("execution.router_metrics.get_router_quality_score") as mock_rq:
            mock_rq.return_value = 0.7  # Between thresholds
            
            result = hybrid_score(
                symbol="BTCUSDT",
                direction="LONG",
                trend_score=0.7,
                expectancy_snapshot=base_snapshots["expectancy"],
                router_health_snapshot=base_snapshots["router_health"],
                funding_snapshot=base_snapshots["funding"],
                basis_snapshot=base_snapshots["basis"],
                regime="normal",
                config=hybrid_config,
                strategy_config=strategy_config,
            )
            
            assert result["router_quality"]["enabled"] is True
            assert result["router_quality"]["score"] == 0.7
            assert result["router_quality"]["multiplier"] == 1.0

    def test_hybrid_score_clamped_after_modulation(self, base_snapshots, hybrid_config):
        """Hybrid score is clamped to [-1, 1] after router quality modulation."""
        strategy_config = {
            "router_quality": {
                "enabled": True,
                "base_score": 0.8,
                "min_score": 0.2,
                "max_score": 1.0,
                "low_quality_threshold": 0.5,
                "high_quality_threshold": 0.9,
                "low_quality_hybrid_multiplier": 0.5,
                "high_quality_hybrid_multiplier": 2.0,  # Very high multiplier
                "slippage_drift_bps_thresholds": {"green": 2.0, "yellow": 6.0},
                "bucket_penalties": {"A_HIGH": 0.0, "B_MEDIUM": -0.05, "C_LOW": -0.15},
                "twap_skip_penalty": 0.10,
            },
            "alpha_decay": {"enabled": False},
        }
        
        # Mock get_router_quality_score to return high score
        with patch("execution.router_metrics.get_router_quality_score") as mock_rq:
            mock_rq.return_value = 0.95
            
            result = hybrid_score(
                symbol="BTCUSDT",
                direction="LONG",
                trend_score=0.9,  # High trend score
                expectancy_snapshot=base_snapshots["expectancy"],
                router_health_snapshot=base_snapshots["router_health"],
                funding_snapshot=base_snapshots["funding"],
                basis_snapshot=base_snapshots["basis"],
                regime="normal",
                config=hybrid_config,
                strategy_config=strategy_config,
            )
            
            # Even with 2.0 multiplier, should be clamped to 1.0
            assert result["hybrid_score"] <= 1.0

    def test_combined_with_alpha_decay(self, base_snapshots, hybrid_config):
        """Router quality modulation works with alpha decay."""
        strategy_config = {
            "router_quality": {
                "enabled": True,
                "base_score": 0.8,
                "min_score": 0.2,
                "max_score": 1.0,
                "low_quality_threshold": 0.5,
                "high_quality_threshold": 0.9,
                "low_quality_hybrid_multiplier": 0.5,
                "high_quality_hybrid_multiplier": 1.05,
                "slippage_drift_bps_thresholds": {"green": 2.0, "yellow": 6.0},
                "bucket_penalties": {"A_HIGH": 0.0, "B_MEDIUM": -0.05, "C_LOW": -0.15},
                "twap_skip_penalty": 0.10,
            },
            "alpha_decay": {
                "enabled": True,
                "half_life_minutes": 45,
                "min_decay_multiplier": 0.35,
            },
        }
        
        with patch("execution.router_metrics.get_router_quality_score") as mock_rq:
            mock_rq.return_value = 0.8  # Mid-range, no multiplier
            
            result = hybrid_score(
                symbol="BTCUSDT",
                direction="LONG",
                trend_score=0.7,
                expectancy_snapshot=base_snapshots["expectancy"],
                router_health_snapshot=base_snapshots["router_health"],
                funding_snapshot=base_snapshots["funding"],
                basis_snapshot=base_snapshots["basis"],
                regime="normal",
                config=hybrid_config,
                strategy_config=strategy_config,
            )
            
            # Both decay and router quality metadata present
            assert "alpha_decay" in result
            assert "router_quality" in result
            assert result["alpha_decay"]["decay_enabled"] is True
            assert result["router_quality"]["enabled"] is True

    def test_import_error_graceful(self, base_snapshots, hybrid_config):
        """If router_metrics import fails, hybrid scoring continues."""
        strategy_config = {
            "router_quality": {"enabled": True},
            "alpha_decay": {"enabled": False},
        }
        
        with patch.dict("sys.modules", {"execution.router_metrics": None}):
            # This should not raise - import error is caught
            result = hybrid_score(
                symbol="BTCUSDT",
                direction="LONG",
                trend_score=0.7,
                expectancy_snapshot=base_snapshots["expectancy"],
                router_health_snapshot=base_snapshots["router_health"],
                funding_snapshot=base_snapshots["funding"],
                basis_snapshot=base_snapshots["basis"],
                regime="normal",
                config=hybrid_config,
                strategy_config=strategy_config,
            )
            
            # Should still have result
            assert "hybrid_score" in result
