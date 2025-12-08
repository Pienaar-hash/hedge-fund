"""
Tests for v7.5_C1 — RV Momentum Integration with Symbol Scoring.

Tests:
- RV momentum disabled doesn't affect hybrid score
- Positive RV score increases hybrid score
- Negative RV score decreases hybrid score
- Combined with other modulations still yields clamped [-1, 1]
"""
from __future__ import annotations

import pytest
from unittest.mock import patch

from execution.intel.symbol_score_v6 import (
    hybrid_score,
    HybridScoreConfig,
)


class TestRvMomentumHybridIntegration:
    """Test RV momentum integration with hybrid scoring."""

    @pytest.fixture
    def base_snapshots(self):
        return {
            "expectancy": {
                "symbols": {
                    "BTCUSDT": {"expectancy": 5.0, "hit_rate": 0.6},
                }
            },
            "router_health": {
                "symbols": [
                    {"symbol": "BTCUSDT", "maker_fill_rate": 0.8, "fallback_rate": 0.1}
                ]
            },
            "funding": {
                "symbols": {"BTCUSDT": {"rate": 0.0001}}
            },
            "basis": {
                "symbols": {"BTCUSDT": {"basis_pct": 0.02}}
            },
        }

    @pytest.fixture
    def hybrid_config(self):
        return HybridScoreConfig(
            trend_weight=0.4,
            carry_weight=0.2,
            expectancy_weight=0.2,
            router_weight=0.2,
            min_hybrid_score_long=0.1,
            min_hybrid_score_short=0.1,
        )

    def test_rv_disabled_no_effect(self, base_snapshots, hybrid_config):
        """RV momentum disabled doesn't affect hybrid score."""
        strategy_config = {
            "rv_momentum": {"enabled": False},
            "alpha_decay": {"enabled": False},
            "router_quality": {"enabled": False},
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
        
        assert result["rv_momentum"]["enabled"] is False
        assert result["rv_momentum"]["score"] == 0.0
        assert result["rv_momentum"]["weight"] == 0.0

    def test_positive_rv_increases_hybrid(self, base_snapshots, hybrid_config):
        """Positive RV score increases hybrid score."""
        strategy_config = {
            "rv_momentum": {
                "enabled": True,
                "hybrid_weight": 0.15,
            },
            "alpha_decay": {"enabled": False},
            "router_quality": {"enabled": False},
        }
        
        # First get baseline with no RV
        with patch("execution.rv_momentum.get_rv_score") as mock_rv:
            mock_rv.return_value = 0.0
            
            baseline_result = hybrid_score(
                symbol="BTCUSDT",
                direction="LONG",
                trend_score=0.6,
                expectancy_snapshot=base_snapshots["expectancy"],
                router_health_snapshot=base_snapshots["router_health"],
                funding_snapshot=base_snapshots["funding"],
                basis_snapshot=base_snapshots["basis"],
                regime="normal",
                config=hybrid_config,
                strategy_config=strategy_config,
            )
        
        # Now with positive RV
        with patch("execution.rv_momentum.get_rv_score") as mock_rv:
            mock_rv.return_value = 0.8  # Strong positive RV
            
            positive_result = hybrid_score(
                symbol="BTCUSDT",
                direction="LONG",
                trend_score=0.6,
                expectancy_snapshot=base_snapshots["expectancy"],
                router_health_snapshot=base_snapshots["router_health"],
                funding_snapshot=base_snapshots["funding"],
                basis_snapshot=base_snapshots["basis"],
                regime="normal",
                config=hybrid_config,
                strategy_config=strategy_config,
            )
        
        # Positive RV should increase score
        assert positive_result["hybrid_score"] >= baseline_result["hybrid_score"]
        assert positive_result["rv_momentum"]["score"] == 0.8
        assert positive_result["rv_momentum"]["enabled"] is True

    def test_negative_rv_decreases_hybrid(self, base_snapshots, hybrid_config):
        """Negative RV score decreases hybrid score."""
        strategy_config = {
            "rv_momentum": {
                "enabled": True,
                "hybrid_weight": 0.15,
            },
            "alpha_decay": {"enabled": False},
            "router_quality": {"enabled": False},
        }
        
        # Baseline with zero RV
        with patch("execution.rv_momentum.get_rv_score") as mock_rv:
            mock_rv.return_value = 0.0
            
            baseline_result = hybrid_score(
                symbol="BTCUSDT",
                direction="LONG",
                trend_score=0.6,
                expectancy_snapshot=base_snapshots["expectancy"],
                router_health_snapshot=base_snapshots["router_health"],
                funding_snapshot=base_snapshots["funding"],
                basis_snapshot=base_snapshots["basis"],
                regime="normal",
                config=hybrid_config,
                strategy_config=strategy_config,
            )
        
        # With negative RV
        with patch("execution.rv_momentum.get_rv_score") as mock_rv:
            mock_rv.return_value = -0.7  # Strong negative RV
            
            negative_result = hybrid_score(
                symbol="BTCUSDT",
                direction="LONG",
                trend_score=0.6,
                expectancy_snapshot=base_snapshots["expectancy"],
                router_health_snapshot=base_snapshots["router_health"],
                funding_snapshot=base_snapshots["funding"],
                basis_snapshot=base_snapshots["basis"],
                regime="normal",
                config=hybrid_config,
                strategy_config=strategy_config,
            )
        
        # Negative RV should decrease score
        assert negative_result["hybrid_score"] <= baseline_result["hybrid_score"]
        assert negative_result["rv_momentum"]["score"] == -0.7

    def test_hybrid_clamped_after_rv_modulation(self, base_snapshots, hybrid_config):
        """Hybrid score is clamped to [-1, 1] after RV modulation."""
        strategy_config = {
            "rv_momentum": {
                "enabled": True,
                "hybrid_weight": 0.5,  # Large weight
            },
            "alpha_decay": {"enabled": False},
            "router_quality": {"enabled": False},
        }
        
        # Extreme positive RV
        with patch("execution.rv_momentum.get_rv_score") as mock_rv:
            mock_rv.return_value = 1.0
            
            result = hybrid_score(
                symbol="BTCUSDT",
                direction="LONG",
                trend_score=0.9,  # High trend
                expectancy_snapshot=base_snapshots["expectancy"],
                router_health_snapshot=base_snapshots["router_health"],
                funding_snapshot=base_snapshots["funding"],
                basis_snapshot=base_snapshots["basis"],
                regime="normal",
                config=hybrid_config,
                strategy_config=strategy_config,
            )
        
        assert result["hybrid_score"] <= 1.0

    def test_rv_combined_with_alpha_decay(self, base_snapshots, hybrid_config):
        """RV momentum works with alpha decay."""
        strategy_config = {
            "rv_momentum": {
                "enabled": True,
                "hybrid_weight": 0.10,
            },
            "alpha_decay": {
                "enabled": True,
                "half_life_minutes": 45,
                "min_decay_multiplier": 0.35,
            },
            "router_quality": {"enabled": False},
        }
        
        with patch("execution.rv_momentum.get_rv_score") as mock_rv:
            mock_rv.return_value = 0.5
            
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
        
        # Both should be present in result
        assert "rv_momentum" in result
        assert "alpha_decay" in result
        assert result["rv_momentum"]["enabled"] is True
        assert result["alpha_decay"]["decay_enabled"] is True

    def test_rv_combined_with_router_quality(self, base_snapshots, hybrid_config):
        """RV momentum works with router quality modulation."""
        strategy_config = {
            "rv_momentum": {
                "enabled": True,
                "hybrid_weight": 0.10,
            },
            "alpha_decay": {"enabled": False},
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
        }
        
        with patch("execution.rv_momentum.get_rv_score") as mock_rv:
            with patch("execution.router_metrics.get_router_quality_score") as mock_rq:
                mock_rv.return_value = 0.4
                mock_rq.return_value = 0.85
                
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
        
        # All three should be present
        assert "rv_momentum" in result
        assert "router_quality" in result
        assert result["rv_momentum"]["enabled"] is True
        assert result["router_quality"]["enabled"] is True

    def test_rv_weight_applied_correctly(self, base_snapshots, hybrid_config):
        """RV weight is applied as additive factor."""
        strategy_config = {
            "rv_momentum": {
                "enabled": True,
                "hybrid_weight": 0.10,
            },
            "alpha_decay": {"enabled": False},
            "router_quality": {"enabled": False},
        }
        
        # Test with specific RV score
        with patch("execution.rv_momentum.get_rv_score") as mock_rv:
            mock_rv.return_value = 1.0  # Max RV score
            
            result = hybrid_score(
                symbol="BTCUSDT",
                direction="LONG",
                trend_score=0.5,
                expectancy_snapshot=base_snapshots["expectancy"],
                router_health_snapshot=base_snapshots["router_health"],
                funding_snapshot=base_snapshots["funding"],
                basis_snapshot=base_snapshots["basis"],
                regime="normal",
                config=hybrid_config,
                strategy_config=strategy_config,
            )
        
        assert result["rv_momentum"]["weight"] == 0.10
        # The RV contribution should be weight * score = 0.10 * 1.0 = 0.10
