"""
Tests for hybrid_score with factor weights integration (v7.5_C3).

Validates:
- With auto_weighting disabled → hybrid same as legacy sum
- With enabled → hybrid uses weighted factor combination
- Orthogonalized factors used when provided
- Final hybrid respects [-1, 1] clamp
"""
from __future__ import annotations

import pytest

from execution.intel.symbol_score_v6 import hybrid_score


def _mock_expectancy_snapshot(symbol: str) -> dict:
    """Create mock expectancy snapshot."""
    return {
        "by_symbol": {
            symbol: {
                "exp_usd": 10.0,
                "exp_per_notional": 0.001,
                "trades": 10,
            }
        }
    }


def _mock_router_health_snapshot(symbol: str) -> dict:
    """Create mock router health snapshot."""
    return {
        "by_symbol": {
            symbol: {
                "fill_ratio": 0.85,
                "maker_ratio": 0.70,
                "latency_ms": 100,
            }
        }
    }


def _mock_funding_snapshot(symbol: str) -> dict:
    """Create mock funding snapshot."""
    return {
        "by_symbol": {
            symbol: {
                "funding_rate": -0.0001,  # Negative for long-favorable
            }
        }
    }


def _mock_basis_snapshot(symbol: str) -> dict:
    """Create mock basis snapshot."""
    return {
        "by_symbol": {
            symbol: {
                "basis": 0.0005,
            }
        }
    }


class TestHybridScoreFactorWeights:
    """Test suite for hybrid_score with factor weights."""

    def test_without_factor_weights_legacy(self):
        """Without factor_weights, should use legacy calculation."""
        symbol = "BTCUSDT"
        result = hybrid_score(
            symbol=symbol,
            direction="LONG",
            trend_score=0.7,
            expectancy_snapshot=_mock_expectancy_snapshot(symbol),
            router_health_snapshot=_mock_router_health_snapshot(symbol),
            funding_snapshot=_mock_funding_snapshot(symbol),
            basis_snapshot=_mock_basis_snapshot(symbol),
            regime="normal",
            factor_weights=None,  # No factor weights
            orthogonalized_factors=None,
        )
        
        assert "hybrid_score" in result
        assert result["factor_weighting"]["using_factor_weights"] is False
        assert result["factor_weighting"]["factor_weights_used"] == {}

    def test_with_factor_weights_enabled(self):
        """With factor_weights provided, should use weighted combination."""
        symbol = "BTCUSDT"
        factor_weights = {
            "trend": 0.4,
            "carry": 0.3,
            "expectancy": 0.2,
            "router": 0.1,
        }
        
        result = hybrid_score(
            symbol=symbol,
            direction="LONG",
            trend_score=0.7,
            expectancy_snapshot=_mock_expectancy_snapshot(symbol),
            router_health_snapshot=_mock_router_health_snapshot(symbol),
            funding_snapshot=_mock_funding_snapshot(symbol),
            basis_snapshot=_mock_basis_snapshot(symbol),
            regime="normal",
            factor_weights=factor_weights,
            orthogonalized_factors=None,
        )
        
        assert result["factor_weighting"]["using_factor_weights"] is True
        assert result["factor_weighting"]["factor_weights_used"]["trend"] == 0.4
        assert result["factor_weighting"]["factor_weights_used"]["carry"] == 0.3

    def test_with_orthogonalized_factors(self):
        """With orthogonalized_factors, should use those values."""
        symbol = "BTCUSDT"
        factor_weights = {
            "trend": 0.5,
            "carry": 0.5,
        }
        orthogonalized = {
            "BTCUSDT": {
                "trend": 0.9,  # Modified ortho value
                "carry": 0.1,  # Modified ortho value
            }
        }
        
        result = hybrid_score(
            symbol=symbol,
            direction="LONG",
            trend_score=0.7,  # This would be 0.7 raw, but ortho is 0.9
            expectancy_snapshot=_mock_expectancy_snapshot(symbol),
            router_health_snapshot=_mock_router_health_snapshot(symbol),
            funding_snapshot=_mock_funding_snapshot(symbol),
            basis_snapshot=_mock_basis_snapshot(symbol),
            regime="normal",
            factor_weights=factor_weights,
            orthogonalized_factors=orthogonalized,
        )
        
        assert result["factor_weighting"]["using_factor_weights"] is True
        assert result["factor_weighting"]["using_orthogonalized"] is True

    def test_hybrid_score_clamped_to_valid_range(self):
        """Hybrid score should be clamped to [-1, 1]."""
        symbol = "BTCUSDT"
        
        result = hybrid_score(
            symbol=symbol,
            direction="LONG",
            trend_score=0.99,  # Very high
            expectancy_snapshot=_mock_expectancy_snapshot(symbol),
            router_health_snapshot=_mock_router_health_snapshot(symbol),
            funding_snapshot=_mock_funding_snapshot(symbol),
            basis_snapshot=_mock_basis_snapshot(symbol),
            regime="normal",
        )
        
        assert result["hybrid_score"] >= -1.0
        assert result["hybrid_score"] <= 1.0

    def test_empty_factor_weights_uses_legacy(self):
        """Empty factor_weights dict should use legacy calculation."""
        symbol = "BTCUSDT"
        
        result = hybrid_score(
            symbol=symbol,
            direction="LONG",
            trend_score=0.7,
            expectancy_snapshot=_mock_expectancy_snapshot(symbol),
            router_health_snapshot=_mock_router_health_snapshot(symbol),
            funding_snapshot=_mock_funding_snapshot(symbol),
            basis_snapshot=_mock_basis_snapshot(symbol),
            regime="normal",
            factor_weights={},  # Empty
        )
        
        assert result["factor_weighting"]["using_factor_weights"] is False

    def test_factor_vector_included(self):
        """Factor vector should be included in result for diagnostics."""
        symbol = "BTCUSDT"
        
        result = hybrid_score(
            symbol=symbol,
            direction="LONG",
            trend_score=0.7,
            expectancy_snapshot=_mock_expectancy_snapshot(symbol),
            router_health_snapshot=_mock_router_health_snapshot(symbol),
            funding_snapshot=_mock_funding_snapshot(symbol),
            basis_snapshot=_mock_basis_snapshot(symbol),
            regime="normal",
        )
        
        assert "factor_vector" in result
        assert "trend" in result["factor_vector"]
        assert "carry" in result["factor_vector"]
        assert "expectancy" in result["factor_vector"]
        assert "router" in result["factor_vector"]

    def test_short_direction_works(self):
        """SHORT direction should work with factor weights."""
        symbol = "BTCUSDT"
        factor_weights = {"trend": 0.5, "carry": 0.5}
        
        result = hybrid_score(
            symbol=symbol,
            direction="SHORT",
            trend_score=0.6,
            expectancy_snapshot=_mock_expectancy_snapshot(symbol),
            router_health_snapshot=_mock_router_health_snapshot(symbol),
            funding_snapshot=_mock_funding_snapshot(symbol),
            basis_snapshot=_mock_basis_snapshot(symbol),
            regime="normal",
            factor_weights=factor_weights,
        )
        
        assert result["direction"] == "SHORT"
        assert result["factor_weighting"]["using_factor_weights"] is True

    def test_passes_threshold_respects_min_threshold(self):
        """passes_threshold should respect min_threshold."""
        symbol = "BTCUSDT"
        
        result = hybrid_score(
            symbol=symbol,
            direction="LONG",
            trend_score=0.3,  # Low score
            expectancy_snapshot=_mock_expectancy_snapshot(symbol),
            router_health_snapshot=_mock_router_health_snapshot(symbol),
            funding_snapshot=_mock_funding_snapshot(symbol),
            basis_snapshot=_mock_basis_snapshot(symbol),
            regime="normal",
        )
        
        assert "passes_threshold" in result
        assert "min_threshold" in result
        # passes_threshold should be hybrid_score >= min_threshold
        expected = result["hybrid_score"] >= result["min_threshold"]
        assert result["passes_threshold"] == expected
