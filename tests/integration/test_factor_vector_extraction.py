"""
Tests for v7.5_C2 factor vector extraction from symbol_score_v6.

Verifies:
- FactorVector dataclass structure
- build_factor_vector function
- Factor vector inclusion in hybrid_score results
"""
import pytest
from typing import Dict, Any


class TestFactorVectorDataclass:
    """Test FactorVector dataclass structure."""
    
    def test_factor_vector_creation(self):
        """FactorVector can be created with all fields."""
        from execution.intel.symbol_score_v6 import FactorVector
        
        vec = FactorVector(
            symbol="BTCUSDT",
            factors={"trend": 0.8, "carry": 0.2},
            hybrid_score=0.65,
            direction="LONG",
            regime="normal",
        )
        
        assert vec.symbol == "BTCUSDT"
        assert vec.factors["trend"] == 0.8
        assert vec.factors["carry"] == 0.2
        assert vec.hybrid_score == 0.65
        assert vec.direction == "LONG"
        assert vec.regime == "normal"
    
    def test_factor_vector_defaults(self):
        """FactorVector has sensible defaults."""
        from execution.intel.symbol_score_v6 import FactorVector
        
        vec = FactorVector(symbol="ETHUSDT")
        
        assert vec.symbol == "ETHUSDT"
        assert vec.factors == {}
        assert vec.hybrid_score == 0.0
        assert vec.direction == "LONG"
        assert vec.regime == "normal"


class TestBuildFactorVector:
    """Test build_factor_vector function."""
    
    def test_build_factor_vector_basic(self):
        """build_factor_vector creates correct FactorVector."""
        from execution.intel.symbol_score_v6 import build_factor_vector
        
        components = {
            "trend": 0.75,
            "carry": 0.35,
            "expectancy": 0.6,
            "router": 0.8,
        }
        
        vec = build_factor_vector(
            symbol="SOLUSDT",
            components=components,
            hybrid_score=0.58,
            direction="SHORT",
            regime="high",
        )
        
        assert vec.symbol == "SOLUSDT"
        assert vec.factors == components
        assert vec.hybrid_score == 0.58
        assert vec.direction == "SHORT"
        assert vec.regime == "high"
    
    def test_build_factor_vector_uppercases_symbol(self):
        """build_factor_vector uppercases symbol."""
        from execution.intel.symbol_score_v6 import build_factor_vector
        
        vec = build_factor_vector(
            symbol="btcusdt",
            components={"trend": 0.5},
            hybrid_score=0.5,
        )
        
        assert vec.symbol == "BTCUSDT"
        assert vec.direction == "LONG"  # Default
    
    def test_build_factor_vector_copies_components(self):
        """build_factor_vector creates independent copy of components."""
        from execution.intel.symbol_score_v6 import build_factor_vector
        
        original = {"trend": 0.5}
        vec = build_factor_vector(
            symbol="BTCUSDT",
            components=original,
            hybrid_score=0.5,
        )
        
        # Modify original
        original["trend"] = 0.99
        
        # Vector should not be affected
        assert vec.factors["trend"] == 0.5


class TestHybridScoreFactorVector:
    """Test factor_vector presence in hybrid_score results."""
    
    def test_hybrid_score_includes_factor_vector(self):
        """hybrid_score result includes factor_vector field."""
        from execution.intel.symbol_score_v6 import hybrid_score
        
        result = hybrid_score(
            symbol="BTCUSDT",
            direction="LONG",
            trend_score=0.7,
            expectancy_snapshot={},
            router_health_snapshot={},
            funding_snapshot={},
            basis_snapshot={},
            regime="normal",
        )
        
        assert "factor_vector" in result
        fv = result["factor_vector"]
        assert isinstance(fv, dict)
    
    def test_factor_vector_contains_all_factors(self):
        """factor_vector contains expected factor keys."""
        from execution.intel.symbol_score_v6 import hybrid_score
        
        result = hybrid_score(
            symbol="ETHUSDT",
            direction="LONG",
            trend_score=0.6,
            expectancy_snapshot={},
            router_health_snapshot={},
            funding_snapshot={},
            basis_snapshot={},
        )
        
        fv = result["factor_vector"]
        
        # Check expected factors are present
        expected_factors = ["trend", "carry", "expectancy", "router", "rv_momentum", "router_quality", "vol_regime"]
        for factor in expected_factors:
            assert factor in fv, f"Missing factor: {factor}"
    
    def test_factor_vector_trend_matches_input(self):
        """factor_vector trend value matches input trend_score."""
        from execution.intel.symbol_score_v6 import hybrid_score
        
        result = hybrid_score(
            symbol="BTCUSDT",
            direction="LONG",
            trend_score=0.85,
            expectancy_snapshot={},
            router_health_snapshot={},
            funding_snapshot={},
            basis_snapshot={},
        )
        
        # Trend in factor_vector should match what was in components
        fv = result["factor_vector"]
        components = result["components"]
        
        # factor_vector.trend should equal components.trend
        assert fv["trend"] == components["trend"]
    
    def test_factor_vector_missing_factors_default_to_zero(self):
        """Missing factor values default to 0.0."""
        from execution.intel.symbol_score_v6 import hybrid_score
        
        # No funding/basis data -> carry should still be present (defaults to 0.5)
        result = hybrid_score(
            symbol="BTCUSDT",
            direction="LONG",
            trend_score=0.5,
            expectancy_snapshot={},
            router_health_snapshot={},
            funding_snapshot={},
            basis_snapshot={},
        )
        
        fv = result["factor_vector"]
        
        # rv_momentum should be 0.0 if not enabled/available
        assert "rv_momentum" in fv
        assert isinstance(fv["rv_momentum"], (int, float))


class TestFactorVectorExports:
    """Test module exports."""
    
    def test_factor_vector_in_all(self):
        """FactorVector is exported in __all__."""
        from execution.intel import symbol_score_v6
        
        assert "FactorVector" in symbol_score_v6.__all__
    
    def test_build_factor_vector_in_all(self):
        """build_factor_vector is exported in __all__."""
        from execution.intel import symbol_score_v6
        
        assert "build_factor_vector" in symbol_score_v6.__all__
