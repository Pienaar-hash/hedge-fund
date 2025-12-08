"""
Tests for v7.5_C2 factor normalization.

Verifies:
- Z-score normalization: mean ~ 0, std ~ 1
- Clamp to ±max_abs_zscore
- Minmax normalization: values in [0, 1]
- Edge cases: zero variance, empty vectors
"""
import pytest
import numpy as np
from typing import List


class TestZscoreNormalization:
    """Test z-score normalization mode."""
    
    def test_zscore_produces_mean_near_zero(self):
        """Z-score normalized factors should have mean near 0."""
        from execution.intel.symbol_score_v6 import FactorVector, build_factor_vector
        from execution.factor_diagnostics import normalize_factor_vectors
        
        # Create vectors with varying trend values
        vectors = [
            build_factor_vector("BTC", {"trend": 0.2}, 0.5),
            build_factor_vector("ETH", {"trend": 0.5}, 0.5),
            build_factor_vector("SOL", {"trend": 0.8}, 0.5),
            build_factor_vector("DOGE", {"trend": 1.0}, 0.5),
        ]
        
        normalized = normalize_factor_vectors(
            vectors=vectors,
            factor_names=["trend"],
            mode="zscore",
            max_abs_zscore=3.0,
        )
        
        trend_values = [nv.factors["trend"] for nv in normalized]
        mean = np.mean(trend_values)
        
        # Mean should be very close to 0
        assert abs(mean) < 0.01, f"Mean should be ~0, got {mean}"
    
    def test_zscore_produces_std_near_one(self):
        """Z-score normalized factors should have std near 1."""
        from execution.intel.symbol_score_v6 import build_factor_vector
        from execution.factor_diagnostics import normalize_factor_vectors
        
        # Create vectors with varying trend values
        vectors = [
            build_factor_vector("BTC", {"trend": 0.2}, 0.5),
            build_factor_vector("ETH", {"trend": 0.5}, 0.5),
            build_factor_vector("SOL", {"trend": 0.8}, 0.5),
            build_factor_vector("DOGE", {"trend": 1.0}, 0.5),
        ]
        
        normalized = normalize_factor_vectors(
            vectors=vectors,
            factor_names=["trend"],
            mode="zscore",
            max_abs_zscore=3.0,
        )
        
        trend_values = [nv.factors["trend"] for nv in normalized]
        std = np.std(trend_values)
        
        # Std should be close to 1 (might be slightly less due to clamping)
        assert 0.5 < std < 1.5, f"Std should be ~1, got {std}"
    
    def test_zscore_clamps_to_max_abs(self):
        """Z-score values should be clamped to ±max_abs_zscore."""
        from execution.intel.symbol_score_v6 import build_factor_vector
        from execution.factor_diagnostics import normalize_factor_vectors
        
        # Create vectors with extreme outlier
        vectors = [
            build_factor_vector("BTC", {"trend": 0.5}, 0.5),
            build_factor_vector("ETH", {"trend": 0.5}, 0.5),
            build_factor_vector("SOL", {"trend": 0.5}, 0.5),
            build_factor_vector("OUTLIER", {"trend": 100.0}, 0.5),  # Extreme outlier
        ]
        
        max_zscore = 2.0
        normalized = normalize_factor_vectors(
            vectors=vectors,
            factor_names=["trend"],
            mode="zscore",
            max_abs_zscore=max_zscore,
        )
        
        for nv in normalized:
            z = nv.factors["trend"]
            assert -max_zscore <= z <= max_zscore, f"Z-score {z} exceeds ±{max_zscore}"
    
    def test_zscore_handles_zero_variance(self):
        """Z-score handles zero variance (all same values) gracefully."""
        from execution.intel.symbol_score_v6 import build_factor_vector
        from execution.factor_diagnostics import normalize_factor_vectors
        
        # All same value -> zero variance
        vectors = [
            build_factor_vector("BTC", {"trend": 0.5}, 0.5),
            build_factor_vector("ETH", {"trend": 0.5}, 0.5),
            build_factor_vector("SOL", {"trend": 0.5}, 0.5),
        ]
        
        normalized = normalize_factor_vectors(
            vectors=vectors,
            factor_names=["trend"],
            mode="zscore",
            max_abs_zscore=3.0,
        )
        
        # All should be 0 (x - mean = 0)
        for nv in normalized:
            assert nv.factors["trend"] == 0.0


class TestMinmaxNormalization:
    """Test minmax normalization mode."""
    
    def test_minmax_produces_range_zero_to_one(self):
        """Minmax normalized factors should be in [0, 1]."""
        from execution.intel.symbol_score_v6 import build_factor_vector
        from execution.factor_diagnostics import normalize_factor_vectors
        
        vectors = [
            build_factor_vector("BTC", {"trend": 0.2}, 0.5),
            build_factor_vector("ETH", {"trend": 0.5}, 0.5),
            build_factor_vector("SOL", {"trend": 0.8}, 0.5),
            build_factor_vector("DOGE", {"trend": 1.0}, 0.5),
        ]
        
        normalized = normalize_factor_vectors(
            vectors=vectors,
            factor_names=["trend"],
            mode="minmax",
            max_abs_zscore=3.0,
        )
        
        for nv in normalized:
            val = nv.factors["trend"]
            assert 0.0 <= val <= 1.0, f"Minmax value {val} not in [0, 1]"
    
    def test_minmax_min_maps_to_zero(self):
        """Minimum value should map to 0."""
        from execution.intel.symbol_score_v6 import build_factor_vector
        from execution.factor_diagnostics import normalize_factor_vectors
        
        vectors = [
            build_factor_vector("MIN", {"trend": 0.1}, 0.5),
            build_factor_vector("MID", {"trend": 0.5}, 0.5),
            build_factor_vector("MAX", {"trend": 0.9}, 0.5),
        ]
        
        normalized = normalize_factor_vectors(
            vectors=vectors,
            factor_names=["trend"],
            mode="minmax",
            max_abs_zscore=3.0,
        )
        
        min_vec = [nv for nv in normalized if nv.symbol == "MIN"][0]
        assert min_vec.factors["trend"] == 0.0
    
    def test_minmax_max_maps_to_one(self):
        """Maximum value should map to 1."""
        from execution.intel.symbol_score_v6 import build_factor_vector
        from execution.factor_diagnostics import normalize_factor_vectors
        
        vectors = [
            build_factor_vector("MIN", {"trend": 0.1}, 0.5),
            build_factor_vector("MID", {"trend": 0.5}, 0.5),
            build_factor_vector("MAX", {"trend": 0.9}, 0.5),
        ]
        
        normalized = normalize_factor_vectors(
            vectors=vectors,
            factor_names=["trend"],
            mode="minmax",
            max_abs_zscore=3.0,
        )
        
        max_vec = [nv for nv in normalized if nv.symbol == "MAX"][0]
        assert max_vec.factors["trend"] == 1.0
    
    def test_minmax_handles_zero_range(self):
        """Minmax handles zero range (all same values) gracefully."""
        from execution.intel.symbol_score_v6 import build_factor_vector
        from execution.factor_diagnostics import normalize_factor_vectors
        
        # All same value -> zero range
        vectors = [
            build_factor_vector("BTC", {"trend": 0.5}, 0.5),
            build_factor_vector("ETH", {"trend": 0.5}, 0.5),
        ]
        
        normalized = normalize_factor_vectors(
            vectors=vectors,
            factor_names=["trend"],
            mode="minmax",
            max_abs_zscore=3.0,
        )
        
        # All should be 0 when range is zero
        for nv in normalized:
            assert nv.factors["trend"] == 0.0


class TestNormalizationEdgeCases:
    """Test edge cases for normalization."""
    
    def test_empty_vectors_returns_empty(self):
        """Empty input returns empty output."""
        from execution.factor_diagnostics import normalize_factor_vectors
        
        normalized = normalize_factor_vectors(
            vectors=[],
            factor_names=["trend"],
            mode="zscore",
            max_abs_zscore=3.0,
        )
        
        assert normalized == []
    
    def test_single_vector_zscore(self):
        """Single vector z-score should produce 0."""
        from execution.intel.symbol_score_v6 import build_factor_vector
        from execution.factor_diagnostics import normalize_factor_vectors
        
        vectors = [build_factor_vector("BTC", {"trend": 0.8}, 0.5)]
        
        normalized = normalize_factor_vectors(
            vectors=vectors,
            factor_names=["trend"],
            mode="zscore",
            max_abs_zscore=3.0,
        )
        
        assert len(normalized) == 1
        # Single value -> z-score is 0 (x - x = 0)
        assert normalized[0].factors["trend"] == 0.0
    
    def test_missing_factor_defaults_to_zero(self):
        """Missing factors in input default to 0.0."""
        from execution.intel.symbol_score_v6 import build_factor_vector
        from execution.factor_diagnostics import normalize_factor_vectors
        
        # Vector with only "trend", normalize with "trend" and "carry"
        vectors = [
            build_factor_vector("BTC", {"trend": 0.8}, 0.5),
            build_factor_vector("ETH", {"trend": 0.5, "carry": 0.3}, 0.5),
        ]
        
        normalized = normalize_factor_vectors(
            vectors=vectors,
            factor_names=["trend", "carry"],
            mode="minmax",
            max_abs_zscore=3.0,
        )
        
        # BTC should have carry normalized (originally 0.0)
        btc = [nv for nv in normalized if nv.symbol == "BTC"][0]
        assert "carry" in btc.factors
    
    def test_preserves_direction_field(self):
        """Normalization preserves direction field."""
        from execution.intel.symbol_score_v6 import build_factor_vector
        from execution.factor_diagnostics import normalize_factor_vectors
        
        vectors = [
            build_factor_vector("BTC", {"trend": 0.5}, 0.5, direction="SHORT"),
        ]
        
        normalized = normalize_factor_vectors(
            vectors=vectors,
            factor_names=["trend"],
            mode="zscore",
            max_abs_zscore=3.0,
        )
        
        assert normalized[0].direction == "SHORT"
    
    def test_multiple_factors_normalized_independently(self):
        """Each factor is normalized independently."""
        from execution.intel.symbol_score_v6 import build_factor_vector
        from execution.factor_diagnostics import normalize_factor_vectors
        
        vectors = [
            build_factor_vector("A", {"trend": 0.0, "carry": 100.0}, 0.5),
            build_factor_vector("B", {"trend": 1.0, "carry": 200.0}, 0.5),
        ]
        
        normalized = normalize_factor_vectors(
            vectors=vectors,
            factor_names=["trend", "carry"],
            mode="minmax",
            max_abs_zscore=3.0,
        )
        
        a = [nv for nv in normalized if nv.symbol == "A"][0]
        b = [nv for nv in normalized if nv.symbol == "B"][0]
        
        # A should have trend=0, carry=0 (min values)
        assert a.factors["trend"] == 0.0
        assert a.factors["carry"] == 0.0
        
        # B should have trend=1, carry=1 (max values)
        assert b.factors["trend"] == 1.0
        assert b.factors["carry"] == 1.0
