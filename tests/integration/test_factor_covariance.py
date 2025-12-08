"""
Tests for v7.5_C2 factor covariance and correlation computation.

Verifies:
- Covariance matrix computation
- Correlation matrix computation
- Factor volatilities from diagonal
- Known relationships (e.g., B = 2*A -> correlation ~1)
"""
import pytest
import numpy as np


class TestCovarianceComputation:
    """Test covariance matrix computation."""
    
    def test_covariance_shape_matches_factors(self):
        """Covariance matrix shape should be (F, F)."""
        from execution.intel.symbol_score_v6 import build_factor_vector
        from execution.factor_diagnostics import compute_factor_covariance
        
        vectors = [
            build_factor_vector("BTC", {"a": 0.5, "b": 0.3, "c": 0.7}, 0.5),
            build_factor_vector("ETH", {"a": 0.6, "b": 0.4, "c": 0.8}, 0.5),
            build_factor_vector("SOL", {"a": 0.7, "b": 0.5, "c": 0.9}, 0.5),
        ]
        
        result = compute_factor_covariance(vectors, ["a", "b", "c"])
        
        assert result.covariance.shape == (3, 3)
        assert result.correlation.shape == (3, 3)
        assert len(result.factors) == 3
    
    def test_covariance_symmetric(self):
        """Covariance matrix should be symmetric."""
        from execution.intel.symbol_score_v6 import build_factor_vector
        from execution.factor_diagnostics import compute_factor_covariance
        
        vectors = [
            build_factor_vector("A", {"x": 0.1, "y": 0.9}, 0.5),
            build_factor_vector("B", {"x": 0.5, "y": 0.5}, 0.5),
            build_factor_vector("C", {"x": 0.9, "y": 0.1}, 0.5),
        ]
        
        result = compute_factor_covariance(vectors, ["x", "y"])
        
        # Check symmetry: cov[i,j] == cov[j,i]
        assert np.allclose(result.covariance, result.covariance.T)
    
    def test_diagonal_is_variance(self):
        """Diagonal of covariance should be factor variances."""
        from execution.intel.symbol_score_v6 import build_factor_vector
        from execution.factor_diagnostics import compute_factor_covariance
        
        # Known values
        vectors = [
            build_factor_vector("A", {"x": 1.0}, 0.5),
            build_factor_vector("B", {"x": 2.0}, 0.5),
            build_factor_vector("C", {"x": 3.0}, 0.5),
        ]
        
        result = compute_factor_covariance(vectors, ["x"])
        
        # Variance of [1, 2, 3] = 1.0 (sample variance with ddof=1)
        expected_var = np.var([1.0, 2.0, 3.0], ddof=1)
        assert np.isclose(result.covariance[0, 0], expected_var)


class TestCorrelationComputation:
    """Test correlation matrix computation."""
    
    def test_correlation_diagonal_is_one(self):
        """Diagonal of correlation matrix should be 1.0."""
        from execution.intel.symbol_score_v6 import build_factor_vector
        from execution.factor_diagnostics import compute_factor_covariance
        
        vectors = [
            build_factor_vector("A", {"x": 0.1, "y": 0.9}, 0.5),
            build_factor_vector("B", {"x": 0.5, "y": 0.5}, 0.5),
            build_factor_vector("C", {"x": 0.9, "y": 0.1}, 0.5),
        ]
        
        result = compute_factor_covariance(vectors, ["x", "y"])
        
        # Diagonal should be 1.0
        for i in range(len(result.factors)):
            assert np.isclose(result.correlation[i, i], 1.0)
    
    def test_correlation_in_valid_range(self):
        """Correlation values should be in [-1, 1]."""
        from execution.intel.symbol_score_v6 import build_factor_vector
        from execution.factor_diagnostics import compute_factor_covariance
        
        np.random.seed(42)
        vectors = [
            build_factor_vector(f"S{i}", {
                "a": np.random.random(),
                "b": np.random.random(),
            }, 0.5)
            for i in range(10)
        ]
        
        result = compute_factor_covariance(vectors, ["a", "b"])
        
        for i in range(len(result.factors)):
            for j in range(len(result.factors)):
                assert -1.0 <= result.correlation[i, j] <= 1.0
    
    def test_perfectly_correlated_factors(self):
        """Perfectly correlated factors should have correlation ~1."""
        from execution.intel.symbol_score_v6 import build_factor_vector
        from execution.factor_diagnostics import compute_factor_covariance
        
        # b = 2 * a (perfectly correlated)
        vectors = [
            build_factor_vector("A", {"a": 1.0, "b": 2.0}, 0.5),
            build_factor_vector("B", {"a": 2.0, "b": 4.0}, 0.5),
            build_factor_vector("C", {"a": 3.0, "b": 6.0}, 0.5),
            build_factor_vector("D", {"a": 4.0, "b": 8.0}, 0.5),
        ]
        
        result = compute_factor_covariance(vectors, ["a", "b"])
        
        # Correlation between a and b should be 1.0
        corr_ab = result.correlation[0, 1]
        assert np.isclose(corr_ab, 1.0, atol=0.01), f"Expected ~1.0, got {corr_ab}"
    
    def test_negatively_correlated_factors(self):
        """Negatively correlated factors should have correlation ~-1."""
        from execution.intel.symbol_score_v6 import build_factor_vector
        from execution.factor_diagnostics import compute_factor_covariance
        
        # b = -a (perfectly negatively correlated)
        vectors = [
            build_factor_vector("A", {"a": 1.0, "b": -1.0}, 0.5),
            build_factor_vector("B", {"a": 2.0, "b": -2.0}, 0.5),
            build_factor_vector("C", {"a": 3.0, "b": -3.0}, 0.5),
            build_factor_vector("D", {"a": 4.0, "b": -4.0}, 0.5),
        ]
        
        result = compute_factor_covariance(vectors, ["a", "b"])
        
        corr_ab = result.correlation[0, 1]
        assert np.isclose(corr_ab, -1.0, atol=0.01), f"Expected ~-1.0, got {corr_ab}"
    
    def test_independent_factors_near_zero_correlation(self):
        """Independent random factors should have correlation near 0."""
        from execution.intel.symbol_score_v6 import build_factor_vector
        from execution.factor_diagnostics import compute_factor_covariance
        
        # Generate independent random values
        np.random.seed(123)
        a_vals = np.random.random(50)
        b_vals = np.random.random(50)  # Independent of a
        
        vectors = [
            build_factor_vector(f"S{i}", {"a": a_vals[i], "b": b_vals[i]}, 0.5)
            for i in range(50)
        ]
        
        result = compute_factor_covariance(vectors, ["a", "b"])
        
        corr_ab = result.correlation[0, 1]
        # Should be close to 0 (within ±0.3 for random data)
        assert abs(corr_ab) < 0.3, f"Expected ~0, got {corr_ab}"


class TestFactorVolatilities:
    """Test factor volatility computation."""
    
    def test_factor_vols_all_present(self):
        """Factor volatilities should be present for all factors."""
        from execution.intel.symbol_score_v6 import build_factor_vector
        from execution.factor_diagnostics import compute_factor_covariance
        
        vectors = [
            build_factor_vector("A", {"x": 0.1, "y": 0.5, "z": 0.9}, 0.5),
            build_factor_vector("B", {"x": 0.5, "y": 0.5, "z": 0.5}, 0.5),
            build_factor_vector("C", {"x": 0.9, "y": 0.5, "z": 0.1}, 0.5),
        ]
        
        result = compute_factor_covariance(vectors, ["x", "y", "z"])
        
        assert "x" in result.factor_vols
        assert "y" in result.factor_vols
        assert "z" in result.factor_vols
    
    def test_factor_vols_from_diagonal(self):
        """Factor vols should equal sqrt of covariance diagonal."""
        from execution.intel.symbol_score_v6 import build_factor_vector
        from execution.factor_diagnostics import compute_factor_covariance
        
        vectors = [
            build_factor_vector("A", {"x": 1.0}, 0.5),
            build_factor_vector("B", {"x": 2.0}, 0.5),
            build_factor_vector("C", {"x": 3.0}, 0.5),
        ]
        
        result = compute_factor_covariance(vectors, ["x"])
        
        # Vol should be sqrt of variance
        expected_vol = np.sqrt(result.covariance[0, 0])
        assert np.isclose(result.factor_vols["x"], expected_vol)
    
    def test_zero_variance_produces_zero_vol(self):
        """Zero variance factor should have zero volatility."""
        from execution.intel.symbol_score_v6 import build_factor_vector
        from execution.factor_diagnostics import compute_factor_covariance
        
        # All same values -> zero variance
        vectors = [
            build_factor_vector("A", {"x": 0.5}, 0.5),
            build_factor_vector("B", {"x": 0.5}, 0.5),
            build_factor_vector("C", {"x": 0.5}, 0.5),
        ]
        
        result = compute_factor_covariance(vectors, ["x"])
        
        assert result.factor_vols["x"] == 0.0


class TestCovarianceEdgeCases:
    """Test edge cases for covariance computation."""
    
    def test_single_symbol_returns_zeros(self):
        """Single symbol should return zero matrices (need 2+ for covariance)."""
        from execution.intel.symbol_score_v6 import build_factor_vector
        from execution.factor_diagnostics import compute_factor_covariance
        
        vectors = [build_factor_vector("A", {"x": 0.5}, 0.5)]
        
        result = compute_factor_covariance(vectors, ["x"])
        
        # Should return zeros for covariance, eye for correlation
        assert result.covariance.shape == (1, 1)
    
    def test_no_factors_returns_empty(self):
        """Empty factor list should return empty matrices."""
        from execution.intel.symbol_score_v6 import build_factor_vector
        from execution.factor_diagnostics import compute_factor_covariance
        
        vectors = [
            build_factor_vector("A", {"x": 0.5}, 0.5),
            build_factor_vector("B", {"x": 0.7}, 0.5),
        ]
        
        result = compute_factor_covariance(vectors, [])
        
        assert result.covariance.shape == (0, 0)
        assert result.correlation.shape == (0, 0)
    
    def test_to_dict_serializable(self):
        """to_dict should produce JSON-serializable output."""
        import json
        from execution.intel.symbol_score_v6 import build_factor_vector
        from execution.factor_diagnostics import compute_factor_covariance
        
        vectors = [
            build_factor_vector("A", {"x": 0.5, "y": 0.3}, 0.5),
            build_factor_vector("B", {"x": 0.7, "y": 0.8}, 0.5),
            build_factor_vector("C", {"x": 0.2, "y": 0.6}, 0.5),
        ]
        
        result = compute_factor_covariance(vectors, ["x", "y"])
        
        # Should not raise
        serialized = json.dumps(result.to_dict())
        assert isinstance(serialized, str)
        
        # Round-trip
        parsed = json.loads(serialized)
        assert "covariance_matrix" in parsed
        assert "correlation_matrix" in parsed
        assert "factor_vols" in parsed
