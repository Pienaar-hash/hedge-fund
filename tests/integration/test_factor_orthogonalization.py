"""
Tests for factor orthogonalization (v7.5_C3).

Validates:
- Gram-Schmidt orthogonalization produces orthogonal columns
- Correlated factors become orthogonal (residual)
- Independent factors remain unchanged
- Degenerate factors are handled gracefully
"""
from __future__ import annotations

import numpy as np
import pytest

from execution.intel.symbol_score_v6 import FactorVector
from execution.factor_diagnostics import (
    orthogonalize_factors,
    OrthogonalizedFactorVectors,
    load_orthogonalization_config,
    OrthogonalizationConfig,
)


def _make_factor_vector(
    symbol: str,
    factors: dict[str, float],
    direction: str = "LONG",
) -> FactorVector:
    """Helper to create FactorVector."""
    return FactorVector(
        symbol=symbol,
        factors=factors,
        hybrid_score=0.5,
        direction=direction,
        regime="normal",
    )


class TestOrthogonalizeFactors:
    """Test suite for orthogonalize_factors function."""

    def test_empty_vectors_returns_empty(self):
        """Empty input returns empty output."""
        result = orthogonalize_factors([], ["trend", "carry"])
        assert isinstance(result, OrthogonalizedFactorVectors)
        assert result.per_symbol == {}

    def test_single_factor_unchanged(self):
        """Single factor should remain unchanged."""
        vectors = [
            _make_factor_vector("BTCUSDT", {"trend": 0.8}),
            _make_factor_vector("ETHUSDT", {"trend": 0.6}),
            _make_factor_vector("SOLUSDT", {"trend": 0.4}),
        ]
        result = orthogonalize_factors(vectors, ["trend"])
        
        assert "BTCUSDT" in result.per_symbol
        assert "ETHUSDT" in result.per_symbol
        assert "SOLUSDT" in result.per_symbol
        
        # Single factor should be unchanged
        assert abs(result.per_symbol["BTCUSDT"]["trend"] - 0.8) < 0.01
        assert abs(result.per_symbol["ETHUSDT"]["trend"] - 0.6) < 0.01
        assert abs(result.per_symbol["SOLUSDT"]["trend"] - 0.4) < 0.01

    def test_perfectly_correlated_factors(self):
        """Factor B = Factor A should make B nearly zero after orthogonalization."""
        # Factor B is exactly Factor A (100% correlated)
        vectors = [
            _make_factor_vector("BTCUSDT", {"A": 1.0, "B": 1.0, "C": 0.1}),
            _make_factor_vector("ETHUSDT", {"A": 0.5, "B": 0.5, "C": 0.9}),
            _make_factor_vector("SOLUSDT", {"A": 0.2, "B": 0.2, "C": 0.5}),
        ]
        result = orthogonalize_factors(vectors, ["A", "B", "C"])
        
        # A should be unchanged (first factor)
        assert abs(result.per_symbol["BTCUSDT"]["A"] - 1.0) < 0.01
        
        # B should be near zero (projection onto A removed)
        for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            assert abs(result.per_symbol[sym]["B"]) < 0.01, f"B for {sym} should be near zero"
        
        # C should still have values (independent)
        c_values = [result.per_symbol[s]["C"] for s in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]]
        assert not all(abs(v) < 0.01 for v in c_values), "C should not be all zeros"

    def test_independent_factors_remain_independent(self):
        """Independent factors should remain mostly unchanged."""
        # Construct orthogonal factors
        vectors = [
            _make_factor_vector("BTCUSDT", {"A": 1.0, "B": 0.0, "C": 0.0}),
            _make_factor_vector("ETHUSDT", {"A": 0.0, "B": 1.0, "C": 0.0}),
            _make_factor_vector("SOLUSDT", {"A": 0.0, "B": 0.0, "C": 1.0}),
        ]
        result = orthogonalize_factors(vectors, ["A", "B", "C"])
        
        # All should remain unchanged since already orthogonal
        assert abs(result.per_symbol["BTCUSDT"]["A"] - 1.0) < 0.01
        assert abs(result.per_symbol["ETHUSDT"]["B"] - 1.0) < 0.01
        assert abs(result.per_symbol["SOLUSDT"]["C"] - 1.0) < 0.01

    def test_orthogonal_columns_dot_product_zero(self):
        """Verify orthogonalized columns have near-zero dot products."""
        vectors = [
            _make_factor_vector("BTCUSDT", {"A": 0.9, "B": 0.7, "C": 0.1}),
            _make_factor_vector("ETHUSDT", {"A": 0.6, "B": 0.8, "C": 0.5}),
            _make_factor_vector("SOLUSDT", {"A": 0.3, "B": 0.5, "C": 0.9}),
            _make_factor_vector("XRPUSDT", {"A": 0.5, "B": 0.4, "C": 0.6}),
        ]
        result = orthogonalize_factors(vectors, ["A", "B", "C"])
        
        # Extract columns
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
        col_A = np.array([result.per_symbol[s]["A"] for s in symbols])
        col_B = np.array([result.per_symbol[s]["B"] for s in symbols])
        col_C = np.array([result.per_symbol[s]["C"] for s in symbols])
        
        # Dot products should be near zero
        tol = 0.01
        assert abs(np.dot(col_A, col_B)) < tol, "A·B should be near zero"
        assert abs(np.dot(col_A, col_C)) < tol, "A·C should be near zero"
        assert abs(np.dot(col_B, col_C)) < tol, "B·C should be near zero"

    def test_degenerate_factor_all_zeros(self):
        """Factor with all zeros should remain zeros."""
        vectors = [
            _make_factor_vector("BTCUSDT", {"A": 1.0, "B": 0.0}),
            _make_factor_vector("ETHUSDT", {"A": 0.5, "B": 0.0}),
            _make_factor_vector("SOLUSDT", {"A": 0.2, "B": 0.0}),
        ]
        result = orthogonalize_factors(vectors, ["A", "B"])
        
        # B should remain zeros
        for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            assert abs(result.per_symbol[sym]["B"]) < 0.01

    def test_to_dict_serializable(self):
        """Verify OrthogonalizedFactorVectors.to_dict() is JSON serializable."""
        vectors = [
            _make_factor_vector("BTCUSDT", {"A": 0.9, "B": 0.7}),
            _make_factor_vector("ETHUSDT", {"A": 0.6, "B": 0.8}),
        ]
        result = orthogonalize_factors(vectors, ["A", "B"])
        
        as_dict = result.to_dict()
        assert isinstance(as_dict, dict)
        assert "per_symbol" in as_dict
        
        # Should be JSON serializable (no numpy types)
        import json
        json_str = json.dumps(as_dict)
        assert len(json_str) > 0


class TestOrthogonalizationConfig:
    """Test suite for orthogonalization config loading."""

    def test_default_config(self):
        """Default config when not in strategy_config."""
        cfg = load_orthogonalization_config({})
        assert isinstance(cfg, OrthogonalizationConfig)
        assert cfg.enabled is False
        assert cfg.method == "gram_schmidt"

    def test_config_from_strategy_config(self):
        """Config loaded from strategy_config."""
        strategy_config = {
            "factor_diagnostics": {
                "orthogonalization": {
                    "enabled": True,
                    "method": "gram_schmidt",
                }
            }
        }
        cfg = load_orthogonalization_config(strategy_config)
        assert cfg.enabled is True
        assert cfg.method == "gram_schmidt"

    def test_disabled_orthogonalization(self):
        """Disabled orthogonalization config."""
        strategy_config = {
            "factor_diagnostics": {
                "orthogonalization": {
                    "enabled": False,
                }
            }
        }
        cfg = load_orthogonalization_config(strategy_config)
        assert cfg.enabled is False
