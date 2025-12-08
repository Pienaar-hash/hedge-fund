"""
Tests for factor auto-weighting (v7.5_C3).

Validates:
- compute_factor_ir computes IR correctly
- compute_raw_factor_weights handles all modes
- normalize_factor_weights clamps and normalizes
- smooth_factor_weights applies EWMA correctly
- build_factor_weights_snapshot integrates all pieces
"""
from __future__ import annotations

import pytest

from execution.factor_diagnostics import (
    compute_factor_ir,
    compute_raw_factor_weights,
    normalize_factor_weights,
    smooth_factor_weights,
    build_factor_weights_snapshot,
    FactorWeights,
    FactorWeightsSnapshot,
    FactorCovarianceSnapshot,
    AutoWeightingConfig,
    load_auto_weighting_config,
)
import numpy as np


class TestComputeFactorIR:
    """Test suite for compute_factor_ir function."""

    def test_basic_ir_computation(self):
        """Basic IR = PnL / Vol."""
        factor_pnl = {"trend": 100.0, "carry": 50.0, "expectancy": -20.0}
        factor_vols = {"trend": 10.0, "carry": 10.0, "expectancy": 10.0}
        
        ir = compute_factor_ir(factor_pnl, factor_vols)
        
        assert abs(ir["trend"] - 10.0) < 0.01  # 100/10
        assert abs(ir["carry"] - 5.0) < 0.01  # 50/10
        assert abs(ir["expectancy"] - (-2.0)) < 0.01  # -20/10

    def test_zero_vol_uses_eps(self):
        """Zero vol should not cause division by zero."""
        factor_pnl = {"trend": 100.0}
        factor_vols = {"trend": 0.0}
        
        ir = compute_factor_ir(factor_pnl, factor_vols, eps=1e-9)
        
        # Should be very large but not inf
        assert ir["trend"] > 1e8
        assert ir["trend"] < float("inf")

    def test_negative_pnl_gives_negative_ir(self):
        """Negative PnL should give negative IR."""
        factor_pnl = {"A": -50.0}
        factor_vols = {"A": 5.0}
        
        ir = compute_factor_ir(factor_pnl, factor_vols)
        
        assert ir["A"] < 0


class TestComputeRawFactorWeights:
    """Test suite for compute_raw_factor_weights function."""

    def test_equal_mode(self):
        """Equal mode gives same weight to all factors."""
        factor_names = ["A", "B", "C"]
        factor_vols = {"A": 1.0, "B": 2.0, "C": 3.0}
        factor_ir = {"A": 10.0, "B": 5.0, "C": 2.0}
        
        weights = compute_raw_factor_weights("equal", factor_names, factor_vols, factor_ir)
        
        assert weights["A"] == 1.0
        assert weights["B"] == 1.0
        assert weights["C"] == 1.0

    def test_vol_inverse_mode(self):
        """Vol inverse mode: lower vol → higher weight."""
        factor_names = ["A", "B", "C"]
        factor_vols = {"A": 0.5, "B": 1.0, "C": 2.0}
        factor_ir = {"A": 0.0, "B": 0.0, "C": 0.0}
        
        weights = compute_raw_factor_weights("vol_inverse", factor_names, factor_vols, factor_ir)
        
        # Lower vol should have higher weight
        assert weights["A"] > weights["B"]
        assert weights["B"] > weights["C"]

    def test_ir_only_mode(self):
        """IR only mode: higher IR → higher weight."""
        factor_names = ["A", "B", "C"]
        factor_vols = {"A": 1.0, "B": 1.0, "C": 1.0}
        factor_ir = {"A": 10.0, "B": 5.0, "C": 2.0}
        
        weights = compute_raw_factor_weights("ir_only", factor_names, factor_vols, factor_ir)
        
        assert weights["A"] == 10.0
        assert weights["B"] == 5.0
        assert weights["C"] == 2.0

    def test_vol_inverse_ir_mode(self):
        """Vol inverse IR mode: IR/vol combination."""
        factor_names = ["A", "B"]
        factor_vols = {"A": 1.0, "B": 2.0}
        factor_ir = {"A": 10.0, "B": 10.0}
        
        weights = compute_raw_factor_weights("vol_inverse_ir", factor_names, factor_vols, factor_ir)
        
        # Same IR, lower vol → higher weight
        assert weights["A"] > weights["B"]

    def test_unknown_mode_defaults_equal(self):
        """Unknown mode should default to equal weights."""
        factor_names = ["A", "B"]
        weights = compute_raw_factor_weights("unknown", factor_names, {"A": 1.0, "B": 1.0}, {"A": 1.0, "B": 1.0})
        
        assert weights["A"] == 1.0
        assert weights["B"] == 1.0


class TestNormalizeFactorWeights:
    """Test suite for normalize_factor_weights function."""

    def test_abs_values_used(self):
        """Negative weights should be converted to positive."""
        raw_weights = {"A": -10.0, "B": 5.0}
        
        result = normalize_factor_weights(raw_weights, min_weight=0.0, max_weight=1.0, normalize_to_one=True)
        
        # Both should be positive
        assert result.weights["A"] > 0
        assert result.weights["B"] > 0

    def test_normalize_to_one(self):
        """Weights should sum to 1 when normalize_to_one=True."""
        raw_weights = {"A": 10.0, "B": 20.0, "C": 30.0}
        
        result = normalize_factor_weights(raw_weights, min_weight=0.0, max_weight=1.0, normalize_to_one=True)
        
        total = sum(result.weights.values())
        assert abs(total - 1.0) < 0.01

    def test_min_weight_clamp(self):
        """Weights should not go below min_weight."""
        raw_weights = {"A": 1.0, "B": 100.0}  # A will be very small after normalization
        
        result = normalize_factor_weights(raw_weights, min_weight=0.10, max_weight=1.0, normalize_to_one=True)
        
        assert result.weights["A"] >= 0.10

    def test_max_weight_clamp(self):
        """Weights should not exceed max_weight."""
        raw_weights = {"A": 1000.0, "B": 1.0}  # A will dominate
        
        result = normalize_factor_weights(raw_weights, min_weight=0.0, max_weight=0.50, normalize_to_one=True)
        
        assert result.weights["A"] <= 0.50

    def test_all_zeros_gives_equal_weights(self):
        """All zero weights should give equal weights."""
        raw_weights = {"A": 0.0, "B": 0.0, "C": 0.0}
        
        result = normalize_factor_weights(raw_weights, min_weight=0.0, max_weight=1.0, normalize_to_one=True)
        
        # Should be equal
        assert abs(result.weights["A"] - result.weights["B"]) < 0.01
        assert abs(result.weights["B"] - result.weights["C"]) < 0.01

    def test_empty_weights(self):
        """Empty weights should return empty."""
        result = normalize_factor_weights({}, min_weight=0.1, max_weight=0.5, normalize_to_one=True)
        assert result.weights == {}


class TestSmoothFactorWeights:
    """Test suite for smooth_factor_weights function."""

    def test_no_prev_returns_current(self):
        """With no previous weights, return current."""
        current = FactorWeights(weights={"A": 0.5, "B": 0.5})
        
        result = smooth_factor_weights(None, current, alpha=0.2)
        
        assert result.weights["A"] == 0.5
        assert result.weights["B"] == 0.5

    def test_alpha_1_gives_current(self):
        """Alpha=1 should give current weights only."""
        prev = FactorWeights(weights={"A": 0.3, "B": 0.7})
        current = FactorWeights(weights={"A": 0.8, "B": 0.2})
        
        result = smooth_factor_weights(prev, current, alpha=1.0)
        
        assert abs(result.weights["A"] - 0.8) < 0.01
        assert abs(result.weights["B"] - 0.2) < 0.01

    def test_alpha_0_gives_prev(self):
        """Alpha=0 should give previous weights only."""
        prev = FactorWeights(weights={"A": 0.3, "B": 0.7})
        current = FactorWeights(weights={"A": 0.8, "B": 0.2})
        
        result = smooth_factor_weights(prev, current, alpha=0.0)
        
        assert abs(result.weights["A"] - 0.3) < 0.01
        assert abs(result.weights["B"] - 0.7) < 0.01

    def test_alpha_05_gives_average(self):
        """Alpha=0.5 should give average."""
        prev = FactorWeights(weights={"A": 0.2, "B": 0.8})
        current = FactorWeights(weights={"A": 0.6, "B": 0.4})
        
        result = smooth_factor_weights(prev, current, alpha=0.5)
        
        assert abs(result.weights["A"] - 0.4) < 0.01  # (0.2 + 0.6) / 2
        assert abs(result.weights["B"] - 0.6) < 0.01  # (0.8 + 0.4) / 2

    def test_new_factors_handled(self):
        """New factors not in prev should be added."""
        prev = FactorWeights(weights={"A": 0.5})
        current = FactorWeights(weights={"A": 0.3, "B": 0.7})
        
        result = smooth_factor_weights(prev, current, alpha=0.5)
        
        # B should be (0 + 0.7) * 0.5 = 0.35
        assert abs(result.weights["B"] - 0.35) < 0.01


class TestBuildFactorWeightsSnapshot:
    """Test suite for build_factor_weights_snapshot function."""

    def test_builds_snapshot_correctly(self):
        """Build snapshot with all pieces."""
        factor_cov = FactorCovarianceSnapshot(
            factors=["A", "B", "C"],
            covariance=np.eye(3),
            correlation=np.eye(3),
            factor_vols={"A": 1.0, "B": 2.0, "C": 3.0},
        )
        factor_pnl = {"A": 100.0, "B": 50.0, "C": 10.0}
        cfg = AutoWeightingConfig(
            enabled=True,
            mode="vol_inverse_ir",
            min_weight=0.05,
            max_weight=0.40,
            normalize_to_one=True,
            smoothing_alpha=1.0,  # No smoothing for test
        )
        
        snapshot = build_factor_weights_snapshot(factor_cov, factor_pnl, cfg, prev_weights=None)
        
        assert isinstance(snapshot, FactorWeightsSnapshot)
        assert len(snapshot.weights) == 3
        assert sum(snapshot.weights.values()) > 0.99
        assert snapshot.mode == "vol_inverse_ir"
        assert snapshot.updated_ts > 0

    def test_snapshot_to_dict_serializable(self):
        """Snapshot to_dict should be JSON serializable."""
        factor_cov = FactorCovarianceSnapshot(
            factors=["A", "B"],
            covariance=np.eye(2),
            correlation=np.eye(2),
            factor_vols={"A": 1.0, "B": 1.0},
        )
        factor_pnl = {"A": 10.0, "B": 10.0}
        cfg = AutoWeightingConfig()
        
        snapshot = build_factor_weights_snapshot(factor_cov, factor_pnl, cfg)
        
        as_dict = snapshot.to_dict()
        
        import json
        json_str = json.dumps(as_dict)
        assert len(json_str) > 0


class TestAutoWeightingConfig:
    """Test suite for auto-weighting config loading."""

    def test_default_config(self):
        """Default config when not in strategy_config."""
        cfg = load_auto_weighting_config({})
        assert isinstance(cfg, AutoWeightingConfig)
        assert cfg.enabled is False

    def test_config_from_strategy_config(self):
        """Config loaded from strategy_config."""
        strategy_config = {
            "factor_diagnostics": {
                "auto_weighting": {
                    "enabled": True,
                    "mode": "vol_inverse",
                    "min_weight": 0.10,
                    "max_weight": 0.35,
                    "normalize_to_one": False,
                    "smoothing_alpha": 0.5,
                }
            }
        }
        cfg = load_auto_weighting_config(strategy_config)
        
        assert cfg.enabled is True
        assert cfg.mode == "vol_inverse"
        assert cfg.min_weight == 0.10
        assert cfg.max_weight == 0.35
        assert cfg.normalize_to_one is False
        assert cfg.smoothing_alpha == 0.5
