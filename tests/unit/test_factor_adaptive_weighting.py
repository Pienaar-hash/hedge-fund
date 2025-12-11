"""
Tests for v7.7_P2 — Adaptive Factor Outcome Weighting.

Tests cover:
1. Base case: adaptive disabled → weights unchanged vs base
2. High IR positive PnL factor → weight increases within max_shift
3. Negative PnL factor or IR < cut threshold → weight decreases
4. All biases respect [-max_shift, +max_shift]
5. Weights still respect [min_weight, max_weight] after adaptation
6. When normalize_to_one = True, sum of weights ≈ 1.0

Pure math tests: no filesystem, no JSON loading.
"""
from __future__ import annotations

import pytest

from execution.factor_diagnostics import (
    AdaptiveConfig,
    AutoWeightingConfig,
    FactorCovarianceSnapshot,
    FactorPerformance,
    FactorWeights,
    FactorWeightsSnapshot,
    apply_adaptive_bias_to_weights,
    build_factor_weights_snapshot,
    compute_adaptive_weight_bias,
    compute_factor_performance,
    load_adaptive_config,
)
import numpy as np

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_adaptive_config() -> AdaptiveConfig:
    """Default adaptive config for tests."""
    return AdaptiveConfig(
        enabled=True,
        ir_boost_threshold=0.25,
        ir_cut_threshold=0.0,
        pnl_min_for_boost=0.0,
        max_shift=0.10,
    )


@pytest.fixture
def default_auto_weight_config() -> AutoWeightingConfig:
    """Default auto-weighting config for tests."""
    return AutoWeightingConfig(
        enabled=True,
        mode="vol_inverse_ir",
        min_weight=0.05,
        max_weight=0.40,
        normalize_to_one=True,
        smoothing_alpha=1.0,  # No smoothing for easier testing
    )


@pytest.fixture
def sample_factor_covariance() -> FactorCovarianceSnapshot:
    """Sample covariance snapshot with 3 factors."""
    factors = ["trend", "carry", "expectancy"]
    n = len(factors)
    return FactorCovarianceSnapshot(
        factors=factors,
        covariance=np.eye(n) * 0.01,  # Low correlation
        correlation=np.eye(n),
        factor_vols={"trend": 0.1, "carry": 0.1, "expectancy": 0.1},
        lookback_days=30,
    )


# ---------------------------------------------------------------------------
# Test: compute_factor_performance
# ---------------------------------------------------------------------------

class TestComputeFactorPerformance:
    """Tests for compute_factor_performance function."""

    def test_combines_ir_and_pnl(self):
        """Combines IR and PnL into FactorPerformance objects."""
        factor_ir = {"trend": 0.5, "carry": -0.1}
        factor_pnl = {"trend": 100.0, "carry": -20.0}

        result = compute_factor_performance(factor_ir, factor_pnl)

        assert "trend" in result
        assert result["trend"].ir == pytest.approx(0.5)
        assert result["trend"].pnl_contrib == pytest.approx(100.0)
        assert result["carry"].ir == pytest.approx(-0.1)
        assert result["carry"].pnl_contrib == pytest.approx(-20.0)

    def test_handles_missing_keys(self):
        """Handles factors present in one dict but not the other."""
        factor_ir = {"trend": 0.5, "vol_regime": 0.2}
        factor_pnl = {"trend": 100.0, "carry": 50.0}

        result = compute_factor_performance(factor_ir, factor_pnl)

        # All keys from both dicts should be present
        assert "trend" in result
        assert "carry" in result
        assert "vol_regime" in result

        # Missing values default to 0
        assert result["carry"].ir == pytest.approx(0.0)
        assert result["vol_regime"].pnl_contrib == pytest.approx(0.0)

    def test_empty_inputs(self):
        """Returns empty dict for empty inputs."""
        result = compute_factor_performance({}, {})
        assert result == {}


# ---------------------------------------------------------------------------
# Test: compute_adaptive_weight_bias
# ---------------------------------------------------------------------------

class TestComputeAdaptiveWeightBias:
    """Tests for compute_adaptive_weight_bias function."""

    def test_high_ir_positive_pnl_gets_positive_bias(self, default_adaptive_config):
        """Factor with high IR and positive PnL gets positive bias."""
        base_weights = {"trend": 0.25, "carry": 0.25}
        factor_perf = {
            "trend": FactorPerformance("trend", ir=0.5, pnl_contrib=100.0),  # High IR, pos PnL
            "carry": FactorPerformance("carry", ir=0.1, pnl_contrib=10.0),   # Low IR
        }

        bias = compute_adaptive_weight_bias(base_weights, factor_perf, default_adaptive_config)

        assert bias["trend"] > 0  # Should get positive bias
        assert bias["trend"] <= default_adaptive_config.max_shift

    def test_negative_pnl_gets_negative_bias(self, default_adaptive_config):
        """Factor with negative PnL gets negative bias."""
        base_weights = {"trend": 0.25, "carry": 0.25}
        factor_perf = {
            "trend": FactorPerformance("trend", ir=0.5, pnl_contrib=-50.0),  # Negative PnL
            "carry": FactorPerformance("carry", ir=0.3, pnl_contrib=10.0),
        }

        bias = compute_adaptive_weight_bias(base_weights, factor_perf, default_adaptive_config)

        assert bias["trend"] < 0  # Negative PnL → negative bias
        assert bias["trend"] >= -default_adaptive_config.max_shift

    def test_low_ir_gets_negative_bias(self, default_adaptive_config):
        """Factor with IR below cut threshold gets negative bias."""
        base_weights = {"trend": 0.25, "carry": 0.25}
        factor_perf = {
            "trend": FactorPerformance("trend", ir=-0.2, pnl_contrib=10.0),  # IR < 0 (cut threshold)
            "carry": FactorPerformance("carry", ir=0.3, pnl_contrib=10.0),
        }

        bias = compute_adaptive_weight_bias(base_weights, factor_perf, default_adaptive_config)

        assert bias["trend"] < 0  # Low IR → negative bias

    def test_bias_clamped_to_max_shift(self, default_adaptive_config):
        """All biases are clamped to [-max_shift, +max_shift]."""
        base_weights = {"trend": 0.25, "carry": 0.25}
        factor_perf = {
            "trend": FactorPerformance("trend", ir=10.0, pnl_contrib=1000.0),  # Very high
            "carry": FactorPerformance("carry", ir=-10.0, pnl_contrib=-1000.0),  # Very low
        }

        bias = compute_adaptive_weight_bias(base_weights, factor_perf, default_adaptive_config)

        assert bias["trend"] <= default_adaptive_config.max_shift
        assert bias["carry"] >= -default_adaptive_config.max_shift

    def test_missing_factor_perf_gets_zero_bias(self, default_adaptive_config):
        """Missing factor performance → zero bias."""
        base_weights = {"trend": 0.25, "carry": 0.25, "vol_regime": 0.25}
        factor_perf = {
            "trend": FactorPerformance("trend", ir=0.5, pnl_contrib=100.0),
            # carry and vol_regime missing
        }

        bias = compute_adaptive_weight_bias(base_weights, factor_perf, default_adaptive_config)

        assert bias["carry"] == pytest.approx(0.0)
        assert bias["vol_regime"] == pytest.approx(0.0)

    def test_neutral_ir_and_pnl_gets_zero_bias(self, default_adaptive_config):
        """Factor with neutral IR/PnL gets zero bias."""
        base_weights = {"trend": 0.25}
        factor_perf = {
            # IR between cut threshold (0) and boost threshold (0.25), PnL non-negative
            "trend": FactorPerformance("trend", ir=0.1, pnl_contrib=5.0),
        }

        bias = compute_adaptive_weight_bias(base_weights, factor_perf, default_adaptive_config)

        assert bias["trend"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test: apply_adaptive_bias_to_weights
# ---------------------------------------------------------------------------

class TestApplyAdaptiveBiasToWeights:
    """Tests for apply_adaptive_bias_to_weights function."""

    def test_applies_bias_correctly(self):
        """Applies bias and clamps to bounds."""
        base_weights = {"trend": 0.20, "carry": 0.20}
        bias = {"trend": 0.05, "carry": -0.03}

        result = apply_adaptive_bias_to_weights(
            base_weights=base_weights,
            bias=bias,
            min_weight=0.05,
            max_weight=0.40,
            normalize_to_one=False,
        )

        assert result["trend"] == pytest.approx(0.25)
        assert result["carry"] == pytest.approx(0.17)

    def test_clamps_to_min_weight(self):
        """Weights are clamped to min_weight."""
        base_weights = {"trend": 0.06}
        bias = {"trend": -0.05}  # Would put it at 0.01

        result = apply_adaptive_bias_to_weights(
            base_weights=base_weights,
            bias=bias,
            min_weight=0.05,
            max_weight=0.40,
            normalize_to_one=False,
        )

        assert result["trend"] == pytest.approx(0.05)

    def test_clamps_to_max_weight(self):
        """Weights are clamped to max_weight."""
        base_weights = {"trend": 0.38}
        bias = {"trend": 0.10}  # Would put it at 0.48

        result = apply_adaptive_bias_to_weights(
            base_weights=base_weights,
            bias=bias,
            min_weight=0.05,
            max_weight=0.40,
            normalize_to_one=False,
        )

        assert result["trend"] == pytest.approx(0.40)

    def test_normalizes_to_one(self):
        """When normalize_to_one=True, weights sum to 1.0."""
        base_weights = {"trend": 0.25, "carry": 0.25, "expectancy": 0.25}
        bias = {"trend": 0.05, "carry": 0.05, "expectancy": 0.05}

        result = apply_adaptive_bias_to_weights(
            base_weights=base_weights,
            bias=bias,
            min_weight=0.05,
            max_weight=0.40,
            normalize_to_one=True,
        )

        total = sum(result.values())
        assert total == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Test: build_factor_weights_snapshot with adaptive
# ---------------------------------------------------------------------------

class TestBuildFactorWeightsSnapshotAdaptive:
    """Tests for build_factor_weights_snapshot with adaptive config."""

    def test_adaptive_disabled_matches_baseline(
        self, sample_factor_covariance, default_auto_weight_config
    ):
        """When adaptive disabled, weights match baseline (no adaptive bias)."""
        factor_pnl = {"trend": 100.0, "carry": 50.0, "expectancy": 30.0}

        # Without adaptive
        baseline = build_factor_weights_snapshot(
            factor_cov=sample_factor_covariance,
            factor_pnl=factor_pnl,
            auto_weight_cfg=default_auto_weight_config,
            adaptive_cfg=None,
        )

        # With adaptive disabled
        disabled_cfg = AdaptiveConfig(enabled=False)
        with_disabled = build_factor_weights_snapshot(
            factor_cov=sample_factor_covariance,
            factor_pnl=factor_pnl,
            auto_weight_cfg=default_auto_weight_config,
            adaptive_cfg=disabled_cfg,
        )

        # Weights should be identical
        for factor in baseline.weights:
            assert baseline.weights[factor] == pytest.approx(
                with_disabled.weights[factor], rel=1e-6
            )

        # Adaptive metadata should reflect disabled state
        assert with_disabled.adaptive_enabled is False
        assert with_disabled.adaptive_bias == {}

    def test_adaptive_enabled_modifies_weights(
        self, sample_factor_covariance, default_auto_weight_config, default_adaptive_config
    ):
        """When adaptive enabled, weights are modified based on IR/PnL."""
        # High IR and PnL for trend → should get boosted
        factor_pnl = {"trend": 100.0, "carry": -10.0, "expectancy": 5.0}

        # Baseline without adaptive
        baseline = build_factor_weights_snapshot(
            factor_cov=sample_factor_covariance,
            factor_pnl=factor_pnl,
            auto_weight_cfg=default_auto_weight_config,
            adaptive_cfg=None,
        )

        # With adaptive enabled
        with_adaptive = build_factor_weights_snapshot(
            factor_cov=sample_factor_covariance,
            factor_pnl=factor_pnl,
            auto_weight_cfg=default_auto_weight_config,
            adaptive_cfg=default_adaptive_config,
        )

        # Weights should differ
        assert with_adaptive.adaptive_enabled is True
        assert with_adaptive.adaptive_bias != {}

        # trend has high IR and positive PnL → should have positive bias
        # carry has negative PnL → should have negative bias
        assert with_adaptive.adaptive_bias.get("carry", 0) <= 0

    def test_adaptive_respects_weight_bounds(
        self, sample_factor_covariance, default_auto_weight_config, default_adaptive_config
    ):
        """Adaptive weights still respect min/max bounds."""
        factor_pnl = {"trend": 1000.0, "carry": -1000.0, "expectancy": 0.0}

        result = build_factor_weights_snapshot(
            factor_cov=sample_factor_covariance,
            factor_pnl=factor_pnl,
            auto_weight_cfg=default_auto_weight_config,
            adaptive_cfg=default_adaptive_config,
        )

        for factor, weight in result.weights.items():
            assert weight >= default_auto_weight_config.min_weight - 1e-6
            assert weight <= default_auto_weight_config.max_weight + 1e-6

    def test_adaptive_weights_sum_to_one(
        self, sample_factor_covariance, default_auto_weight_config, default_adaptive_config
    ):
        """When normalize_to_one=True, adaptive weights still sum to 1.0."""
        factor_pnl = {"trend": 100.0, "carry": 50.0, "expectancy": 30.0}

        result = build_factor_weights_snapshot(
            factor_cov=sample_factor_covariance,
            factor_pnl=factor_pnl,
            auto_weight_cfg=default_auto_weight_config,
            adaptive_cfg=default_adaptive_config,
        )

        total = sum(result.weights.values())
        assert total == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Test: load_adaptive_config
# ---------------------------------------------------------------------------

class TestLoadAdaptiveConfig:
    """Tests for load_adaptive_config function."""

    def test_load_default_when_none(self):
        """Returns disabled config when no strategy_config."""
        cfg = load_adaptive_config(None)
        assert cfg.enabled is False

    def test_load_from_nested_config(self):
        """Loads adaptive config from nested strategy_config."""
        strategy_config = {
            "factor_diagnostics": {
                "auto_weighting": {
                    "enabled": True,
                    "adaptive": {
                        "enabled": True,
                        "ir_boost_threshold": 0.30,
                        "max_shift": 0.15,
                    },
                },
            },
        }

        cfg = load_adaptive_config(strategy_config)

        assert cfg.enabled is True
        assert cfg.ir_boost_threshold == pytest.approx(0.30)
        assert cfg.max_shift == pytest.approx(0.15)
        # Defaults for unspecified values
        assert cfg.ir_cut_threshold == pytest.approx(0.0)
        assert cfg.pnl_min_for_boost == pytest.approx(0.0)

    def test_load_missing_adaptive_key_is_disabled(self):
        """Missing adaptive key → disabled config."""
        strategy_config = {
            "factor_diagnostics": {
                "auto_weighting": {
                    "enabled": True,
                    # No "adaptive" key
                },
            },
        }

        cfg = load_adaptive_config(strategy_config)

        assert cfg.enabled is False

    def test_load_empty_adaptive_block_is_disabled(self):
        """Empty adaptive block → disabled config."""
        strategy_config = {
            "factor_diagnostics": {
                "auto_weighting": {
                    "enabled": True,
                    "adaptive": {},
                },
            },
        }

        cfg = load_adaptive_config(strategy_config)

        assert cfg.enabled is False


# ---------------------------------------------------------------------------
# Test: Edge Cases and Boundary Conditions
# ---------------------------------------------------------------------------

class TestAdaptiveEdgeCases:
    """Test edge cases for adaptive weighting."""

    def test_all_factors_get_boosted(self, default_adaptive_config):
        """When all factors have high IR/PnL, all get positive bias (clamped)."""
        base_weights = {"a": 0.33, "b": 0.33, "c": 0.34}
        factor_perf = {
            "a": FactorPerformance("a", ir=0.5, pnl_contrib=100.0),
            "b": FactorPerformance("b", ir=0.6, pnl_contrib=150.0),
            "c": FactorPerformance("c", ir=0.4, pnl_contrib=80.0),
        }

        bias = compute_adaptive_weight_bias(base_weights, factor_perf, default_adaptive_config)

        # All should have positive bias
        assert all(b > 0 for b in bias.values())
        # All clamped to max_shift
        assert all(b <= default_adaptive_config.max_shift for b in bias.values())

    def test_all_factors_get_cut(self, default_adaptive_config):
        """When all factors have negative PnL, all get negative bias."""
        base_weights = {"a": 0.33, "b": 0.33, "c": 0.34}
        factor_perf = {
            "a": FactorPerformance("a", ir=0.3, pnl_contrib=-100.0),
            "b": FactorPerformance("b", ir=0.2, pnl_contrib=-150.0),
            "c": FactorPerformance("c", ir=-0.1, pnl_contrib=-80.0),
        }

        bias = compute_adaptive_weight_bias(base_weights, factor_perf, default_adaptive_config)

        # All should have negative bias
        assert all(b < 0 for b in bias.values())
        # All clamped to -max_shift
        assert all(b >= -default_adaptive_config.max_shift for b in bias.values())

    def test_zero_max_shift_means_no_change(self):
        """With max_shift=0, no bias is applied."""
        cfg = AdaptiveConfig(enabled=True, max_shift=0.0)
        base_weights = {"trend": 0.25}
        factor_perf = {
            "trend": FactorPerformance("trend", ir=10.0, pnl_contrib=1000.0),
        }

        bias = compute_adaptive_weight_bias(base_weights, factor_perf, cfg)

        assert bias["trend"] == pytest.approx(0.0)

    def test_empty_base_weights(self, default_adaptive_config):
        """Empty base weights returns empty bias."""
        bias = compute_adaptive_weight_bias({}, {}, default_adaptive_config)
        assert bias == {}

    def test_snapshot_to_dict_includes_adaptive_fields(
        self, sample_factor_covariance, default_auto_weight_config, default_adaptive_config
    ):
        """FactorWeightsSnapshot.to_dict includes adaptive metadata."""
        factor_pnl = {"trend": 100.0, "carry": 50.0, "expectancy": 30.0}

        result = build_factor_weights_snapshot(
            factor_cov=sample_factor_covariance,
            factor_pnl=factor_pnl,
            auto_weight_cfg=default_auto_weight_config,
            adaptive_cfg=default_adaptive_config,
        )

        d = result.to_dict()

        assert "adaptive_enabled" in d
        assert "adaptive_bias" in d
        assert d["adaptive_enabled"] is True
        assert isinstance(d["adaptive_bias"], dict)
