"""
Integration tests for regime-sensitive factor weight adjustments.
Tests end-to-end flow from regime state through factor weight computation.
PATCHSET_V7.7_P6
"""

import pytest
from unittest.mock import patch, MagicMock
from typing import Dict

from execution.factor_diagnostics import (
    FactorRegimeCurvesConfig,
    FactorWeightsSnapshot,
    AutoWeightingConfig,
    apply_regime_curves_to_weights,
    load_factor_regime_curves_config,
)


@pytest.fixture
def base_weights():
    """Base factor weights before regime adjustment."""
    return {
        "momentum": 0.30,
        "carry": 0.25,
        "value": 0.25,
        "liquidity": 0.20,
    }


@pytest.fixture
def regime_curves_config():
    """Default regime curves config."""
    return FactorRegimeCurvesConfig()


@pytest.fixture
def auto_weighting_config():
    """Default auto weighting config."""
    return AutoWeightingConfig(
        enabled=True,
        min_weight=0.05,
        max_weight=0.40,
        normalize_to_one=True,
    )


class TestEndToEndFactorWeightsWithRegime:
    """End-to-end tests for factor weights with regime curves."""

    def test_weights_unchanged_in_normal_regime(self, base_weights, regime_curves_config, auto_weighting_config):
        """Normal regime should not change weights."""
        adjusted, modifiers = apply_regime_curves_to_weights(
            weights=base_weights.copy(),
            vol_regime="NORMAL",
            dd_state="NORMAL",
            regime_curves=regime_curves_config,
            min_weight=auto_weighting_config.min_weight,
            max_weight=auto_weighting_config.max_weight,
            normalize_to_one=auto_weighting_config.normalize_to_one,
        )

        # Weights should remain the same (up to normalization)
        total = sum(adjusted.values())
        assert abs(total - 1.0) < 0.01  # Normalized

        # Individual weights should be proportionally same
        for factor in base_weights:
            original_ratio = base_weights[factor] / sum(base_weights.values())
            adjusted_ratio = adjusted[factor] / total
            assert abs(original_ratio - adjusted_ratio) < 0.01

    def test_weights_reduced_in_crisis(self, base_weights, regime_curves_config, auto_weighting_config):
        """CRISIS regime should reduce all weights proportionally."""
        adjusted, modifiers = apply_regime_curves_to_weights(
            weights=base_weights.copy(),
            vol_regime="CRISIS",
            dd_state="NORMAL",
            regime_curves=regime_curves_config,
            min_weight=auto_weighting_config.min_weight,
            max_weight=auto_weighting_config.max_weight,
            normalize_to_one=auto_weighting_config.normalize_to_one,
        )

        # Multiplier should be 0.70 (factor diagnostics uses different defaults)
        assert modifiers["vol_multiplier"] == 0.70
        
        # Weights still normalized to 1.0
        total = sum(adjusted.values())
        assert abs(total - 1.0) < 0.01

    def test_weights_reduced_in_drawdown(self, base_weights, regime_curves_config, auto_weighting_config):
        """DRAWDOWN state should reduce weights."""
        adjusted, modifiers = apply_regime_curves_to_weights(
            weights=base_weights.copy(),
            vol_regime="NORMAL",
            dd_state="DRAWDOWN",
            regime_curves=regime_curves_config,
            min_weight=auto_weighting_config.min_weight,
            max_weight=auto_weighting_config.max_weight,
            normalize_to_one=auto_weighting_config.normalize_to_one,
        )

        # Multiplier should be 0.65
        assert modifiers["dd_multiplier"] == 0.65

    def test_combined_crisis_drawdown(self, base_weights, regime_curves_config, auto_weighting_config):
        """CRISIS + DRAWDOWN should compound."""
        adjusted, modifiers = apply_regime_curves_to_weights(
            weights=base_weights.copy(),
            vol_regime="CRISIS",
            dd_state="DRAWDOWN",
            regime_curves=regime_curves_config,
            min_weight=auto_weighting_config.min_weight,
            max_weight=auto_weighting_config.max_weight,
            normalize_to_one=auto_weighting_config.normalize_to_one,
        )

        # Combined = 0.70 * 0.65 = 0.455
        expected = 0.70 * 0.65
        assert modifiers["combined_multiplier"] == pytest.approx(expected, rel=0.01)

    def test_weights_boosted_in_low_vol(self, base_weights, regime_curves_config, auto_weighting_config):
        """LOW vol regime should boost weights."""
        adjusted, modifiers = apply_regime_curves_to_weights(
            weights=base_weights.copy(),
            vol_regime="LOW",
            dd_state="NORMAL",
            regime_curves=regime_curves_config,
            min_weight=auto_weighting_config.min_weight,
            max_weight=auto_weighting_config.max_weight,
            normalize_to_one=auto_weighting_config.normalize_to_one,
        )

        # LOW = 1.05 multiplier
        assert modifiers["vol_multiplier"] == 1.05


class TestFactorWeightSnapshotIntegration:
    """Test FactorWeightsSnapshot with regime modifiers."""

    def test_snapshot_includes_regime_modifiers(self):
        """Snapshot should include regime_modifiers field."""
        snapshot = FactorWeightsSnapshot(
            weights={"momentum": 0.5, "carry": 0.5},
            factor_vols={"momentum": 0.1, "carry": 0.15},
            factor_ir={"momentum": 0.3, "carry": 0.2},
            regime_modifiers={
                "vol_regime": "NORMAL",
                "vol_multiplier": 1.0,
                "dd_state": "NORMAL",
                "dd_multiplier": 1.0,
                "combined_multiplier": 1.0,
            },
        )

        # Should have regime_modifiers
        assert hasattr(snapshot, "regime_modifiers")
        assert "vol_multiplier" in snapshot.regime_modifiers
        assert "dd_multiplier" in snapshot.regime_modifiers

    def test_snapshot_to_dict_includes_regime(self):
        """to_dict() should include regime_modifiers."""
        snapshot = FactorWeightsSnapshot(
            weights={"momentum": 0.5, "carry": 0.5},
            factor_vols={"momentum": 0.1, "carry": 0.15},
            factor_ir={"momentum": 0.3, "carry": 0.2},
            regime_modifiers={
                "vol_regime": "HIGH",
                "vol_multiplier": 0.90,
                "dd_state": "RECOVERY",
                "dd_multiplier": 0.90,
                "combined_multiplier": 0.81,
            },
        )

        snapshot_dict = snapshot.to_dict()
        assert "regime_modifiers" in snapshot_dict
        assert snapshot_dict["regime_modifiers"]["vol_multiplier"] == 0.90


class TestBackwardCompatibilityIntegration:
    """Test backward compatibility when regime curves not provided."""

    def test_no_regime_config_uses_defaults(self, base_weights, auto_weighting_config):
        """Missing regime config should use 1.0 multipliers."""
        # Using default regime curves config
        default_config = FactorRegimeCurvesConfig()
        adjusted, modifiers = apply_regime_curves_to_weights(
            weights=base_weights.copy(),
            vol_regime="NORMAL",
            dd_state="NORMAL",
            regime_curves=default_config,
            min_weight=auto_weighting_config.min_weight,
            max_weight=auto_weighting_config.max_weight,
            normalize_to_one=auto_weighting_config.normalize_to_one,
        )

        # Should have default modifiers
        assert modifiers["vol_multiplier"] == 1.0
        assert modifiers["dd_multiplier"] == 1.0
        assert modifiers["combined_multiplier"] == 1.0

    def test_unknown_regime_uses_defaults(self, base_weights, regime_curves_config, auto_weighting_config):
        """Unknown regime strings should use 1.0 multipliers."""
        adjusted, modifiers = apply_regime_curves_to_weights(
            weights=base_weights.copy(),
            vol_regime="UNKNOWN",  # Unknown regime
            dd_state="UNKNOWN",  # Unknown dd_state
            regime_curves=regime_curves_config,
            min_weight=auto_weighting_config.min_weight,
            max_weight=auto_weighting_config.max_weight,
            normalize_to_one=auto_weighting_config.normalize_to_one,
        )

        # Should fall back to 1.0
        assert modifiers["vol_multiplier"] == 1.0
        assert modifiers["dd_multiplier"] == 1.0

    def test_empty_weights_handled(self, regime_curves_config, auto_weighting_config):
        """Empty weights dict should be handled gracefully."""
        adjusted, modifiers = apply_regime_curves_to_weights(
            weights={},
            vol_regime="NORMAL",
            dd_state="NORMAL",
            regime_curves=regime_curves_config,
            min_weight=auto_weighting_config.min_weight,
            max_weight=auto_weighting_config.max_weight,
            normalize_to_one=auto_weighting_config.normalize_to_one,
        )

        assert adjusted == {}
        assert modifiers["combined_multiplier"] == 1.0


class TestRegimeTransitions:
    """Test behavior during regime transitions."""

    def test_transition_from_normal_to_high(self, base_weights, regime_curves_config, auto_weighting_config):
        """Transition NORMAL → HIGH should reduce weights."""
        _, normal_modifiers = apply_regime_curves_to_weights(
            weights=base_weights.copy(),
            vol_regime="NORMAL",
            dd_state="NORMAL",
            regime_curves=regime_curves_config,
            min_weight=auto_weighting_config.min_weight,
            max_weight=auto_weighting_config.max_weight,
            normalize_to_one=auto_weighting_config.normalize_to_one,
        )

        _, high_modifiers = apply_regime_curves_to_weights(
            weights=base_weights.copy(),
            vol_regime="HIGH",
            dd_state="NORMAL",
            regime_curves=regime_curves_config,
            min_weight=auto_weighting_config.min_weight,
            max_weight=auto_weighting_config.max_weight,
            normalize_to_one=auto_weighting_config.normalize_to_one,
        )

        # HIGH should have lower multiplier
        assert high_modifiers["vol_multiplier"] < normal_modifiers["vol_multiplier"]

    def test_transition_from_recovery_to_normal(self, base_weights, regime_curves_config, auto_weighting_config):
        """Transition RECOVERY → NORMAL should increase weights."""
        _, recovery_modifiers = apply_regime_curves_to_weights(
            weights=base_weights.copy(),
            vol_regime="NORMAL",
            dd_state="RECOVERY",
            regime_curves=regime_curves_config,
            min_weight=auto_weighting_config.min_weight,
            max_weight=auto_weighting_config.max_weight,
            normalize_to_one=auto_weighting_config.normalize_to_one,
        )

        _, normal_modifiers = apply_regime_curves_to_weights(
            weights=base_weights.copy(),
            vol_regime="NORMAL",
            dd_state="NORMAL",
            regime_curves=regime_curves_config,
            min_weight=auto_weighting_config.min_weight,
            max_weight=auto_weighting_config.max_weight,
            normalize_to_one=auto_weighting_config.normalize_to_one,
        )

        # NORMAL should have higher multiplier than RECOVERY
        assert normal_modifiers["dd_multiplier"] > recovery_modifiers["dd_multiplier"]


class TestConfigLoading:
    """Test loading regime curves config from strategy config."""

    def test_load_uses_defaults_when_empty(self):
        """Should use defaults when config is empty."""
        config = load_factor_regime_curves_config({})

        # Should have default values
        assert config.volatility["LOW"] == 1.05
        assert config.volatility["CRISIS"] == 0.70
        assert config.drawdown["DRAWDOWN"] == 0.65

    def test_config_has_expected_structure(self):
        """Config should have expected structure."""
        config = FactorRegimeCurvesConfig()
        
        # Check volatility keys
        assert "LOW" in config.volatility
        assert "NORMAL" in config.volatility
        assert "HIGH" in config.volatility
        assert "CRISIS" in config.volatility
        
        # Check drawdown keys
        assert "NORMAL" in config.drawdown
        assert "RECOVERY" in config.drawdown
        assert "DRAWDOWN" in config.drawdown
