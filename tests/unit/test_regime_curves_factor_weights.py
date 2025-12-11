# mypy: ignore-errors
"""
Unit tests for regime curves in factor weights (v7.7_P6).
"""
from __future__ import annotations

import pytest
from typing import Any, Dict

from execution.factor_diagnostics import (
    FactorRegimeCurvesConfig,
    FactorWeightsSnapshot,
    apply_regime_curves_to_weights,
    get_dd_state_from_snapshot,
    get_vol_regime_from_snapshot,
    load_factor_regime_curves_config,
)


class TestFactorRegimeCurvesConfig:
    """Test FactorRegimeCurvesConfig dataclass and loader."""

    def test_default_volatility_curves(self):
        """Default volatility curves for factors should be sensible."""
        cfg = FactorRegimeCurvesConfig()
        
        assert cfg.volatility["LOW"] == 1.05
        assert cfg.volatility["NORMAL"] == 1.00
        assert cfg.volatility["HIGH"] == 0.90
        assert cfg.volatility["CRISIS"] == 0.70

    def test_default_drawdown_curves(self):
        """Default drawdown curves for factors should be sensible."""
        cfg = FactorRegimeCurvesConfig()
        
        assert cfg.drawdown["NORMAL"] == 1.00
        assert cfg.drawdown["RECOVERY"] == 0.90
        assert cfg.drawdown["DRAWDOWN"] == 0.65

    def test_load_factor_regime_curves_config_from_dict(self):
        """Load factor regime curves from config dict."""
        strategy_config = {
            "factor_diagnostics": {
                "auto_weighting": {
                    "regime_curves": {
                        "volatility": {
                            "LOW": 1.10,
                            "NORMAL": 1.00,
                            "HIGH": 0.85,
                            "CRISIS": 0.60,
                        },
                        "drawdown": {
                            "NORMAL": 1.00,
                            "RECOVERY": 0.80,
                            "DRAWDOWN": 0.55,
                        },
                    }
                }
            }
        }
        
        cfg = load_factor_regime_curves_config(strategy_config)
        
        assert cfg.volatility["LOW"] == 1.10
        assert cfg.volatility["CRISIS"] == 0.60
        assert cfg.drawdown["RECOVERY"] == 0.80
        assert cfg.drawdown["DRAWDOWN"] == 0.55

    def test_load_factor_regime_curves_config_empty(self):
        """Return defaults when config is empty."""
        cfg = load_factor_regime_curves_config({})
        
        assert cfg.volatility["NORMAL"] == 1.00
        assert cfg.drawdown["NORMAL"] == 1.00


class TestGetVolRegimeFromSnapshot:
    """Test get_vol_regime_from_snapshot function."""

    def test_extract_vol_regime_lowercase(self):
        """Extract vol_regime and normalize to uppercase."""
        risk_snapshot = {"vol_regime": "high"}
        
        result = get_vol_regime_from_snapshot(risk_snapshot)
        
        assert result == "HIGH"

    def test_extract_vol_regime_uppercase(self):
        """Extract vol_regime that's already uppercase."""
        risk_snapshot = {"vol_regime": "CRISIS"}
        
        result = get_vol_regime_from_snapshot(risk_snapshot)
        
        assert result == "CRISIS"

    def test_missing_vol_regime_defaults_normal(self):
        """Default to NORMAL when vol_regime is missing."""
        risk_snapshot = {"other_field": "value"}
        
        result = get_vol_regime_from_snapshot(risk_snapshot)
        
        assert result == "NORMAL"

    def test_none_snapshot_defaults_normal(self):
        """Default to NORMAL when snapshot is None."""
        result = get_vol_regime_from_snapshot(None)
        
        assert result == "NORMAL"

    def test_empty_snapshot_defaults_normal(self):
        """Default to NORMAL when snapshot is empty."""
        result = get_vol_regime_from_snapshot({})
        
        assert result == "NORMAL"


class TestGetDdStateFromSnapshot:
    """Test get_dd_state_from_snapshot function."""

    def test_extract_dd_state(self):
        """Extract dd_state from snapshot."""
        risk_snapshot = {"dd_state": "DRAWDOWN"}
        
        result = get_dd_state_from_snapshot(risk_snapshot)
        
        assert result == "DRAWDOWN"

    def test_missing_dd_state_defaults_normal(self):
        """Default to NORMAL when dd_state is missing."""
        risk_snapshot = {"other_field": "value"}
        
        result = get_dd_state_from_snapshot(risk_snapshot)
        
        assert result == "NORMAL"

    def test_none_snapshot_defaults_normal(self):
        """Default to NORMAL when snapshot is None."""
        result = get_dd_state_from_snapshot(None)
        
        assert result == "NORMAL"


class TestApplyRegimeCurvesToWeights:
    """Test apply_regime_curves_to_weights function."""

    @pytest.fixture
    def base_weights(self) -> Dict[str, float]:
        """Base equal weights."""
        return {
            "trend": 0.25,
            "carry": 0.25,
            "rv_momentum": 0.25,
            "expectancy": 0.25,
        }

    @pytest.fixture
    def regime_cfg(self) -> FactorRegimeCurvesConfig:
        """Default regime curves config."""
        return FactorRegimeCurvesConfig()

    def test_normal_regime_no_change(self, base_weights, regime_cfg):
        """NORMAL vol + NORMAL dd should not change weights significantly."""
        adjusted, modifiers = apply_regime_curves_to_weights(
            weights=base_weights,
            vol_regime="NORMAL",
            dd_state="NORMAL",
            regime_curves=regime_cfg,
            min_weight=0.05,
            max_weight=0.40,
            normalize_to_one=True,
        )
        
        # Combined multiplier is 1.0, so weights should be similar
        assert modifiers["combined_multiplier"] == 1.0
        # After normalization, should sum to 1.0
        assert sum(adjusted.values()) == pytest.approx(1.0, abs=0.01)

    def test_high_vol_reduces_weights(self, base_weights, regime_cfg):
        """HIGH vol should reduce raw weights before normalization."""
        adjusted, modifiers = apply_regime_curves_to_weights(
            weights=base_weights,
            vol_regime="HIGH",
            dd_state="NORMAL",
            regime_curves=regime_cfg,
            min_weight=0.05,
            max_weight=0.40,
            normalize_to_one=True,
        )
        
        assert modifiers["vol_multiplier"] == 0.90
        assert modifiers["combined_multiplier"] == 0.90
        # After normalization, should still sum to 1.0
        assert sum(adjusted.values()) == pytest.approx(1.0, abs=0.01)

    def test_crisis_vol_with_drawdown(self, base_weights, regime_cfg):
        """CRISIS vol + DRAWDOWN should have significant combined effect."""
        adjusted, modifiers = apply_regime_curves_to_weights(
            weights=base_weights,
            vol_regime="CRISIS",
            dd_state="DRAWDOWN",
            regime_curves=regime_cfg,
            min_weight=0.05,
            max_weight=0.40,
            normalize_to_one=True,
        )
        
        # 0.70 * 0.65 = 0.455
        assert modifiers["vol_multiplier"] == 0.70
        assert modifiers["dd_multiplier"] == 0.65
        expected_combined = 0.70 * 0.65
        assert modifiers["combined_multiplier"] == pytest.approx(expected_combined, abs=0.01)

    def test_weights_clamped_to_bounds(self, regime_cfg):
        """Weights should be clamped to min/max bounds."""
        # Very small weights that might go below min
        small_weights = {
            "trend": 0.01,
            "carry": 0.01,
            "rv_momentum": 0.01,
            "expectancy": 0.97,
        }
        
        adjusted, modifiers = apply_regime_curves_to_weights(
            weights=small_weights,
            vol_regime="CRISIS",  # 0.70 multiplier
            dd_state="DRAWDOWN",  # 0.65 multiplier
            regime_curves=regime_cfg,
            min_weight=0.05,
            max_weight=0.40,
            normalize_to_one=False,
        )
        
        # All weights should be >= min_weight after clamping
        for w in adjusted.values():
            assert w >= 0.05

    def test_regime_modifiers_metadata_complete(self, base_weights, regime_cfg):
        """Regime modifiers should contain all expected keys."""
        adjusted, modifiers = apply_regime_curves_to_weights(
            weights=base_weights,
            vol_regime="HIGH",
            dd_state="RECOVERY",
            regime_curves=regime_cfg,
            min_weight=0.05,
            max_weight=0.40,
            normalize_to_one=True,
        )
        
        assert "vol_regime" in modifiers
        assert "vol_multiplier" in modifiers
        assert "dd_state" in modifiers
        assert "dd_multiplier" in modifiers
        assert "combined_multiplier" in modifiers


class TestFactorWeightsSnapshotRegimeModifiers:
    """Test FactorWeightsSnapshot includes regime_modifiers."""

    def test_snapshot_has_regime_modifiers_field(self):
        """FactorWeightsSnapshot should have regime_modifiers field."""
        snapshot = FactorWeightsSnapshot(
            weights={"trend": 0.25, "carry": 0.25},
            factor_vols={"trend": 0.1, "carry": 0.1},
            factor_ir={"trend": 0.5, "carry": 0.4},
            regime_modifiers={
                "vol_regime": "HIGH",
                "vol_multiplier": 0.90,
                "dd_state": "NORMAL",
                "dd_multiplier": 1.0,
                "combined_multiplier": 0.90,
            },
        )
        
        assert snapshot.regime_modifiers["vol_regime"] == "HIGH"
        assert snapshot.regime_modifiers["vol_multiplier"] == 0.90

    def test_snapshot_to_dict_includes_regime_modifiers(self):
        """to_dict should include regime_modifiers."""
        snapshot = FactorWeightsSnapshot(
            weights={"trend": 0.25},
            regime_modifiers={"vol_regime": "CRISIS"},
        )
        
        d = snapshot.to_dict()
        
        assert "regime_modifiers" in d
        assert d["regime_modifiers"]["vol_regime"] == "CRISIS"

    def test_empty_regime_modifiers_default(self):
        """Default regime_modifiers should be empty dict."""
        snapshot = FactorWeightsSnapshot()
        
        assert snapshot.regime_modifiers == {}
