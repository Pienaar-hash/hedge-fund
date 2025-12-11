# mypy: ignore-errors
"""
Unit tests for regime curves in conviction engine (v7.7_P6).
"""
from __future__ import annotations

import pytest
from typing import Any, Dict

from execution.conviction_engine import (
    ConvictionConfig,
    ConvictionContext,
    RegimeCurvesConfig,
    compute_conviction,
    compute_conviction_score,
    compute_conviction_score_with_regime,
    get_dd_regime_multiplier,
    get_vol_regime_multiplier,
    load_conviction_config,
    load_regime_curves_config,
)


class TestRegimeCurvesConfig:
    """Test RegimeCurvesConfig dataclass and loader."""

    def test_default_volatility_curves(self):
        """Default volatility curves should be sensible."""
        cfg = RegimeCurvesConfig()
        
        assert cfg.volatility["LOW"] == 1.10
        assert cfg.volatility["NORMAL"] == 1.00
        assert cfg.volatility["HIGH"] == 0.75
        assert cfg.volatility["CRISIS"] == 0.40

    def test_default_drawdown_curves(self):
        """Default drawdown curves should be sensible."""
        cfg = RegimeCurvesConfig()
        
        assert cfg.drawdown["NORMAL"] == 1.00
        assert cfg.drawdown["RECOVERY"] == 0.85
        assert cfg.drawdown["DRAWDOWN"] == 0.50

    def test_default_smoothing_alpha(self):
        """Default smoothing alpha should be 0.1."""
        cfg = RegimeCurvesConfig()
        assert cfg.smoothing_alpha == 0.1

    def test_load_regime_curves_config_from_dict(self):
        """Load regime curves from config dict."""
        conviction_cfg = {
            "regime_curves": {
                "volatility": {
                    "LOW": 1.20,
                    "NORMAL": 1.00,
                    "HIGH": 0.80,
                    "CRISIS": 0.30,
                },
                "drawdown": {
                    "NORMAL": 1.00,
                    "RECOVERY": 0.75,
                    "DRAWDOWN": 0.40,
                },
                "smoothing_alpha": 0.2,
            }
        }
        
        cfg = load_regime_curves_config(conviction_cfg)
        
        assert cfg.volatility["LOW"] == 1.20
        assert cfg.volatility["CRISIS"] == 0.30
        assert cfg.drawdown["RECOVERY"] == 0.75
        assert cfg.smoothing_alpha == 0.2

    def test_load_regime_curves_config_empty(self):
        """Return defaults when config is empty."""
        cfg = load_regime_curves_config({})
        
        assert cfg.volatility["NORMAL"] == 1.00
        assert cfg.drawdown["NORMAL"] == 1.00


class TestVolRegimeMultiplier:
    """Test get_vol_regime_multiplier function."""

    def test_low_vol_increases_conviction(self):
        """LOW volatility should increase conviction (multiplier > 1)."""
        cfg = RegimeCurvesConfig()
        mult = get_vol_regime_multiplier("low", cfg)
        
        assert mult == 1.10
        assert mult > 1.0

    def test_normal_vol_neutral(self):
        """NORMAL volatility should be neutral (multiplier = 1)."""
        cfg = RegimeCurvesConfig()
        mult = get_vol_regime_multiplier("normal", cfg)
        
        assert mult == 1.00

    def test_high_vol_decreases_conviction(self):
        """HIGH volatility should decrease conviction (multiplier < 1)."""
        cfg = RegimeCurvesConfig()
        mult = get_vol_regime_multiplier("high", cfg)
        
        assert mult == 0.75
        assert mult < 1.0

    def test_crisis_vol_significantly_decreases_conviction(self):
        """CRISIS volatility should significantly decrease conviction."""
        cfg = RegimeCurvesConfig()
        mult = get_vol_regime_multiplier("crisis", cfg)
        
        assert mult == 0.40
        assert mult < 0.5

    def test_uppercase_lowercase_handled(self):
        """Both uppercase and lowercase regime labels should work."""
        cfg = RegimeCurvesConfig()
        
        assert get_vol_regime_multiplier("LOW", cfg) == 1.10
        assert get_vol_regime_multiplier("low", cfg) == 1.10
        assert get_vol_regime_multiplier("HIGH", cfg) == 0.75
        assert get_vol_regime_multiplier("high", cfg) == 0.75

    def test_unknown_regime_defaults_to_one(self):
        """Unknown regime should default to multiplier of 1.0."""
        cfg = RegimeCurvesConfig()
        mult = get_vol_regime_multiplier("unknown_regime", cfg)
        
        assert mult == 1.0


class TestDdRegimeMultiplier:
    """Test get_dd_regime_multiplier function."""

    def test_normal_dd_neutral(self):
        """NORMAL drawdown state should be neutral."""
        cfg = RegimeCurvesConfig()
        mult = get_dd_regime_multiplier("NORMAL", cfg)
        
        assert mult == 1.00

    def test_recovery_dd_reduces_conviction(self):
        """RECOVERY should reduce conviction somewhat."""
        cfg = RegimeCurvesConfig()
        mult = get_dd_regime_multiplier("RECOVERY", cfg)
        
        assert mult == 0.85
        assert mult < 1.0

    def test_drawdown_significantly_reduces_conviction(self):
        """DRAWDOWN should significantly reduce conviction."""
        cfg = RegimeCurvesConfig()
        mult = get_dd_regime_multiplier("DRAWDOWN", cfg)
        
        assert mult == 0.50
        assert mult < 0.6

    def test_unknown_dd_state_defaults_to_one(self):
        """Unknown DD state should default to multiplier of 1.0."""
        cfg = RegimeCurvesConfig()
        mult = get_dd_regime_multiplier("unknown", cfg)
        
        assert mult == 1.0


class TestComputeConvictionScoreWithRegime:
    """Test compute_conviction_score_with_regime function."""

    @pytest.fixture
    def base_context(self) -> ConvictionContext:
        """Base context with moderate scores."""
        return ConvictionContext(
            hybrid_score=0.7,
            expectancy_alpha=0.6,
            router_quality=0.9,
            trend_strength=0.5,
            vol_regime="normal",
            dd_state="NORMAL",
            risk_mode="OK",
        )

    def test_normal_regime_no_penalty(self, base_context):
        """NORMAL vol + NORMAL dd should have combined multiplier ~1.0."""
        cfg = ConvictionConfig()
        
        score, modifiers = compute_conviction_score_with_regime(base_context, cfg)
        
        assert modifiers["vol_multiplier"] == 1.0
        assert modifiers["dd_multiplier"] == 1.0
        assert modifiers["combined_multiplier"] == 1.0

    def test_high_vol_reduces_score(self, base_context):
        """HIGH vol regime should reduce conviction score."""
        cfg = ConvictionConfig()
        base_context.vol_regime = "high"
        
        score, modifiers = compute_conviction_score_with_regime(base_context, cfg)
        
        assert modifiers["vol_multiplier"] == 0.75
        assert modifiers["combined_multiplier"] < 1.0

    def test_drawdown_reduces_score(self, base_context):
        """DRAWDOWN state should reduce conviction score."""
        cfg = ConvictionConfig()
        base_context.dd_state = "DRAWDOWN"
        
        score, modifiers = compute_conviction_score_with_regime(base_context, cfg)
        
        assert modifiers["dd_multiplier"] == 0.50
        assert modifiers["combined_multiplier"] < 1.0

    def test_crisis_plus_drawdown_severe_reduction(self, base_context):
        """CRISIS vol + DRAWDOWN should severely reduce conviction."""
        cfg = ConvictionConfig()
        base_context.vol_regime = "crisis"
        base_context.dd_state = "DRAWDOWN"
        
        score, modifiers = compute_conviction_score_with_regime(base_context, cfg)
        
        # 0.40 * 0.50 = 0.20
        assert modifiers["combined_multiplier"] == pytest.approx(0.20, rel=0.01)

    def test_score_clamped_to_zero_one(self, base_context):
        """Final score should always be in [0.0, 1.0]."""
        cfg = ConvictionConfig()
        
        # Set extreme values
        base_context.hybrid_score = 1.0
        base_context.expectancy_alpha = 1.0
        base_context.router_quality = 1.0
        base_context.trend_strength = 1.0
        base_context.vol_regime = "low"  # 1.10 multiplier
        base_context.dd_state = "NORMAL"  # 1.00 multiplier
        
        score, modifiers = compute_conviction_score_with_regime(base_context, cfg)
        
        assert 0.0 <= score <= 1.0

    def test_modifiers_metadata_complete(self, base_context):
        """Modifiers should contain all expected keys."""
        cfg = ConvictionConfig()
        
        score, modifiers = compute_conviction_score_with_regime(base_context, cfg)
        
        assert "vol_regime" in modifiers
        assert "vol_multiplier" in modifiers
        assert "dd_state" in modifiers
        assert "dd_multiplier" in modifiers
        assert "combined_multiplier" in modifiers
        assert "base_score" in modifiers
        assert "adjusted_score" in modifiers
        assert "smoothed" in modifiers


class TestComputeConvictionWithRegime:
    """Test compute_conviction includes regime modifiers."""

    def test_conviction_result_has_regime_modifiers(self):
        """ConvictionResult should include regime_modifiers field."""
        ctx = ConvictionContext(
            hybrid_score=0.7,
            expectancy_alpha=0.6,
            router_quality=0.9,
            trend_strength=0.5,
            vol_regime="high",
            dd_state="RECOVERY",
        )
        cfg = ConvictionConfig()
        
        result = compute_conviction(ctx, cfg)
        
        assert hasattr(result, "regime_modifiers")
        assert isinstance(result.regime_modifiers, dict)
        assert result.regime_modifiers.get("vol_multiplier") == 0.75
        assert result.regime_modifiers.get("dd_multiplier") == 0.85


class TestSmoothingAlpha:
    """Test EMA smoothing of regime-adjusted conviction."""

    def test_smoothing_with_alpha_one_no_memory(self):
        """Alpha=1.0 should use only current value."""
        ctx = ConvictionContext(
            hybrid_score=0.5,
            expectancy_alpha=0.5,
            router_quality=0.8,
            trend_strength=0.5,
        )
        cfg = ConvictionConfig()
        cfg.regime_curves.smoothing_alpha = 1.0
        
        # First call
        score1, _ = compute_conviction_score_with_regime(ctx, cfg, symbol="TEST_ALPHA")
        
        # Change inputs significantly
        ctx.hybrid_score = 1.0
        
        # Second call should immediately reflect new score (no memory)
        score2, _ = compute_conviction_score_with_regime(ctx, cfg, symbol="TEST_ALPHA")
        
        # With alpha=1.0, score should change immediately
        assert score2 > score1
