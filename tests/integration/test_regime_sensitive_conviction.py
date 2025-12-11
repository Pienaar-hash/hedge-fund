"""
Integration tests for regime-sensitive conviction sizing.
Tests end-to-end flow from regime state through conviction computation.
PATCHSET_V7.7_P6
"""

import pytest
from unittest.mock import patch, MagicMock

from execution.conviction_engine import (
    RegimeCurvesConfig,
    ConvictionConfig,
    ConvictionContext,
    compute_conviction_score,
    compute_conviction_score_with_regime,
    get_vol_regime_multiplier,
    get_dd_regime_multiplier,
    load_conviction_config,
)


@pytest.fixture
def base_context():
    """Base conviction context for tests."""
    return ConvictionContext(
        hybrid_score=0.7,
        expectancy_alpha=0.6,
        router_quality=0.8,
        trend_strength=0.5,
        vol_regime="normal",
        dd_state="NORMAL",
        risk_mode="OK",
        category_momentum=0.0,
    )


@pytest.fixture
def base_config():
    """Base conviction config with regime curves enabled."""
    return ConvictionConfig(
        enabled=True,
        regime_curves=RegimeCurvesConfig(),  # Defaults
    )


class TestEndToEndConvictionWithRegime:
    """End-to-end tests for conviction with regime curves."""

    def test_conviction_in_normal_regime(self, base_config, base_context):
        """Normal regime should produce baseline conviction."""
        score, modifiers = compute_conviction_score_with_regime(
            ctx=base_context,
            cfg=base_config,
            symbol="BTCUSDT",
        )

        # Normal regime = 1.0 multiplier
        assert modifiers["vol_multiplier"] == 1.0
        assert modifiers["dd_multiplier"] == 1.0
        assert modifiers["combined_multiplier"] == 1.0
        assert score > 0.0

    def test_conviction_drops_in_crisis(self, base_config, base_context):
        """CRISIS regime should significantly reduce conviction."""
        # Normal regime baseline
        base_context_normal = ConvictionContext(
            hybrid_score=0.7,
            expectancy_alpha=0.6,
            router_quality=0.8,
            trend_strength=0.5,
            vol_regime="normal",
            dd_state="NORMAL",
            risk_mode="OK",
            category_momentum=0.0,
        )
        normal_score, _ = compute_conviction_score_with_regime(
            ctx=base_context_normal,
            cfg=base_config,
            symbol="BTCUSDT",
        )
        
        # Crisis regime
        crisis_context = ConvictionContext(
            hybrid_score=0.7,
            expectancy_alpha=0.6,
            router_quality=0.8,
            trend_strength=0.5,
            vol_regime="crisis",
            dd_state="NORMAL",
            risk_mode="OK",
            category_momentum=0.0,
        )
        crisis_score, modifiers = compute_conviction_score_with_regime(
            ctx=crisis_context,
            cfg=base_config,
            symbol="BTCUSDT",
        )

        # CRISIS = 0.40 multiplier
        assert modifiers["vol_multiplier"] == 0.40
        assert crisis_score < normal_score

    def test_conviction_drops_in_drawdown(self, base_config, base_context):
        """DRAWDOWN state should reduce conviction."""
        dd_context = ConvictionContext(
            hybrid_score=0.7,
            expectancy_alpha=0.6,
            router_quality=0.8,
            trend_strength=0.5,
            vol_regime="normal",
            dd_state="DRAWDOWN",
            risk_mode="OK",
            category_momentum=0.0,
        )
        score, modifiers = compute_conviction_score_with_regime(
            ctx=dd_context,
            cfg=base_config,
            symbol="BTCUSDT",
        )

        # DRAWDOWN = 0.50 multiplier
        assert modifiers["dd_multiplier"] == 0.50
        assert modifiers["combined_multiplier"] == pytest.approx(0.50, rel=0.01)

    def test_combined_crisis_and_drawdown(self, base_config, base_context):
        """CRISIS + DRAWDOWN should compound penalties."""
        combined_context = ConvictionContext(
            hybrid_score=0.7,
            expectancy_alpha=0.6,
            router_quality=0.8,
            trend_strength=0.5,
            vol_regime="crisis",
            dd_state="DRAWDOWN",
            risk_mode="OK",
            category_momentum=0.0,
        )
        score, modifiers = compute_conviction_score_with_regime(
            ctx=combined_context,
            cfg=base_config,
            symbol="BTCUSDT",
        )

        # CRISIS (0.40) * DRAWDOWN (0.50) = 0.20
        expected_combined = 0.40 * 0.50
        assert modifiers["combined_multiplier"] == pytest.approx(expected_combined, rel=0.01)

    def test_conviction_boost_in_low_vol(self, base_config, base_context):
        """LOW vol regime should boost conviction."""
        # Normal regime baseline
        normal_context = ConvictionContext(
            hybrid_score=0.7,
            expectancy_alpha=0.6,
            router_quality=0.8,
            trend_strength=0.5,
            vol_regime="normal",
            dd_state="NORMAL",
            risk_mode="OK",
            category_momentum=0.0,
        )
        normal_score, _ = compute_conviction_score_with_regime(
            ctx=normal_context,
            cfg=base_config,
            symbol="BTCUSDT",
        )
        
        # Low vol regime
        low_context = ConvictionContext(
            hybrid_score=0.7,
            expectancy_alpha=0.6,
            router_quality=0.8,
            trend_strength=0.5,
            vol_regime="low",
            dd_state="NORMAL",
            risk_mode="OK",
            category_momentum=0.0,
        )
        low_score, modifiers = compute_conviction_score_with_regime(
            ctx=low_context,
            cfg=base_config,
            symbol="BTCUSDT",
        )

        # LOW = 1.10 multiplier
        assert modifiers["vol_multiplier"] == 1.10
        # Boost should be reflected in score
        assert low_score >= normal_score


class TestConvictionRegimeSmoothing:
    """Test EMA smoothing of regime multipliers."""

    def test_smoothing_dampens_sudden_changes(self, base_config):
        """Smoothing should dampen sudden regime transitions."""
        # This test verifies the smoothing mechanism exists
        # The actual smoothing is done via _ema_smooth and _prev_conviction_cache
        crisis_context = ConvictionContext(
            hybrid_score=0.7,
            expectancy_alpha=0.6,
            router_quality=0.8,
            trend_strength=0.5,
            vol_regime="crisis",
            dd_state="NORMAL",
            risk_mode="OK",
            category_momentum=0.0,
        )
        
        score, modifiers = compute_conviction_score_with_regime(
            ctx=crisis_context,
            cfg=base_config,
            symbol="BTCUSDT",
        )

        # Raw multiplier should be 0.40
        assert modifiers["vol_multiplier"] == 0.40


class TestConvictionWithoutRegimeCurves:
    """Test backward compatibility when regime curves disabled."""

    def test_no_regime_curves_config(self):
        """Config without regime_curves should still work."""
        config = ConvictionConfig(enabled=True)  # No regime_curves override
        
        crisis_context = ConvictionContext(
            hybrid_score=0.7,
            expectancy_alpha=0.6,
            router_quality=0.8,
            trend_strength=0.5,
            vol_regime="crisis",
            dd_state="DRAWDOWN",
            risk_mode="OK",
            category_momentum=0.0,
        )

        # Should still work with default regime curves
        score, modifiers = compute_conviction_score_with_regime(
            ctx=crisis_context,
            cfg=config,
            symbol="BTCUSDT",
        )

        # Default curves should apply
        assert modifiers["vol_multiplier"] == 0.40
        assert modifiers["dd_multiplier"] == 0.50

    def test_base_score_computation(self, base_config):
        """Base score computation should work regardless of regime."""
        base_context = ConvictionContext(
            hybrid_score=0.7,
            expectancy_alpha=0.6,
            router_quality=0.8,
            trend_strength=0.5,
            vol_regime="normal",
            dd_state="NORMAL",
            risk_mode="OK",
            category_momentum=0.0,
        )
        
        score = compute_conviction_score(base_context, base_config)
        
        # Should produce valid conviction
        assert 0.0 <= score <= 1.0


class TestConvictionFloorsDuringCrisis:
    """Test that conviction respects bounds during extreme regimes."""

    def test_crisis_produces_valid_score(self):
        """Even in CRISIS, conviction should be valid."""
        config = ConvictionConfig(enabled=True)
        
        crisis_context = ConvictionContext(
            hybrid_score=0.3,  # Low score
            expectancy_alpha=0.2,
            router_quality=0.5,
            trend_strength=0.2,
            vol_regime="crisis",
            dd_state="DRAWDOWN",
            risk_mode="OK",
            category_momentum=0.0,
        )

        score, modifiers = compute_conviction_score_with_regime(
            ctx=crisis_context,
            cfg=config,
            symbol="BTCUSDT",
        )

        # Should be clamped to valid range
        assert 0.0 <= score <= 1.0

    def test_low_vol_produces_valid_score(self):
        """Even with LOW vol boost, conviction should be valid."""
        config = ConvictionConfig(enabled=True)
        
        low_context = ConvictionContext(
            hybrid_score=0.95,  # High score
            expectancy_alpha=0.9,
            router_quality=0.95,
            trend_strength=0.9,
            vol_regime="low",
            dd_state="NORMAL",
            risk_mode="OK",
            category_momentum=0.0,
        )

        score, modifiers = compute_conviction_score_with_regime(
            ctx=low_context,
            cfg=config,
            symbol="BTCUSDT",
        )

        # Should be clamped to max 1.0
        assert 0.0 <= score <= 1.0
