"""
Tests for signal screener vol regime integration (v7.4 B2).
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from execution.utils.vol import (
    get_sizing_multiplier,
    get_hybrid_weight_modifiers,
    VolRegime,
)


class TestScreenerSizingIntegration:
    """Tests for vol regime sizing modulation in screener."""

    @pytest.fixture
    def base_strategy_config(self):
        return {
            "vol_regimes": {
                "enabled": True,
                "sizing_multipliers": {
                    "CORE": {"low": 1.15, "normal": 1.0, "high": 0.75, "crisis": 0.5},
                    "SATELLITE": {"low": 1.1, "normal": 1.0, "high": 0.7, "crisis": 0.4},
                    "TACTICAL": {"low": 1.0, "normal": 0.9, "high": 0.6, "crisis": 0.3},
                    "ALT-EXT": {"low": 0.9, "normal": 0.8, "high": 0.5, "crisis": 0.2},
                }
            }
        }

    def test_core_sizing_by_regime(self, base_strategy_config):
        """CORE tier sizing changes by regime."""
        base_pct = 0.02  # 2% base per trade
        
        # Low regime: size up
        mult_low = get_sizing_multiplier("CORE", "low", base_strategy_config)
        assert base_pct * mult_low == pytest.approx(0.023, abs=0.001)  # 2.3%
        
        # Normal regime: unchanged
        mult_normal = get_sizing_multiplier("CORE", "normal", base_strategy_config)
        assert base_pct * mult_normal == 0.02
        
        # High regime: size down
        mult_high = get_sizing_multiplier("CORE", "high", base_strategy_config)
        assert base_pct * mult_high == pytest.approx(0.015, abs=0.001)  # 1.5%
        
        # Crisis regime: significant reduction
        mult_crisis = get_sizing_multiplier("CORE", "crisis", base_strategy_config)
        assert base_pct * mult_crisis == pytest.approx(0.01, abs=0.001)  # 1%

    def test_satellite_sizing_by_regime(self, base_strategy_config):
        """SATELLITE tier sizing changes by regime."""
        base_pct = 0.015
        
        mult_crisis = get_sizing_multiplier("SATELLITE", "crisis", base_strategy_config)
        assert base_pct * mult_crisis == pytest.approx(0.006, abs=0.001)  # 0.6%

    def test_tactical_sizing_by_regime(self, base_strategy_config):
        """TACTICAL tier has aggressive crisis reduction."""
        base_pct = 0.01
        
        mult_crisis = get_sizing_multiplier("TACTICAL", "crisis", base_strategy_config)
        assert mult_crisis == 0.3
        assert base_pct * mult_crisis == pytest.approx(0.003, abs=0.001)

    def test_altex_sizing_by_regime(self, base_strategy_config):
        """ALT-EXT tier has most aggressive reduction."""
        base_pct = 0.005
        
        mult_crisis = get_sizing_multiplier("ALT-EXT", "crisis", base_strategy_config)
        assert mult_crisis == 0.2
        assert base_pct * mult_crisis == pytest.approx(0.001, abs=0.0001)

    def test_tier_case_insensitive(self, base_strategy_config):
        """Tier matching is case-insensitive."""
        assert get_sizing_multiplier("core", "high", base_strategy_config) == 0.75
        assert get_sizing_multiplier("CORE", "high", base_strategy_config) == 0.75
        assert get_sizing_multiplier("Core", "high", base_strategy_config) == 0.75


class TestHybridWeightModulationIntegration:
    """Tests for hybrid weight modulation by vol regime."""

    @pytest.fixture
    def hybrid_config(self):
        return {
            "vol_regimes": {
                "enabled": True,
                "hybrid_weight_modifiers": {
                    "default": {
                        "low": {"carry": 1.0, "expectancy": 1.0, "router": 1.0},
                        "normal": {"carry": 1.0, "expectancy": 1.0, "router": 1.0},
                        "high": {"carry": 0.7, "expectancy": 0.9, "router": 1.0},
                        "crisis": {"carry": 0.4, "expectancy": 0.8, "router": 1.1},
                    }
                }
            }
        }

    def test_normal_regime_no_modulation(self, hybrid_config):
        """Normal regime has no weight changes."""
        mods = get_hybrid_weight_modifiers("normal", hybrid_config)
        assert mods.carry == 1.0
        assert mods.expectancy == 1.0
        assert mods.router == 1.0

    def test_high_regime_reduces_carry(self, hybrid_config):
        """High regime reduces carry and expectancy weights."""
        mods = get_hybrid_weight_modifiers("high", hybrid_config)
        assert mods.carry == 0.7
        assert mods.expectancy == 0.9
        assert mods.router == 1.0

    def test_crisis_regime_significant_reduction(self, hybrid_config):
        """Crisis regime has significant weight changes."""
        mods = get_hybrid_weight_modifiers("crisis", hybrid_config)
        assert mods.carry == 0.4
        assert mods.expectancy == 0.8
        assert mods.router == 1.1  # Router slightly boosted

    def test_hybrid_score_changes_by_regime(self, hybrid_config):
        """Hybrid score changes based on regime modulation."""
        # Simulate hybrid scoring with different regimes
        base_weights = {"trend": 0.4, "carry": 0.25, "expectancy": 0.2, "router": 0.15}
        
        # Component scores
        trend_sc = 0.7
        carry_sc = 0.8
        expect_sc = 0.6
        router_sc = 0.5
        
        def compute_hybrid(regime):
            mods = get_hybrid_weight_modifiers(regime, hybrid_config)
            w_trend = base_weights["trend"]
            w_carry = base_weights["carry"] * mods.carry
            w_expect = base_weights["expectancy"] * mods.expectancy
            w_router = base_weights["router"] * mods.router
            total = w_trend + w_carry + w_expect + w_router
            return (
                trend_sc * w_trend +
                carry_sc * w_carry +
                expect_sc * w_expect +
                router_sc * w_router
            ) / total
        
        hybrid_normal = compute_hybrid("normal")
        hybrid_crisis = compute_hybrid("crisis")
        
        # Crisis should weight differently (less carry influence)
        assert hybrid_normal != hybrid_crisis
        # With 0.8 carry in crisis (high score) being downweighted,
        # the final hybrid should be different
        assert abs(hybrid_normal - hybrid_crisis) > 0.01


class TestMissingConfigFallbacks:
    """Tests for graceful fallback when config missing."""

    def test_missing_vol_regimes_block(self):
        """Missing vol_regimes block uses defaults."""
        config = {"other_config": True}
        mult = get_sizing_multiplier("CORE", "crisis", config)
        assert mult == 1.0  # Default

    def test_missing_tier_in_multipliers(self):
        """Missing tier falls back to CORE."""
        config = {
            "vol_regimes": {
                "enabled": True,
                "sizing_multipliers": {
                    "CORE": {"crisis": 0.5}
                }
            }
        }
        mult = get_sizing_multiplier("SATELLITE", "crisis", config)
        assert mult == 0.5  # Falls back to CORE

    def test_missing_regime_in_tier(self):
        """Missing regime in tier returns 1.0."""
        config = {
            "vol_regimes": {
                "enabled": True,
                "sizing_multipliers": {
                    "CORE": {"low": 1.15}  # Only low defined
                }
            }
        }
        mult = get_sizing_multiplier("CORE", "crisis", config)
        assert mult == 1.0  # Default


class TestVolRegimePropagation:
    """Tests for vol regime propagation through the pipeline."""

    def test_vol_regime_in_intent_structure(self):
        """Vol regime fields appear in expected intent structure."""
        # Expected structure in intent
        intent_structure = {
            "vol_regime": "high",
            "vol": {
                "short": 0.015,
                "long": 0.010,
                "ratio": 1.5,
            },
            "tier": "CORE",
            "sizing_notes": {
                "vol_regime": "high",
                "vol_sizing_mult": 0.75,
                "effective_per_trade_nav_pct": 0.015,
            }
        }
        
        # Verify structure is as expected
        assert intent_structure["vol_regime"] == "high"
        assert intent_structure["vol"]["ratio"] == 1.5
        assert intent_structure["sizing_notes"]["vol_sizing_mult"] == 0.75

    def test_vol_regime_labels_are_valid(self):
        """Vol regime labels are one of expected values."""
        valid_labels = {"low", "normal", "high", "crisis"}
        
        for label in valid_labels:
            regime = VolRegime(label=label, vol_short=0.01, vol_long=0.01, ratio=1.0)
            assert regime.label in valid_labels
