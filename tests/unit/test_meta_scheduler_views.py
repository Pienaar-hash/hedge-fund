"""
Unit tests for Meta-Scheduler public API views (v7.8_P1).

Tests:
- get_factor_meta_weights() with None/default state
- get_conviction_meta_strength() defaults
- get_category_meta_overlays() defaults
- is_meta_scheduler_active() logic
"""

import pytest

from execution.meta_scheduler import (
    MetaSchedulerConfig,
    MetaSchedulerState,
    FactorMetaState,
    ConvictionMetaState,
    CategoryMetaState,
    create_neutral_state,
    get_factor_meta_weights,
    get_conviction_meta_strength,
    get_category_meta_overlays,
    is_meta_scheduler_active,
    NEUTRAL_MULTIPLIER,
)


# ---------------------------------------------------------------------------
# get_factor_meta_weights Tests
# ---------------------------------------------------------------------------


class TestGetFactorMetaWeights:
    """Tests for get_factor_meta_weights()."""

    def test_none_state_returns_empty(self):
        """None state returns empty dict."""
        weights = get_factor_meta_weights(None)
        assert weights == {}

    def test_empty_state_returns_empty(self):
        """Neutral state returns empty dict."""
        state = create_neutral_state()
        weights = get_factor_meta_weights(state)
        assert weights == {}

    def test_returns_copy_of_weights(self):
        """Returns a copy, not the original dict."""
        state = MetaSchedulerState(
            factor_state=FactorMetaState(
                meta_weights={"momentum": 1.05, "carry": 0.95}
            ),
        )
        
        weights = get_factor_meta_weights(state)
        
        assert weights == {"momentum": 1.05, "carry": 0.95}
        
        # Modifying returned dict shouldn't affect state
        weights["momentum"] = 2.0
        assert state.factor_state.meta_weights["momentum"] == 1.05

    def test_returns_all_factors(self):
        """Returns weights for all tracked factors."""
        state = MetaSchedulerState(
            factor_state=FactorMetaState(
                meta_weights={
                    "momentum": 1.08,
                    "carry": 0.95,
                    "value": 1.02,
                    "volatility": 0.98,
                }
            ),
        )
        
        weights = get_factor_meta_weights(state)
        
        assert len(weights) == 4
        assert weights["momentum"] == 1.08
        assert weights["carry"] == 0.95


# ---------------------------------------------------------------------------
# get_conviction_meta_strength Tests
# ---------------------------------------------------------------------------


class TestGetConvictionMetaStrength:
    """Tests for get_conviction_meta_strength()."""

    def test_none_state_returns_neutral(self):
        """None state returns neutral multiplier (1.0)."""
        strength = get_conviction_meta_strength(None)
        assert strength == NEUTRAL_MULTIPLIER

    def test_neutral_state_returns_neutral(self):
        """Neutral state returns 1.0."""
        state = create_neutral_state()
        strength = get_conviction_meta_strength(state)
        assert strength == NEUTRAL_MULTIPLIER

    def test_returns_actual_strength(self):
        """Returns the actual global_strength value."""
        state = MetaSchedulerState(
            conviction_state=ConvictionMetaState(
                global_strength=1.12,
                ema_health=0.75,
            ),
        )
        
        strength = get_conviction_meta_strength(state)
        
        assert strength == 1.12

    def test_low_strength(self):
        """Returns low strength correctly."""
        state = MetaSchedulerState(
            conviction_state=ConvictionMetaState(
                global_strength=0.88,
                ema_health=0.35,
            ),
        )
        
        strength = get_conviction_meta_strength(state)
        
        assert strength == 0.88


# ---------------------------------------------------------------------------
# get_category_meta_overlays Tests
# ---------------------------------------------------------------------------


class TestGetCategoryMetaOverlays:
    """Tests for get_category_meta_overlays()."""

    def test_none_state_returns_empty(self):
        """None state returns empty dict."""
        overlays = get_category_meta_overlays(None)
        assert overlays == {}

    def test_neutral_state_returns_empty(self):
        """Neutral state returns empty dict."""
        state = create_neutral_state()
        overlays = get_category_meta_overlays(state)
        assert overlays == {}

    def test_returns_copy_of_overlays(self):
        """Returns a copy, not the original dict."""
        state = MetaSchedulerState(
            category_state=CategoryMetaState(
                category_overlays={"btc": 1.10, "eth": 0.90}
            ),
        )
        
        overlays = get_category_meta_overlays(state)
        
        assert overlays == {"btc": 1.10, "eth": 0.90}
        
        # Modifying returned dict shouldn't affect state
        overlays["btc"] = 2.0
        assert state.category_state.category_overlays["btc"] == 1.10

    def test_returns_all_categories(self):
        """Returns overlays for all tracked categories."""
        state = MetaSchedulerState(
            category_state=CategoryMetaState(
                category_overlays={
                    "btc": 1.08,
                    "eth": 1.02,
                    "alt-l1": 0.95,
                    "defi": 0.90,
                }
            ),
        )
        
        overlays = get_category_meta_overlays(state)
        
        assert len(overlays) == 4
        assert overlays["btc"] == 1.08
        assert overlays["defi"] == 0.90


# ---------------------------------------------------------------------------
# is_meta_scheduler_active Tests
# ---------------------------------------------------------------------------


class TestIsMetaSchedulerActive:
    """Tests for is_meta_scheduler_active()."""

    def test_disabled_config_returns_false(self):
        """Disabled config returns False regardless of state."""
        cfg = MetaSchedulerConfig(enabled=False)
        state = MetaSchedulerState(stats={"sample_count": 100})
        
        assert is_meta_scheduler_active(cfg, state) is False

    def test_none_state_returns_false(self):
        """None state returns False even if enabled."""
        cfg = MetaSchedulerConfig(enabled=True, min_samples=10)
        
        assert is_meta_scheduler_active(cfg, None) is False

    def test_below_min_samples_returns_false(self):
        """Below min_samples returns False."""
        cfg = MetaSchedulerConfig(enabled=True, min_samples=100)
        state = MetaSchedulerState(stats={"sample_count": 50})
        
        assert is_meta_scheduler_active(cfg, state) is False

    def test_at_min_samples_returns_true(self):
        """At min_samples returns True."""
        cfg = MetaSchedulerConfig(enabled=True, min_samples=100)
        state = MetaSchedulerState(stats={"sample_count": 100})
        
        assert is_meta_scheduler_active(cfg, state) is True

    def test_above_min_samples_returns_true(self):
        """Above min_samples returns True."""
        cfg = MetaSchedulerConfig(enabled=True, min_samples=100)
        state = MetaSchedulerState(stats={"sample_count": 500})
        
        assert is_meta_scheduler_active(cfg, state) is True

    def test_missing_sample_count_returns_false(self):
        """Missing sample_count in stats returns False."""
        cfg = MetaSchedulerConfig(enabled=True, min_samples=10)
        state = MetaSchedulerState(stats={})
        
        assert is_meta_scheduler_active(cfg, state) is False


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for views."""

    def test_factor_weights_with_zero_values(self):
        """Handles zero values in factor weights."""
        state = MetaSchedulerState(
            factor_state=FactorMetaState(
                meta_weights={"momentum": 0.0, "carry": 1.0}
            ),
        )
        
        weights = get_factor_meta_weights(state)
        
        assert weights["momentum"] == 0.0
        assert weights["carry"] == 1.0

    def test_conviction_at_bounds(self):
        """Handles conviction at extreme bounds."""
        # Low bound
        state_low = MetaSchedulerState(
            conviction_state=ConvictionMetaState(global_strength=0.85),
        )
        assert get_conviction_meta_strength(state_low) == 0.85
        
        # High bound
        state_high = MetaSchedulerState(
            conviction_state=ConvictionMetaState(global_strength=1.15),
        )
        assert get_conviction_meta_strength(state_high) == 1.15

    def test_empty_string_category(self):
        """Handles empty string category name."""
        state = MetaSchedulerState(
            category_state=CategoryMetaState(
                category_overlays={"": 1.05, "btc": 1.10}
            ),
        )
        
        overlays = get_category_meta_overlays(state)
        
        assert overlays[""] == 1.05
        assert overlays["btc"] == 1.10
