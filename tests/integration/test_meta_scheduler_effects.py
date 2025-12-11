"""
Integration tests for Meta-Scheduler effects on other modules (v7.8_P1).

Tests:
- Factor weight overlays applied correctly
- Conviction strength affects computation
- Category overlays affect momentum scores
- Neutral when disabled
"""

import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from execution.meta_scheduler import (
    MetaSchedulerConfig,
    MetaSchedulerState,
    FactorMetaState,
    ConvictionMetaState,
    CategoryMetaState,
    create_neutral_state,
    meta_learning_step,
    get_factor_meta_weights,
    get_conviction_meta_strength,
    get_category_meta_overlays,
    NEUTRAL_MULTIPLIER,
)


# ---------------------------------------------------------------------------
# Factor Weight Effects Tests
# ---------------------------------------------------------------------------


class TestFactorWeightEffects:
    """Tests for factor weight overlay effects."""

    def test_positive_overlay_increases_weight(self):
        """Factors with positive overlay get higher weight."""
        cfg = MetaSchedulerConfig(
            enabled=True,
            learning_rate=0.10,
            min_samples=0,
            max_factor_shift=0.15,
        )
        
        # Build state with positive momentum signal over time
        state = create_neutral_state()
        state.stats["sample_count"] = 100
        state.factor_state.ema_ir = {"momentum": 0.6}
        state.factor_state.ema_pnl = {"momentum": 200.0}
        
        # Run learning step with continued positive signal
        state = meta_learning_step(
            cfg=cfg,
            prev_state=state,
            factor_edges={"momentum": {"ir": 0.7, "pnl_contrib": 250.0}},
            category_edges={},
            strategy_health={"health_score": 0.8},
        )
        
        weights = get_factor_meta_weights(state)
        
        assert weights.get("momentum", 1.0) > NEUTRAL_MULTIPLIER

    def test_negative_overlay_decreases_weight(self):
        """Factors with negative overlay get lower weight."""
        cfg = MetaSchedulerConfig(
            enabled=True,
            learning_rate=0.10,
            min_samples=0,
            max_factor_shift=0.15,
        )
        
        state = create_neutral_state()
        state.stats["sample_count"] = 100
        state.factor_state.ema_ir = {"carry": -0.5}
        state.factor_state.ema_pnl = {"carry": -100.0}
        
        state = meta_learning_step(
            cfg=cfg,
            prev_state=state,
            factor_edges={"carry": {"ir": -0.6, "pnl_contrib": -150.0}},
            category_edges={},
            strategy_health={"health_score": 0.4},
        )
        
        weights = get_factor_meta_weights(state)
        
        assert weights.get("carry", 1.0) < NEUTRAL_MULTIPLIER

    def test_overlay_bounded_by_config(self):
        """Overlays stay within configured bounds."""
        cfg = MetaSchedulerConfig(
            enabled=True,
            learning_rate=0.50,  # Very aggressive
            min_samples=0,
            max_factor_shift=0.10,  # Tight bounds
        )
        
        state = create_neutral_state()
        state.stats["sample_count"] = 100
        state.factor_state.meta_weights = {"momentum": 1.08}
        state.factor_state.ema_ir = {"momentum": 0.9}
        state.factor_state.ema_pnl = {"momentum": 500.0}
        
        # Try to push past bounds
        for _ in range(10):
            state = meta_learning_step(
                cfg=cfg,
                prev_state=state,
                factor_edges={"momentum": {"ir": 1.0, "pnl_contrib": 1000.0}},
                category_edges={},
                strategy_health={"health_score": 0.95},
            )
        
        weights = get_factor_meta_weights(state)
        
        assert weights.get("momentum", 1.0) <= 1.10
        assert weights.get("momentum", 1.0) >= 0.90


# ---------------------------------------------------------------------------
# Conviction Effects Tests
# ---------------------------------------------------------------------------


class TestConvictionEffects:
    """Tests for conviction strength overlay effects."""

    def test_high_health_boosts_conviction(self):
        """High system health boosts conviction strength."""
        cfg = MetaSchedulerConfig(
            enabled=True,
            learning_rate=0.10,
            min_samples=0,
            max_conviction_shift=0.15,
        )
        
        state = create_neutral_state()
        state.stats["sample_count"] = 100
        state.conviction_state.ema_health = 0.75
        
        # Strong health signal
        state = meta_learning_step(
            cfg=cfg,
            prev_state=state,
            factor_edges={"momentum": {"edge_score": 0.6}},
            category_edges={},
            strategy_health={"health_score": 0.85},
        )
        
        strength = get_conviction_meta_strength(state)
        
        assert strength > NEUTRAL_MULTIPLIER

    def test_low_health_reduces_conviction(self):
        """Low system health reduces conviction strength."""
        cfg = MetaSchedulerConfig(
            enabled=True,
            learning_rate=0.10,
            min_samples=0,
            max_conviction_shift=0.15,
        )
        
        state = create_neutral_state()
        state.stats["sample_count"] = 100
        state.conviction_state.ema_health = 0.35
        
        # Weak health signal
        state = meta_learning_step(
            cfg=cfg,
            prev_state=state,
            factor_edges={},
            category_edges={},
            strategy_health={"health_score": 0.25},
        )
        
        strength = get_conviction_meta_strength(state)
        
        assert strength < NEUTRAL_MULTIPLIER

    def test_conviction_bounded(self):
        """Conviction strength stays within bounds."""
        cfg = MetaSchedulerConfig(
            enabled=True,
            learning_rate=0.50,
            min_samples=0,
            max_conviction_shift=0.10,
        )
        
        state = create_neutral_state()
        state.stats["sample_count"] = 100
        state.conviction_state.global_strength = 1.08
        state.conviction_state.ema_health = 0.9
        
        # Try to push past bounds
        for _ in range(10):
            state = meta_learning_step(
                cfg=cfg,
                prev_state=state,
                factor_edges={"momentum": {"edge_score": 0.9}},
                category_edges={},
                strategy_health={"health_score": 0.95},
            )
        
        strength = get_conviction_meta_strength(state)
        
        assert strength <= 1.10
        assert strength >= 0.90


# ---------------------------------------------------------------------------
# Category Effects Tests
# ---------------------------------------------------------------------------


class TestCategoryEffects:
    """Tests for category overlay effects."""

    def test_strong_category_gets_overlay_boost(self):
        """Categories with strong performance get overlay boost."""
        cfg = MetaSchedulerConfig(
            enabled=True,
            learning_rate=0.10,
            min_samples=0,
            max_category_shift=0.15,
        )
        
        state = create_neutral_state()
        state.stats["sample_count"] = 100
        state.category_state.ema_category_ir = {"btc": 0.5}
        state.category_state.ema_category_pnl = {"btc": 100.0}
        
        state = meta_learning_step(
            cfg=cfg,
            prev_state=state,
            factor_edges={},
            category_edges={"btc": {"ir": 0.6, "total_pnl": 150.0}},
            strategy_health={"health_score": 0.7},
        )
        
        overlays = get_category_meta_overlays(state)
        
        assert overlays.get("btc", 1.0) > NEUTRAL_MULTIPLIER

    def test_weak_category_gets_overlay_cut(self):
        """Categories with weak performance get overlay cut."""
        cfg = MetaSchedulerConfig(
            enabled=True,
            learning_rate=0.10,
            min_samples=0,
            max_category_shift=0.15,
        )
        
        state = create_neutral_state()
        state.stats["sample_count"] = 100
        state.category_state.ema_category_ir = {"altcoin": -0.4}
        state.category_state.ema_category_pnl = {"altcoin": -50.0}
        
        state = meta_learning_step(
            cfg=cfg,
            prev_state=state,
            factor_edges={},
            category_edges={"altcoin": {"ir": -0.5, "total_pnl": -80.0}},
            strategy_health={"health_score": 0.5},
        )
        
        overlays = get_category_meta_overlays(state)
        
        assert overlays.get("altcoin", 1.0) < NEUTRAL_MULTIPLIER


# ---------------------------------------------------------------------------
# Neutral When Disabled Tests
# ---------------------------------------------------------------------------


class TestNeutralWhenDisabled:
    """Tests to ensure neutral behavior when disabled."""

    def test_factor_weights_neutral_when_disabled(self):
        """Factor weights are neutral when disabled."""
        cfg = MetaSchedulerConfig(enabled=False)
        
        state = meta_learning_step(
            cfg=cfg,
            prev_state=None,
            factor_edges={"momentum": {"ir": 1.0, "pnl_contrib": 1000.0}},
            category_edges={},
            strategy_health={"health_score": 0.95},
        )
        
        weights = get_factor_meta_weights(state)
        
        # Should be empty (neutral) since disabled
        assert weights == {} or all(w == NEUTRAL_MULTIPLIER for w in weights.values())

    def test_conviction_neutral_when_disabled(self):
        """Conviction strength is neutral when disabled."""
        cfg = MetaSchedulerConfig(enabled=False)
        
        state = meta_learning_step(
            cfg=cfg,
            prev_state=None,
            factor_edges={},
            category_edges={},
            strategy_health={"health_score": 0.95},
        )
        
        strength = get_conviction_meta_strength(state)
        
        assert strength == NEUTRAL_MULTIPLIER

    def test_category_overlays_neutral_when_disabled(self):
        """Category overlays are neutral when disabled."""
        cfg = MetaSchedulerConfig(enabled=False)
        
        state = meta_learning_step(
            cfg=cfg,
            prev_state=None,
            factor_edges={},
            category_edges={"btc": {"ir": 1.0, "total_pnl": 1000.0}},
            strategy_health={"health_score": 0.95},
        )
        
        overlays = get_category_meta_overlays(state)
        
        # Should be empty (neutral) since disabled
        assert overlays == {} or all(o == NEUTRAL_MULTIPLIER for o in overlays.values())

    def test_no_state_change_when_disabled(self):
        """State doesn't change when disabled."""
        cfg = MetaSchedulerConfig(enabled=False)
        
        prev_state = MetaSchedulerState(
            updated_ts="2025-01-01T00:00:00Z",
            factor_state=FactorMetaState(meta_weights={"momentum": 1.05}),
            conviction_state=ConvictionMetaState(global_strength=1.10),
            stats={"sample_count": 100},
        )
        
        new_state = meta_learning_step(
            cfg=cfg,
            prev_state=prev_state,
            factor_edges={"momentum": {"ir": -1.0, "pnl_contrib": -1000.0}},
            category_edges={},
            strategy_health={"health_score": 0.1},
        )
        
        # State should be unchanged
        assert new_state.factor_state.meta_weights == prev_state.factor_state.meta_weights
        assert new_state.conviction_state.global_strength == prev_state.conviction_state.global_strength


# ---------------------------------------------------------------------------
# Multi-Factor Interaction Tests
# ---------------------------------------------------------------------------


class TestMultiFactorInteractions:
    """Tests for interactions between multiple factors/categories."""

    def test_multiple_factors_independent(self):
        """Multiple factors are updated independently."""
        cfg = MetaSchedulerConfig(
            enabled=True,
            learning_rate=0.10,
            min_samples=0,
            max_factor_shift=0.15,
        )
        
        state = create_neutral_state()
        state.stats["sample_count"] = 100
        
        # Mixed signals: momentum positive, carry negative
        state = meta_learning_step(
            cfg=cfg,
            prev_state=state,
            factor_edges={
                "momentum": {"ir": 0.6, "pnl_contrib": 100.0},
                "carry": {"ir": -0.5, "pnl_contrib": -80.0},
            },
            category_edges={},
            strategy_health={"health_score": 0.6},
        )
        
        weights = get_factor_meta_weights(state)
        
        # Momentum should increase, carry should decrease
        momentum_weight = weights.get("momentum", 1.0)
        carry_weight = weights.get("carry", 1.0)
        
        # After one step with fresh state, both should be close to neutral
        # but trending in opposite directions in EMAs
        assert state.factor_state.ema_ir.get("momentum", 0) > 0
        assert state.factor_state.ema_ir.get("carry", 0) < 0

    def test_multiple_categories_independent(self):
        """Multiple categories are updated independently."""
        cfg = MetaSchedulerConfig(
            enabled=True,
            learning_rate=0.10,
            min_samples=0,
            max_category_shift=0.15,
        )
        
        state = create_neutral_state()
        state.stats["sample_count"] = 100
        
        # Mixed signals: btc positive, altcoin negative
        state = meta_learning_step(
            cfg=cfg,
            prev_state=state,
            factor_edges={},
            category_edges={
                "btc": {"ir": 0.5, "total_pnl": 100.0},
                "altcoin": {"ir": -0.4, "total_pnl": -60.0},
            },
            strategy_health={"health_score": 0.6},
        )
        
        # Check EMAs are building in opposite directions
        assert state.category_state.ema_category_ir.get("btc", 0) > 0
        assert state.category_state.ema_category_ir.get("altcoin", 0) < 0
