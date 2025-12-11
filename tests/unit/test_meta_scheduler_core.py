"""
Unit tests for Meta-Scheduler core functionality (v7.8_P1).

Tests:
- Config loading
- State serialization/deserialization
- EMA updates
- Overlay computations
- Bounds enforcement
"""

import pytest
from typing import Dict, Any

from execution.meta_scheduler import (
    MetaSchedulerConfig,
    MetaSchedulerState,
    FactorMetaState,
    ConvictionMetaState,
    CategoryMetaState,
    load_meta_scheduler_config,
    create_neutral_state,
    meta_learning_step,
    NEUTRAL_MULTIPLIER,
)


# ---------------------------------------------------------------------------
# Config Loading Tests
# ---------------------------------------------------------------------------


class TestMetaSchedulerConfig:
    """Tests for MetaSchedulerConfig loading."""

    def test_default_config(self):
        """Default config has expected values."""
        cfg = MetaSchedulerConfig()
        
        assert cfg.enabled is False
        assert cfg.learning_rate == 0.05
        assert cfg.min_samples == 50
        assert cfg.max_factor_shift == 0.10
        assert cfg.max_conviction_shift == 0.15
        assert cfg.max_category_shift == 0.15
        assert cfg.decay == 0.90

    def test_load_from_strategy_config(self):
        """Config loads from strategy_config dict."""
        strategy_cfg = {
            "meta_scheduler": {
                "enabled": True,
                "learning_rate": 0.10,
                "min_samples": 100,
                "max_factor_shift": 0.20,
            }
        }
        
        cfg = load_meta_scheduler_config(strategy_cfg)
        
        assert cfg.enabled is True
        assert cfg.learning_rate == 0.10
        assert cfg.min_samples == 100
        assert cfg.max_factor_shift == 0.20
        # Defaults for unspecified
        assert cfg.max_conviction_shift == 0.15

    def test_load_from_none(self):
        """Config from None returns defaults."""
        cfg = load_meta_scheduler_config(None)
        
        assert cfg.enabled is False
        assert cfg.learning_rate == 0.05

    def test_load_from_empty(self):
        """Config from empty dict returns defaults."""
        cfg = load_meta_scheduler_config({})
        
        assert cfg.enabled is False


# ---------------------------------------------------------------------------
# State Serialization Tests
# ---------------------------------------------------------------------------


class TestMetaSchedulerState:
    """Tests for state serialization."""

    def test_neutral_state(self):
        """create_neutral_state returns valid neutral state."""
        state = create_neutral_state()
        
        assert state.factor_state.meta_weights == {}
        assert state.conviction_state.global_strength == NEUTRAL_MULTIPLIER
        assert state.category_state.category_overlays == {}
        assert state.stats.get("sample_count") == 0

    def test_state_to_dict(self):
        """State converts to dict correctly."""
        state = MetaSchedulerState(
            updated_ts="2025-01-01T00:00:00Z",
            factor_state=FactorMetaState(
                meta_weights={"momentum": 1.05},
                ema_ir={"momentum": 0.3},
                ema_pnl={"momentum": 100.0},
            ),
            conviction_state=ConvictionMetaState(
                global_strength=1.10,
                ema_health=0.75,
            ),
            category_state=CategoryMetaState(
                category_overlays={"btc": 1.05},
            ),
            stats={"sample_count": 100},
        )
        
        data = state.to_dict()
        
        assert data["updated_ts"] == "2025-01-01T00:00:00Z"
        assert data["factor_state"]["meta_weights"]["momentum"] == 1.05
        assert data["conviction_state"]["global_strength"] == 1.10
        assert data["category_state"]["category_overlays"]["btc"] == 1.05

    def test_state_from_dict(self):
        """State reconstructs from dict correctly."""
        data = {
            "updated_ts": "2025-01-01T00:00:00Z",
            "factor_state": {
                "meta_weights": {"momentum": 1.05},
                "ema_ir": {"momentum": 0.3},
                "ema_pnl": {},
            },
            "conviction_state": {
                "global_strength": 1.10,
                "ema_health": 0.75,
            },
            "category_state": {
                "category_overlays": {"btc": 1.05},
            },
            "stats": {"sample_count": 100},
        }
        
        state = MetaSchedulerState.from_dict(data)
        
        assert state.factor_state.meta_weights["momentum"] == 1.05
        assert state.conviction_state.global_strength == 1.10
        assert state.category_state.category_overlays["btc"] == 1.05

    def test_roundtrip(self):
        """State survives to_dict → from_dict roundtrip."""
        original = MetaSchedulerState(
            updated_ts="2025-01-01T00:00:00Z",
            factor_state=FactorMetaState(
                meta_weights={"momentum": 1.08, "carry": 0.95},
                ema_ir={"momentum": 0.4, "carry": -0.1},
                ema_pnl={"momentum": 50.0, "carry": -20.0},
            ),
            conviction_state=ConvictionMetaState(
                global_strength=1.12,
                ema_health=0.68,
            ),
            category_state=CategoryMetaState(
                category_overlays={"btc": 1.10, "eth": 0.90},
                ema_category_ir={"btc": 0.5, "eth": -0.2},
                ema_category_pnl={"btc": 100.0, "eth": -50.0},
            ),
            stats={"sample_count": 75, "learning_active": True},
        )
        
        data = original.to_dict()
        reconstructed = MetaSchedulerState.from_dict(data)
        
        assert reconstructed.factor_state.meta_weights == original.factor_state.meta_weights
        assert reconstructed.conviction_state.global_strength == original.conviction_state.global_strength
        assert reconstructed.category_state.category_overlays == original.category_state.category_overlays


# ---------------------------------------------------------------------------
# Learning Step Tests: Factor Overlays
# ---------------------------------------------------------------------------


class TestFactorOverlays:
    """Tests for factor overlay computation."""

    def test_positive_ir_pnl_increases_overlay(self):
        """Factors with positive IR and PnL get increased overlay."""
        cfg = MetaSchedulerConfig(
            enabled=True,
            learning_rate=0.10,
            min_samples=0,  # Immediate learning
            max_factor_shift=0.20,
            ir_threshold=0.2,
        )
        
        prev_state = create_neutral_state()
        prev_state.factor_state.meta_weights = {"momentum": 1.0}
        # Initialize EMAs high enough to pass threshold after update
        prev_state.factor_state.ema_ir = {"momentum": 0.5}
        prev_state.factor_state.ema_pnl = {"momentum": 100.0}
        prev_state.stats["sample_count"] = 100
        
        factor_edges = {
            "momentum": {"edge_score": 0.8, "ir": 0.5, "pnl_contrib": 100.0},
        }
        
        new_state = meta_learning_step(
            cfg=cfg,
            prev_state=prev_state,
            factor_edges=factor_edges,
            category_edges={},
            strategy_health={"health_score": 0.7},
        )
        
        # Should increase
        assert new_state.factor_state.meta_weights["momentum"] > 1.0

    def test_negative_ir_decreases_overlay(self):
        """Factors with negative IR get decreased overlay."""
        cfg = MetaSchedulerConfig(
            enabled=True,
            learning_rate=0.10,
            min_samples=0,
            max_factor_shift=0.20,
            ir_threshold=0.2,
        )
        
        prev_state = create_neutral_state()
        prev_state.factor_state.meta_weights = {"momentum": 1.0}
        prev_state.factor_state.ema_ir = {"momentum": -0.5}
        prev_state.factor_state.ema_pnl = {"momentum": -50.0}
        prev_state.stats["sample_count"] = 100
        
        factor_edges = {
            "momentum": {"edge_score": -0.5, "ir": -0.5, "pnl_contrib": -50.0},
        }
        
        new_state = meta_learning_step(
            cfg=cfg,
            prev_state=prev_state,
            factor_edges=factor_edges,
            category_edges={},
            strategy_health={"health_score": 0.5},
        )
        
        # Should decrease
        assert new_state.factor_state.meta_weights["momentum"] < 1.0

    def test_overlay_respects_max_bound(self):
        """Overlay cannot exceed 1 + max_factor_shift."""
        cfg = MetaSchedulerConfig(
            enabled=True,
            learning_rate=0.50,  # Very aggressive
            min_samples=0,
            max_factor_shift=0.10,  # Max 1.10
        )
        
        prev_state = create_neutral_state()
        prev_state.factor_state.meta_weights = {"momentum": 1.08}
        prev_state.factor_state.ema_ir = {"momentum": 0.8}
        prev_state.factor_state.ema_pnl = {"momentum": 200.0}
        prev_state.stats["sample_count"] = 100
        
        factor_edges = {
            "momentum": {"ir": 1.0, "pnl_contrib": 500.0},
        }
        
        new_state = meta_learning_step(
            cfg=cfg,
            prev_state=prev_state,
            factor_edges=factor_edges,
            category_edges={},
            strategy_health={"health_score": 0.9},
        )
        
        # Should be capped at 1.10
        assert new_state.factor_state.meta_weights["momentum"] <= 1.10

    def test_overlay_respects_min_bound(self):
        """Overlay cannot go below 1 - max_factor_shift."""
        cfg = MetaSchedulerConfig(
            enabled=True,
            learning_rate=0.50,
            min_samples=0,
            max_factor_shift=0.10,  # Min 0.90
        )
        
        prev_state = create_neutral_state()
        prev_state.factor_state.meta_weights = {"momentum": 0.92}
        prev_state.factor_state.ema_ir = {"momentum": -0.8}
        prev_state.factor_state.ema_pnl = {"momentum": -200.0}
        prev_state.stats["sample_count"] = 100
        
        factor_edges = {
            "momentum": {"ir": -1.0, "pnl_contrib": -500.0},
        }
        
        new_state = meta_learning_step(
            cfg=cfg,
            prev_state=prev_state,
            factor_edges=factor_edges,
            category_edges={},
            strategy_health={"health_score": 0.3},
        )
        
        # Should be capped at 0.90
        assert new_state.factor_state.meta_weights["momentum"] >= 0.90


# ---------------------------------------------------------------------------
# Learning Step Tests: Conviction Strength
# ---------------------------------------------------------------------------


class TestConvictionStrength:
    """Tests for conviction meta strength computation."""

    def test_high_health_increases_strength(self):
        """High health score increases conviction strength."""
        cfg = MetaSchedulerConfig(
            enabled=True,
            learning_rate=0.10,
            min_samples=0,
            max_conviction_shift=0.15,
        )
        
        prev_state = create_neutral_state()
        prev_state.conviction_state.global_strength = 1.0
        prev_state.conviction_state.ema_health = 0.7
        prev_state.stats["sample_count"] = 100
        
        factor_edges = {
            "momentum": {"edge_score": 0.5},
        }
        
        new_state = meta_learning_step(
            cfg=cfg,
            prev_state=prev_state,
            factor_edges=factor_edges,
            category_edges={},
            strategy_health={"health_score": 0.85},
        )
        
        # Should increase
        assert new_state.conviction_state.global_strength > 1.0

    def test_low_health_decreases_strength(self):
        """Low health score decreases conviction strength."""
        cfg = MetaSchedulerConfig(
            enabled=True,
            learning_rate=0.10,
            min_samples=0,
            max_conviction_shift=0.15,
        )
        
        prev_state = create_neutral_state()
        prev_state.conviction_state.global_strength = 1.0
        prev_state.conviction_state.ema_health = 0.4
        prev_state.stats["sample_count"] = 100
        
        new_state = meta_learning_step(
            cfg=cfg,
            prev_state=prev_state,
            factor_edges={},
            category_edges={},
            strategy_health={"health_score": 0.30},
        )
        
        # Should decrease
        assert new_state.conviction_state.global_strength < 1.0

    def test_conviction_respects_bounds(self):
        """Conviction strength respects max_conviction_shift bounds."""
        cfg = MetaSchedulerConfig(
            enabled=True,
            learning_rate=0.50,
            min_samples=0,
            max_conviction_shift=0.10,  # [0.90, 1.10]
        )
        
        prev_state = create_neutral_state()
        prev_state.conviction_state.global_strength = 1.08
        prev_state.conviction_state.ema_health = 0.9
        prev_state.stats["sample_count"] = 100
        
        new_state = meta_learning_step(
            cfg=cfg,
            prev_state=prev_state,
            factor_edges={"momentum": {"edge_score": 0.8}},
            category_edges={},
            strategy_health={"health_score": 0.95},
        )
        
        assert 0.90 <= new_state.conviction_state.global_strength <= 1.10


# ---------------------------------------------------------------------------
# Learning Step Tests: Category Overlays
# ---------------------------------------------------------------------------


class TestCategoryOverlays:
    """Tests for category overlay computation."""

    def test_strong_category_gets_boost(self):
        """Categories with strong IR get increased overlay."""
        cfg = MetaSchedulerConfig(
            enabled=True,
            learning_rate=0.10,
            min_samples=0,
            max_category_shift=0.15,
            ir_threshold=0.2,
        )
        
        prev_state = create_neutral_state()
        prev_state.category_state.category_overlays = {"btc": 1.0}
        prev_state.category_state.ema_category_ir = {"btc": 0.5}
        prev_state.category_state.ema_category_pnl = {"btc": 100.0}
        prev_state.stats["sample_count"] = 100
        
        category_edges = {
            "btc": {"ir": 0.6, "total_pnl": 150.0, "edge_score": 0.6},
        }
        
        new_state = meta_learning_step(
            cfg=cfg,
            prev_state=prev_state,
            factor_edges={},
            category_edges=category_edges,
            strategy_health={"health_score": 0.7},
        )
        
        assert new_state.category_state.category_overlays["btc"] > 1.0

    def test_weak_category_gets_cut(self):
        """Categories with weak IR get decreased overlay."""
        cfg = MetaSchedulerConfig(
            enabled=True,
            learning_rate=0.10,
            min_samples=0,
            max_category_shift=0.15,
            ir_threshold=0.2,
        )
        
        prev_state = create_neutral_state()
        prev_state.category_state.category_overlays = {"altcoin": 1.0}
        prev_state.category_state.ema_category_ir = {"altcoin": -0.4}
        prev_state.category_state.ema_category_pnl = {"altcoin": -50.0}
        prev_state.stats["sample_count"] = 100
        
        category_edges = {
            "altcoin": {"ir": -0.5, "total_pnl": -80.0, "edge_score": -0.5},
        }
        
        new_state = meta_learning_step(
            cfg=cfg,
            prev_state=prev_state,
            factor_edges={},
            category_edges=category_edges,
            strategy_health={"health_score": 0.5},
        )
        
        assert new_state.category_state.category_overlays["altcoin"] < 1.0

    def test_category_overlay_respects_bounds(self):
        """Category overlays respect max_category_shift bounds."""
        cfg = MetaSchedulerConfig(
            enabled=True,
            learning_rate=0.50,
            min_samples=0,
            max_category_shift=0.10,  # [0.90, 1.10]
        )
        
        prev_state = create_neutral_state()
        prev_state.category_state.category_overlays = {"btc": 1.08}
        prev_state.category_state.ema_category_ir = {"btc": 0.9}
        prev_state.category_state.ema_category_pnl = {"btc": 500.0}
        prev_state.stats["sample_count"] = 100
        
        category_edges = {
            "btc": {"ir": 1.0, "total_pnl": 1000.0},
        }
        
        new_state = meta_learning_step(
            cfg=cfg,
            prev_state=prev_state,
            factor_edges={},
            category_edges=category_edges,
            strategy_health={"health_score": 0.9},
        )
        
        assert 0.90 <= new_state.category_state.category_overlays["btc"] <= 1.10


# ---------------------------------------------------------------------------
# Min Samples Tests
# ---------------------------------------------------------------------------


class TestMinSamples:
    """Tests for min_samples gating."""

    def test_no_learning_before_min_samples(self):
        """Overlays stay neutral before min_samples reached."""
        cfg = MetaSchedulerConfig(
            enabled=True,
            learning_rate=0.50,  # Aggressive
            min_samples=100,
            max_factor_shift=0.20,
        )
        
        prev_state = create_neutral_state()
        prev_state.stats["sample_count"] = 50  # Below min_samples
        
        factor_edges = {
            "momentum": {"ir": 1.0, "pnl_contrib": 1000.0},
        }
        
        new_state = meta_learning_step(
            cfg=cfg,
            prev_state=prev_state,
            factor_edges=factor_edges,
            category_edges={},
            strategy_health={"health_score": 0.9},
        )
        
        # Should not have changed overlays
        assert new_state.factor_state.meta_weights.get("momentum", 1.0) == 1.0
        assert new_state.stats.get("learning_active") is False

    def test_learning_after_min_samples(self):
        """Learning kicks in after min_samples reached."""
        cfg = MetaSchedulerConfig(
            enabled=True,
            learning_rate=0.10,
            min_samples=10,
            max_factor_shift=0.20,
        )
        
        prev_state = create_neutral_state()
        prev_state.factor_state.meta_weights = {"momentum": 1.0}
        prev_state.factor_state.ema_ir = {"momentum": 0.5}
        prev_state.factor_state.ema_pnl = {"momentum": 100.0}
        prev_state.stats["sample_count"] = 50  # Above min_samples
        
        factor_edges = {
            "momentum": {"ir": 0.6, "pnl_contrib": 150.0},
        }
        
        new_state = meta_learning_step(
            cfg=cfg,
            prev_state=prev_state,
            factor_edges=factor_edges,
            category_edges={},
            strategy_health={"health_score": 0.8},
        )
        
        # Should have learning active and changed overlay
        assert new_state.stats.get("learning_active") is True


# ---------------------------------------------------------------------------
# Disabled Mode Tests
# ---------------------------------------------------------------------------


class TestDisabledMode:
    """Tests for disabled meta-scheduler."""

    def test_disabled_returns_prev_state(self):
        """Disabled meta-scheduler returns previous state unchanged."""
        cfg = MetaSchedulerConfig(enabled=False)
        
        prev_state = MetaSchedulerState(
            updated_ts="2025-01-01T00:00:00Z",
            factor_state=FactorMetaState(meta_weights={"momentum": 1.05}),
            conviction_state=ConvictionMetaState(global_strength=1.10),
            category_state=CategoryMetaState(),
            stats={"sample_count": 100},
        )
        
        new_state = meta_learning_step(
            cfg=cfg,
            prev_state=prev_state,
            factor_edges={"momentum": {"ir": 1.0}},
            category_edges={},
            strategy_health={"health_score": 0.9},
        )
        
        # Should return previous state unchanged
        assert new_state.factor_state.meta_weights == prev_state.factor_state.meta_weights
        assert new_state.conviction_state.global_strength == prev_state.conviction_state.global_strength

    def test_disabled_with_none_state(self):
        """Disabled meta-scheduler with None state returns neutral."""
        cfg = MetaSchedulerConfig(enabled=False)
        
        new_state = meta_learning_step(
            cfg=cfg,
            prev_state=None,
            factor_edges={},
            category_edges={},
            strategy_health=None,
        )
        
        assert new_state.conviction_state.global_strength == NEUTRAL_MULTIPLIER
