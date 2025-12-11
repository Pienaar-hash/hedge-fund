"""
Unit tests for alpha_router.py (v7.8_P2 — Alpha Router / "Overmind")

Tests:
- Config loading and defaults
- State serialization round-trip
- Health-based allocation computation
- Regime penalties (vol/DD)
- Router quality penalties
- Meta-scheduler influence
- EMA smoothing
- Bounds enforcement (floor/ceiling)
"""
import pytest
from datetime import datetime, timezone
from typing import Any, Dict

from execution.alpha_router import (
    AlphaRouterConfig,
    AlphaRouterState,
    load_alpha_router_config,
    create_neutral_state,
    compute_target_allocation,
    is_alpha_router_active,
    get_target_allocation,
    get_allocation_components,
    apply_allocation_to_limits,
    load_allocation_for_sizing,
    NEUTRAL_ALLOCATION,
)


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------


class TestConfigLoading:
    """Tests for AlphaRouterConfig loading."""

    def test_default_config_disabled(self):
        """Default config should have enabled=False."""
        cfg = load_alpha_router_config(None)
        assert cfg.enabled is False
        assert cfg.allocation_floor == 0.20
        assert cfg.allocation_ceiling == 1.00

    def test_load_from_strategy_config(self):
        """Config should load from strategy_config dict."""
        strategy_cfg = {
            "alpha_router": {
                "enabled": True,
                "allocation_floor": 0.25,
                "allocation_ceiling": 0.95,
                "meta_influence": 0.15,
            }
        }
        cfg = load_alpha_router_config(strategy_cfg)
        assert cfg.enabled is True
        assert cfg.allocation_floor == 0.25
        assert cfg.allocation_ceiling == 0.95
        assert cfg.meta_influence == 0.15

    def test_partial_config_uses_defaults(self):
        """Partial config should use defaults for missing fields."""
        strategy_cfg = {
            "alpha_router": {
                "enabled": True,
            }
        }
        cfg = load_alpha_router_config(strategy_cfg)
        assert cfg.enabled is True
        assert cfg.allocation_floor == 0.20  # default
        assert cfg.smoothing_alpha == 0.25  # default

    def test_empty_alpha_router_block(self):
        """Empty alpha_router block should use all defaults."""
        strategy_cfg = {"alpha_router": {}}
        cfg = load_alpha_router_config(strategy_cfg)
        assert cfg.enabled is False
        assert cfg.allocation_floor == 0.20

    def test_default_regime_penalties(self):
        """Default regime penalties should be present."""
        cfg = load_alpha_router_config(None)
        assert cfg.regime_penalties["LOW"] == 1.0
        assert cfg.regime_penalties["NORMAL"] == 1.0
        assert cfg.regime_penalties["HIGH"] == 0.80
        assert cfg.regime_penalties["CRISIS"] == 0.50

    def test_default_dd_penalties(self):
        """Default DD penalties should be present."""
        cfg = load_alpha_router_config(None)
        assert cfg.dd_penalties["NORMAL"] == 1.0
        assert cfg.dd_penalties["RECOVERY"] == 0.85
        assert cfg.dd_penalties["DRAWDOWN"] == 0.65


# ---------------------------------------------------------------------------
# State Tests
# ---------------------------------------------------------------------------


class TestStateSerializtion:
    """Tests for AlphaRouterState serialization."""

    def test_state_to_dict(self):
        """State should serialize to dict correctly."""
        state = AlphaRouterState(
            updated_ts="2025-01-01T00:00:00+00:00",
            target_allocation=0.75,
            raw_components={"health_base": 0.8, "vol_penalty": 0.95},
            smoothed=True,
            prev_allocation=0.70,
        )
        d = state.to_dict()
        assert d["updated_ts"] == "2025-01-01T00:00:00+00:00"
        assert d["target_allocation"] == 0.75
        assert d["smoothed"] is True
        assert d["raw_components"]["health_base"] == 0.8

    def test_state_from_dict(self):
        """State should deserialize from dict correctly."""
        data = {
            "updated_ts": "2025-01-01T00:00:00+00:00",
            "target_allocation": 0.65,
            "raw_components": {"health_base": 0.7},
            "smoothed": False,
            "prev_allocation": 0.60,
        }
        state = AlphaRouterState.from_dict(data)
        assert state.target_allocation == 0.65
        assert state.smoothed is False
        assert state.raw_components["health_base"] == 0.7

    def test_state_roundtrip(self):
        """State should survive serialize/deserialize cycle."""
        original = AlphaRouterState(
            updated_ts="2025-01-01T12:00:00+00:00",
            target_allocation=0.82,
            raw_components={
                "health_base": 0.9,
                "vol_penalty": 1.0,
                "dd_penalty": 0.85,
            },
            smoothed=True,
            prev_allocation=0.80,
        )
        d = original.to_dict()
        restored = AlphaRouterState.from_dict(d)
        assert restored.target_allocation == 0.82
        assert restored.prev_allocation == 0.80
        assert restored.raw_components["dd_penalty"] == 0.85

    def test_create_neutral_state(self):
        """Neutral state should have allocation = 1.0."""
        state = create_neutral_state()
        assert state.target_allocation == NEUTRAL_ALLOCATION
        assert state.prev_allocation == NEUTRAL_ALLOCATION
        assert "health_base" in state.raw_components


# ---------------------------------------------------------------------------
# Health-Based Allocation Tests
# ---------------------------------------------------------------------------


class TestHealthAllocation:
    """Tests for health-based allocation computation."""

    def test_strong_health_full_allocation(self):
        """Strong health score should yield full allocation."""
        cfg = AlphaRouterConfig(
            enabled=True,
            health_thresholds={"strong": 0.75, "weak": 0.45},
        )
        health = {"health_score": 0.85}
        state = compute_target_allocation(
            health=health,
            meta_state=None,
            router_quality=0.8,
            vol_regime="NORMAL",
            dd_state="NORMAL",
            cfg=cfg,
        )
        assert state.target_allocation >= 0.95

    def test_weak_health_reduced_allocation(self):
        """Weak health score should yield reduced allocation."""
        cfg = AlphaRouterConfig(
            enabled=True,
            health_thresholds={"strong": 0.75, "weak": 0.45},
            smoothing_alpha=0.0,  # No smoothing
        )
        health = {"health_score": 0.30}  # Below weak threshold
        state = compute_target_allocation(
            health=health,
            meta_state=None,
            router_quality=0.8,
            vol_regime="NORMAL",
            dd_state="NORMAL",
            cfg=cfg,
        )
        # Health below weak → base = 0.30
        assert state.raw_components["health_base"] == 0.30

    def test_mid_health_interpolated(self):
        """Mid-range health should interpolate linearly."""
        cfg = AlphaRouterConfig(
            enabled=True,
            health_thresholds={"strong": 0.80, "weak": 0.40},
            smoothing_alpha=0.0,
        )
        health = {"health_score": 0.60}  # Midpoint
        state = compute_target_allocation(
            health=health,
            meta_state=None,
            router_quality=0.8,
            vol_regime="NORMAL",
            dd_state="NORMAL",
            cfg=cfg,
        )
        # Midpoint between 0.30 and 1.0 should be ~0.65
        assert 0.60 <= state.raw_components["health_base"] <= 0.70


# ---------------------------------------------------------------------------
# Regime Penalty Tests
# ---------------------------------------------------------------------------


class TestRegimePenalties:
    """Tests for volatility and DD regime penalties."""

    def test_high_vol_regime_penalty(self):
        """HIGH vol regime should apply penalty."""
        cfg = AlphaRouterConfig(
            enabled=True,
            regime_penalties={"LOW": 1.0, "NORMAL": 1.0, "HIGH": 0.80, "CRISIS": 0.50},
            smoothing_alpha=0.0,
        )
        health = {"health_score": 0.80}
        state = compute_target_allocation(
            health=health,
            meta_state=None,
            router_quality=0.8,
            vol_regime="HIGH",
            dd_state="NORMAL",
            cfg=cfg,
        )
        assert state.raw_components["vol_penalty"] == 0.80

    def test_crisis_vol_regime_severe_penalty(self):
        """CRISIS vol regime should apply severe penalty."""
        cfg = AlphaRouterConfig(
            enabled=True,
            regime_penalties={"LOW": 1.0, "NORMAL": 1.0, "HIGH": 0.80, "CRISIS": 0.50},
            smoothing_alpha=0.0,
        )
        health = {"health_score": 0.80}
        state = compute_target_allocation(
            health=health,
            meta_state=None,
            router_quality=0.8,
            vol_regime="CRISIS",
            dd_state="NORMAL",
            cfg=cfg,
        )
        assert state.raw_components["vol_penalty"] == 0.50

    def test_dd_state_penalty(self):
        """DRAWDOWN state should apply penalty."""
        cfg = AlphaRouterConfig(
            enabled=True,
            dd_penalties={"NORMAL": 1.0, "RECOVERY": 0.85, "DRAWDOWN": 0.65},
            smoothing_alpha=0.0,
        )
        health = {"health_score": 0.80}
        state = compute_target_allocation(
            health=health,
            meta_state=None,
            router_quality=0.8,
            vol_regime="NORMAL",
            dd_state="DRAWDOWN",
            cfg=cfg,
        )
        assert state.raw_components["dd_penalty"] == 0.65

    def test_combined_regime_penalties(self):
        """Combined HIGH vol + DRAWDOWN should stack penalties."""
        cfg = AlphaRouterConfig(
            enabled=True,
            regime_penalties={"HIGH": 0.80},
            dd_penalties={"DRAWDOWN": 0.65},
            smoothing_alpha=0.0,
        )
        health = {"health_score": 0.80}  # Strong health → base ~1.0
        state = compute_target_allocation(
            health=health,
            meta_state=None,
            router_quality=0.8,
            vol_regime="HIGH",
            dd_state="DRAWDOWN",
            cfg=cfg,
        )
        # 1.0 * 0.80 * 0.65 = 0.52
        assert state.raw_components["vol_penalty"] == 0.80
        assert state.raw_components["dd_penalty"] == 0.65


# ---------------------------------------------------------------------------
# Router Quality Penalty Tests
# ---------------------------------------------------------------------------


class TestRouterQualityPenalty:
    """Tests for router execution quality penalties."""

    def test_good_router_quality_no_penalty(self):
        """Good router quality should have no penalty."""
        cfg = AlphaRouterConfig(
            enabled=True,
            router_quality_thresholds={"good": 0.65, "moderate": 0.45},
            smoothing_alpha=0.0,
        )
        health = {"health_score": 0.80}
        state = compute_target_allocation(
            health=health,
            meta_state=None,
            router_quality=0.75,  # Above good threshold
            vol_regime="NORMAL",
            dd_state="NORMAL",
            cfg=cfg,
        )
        assert state.raw_components["router_penalty"] == 1.0

    def test_moderate_router_quality_small_penalty(self):
        """Moderate router quality should have small penalty."""
        cfg = AlphaRouterConfig(
            enabled=True,
            router_quality_thresholds={"good": 0.65, "moderate": 0.45},
            smoothing_alpha=0.0,
        )
        health = {"health_score": 0.80}
        state = compute_target_allocation(
            health=health,
            meta_state=None,
            router_quality=0.55,  # Between moderate and good
            vol_regime="NORMAL",
            dd_state="NORMAL",
            cfg=cfg,
        )
        assert state.raw_components["router_penalty"] == 0.90

    def test_poor_router_quality_larger_penalty(self):
        """Poor router quality should have larger penalty."""
        cfg = AlphaRouterConfig(
            enabled=True,
            router_quality_thresholds={"good": 0.65, "moderate": 0.45},
            smoothing_alpha=0.0,
        )
        health = {"health_score": 0.80}
        state = compute_target_allocation(
            health=health,
            meta_state=None,
            router_quality=0.30,  # Below moderate
            vol_regime="NORMAL",
            dd_state="NORMAL",
            cfg=cfg,
        )
        assert state.raw_components["router_penalty"] == 0.70


# ---------------------------------------------------------------------------
# EMA Smoothing Tests
# ---------------------------------------------------------------------------


class TestEMASmoothing:
    """Tests for allocation EMA smoothing."""

    def test_smoothing_applied_with_prev_state(self):
        """Smoothing should be applied when prev_state exists."""
        cfg = AlphaRouterConfig(
            enabled=True,
            smoothing_alpha=0.5,  # 50% smoothing
        )
        prev_state = AlphaRouterState(
            target_allocation=0.80,
            prev_allocation=0.75,
        )
        health = {"health_score": 0.40}  # Lower health → would give ~0.30 base
        state = compute_target_allocation(
            health=health,
            meta_state=None,
            router_quality=0.8,
            vol_regime="NORMAL",
            dd_state="NORMAL",
            cfg=cfg,
            prev_state=prev_state,
        )
        # With smoothing, should be between prev (0.80) and new
        assert state.smoothed is True
        assert state.target_allocation < 0.80
        assert state.target_allocation > cfg.allocation_floor

    def test_no_smoothing_without_prev_state(self):
        """No smoothing without prev_state."""
        cfg = AlphaRouterConfig(
            enabled=True,
            smoothing_alpha=0.5,
        )
        health = {"health_score": 0.80}
        state = compute_target_allocation(
            health=health,
            meta_state=None,
            router_quality=0.8,
            vol_regime="NORMAL",
            dd_state="NORMAL",
            cfg=cfg,
            prev_state=None,
        )
        assert state.smoothed is False

    def test_zero_alpha_no_smoothing(self):
        """smoothing_alpha=0 should not apply smoothing."""
        cfg = AlphaRouterConfig(
            enabled=True,
            smoothing_alpha=0.0,
        )
        prev_state = AlphaRouterState(
            target_allocation=0.60,
            prev_allocation=0.55,
        )
        health = {"health_score": 0.90}
        state = compute_target_allocation(
            health=health,
            meta_state=None,
            router_quality=0.8,
            vol_regime="NORMAL",
            dd_state="NORMAL",
            cfg=cfg,
            prev_state=prev_state,
        )
        # With alpha=0, smoothing is effectively disabled
        assert state.smoothed is False


# ---------------------------------------------------------------------------
# Bounds Tests
# ---------------------------------------------------------------------------


class TestBounds:
    """Tests for allocation floor/ceiling bounds."""

    def test_floor_enforced(self):
        """Allocation should not go below floor."""
        cfg = AlphaRouterConfig(
            enabled=True,
            allocation_floor=0.20,
            allocation_ceiling=1.00,
            smoothing_alpha=0.0,
        )
        health = {"health_score": 0.10}  # Very low health
        state = compute_target_allocation(
            health=health,
            meta_state=None,
            router_quality=0.2,  # Poor router
            vol_regime="CRISIS",  # 0.50 penalty
            dd_state="DRAWDOWN",  # 0.65 penalty
            cfg=cfg,
        )
        assert state.target_allocation >= cfg.allocation_floor

    def test_ceiling_enforced(self):
        """Allocation should not exceed ceiling."""
        cfg = AlphaRouterConfig(
            enabled=True,
            allocation_floor=0.20,
            allocation_ceiling=0.90,  # Custom ceiling
            smoothing_alpha=0.0,
        )
        health = {"health_score": 0.95}  # Excellent health
        state = compute_target_allocation(
            health=health,
            meta_state=None,
            router_quality=1.0,
            vol_regime="LOW",  # 1.0 (no penalty)
            dd_state="NORMAL",
            cfg=cfg,
        )
        assert state.target_allocation <= cfg.allocation_ceiling


# ---------------------------------------------------------------------------
# Activation Check Tests
# ---------------------------------------------------------------------------


class TestActivationCheck:
    """Tests for is_alpha_router_active."""

    def test_active_when_enabled(self):
        """Should be active when enabled=True."""
        cfg = AlphaRouterConfig(enabled=True)
        assert is_alpha_router_active(cfg) is True

    def test_inactive_when_disabled(self):
        """Should be inactive when enabled=False."""
        cfg = AlphaRouterConfig(enabled=False)
        assert is_alpha_router_active(cfg) is False


# ---------------------------------------------------------------------------
# Limits Application Tests
# ---------------------------------------------------------------------------


class TestApplyAllocationToLimits:
    """Tests for apply_allocation_to_limits."""

    def test_full_allocation_no_change(self):
        """Full allocation (1.0) should not change limits."""
        exp, trade, sym = apply_allocation_to_limits(
            allocation=1.0,
            max_total_exposure_pct=0.50,
            max_per_trade_nav_pct=0.06,
            symbol_cap_pct=0.10,
        )
        assert exp == 0.50
        assert trade == 0.06
        assert sym == 0.10

    def test_half_allocation_halves_limits(self):
        """50% allocation should halve limits."""
        exp, trade, sym = apply_allocation_to_limits(
            allocation=0.50,
            max_total_exposure_pct=0.50,
            max_per_trade_nav_pct=0.06,
            symbol_cap_pct=0.10,
        )
        assert exp == 0.25
        assert trade == 0.03
        assert sym == 0.05

    def test_min_allocation_enforced(self):
        """Very low allocation should be clamped to minimum."""
        exp, trade, sym = apply_allocation_to_limits(
            allocation=0.01,  # Very low
            max_total_exposure_pct=0.50,
            max_per_trade_nav_pct=0.06,
            symbol_cap_pct=0.10,
        )
        # MIN_ALLOCATION is 0.05
        assert exp == 0.50 * 0.05
        assert trade == 0.06 * 0.05
        assert sym == 0.10 * 0.05


# ---------------------------------------------------------------------------
# Views API Tests
# ---------------------------------------------------------------------------


class TestViewsAPI:
    """Tests for get_target_allocation and get_allocation_components."""

    def test_get_target_allocation_from_state(self):
        """Should return allocation from provided state."""
        state = AlphaRouterState(target_allocation=0.72)
        alloc = get_target_allocation(state=state)
        assert alloc == 0.72

    def test_get_target_allocation_none_returns_neutral(self):
        """Should return NEUTRAL_ALLOCATION when state is None."""
        alloc = get_target_allocation(state=None, path="/nonexistent/path.json")
        assert alloc == NEUTRAL_ALLOCATION

    def test_get_allocation_components_from_state(self):
        """Should return components from provided state."""
        state = AlphaRouterState(
            raw_components={"health_base": 0.9, "vol_penalty": 0.8}
        )
        components = get_allocation_components(state=state)
        assert components["health_base"] == 0.9
        assert components["vol_penalty"] == 0.8

    def test_get_allocation_components_none_returns_empty(self):
        """Should return empty dict when state is None."""
        components = get_allocation_components(state=None, path="/nonexistent/path.json")
        assert components == {}


# ---------------------------------------------------------------------------
# Integration: load_allocation_for_sizing
# ---------------------------------------------------------------------------


class TestLoadAllocationForSizing:
    """Tests for load_allocation_for_sizing helper."""

    def test_returns_neutral_when_disabled(self):
        """Should return NEUTRAL_ALLOCATION when disabled."""
        cfg = AlphaRouterConfig(enabled=False)
        alloc = load_allocation_for_sizing(cfg=cfg)
        assert alloc == NEUTRAL_ALLOCATION

    def test_returns_neutral_when_cfg_none_defaults_disabled(self):
        """Should return NEUTRAL_ALLOCATION when cfg=None (defaults to disabled)."""
        alloc = load_allocation_for_sizing(cfg=None, path="/nonexistent/path.json")
        assert alloc == NEUTRAL_ALLOCATION
