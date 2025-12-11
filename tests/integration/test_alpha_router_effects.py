"""
Integration tests for alpha_router.py (v7.8_P2)

Tests:
- State file round-trip persistence
- Schema matches manifest
- Integration with strategy health inputs
- End-to-end allocation flow
"""
import json
import pytest
import tempfile
from pathlib import Path
from typing import Any, Dict

from execution.alpha_router import (
    AlphaRouterConfig,
    AlphaRouterState,
    create_neutral_state,
    compute_target_allocation,
    write_alpha_router_state,
    load_alpha_router_state,
    run_alpha_router_step,
    NEUTRAL_ALLOCATION,
)


# ---------------------------------------------------------------------------
# State File Persistence Tests
# ---------------------------------------------------------------------------


class TestStatePersistence:
    """Tests for state file round-trip persistence."""

    def test_write_and_load_state(self, tmp_path: Path):
        """State should persist through write/load cycle."""
        state_path = tmp_path / "alpha_router_state.json"
        
        original = AlphaRouterState(
            updated_ts="2025-01-15T10:30:00+00:00",
            target_allocation=0.72,
            raw_components={
                "health_score": 0.65,
                "health_base": 0.75,
                "vol_regime": "HIGH",
                "vol_penalty": 0.80,
                "dd_state": "NORMAL",
                "dd_penalty": 1.0,
                "router_quality": 0.70,
                "router_penalty": 1.0,
                "meta_adjustment": 0.02,
                "raw_allocation": 0.72,
            },
            smoothed=True,
            prev_allocation=0.68,
        )
        
        write_alpha_router_state(original, state_path)
        loaded = load_alpha_router_state(state_path)
        
        assert loaded is not None
        assert loaded.target_allocation == original.target_allocation
        assert loaded.smoothed == original.smoothed
        assert loaded.prev_allocation == original.prev_allocation
        assert loaded.raw_components["health_score"] == 0.65

    def test_load_nonexistent_returns_none(self, tmp_path: Path):
        """Loading from nonexistent file should return None."""
        state_path = tmp_path / "nonexistent.json"
        loaded = load_alpha_router_state(state_path)
        assert loaded is None

    def test_atomic_write_creates_directories(self, tmp_path: Path):
        """Writer should create parent directories if needed."""
        state_path = tmp_path / "nested" / "dir" / "alpha_router_state.json"
        state = create_neutral_state()
        
        write_alpha_router_state(state, state_path)
        
        assert state_path.exists()
        loaded = load_alpha_router_state(state_path)
        assert loaded is not None


# ---------------------------------------------------------------------------
# Schema Tests
# ---------------------------------------------------------------------------


class TestStateSchema:
    """Tests for state schema compliance."""

    def test_state_has_required_fields(self, tmp_path: Path):
        """State should have all required fields per manifest."""
        state_path = tmp_path / "alpha_router_state.json"
        
        cfg = AlphaRouterConfig(enabled=True, smoothing_alpha=0.0)
        health = {"health_score": 0.70}
        
        state = compute_target_allocation(
            health=health,
            meta_state=None,
            router_quality=0.75,
            vol_regime="NORMAL",
            dd_state="NORMAL",
            cfg=cfg,
        )
        
        write_alpha_router_state(state, state_path)
        
        # Load raw JSON to check schema
        with open(state_path, "r") as f:
            data = json.load(f)
        
        # Required fields per manifest
        assert "updated_ts" in data
        assert "target_allocation" in data
        assert "raw_components" in data
        assert "smoothed" in data
        assert "prev_allocation" in data
        
        # target_allocation should be float in [0, 1]
        assert isinstance(data["target_allocation"], (int, float))
        assert 0 <= data["target_allocation"] <= 1
        
        # raw_components should have allocation breakdown
        components = data["raw_components"]
        assert "health_base" in components
        assert "vol_penalty" in components
        assert "dd_penalty" in components
        assert "router_penalty" in components

    def test_state_json_valid(self, tmp_path: Path):
        """State file should be valid JSON."""
        state_path = tmp_path / "alpha_router_state.json"
        state = create_neutral_state()
        
        write_alpha_router_state(state, state_path)
        
        # Should not raise
        with open(state_path, "r") as f:
            data = json.load(f)
        
        assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# End-to-End Flow Tests
# ---------------------------------------------------------------------------


class TestE2EFlow:
    """Tests for end-to-end allocation flow."""

    def test_strong_health_normal_regime_high_allocation(self, tmp_path: Path):
        """Strong health + normal regime should yield high allocation."""
        state_path = tmp_path / "alpha_router_state.json"
        
        strategy_cfg = {
            "alpha_router": {
                "enabled": True,
                "allocation_floor": 0.20,
                "allocation_ceiling": 1.00,
            }
        }
        
        state = run_alpha_router_step(
            health={"health_score": 0.85},
            meta_state=None,
            router_quality=0.80,
            vol_regime="NORMAL",
            dd_state="NORMAL",
            strategy_cfg=strategy_cfg,
            state_path=state_path,
        )
        
        # Strong health + no penalties → should be near 1.0
        assert state.target_allocation >= 0.90
        assert state_path.exists()

    def test_weak_health_crisis_regime_low_allocation(self, tmp_path: Path):
        """Weak health + crisis regime should yield low allocation."""
        state_path = tmp_path / "alpha_router_state.json"
        
        strategy_cfg = {
            "alpha_router": {
                "enabled": True,
                "allocation_floor": 0.15,
                "allocation_ceiling": 1.00,
            }
        }
        
        state = run_alpha_router_step(
            health={"health_score": 0.35},  # Below weak threshold
            meta_state=None,
            router_quality=0.40,  # Poor
            vol_regime="CRISIS",  # 0.50 penalty
            dd_state="DRAWDOWN",  # 0.65 penalty
            strategy_cfg=strategy_cfg,
            state_path=state_path,
        )
        
        # Should be clamped to floor
        assert state.target_allocation == 0.15
        assert state_path.exists()

    def test_disabled_returns_neutral_no_file(self, tmp_path: Path):
        """Disabled config should return neutral and not write file."""
        state_path = tmp_path / "alpha_router_state.json"
        
        strategy_cfg = {
            "alpha_router": {
                "enabled": False,  # Disabled
            }
        }
        
        state = run_alpha_router_step(
            health={"health_score": 0.50},
            meta_state=None,
            router_quality=0.70,
            vol_regime="NORMAL",
            dd_state="NORMAL",
            strategy_cfg=strategy_cfg,
            state_path=state_path,
        )
        
        assert state.target_allocation == NEUTRAL_ALLOCATION
        # Should NOT write file when disabled
        assert not state_path.exists()

    def test_smoothing_across_calls(self, tmp_path: Path):
        """Allocation should smooth across consecutive calls."""
        state_path = tmp_path / "alpha_router_state.json"
        
        strategy_cfg = {
            "alpha_router": {
                "enabled": True,
                "smoothing_alpha": 0.3,  # 30% new, 70% old
            }
        }
        
        # First call - high health
        state1 = run_alpha_router_step(
            health={"health_score": 0.90},
            meta_state=None,
            router_quality=0.80,
            vol_regime="NORMAL",
            dd_state="NORMAL",
            strategy_cfg=strategy_cfg,
            state_path=state_path,
        )
        
        # Second call - lower health
        state2 = run_alpha_router_step(
            health={"health_score": 0.40},  # Much lower
            meta_state=None,
            router_quality=0.80,
            vol_regime="NORMAL",
            dd_state="NORMAL",
            strategy_cfg=strategy_cfg,
            state_path=state_path,
        )
        
        # With smoothing, second call should not drop as sharply
        assert state2.smoothed is True
        assert state2.prev_allocation == state1.target_allocation
        # Should be between old and new raw values
        assert state2.target_allocation < state1.target_allocation
        assert state2.target_allocation > 0.30  # Would be ~0.30 without smoothing


# ---------------------------------------------------------------------------
# Regime Combination Tests
# ---------------------------------------------------------------------------


class TestRegimeCombinations:
    """Tests for various regime combinations."""

    def test_all_normal_max_allocation(self, tmp_path: Path):
        """All normal conditions should give max allocation."""
        cfg = AlphaRouterConfig(
            enabled=True,
            smoothing_alpha=0.0,
        )
        
        state = compute_target_allocation(
            health={"health_score": 0.85},
            meta_state=None,
            router_quality=0.80,
            vol_regime="NORMAL",
            dd_state="NORMAL",
            cfg=cfg,
        )
        
        assert state.target_allocation >= 0.95

    def test_high_vol_only(self, tmp_path: Path):
        """HIGH vol only should moderately reduce allocation."""
        cfg = AlphaRouterConfig(
            enabled=True,
            smoothing_alpha=0.0,
        )
        
        state = compute_target_allocation(
            health={"health_score": 0.85},
            meta_state=None,
            router_quality=0.80,
            vol_regime="HIGH",  # 0.80 penalty
            dd_state="NORMAL",
            cfg=cfg,
        )
        
        # ~1.0 * 0.80 * 1.0 * 1.0 = 0.80
        assert 0.75 <= state.target_allocation <= 0.85

    def test_drawdown_only(self, tmp_path: Path):
        """DRAWDOWN only should reduce allocation."""
        cfg = AlphaRouterConfig(
            enabled=True,
            smoothing_alpha=0.0,
        )
        
        state = compute_target_allocation(
            health={"health_score": 0.85},
            meta_state=None,
            router_quality=0.80,
            vol_regime="NORMAL",
            dd_state="DRAWDOWN",  # 0.65 penalty
            cfg=cfg,
        )
        
        # ~1.0 * 1.0 * 0.65 * 1.0 = 0.65
        assert 0.60 <= state.target_allocation <= 0.70

    def test_poor_router_only(self, tmp_path: Path):
        """Poor router quality only should reduce allocation."""
        cfg = AlphaRouterConfig(
            enabled=True,
            smoothing_alpha=0.0,
        )
        
        state = compute_target_allocation(
            health={"health_score": 0.85},
            meta_state=None,
            router_quality=0.30,  # Poor → 0.70 penalty
            vol_regime="NORMAL",
            dd_state="NORMAL",
            cfg=cfg,
        )
        
        # ~1.0 * 1.0 * 1.0 * 0.70 = 0.70
        assert 0.65 <= state.target_allocation <= 0.75

    def test_everything_bad_hits_floor(self, tmp_path: Path):
        """All bad conditions should hit floor."""
        cfg = AlphaRouterConfig(
            enabled=True,
            allocation_floor=0.15,
            smoothing_alpha=0.0,
        )
        
        state = compute_target_allocation(
            health={"health_score": 0.20},  # Very low → 0.30 base
            meta_state=None,
            router_quality=0.20,  # Poor → 0.70
            vol_regime="CRISIS",  # 0.50
            dd_state="DRAWDOWN",  # 0.65
            cfg=cfg,
        )
        
        # 0.30 * 0.50 * 0.65 * 0.70 = 0.0683 → clamped to floor
        assert state.target_allocation == 0.15


# ---------------------------------------------------------------------------
# Effects Integration Tests
# ---------------------------------------------------------------------------


class TestEffectsIntegration:
    """Tests for integration effects on sizing."""

    def test_allocation_reduces_sizing_limits(self):
        """Allocation < 1.0 should reduce all sizing limits proportionally."""
        from execution.alpha_router import apply_allocation_to_limits
        
        # 60% allocation
        exp, trade, sym = apply_allocation_to_limits(
            allocation=0.60,
            max_total_exposure_pct=0.50,
            max_per_trade_nav_pct=0.06,
            symbol_cap_pct=0.10,
        )
        
        assert exp == pytest.approx(0.30, rel=0.01)
        assert trade == pytest.approx(0.036, rel=0.01)
        assert sym == pytest.approx(0.06, rel=0.01)

    def test_allocation_never_increases_limits(self):
        """Allocation should never increase limits above configured values."""
        from execution.alpha_router import apply_allocation_to_limits
        
        # Even if we somehow got > 1.0, limits should be clamped
        exp, trade, sym = apply_allocation_to_limits(
            allocation=1.5,  # Should be clamped to 1.0
            max_total_exposure_pct=0.50,
            max_per_trade_nav_pct=0.06,
            symbol_cap_pct=0.10,
        )
        
        assert exp <= 0.50
        assert trade <= 0.06
        assert sym <= 0.10
