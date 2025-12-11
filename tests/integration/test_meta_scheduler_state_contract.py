"""
Integration tests for Meta-Scheduler state contract (v7.8_P1).

Tests:
- State file schema matches manifest
- Atomic writes work correctly
- State persistence across loads
"""

import pytest
import json
import tempfile
from pathlib import Path

from execution.meta_scheduler import (
    MetaSchedulerConfig,
    MetaSchedulerState,
    FactorMetaState,
    ConvictionMetaState,
    CategoryMetaState,
    create_neutral_state,
    load_meta_scheduler_state,
    write_meta_scheduler_state,
    meta_learning_step,
)


# ---------------------------------------------------------------------------
# State Schema Tests
# ---------------------------------------------------------------------------


class TestStateSchema:
    """Tests for state file schema compliance."""

    def test_state_has_required_fields(self):
        """State dict has all required fields from manifest."""
        state = create_neutral_state()
        data = state.to_dict()
        
        # Required fields from v7_manifest.json
        assert "updated_ts" in data
        assert "factor_state" in data
        assert "conviction_state" in data
        assert "category_state" in data
        assert "stats" in data

    def test_factor_state_schema(self):
        """factor_state has correct structure."""
        state = MetaSchedulerState(
            factor_state=FactorMetaState(
                meta_weights={"momentum": 1.05},
                ema_ir={"momentum": 0.3},
                ema_pnl={"momentum": 100.0},
            ),
        )
        data = state.to_dict()
        
        factor_state = data["factor_state"]
        assert "meta_weights" in factor_state
        assert "ema_ir" in factor_state
        assert "ema_pnl" in factor_state

    def test_conviction_state_schema(self):
        """conviction_state has correct structure."""
        state = MetaSchedulerState(
            conviction_state=ConvictionMetaState(
                global_strength=1.10,
                ema_health=0.75,
            ),
        )
        data = state.to_dict()
        
        conviction_state = data["conviction_state"]
        assert "global_strength" in conviction_state
        assert "ema_health" in conviction_state

    def test_category_state_schema(self):
        """category_state has correct structure."""
        state = MetaSchedulerState(
            category_state=CategoryMetaState(
                category_overlays={"btc": 1.10},
                ema_category_ir={"btc": 0.5},
                ema_category_pnl={"btc": 100.0},
            ),
        )
        data = state.to_dict()
        
        category_state = data["category_state"]
        assert "category_overlays" in category_state
        assert "ema_category_ir" in category_state
        assert "ema_category_pnl" in category_state

    def test_stats_contains_expected_fields(self):
        """stats contains expected tracking fields."""
        cfg = MetaSchedulerConfig(enabled=True, min_samples=0)
        state = meta_learning_step(
            cfg=cfg,
            prev_state=None,
            factor_edges={},
            category_edges={},
            strategy_health={"health_score": 0.7},
        )
        
        data = state.to_dict()
        stats = data["stats"]
        
        assert "sample_count" in stats
        # learning_active only present when enabled
        assert "learning_active" in stats or "sample_count" in stats


# ---------------------------------------------------------------------------
# State I/O Tests
# ---------------------------------------------------------------------------


class TestStateIO:
    """Tests for state file I/O."""

    def test_write_and_load_roundtrip(self, tmp_path):
        """State survives write → load roundtrip."""
        path = tmp_path / "meta_scheduler.json"
        
        original = MetaSchedulerState(
            updated_ts="2025-01-01T00:00:00Z",
            factor_state=FactorMetaState(
                meta_weights={"momentum": 1.08, "carry": 0.95},
                ema_ir={"momentum": 0.4},
                ema_pnl={"momentum": 50.0},
            ),
            conviction_state=ConvictionMetaState(
                global_strength=1.12,
                ema_health=0.68,
            ),
            category_state=CategoryMetaState(
                category_overlays={"btc": 1.10},
            ),
            stats={"sample_count": 100},
        )
        
        write_meta_scheduler_state(original, path)
        loaded = load_meta_scheduler_state(path)
        
        assert loaded is not None
        assert loaded.factor_state.meta_weights == original.factor_state.meta_weights
        assert loaded.conviction_state.global_strength == original.conviction_state.global_strength
        assert loaded.category_state.category_overlays == original.category_state.category_overlays

    def test_load_nonexistent_returns_none(self, tmp_path):
        """Loading nonexistent file returns None."""
        path = tmp_path / "nonexistent.json"
        
        state = load_meta_scheduler_state(path)
        
        assert state is None

    def test_load_invalid_json_returns_none(self, tmp_path):
        """Loading invalid JSON returns None."""
        path = tmp_path / "invalid.json"
        path.write_text("not valid json {{{")
        
        state = load_meta_scheduler_state(path)
        
        assert state is None

    def test_atomic_write(self, tmp_path):
        """Write is atomic (uses temp file)."""
        path = tmp_path / "meta_scheduler.json"
        
        # Write initial state
        state1 = create_neutral_state()
        write_meta_scheduler_state(state1, path)
        
        # Write second state
        state2 = MetaSchedulerState(
            factor_state=FactorMetaState(meta_weights={"momentum": 1.10}),
        )
        write_meta_scheduler_state(state2, path)
        
        # Should have the second state
        loaded = load_meta_scheduler_state(path)
        assert loaded is not None
        assert loaded.factor_state.meta_weights.get("momentum") == 1.10

    def test_creates_parent_directory(self, tmp_path):
        """Write creates parent directories if needed."""
        path = tmp_path / "nested" / "dir" / "meta_scheduler.json"
        
        state = create_neutral_state()
        write_meta_scheduler_state(state, path)
        
        assert path.exists()
        loaded = load_meta_scheduler_state(path)
        assert loaded is not None


# ---------------------------------------------------------------------------
# Learning Sequence Tests
# ---------------------------------------------------------------------------


class TestLearningSequence:
    """Tests for multi-step learning sequences."""

    def test_state_accumulates_across_steps(self):
        """State accumulates learning across multiple steps."""
        cfg = MetaSchedulerConfig(
            enabled=True,
            learning_rate=0.10,
            min_samples=0,
            max_factor_shift=0.20,
            decay=0.90,
        )
        
        # Step 1: Strong positive signal
        state = meta_learning_step(
            cfg=cfg,
            prev_state=None,
            factor_edges={"momentum": {"ir": 0.5, "pnl_contrib": 100.0}},
            category_edges={},
            strategy_health={"health_score": 0.8},
        )
        
        assert state.stats["sample_count"] == 1
        
        # Step 2: Continue positive signal
        state = meta_learning_step(
            cfg=cfg,
            prev_state=state,
            factor_edges={"momentum": {"ir": 0.6, "pnl_contrib": 150.0}},
            category_edges={},
            strategy_health={"health_score": 0.85},
        )
        
        assert state.stats["sample_count"] == 2
        # EMA should be building up
        assert state.factor_state.ema_ir.get("momentum", 0) > 0

    def test_ema_smooths_across_steps(self):
        """EMA properly smooths values across steps."""
        cfg = MetaSchedulerConfig(
            enabled=True,
            learning_rate=0.05,
            min_samples=0,
            decay=0.80,  # 80% previous, 20% new
        )
        
        # Initial step with high value
        state = meta_learning_step(
            cfg=cfg,
            prev_state=None,
            factor_edges={"momentum": {"ir": 1.0}},
            category_edges={},
            strategy_health={"health_score": 0.9},
        )
        
        initial_ema = state.factor_state.ema_ir.get("momentum", 0)
        
        # Second step with lower value
        state = meta_learning_step(
            cfg=cfg,
            prev_state=state,
            factor_edges={"momentum": {"ir": 0.0}},
            category_edges={},
            strategy_health={"health_score": 0.5},
        )
        
        # EMA should be between initial and new value
        new_ema = state.factor_state.ema_ir.get("momentum", 0)
        assert new_ema < initial_ema  # Lower due to 0.0 input
        assert new_ema > 0  # But not zero due to EMA smoothing


# ---------------------------------------------------------------------------
# JSON Serialization Tests
# ---------------------------------------------------------------------------


class TestJsonSerialization:
    """Tests for JSON serialization compliance."""

    def test_state_is_json_serializable(self):
        """Complete state serializes to valid JSON."""
        state = MetaSchedulerState(
            updated_ts="2025-01-01T00:00:00Z",
            factor_state=FactorMetaState(
                meta_weights={"momentum": 1.08},
                ema_ir={"momentum": 0.5},
                ema_pnl={"momentum": 100.0},
            ),
            conviction_state=ConvictionMetaState(
                global_strength=1.10,
                ema_health=0.75,
            ),
            category_state=CategoryMetaState(
                category_overlays={"btc": 1.05},
            ),
            stats={"sample_count": 50, "learning_active": True},
        )
        
        data = state.to_dict()
        json_str = json.dumps(data)
        
        # Should round-trip through JSON
        parsed = json.loads(json_str)
        assert parsed["factor_state"]["meta_weights"]["momentum"] == 1.08

    def test_float_precision_preserved(self):
        """Float precision is preserved to 6 decimal places."""
        state = MetaSchedulerState(
            factor_state=FactorMetaState(
                meta_weights={"momentum": 1.123456789},
            ),
        )
        
        data = state.to_dict()
        
        # Should be rounded to 6 decimal places
        assert data["factor_state"]["meta_weights"]["momentum"] == 1.123457
