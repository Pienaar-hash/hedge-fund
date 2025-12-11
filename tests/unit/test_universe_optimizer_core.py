"""
Unit tests for universe_optimizer.py (v7.8_P3).

Tests the core functionality of the Universe Optimizer module:
- Config loading and defaults
- Symbol scoring computation
- Universe selection
- State I/O
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------


class TestUniverseOptimizerConfig:
    """Tests for UniverseOptimizerConfig loading and defaults."""

    def test_default_config(self):
        """Default config should have expected values."""
        from execution.universe_optimizer import UniverseOptimizerConfig

        cfg = UniverseOptimizerConfig()

        assert cfg.enabled is False
        assert cfg.min_universe_size == 4
        assert cfg.max_universe_size == 20
        assert cfg.volatility_regime_shrink is True
        assert cfg.drawdown_shrink is True
        assert cfg.category_diversification_min == 2
        assert cfg.score_threshold == 0.30
        assert "edge_score" in cfg.score_weights
        assert "category_score" in cfg.score_weights

    def test_score_weights_are_present(self):
        """Config should have all score weight components."""
        from execution.universe_optimizer import UniverseOptimizerConfig

        cfg = UniverseOptimizerConfig()
        
        # Check all weight components exist
        assert "edge_score" in cfg.score_weights
        assert "category_score" in cfg.score_weights
        assert "meta_overlay" in cfg.score_weights
        assert "strategy_health" in cfg.score_weights
        assert "allocation_confidence" in cfg.score_weights

    def test_load_config_from_dict(self):
        """Config should load from strategy_config dict."""
        from execution.universe_optimizer import load_universe_optimizer_config

        strategy_config = {
            "universe_optimizer": {
                "enabled": True,
                "min_universe_size": 8,
                "max_universe_size": 30,
                "score_threshold": 0.40,
            }
        }

        cfg = load_universe_optimizer_config(strategy_config)

        assert cfg.enabled is True
        assert cfg.min_universe_size == 8
        assert cfg.max_universe_size == 30
        assert cfg.score_threshold == 0.40

    def test_load_config_missing_section(self):
        """Config without universe_optimizer section returns defaults."""
        from execution.universe_optimizer import load_universe_optimizer_config

        strategy_config = {"sizing": {"default_leverage": 3}}
        cfg = load_universe_optimizer_config(strategy_config)

        assert cfg.enabled is False
        assert cfg.min_universe_size == 4

    def test_load_config_none_returns_defaults(self):
        """Passing None returns defaults."""
        from execution.universe_optimizer import load_universe_optimizer_config

        cfg = load_universe_optimizer_config(None)
        assert cfg.enabled is False


# ---------------------------------------------------------------------------
# Symbol Scoring Tests
# ---------------------------------------------------------------------------


class TestSymbolScoring:
    """Tests for compute_symbol_composite_score."""

    def test_basic_scoring(self):
        """Score should be computed from inputs."""
        from execution.universe_optimizer import (
            UniverseOptimizerConfig,
            compute_symbol_composite_score,
        )

        cfg = UniverseOptimizerConfig()

        symbol_edges = {
            "BTCUSDT": {"edge": 0.8},
        }
        category_edges = {
            "major": {"edge": 0.6},
        }

        score = compute_symbol_composite_score(
            symbol="BTCUSDT",
            symbol_edges=symbol_edges,
            category_edges=category_edges,
            meta_overlay=1.0,
            strategy_health=0.7,
            allocation_confidence=0.9,
            cfg=cfg,
        )

        # Score should be in valid range
        assert 0.0 <= score <= 1.0
        # With good inputs, should be moderately high
        assert score > 0.3

    def test_score_clamped_0_to_1(self):
        """Score should be clamped to [0, 1]."""
        from execution.universe_optimizer import (
            UniverseOptimizerConfig,
            compute_symbol_composite_score,
        )

        cfg = UniverseOptimizerConfig()

        # All max inputs
        score_high = compute_symbol_composite_score(
            symbol="BTCUSDT",
            symbol_edges={"BTCUSDT": {"edge": 2.0}},
            category_edges={"major": {"edge": 2.0}},
            meta_overlay=2.0,
            strategy_health=2.0,
            allocation_confidence=2.0,
            cfg=cfg,
        )
        assert score_high <= 1.0

        # All min inputs
        score_low = compute_symbol_composite_score(
            symbol="BTCUSDT",
            symbol_edges={"BTCUSDT": {"edge": -1.0}},
            category_edges={},
            meta_overlay=0.0,
            strategy_health=0.0,
            allocation_confidence=0.0,
            cfg=cfg,
        )
        assert score_low >= 0.0


# ---------------------------------------------------------------------------
# Universe Selection Tests
# ---------------------------------------------------------------------------


class TestUniverseSelection:
    """Tests for select_optimized_universe."""

    def test_basic_selection(self):
        """Should select top symbols by score."""
        from execution.universe_optimizer import (
            UniverseOptimizerConfig,
            select_optimized_universe,
        )

        cfg = UniverseOptimizerConfig(
            min_universe_size=2,
            max_universe_size=5,
            score_threshold=0.0,
            base_universe=[],  # No base for this test
        )

        symbol_scores = {
            "BTCUSDT": 0.9,
            "ETHUSDT": 0.8,
            "SOLUSDT": 0.7,
            "DOGEUSDT": 0.6,
            "LINKUSDT": 0.5,
            "XRPUSDT": 0.4,
        }
        symbol_edges = {sym: {"edge": 0.5} for sym in symbol_scores}

        selected, notes = select_optimized_universe(
            symbol_scores=symbol_scores,
            symbol_edges=symbol_edges,
            effective_max_size=5,
            cfg=cfg,
        )

        # Should select top 5 by score
        assert len(selected) == 5
        assert "BTCUSDT" in selected
        assert "ETHUSDT" in selected
        assert "XRPUSDT" not in selected  # lowest score

    def test_respects_max_size(self):
        """Should not exceed effective_max_size."""
        from execution.universe_optimizer import (
            UniverseOptimizerConfig,
            select_optimized_universe,
        )

        cfg = UniverseOptimizerConfig(
            min_universe_size=2,
            max_universe_size=10,
            score_threshold=0.0,
            base_universe=[],
        )

        symbol_scores = {f"SYM{i}USDT": 0.5 + i * 0.01 for i in range(10)}
        symbol_edges = {sym: {"edge": 0.5} for sym in symbol_scores}

        selected, notes = select_optimized_universe(
            symbol_scores=symbol_scores,
            symbol_edges=symbol_edges,
            effective_max_size=3,  # Smaller than max
            cfg=cfg,
        )

        assert len(selected) <= 3

    def test_score_threshold_filters(self):
        """Should filter symbols below score_threshold."""
        from execution.universe_optimizer import (
            UniverseOptimizerConfig,
            select_optimized_universe,
        )

        cfg = UniverseOptimizerConfig(
            min_universe_size=1,
            max_universe_size=10,
            score_threshold=0.5,
            base_universe=[],
        )

        symbol_scores = {
            "BTCUSDT": 0.9,
            "ETHUSDT": 0.6,
            "DOGEUSDT": 0.3,  # Below threshold
        }
        symbol_edges = {sym: {"edge": 0.5} for sym in symbol_scores}

        selected, notes = select_optimized_universe(
            symbol_scores=symbol_scores,
            symbol_edges=symbol_edges,
            effective_max_size=10,
            cfg=cfg,
        )

        assert "BTCUSDT" in selected
        assert "ETHUSDT" in selected
        # DOGE below threshold should be excluded
        assert "DOGEUSDT" not in selected


# ---------------------------------------------------------------------------
# State I/O Tests
# ---------------------------------------------------------------------------


class TestStateIO:
    """Tests for state loading and writing."""

    def test_state_roundtrip(self, tmp_path):
        """State should survive JSON serialization."""
        from execution.universe_optimizer import (
            UniverseOptimizerState,
            write_universe_optimizer_state,
            load_universe_optimizer_state,
        )

        state = UniverseOptimizerState(
            updated_ts="2025-01-01T00:00:00Z",
            allowed_symbols=["BTCUSDT", "ETHUSDT"],
            symbol_scores={"BTCUSDT": 0.8, "ETHUSDT": 0.6},
            category_scores={"major": 0.7},
            total_universe_size=10,
            effective_max_size=5,
            notes=["Test note"],
        )

        state_path = tmp_path / "universe_optimizer.json"

        write_universe_optimizer_state(state, state_path)
        loaded = load_universe_optimizer_state(state_path)

        assert loaded is not None
        assert loaded.allowed_symbols == ["BTCUSDT", "ETHUSDT"]
        assert loaded.symbol_scores["BTCUSDT"] == 0.8
        assert loaded.total_universe_size == 10

    def test_load_missing_state_returns_none(self, tmp_path):
        """Loading missing state file returns None."""
        from execution.universe_optimizer import load_universe_optimizer_state

        state_path = tmp_path / "nonexistent.json"
        state = load_universe_optimizer_state(state_path)

        assert state is None

    def test_state_to_dict(self):
        """State.to_dict() should produce valid JSON-serializable dict."""
        from execution.universe_optimizer import UniverseOptimizerState

        state = UniverseOptimizerState(
            updated_ts="2025-01-01T00:00:00Z",
            allowed_symbols=["BTCUSDT"],
            symbol_scores={"BTCUSDT": 0.8888},
            category_scores={},
            total_universe_size=5,
            effective_max_size=5,
            notes=[],
        )

        d = state.to_dict()

        assert d["updated_ts"] == "2025-01-01T00:00:00Z"
        assert d["allowed_symbols"] == ["BTCUSDT"]
        # Scores should be rounded
        assert d["symbol_scores"]["BTCUSDT"] == 0.8888

        # Should be JSON serializable
        json_str = json.dumps(d)
        assert json_str


# ---------------------------------------------------------------------------
# get_allowed_symbols Tests
# ---------------------------------------------------------------------------


class TestGetAllowedSymbols:
    """Tests for get_allowed_symbols public API."""

    def test_returns_list_when_state_exists(self, tmp_path):
        """Should return allowed symbols list when state exists."""
        from execution.universe_optimizer import (
            UniverseOptimizerState,
            write_universe_optimizer_state,
            get_allowed_symbols,
        )

        state = UniverseOptimizerState(
            updated_ts="2025-01-01T00:00:00Z",
            allowed_symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            symbol_scores={},
            category_scores={},
            total_universe_size=10,
            effective_max_size=10,
            notes=[],
        )

        state_path = tmp_path / "universe_optimizer.json"
        write_universe_optimizer_state(state, state_path)
        
        result = get_allowed_symbols(path=state_path)

        assert result == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    def test_returns_none_when_no_state(self, tmp_path):
        """Should return None when no state file exists."""
        from execution.universe_optimizer import get_allowed_symbols

        state_path = tmp_path / "nonexistent.json"
        result = get_allowed_symbols(path=state_path)

        assert result is None

    def test_returns_from_state_object(self):
        """Should return from pre-loaded state object."""
        from execution.universe_optimizer import (
            UniverseOptimizerState,
            get_allowed_symbols,
        )

        state = UniverseOptimizerState(
            updated_ts="2025-01-01T00:00:00Z",
            allowed_symbols=["BTCUSDT"],
            symbol_scores={},
            category_scores={},
            total_universe_size=5,
            effective_max_size=5,
            notes=[],
        )

        result = get_allowed_symbols(state=state)
        assert result == ["BTCUSDT"]


# ---------------------------------------------------------------------------
# is_symbol_allowed Tests
# ---------------------------------------------------------------------------


class TestIsSymbolAllowed:
    """Tests for is_symbol_allowed function."""

    def test_symbol_in_allowed_list(self, tmp_path):
        """Should return True for symbol in allowed list."""
        from execution.universe_optimizer import (
            UniverseOptimizerState,
            write_universe_optimizer_state,
            is_symbol_allowed,
        )

        state = UniverseOptimizerState(
            updated_ts="2025-01-01T00:00:00Z",
            allowed_symbols=["BTCUSDT", "ETHUSDT"],
            symbol_scores={},
            category_scores={},
            total_universe_size=5,
            effective_max_size=5,
            notes=[],
        )

        state_path = tmp_path / "universe_optimizer.json"
        write_universe_optimizer_state(state, state_path)

        assert is_symbol_allowed("BTCUSDT", path=state_path) is True
        assert is_symbol_allowed("ETHUSDT", path=state_path) is True
        assert is_symbol_allowed("DOGEUSDT", path=state_path) is False

    def test_no_state_returns_true(self, tmp_path):
        """Should return True (no filtering) when no state."""
        from execution.universe_optimizer import is_symbol_allowed

        state_path = tmp_path / "nonexistent.json"
        result = is_symbol_allowed("BTCUSDT", path=state_path)

        # When optimizer not active, all symbols allowed
        assert result is True


# ---------------------------------------------------------------------------
# filter_candidates_by_universe Tests
# ---------------------------------------------------------------------------


class TestFilterCandidates:
    """Tests for filter_candidates_by_universe."""

    def test_filters_candidates(self, tmp_path):
        """Should filter candidates to allowed universe."""
        from execution.universe_optimizer import (
            UniverseOptimizerConfig,
            UniverseOptimizerState,
            filter_candidates_by_universe,
        )

        cfg = UniverseOptimizerConfig(enabled=True)

        state = UniverseOptimizerState(
            updated_ts="2025-01-01T00:00:00Z",
            allowed_symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],  # 3 symbols for min_size_fallback
            symbol_scores={},
            category_scores={},
            total_universe_size=5,
            effective_max_size=5,
            notes=[],
        )

        candidates = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]
        filtered, was_filtered = filter_candidates_by_universe(
            candidates=candidates,
            cfg=cfg,
            state=state,
        )

        assert "BTCUSDT" in filtered
        assert "ETHUSDT" in filtered
        assert "SOLUSDT" in filtered
        assert "DOGEUSDT" not in filtered
        assert was_filtered is True

    def test_disabled_returns_all(self):
        """Should return all candidates when disabled."""
        from execution.universe_optimizer import (
            UniverseOptimizerConfig,
            filter_candidates_by_universe,
        )

        cfg = UniverseOptimizerConfig(enabled=False)

        candidates = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        filtered, was_filtered = filter_candidates_by_universe(
            candidates=candidates,
            cfg=cfg,
        )

        # All candidates pass when disabled
        assert filtered == candidates
        assert was_filtered is False

    def test_no_state_returns_all(self, tmp_path):
        """Should return all candidates when state file doesn't exist."""
        from execution.universe_optimizer import (
            UniverseOptimizerConfig,
            UniverseOptimizerState,
            filter_candidates_by_universe,
        )

        cfg = UniverseOptimizerConfig(enabled=True)
        
        # Create an empty state to simulate "no valid state"
        empty_state = UniverseOptimizerState(
            allowed_symbols=[],  # Empty = no filtering
        )

        candidates = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        filtered, was_filtered = filter_candidates_by_universe(
            candidates=candidates,
            cfg=cfg,
            state=empty_state,
        )

        # All candidates pass when state has no symbols
        assert filtered == candidates
        assert was_filtered is False


# ---------------------------------------------------------------------------
# run_universe_optimizer_step Tests
# ---------------------------------------------------------------------------


class TestRunOptimizerStep:
    """Tests for run_universe_optimizer_step orchestration."""

    def test_disabled_returns_empty_state(self, tmp_path):
        """When disabled, should return empty state without writing."""
        from execution.universe_optimizer import run_universe_optimizer_step

        strategy_cfg = {
            "universe_optimizer": {"enabled": False}
        }

        state_path = tmp_path / "universe_optimizer.json"

        state = run_universe_optimizer_step(
            candidate_symbols=["BTCUSDT", "ETHUSDT"],
            symbol_edges={"BTCUSDT": {"edge": 0.8}},
            category_edges={},
            vol_regime="NORMAL",
            dd_state="NORMAL",
            strategy_cfg=strategy_cfg,
            state_path=state_path,
        )

        # Should return empty state
        assert state.allowed_symbols == []
        # Should NOT write file
        assert not state_path.exists()

    def test_enabled_writes_state(self, tmp_path):
        """When enabled, should write state file."""
        from execution.universe_optimizer import run_universe_optimizer_step

        strategy_cfg = {
            "universe_optimizer": {
                "enabled": True,
                "min_universe_size": 2,
                "max_universe_size": 10,
                "score_threshold": 0.0,
            }
        }

        state_path = tmp_path / "universe_optimizer.json"

        state = run_universe_optimizer_step(
            candidate_symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            symbol_edges={
                "BTCUSDT": {"edge": 0.8},
                "ETHUSDT": {"edge": 0.6},
                "SOLUSDT": {"edge": 0.4},
            },
            category_edges={},
            vol_regime="NORMAL",
            dd_state="NORMAL",
            strategy_cfg=strategy_cfg,
            state_path=state_path,
        )

        # Should have symbols in universe
        assert len(state.allowed_symbols) >= 2
        # Should write file
        assert state_path.exists()

        # Verify file content
        data = json.loads(state_path.read_text())
        assert "allowed_symbols" in data
        assert "symbol_scores" in data
