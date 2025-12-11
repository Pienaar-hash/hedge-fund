"""
Integration tests for universe_optimizer.py (v7.8_P3).

Tests the integration of Universe Optimizer with:
- edge_scanner.py (update_universe_optimizer)
- signal_screener.py (get_allowed_symbols filtering)
- State contract compliance
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, Any


# ---------------------------------------------------------------------------
# Integration with edge_scanner Tests
# ---------------------------------------------------------------------------


class TestEdgeScannerIntegration:
    """Tests for update_universe_optimizer in edge_scanner."""

    def test_update_universe_optimizer_disabled(self, tmp_path):
        """Should not write state when disabled."""
        from execution.edge_scanner import update_universe_optimizer, EdgeInsights

        # Create minimal EdgeInsights
        edge_insights = EdgeInsights(
            updated_ts="2025-01-01T00:00:00Z",
            edge_summary={},
            factor_edges={},
            symbol_edges={},
            category_edges={},
            config_echo={},
        )

        strategy_config = {
            "universe_optimizer": {
                "enabled": False,
            }
        }

        state_path = tmp_path / "universe_optimizer.json"

        # Patch the default path
        with patch(
            "execution.universe_optimizer.DEFAULT_UNIVERSE_OPTIMIZER_PATH",
            state_path,
        ):
            update_universe_optimizer(edge_insights, strategy_config)

        # State file should NOT be created (disabled)
        assert not state_path.exists()

    def test_update_universe_optimizer_enabled(self, tmp_path):
        """Should write state when enabled using run_universe_optimizer_step directly."""
        from execution.universe_optimizer import run_universe_optimizer_step

        strategy_config = {
            "universe_optimizer": {
                "enabled": True,
                "min_universe_size": 2,
                "max_universe_size": 10,
                "score_threshold": 0.0,
            }
        }

        state_path = tmp_path / "universe_optimizer.json"

        # Call run_universe_optimizer_step directly with state_path parameter
        state = run_universe_optimizer_step(
            candidate_symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            symbol_edges={
                "BTCUSDT": {"edge": 0.8},
                "ETHUSDT": {"edge": 0.6},
                "SOLUSDT": {"edge": 0.4},
            },
            category_edges={
                "major": {"edge": 0.7},
            },
            vol_regime="NORMAL",
            dd_state="NORMAL",
            strategy_cfg=strategy_config,
            state_path=state_path,
        )

        # State file should be created
        assert state_path.exists()
        data = json.loads(state_path.read_text())
        assert "allowed_symbols" in data
        assert "symbol_scores" in data
        # Should have at least 2 symbols
        assert len(data["allowed_symbols"]) >= 2

        # State file should be created
        assert state_path.exists()
        data = json.loads(state_path.read_text())
        assert "allowed_symbols" in data
        assert "symbol_scores" in data


# ---------------------------------------------------------------------------
# Integration with signal_screener Tests
# ---------------------------------------------------------------------------


class TestSignalScreenerIntegration:
    """Tests for get_allowed_symbols integration in signal_screener."""

    def test_get_allowed_symbols_filters_candidates(self, tmp_path):
        """When enabled, get_allowed_symbols should filter candidates."""
        from execution.universe_optimizer import (
            UniverseOptimizerState,
            write_universe_optimizer_state,
            get_allowed_symbols,
        )

        state = UniverseOptimizerState(
            updated_ts="2025-01-01T00:00:00Z",
            allowed_symbols=["BTCUSDT", "ETHUSDT"],
            symbol_scores={"BTCUSDT": 0.8, "ETHUSDT": 0.6},
            category_scores={},
            total_universe_size=10,
            effective_max_size=10,
            notes=[],
        )

        state_path = tmp_path / "universe_optimizer.json"
        write_universe_optimizer_state(state, state_path)
        
        allowed = get_allowed_symbols(path=state_path)

        # Universe should contain only BTC and ETH
        assert allowed == ["BTCUSDT", "ETHUSDT"]

        # Test intersection logic (as done in signal_screener)
        base_allowed = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"}
        optimized_set = {s.upper() for s in allowed}
        filtered = base_allowed & optimized_set

        assert "BTCUSDT" in filtered
        assert "ETHUSDT" in filtered
        assert "SOLUSDT" not in filtered
        assert "DOGEUSDT" not in filtered

    def test_empty_state_returns_none(self, tmp_path):
        """No state file should return None (no filtering)."""
        from execution.universe_optimizer import get_allowed_symbols

        state_path = tmp_path / "nonexistent.json"
        allowed = get_allowed_symbols(path=state_path)

        assert allowed is None


# ---------------------------------------------------------------------------
# State Contract Tests
# ---------------------------------------------------------------------------


class TestStateContract:
    """Tests verifying state file contract."""

    def test_state_has_required_fields(self, tmp_path):
        """State file should have all required fields per v7_manifest."""
        from execution.universe_optimizer import (
            run_universe_optimizer_step,
        )

        strategy_cfg = {
            "universe_optimizer": {
                "enabled": True,
                "min_universe_size": 2,
                "max_universe_size": 5,
            }
        }

        state_path = tmp_path / "universe_optimizer.json"

        run_universe_optimizer_step(
            candidate_symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            symbol_edges={"BTCUSDT": {"edge": 0.8}, "ETHUSDT": {"edge": 0.6}},
            category_edges={"major": {"edge": 0.7}},
            vol_regime="NORMAL",
            dd_state="NORMAL",
            strategy_cfg=strategy_cfg,
            state_path=state_path,
        )

        data = json.loads(state_path.read_text())

        # Required fields per v7_manifest
        assert "updated_ts" in data
        assert "allowed_symbols" in data
        assert "symbol_scores" in data
        assert "category_scores" in data
        assert "total_universe_size" in data
        assert "effective_max_size" in data
        assert "notes" in data

        # Type checks
        assert isinstance(data["updated_ts"], str)
        assert isinstance(data["allowed_symbols"], list)
        assert isinstance(data["symbol_scores"], dict)
        assert isinstance(data["category_scores"], dict)
        assert isinstance(data["total_universe_size"], int)
        assert isinstance(data["effective_max_size"], int)
        assert isinstance(data["notes"], list)


# ---------------------------------------------------------------------------
# Effect Tests (end-to-end behavior)
# ---------------------------------------------------------------------------


class TestUniverseOptimizerEffects:
    """Tests verifying expected runtime behavior."""

    def test_high_edge_symbols_included(self, tmp_path):
        """High-edge symbols should be included in universe."""
        from execution.universe_optimizer import run_universe_optimizer_step

        strategy_cfg = {
            "universe_optimizer": {
                "enabled": True,
                "min_universe_size": 2,
                "max_universe_size": 5,
                "score_threshold": 0.0,
            }
        }

        symbol_edges = {
            "BTCUSDT": {"edge": 0.9},  # highest
            "ETHUSDT": {"edge": 0.7},
            "SOLUSDT": {"edge": 0.5},
            "DOGEUSDT": {"edge": 0.3},
            "LINKUSDT": {"edge": 0.1},  # lowest
        }

        state_path = tmp_path / "universe_optimizer.json"

        state = run_universe_optimizer_step(
            candidate_symbols=list(symbol_edges.keys()),
            symbol_edges=symbol_edges,
            category_edges={},
            vol_regime="NORMAL",
            dd_state="NORMAL",
            strategy_cfg=strategy_cfg,
            state_path=state_path,
        )

        # BTC and ETH should definitely be in
        assert "BTCUSDT" in state.allowed_symbols
        assert "ETHUSDT" in state.allowed_symbols

    def test_regime_shrinks_effective_max(self, tmp_path):
        """HIGH/CRISIS vol regime should shrink effective_max_size."""
        from execution.universe_optimizer import run_universe_optimizer_step

        strategy_cfg = {
            "universe_optimizer": {
                "enabled": True,
                "min_universe_size": 2,
                "max_universe_size": 10,
                "score_threshold": 0.0,
            }
        }

        symbol_edges = {f"SYM{i}USDT": {"edge": 0.5 + i * 0.01} for i in range(10)}

        state_path_normal = tmp_path / "universe_optimizer_normal.json"
        state_path_high = tmp_path / "universe_optimizer_high.json"

        state_normal = run_universe_optimizer_step(
            candidate_symbols=list(symbol_edges.keys()),
            symbol_edges=symbol_edges,
            category_edges={},
            vol_regime="NORMAL",
            dd_state="NORMAL",
            strategy_cfg=strategy_cfg,
            state_path=state_path_normal,
        )

        state_high = run_universe_optimizer_step(
            candidate_symbols=list(symbol_edges.keys()),
            symbol_edges=symbol_edges,
            category_edges={},
            vol_regime="HIGH",
            dd_state="NORMAL",
            strategy_cfg=strategy_cfg,
            state_path=state_path_high,
        )

        # HIGH vol should have smaller effective_max_size
        assert state_high.effective_max_size < state_normal.effective_max_size


# ---------------------------------------------------------------------------
# Manifest Alignment Tests
# ---------------------------------------------------------------------------


class TestManifestAlignment:
    """Tests ensuring state aligns with v7_manifest.json."""

    def test_state_path_matches_manifest(self):
        """DEFAULT_UNIVERSE_OPTIMIZER_PATH should match v7_manifest.json."""
        from execution.universe_optimizer import DEFAULT_UNIVERSE_OPTIMIZER_PATH

        expected_path = Path("logs/state/universe_optimizer.json")
        assert DEFAULT_UNIVERSE_OPTIMIZER_PATH == expected_path

    def test_manifest_has_universe_optimizer(self):
        """v7_manifest.json should include universe_optimizer."""
        manifest_path = Path("v7_manifest.json")
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            state_files = manifest.get("state_files", {})
            assert "universe_optimizer" in state_files

            entry = state_files["universe_optimizer"]
            assert entry.get("path") == "logs/state/universe_optimizer.json"
            assert entry.get("owner") == "executor"
            assert entry.get("optional") is True
