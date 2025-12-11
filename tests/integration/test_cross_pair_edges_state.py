"""
Integration tests for cross-pair edges state surface.

Tests:
- State file schema validation
- Manifest alignment
- State surface consistency
- Integration with EdgeInsights

v7.8_P5: Cross-Pair Statistical Arbitrage Engine.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_cross_pair_state():
    """Sample cross-pair edges state for testing."""
    return {
        "updated_ts": 1700000000.0,
        "pair_edges": {
            "BTCUSDT/ETHUSDT": {
                "pair": ["BTCUSDT", "ETHUSDT"],
                "ema_score": 0.72,
                "long_leg": "ETHUSDT",
                "short_leg": "BTCUSDT",
                "signal": "ENTER",
                "stats": {
                    "base": "BTCUSDT",
                    "quote": "ETHUSDT",
                    "hedge_ratio": 16.5,
                    "spread_mean": 0.0,
                    "spread_std": 120.5,
                    "spread_z": 2.1,
                    "corr": 0.88,
                    "half_life_est": 22.5,
                    "residual_momo": 0.005,
                    "liquidity_ok": True,
                    "eligible": True,
                }
            },
            "SOLUSDT/SUIUSDT": {
                "pair": ["SOLUSDT", "SUIUSDT"],
                "ema_score": 0.45,
                "long_leg": "",
                "short_leg": "",
                "signal": "NONE",
                "stats": {
                    "base": "SOLUSDT",
                    "quote": "SUIUSDT",
                    "hedge_ratio": 0.8,
                    "spread_mean": 0.0,
                    "spread_std": 0.05,
                    "spread_z": 0.8,
                    "corr": 0.75,
                    "half_life_est": 35.0,
                    "residual_momo": 0.002,
                    "liquidity_ok": True,
                    "eligible": True,
                }
            }
        },
        "pairs_analyzed": 3,
        "pairs_eligible": 2,
        "cycle_count": 15,
        "notes": "Cross-pair scan complete",
    }


@pytest.fixture
def state_dir(tmp_path):
    """Temporary state directory."""
    state_path = tmp_path / "state"
    state_path.mkdir(parents=True, exist_ok=True)
    return state_path


# ---------------------------------------------------------------------------
# Test: State File Schema
# ---------------------------------------------------------------------------

class TestCrossPairStateSchema:
    """Tests for cross-pair edges state file schema."""

    def test_required_fields_present(self, sample_cross_pair_state):
        """Test that required fields are present in state."""
        required_fields = [
            "updated_ts",
            "pair_edges",
            "pairs_analyzed",
            "pairs_eligible",
        ]
        for field in required_fields:
            assert field in sample_cross_pair_state, f"Missing required field: {field}"

    def test_updated_ts_is_numeric(self, sample_cross_pair_state):
        """Test that updated_ts is a numeric timestamp."""
        assert isinstance(sample_cross_pair_state["updated_ts"], (int, float))
        assert sample_cross_pair_state["updated_ts"] > 0

    def test_pair_edges_is_dict(self, sample_cross_pair_state):
        """Test that pair_edges is a dictionary."""
        assert isinstance(sample_cross_pair_state["pair_edges"], dict)

    def test_pair_edge_structure(self, sample_cross_pair_state):
        """Test that each pair edge has required structure."""
        required_edge_fields = ["pair", "ema_score", "long_leg", "short_leg", "signal", "stats"]
        
        for key, edge in sample_cross_pair_state["pair_edges"].items():
            for field in required_edge_fields:
                assert field in edge, f"Missing field {field} in edge {key}"

    def test_pair_stats_structure(self, sample_cross_pair_state):
        """Test that pair stats have required structure."""
        required_stats_fields = [
            "base", "quote", "hedge_ratio", "spread_mean", "spread_std",
            "spread_z", "corr", "half_life_est", "liquidity_ok", "eligible"
        ]
        
        for key, edge in sample_cross_pair_state["pair_edges"].items():
            stats = edge["stats"]
            for field in required_stats_fields:
                assert field in stats, f"Missing stats field {field} in edge {key}"

    def test_signal_values_valid(self, sample_cross_pair_state):
        """Test that signal values are one of ENTER/EXIT/NONE."""
        valid_signals = {"ENTER", "EXIT", "NONE"}
        
        for key, edge in sample_cross_pair_state["pair_edges"].items():
            assert edge["signal"] in valid_signals, f"Invalid signal {edge['signal']} in {key}"

    def test_score_in_valid_range(self, sample_cross_pair_state):
        """Test that ema_score is in [0, 1] range."""
        for key, edge in sample_cross_pair_state["pair_edges"].items():
            score = edge["ema_score"]
            assert 0 <= score <= 1, f"Score {score} out of range in {key}"

    def test_correlation_in_valid_range(self, sample_cross_pair_state):
        """Test that correlation is in [-1, 1] range."""
        for key, edge in sample_cross_pair_state["pair_edges"].items():
            corr = edge["stats"]["corr"]
            assert -1 <= corr <= 1, f"Correlation {corr} out of range in {key}"


# ---------------------------------------------------------------------------
# Test: Manifest Alignment
# ---------------------------------------------------------------------------

class TestManifestAlignment:
    """Tests for v7_manifest.json alignment."""

    def test_cross_pair_edges_in_manifest(self):
        """Test that cross_pair_edges is defined in manifest."""
        manifest_path = Path("v7_manifest.json")
        if not manifest_path.exists():
            pytest.skip("v7_manifest.json not found")
        
        manifest = json.loads(manifest_path.read_text())
        state_files = manifest.get("state_files", {})
        
        assert "cross_pair_edges" in state_files, "cross_pair_edges not in manifest"

    def test_cross_pair_edges_manifest_fields(self):
        """Test manifest entry has required fields."""
        manifest_path = Path("v7_manifest.json")
        if not manifest_path.exists():
            pytest.skip("v7_manifest.json not found")
        
        manifest = json.loads(manifest_path.read_text())
        entry = manifest.get("state_files", {}).get("cross_pair_edges", {})
        
        # Check required manifest fields
        assert "path" in entry, "Missing 'path' in manifest entry"
        assert "owner" in entry, "Missing 'owner' in manifest entry"
        assert "cross_pair_edges.json" in entry["path"], "Incorrect file path"
        # Owner can be executor or cross_pair_engine
        assert entry["owner"] in ("executor", "cross_pair_engine", "intel"), "Unexpected owner"


# ---------------------------------------------------------------------------
# Test: State File Write/Read Cycle
# ---------------------------------------------------------------------------

class TestStateFileIO:
    """Tests for state file I/O operations."""

    def test_write_and_read_state(self, state_dir, sample_cross_pair_state):
        """Test writing and reading back state file."""
        state_file = state_dir / "cross_pair_edges.json"
        
        # Write state
        state_file.write_text(json.dumps(sample_cross_pair_state, indent=2))
        
        # Read back
        loaded = json.loads(state_file.read_text())
        
        assert loaded["updated_ts"] == sample_cross_pair_state["updated_ts"]
        assert loaded["pairs_analyzed"] == sample_cross_pair_state["pairs_analyzed"]
        assert len(loaded["pair_edges"]) == len(sample_cross_pair_state["pair_edges"])

    def test_state_file_is_valid_json(self, state_dir, sample_cross_pair_state):
        """Test that state file is valid JSON."""
        state_file = state_dir / "cross_pair_edges.json"
        state_file.write_text(json.dumps(sample_cross_pair_state))
        
        # Should not raise
        content = state_file.read_text()
        parsed = json.loads(content)
        assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# Test: Integration with EdgeInsights
# ---------------------------------------------------------------------------

class TestEdgeInsightsIntegration:
    """Tests for integration with EdgeInsights surface."""

    def test_edge_insights_pair_edges_field(self):
        """Test that EdgeInsights has pair_edges field."""
        from execution.edge_scanner import EdgeInsights
        
        # Create EdgeInsights with pair_edges
        insights = EdgeInsights(
            updated_ts="2024-01-01T00:00:00Z",
            pair_edges={"BTCUSDT/ETHUSDT": {"ema_score": 0.75}},
        )
        
        assert insights.pair_edges is not None
        assert "BTCUSDT/ETHUSDT" in insights.pair_edges

    def test_edge_insights_pair_edges_optional(self):
        """Test that EdgeInsights pair_edges field is optional."""
        from execution.edge_scanner import EdgeInsights
        
        # Create EdgeInsights without pair_edges
        insights = EdgeInsights(
            updated_ts="2024-01-01T00:00:00Z",
        )
        
        # Should default to None
        assert insights.pair_edges is None


# ---------------------------------------------------------------------------
# Test: Universe Optimizer Integration
# ---------------------------------------------------------------------------

class TestUniverseOptimizerIntegration:
    """Tests for integration with Universe Optimizer."""

    def test_universe_optimizer_config_has_cross_pair_bias(self):
        """Test that Universe Optimizer config has cross-pair bias settings."""
        from execution.universe_optimizer import UniverseOptimizerConfig
        import dataclasses
        
        fields = {f.name for f in dataclasses.fields(UniverseOptimizerConfig)}
        
        assert "use_cross_pair_bias" in fields, "Missing use_cross_pair_bias field"
        assert "cross_pair_bias_max" in fields, "Missing cross_pair_bias_max field"
        assert "cross_pair_bias_threshold" in fields, "Missing cross_pair_bias_threshold field"

    def test_universe_optimizer_config_defaults(self):
        """Test that cross-pair bias config has sensible defaults."""
        from execution.universe_optimizer import UniverseOptimizerConfig
        
        config = UniverseOptimizerConfig()
        assert config.use_cross_pair_bias is False  # Disabled by default
        assert config.cross_pair_bias_max >= 0
        assert config.cross_pair_bias_threshold >= 0


# ---------------------------------------------------------------------------
# Test: Dashboard Loader
# ---------------------------------------------------------------------------

class TestDashboardLoader:
    """Tests for dashboard state loader."""

    def test_load_cross_pair_state_function_exists(self):
        """Test that load_cross_pair_state function exists in dashboard."""
        from dashboard.edge_discovery_panel import load_cross_pair_state
        
        assert callable(load_cross_pair_state)

    def test_load_cross_pair_state_returns_dict(self):
        """Test that loader returns dict."""
        from dashboard.edge_discovery_panel import load_cross_pair_state
        
        result = load_cross_pair_state()
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Test: Cross-Pair Engine Module Exports
# ---------------------------------------------------------------------------

class TestModuleExports:
    """Tests for cross-pair engine module exports."""

    def test_exports_dataclasses(self):
        """Test that dataclasses are exported."""
        from execution.cross_pair_engine import (
            CrossPairConfig,
            PairStats,
            PairEdge,
            CrossPairState,
        )
        
        import dataclasses
        assert dataclasses.is_dataclass(CrossPairConfig)
        assert dataclasses.is_dataclass(PairStats)
        assert dataclasses.is_dataclass(PairEdge)
        assert dataclasses.is_dataclass(CrossPairState)

    def test_exports_functions(self):
        """Test that key functions are exported."""
        from execution.cross_pair_engine import (
            load_cross_pair_config,
            compute_ols_hedge_ratio,
            compute_spread_series,
            compute_correlation,
            compute_half_life,
            compute_pair_stats,
            compute_pair_edge,
            run_cross_pair_scan,
            save_cross_pair_state,
            load_cross_pair_state,
        )
        
        assert callable(load_cross_pair_config)
        assert callable(compute_ols_hedge_ratio)
        assert callable(compute_spread_series)
        assert callable(compute_correlation)
        assert callable(compute_half_life)
        assert callable(compute_pair_stats)
        assert callable(compute_pair_edge)
        assert callable(run_cross_pair_scan)
        assert callable(save_cross_pair_state)
        assert callable(load_cross_pair_state)
