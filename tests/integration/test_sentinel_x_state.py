"""
Integration tests for Sentinel-X state surface and manifest alignment (v7.8_P6).

Tests cover:
- State file schema compliance
- Manifest alignment
- EdgeInsights integration
- Conviction engine integration
- Factor diagnostics integration
- Alpha Router integration
- Universe Optimizer integration
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_state_data():
    """Sample valid Sentinel-X state data."""
    return {
        "updated_ts": "2024-01-15T10:30:00Z",
        "cycle_count": 42,
        "primary_regime": "TREND_UP",
        "secondary_regime": "BREAKOUT",
        "regime_probs": {
            "TREND_UP": 0.45,
            "TREND_DOWN": 0.05,
            "MEAN_REVERT": 0.10,
            "BREAKOUT": 0.25,
            "CHOPPY": 0.10,
            "CRISIS": 0.05,
        },
        "smoothed_probs": {
            "TREND_UP": 0.42,
            "TREND_DOWN": 0.06,
            "MEAN_REVERT": 0.12,
            "BREAKOUT": 0.24,
            "CHOPPY": 0.11,
            "CRISIS": 0.05,
        },
        "features": {
            "returns_mean": 0.0012,
            "returns_std": 0.0045,
            "trend_slope": 0.0035,
            "trend_r2": 0.72,
            "vol_regime_z": 0.45,
        },
        "crisis_flag": False,
        "crisis_reason": "",
        "history_meta": {
            "last_n_labels": ["CHOPPY", "TREND_UP", "TREND_UP"],
            "consecutive_count": 0,
            "pending_regime": None,
            "last_primary": "TREND_UP",
        },
        "meta": {
            "model_type": "gradient_boosting",
            "lookback_bars": 240,
        },
        "notes": "",
        "errors": [],
    }


@pytest.fixture
def tmp_state_dir():
    """Create temporary directory for state files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ---------------------------------------------------------------------------
# Schema Compliance Tests
# ---------------------------------------------------------------------------


class TestSentinelXStateSchema:
    """Tests for Sentinel-X state schema compliance."""

    def test_state_has_required_fields(self, sample_state_data):
        """Test state has all required fields from manifest."""
        required_fields = [
            "updated_ts",
            "cycle_count",
            "primary_regime",
            "secondary_regime",
            "regime_probs",
            "smoothed_probs",
            "features",
            "crisis_flag",
            "crisis_reason",
            "history_meta",
            "meta",
            "notes",
            "errors",
        ]
        
        for field in required_fields:
            assert field in sample_state_data, f"Missing required field: {field}"

    def test_state_to_dict_serializable(self, sample_state_data, tmp_state_dir):
        """Test state can be serialized to JSON."""
        from execution.sentinel_x import SentinelXState, save_sentinel_x_state
        
        state = SentinelXState(
            updated_ts=sample_state_data["updated_ts"],
            cycle_count=sample_state_data["cycle_count"],
            primary_regime=sample_state_data["primary_regime"],
            secondary_regime=sample_state_data["secondary_regime"],
            regime_probs=sample_state_data["regime_probs"],
            smoothed_probs=sample_state_data["smoothed_probs"],
            features=sample_state_data["features"],
            crisis_flag=sample_state_data["crisis_flag"],
            crisis_reason=sample_state_data["crisis_reason"],
            history_meta=sample_state_data["history_meta"],
            meta=sample_state_data["meta"],
        )
        
        state_path = tmp_state_dir / "sentinel_x.json"
        assert save_sentinel_x_state(state, state_path) is True
        
        # Verify JSON is valid
        loaded = json.loads(state_path.read_text())
        assert loaded["primary_regime"] == "TREND_UP"

    def test_regime_probs_sum_to_one(self, sample_state_data):
        """Test regime probabilities sum to approximately 1."""
        probs = sample_state_data["regime_probs"]
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.05, f"Probs sum to {total}, expected ~1.0"

    def test_valid_regime_labels(self, sample_state_data):
        """Test regime labels are valid."""
        valid_regimes = {
            "TREND_UP", "TREND_DOWN", "MEAN_REVERT",
            "BREAKOUT", "CHOPPY", "CRISIS"
        }
        
        assert sample_state_data["primary_regime"] in valid_regimes
        if sample_state_data["secondary_regime"]:
            assert sample_state_data["secondary_regime"] in valid_regimes


# ---------------------------------------------------------------------------
# Manifest Alignment Tests
# ---------------------------------------------------------------------------


class TestManifestAlignment:
    """Tests for v7_manifest.json alignment."""

    def test_sentinel_x_in_manifest(self):
        """Test sentinel_x entry exists in manifest."""
        manifest_path = Path("v7_manifest.json")
        if not manifest_path.exists():
            pytest.skip("v7_manifest.json not found")
        
        manifest = json.loads(manifest_path.read_text())
        state_files = manifest.get("state_files", {})
        
        assert "sentinel_x" in state_files, "sentinel_x not in manifest"

    def test_manifest_path_correct(self):
        """Test manifest path matches expected."""
        manifest_path = Path("v7_manifest.json")
        if not manifest_path.exists():
            pytest.skip("v7_manifest.json not found")
        
        manifest = json.loads(manifest_path.read_text())
        sentinel_entry = manifest.get("state_files", {}).get("sentinel_x", {})
        
        expected_path = "logs/state/sentinel_x.json"
        assert sentinel_entry.get("path") == expected_path

    def test_manifest_fields_documented(self):
        """Test manifest has fields documentation."""
        manifest_path = Path("v7_manifest.json")
        if not manifest_path.exists():
            pytest.skip("v7_manifest.json not found")
        
        manifest = json.loads(manifest_path.read_text())
        sentinel_entry = manifest.get("state_files", {}).get("sentinel_x", {})
        
        fields = sentinel_entry.get("fields", {})
        assert "primary_regime" in fields
        assert "regime_probs" in fields
        assert "crisis_flag" in fields


# ---------------------------------------------------------------------------
# EdgeInsights Integration Tests
# ---------------------------------------------------------------------------


class TestEdgeInsightsIntegration:
    """Tests for EdgeInsights integration."""

    def test_edge_insights_has_sentinel_x_field(self):
        """Test EdgeInsights dataclass has sentinel_x field."""
        from execution.edge_scanner import EdgeInsights
        
        insights = EdgeInsights()
        assert hasattr(insights, "sentinel_x")
        assert insights.sentinel_x is None  # Optional field

    def test_edge_insights_to_dict_includes_sentinel_x(self):
        """Test to_dict includes sentinel_x when set."""
        from execution.edge_scanner import EdgeInsights
        
        sentinel_data = {
            "primary_regime": "TREND_UP",
            "regime_probs": {"TREND_UP": 0.7},
        }
        
        insights = EdgeInsights(sentinel_x=sentinel_data)
        result = insights.to_dict()
        
        assert "sentinel_x" in result
        assert result["sentinel_x"]["primary_regime"] == "TREND_UP"

    def test_update_sentinel_x_function_exists(self):
        """Test update_sentinel_x function is exported."""
        from execution.edge_scanner import update_sentinel_x
        
        assert callable(update_sentinel_x)


# ---------------------------------------------------------------------------
# Conviction Engine Integration Tests
# ---------------------------------------------------------------------------


class TestConvictionEngineIntegration:
    """Tests for conviction engine integration."""

    def test_conviction_context_has_sentinel_fields(self):
        """Test ConvictionContext has Sentinel-X fields."""
        from execution.conviction_engine import ConvictionContext
        
        ctx = ConvictionContext()
        assert hasattr(ctx, "sentinel_x_weight")
        assert hasattr(ctx, "sentinel_x_regime")
        assert ctx.sentinel_x_weight == 1.0  # Default neutral

    def test_conviction_with_sentinel_weight(self):
        """Test conviction computation with Sentinel-X weight."""
        from execution.conviction_engine import (
            ConvictionContext,
            ConvictionConfig,
            compute_conviction_score_with_regime,
            clear_conviction_cache,
        )
        
        clear_conviction_cache()
        
        cfg = ConvictionConfig(enabled=True)
        
        # Without Sentinel-X weight
        ctx1 = ConvictionContext(
            hybrid_score=0.7,
            router_quality=0.8,
            sentinel_x_weight=1.0,
        )
        score1, modifiers1 = compute_conviction_score_with_regime(ctx1, cfg)
        
        clear_conviction_cache()
        
        # With Sentinel-X boost
        ctx2 = ConvictionContext(
            hybrid_score=0.7,
            router_quality=0.8,
            sentinel_x_weight=1.10,
            sentinel_x_regime="TREND_UP",
        )
        score2, modifiers2 = compute_conviction_score_with_regime(ctx2, cfg)
        
        # Score with boost should be higher
        assert score2 > score1
        assert modifiers2["sentinel_x_weight"] == 1.10
        assert modifiers2["sentinel_x_regime"] == "TREND_UP"

    def test_get_sentinel_x_conviction_weight(self):
        """Test helper function exists and works."""
        from execution.conviction_engine import get_sentinel_x_conviction_weight
        
        # Without state file, should return (1.0, "")
        weight, regime = get_sentinel_x_conviction_weight()
        assert isinstance(weight, float)
        assert isinstance(regime, str)


# ---------------------------------------------------------------------------
# Factor Diagnostics Integration Tests
# ---------------------------------------------------------------------------


class TestFactorDiagnosticsIntegration:
    """Tests for factor diagnostics integration."""

    def test_factor_weights_snapshot_has_sentinel_fields(self):
        """Test FactorWeightsSnapshot has Sentinel-X fields."""
        from execution.factor_diagnostics import FactorWeightsSnapshot
        
        snapshot = FactorWeightsSnapshot()
        assert hasattr(snapshot, "sentinel_x_overlay_enabled")
        assert hasattr(snapshot, "sentinel_x_overlay")
        assert hasattr(snapshot, "sentinel_x_regime")

    def test_factor_weights_to_dict_includes_sentinel(self):
        """Test to_dict includes Sentinel-X fields."""
        from execution.factor_diagnostics import FactorWeightsSnapshot
        
        snapshot = FactorWeightsSnapshot(
            sentinel_x_overlay_enabled=True,
            sentinel_x_overlay={"trend": 1.15, "momentum": 1.10},
            sentinel_x_regime="TREND_UP",
        )
        
        result = snapshot.to_dict()
        assert result["sentinel_x_overlay_enabled"] is True
        assert result["sentinel_x_regime"] == "TREND_UP"
        assert "trend" in result["sentinel_x_overlay"]


# ---------------------------------------------------------------------------
# Alpha Router Integration Tests
# ---------------------------------------------------------------------------


class TestAlphaRouterIntegration:
    """Tests for Alpha Router integration."""

    def test_compute_sentinel_x_factor(self):
        """Test Sentinel-X factor computation."""
        from execution.alpha_router import _compute_sentinel_x_factor, AlphaRouterConfig
        
        cfg = AlphaRouterConfig()
        
        # With state
        state = {"primary_regime": "TREND_UP"}
        factor, regime = _compute_sentinel_x_factor(state, cfg)
        assert factor == 1.05
        assert regime == "TREND_UP"
        
        # Crisis regime
        state = {"primary_regime": "CRISIS"}
        factor, regime = _compute_sentinel_x_factor(state, cfg)
        assert factor == 0.60
        assert regime == "CRISIS"
        
        # No state
        factor, regime = _compute_sentinel_x_factor(None, cfg)
        assert factor == 1.0
        assert regime == ""

    def test_compute_target_allocation_with_sentinel(self):
        """Test target allocation includes Sentinel-X factor."""
        from execution.alpha_router import (
            compute_target_allocation,
            AlphaRouterConfig,
        )
        
        cfg = AlphaRouterConfig(enabled=True)
        health = {"health_score": 0.7}
        
        # Without Sentinel-X
        state1 = compute_target_allocation(
            health=health,
            meta_state=None,
            router_quality=0.8,
            vol_regime="NORMAL",
            dd_state="NORMAL",
            cfg=cfg,
            sentinel_x_state=None,
        )
        
        # With CRISIS Sentinel-X
        state2 = compute_target_allocation(
            health=health,
            meta_state=None,
            router_quality=0.8,
            vol_regime="NORMAL",
            dd_state="NORMAL",
            cfg=cfg,
            sentinel_x_state={"primary_regime": "CRISIS"},
        )
        
        # CRISIS should reduce allocation
        assert state2.target_allocation < state1.target_allocation
        assert state2.raw_components["sentinel_x_factor"] == 0.60
        assert state2.raw_components["sentinel_x_regime"] == "CRISIS"


# ---------------------------------------------------------------------------
# Universe Optimizer Integration Tests
# ---------------------------------------------------------------------------


class TestUniverseOptimizerIntegration:
    """Tests for Universe Optimizer integration."""

    def test_effective_max_size_with_sentinel(self):
        """Test effective max size includes Sentinel-X shrink."""
        from execution.universe_optimizer import (
            _compute_effective_max_size,
            UniverseOptimizerConfig,
        )
        
        cfg = UniverseOptimizerConfig(
            min_universe_size=4,
            max_universe_size=20,
        )
        
        # Without Sentinel-X
        size1 = _compute_effective_max_size(
            base_max=20,
            vol_regime="NORMAL",
            dd_state="NORMAL",
            cfg=cfg,
            sentinel_x_state=None,
        )
        
        # With CRISIS Sentinel-X
        size2 = _compute_effective_max_size(
            base_max=20,
            vol_regime="NORMAL",
            dd_state="NORMAL",
            cfg=cfg,
            sentinel_x_state={"primary_regime": "CRISIS"},
        )
        
        # CRISIS should shrink universe
        assert size2 < size1

    def test_compute_optimized_universe_signature(self):
        """Test compute_optimized_universe accepts sentinel_x_state."""
        from execution.universe_optimizer import compute_optimized_universe
        import inspect
        
        sig = inspect.signature(compute_optimized_universe)
        params = list(sig.parameters.keys())
        
        assert "sentinel_x_state" in params


# ---------------------------------------------------------------------------
# Strategy Config Integration Tests
# ---------------------------------------------------------------------------


class TestStrategyConfigIntegration:
    """Tests for strategy_config.json integration."""

    def test_sentinel_x_in_strategy_config(self):
        """Test sentinel_x block exists in strategy_config.json."""
        config_path = Path("config/strategy_config.json")
        if not config_path.exists():
            pytest.skip("strategy_config.json not found")
        
        config = json.loads(config_path.read_text())
        
        assert "sentinel_x" in config, "sentinel_x block not in strategy_config.json"

    def test_sentinel_x_config_fields(self):
        """Test sentinel_x config has expected fields."""
        config_path = Path("config/strategy_config.json")
        if not config_path.exists():
            pytest.skip("strategy_config.json not found")
        
        config = json.loads(config_path.read_text())
        sentinel_cfg = config.get("sentinel_x", {})
        
        # Check key fields exist
        assert "enabled" in sentinel_cfg
        assert "lookback_bars" in sentinel_cfg
        assert "regimes" in sentinel_cfg
        assert "prob_thresholds" in sentinel_cfg
        assert "crisis_hard_rules" in sentinel_cfg
        assert "conviction_weights" in sentinel_cfg
        assert "factor_weights" in sentinel_cfg

    def test_sentinel_x_disabled_by_default(self):
        """Test sentinel_x is disabled by default."""
        config_path = Path("config/strategy_config.json")
        if not config_path.exists():
            pytest.skip("strategy_config.json not found")
        
        config = json.loads(config_path.read_text())
        sentinel_cfg = config.get("sentinel_x", {})
        
        assert sentinel_cfg.get("enabled") is False, "sentinel_x should be disabled by default"
