"""
Integration tests for Alpha Decay & Survival Curves (v7.8_P7).

Tests state file schema, manifest alignment, and integration hooks.
"""
import json
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# State Schema Tests
# ---------------------------------------------------------------------------


class TestAlphaDecayStateSchema:
    """Tests for alpha_decay.json schema compliance."""

    def test_state_schema_required_fields(self):
        """Test that state has all required fields."""
        from execution.alpha_decay import AlphaDecayState
        
        state = AlphaDecayState(
            updated_ts="2024-01-01T00:00:00+00:00",
            cycle_count=1,
        )
        d = state.to_dict()
        
        # Required fields from manifest
        required = [
            "updated_ts",
            "cycle_count",
            "symbols",
            "categories",
            "factors",
            "avg_symbol_survival",
            "avg_category_survival",
            "avg_factor_survival",
            "overall_alpha_health",
            "weakest_symbols",
            "strongest_symbols",
            "weakest_categories",
            "weakest_factors",
            "meta",
        ]
        
        for field in required:
            assert field in d, f"Missing required field: {field}"

    def test_symbol_stats_schema(self):
        """Test SymbolDecayStats schema."""
        from execution.alpha_decay import SymbolDecayStats
        
        stats = SymbolDecayStats(
            symbol="BTCUSDT",
            decay_rate=-0.02,
            half_life=35,
            survival_prob=0.6,
            deterioration_prob=0.4,
            ema_edge_score=0.5,
        )
        d = stats.to_dict()
        
        required = [
            "symbol",
            "decay_rate",
            "half_life",
            "survival_prob",
            "deterioration_prob",
            "ema_edge_score",
            "sample_count",
            "last_edge_score",
            "trend_direction",
            "days_since_peak",
        ]
        
        for field in required:
            assert field in d, f"Missing SymbolDecayStats field: {field}"

    def test_category_stats_schema(self):
        """Test CategoryDecayStats schema."""
        from execution.alpha_decay import CategoryDecayStats
        
        stats = CategoryDecayStats(
            category="L1",
            decay_rate=-0.01,
            half_life=70,
            survival_prob=0.8,
            deterioration_prob=0.2,
        )
        d = stats.to_dict()
        
        required = [
            "category",
            "decay_rate",
            "half_life",
            "survival_prob",
            "deterioration_prob",
            "symbol_count",
            "avg_symbol_survival",
            "weakest_symbol",
            "strongest_symbol",
        ]
        
        for field in required:
            assert field in d, f"Missing CategoryDecayStats field: {field}"

    def test_factor_stats_schema(self):
        """Test FactorDecayStats schema."""
        from execution.alpha_decay import FactorDecayStats
        
        stats = FactorDecayStats(
            factor="trend",
            decay_rate=-0.01,
            survival_prob=0.7,
            adjusted_factor_weight_multiplier=0.95,
        )
        d = stats.to_dict()
        
        required = [
            "factor",
            "decay_rate",
            "survival_prob",
            "adjusted_factor_weight_multiplier",
            "pnl_contribution",
            "ir_rolling",
            "trend_direction",
            "days_positive",
        ]
        
        for field in required:
            assert field in d, f"Missing FactorDecayStats field: {field}"


# ---------------------------------------------------------------------------
# Manifest Alignment Tests
# ---------------------------------------------------------------------------


class TestAlphaDecayManifest:
    """Tests for v7_manifest.json alignment."""

    def test_alpha_decay_in_manifest(self):
        """Test alpha_decay is registered in manifest."""
        manifest_path = Path("v7_manifest.json")
        if not manifest_path.exists():
            pytest.skip("v7_manifest.json not found")
        
        manifest = json.loads(manifest_path.read_text())
        state_files = manifest.get("state_files", {})
        
        assert "alpha_decay" in state_files, "alpha_decay not in manifest"

    def test_alpha_decay_manifest_fields(self):
        """Test alpha_decay manifest entry has required fields."""
        manifest_path = Path("v7_manifest.json")
        if not manifest_path.exists():
            pytest.skip("v7_manifest.json not found")
        
        manifest = json.loads(manifest_path.read_text())
        entry = manifest.get("state_files", {}).get("alpha_decay", {})
        
        assert entry.get("path") == "logs/state/alpha_decay.json"
        assert entry.get("owner") == "executor"
        assert "fields" in entry


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------


class TestAlphaDecayConfig:
    """Tests for strategy_config.json alpha_decay block."""

    def test_config_block_exists(self):
        """Test alpha_decay block exists in strategy_config."""
        config_path = Path("config/strategy_config.json")
        if not config_path.exists():
            pytest.skip("strategy_config.json not found")
        
        config = json.loads(config_path.read_text())
        
        assert "alpha_decay" in config, "alpha_decay block not in strategy_config"

    def test_config_block_fields(self):
        """Test alpha_decay config has expected fields."""
        config_path = Path("config/strategy_config.json")
        if not config_path.exists():
            pytest.skip("strategy_config.json not found")
        
        config = json.loads(config_path.read_text())
        decay_cfg = config.get("alpha_decay", {})
        
        expected = [
            "enabled",
            "lookback_days",
            "min_samples",
            "smoothing_alpha",
            "symbol_half_life_floor",
            "symbol_half_life_ceiling",
            "decay_penalty_strength",
            "sentinel_x_integration",
        ]
        
        for field in expected:
            assert field in decay_cfg, f"Missing config field: {field}"

    def test_config_disabled_by_default(self):
        """Test alpha_decay is disabled by default."""
        config_path = Path("config/strategy_config.json")
        if not config_path.exists():
            pytest.skip("strategy_config.json not found")
        
        config = json.loads(config_path.read_text())
        decay_cfg = config.get("alpha_decay", {})
        
        assert decay_cfg.get("enabled") is False


# ---------------------------------------------------------------------------
# Integration Hook Tests
# ---------------------------------------------------------------------------


class TestUniverseOptimizerIntegration:
    """Tests for Universe Optimizer integration."""

    def test_get_alpha_decay_symbol_penalty_exists(self):
        """Test penalty function exists in universe_optimizer."""
        from execution.universe_optimizer import get_alpha_decay_symbol_penalty
        
        # Should return 1.0 for None state
        result = get_alpha_decay_symbol_penalty("BTCUSDT", None)
        assert result == 1.0

    def test_compute_symbol_composite_score_accepts_decay_state(self):
        """Test composite score function accepts alpha_decay_state."""
        from execution.universe_optimizer import (
            compute_symbol_composite_score,
            UniverseOptimizerConfig,
        )
        
        cfg = UniverseOptimizerConfig()
        score = compute_symbol_composite_score(
            symbol="BTCUSDT",
            symbol_edges={},
            category_edges={},
            meta_overlay=1.0,
            strategy_health=0.5,
            allocation_confidence=0.5,
            cfg=cfg,
            alpha_decay_state=None,  # Should accept this param
        )
        
        assert isinstance(score, float)
        assert 0 <= score <= 1


class TestAlphaRouterIntegration:
    """Tests for Alpha Router integration."""

    def test_get_alpha_decay_router_adjustment_exists(self):
        """Test adjustment function exists in alpha_router."""
        from execution.alpha_router import get_alpha_decay_router_adjustment
        
        # Should return 1.0 for None state
        result = get_alpha_decay_router_adjustment(None)
        assert result == 1.0

    def test_adjustment_with_state(self):
        """Test adjustment with actual state."""
        from execution.alpha_router import get_alpha_decay_router_adjustment
        
        state = {"avg_symbol_survival": 0.7}
        result = get_alpha_decay_router_adjustment(state)
        
        # 0.8 + 0.2 * 0.7 = 0.94
        assert abs(result - 0.94) < 0.01


class TestConvictionEngineIntegration:
    """Tests for Conviction Engine integration."""

    def test_conviction_context_has_decay_field(self):
        """Test ConvictionContext has alpha_decay_adjustment field."""
        from execution.conviction_engine import ConvictionContext
        
        ctx = ConvictionContext()
        assert hasattr(ctx, "alpha_decay_adjustment")
        assert ctx.alpha_decay_adjustment == 1.0  # Default

    def test_regime_modifiers_include_decay(self):
        """Test regime modifiers include alpha_decay_adjustment."""
        from execution.conviction_engine import (
            ConvictionContext,
            compute_conviction_score_with_regime,
        )
        
        ctx = ConvictionContext(
            hybrid_score=0.6,
            alpha_decay_adjustment=0.95,
        )
        
        score, modifiers = compute_conviction_score_with_regime(ctx)
        
        assert "alpha_decay_adjustment" in modifiers
        assert modifiers["alpha_decay_adjustment"] == 0.95


class TestFactorDiagnosticsIntegration:
    """Tests for Factor Diagnostics integration."""

    def test_factor_weights_snapshot_has_decay_fields(self):
        """Test FactorWeightsSnapshot has alpha_decay fields."""
        from execution.factor_diagnostics import FactorWeightsSnapshot
        
        snapshot = FactorWeightsSnapshot()
        
        assert hasattr(snapshot, "alpha_decay_overlay_enabled")
        assert hasattr(snapshot, "alpha_decay_multipliers")

    def test_factor_weights_to_dict_includes_decay(self):
        """Test to_dict includes alpha_decay fields."""
        from execution.factor_diagnostics import FactorWeightsSnapshot
        
        snapshot = FactorWeightsSnapshot(
            alpha_decay_overlay_enabled=True,
            alpha_decay_multipliers={"trend": 0.95},
        )
        d = snapshot.to_dict()
        
        assert d["alpha_decay_overlay_enabled"] is True
        assert d["alpha_decay_multipliers"]["trend"] == 0.95


class TestEdgeScannerIntegration:
    """Tests for EdgeScanner integration."""

    def test_edge_insights_has_decay_summary(self):
        """Test EdgeInsights has alpha_decay_summary field."""
        from execution.edge_scanner import EdgeInsights
        
        insights = EdgeInsights()
        assert hasattr(insights, "alpha_decay_summary")

    def test_edge_insights_to_dict_includes_decay(self):
        """Test to_dict includes alpha_decay_summary when set."""
        from execution.edge_scanner import EdgeInsights
        
        insights = EdgeInsights(
            alpha_decay_summary={
                "overall_alpha_health": 0.7,
                "weakest_symbols": ["XYZUSDT"],
            }
        )
        d = insights.to_dict()
        
        assert "alpha_decay_summary" in d
        assert d["alpha_decay_summary"]["overall_alpha_health"] == 0.7

    def test_update_alpha_decay_function_exists(self):
        """Test update_alpha_decay function exists in edge_scanner."""
        from execution.edge_scanner import update_alpha_decay, load_alpha_decay_state
        
        # Functions should be importable
        assert callable(update_alpha_decay)
        assert callable(load_alpha_decay_state)


# ---------------------------------------------------------------------------
# Disabled Behavior Tests
# ---------------------------------------------------------------------------


class TestDisabledBehavior:
    """Tests to verify no behavior change when disabled."""

    def test_run_alpha_decay_full_returns_none_when_disabled(self):
        """Test run_alpha_decay_full returns None when disabled."""
        from unittest.mock import patch
        from execution.alpha_decay import run_alpha_decay_full, AlphaDecayConfig
        
        disabled_cfg = AlphaDecayConfig(enabled=False)
        
        with patch("execution.alpha_decay.load_config_from_strategy", return_value=disabled_cfg):
            result = run_alpha_decay_full()
        
        assert result is None

    def test_symbol_penalty_no_effect_when_disabled(self):
        """Test symbol penalty returns 1.0 when disabled."""
        from execution.alpha_decay import get_symbol_decay_penalty, AlphaDecayConfig
        
        cfg = AlphaDecayConfig(enabled=False)
        penalty = get_symbol_decay_penalty("BTCUSDT", None, cfg)
        
        assert penalty == 1.0

    def test_router_adjustment_no_effect_when_disabled(self):
        """Test router adjustment returns 1.0 when disabled."""
        from execution.alpha_decay import get_alpha_router_adjustment, AlphaDecayConfig
        
        cfg = AlphaDecayConfig(enabled=False)
        adj = get_alpha_router_adjustment(None, cfg)
        
        assert adj == 1.0

    def test_conviction_adjustment_no_effect_when_disabled(self):
        """Test conviction adjustment returns 1.0 when disabled."""
        from execution.alpha_decay import get_conviction_decay_adjustment, AlphaDecayConfig
        
        cfg = AlphaDecayConfig(enabled=False)
        adj = get_conviction_decay_adjustment(None, cfg)
        
        assert adj == 1.0

    def test_factor_multipliers_empty_when_disabled(self):
        """Test factor multipliers returns empty dict when disabled."""
        from execution.alpha_decay import get_factor_decay_multipliers, AlphaDecayConfig
        
        cfg = AlphaDecayConfig(enabled=False)
        mults = get_factor_decay_multipliers(None, cfg)
        
        assert mults == {}


# ---------------------------------------------------------------------------
# Run Step Integration Test
# ---------------------------------------------------------------------------


class TestRunAlphaDecayStep:
    """Tests for full run_alpha_decay_step."""

    def test_run_step_produces_valid_state(self):
        """Test run_alpha_decay_step produces valid state."""
        from execution.alpha_decay import (
            run_alpha_decay_step,
            AlphaDecayConfig,
            AlphaDecayHistory,
        )
        
        cfg = AlphaDecayConfig(
            enabled=True,
            min_samples=5,  # Lower for testing
        )
        history = AlphaDecayHistory()
        
        # Add some history
        import time
        base_time = time.time() - 30 * 86400  # 30 days ago
        for i in range(20):
            history.add_symbol_edge("BTCUSDT", 0.5 - i * 0.01, base_time + i * 86400)
            history.add_factor_pnl("trend", 100 - i * 2, base_time + i * 86400)
        
        state = run_alpha_decay_step(
            config=cfg,
            history=history,
            prev_state=None,
            edge_insights=None,
            factor_diagnostics=None,
            hybrid_scores=None,
            sentinel_x_state=None,
            symbol_to_category=None,
        )
        
        assert state.updated_ts != ""
        assert state.cycle_count == 1
        assert isinstance(state.avg_symbol_survival, float)
        assert isinstance(state.overall_alpha_health, float)
