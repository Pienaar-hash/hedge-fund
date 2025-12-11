"""
Integration tests for Cerberus Multi-Strategy Portfolio Router — v7.8_P8
"""
import json
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# State File Schema Tests
# ---------------------------------------------------------------------------


class TestCerberusStateSchema:
    """Tests for Cerberus state file schema."""

    def test_state_file_schema(self) -> None:
        """Test that state file has correct schema."""
        from execution.cerberus_router import (
            CerberusConfig,
            run_cerberus_step,
            write_cerberus_state,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cerberus_state.json"

            cfg = CerberusConfig(enabled=True)
            state = run_cerberus_step(cfg)
            write_cerberus_state(state, path)

            # Load and verify schema
            data = json.loads(path.read_text())

            # Required top-level fields
            assert "updated_ts" in data
            assert "cycle_count" in data
            assert "head_state" in data
            assert "regime" in data
            assert "overall_health" in data
            assert "avg_decay_survival" in data
            assert "notes" in data

            # head_state structure
            head_state = data["head_state"]
            assert "heads" in head_state
            assert "mean_multiplier" in head_state
            assert "normalized" in head_state

            # Each head has required fields
            for head_name, metrics in head_state["heads"].items():
                assert "multiplier" in metrics
                assert "ema_score" in metrics
                assert "signal_score" in metrics
                assert "trend_direction" in metrics

    def test_state_values_bounded(self) -> None:
        """Test that state values are within expected bounds."""
        from execution.cerberus_router import (
            CerberusConfig,
            run_cerberus_step,
        )

        cfg = CerberusConfig(enabled=True)
        state = run_cerberus_step(cfg)

        # Check multipliers are within bounds
        for head, metrics in state.head_state.heads.items():
            assert 0.0 <= metrics.multiplier <= 3.0
            assert 0.0 <= metrics.ema_score <= 1.0
            assert 0.0 <= metrics.signal_score <= 1.0

        # Check health and survival in [0, 1]
        assert 0.0 <= state.overall_health <= 1.0
        assert 0.0 <= state.avg_decay_survival <= 1.0


# ---------------------------------------------------------------------------
# Manifest Alignment Tests
# ---------------------------------------------------------------------------


class TestManifestAlignment:
    """Tests for v7_manifest.json alignment."""

    def test_manifest_contains_cerberus_entry(self) -> None:
        """Test that manifest has cerberus_state entry."""
        manifest_path = Path("v7_manifest.json")
        if not manifest_path.exists():
            pytest.skip("v7_manifest.json not found")

        manifest = json.loads(manifest_path.read_text())
        state_files = manifest.get("state_files", {})

        assert "cerberus_state" in state_files

        entry = state_files["cerberus_state"]
        assert entry.get("path") == "logs/state/cerberus_state.json"
        assert entry.get("owner") == "executor"
        assert entry.get("optional") is True

        # Check documented fields
        fields = entry.get("fields", {})
        assert "updated_ts" in fields
        assert "head_state" in fields
        assert "regime" in fields


# ---------------------------------------------------------------------------
# Config Integration Tests
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    """Tests for config file integration."""

    def test_config_block_exists(self) -> None:
        """Test that cerberus_router block exists in strategy_config.json."""
        config_path = Path("config/strategy_config.json")
        if not config_path.exists():
            pytest.skip("strategy_config.json not found")

        config = json.loads(config_path.read_text())
        assert "cerberus_router" in config

        cerberus_cfg = config["cerberus_router"]
        assert "enabled" in cerberus_cfg
        assert "strategy_heads" in cerberus_cfg
        assert "learning_rate" in cerberus_cfg
        assert "bounds" in cerberus_cfg
        assert "regime_weights" in cerberus_cfg

    def test_config_loads_correctly(self) -> None:
        """Test that config loads from file correctly."""
        from execution.cerberus_router import load_cerberus_config

        config_path = Path("config/strategy_config.json")
        if not config_path.exists():
            pytest.skip("strategy_config.json not found")

        cfg = load_cerberus_config(config_path)
        assert cfg is not None
        # Default is disabled
        assert cfg.enabled is False
        # All heads should exist
        assert len(cfg.strategy_heads) == 6


# ---------------------------------------------------------------------------
# Downstream Integration Tests
# ---------------------------------------------------------------------------


class TestFactorDiagnosticsIntegration:
    """Tests for factor diagnostics integration."""

    def test_factor_weights_snapshot_has_cerberus_fields(self) -> None:
        """Test that FactorWeightsSnapshot has cerberus overlay fields."""
        from execution.factor_diagnostics import FactorWeightsSnapshot

        snapshot = FactorWeightsSnapshot()
        assert hasattr(snapshot, "cerberus_overlay_enabled")
        assert hasattr(snapshot, "cerberus_overlay")

    def test_factor_weights_snapshot_to_dict(self) -> None:
        """Test that cerberus fields are serialized."""
        from execution.factor_diagnostics import FactorWeightsSnapshot

        snapshot = FactorWeightsSnapshot(
            cerberus_overlay_enabled=True,
            cerberus_overlay={"trend": 1.1, "carry": 0.9},
        )
        d = snapshot.to_dict()
        assert d["cerberus_overlay_enabled"] is True
        assert d["cerberus_overlay"]["trend"] == 1.1


class TestConvictionEngineIntegration:
    """Tests for conviction engine integration."""

    def test_conviction_context_has_cerberus_field(self) -> None:
        """Test that ConvictionContext has cerberus multiplier field."""
        from execution.conviction_engine import ConvictionContext

        ctx = ConvictionContext()
        assert hasattr(ctx, "cerberus_conviction_multiplier")
        assert ctx.cerberus_conviction_multiplier == 1.0

    def test_conviction_context_applies_multiplier(self) -> None:
        """Test that cerberus multiplier is applied in scoring."""
        from execution.conviction_engine import (
            ConvictionConfig,
            ConvictionContext,
            compute_conviction_score_with_regime,
        )

        cfg = ConvictionConfig(enabled=True)

        # Base context
        ctx_base = ConvictionContext(
            hybrid_score=0.7,
            expectancy_alpha=0.6,
            router_quality=0.8,
            trend_strength=0.5,
            cerberus_conviction_multiplier=1.0,
        )
        score_base, _ = compute_conviction_score_with_regime(ctx_base, cfg)

        # Context with boosted cerberus multiplier
        ctx_boosted = ConvictionContext(
            hybrid_score=0.7,
            expectancy_alpha=0.6,
            router_quality=0.8,
            trend_strength=0.5,
            cerberus_conviction_multiplier=1.2,
        )
        score_boosted, mods = compute_conviction_score_with_regime(ctx_boosted, cfg)

        assert score_boosted >= score_base
        assert "cerberus_conviction_multiplier" in mods


class TestUniverseOptimizerIntegration:
    """Tests for universe optimizer integration."""

    def test_compute_symbol_score_accepts_cerberus_state(self) -> None:
        """Test that compute_symbol_composite_score accepts cerberus_state."""
        from execution.universe_optimizer import (
            UniverseOptimizerConfig,
            compute_symbol_composite_score,
        )

        cfg = UniverseOptimizerConfig()
        
        score = compute_symbol_composite_score(
            symbol="BTCUSDT",
            symbol_edges={},
            category_edges={},
            meta_overlay=1.0,
            strategy_health=0.7,
            allocation_confidence=0.8,
            cfg=cfg,
            alpha_decay_state=None,
            cerberus_state=None,
        )
        assert 0.0 <= score <= 1.0

    def test_cerberus_category_multiplier_applied(self) -> None:
        """Test that Cerberus category multiplier affects scores."""
        from execution.universe_optimizer import (
            UniverseOptimizerConfig,
            compute_symbol_composite_score,
        )

        cfg = UniverseOptimizerConfig()

        # Without Cerberus
        score_base = compute_symbol_composite_score(
            symbol="BTCUSDT",
            symbol_edges={},
            category_edges={},
            meta_overlay=1.0,
            strategy_health=0.7,
            allocation_confidence=0.8,
            cfg=cfg,
            cerberus_state=None,
        )

        # With boosted Cerberus CATEGORY head
        cerberus_state = {
            "head_state": {
                "heads": {
                    "CATEGORY": {"multiplier": 1.5},
                }
            }
        }
        score_boosted = compute_symbol_composite_score(
            symbol="BTCUSDT",
            symbol_edges={},
            category_edges={},
            meta_overlay=1.0,
            strategy_health=0.7,
            allocation_confidence=0.8,
            cfg=cfg,
            cerberus_state=cerberus_state,
        )

        # Boosted should be >= base
        assert score_boosted >= score_base


class TestAlphaRouterIntegration:
    """Tests for alpha router integration."""

    def test_get_cerberus_router_adjustment(self) -> None:
        """Test Cerberus router adjustment function."""
        from execution.alpha_router import get_cerberus_router_adjustment

        # No state
        adj = get_cerberus_router_adjustment(None)
        assert adj == 1.0

        # With state
        cerberus_state = {
            "head_state": {
                "heads": {
                    "TREND": {"multiplier": 1.2},
                    "MEAN_REVERT": {"multiplier": 0.8},
                    "RELATIVE_VALUE": {"multiplier": 1.0},
                    "CATEGORY": {"multiplier": 1.1},
                    "VOL_HARVEST": {"multiplier": 0.9},
                    "EMERGENT_ALPHA": {"multiplier": 1.0},
                }
            }
        }
        adj = get_cerberus_router_adjustment(cerberus_state)
        # Weighted average should be close to 1.0
        assert 0.5 <= adj <= 1.5


class TestCrossfireIntegration:
    """Tests for Crossfire integration."""

    def test_get_cerberus_crossfire_multiplier(self) -> None:
        """Test Cerberus crossfire multiplier function."""
        from execution.cross_pair_engine import get_cerberus_crossfire_multiplier

        # No state
        mult = get_cerberus_crossfire_multiplier(None)
        assert mult == 1.0

        # With boosted RV head
        cerberus_state = {
            "head_state": {
                "heads": {
                    "RELATIVE_VALUE": {"multiplier": 1.4},
                }
            }
        }
        mult = get_cerberus_crossfire_multiplier(cerberus_state)
        assert mult == 1.4


# ---------------------------------------------------------------------------
# Disabled Behavior Tests
# ---------------------------------------------------------------------------


class TestDisabledBehavior:
    """Tests for disabled mode behavior."""

    def test_disabled_returns_neutral_multipliers(self) -> None:
        """Test that disabled config returns all 1.0 multipliers."""
        from execution.cerberus_router import (
            CerberusConfig,
            get_cerberus_all_multipliers,
        )

        cfg = CerberusConfig(enabled=False)
        mults = get_cerberus_all_multipliers(None, cfg)
        assert all(m == 1.0 for m in mults.values())

    def test_disabled_no_effect_on_factor_overlay(self) -> None:
        """Test that disabled config gives 1.0 factor overlay."""
        from execution.cerberus_router import (
            CerberusConfig,
            get_cerberus_factor_weight_overlay,
        )

        cfg = CerberusConfig(enabled=False)
        overlay = get_cerberus_factor_weight_overlay("trend", None, cfg)
        assert overlay == 1.0

    def test_disabled_no_effect_on_conviction(self) -> None:
        """Test that disabled config gives 1.0 conviction multiplier."""
        from execution.cerberus_router import (
            CerberusConfig,
            get_cerberus_conviction_multiplier,
        )

        cfg = CerberusConfig(enabled=False)
        mult = get_cerberus_conviction_multiplier(None, cfg)
        assert mult == 1.0


# ---------------------------------------------------------------------------
# Dashboard Panel Tests
# ---------------------------------------------------------------------------


class TestDashboardPanel:
    """Tests for dashboard panel."""

    def test_load_cerberus_state_missing(self) -> None:
        """Test loading state from missing file."""
        from dashboard.cerberus_panel import load_cerberus_state

        state = load_cerberus_state("/nonexistent/path.json")
        assert state == {}

    def test_load_cerberus_state_valid(self) -> None:
        """Test loading valid state file."""
        from dashboard.cerberus_panel import load_cerberus_state
        from execution.cerberus_router import (
            CerberusConfig,
            run_cerberus_step,
            write_cerberus_state,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cerberus_state.json"

            cfg = CerberusConfig(enabled=True)
            state = run_cerberus_step(cfg)
            write_cerberus_state(state, path)

            loaded = load_cerberus_state(path)
            assert "head_state" in loaded
            assert "regime" in loaded
