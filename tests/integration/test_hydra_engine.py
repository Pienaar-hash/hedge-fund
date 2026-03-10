"""
Integration tests for Hydra Multi-Strategy Execution Engine — v7.9_P1

Tests cover:
- State file schema and manifest alignment
- Config integration with strategy_config.json
- Executor integration hooks
- Dashboard panel rendering
- Logging to JSONL
- Disabled behavior (v7.8 baseline)
"""

import json
import pytest
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch, MagicMock

from execution.hydra_engine import (
    STRATEGY_HEADS,
    HydraConfig,
    HydraState,
    HydraMergedIntent,
    load_hydra_config,
    load_hydra_state,
    save_hydra_state,
    log_hydra_intents,
    run_hydra_step,
    is_hydra_enabled,
    hydra_merged_intent_to_execution_intent,
)

from execution.hydra_integration import (
    run_hydra_pipeline,
    convert_hydra_intents_to_execution,
    get_hydra_attribution_for_order,
    merge_with_single_strategy_intents,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def real_strategy_config() -> Dict[str, Any]:
    """Load real strategy_config.json."""
    cfg_path = Path("config/strategy_config.json")
    if cfg_path.exists():
        return json.loads(cfg_path.read_text())
    return {}


@pytest.fixture
def v7_manifest() -> Dict[str, Any]:
    """Load v7_manifest.json."""
    manifest_path = Path("v7_manifest.json")
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    return {}


@pytest.fixture
def tmp_state_path(tmp_path: Path) -> Path:
    """Create temporary state path."""
    return tmp_path / "hydra_state.json"


@pytest.fixture
def tmp_log_path(tmp_path: Path) -> Path:
    """Create temporary log path."""
    return tmp_path / "hydra_intents.jsonl"


# ---------------------------------------------------------------------------
# Test: State Schema
# ---------------------------------------------------------------------------


class TestHydraStateSchema:
    """Tests for hydra_state.json schema compliance."""

    def test_state_has_required_fields(self, tmp_state_path: Path):
        """Test that saved state has all required fields."""
        state = HydraState(
            updated_ts="2024-01-01T00:00:00Z",
            head_budgets={h: 0.30 for h in STRATEGY_HEADS},
            head_usage={h: 0.15 for h in STRATEGY_HEADS},
            head_positions={h: 5 for h in STRATEGY_HEADS},
            merged_intents=[{"symbol": "BTCUSDT", "net_side": "long", "nav_pct": 0.05}],
            cycle_count=10,
            notes=["Test note"],
            errors=[],
            meta={"test": True},
        )

        save_hydra_state(state, tmp_state_path)
        data = json.loads(tmp_state_path.read_text())

        # Required fields per manifest
        assert "updated_ts" in data
        assert "head_budgets" in data
        assert "head_usage" in data
        assert "head_positions" in data
        assert "merged_intents" in data
        assert "cycle_count" in data
        assert "notes" in data
        assert "errors" in data
        assert "meta" in data

    def test_state_types_are_correct(self, tmp_state_path: Path):
        """Test that state field types are correct."""
        state = HydraState(
            updated_ts="2024-01-01T00:00:00Z",
            head_budgets={"TREND": 0.50},
            head_usage={"TREND": 0.25},
            head_positions={"TREND": 5},
            merged_intents=[],
            cycle_count=10,
        )

        save_hydra_state(state, tmp_state_path)
        data = json.loads(tmp_state_path.read_text())

        assert isinstance(data["updated_ts"], str)
        assert isinstance(data["head_budgets"], dict)
        assert isinstance(data["head_usage"], dict)
        assert isinstance(data["head_positions"], dict)
        assert isinstance(data["merged_intents"], list)
        assert isinstance(data["cycle_count"], int)


# ---------------------------------------------------------------------------
# Test: Manifest Alignment
# ---------------------------------------------------------------------------


class TestManifestAlignment:
    """Tests for alignment with v7_manifest.json."""

    def test_hydra_state_in_manifest(self, v7_manifest: Dict[str, Any]):
        """Test that hydra_state is defined in manifest."""
        state_files = v7_manifest.get("state_files", {})
        assert "hydra_state" in state_files

    def test_manifest_path_matches_default(self, v7_manifest: Dict[str, Any]):
        """Test that manifest path matches module default."""
        state_files = v7_manifest.get("state_files", {})
        hydra_entry = state_files.get("hydra_state", {})
        expected_path = "logs/state/hydra_state.json"
        assert hydra_entry.get("path") == expected_path

    def test_manifest_fields_documented(self, v7_manifest: Dict[str, Any]):
        """Test that manifest documents required fields."""
        state_files = v7_manifest.get("state_files", {})
        hydra_entry = state_files.get("hydra_state", {})
        fields = hydra_entry.get("fields", {})

        expected_fields = [
            "updated_ts",
            "head_budgets",
            "head_usage",
            "head_positions",
            "merged_intents",
            "cycle_count",
            "notes",
            "errors",
            "meta",
        ]

        for field in expected_fields:
            assert field in fields, f"Field {field} not documented in manifest"


# ---------------------------------------------------------------------------
# Test: Config Integration
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    """Tests for config integration with strategy_config.json."""

    def test_loads_from_real_config(self, real_strategy_config: Dict[str, Any]):
        """Test loading Hydra config from real strategy_config.json."""
        if not real_strategy_config:
            pytest.skip("strategy_config.json not available")

        cfg = load_hydra_config(strategy_config=real_strategy_config)

        # Should load without error
        assert isinstance(cfg, HydraConfig)

        # Check hydra_execution block if present
        if "hydra_execution" in real_strategy_config:
            hydra_block = real_strategy_config["hydra_execution"]
            assert cfg.enabled == hydra_block.get("enabled", False)

    def test_config_has_all_heads(self, real_strategy_config: Dict[str, Any]):
        """Test that config has all 6 heads."""
        cfg = load_hydra_config(strategy_config=real_strategy_config)

        for head in STRATEGY_HEADS:
            assert head in cfg.heads

    def test_disabled_by_default(self):
        """Test that Hydra is disabled by default."""
        cfg = load_hydra_config(strategy_config={})
        assert cfg.enabled is False


# ---------------------------------------------------------------------------
# Test: Executor Integration
# ---------------------------------------------------------------------------


class TestExecutorIntegration:
    """Tests for executor integration hooks."""

    def test_run_hydra_pipeline_disabled(self):
        """Test that disabled Hydra returns empty results."""
        strategy_config = {"hydra_execution": {"enabled": False}}

        merged, state = run_hydra_pipeline(
            strategy_config=strategy_config,
            cerberus_multipliers={},
            symbols=["BTCUSDT"],
            nav_usd=10000,
        )

        assert len(merged) == 0
        assert isinstance(state, HydraState)

    def test_run_hydra_pipeline_enabled(self, tmp_state_path: Path, tmp_log_path: Path):
        """Test that enabled Hydra generates intents."""
        strategy_config = {"hydra_execution": {"enabled": True}}

        merged, state = run_hydra_pipeline(
            strategy_config=strategy_config,
            cerberus_multipliers={"TREND": 1.0},
            symbols=["BTCUSDT", "ETHUSDT"],
            nav_usd=10000,
            hybrid_scores={"BTCUSDT": 0.7, "ETHUSDT": 0.5},
            state_path=tmp_state_path,
            log_path=tmp_log_path,
        )

        assert isinstance(state, HydraState)
        assert state.updated_ts != ""

    def test_convert_intents_to_execution(self):
        """Test conversion of merged intents to execution format."""
        merged = [
            HydraMergedIntent(
                symbol="BTCUSDT",
                net_side="long",
                nav_pct=0.05,
                heads=["TREND"],
                head_contributions={"TREND": 0.05},
                score=0.8,
                rationale="Test",
            ),
        ]

        execution = convert_hydra_intents_to_execution(
            merged_intents=merged,
            nav_usd=10000,
            prices={"BTCUSDT": 50000},
        )

        assert len(execution) == 1
        assert execution[0]["symbol"] == "BTCUSDT"
        assert execution[0]["side"] == "LONG"
        assert execution[0]["notional_usd"] == 500
        assert execution[0]["price"] == 50000
        assert execution[0]["gross_usd"] == 500

    def test_get_attribution_for_order(self):
        """Test getting attribution metadata for orders."""
        state = HydraState(
            merged_intents=[
                {
                    "symbol": "BTCUSDT",
                    "net_side": "long",
                    "nav_pct": 0.05,
                    "heads": ["TREND", "CATEGORY"],
                    "head_contributions": {"TREND": 0.03, "CATEGORY": 0.02},
                },
            ]
        )

        attr = get_hydra_attribution_for_order("BTCUSDT", "long", state)

        assert attr["source"] == "hydra"
        assert "TREND" in attr["strategy_heads"]
        assert attr["head_contributions"]["TREND"] == 0.03

    def test_merge_with_legacy_intents_hydra_wins(self):
        """Test that Hydra intents override legacy when preferred."""
        hydra = [{"symbol": "BTCUSDT", "side": "LONG", "qty": 0.1}]
        legacy = [
            {"symbol": "BTCUSDT", "side": "SHORT", "qty": 0.05},
            {"symbol": "ETHUSDT", "side": "LONG", "qty": 0.2},
        ]

        merged = merge_with_single_strategy_intents(hydra, legacy, prefer_hydra=True)

        # BTCUSDT should be from Hydra, ETHUSDT from legacy
        btc = [i for i in merged if i["symbol"] == "BTCUSDT"][0]
        assert btc["side"] == "LONG"
        assert len(merged) == 2


# ---------------------------------------------------------------------------
# Test: Dashboard Panel
# ---------------------------------------------------------------------------


class TestDashboardPanel:
    """Tests for dashboard panel rendering."""

    def test_panel_import(self):
        """Test that panel can be imported."""
        from dashboard.hydra_panel import (
            load_hydra_state,
            render_hydra_panel,
            render_hydra_widget,
        )

        assert callable(render_hydra_panel)
        assert callable(render_hydra_widget)

    def test_load_state_missing_file(self):
        """Test that missing file returns empty dict."""
        from dashboard.hydra_panel import load_hydra_state

        state = load_hydra_state("/nonexistent/path.json")
        assert state == {}

    def test_load_state_valid_file(self, tmp_state_path: Path):
        """Test loading valid state file."""
        from dashboard.hydra_panel import load_hydra_state

        state_data = {
            "updated_ts": "2024-01-01T00:00:00Z",
            "head_budgets": {"TREND": 0.50},
            "head_usage": {"TREND": 0.25},
        }
        tmp_state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_state_path.write_text(json.dumps(state_data))

        loaded = load_hydra_state(tmp_state_path)
        assert loaded["updated_ts"] == "2024-01-01T00:00:00Z"


# ---------------------------------------------------------------------------
# Test: JSONL Logging
# ---------------------------------------------------------------------------


class TestJSONLLogging:
    """Tests for hydra_intents.jsonl logging."""

    def test_log_creates_file(self, tmp_log_path: Path):
        """Test that logging creates the file."""
        intents = [
            HydraMergedIntent(
                symbol="BTCUSDT",
                net_side="long",
                nav_pct=0.05,
                heads=["TREND"],
                head_contributions={"TREND": 0.05},
                score=0.8,
                rationale="Test",
            ),
        ]

        count = log_hydra_intents(intents, tmp_log_path)

        assert count == 1
        assert tmp_log_path.exists()

    def test_log_appends(self, tmp_log_path: Path):
        """Test that logging appends to existing file."""
        intent = HydraMergedIntent(
            symbol="BTCUSDT",
            net_side="long",
            nav_pct=0.05,
            heads=["TREND"],
            head_contributions={"TREND": 0.05},
            score=0.8,
            rationale="Test",
        )

        log_hydra_intents([intent], tmp_log_path)
        log_hydra_intents([intent], tmp_log_path)

        lines = tmp_log_path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_log_entry_has_required_fields(self, tmp_log_path: Path):
        """Test that log entries have required fields."""
        intent = HydraMergedIntent(
            symbol="BTCUSDT",
            net_side="long",
            nav_pct=0.05,
            heads=["TREND"],
            head_contributions={"TREND": 0.05},
            score=0.8,
            rationale="Test",
        )

        log_hydra_intents([intent], tmp_log_path)

        line = tmp_log_path.read_text().strip()
        entry = json.loads(line)

        assert "ts" in entry
        assert "symbol" in entry
        assert "net_side" in entry
        assert "nav_pct" in entry
        assert "heads" in entry
        assert "head_contributions" in entry
        assert "score" in entry


# ---------------------------------------------------------------------------
# Test: Disabled Behavior (v7.8 Baseline)
# ---------------------------------------------------------------------------


class TestDisabledBehavior:
    """Tests that disabled Hydra matches v7.8 baseline."""

    def test_disabled_hydra_no_state_written(self, tmp_state_path: Path):
        """Test that disabled Hydra writes minimal state."""
        cfg = HydraConfig(enabled=False)

        merged, state = run_hydra_step(
            cfg=cfg,
            cerberus_multipliers={},
            symbols=["BTCUSDT"],
            hybrid_scores={"BTCUSDT": 0.8},
            zscore_map={},
            pair_edges=[],
            category_scores={},
            symbol_categories={},
            vol_targets={},
            realized_vols={},
            universe_scores={},
            alpha_miner_signals={},
            nav_usd=10000,
            state_path=tmp_state_path,
        )

        assert len(merged) == 0
        assert "Hydra disabled" in state.notes

    def test_disabled_hydra_pipeline_passthrough(self):
        """Test that disabled Hydra pipeline returns empty."""
        strategy_config = {"hydra_execution": {"enabled": False}}

        merged, state = run_hydra_pipeline(
            strategy_config=strategy_config,
            cerberus_multipliers={"TREND": 1.5},
            symbols=["BTCUSDT", "ETHUSDT"],
            nav_usd=10000,
            hybrid_scores={"BTCUSDT": 0.9},
        )

        assert len(merged) == 0

    def test_is_hydra_enabled_returns_false(self):
        """Test that is_hydra_enabled returns False when disabled."""
        assert is_hydra_enabled({}) is False
        assert is_hydra_enabled({"hydra_execution": {"enabled": False}}) is False

    def test_attribution_fallback_to_legacy(self):
        """Test that attribution falls back to legacy when no state."""
        attr = get_hydra_attribution_for_order("BTCUSDT", "long", None)

        assert attr["source"] == "legacy"
        assert attr["strategy_heads"] == []


# ---------------------------------------------------------------------------
# Test: End-to-End Pipeline
# ---------------------------------------------------------------------------


class TestEndToEndPipeline:
    """End-to-end tests for the full Hydra pipeline."""

    def test_full_pipeline_with_all_heads(self, tmp_state_path: Path, tmp_log_path: Path):
        """Test full pipeline with all head generators."""
        strategy_config = {"hydra_execution": {"enabled": True}}

        merged, state = run_hydra_pipeline(
            strategy_config=strategy_config,
            cerberus_multipliers={h: 1.0 for h in STRATEGY_HEADS},
            symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            nav_usd=10000,
            cycle_count=5,
            hybrid_scores={"BTCUSDT": 0.7, "ETHUSDT": 0.5, "SOLUSDT": -0.3},
            zscore_map={"BTCUSDT": 0.5, "ETHUSDT": 2.5, "SOLUSDT": -1.8},
            pair_edges=[{"long_symbol": "BTCUSDT", "short_symbol": "ETHUSDT", "edge_score": 0.6}],
            category_scores={"L1": 0.4, "DEFI": -0.3},
            symbol_categories={"BTCUSDT": "L1", "ETHUSDT": "L1", "SOLUSDT": "DEFI"},
            vol_targets={"BTCUSDT": 0.20},
            realized_vols={"BTCUSDT": 0.25},
            universe_scores={"BTCUSDT": 0.6, "SOLUSDT": 0.8},
            alpha_miner_signals={"BTCUSDT": {"alpha_score": 0.5}},
            state_path=tmp_state_path,
            log_path=tmp_log_path,
        )

        # State should be populated
        assert state.updated_ts != ""
        assert state.cycle_count == 5

        # Should have some head usage
        assert any(v > 0 for v in state.head_usage.values())

    def test_pipeline_respects_budget_limits(self, tmp_state_path: Path):
        """Test that pipeline respects per-head budget limits."""
        strategy_config = {
            "hydra_execution": {
                "enabled": True,
                "heads": {
                    "TREND": {"enabled": True, "max_nav_pct": 0.10},
                },
            }
        }

        # Generate many high-scoring trend signals
        merged, state = run_hydra_pipeline(
            strategy_config=strategy_config,
            cerberus_multipliers={"TREND": 1.0},
            symbols=[f"SYM{i}USDT" for i in range(20)],
            nav_usd=10000,
            hybrid_scores={f"SYM{i}USDT": 0.8 for i in range(20)},
            state_path=tmp_state_path,
        )

        # TREND usage should be capped at max_nav_pct
        trend_usage = state.head_usage.get("TREND", 0.0)
        assert trend_usage <= 0.10 + 0.001  # Small epsilon for float precision

    def test_pipeline_handles_conflicting_signals(self, tmp_state_path: Path):
        """Test that pipeline handles conflicting signals correctly."""
        strategy_config = {"hydra_execution": {"enabled": True}}

        # Create scenario where TREND and MEAN_REVERT conflict on BTCUSDT
        merged, state = run_hydra_pipeline(
            strategy_config=strategy_config,
            cerberus_multipliers={"TREND": 1.0, "MEAN_REVERT": 1.0},
            symbols=["BTCUSDT"],
            nav_usd=10000,
            hybrid_scores={"BTCUSDT": 0.8},  # Strong trend signal (long)
            zscore_map={"BTCUSDT": 2.5},  # Overbought (mean revert short)
            state_path=tmp_state_path,
        )

        # Should have resolved the conflict
        if merged:
            btc_intent = [m for m in merged if m.symbol == "BTCUSDT"]
            if btc_intent:
                # Net side should be determined
                assert btc_intent[0].net_side in ("long", "short", "flat")
