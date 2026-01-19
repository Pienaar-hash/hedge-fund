"""
Integration tests for Hydra PnL Attribution — v7.9_P2

Tests:
- State file contract (v7_manifest alignment)
- Cerberus router integration
- Hydra engine integration
- Config schema validation
- End-to-end pipeline
"""

import json
import pytest
import tempfile
from pathlib import Path

from execution.hydra_pnl import (
    STRATEGY_HEADS,
    HydraPnlConfig,
    HydraPnlState,
    HeadPnlStats,
    load_hydra_pnl_config,
    load_hydra_pnl_state,
    save_hydra_pnl_state,
    run_hydra_pnl_step,
    get_head_throttle_scales,
    apply_pnl_throttle_to_cerberus,
    apply_pnl_throttle_to_hydra_budgets,
)


class TestStateFileContract:
    """Tests for state file schema contract (v7_manifest alignment)."""

    def test_state_file_schema(self):
        """Verify state file has required fields per v7_manifest."""
        state = HydraPnlState()
        state.heads["TREND"].realized_pnl = 100.0
        state.updated_ts = "2024-01-01T00:00:00Z"
        state.meta = {"lookback_days": 90}

        d = state.to_dict()

        # Required fields per manifest
        assert "updated_ts" in d
        assert "heads" in d
        assert "meta" in d

        # Heads dict structure
        assert "TREND" in d["heads"]
        trend = d["heads"]["TREND"]
        assert "equity" in trend
        assert "drawdown" in trend
        assert "win_rate" in trend
        assert "throttle_scale" in trend
        assert "kill_switch_active" in trend

    def test_all_heads_present(self):
        """Verify all 6 canonical heads are in state."""
        state = HydraPnlState()
        d = state.to_dict()

        for head in STRATEGY_HEADS:
            assert head in d["heads"]

    def test_state_roundtrip(self):
        """Verify state survives save/load cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "hydra_pnl.json"

            state = HydraPnlState()
            state.heads["TREND"].realized_pnl = 500.0
            state.heads["TREND"].throttle_scale = 0.8
            state.heads["MEAN_REVERT"].kill_switch_active = True
            state.heads["MEAN_REVERT"].cooldown_remaining = 100
            state.updated_ts = "2024-01-15T10:30:00Z"
            state.meta = {"heads_killed": 1}

            save_hydra_pnl_state(state, path)

            # Verify file contents
            content = json.loads(path.read_text())
            assert content["updated_ts"] == "2024-01-15T10:30:00Z"
            assert content["heads"]["TREND"]["realized_pnl"] == 500.0

            # Load and verify
            loaded = load_hydra_pnl_state(path)
            assert loaded.heads["TREND"].realized_pnl == 500.0
            assert loaded.heads["TREND"].throttle_scale == 0.8
            assert loaded.heads["MEAN_REVERT"].kill_switch_active is True
            assert loaded.heads["MEAN_REVERT"].cooldown_remaining == 100


class TestConfigSchemaValidation:
    """Tests for config schema validation."""

    def test_config_from_strategy_config_json(self):
        """Test loading config from strategy_config.json format."""
        config_data = {
            "hydra_pnl": {
                "_comment": "Hydra PnL Attribution & Drawdown Engine — v7.9_P2",
                "enabled": True,
                "lookback_days": 60,
                "min_trades_for_stats": 25,
                "kill_switch": {
                    "max_drawdown": 0.30,
                    "min_win_rate": 0.35,
                    "min_R_multiple": -0.8,
                    "cooldown_cycles": 500,
                },
                "throttling": {
                    "dd_soft_threshold": 0.08,
                    "dd_hard_threshold": 0.18,
                    "soft_scale": 0.6,
                    "hard_scale": 0.1,
                },
            }
        }

        cfg = load_hydra_pnl_config(strategy_config=config_data)

        assert cfg.enabled is True
        assert cfg.lookback_days == 60
        assert cfg.min_trades_for_stats == 25
        assert cfg.kill_switch.max_drawdown == 0.30
        assert cfg.kill_switch.min_win_rate == 0.35
        assert cfg.kill_switch.min_R_multiple == -0.8
        assert cfg.kill_switch.cooldown_cycles == 500
        assert cfg.throttling.dd_soft_threshold == 0.08
        assert cfg.throttling.dd_hard_threshold == 0.18
        assert cfg.throttling.soft_scale == 0.6
        assert cfg.throttling.hard_scale == 0.1

    def test_partial_config(self):
        """Test loading partial config uses defaults."""
        config_data = {
            "hydra_pnl": {
                "enabled": True,
                "kill_switch": {
                    "max_drawdown": 0.20,
                },
            }
        }

        cfg = load_hydra_pnl_config(strategy_config=config_data)

        assert cfg.enabled is True
        assert cfg.kill_switch.max_drawdown == 0.20
        # Defaults for unspecified
        assert cfg.kill_switch.min_win_rate == 0.40
        assert cfg.throttling.dd_soft_threshold == 0.10


class TestCerberusIntegration:
    """Tests for Cerberus router integration."""

    def test_throttle_applies_to_cerberus_multipliers(self):
        """Test that PnL throttle modifies Cerberus head multipliers."""
        pnl_state = HydraPnlState()
        pnl_state.heads["TREND"].throttle_scale = 1.0
        pnl_state.heads["MEAN_REVERT"].throttle_scale = 0.5
        pnl_state.heads["VOL_HARVEST"].throttle_scale = 0.0
        pnl_state.heads["RELATIVE_VALUE"].throttle_scale = 0.75

        cerberus_mults = {
            "TREND": 1.2,
            "MEAN_REVERT": 1.4,
            "VOL_HARVEST": 0.8,
            "RELATIVE_VALUE": 1.0,
            "CATEGORY": 1.0,
            "EMERGENT_ALPHA": 0.9,
        }

        throttled = apply_pnl_throttle_to_cerberus(cerberus_mults, pnl_state)

        # TREND: 1.2 * 1.0 = 1.2
        assert throttled["TREND"] == pytest.approx(1.2)
        # MEAN_REVERT: 1.4 * 0.5 = 0.7
        assert throttled["MEAN_REVERT"] == pytest.approx(0.7)
        # VOL_HARVEST: 0.8 * 0.0 = 0.0
        assert throttled["VOL_HARVEST"] == pytest.approx(0.0)
        # RELATIVE_VALUE: 1.0 * 0.75 = 0.75
        assert throttled["RELATIVE_VALUE"] == pytest.approx(0.75)

    def test_killed_head_has_zero_multiplier(self):
        """Test that kill-switched head results in zero multiplier."""
        pnl_state = HydraPnlState()
        pnl_state.heads["MEAN_REVERT"].kill_switch_active = True
        pnl_state.heads["MEAN_REVERT"].throttle_scale = 0.0

        cerberus_mults = {"MEAN_REVERT": 1.5}

        throttled = apply_pnl_throttle_to_cerberus(cerberus_mults, pnl_state)
        assert throttled["MEAN_REVERT"] == 0.0


class TestHydraEngineIntegration:
    """Tests for Hydra engine integration."""

    def test_throttle_applies_to_hydra_budgets(self):
        """Test that PnL throttle modifies Hydra NAV budgets."""
        pnl_state = HydraPnlState()
        pnl_state.heads["TREND"].throttle_scale = 1.0
        pnl_state.heads["CATEGORY"].throttle_scale = 0.5
        pnl_state.heads["VOL_HARVEST"].throttle_scale = 0.0

        budgets = {
            "TREND": 0.50,
            "CATEGORY": 0.20,
            "VOL_HARVEST": 0.15,
        }

        throttled = apply_pnl_throttle_to_hydra_budgets(budgets, pnl_state)

        assert throttled["TREND"] == pytest.approx(0.50)
        assert throttled["CATEGORY"] == pytest.approx(0.10)
        assert throttled["VOL_HARVEST"] == pytest.approx(0.0)

    def test_killed_head_has_zero_budget(self):
        """Test that kill-switched head gets zero budget."""
        pnl_state = HydraPnlState()
        pnl_state.heads["EMERGENT_ALPHA"].kill_switch_active = True
        pnl_state.heads["EMERGENT_ALPHA"].throttle_scale = 0.0

        budgets = {"EMERGENT_ALPHA": 0.15}

        throttled = apply_pnl_throttle_to_hydra_budgets(budgets, pnl_state)
        assert throttled["EMERGENT_ALPHA"] == 0.0


class TestEndToEndPipeline:
    """End-to-end tests for the full pipeline."""

    def test_full_cycle_with_fills_and_positions(self):
        """Test a complete PnL cycle with fills and positions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "hydra_pnl.json"
            log_path = Path(tmpdir) / "events.jsonl"

            cfg = HydraPnlConfig(
                enabled=True,
                lookback_days=30,
                min_trades_for_stats=5,
            )

            # First cycle: some winning trades
            fills_1 = [
                {"pnl": 100.0, "fee": 0.5, "head_contributions": {"TREND": 0.7, "CATEGORY": 0.3}, "R_multiple": 1.5},
                {"pnl": 50.0, "fee": 0.3, "head_contributions": {"MEAN_REVERT": 1.0}, "R_multiple": 1.0},
            ]

            state = run_hydra_pnl_step(
                cfg=cfg,
                fills=fills_1,
                positions=[],
                head_contributions_by_symbol={},
                nav_usd=10000.0,
                state_path=state_path,
                log_path=log_path,
            )

            # Verify attribution
            assert state.heads["TREND"].realized_pnl == pytest.approx(70.0)
            assert state.heads["CATEGORY"].realized_pnl == pytest.approx(30.0)
            assert state.heads["MEAN_REVERT"].realized_pnl == pytest.approx(50.0)

            # Second cycle: add positions for unrealized PnL
            positions = [
                {"symbol": "BTCUSDT", "unrealized_pnl": 200.0, "notional_usd": 2000.0},
                {"symbol": "ETHUSDT", "unrealized_pnl": -75.0, "notional_usd": 1500.0},
            ]
            contributions = {
                "BTCUSDT": {"TREND": 0.8, "VOL_HARVEST": 0.2},
                "ETHUSDT": {"MEAN_REVERT": 1.0},
            }

            state = run_hydra_pnl_step(
                cfg=cfg,
                fills=[],
                positions=positions,
                head_contributions_by_symbol=contributions,
                nav_usd=10000.0,
                state_path=state_path,
                log_path=log_path,
            )

            # Verify unrealized attribution
            assert state.heads["TREND"].unrealized_pnl == pytest.approx(160.0)
            assert state.heads["VOL_HARVEST"].unrealized_pnl == pytest.approx(40.0)
            assert state.heads["MEAN_REVERT"].unrealized_pnl == pytest.approx(-75.0)

    def test_kill_switch_triggers_and_cooldown(self):
        """Test kill switch activation and cooldown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "hydra_pnl.json"
            log_path = Path(tmpdir) / "events.jsonl"

            cfg = HydraPnlConfig(enabled=True, min_trades_for_stats=3)
            cfg.kill_switch.max_drawdown = 0.20
            cfg.kill_switch.cooldown_cycles = 5

            # Create state with high drawdown
            state = HydraPnlState()
            state.heads["TREND"].trades = 10
            state.heads["TREND"].wins = 3
            state.heads["TREND"].realized_pnl = 100.0
            state.heads["TREND"].max_equity = 500.0
            state.heads["TREND"].equity = 350.0  # 30% drawdown
            state.heads["TREND"].drawdown = 0.30
            state.heads["TREND"].update_win_rate()

            save_hydra_pnl_state(state, state_path)

            # Run step to trigger kill switch
            state = run_hydra_pnl_step(
                cfg=cfg,
                fills=[],
                positions=[],
                head_contributions_by_symbol={},
                nav_usd=10000.0,
                state_path=state_path,
                log_path=log_path,
            )

            assert state.heads["TREND"].kill_switch_active is True
            assert state.heads["TREND"].cooldown_remaining == 5
            assert state.heads["TREND"].throttle_scale == 0.0

            # Verify event logged
            events = log_path.read_text()
            assert "KILL_SWITCH_ON" in events
            assert "TREND" in events

    def test_throttle_scales_with_drawdown(self):
        """Test throttle scaling based on drawdown levels."""
        cfg = HydraPnlConfig(enabled=True)
        cfg.throttling.dd_soft_threshold = 0.10
        cfg.throttling.dd_hard_threshold = 0.20
        cfg.throttling.soft_scale = 0.5
        cfg.throttling.hard_scale = 0.0

        state = HydraPnlState()

        # Set up proper equity/max_equity to produce the desired drawdowns
        # Below soft (5% DD): max=1000, equity=950
        state.heads["TREND"].max_equity = 1000.0
        state.heads["TREND"].realized_pnl = 950.0
        state.heads["TREND"].equity = 950.0
        state.heads["TREND"].drawdown = 0.05

        # Between soft and hard (15% DD): max=1000, equity=850
        state.heads["MEAN_REVERT"].max_equity = 1000.0
        state.heads["MEAN_REVERT"].realized_pnl = 850.0
        state.heads["MEAN_REVERT"].equity = 850.0
        state.heads["MEAN_REVERT"].drawdown = 0.15

        # Above hard (25% DD): max=1000, equity=750
        state.heads["VOL_HARVEST"].max_equity = 1000.0
        state.heads["VOL_HARVEST"].realized_pnl = 750.0
        state.heads["VOL_HARVEST"].equity = 750.0
        state.heads["VOL_HARVEST"].drawdown = 0.25

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "hydra_pnl.json"
            save_hydra_pnl_state(state, state_path)

            result = run_hydra_pnl_step(
                cfg=cfg,
                fills=[],
                positions=[],
                head_contributions_by_symbol={},
                nav_usd=10000.0,
                state_path=state_path,
            )

            # TREND: 5% DD < 10% soft → 1.0
            assert result.heads["TREND"].throttle_scale == 1.0

            # MEAN_REVERT: 15% DD (midway) → 0.75
            assert result.heads["MEAN_REVERT"].throttle_scale == pytest.approx(0.75)

            # VOL_HARVEST: 25% DD > 20% hard → 0.0
            assert result.heads["VOL_HARVEST"].throttle_scale == 0.0


class TestManifestAlignment:
    """Tests to verify alignment with v7_manifest.json."""

    def test_state_path_matches_manifest(self):
        """Verify default state path matches manifest entry."""
        from execution.hydra_pnl import DEFAULT_STATE_PATH

        # Should match manifest: "path": "logs/state/hydra_pnl.json"
        assert str(DEFAULT_STATE_PATH) == "logs/state/hydra_pnl.json"

    def test_state_schema_has_manifest_fields(self):
        """Verify state has all fields declared in manifest."""
        state = HydraPnlState()
        state.updated_ts = "2024-01-01T00:00:00Z"
        state.meta = {"lookback_days": 90}

        d = state.to_dict()

        # Per manifest fields:
        # "updated_ts": "ISO timestamp"
        assert "updated_ts" in d
        assert isinstance(d["updated_ts"], str)

        # "heads": "Dict[str, HeadPnlStats] - per-head equity, drawdown, win_rate, throttle_scale"
        assert "heads" in d
        assert isinstance(d["heads"], dict)
        for head_data in d["heads"].values():
            assert "equity" in head_data
            assert "drawdown" in head_data
            assert "win_rate" in head_data
            assert "throttle_scale" in head_data

        # "meta": "Dict - lookback_days, total_realized_pnl, total_unrealized_pnl, heads_killed"
        assert "meta" in d
        assert isinstance(d["meta"], dict)


class TestDashboardIntegration:
    """Tests for dashboard panel integration."""

    def test_panel_loads_state(self):
        """Test that dashboard panel can load state correctly."""
        from dashboard.hydra_panel import load_hydra_pnl_state as panel_load_state

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "hydra_pnl.json"

            state = HydraPnlState()
            state.heads["TREND"].realized_pnl = 500.0
            state.heads["TREND"].drawdown = 0.15
            state.heads["TREND"].throttle_scale = 0.75
            state.updated_ts = "2024-01-01T12:00:00Z"

            save_hydra_pnl_state(state, path)

            loaded = panel_load_state(path)

            assert loaded["updated_ts"] == "2024-01-01T12:00:00Z"
            assert loaded["heads"]["TREND"]["realized_pnl"] == 500.0
            assert loaded["heads"]["TREND"]["drawdown"] == 0.15
            assert loaded["heads"]["TREND"]["throttle_scale"] == 0.75

    def test_panel_handles_missing_state(self):
        """Test that dashboard panel handles missing state gracefully."""
        from dashboard.hydra_panel import load_hydra_pnl_state as panel_load_state

        loaded = panel_load_state("/nonexistent/path.json")
        assert loaded == {}
