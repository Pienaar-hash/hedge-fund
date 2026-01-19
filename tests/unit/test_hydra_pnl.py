"""
Unit tests for execution/hydra_pnl.py — v7.9_P2

Tests Hydra PnL Attribution & Drawdown Engine:
- Config loading
- HeadPnlStats dataclass
- HydraPnlState dataclass
- PnL attribution
- Kill switch logic
- Throttle scaling
- Integration helpers
"""

import json
import pytest
import tempfile
from pathlib import Path

from execution.hydra_pnl import (
    # Constants
    STRATEGY_HEADS,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MIN_TRADES,
    DEFAULT_MAX_DRAWDOWN,
    DEFAULT_MIN_WIN_RATE,
    DEFAULT_MIN_R_MULTIPLE,
    DEFAULT_COOLDOWN_CYCLES,
    DEFAULT_DD_SOFT_THRESHOLD,
    DEFAULT_DD_HARD_THRESHOLD,
    DEFAULT_SOFT_SCALE,
    DEFAULT_HARD_SCALE,
    # Data classes
    KillSwitchConfig,
    ThrottlingConfig,
    HydraPnlConfig,
    HeadPnlStats,
    HydraPnlState,
    # Config loading
    load_hydra_pnl_config,
    # State I/O
    load_hydra_pnl_state,
    save_hydra_pnl_state,
    # Event logging
    log_pnl_event,
    # PnL Attribution
    attribute_fill_pnl,
    update_unrealized_pnl,
    update_gross_exposure,
    # Kill switch
    check_kill_switch,
    evaluate_all_kill_switches,
    # Throttling
    compute_throttle_scale,
    update_throttle_scales,
    # Integration helpers
    get_head_throttle_scales,
    get_head_kill_switch_status,
    is_head_active,
    get_head_stats_summary,
    get_hydra_pnl_summary,
    # Pipeline
    run_hydra_pnl_step,
    # Cerberus integration
    apply_pnl_throttle_to_cerberus,
    # Hydra integration
    apply_pnl_throttle_to_hydra_budgets,
    is_hydra_pnl_enabled,
)


class TestKillSwitchConfig:
    """Tests for KillSwitchConfig dataclass."""

    def test_default_values(self):
        cfg = KillSwitchConfig()
        assert cfg.max_drawdown == DEFAULT_MAX_DRAWDOWN
        assert cfg.min_win_rate == DEFAULT_MIN_WIN_RATE
        assert cfg.min_R_multiple == DEFAULT_MIN_R_MULTIPLE
        assert cfg.cooldown_cycles == DEFAULT_COOLDOWN_CYCLES

    def test_custom_values(self):
        cfg = KillSwitchConfig(
            max_drawdown=0.30,
            min_win_rate=0.35,
            min_R_multiple=-1.0,
            cooldown_cycles=500,
        )
        assert cfg.max_drawdown == 0.30
        assert cfg.min_win_rate == 0.35
        assert cfg.min_R_multiple == -1.0
        assert cfg.cooldown_cycles == 500


class TestThrottlingConfig:
    """Tests for ThrottlingConfig dataclass."""

    def test_default_values(self):
        cfg = ThrottlingConfig()
        assert cfg.dd_soft_threshold == DEFAULT_DD_SOFT_THRESHOLD
        assert cfg.dd_hard_threshold == DEFAULT_DD_HARD_THRESHOLD
        assert cfg.soft_scale == DEFAULT_SOFT_SCALE
        assert cfg.hard_scale == DEFAULT_HARD_SCALE

    def test_custom_values(self):
        cfg = ThrottlingConfig(
            dd_soft_threshold=0.05,
            dd_hard_threshold=0.15,
            soft_scale=0.7,
            hard_scale=0.1,
        )
        assert cfg.dd_soft_threshold == 0.05
        assert cfg.dd_hard_threshold == 0.15
        assert cfg.soft_scale == 0.7
        assert cfg.hard_scale == 0.1


class TestHydraPnlConfig:
    """Tests for HydraPnlConfig dataclass."""

    def test_default_values(self):
        cfg = HydraPnlConfig()
        assert cfg.enabled is False
        assert cfg.lookback_days == DEFAULT_LOOKBACK_DAYS
        assert cfg.min_trades_for_stats == DEFAULT_MIN_TRADES
        assert isinstance(cfg.kill_switch, KillSwitchConfig)
        assert isinstance(cfg.throttling, ThrottlingConfig)

    def test_enabled_config(self):
        cfg = HydraPnlConfig(enabled=True, lookback_days=60, min_trades_for_stats=20)
        assert cfg.enabled is True
        assert cfg.lookback_days == 60
        assert cfg.min_trades_for_stats == 20

    def test_lookback_clamped(self):
        cfg = HydraPnlConfig(lookback_days=500)
        assert cfg.lookback_days == 365  # Max 365

        cfg = HydraPnlConfig(lookback_days=-10)
        assert cfg.lookback_days == 1  # Min 1

    def test_min_trades_clamped(self):
        cfg = HydraPnlConfig(min_trades_for_stats=2000)
        assert cfg.min_trades_for_stats == 1000  # Max 1000


class TestHeadPnlStats:
    """Tests for HeadPnlStats dataclass."""

    def test_default_values(self):
        stats = HeadPnlStats(head="TREND")
        assert stats.head == "TREND"
        assert stats.equity == 0.0
        assert stats.max_equity == 0.0
        assert stats.drawdown == 0.0
        assert stats.realized_pnl == 0.0
        assert stats.unrealized_pnl == 0.0
        assert stats.trades == 0
        assert stats.wins == 0
        assert stats.win_rate == 0.0
        assert stats.throttle_scale == 1.0
        assert stats.kill_switch_active is False

    def test_update_equity(self):
        stats = HeadPnlStats(head="TREND", realized_pnl=100.0, unrealized_pnl=50.0)
        stats.update_equity()
        assert stats.equity == 150.0
        assert stats.max_equity == 150.0
        assert stats.drawdown == 0.0

        # Now drop equity
        stats.unrealized_pnl = -50.0
        stats.update_equity()
        assert stats.equity == 50.0
        assert stats.max_equity == 150.0  # Still 150
        assert stats.drawdown == pytest.approx((150 - 50) / 150)

    def test_record_trade_win(self):
        stats = HeadPnlStats(head="TREND")
        stats.record_trade(pnl=100.0, R_multiple=1.5)

        assert stats.trades == 1
        assert stats.wins == 1
        assert stats.realized_pnl == 100.0
        assert stats.win_rate == 1.0
        assert stats.total_R == 1.5
        assert stats.trades_with_R == 1
        assert stats.avg_R == 1.5
        assert stats.last_trade_ts > 0

    def test_record_trade_loss(self):
        stats = HeadPnlStats(head="TREND")
        stats.record_trade(pnl=-50.0, R_multiple=-0.5)

        assert stats.trades == 1
        assert stats.wins == 0
        assert stats.realized_pnl == -50.0
        assert stats.win_rate == 0.0
        assert stats.avg_R == -0.5

    def test_to_dict(self):
        stats = HeadPnlStats(head="TREND", realized_pnl=100.0, trades=5)
        d = stats.to_dict()
        assert d["head"] == "TREND"
        assert d["realized_pnl"] == 100.0
        assert d["trades"] == 5

    def test_from_dict(self):
        d = {
            "head": "MEAN_REVERT",
            "equity": 500.0,
            "drawdown": 0.05,
            "trades": 10,
            "throttle_scale": 0.8,
        }
        stats = HeadPnlStats.from_dict(d)
        assert stats.head == "MEAN_REVERT"
        assert stats.equity == 500.0
        assert stats.drawdown == 0.05
        assert stats.trades == 10
        assert stats.throttle_scale == 0.8


class TestHydraPnlState:
    """Tests for HydraPnlState dataclass."""

    def test_default_heads_created(self):
        state = HydraPnlState()
        assert len(state.heads) == 6
        for head in STRATEGY_HEADS:
            assert head in state.heads
            assert state.heads[head].head == head

    def test_get_total_realized_pnl(self):
        state = HydraPnlState()
        state.heads["TREND"].realized_pnl = 100.0
        state.heads["MEAN_REVERT"].realized_pnl = 50.0
        assert state.get_total_realized_pnl() == 150.0

    def test_get_total_equity(self):
        state = HydraPnlState()
        state.heads["TREND"].realized_pnl = 100.0
        state.heads["TREND"].unrealized_pnl = 20.0
        state.heads["TREND"].update_equity()
        state.heads["MEAN_REVERT"].realized_pnl = 50.0
        state.heads["MEAN_REVERT"].update_equity()
        assert state.get_total_equity() == 170.0

    def test_get_max_head_drawdown(self):
        state = HydraPnlState()
        state.heads["TREND"].drawdown = 0.10
        state.heads["MEAN_REVERT"].drawdown = 0.25
        state.heads["VOL_HARVEST"].drawdown = 0.15
        assert state.get_max_head_drawdown() == 0.25

    def test_get_worst_heads(self):
        state = HydraPnlState()
        state.heads["TREND"].drawdown = 0.10
        state.heads["MEAN_REVERT"].drawdown = 0.30
        state.heads["RELATIVE_VALUE"].drawdown = 0.20
        worst = state.get_worst_heads(2)
        assert worst[0] == "MEAN_REVERT"
        assert worst[1] == "RELATIVE_VALUE"

    def test_get_best_heads(self):
        state = HydraPnlState()
        state.heads["TREND"].equity = 100.0
        state.heads["MEAN_REVERT"].equity = 200.0
        state.heads["CATEGORY"].equity = 50.0
        best = state.get_best_heads(2)
        assert best[0] == "MEAN_REVERT"
        assert best[1] == "TREND"

    def test_to_dict_from_dict(self):
        state = HydraPnlState()
        state.heads["TREND"].realized_pnl = 100.0
        state.updated_ts = "2024-01-01T00:00:00Z"

        d = state.to_dict()
        restored = HydraPnlState.from_dict(d)

        assert restored.updated_ts == "2024-01-01T00:00:00Z"
        assert restored.heads["TREND"].realized_pnl == 100.0


class TestConfigLoading:
    """Tests for config loading."""

    def test_load_empty_config(self):
        cfg = load_hydra_pnl_config(strategy_config={})
        assert cfg.enabled is False
        assert cfg.lookback_days == DEFAULT_LOOKBACK_DAYS

    def test_load_from_dict(self):
        strategy_cfg = {
            "hydra_pnl": {
                "enabled": True,
                "lookback_days": 60,
                "min_trades_for_stats": 25,
                "kill_switch": {
                    "max_drawdown": 0.30,
                    "min_win_rate": 0.45,
                },
                "throttling": {
                    "dd_soft_threshold": 0.08,
                },
            }
        }
        cfg = load_hydra_pnl_config(strategy_config=strategy_cfg)
        assert cfg.enabled is True
        assert cfg.lookback_days == 60
        assert cfg.min_trades_for_stats == 25
        assert cfg.kill_switch.max_drawdown == 0.30
        assert cfg.kill_switch.min_win_rate == 0.45
        assert cfg.throttling.dd_soft_threshold == 0.08


class TestStateIO:
    """Tests for state I/O."""

    def test_load_missing_file(self):
        state = load_hydra_pnl_state("/nonexistent/path.json")
        assert isinstance(state, HydraPnlState)
        assert len(state.heads) == 6

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "hydra_pnl.json"

            state = HydraPnlState()
            state.heads["TREND"].realized_pnl = 500.0
            state.updated_ts = "2024-01-01T12:00:00Z"

            assert save_hydra_pnl_state(state, path)

            loaded = load_hydra_pnl_state(path)
            assert loaded.heads["TREND"].realized_pnl == 500.0
            assert loaded.updated_ts == "2024-01-01T12:00:00Z"


class TestEventLogging:
    """Tests for event logging."""

    def test_log_event(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "events.jsonl"

            assert log_pnl_event(
                head="TREND",
                event="KILL_SWITCH_ON",
                reason="drawdown > 25%",
                log_path=path,
                extra={"cooldown": 720},
            )

            content = path.read_text()
            assert "TREND" in content
            assert "KILL_SWITCH_ON" in content
            assert "drawdown > 25%" in content


class TestPnlAttribution:
    """Tests for PnL attribution."""

    def test_attribute_fill_pnl(self):
        state = HydraPnlState()
        contributions = {"TREND": 0.6, "MEAN_REVERT": 0.4}

        attributed = attribute_fill_pnl(
            fill_pnl=100.0,
            fill_fee=0.5,
            head_contributions=contributions,
            state=state,
            R_multiple=2.0,
        )

        assert attributed["TREND"] == pytest.approx(60.0)
        assert attributed["MEAN_REVERT"] == pytest.approx(40.0)
        assert state.heads["TREND"].realized_pnl == pytest.approx(60.0)
        assert state.heads["MEAN_REVERT"].realized_pnl == pytest.approx(40.0)

    def test_update_unrealized_pnl(self):
        state = HydraPnlState()
        positions = [
            {"symbol": "BTCUSDT", "unrealized_pnl": 100.0},
            {"symbol": "ETHUSDT", "unrealized_pnl": -50.0},
        ]
        contributions = {
            "BTCUSDT": {"TREND": 1.0},
            "ETHUSDT": {"MEAN_REVERT": 0.5, "TREND": 0.5},
        }

        update_unrealized_pnl(positions, contributions, state)

        assert state.heads["TREND"].unrealized_pnl == pytest.approx(100.0 - 25.0)
        assert state.heads["MEAN_REVERT"].unrealized_pnl == pytest.approx(-25.0)

    def test_update_gross_exposure(self):
        state = HydraPnlState()
        positions = [
            {"symbol": "BTCUSDT", "notional_usd": 1000.0},
            {"symbol": "ETHUSDT", "notional_usd": 500.0},
        ]
        contributions = {
            "BTCUSDT": {"TREND": 1.0},
            "ETHUSDT": {"MEAN_REVERT": 1.0},
        }

        update_gross_exposure(positions, contributions, state, nav_usd=10000.0)

        assert state.heads["TREND"].gross_exposure == pytest.approx(0.10)
        assert state.heads["MEAN_REVERT"].gross_exposure == pytest.approx(0.05)


class TestKillSwitch:
    """Tests for kill switch logic."""

    def test_no_kill_insufficient_trades(self):
        cfg = HydraPnlConfig(enabled=True, min_trades_for_stats=30)
        stats = HeadPnlStats(head="TREND", trades=10, drawdown=0.50)

        result = check_kill_switch(stats, cfg)
        assert result is False

    def test_kill_on_drawdown(self):
        cfg = HydraPnlConfig(enabled=True, min_trades_for_stats=10)
        cfg.kill_switch.max_drawdown = 0.25
        stats = HeadPnlStats(head="TREND", trades=20, drawdown=0.30)

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "events.jsonl"
            result = check_kill_switch(stats, cfg, log_path)

            assert result is True
            assert stats.kill_switch_active is True
            assert stats.cooldown_remaining == cfg.kill_switch.cooldown_cycles

    def test_kill_on_win_rate(self):
        cfg = HydraPnlConfig(enabled=True, min_trades_for_stats=10)
        cfg.kill_switch.min_win_rate = 0.40
        stats = HeadPnlStats(head="TREND", trades=20, win_rate=0.30)

        result = check_kill_switch(stats, cfg)
        assert result is True

    def test_kill_on_R_multiple(self):
        cfg = HydraPnlConfig(enabled=True, min_trades_for_stats=10)
        cfg.kill_switch.min_R_multiple = -0.5
        stats = HeadPnlStats(head="TREND", trades=20, trades_with_R=20, avg_R=-0.8)

        result = check_kill_switch(stats, cfg)
        assert result is True

    def test_cooldown_decrements(self):
        cfg = HydraPnlConfig(enabled=True, min_trades_for_stats=10)
        stats = HeadPnlStats(
            head="TREND",
            kill_switch_active=True,
            cooldown_remaining=5,
        )

        result = check_kill_switch(stats, cfg)
        assert stats.cooldown_remaining == 4
        assert stats.kill_switch_active is True

    def test_cooldown_completes(self):
        cfg = HydraPnlConfig(enabled=True)
        stats = HeadPnlStats(
            head="TREND",
            kill_switch_active=True,
            cooldown_remaining=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "events.jsonl"
            result = check_kill_switch(stats, cfg, log_path)

            assert stats.cooldown_remaining == 0
            assert stats.kill_switch_active is False


class TestThrottling:
    """Tests for throttle scaling."""

    def test_no_throttle_below_soft(self):
        cfg = ThrottlingConfig(dd_soft_threshold=0.10, dd_hard_threshold=0.20)
        scale = compute_throttle_scale(0.05, cfg)
        assert scale == 1.0

    def test_soft_throttle(self):
        cfg = ThrottlingConfig(
            dd_soft_threshold=0.10,
            dd_hard_threshold=0.20,
            soft_scale=0.5,
        )
        # At 15% DD (midpoint between soft 10% and hard 20%)
        scale = compute_throttle_scale(0.15, cfg)
        # k = (0.15 - 0.10) / (0.20 - 0.10) = 0.5
        # scale = 1.0 - 0.5 * (1.0 - 0.5) = 1.0 - 0.25 = 0.75
        assert scale == pytest.approx(0.75)

    def test_hard_throttle(self):
        cfg = ThrottlingConfig(dd_hard_threshold=0.20, hard_scale=0.0)
        scale = compute_throttle_scale(0.25, cfg)
        assert scale == 0.0

    def test_update_throttle_scales(self):
        state = HydraPnlState()
        state.heads["TREND"].drawdown = 0.05  # Below soft
        state.heads["MEAN_REVERT"].drawdown = 0.15  # Between soft/hard
        state.heads["VOL_HARVEST"].drawdown = 0.25  # Above hard

        cfg = HydraPnlConfig(enabled=True)

        scales = update_throttle_scales(state, cfg)

        assert scales["TREND"] == 1.0
        assert 0 < scales["MEAN_REVERT"] < 1.0
        assert scales["VOL_HARVEST"] == 0.0


class TestIntegrationHelpers:
    """Tests for integration helpers."""

    def test_get_head_throttle_scales(self):
        state = HydraPnlState()
        state.heads["TREND"].throttle_scale = 1.0
        state.heads["MEAN_REVERT"].throttle_scale = 0.5

        scales = get_head_throttle_scales(state)
        assert scales["TREND"] == 1.0
        assert scales["MEAN_REVERT"] == 0.5

    def test_get_head_kill_switch_status(self):
        state = HydraPnlState()
        state.heads["TREND"].kill_switch_active = False
        state.heads["MEAN_REVERT"].kill_switch_active = True

        status = get_head_kill_switch_status(state)
        assert status["TREND"] is False
        assert status["MEAN_REVERT"] is True

    def test_is_head_active(self):
        state = HydraPnlState()
        state.heads["TREND"].kill_switch_active = False
        state.heads["MEAN_REVERT"].kill_switch_active = True

        assert is_head_active("TREND", state) is True
        assert is_head_active("MEAN_REVERT", state) is False

    def test_get_head_stats_summary(self):
        state = HydraPnlState()
        state.heads["TREND"].equity = 100.0
        state.heads["TREND"].drawdown = 0.05

        summary = get_head_stats_summary(state)
        assert summary["TREND"]["equity"] == 100.0
        assert summary["TREND"]["drawdown"] == 0.05

    def test_get_hydra_pnl_summary(self):
        state = HydraPnlState()
        state.heads["TREND"].realized_pnl = 100.0
        state.heads["MEAN_REVERT"].realized_pnl = 50.0
        state.heads["TREND"].drawdown = 0.10
        state.heads["VOL_HARVEST"].kill_switch_active = True

        summary = get_hydra_pnl_summary(state)
        assert summary["total_realized_pnl"] == 150.0
        assert summary["heads_killed"] == 1


class TestCerberusIntegration:
    """Tests for Cerberus integration."""

    def test_apply_pnl_throttle_to_cerberus(self):
        state = HydraPnlState()
        state.heads["TREND"].throttle_scale = 1.0
        state.heads["MEAN_REVERT"].throttle_scale = 0.5
        state.heads["VOL_HARVEST"].throttle_scale = 0.0

        cerberus_mults = {
            "TREND": 1.2,
            "MEAN_REVERT": 0.8,
            "VOL_HARVEST": 1.0,
        }

        throttled = apply_pnl_throttle_to_cerberus(cerberus_mults, state)

        assert throttled["TREND"] == pytest.approx(1.2)
        assert throttled["MEAN_REVERT"] == pytest.approx(0.4)
        assert throttled["VOL_HARVEST"] == pytest.approx(0.0)


class TestHydraIntegration:
    """Tests for Hydra integration."""

    def test_apply_pnl_throttle_to_hydra_budgets(self):
        state = HydraPnlState()
        state.heads["TREND"].throttle_scale = 1.0
        state.heads["CATEGORY"].throttle_scale = 0.5

        budgets = {
            "TREND": 0.50,
            "CATEGORY": 0.20,
        }

        throttled = apply_pnl_throttle_to_hydra_budgets(budgets, state)

        assert throttled["TREND"] == pytest.approx(0.50)
        assert throttled["CATEGORY"] == pytest.approx(0.10)


class TestRunHydraPnlStep:
    """Tests for the pipeline runner."""

    def test_disabled_returns_empty_state(self):
        cfg = HydraPnlConfig(enabled=False)
        state = run_hydra_pnl_step(cfg)

        assert state.meta.get("enabled") is False

    def test_basic_step_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "hydra_pnl.json"
            log_path = Path(tmpdir) / "events.jsonl"

            cfg = HydraPnlConfig(enabled=True, min_trades_for_stats=5)

            fills = [
                {"pnl": 100.0, "fee": 0.5, "head_contributions": {"TREND": 1.0}},
                {"pnl": -50.0, "fee": 0.3, "head_contributions": {"MEAN_REVERT": 1.0}},
            ]
            positions = [
                {"symbol": "BTCUSDT", "unrealized_pnl": 25.0, "notional_usd": 1000.0},
            ]
            contributions = {"BTCUSDT": {"TREND": 1.0}}

            state = run_hydra_pnl_step(
                cfg=cfg,
                fills=fills,
                positions=positions,
                head_contributions_by_symbol=contributions,
                nav_usd=10000.0,
                state_path=state_path,
                log_path=log_path,
            )

            assert state.heads["TREND"].realized_pnl == 100.0
            assert state.heads["MEAN_REVERT"].realized_pnl == -50.0
            assert state.heads["TREND"].unrealized_pnl == 25.0
            assert state.updated_ts != ""


class TestIsHydraPnlEnabled:
    """Tests for is_hydra_pnl_enabled."""

    def test_disabled_by_default(self):
        assert is_hydra_pnl_enabled({}) is False

    def test_enabled(self):
        cfg = {"hydra_pnl": {"enabled": True}}
        assert is_hydra_pnl_enabled(cfg) is True

    def test_explicitly_disabled(self):
        cfg = {"hydra_pnl": {"enabled": False}}
        assert is_hydra_pnl_enabled(cfg) is False
