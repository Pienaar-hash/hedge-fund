"""
Tests for churn_guard module (v7.9-E2).

Tests the two anti-churn safety rails:
  1. min_hold_seconds: No exits before hold period (except crisis)
  2. cooldown_seconds: No re-entry after exit for cooldown period
"""

import pytest
import time
from unittest.mock import patch, MagicMock

from execution.churn_guard import (
    ChurnConfig,
    ChurnState,
    check_entry_allowed,
    check_exit_allowed,
    record_entry,
    record_exit,
    load_churn_config,
    get_state_snapshot,
    reset_state,
    _key,
    DEFAULT_MIN_HOLD_SECONDS,
    DEFAULT_COOLDOWN_SECONDS,
    HOLD_BYPASS_REASONS,
)


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset churn guard state before and after each test."""
    reset_state()
    yield
    reset_state()


# ── Config loading ─────────────────────────────────────────────────────────

class TestChurnConfig:
    def test_defaults(self):
        cfg = ChurnConfig()
        assert cfg.min_hold_seconds == DEFAULT_MIN_HOLD_SECONDS
        assert cfg.cooldown_seconds == DEFAULT_COOLDOWN_SECONDS
        assert cfg.crisis_override is True

    def test_load_from_dict(self):
        runtime = {
            "churn_guard": {
                "min_hold_seconds": 60,
                "cooldown_seconds": 120,
                "crisis_override": False,
            }
        }
        cfg = load_churn_config(runtime)
        assert cfg.min_hold_seconds == 60.0
        assert cfg.cooldown_seconds == 120.0
        assert cfg.crisis_override is False

    def test_load_missing_section(self):
        cfg = load_churn_config({})
        assert cfg.min_hold_seconds == DEFAULT_MIN_HOLD_SECONDS

    def test_load_none(self):
        cfg = load_churn_config({"churn_guard": None})
        assert cfg.min_hold_seconds == DEFAULT_MIN_HOLD_SECONDS

    def test_partial_override(self):
        cfg = load_churn_config({"churn_guard": {"min_hold_seconds": 30}})
        assert cfg.min_hold_seconds == 30.0
        assert cfg.cooldown_seconds == DEFAULT_COOLDOWN_SECONDS


# ── Key generation ─────────────────────────────────────────────────────────

class TestKey:
    def test_canonical_key(self):
        assert _key("BTCUSDT", "LONG") == "BTCUSDT_LONG"

    def test_normalizes_case(self):
        assert _key("ethusdt", "long") == "ETHUSDT_LONG"


# ── Min-hold gate (exit side) ─────────────────────────────────────────────

class TestMinHold:
    def test_exit_allowed_no_entry_recorded(self):
        """No entry recorded → allow exit (position predates session)."""
        allowed, reason = check_exit_allowed("BTCUSDT", "LONG", now=1000.0)
        assert allowed is True
        assert reason == ""

    def test_exit_vetoed_within_hold(self):
        """Exit within min_hold_seconds → veto."""
        cfg = ChurnConfig(min_hold_seconds=120)
        record_entry("BTCUSDT", "LONG", ts=1000.0)
        allowed, reason = check_exit_allowed(
            "BTCUSDT", "LONG", now=1050.0, config=cfg,
        )
        assert allowed is False
        assert "min_hold_veto" in reason
        assert "50s" in reason

    def test_exit_allowed_after_hold(self):
        """Exit after min_hold_seconds → allow."""
        cfg = ChurnConfig(min_hold_seconds=120)
        record_entry("BTCUSDT", "LONG", ts=1000.0)
        allowed, _ = check_exit_allowed(
            "BTCUSDT", "LONG", now=1200.0, config=cfg,
        )
        assert allowed is True

    def test_exit_allowed_exactly_at_threshold(self):
        cfg = ChurnConfig(min_hold_seconds=120)
        record_entry("BTCUSDT", "LONG", ts=1000.0)
        allowed, _ = check_exit_allowed(
            "BTCUSDT", "LONG", now=1120.0, config=cfg,
        )
        assert allowed is True

    def test_crisis_override_bypasses_hold(self):
        """CRISIS_OVERRIDE exits always bypass min_hold."""
        cfg = ChurnConfig(min_hold_seconds=120, crisis_override=True)
        record_entry("BTCUSDT", "LONG", ts=1000.0)
        allowed, _ = check_exit_allowed(
            "BTCUSDT", "LONG", exit_reason="CRISIS_OVERRIDE",
            now=1010.0, config=cfg,
        )
        assert allowed is True

    def test_seatbelt_bypasses_hold(self):
        """SEATBELT exits always bypass min_hold."""
        cfg = ChurnConfig(min_hold_seconds=120, crisis_override=True)
        record_entry("BTCUSDT", "LONG", ts=1000.0)
        allowed, _ = check_exit_allowed(
            "BTCUSDT", "LONG", exit_reason="SEATBELT",
            now=1010.0, config=cfg,
        )
        assert allowed is True

    def test_crisis_override_disabled(self):
        """If crisis_override=False, even CRISIS doesn't bypass hold."""
        cfg = ChurnConfig(min_hold_seconds=120, crisis_override=False)
        record_entry("BTCUSDT", "LONG", ts=1000.0)
        allowed, _ = check_exit_allowed(
            "BTCUSDT", "LONG", exit_reason="CRISIS_OVERRIDE",
            now=1010.0, config=cfg,
        )
        assert allowed is False

    def test_regime_flip_bypasses_hold(self):
        """REGIME_FLIP is a bypass reason — doctrine exits must not be blocked."""
        cfg = ChurnConfig(min_hold_seconds=120)
        record_entry("BTCUSDT", "LONG", ts=1000.0)
        allowed, _ = check_exit_allowed(
            "BTCUSDT", "LONG", exit_reason="REGIME_FLIP",
            now=1010.0, config=cfg,
        )
        assert allowed is True

    def test_regime_change_canonical_bypasses_hold(self):
        """Canonical REGIME_CHANGE also bypasses hold."""
        cfg = ChurnConfig(min_hold_seconds=120)
        record_entry("BTCUSDT", "LONG", ts=1000.0)
        allowed, _ = check_exit_allowed(
            "BTCUSDT", "LONG", exit_reason="REGIME_CHANGE",
            now=1010.0, config=cfg,
        )
        assert allowed is True

    def test_different_symbols_independent(self):
        """Entries for different symbols are tracked independently."""
        cfg = ChurnConfig(min_hold_seconds=120)
        record_entry("BTCUSDT", "LONG", ts=1000.0)
        record_entry("ETHUSDT", "LONG", ts=1050.0)
        # BTC is within hold, ETH is also within hold
        allowed_btc, _ = check_exit_allowed("BTCUSDT", "LONG", now=1100.0, config=cfg)
        allowed_eth, _ = check_exit_allowed("ETHUSDT", "LONG", now=1100.0, config=cfg)
        assert allowed_btc is False  # only 100s held
        assert allowed_eth is False  # only 50s held


# ── Cooldown gate (entry side) ────────────────────────────────────────────

class TestCooldown:
    def test_entry_allowed_no_exit_recorded(self):
        """No exit recorded → allow entry."""
        allowed, _ = check_entry_allowed("BTCUSDT", "LONG", now=1000.0)
        assert allowed is True

    def test_entry_vetoed_within_cooldown(self):
        """Entry within cooldown_seconds of exit → veto."""
        cfg = ChurnConfig(cooldown_seconds=300)
        record_exit("BTCUSDT", "LONG", ts=1000.0)
        allowed, reason = check_entry_allowed(
            "BTCUSDT", "LONG", now=1100.0, config=cfg,
        )
        assert allowed is False
        assert "cooldown_veto" in reason
        assert "200s remaining" in reason

    def test_entry_allowed_after_cooldown(self):
        """Entry after cooldown_seconds → allow."""
        cfg = ChurnConfig(cooldown_seconds=300)
        record_exit("BTCUSDT", "LONG", ts=1000.0)
        allowed, _ = check_entry_allowed(
            "BTCUSDT", "LONG", now=1400.0, config=cfg,
        )
        assert allowed is True

    def test_cooldown_clears_after_expiry(self):
        """After cooldown expires, exit time is cleared from state."""
        cfg = ChurnConfig(cooldown_seconds=60)
        record_exit("BTCUSDT", "LONG", ts=1000.0)
        # First check at 1070s — cooldown expired
        allowed, _ = check_entry_allowed(
            "BTCUSDT", "LONG", now=1070.0, config=cfg,
        )
        assert allowed is True
        # State should be cleared
        snap = get_state_snapshot()
        assert "BTCUSDT_LONG" not in snap["active_cooldowns"]

    def test_different_sides_independent(self):
        """LONG and SHORT cooldowns are tracked independently."""
        cfg = ChurnConfig(cooldown_seconds=300)
        record_exit("BTCUSDT", "LONG", ts=1000.0)
        # SHORT was never exited → allowed
        allowed_short, _ = check_entry_allowed(
            "BTCUSDT", "SHORT", now=1050.0, config=cfg,
        )
        assert allowed_short is True
        # LONG is still cooling → vetoed
        allowed_long, _ = check_entry_allowed(
            "BTCUSDT", "LONG", now=1050.0, config=cfg,
        )
        assert allowed_long is False


# ── State recording ───────────────────────────────────────────────────────

class TestStateRecording:
    def test_record_entry(self):
        record_entry("BTCUSDT", "LONG", ts=1000.0)
        snap = get_state_snapshot()
        assert "BTCUSDT_LONG" in snap["active_entries"]
        assert snap["active_entries"]["BTCUSDT_LONG"]["entry_ts"] == 1000.0

    def test_record_exit_clears_entry(self):
        record_entry("BTCUSDT", "LONG", ts=1000.0)
        record_exit("BTCUSDT", "LONG", ts=1100.0)
        snap = get_state_snapshot()
        assert "BTCUSDT_LONG" not in snap["active_entries"]
        assert "BTCUSDT_LONG" in snap["active_cooldowns"]
        assert snap["active_cooldowns"]["BTCUSDT_LONG"]["exit_ts"] == 1100.0

    def test_reset_state(self):
        record_entry("BTCUSDT", "LONG", ts=1000.0)
        record_exit("ETHUSDT", "SHORT", ts=1100.0)
        reset_state()
        snap = get_state_snapshot()
        assert snap["active_entries"] == {}
        assert snap["active_cooldowns"] == {}


# ── Full cycle integration ────────────────────────────────────────────────

class TestFullCycle:
    def test_entry_exit_entry_cycle(self):
        """Full cycle: enter → wait → exit → wait → re-enter."""
        cfg = ChurnConfig(min_hold_seconds=60, cooldown_seconds=120)

        # Enter at t=0
        record_entry("BTCUSDT", "LONG", ts=0.0)

        # Try exit at t=30 → vetoed (under hold)
        allowed, _ = check_exit_allowed(
            "BTCUSDT", "LONG", now=30.0, config=cfg,
        )
        assert allowed is False

        # Exit at t=90 → allowed (past hold)
        allowed, _ = check_exit_allowed(
            "BTCUSDT", "LONG", now=90.0, config=cfg,
        )
        assert allowed is True
        record_exit("BTCUSDT", "LONG", ts=90.0)

        # Try re-entry at t=150 → vetoed (under cooldown)
        allowed, _ = check_entry_allowed(
            "BTCUSDT", "LONG", now=150.0, config=cfg,
        )
        assert allowed is False

        # Re-entry at t=250 → allowed (past cooldown)
        allowed, _ = check_entry_allowed(
            "BTCUSDT", "LONG", now=250.0, config=cfg,
        )
        assert allowed is True

    def test_churn_scenario_regime_flips_bypass_hold(self):
        """REGIME_FLIP exits bypass min_hold — they are doctrine-authority exits.
        
        Previously these were vetoed (Feb 12 churn scenario). After v7.9-KS,
        regime-flip exits must NEVER be blocked by the churn guard.
        """
        cfg = ChurnConfig(min_hold_seconds=120, cooldown_seconds=300)
        record_entry("ETHUSDT", "LONG", ts=1000.0)

        # Try 10 REGIME_FLIP exits, 60s apart — all should be allowed
        vetoed = 0
        for i in range(10):
            t = 1060.0 + i * 60  # 1060, 1120, 1180...
            allowed, _ = check_exit_allowed(
                "ETHUSDT", "LONG", exit_reason="REGIME_FLIP",
                now=t, config=cfg,
            )
            if not allowed:
                vetoed += 1

        assert vetoed == 0  # REGIME_FLIP always bypasses

    def test_churn_scenario_non_safety_exits_still_held(self):
        """Non-safety exit reasons (e.g. TIME_STOP) still respect min_hold."""
        cfg = ChurnConfig(min_hold_seconds=120, cooldown_seconds=300)
        record_entry("ETHUSDT", "LONG", ts=1000.0)

        vetoed = 0
        for i in range(10):
            t = 1060.0 + i * 60
            allowed, _ = check_exit_allowed(
                "ETHUSDT", "LONG", exit_reason="TIME_STOP",
                now=t, config=cfg,
            )
            if not allowed:
                vetoed += 1

        # First exit at 1060 (held 60s < 120s): vetoed
        # Second at 1120 (held 120s): allowed
        assert vetoed == 1


# ── Bypass reason coverage ────────────────────────────────────────────────

class TestBypassReasons:
    def test_bypass_reasons_frozen(self):
        # Raw doctrine_kernel values
        assert "CRISIS_OVERRIDE" in HOLD_BYPASS_REASONS
        assert "SEATBELT" in HOLD_BYPASS_REASONS
        assert "REGIME_FLIP" in HOLD_BYPASS_REASONS
        assert "STOP_LOSS_SEATBELT" in HOLD_BYPASS_REASONS
        assert "REGIME_CONFIDENCE_COLLAPSE" in HOLD_BYPASS_REASONS
        # Canonical exit_reason_map values
        assert "CRISIS" in HOLD_BYPASS_REASONS
        assert "REGIME_CHANGE" in HOLD_BYPASS_REASONS

    def test_case_insensitive_bypass(self):
        """Exit reason is converted to upper for matching."""
        cfg = ChurnConfig(min_hold_seconds=120)
        record_entry("BTC", "LONG", ts=0.0)
        allowed, _ = check_exit_allowed(
            "BTC", "LONG", exit_reason="crisis_override",
            now=10.0, config=cfg,
        )
        assert allowed is True
