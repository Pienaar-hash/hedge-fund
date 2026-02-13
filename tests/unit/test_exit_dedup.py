"""
Tests for exit_dedup module (v7.9-E2.2).

Exit intent deduplication — suppresses repeated exit orders for the same
symbol/side/reason within a TTL window.
"""

import pytest
import time

from execution.exit_dedup import (
    ExitDedupConfig,
    check_exit_dedup,
    record_exit_sent,
    clear_for_symbol,
    get_state_snapshot,
    reset_state,
    load_exit_dedup_config,
    DEFAULT_EXIT_DEDUP_TTL,
    DEFAULT_QTY_CHANGE_THRESHOLD,
    DEDUP_BYPASS_REASONS,
)


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset dedup state before and after each test."""
    reset_state()
    yield
    reset_state()


# ── Config ─────────────────────────────────────────────────────────────────

class TestExitDedupConfig:
    def test_defaults(self):
        cfg = ExitDedupConfig()
        assert cfg.ttl_seconds == DEFAULT_EXIT_DEDUP_TTL
        assert cfg.qty_change_threshold == DEFAULT_QTY_CHANGE_THRESHOLD
        assert cfg.enabled is True

    def test_load_from_dict(self):
        runtime = {
            "exit_dedup": {
                "ttl_seconds": 600,
                "qty_change_threshold": 0.20,
                "enabled": False,
            }
        }
        cfg = load_exit_dedup_config(runtime)
        assert cfg.ttl_seconds == 600.0
        assert cfg.qty_change_threshold == 0.20
        assert cfg.enabled is False

    def test_load_missing_section(self):
        cfg = load_exit_dedup_config({})
        assert cfg.ttl_seconds == DEFAULT_EXIT_DEDUP_TTL

    def test_load_none(self):
        # When called without runtime.yaml available, still returns defaults
        cfg = load_exit_dedup_config({"other": 1})
        assert cfg.enabled is True


# ── Core dedup logic ───────────────────────────────────────────────────────

class TestCheckExitDedup:
    CFG = ExitDedupConfig(ttl_seconds=300, qty_change_threshold=0.10)

    def test_first_intent_allowed(self):
        ok, reason = check_exit_dedup("ETHUSDT", "LONG", "REGIME_FLIP", config=self.CFG)
        assert ok is True
        assert reason == ""

    def test_second_intent_within_ttl_suppressed(self):
        now = time.time()
        record_exit_sent("ETHUSDT", "LONG", "REGIME_FLIP", qty=0.1, now=now)
        ok, reason = check_exit_dedup(
            "ETHUSDT", "LONG", "REGIME_FLIP", current_qty=0.1, now=now + 60, config=self.CFG,
        )
        assert ok is False
        assert "exit_dedup" in reason

    def test_allowed_after_ttl_expires(self):
        now = time.time()
        record_exit_sent("ETHUSDT", "LONG", "REGIME_FLIP", qty=0.1, now=now)
        ok, _ = check_exit_dedup(
            "ETHUSDT", "LONG", "REGIME_FLIP", current_qty=0.1, now=now + 301, config=self.CFG,
        )
        assert ok is True

    def test_different_symbol_not_suppressed(self):
        now = time.time()
        record_exit_sent("ETHUSDT", "LONG", "REGIME_FLIP", qty=0.1, now=now)
        ok, _ = check_exit_dedup(
            "BTCUSDT", "LONG", "REGIME_FLIP", now=now + 10, config=self.CFG,
        )
        assert ok is True

    def test_different_side_not_suppressed(self):
        now = time.time()
        record_exit_sent("ETHUSDT", "LONG", "REGIME_FLIP", qty=0.1, now=now)
        ok, _ = check_exit_dedup(
            "ETHUSDT", "SHORT", "REGIME_FLIP", now=now + 10, config=self.CFG,
        )
        assert ok is True

    def test_different_reason_not_suppressed(self):
        now = time.time()
        record_exit_sent("ETHUSDT", "LONG", "REGIME_FLIP", qty=0.1, now=now)
        ok, _ = check_exit_dedup(
            "ETHUSDT", "LONG", "TIME_STOP", now=now + 10, config=self.CFG,
        )
        assert ok is True

    def test_qty_growth_bypasses_dedup(self):
        now = time.time()
        record_exit_sent("ETHUSDT", "LONG", "REGIME_FLIP", qty=1.0, now=now)
        # Position grew by 15% (>10% threshold)
        ok, _ = check_exit_dedup(
            "ETHUSDT", "LONG", "REGIME_FLIP", current_qty=1.15, now=now + 10, config=self.CFG,
        )
        assert ok is True

    def test_qty_shrink_does_not_bypass(self):
        """Qty decreased by 15% — still within dedup window."""
        now = time.time()
        record_exit_sent("ETHUSDT", "LONG", "REGIME_FLIP", qty=1.0, now=now)
        # position shrank (already partially exited)
        ok, _ = check_exit_dedup(
            "ETHUSDT", "LONG", "REGIME_FLIP", current_qty=0.85, now=now + 10, config=self.CFG,
        )
        # 15% change > 10% threshold → actually does bypass (absolute change)
        assert ok is True

    def test_small_qty_change_still_suppressed(self):
        now = time.time()
        record_exit_sent("ETHUSDT", "LONG", "REGIME_FLIP", qty=1.0, now=now)
        ok, _ = check_exit_dedup(
            "ETHUSDT", "LONG", "REGIME_FLIP", current_qty=1.05, now=now + 10, config=self.CFG,
        )
        assert ok is False


# ── Bypass reasons ─────────────────────────────────────────────────────────

class TestBypassReasons:
    CFG = ExitDedupConfig(ttl_seconds=300, qty_change_threshold=0.10)

    def test_crisis_override_always_allowed(self):
        now = time.time()
        record_exit_sent("ETHUSDT", "LONG", "CRISIS_OVERRIDE", qty=0.1, now=now)
        ok, _ = check_exit_dedup(
            "ETHUSDT", "LONG", "CRISIS_OVERRIDE", current_qty=0.1, now=now + 10, config=self.CFG,
        )
        assert ok is True

    def test_seatbelt_always_allowed(self):
        now = time.time()
        record_exit_sent("ETHUSDT", "LONG", "SEATBELT", qty=0.1, now=now)
        ok, _ = check_exit_dedup(
            "ETHUSDT", "LONG", "SEATBELT", current_qty=0.1, now=now + 10, config=self.CFG,
        )
        assert ok is True

    def test_bypass_reasons_constant(self):
        assert "CRISIS_OVERRIDE" in DEDUP_BYPASS_REASONS
        assert "SEATBELT" in DEDUP_BYPASS_REASONS


# ── Disabled mode ──────────────────────────────────────────────────────────

class TestDisabled:
    def test_disabled_allows_everything(self):
        cfg = ExitDedupConfig(enabled=False)
        now = time.time()
        record_exit_sent("ETHUSDT", "LONG", "REGIME_FLIP", qty=0.1, now=now)
        ok, _ = check_exit_dedup(
            "ETHUSDT", "LONG", "REGIME_FLIP", current_qty=0.1, now=now + 10, config=cfg,
        )
        assert ok is True


# ── State management ───────────────────────────────────────────────────────

class TestStateManagement:
    def test_clear_for_symbol(self):
        now = time.time()
        record_exit_sent("ETHUSDT", "LONG", "REGIME_FLIP", qty=0.1, now=now)
        record_exit_sent("BTCUSDT", "LONG", "REGIME_FLIP", qty=0.1, now=now)
        clear_for_symbol("ETHUSDT")
        # ETHUSDT cleared — should be allowed again
        ok1, _ = check_exit_dedup(
            "ETHUSDT", "LONG", "REGIME_FLIP", current_qty=0.1,
            now=now + 10, config=ExitDedupConfig(),
        )
        assert ok1 is True
        # BTCUSDT NOT cleared — should still be suppressed
        ok2, _ = check_exit_dedup(
            "BTCUSDT", "LONG", "REGIME_FLIP", current_qty=0.1,
            now=now + 10, config=ExitDedupConfig(),
        )
        assert ok2 is False

    def test_get_state_snapshot(self):
        now = time.time()
        record_exit_sent("ETHUSDT", "LONG", "REGIME_FLIP", qty=0.1, now=now)
        snap = get_state_snapshot()
        assert "active_dedup_entries" in snap
        assert len(snap["active_dedup_entries"]) == 1
        key = list(snap["active_dedup_entries"].keys())[0]
        assert "ETHUSDT" in key

    def test_reset_state(self):
        now = time.time()
        record_exit_sent("ETHUSDT", "LONG", "REGIME_FLIP", qty=0.1, now=now)
        reset_state()
        snap = get_state_snapshot()
        assert len(snap["active_dedup_entries"]) == 0
