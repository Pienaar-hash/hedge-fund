"""
Kill Switch × Doctrine Exit — Invariant Tests (v7.9-KS)

Safety axiom:
  KILL_SWITCH may only block risk-increasing orders.
  It must NEVER block reduceOnly exits issued under doctrine authority.

These tests verify:
  1. Genuine doctrine exits (reduceOnly=True + doctrine_exit=True) bypass kill switch
  2. Spoofed doctrine_exit=True WITHOUT reduceOnly is still blocked
  3. reduceOnly=True WITHOUT doctrine_exit is still blocked
  4. Normal entries are blocked as usual
"""

import pytest


def _is_doctrine_exit(intent: dict) -> bool:
    """Reproduce the exact guard from executor_live._send_order (line ~3687)."""
    return bool(intent.get("doctrine_exit")) and bool(intent.get("reduceOnly"))


class TestKillSwitchDoctrineExitGuard:
    """Unit-test the two-flag guard in isolation (no executor import needed)."""

    def test_genuine_doctrine_exit_bypasses(self):
        """reduceOnly=True + doctrine_exit=True → bypass kill switch."""
        intent = {
            "symbol": "SOLUSDT",
            "signal": "SELL",
            "reduceOnly": True,
            "doctrine_exit": True,
            "positionSide": "LONG",
            "quantity": 0.55,
            "metadata": {"exit_reason": "REGIME_CHANGE", "exit_source": "doctrine_kernel"},
        }
        assert _is_doctrine_exit(intent) is True

    def test_spoofed_doctrine_exit_without_reduce_only_blocked(self):
        """doctrine_exit=True but reduceOnly missing → kill switch blocks.

        This prevents an entry path from accidentally (or maliciously) setting
        doctrine_exit=True to escape the kill switch.
        """
        intent = {
            "symbol": "SOLUSDT",
            "signal": "BUY",
            "doctrine_exit": True,
            # reduceOnly intentionally absent
            "positionSide": "LONG",
            "quantity": 0.55,
        }
        assert _is_doctrine_exit(intent) is False

    def test_spoofed_doctrine_exit_with_reduce_only_false_blocked(self):
        """doctrine_exit=True + reduceOnly=False → still blocked."""
        intent = {
            "symbol": "SOLUSDT",
            "signal": "BUY",
            "doctrine_exit": True,
            "reduceOnly": False,
            "positionSide": "LONG",
            "quantity": 0.55,
        }
        assert _is_doctrine_exit(intent) is False

    def test_reduce_only_without_doctrine_exit_blocked(self):
        """reduceOnly=True but no doctrine_exit → kill switch blocks.

        Non-doctrine reduceOnly orders (e.g., manual partial close, TP)
        should still be subject to kill switch.
        """
        intent = {
            "symbol": "SOLUSDT",
            "signal": "SELL",
            "reduceOnly": True,
            # doctrine_exit intentionally absent
            "positionSide": "LONG",
            "quantity": 0.55,
        }
        assert _is_doctrine_exit(intent) is False

    def test_normal_entry_blocked(self):
        """Standard entry intent has neither flag → blocked by kill switch."""
        intent = {
            "symbol": "SOLUSDT",
            "signal": "BUY",
            "positionSide": "LONG",
            "quantity": 0.55,
        }
        assert _is_doctrine_exit(intent) is False

    def test_empty_intent_blocked(self):
        """Empty dict → blocked (safe default)."""
        assert _is_doctrine_exit({}) is False

    def test_none_values_blocked(self):
        """Explicit None values → blocked (falsy)."""
        intent = {"doctrine_exit": None, "reduceOnly": None}
        assert _is_doctrine_exit(intent) is False

    def test_string_true_considered_truthy(self):
        """String 'True' is truthy — accepted (matches bool() semantics)."""
        intent = {"doctrine_exit": "True", "reduceOnly": "True"}
        assert _is_doctrine_exit(intent) is True

    def test_zero_values_blocked(self):
        """Zero (0) is falsy — blocked."""
        intent = {"doctrine_exit": 0, "reduceOnly": 0}
        assert _is_doctrine_exit(intent) is False


class TestDedupBypassReasonCoverage:
    """Verify that DEDUP_BYPASS_REASONS covers both raw and canonical forms."""

    def test_bypass_reasons_include_both_layers(self):
        from execution.exit_dedup import DEDUP_BYPASS_REASONS

        # Raw doctrine_kernel enum values
        assert "REGIME_FLIP" in DEDUP_BYPASS_REASONS
        assert "CRISIS_OVERRIDE" in DEDUP_BYPASS_REASONS
        assert "STOP_LOSS_SEATBELT" in DEDUP_BYPASS_REASONS
        assert "REGIME_CONFIDENCE_COLLAPSE" in DEDUP_BYPASS_REASONS

        # Canonical exit_reason_map names
        assert "REGIME_CHANGE" in DEDUP_BYPASS_REASONS
        assert "CRISIS" in DEDUP_BYPASS_REASONS
        assert "SEATBELT" in DEDUP_BYPASS_REASONS

    def test_regime_flip_raw_bypasses_dedup(self):
        """Raw REGIME_FLIP (from candidate.exit_reason) bypasses dedup."""
        from execution.exit_dedup import (
            ExitDedupConfig,
            check_exit_dedup,
            record_exit_sent,
            reset_state,
        )
        import time

        reset_state()
        now = time.time()
        record_exit_sent("ETHUSDT", "LONG", "REGIME_FLIP", qty=0.1, now=now)
        ok, _ = check_exit_dedup(
            "ETHUSDT", "LONG", "REGIME_FLIP",
            current_qty=0.1, now=now + 10, config=ExitDedupConfig(),
        )
        assert ok is True
        reset_state()

    def test_regime_change_canonical_bypasses_dedup(self):
        """Canonical REGIME_CHANGE (from normalized intent) bypasses dedup."""
        from execution.exit_dedup import (
            ExitDedupConfig,
            check_exit_dedup,
            record_exit_sent,
            reset_state,
        )
        import time

        reset_state()
        now = time.time()
        record_exit_sent("ETHUSDT", "LONG", "REGIME_CHANGE", qty=0.1, now=now)
        ok, _ = check_exit_dedup(
            "ETHUSDT", "LONG", "REGIME_CHANGE",
            current_qty=0.1, now=now + 10, config=ExitDedupConfig(),
        )
        assert ok is True
        reset_state()
