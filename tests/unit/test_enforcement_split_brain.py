# tests/unit/test_enforcement_split_brain.py
"""
Phase C.1 — Split-Brain Detection Tests

Tests the split-brain counter that detects divergence between
B.5 rehearsal and C.1 enforcement verdicts.

Expected invariant: split_brain_count ≈ 0.
Nonzero = the two code paths have drifted.

Cases:
  1. Both agree (no split brain) — permit present, both pass
  2. Both agree (no split brain) — permit absent, both block
  3. Split brain: enforcement denies, rehearsal allows
  4. rehearsal_would_block=None → skipped (no comparison)
  5. Counter increments on repeated divergences
  6. Metrics serialised via to_dict() include split-brain fields
  7. reset_rehearsal clears split-brain counters
"""
from __future__ import annotations

import pytest

from execution.enforcement_rehearsal import (
    record_split_brain_check,
    get_enforcement_metrics,
    EnforcementMetrics,
    init_enforcement,
    reset_rehearsal,
    init_rehearsal,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_state():
    """Reset module state before/after each test."""
    reset_rehearsal()
    yield
    reset_rehearsal()


# ---------------------------------------------------------------------------
# Test: both agree — no split brain
# ---------------------------------------------------------------------------

class TestNoSplitBrain:
    """When rehearsal and enforcement agree, split_brain_count stays 0."""

    def test_both_pass(self):
        """rehearsal=allow (would_block=False), enforcement=allow (denied=False)."""
        record_split_brain_check(
            symbol="BTCUSDT",
            direction="BUY",
            rehearsal_would_block=False,
            enforcement_denied=False,
        )
        m = get_enforcement_metrics()
        assert m["split_brain_count"] == 0
        assert m["last_split_brain_ts"] == ""
        assert m["last_split_brain_symbol"] == ""

    def test_both_block(self):
        """rehearsal=block (would_block=True), enforcement=deny (denied=True)."""
        record_split_brain_check(
            symbol="ETHUSDT",
            direction="SELL",
            rehearsal_would_block=True,
            enforcement_denied=True,
        )
        m = get_enforcement_metrics()
        assert m["split_brain_count"] == 0

    def test_enforcement_allows_rehearsal_blocks(self):
        """Rehearsal blocks but enforcement allows — not flagged as split-brain.
        Split-brain specifically tracks enforcement-denied + rehearsal-allowed."""
        record_split_brain_check(
            symbol="SOLUSDT",
            direction="BUY",
            rehearsal_would_block=True,
            enforcement_denied=False,
        )
        m = get_enforcement_metrics()
        assert m["split_brain_count"] == 0


# ---------------------------------------------------------------------------
# Test: split brain detected
# ---------------------------------------------------------------------------

class TestSplitBrainDetected:
    """Enforcement denied but rehearsal would have allowed → split brain."""

    def test_single_divergence(self):
        record_split_brain_check(
            symbol="BTCUSDT",
            direction="BUY",
            rehearsal_would_block=False,
            enforcement_denied=True,
        )
        m = get_enforcement_metrics()
        assert m["split_brain_count"] == 1
        assert m["last_split_brain_symbol"] == "BTCUSDT"
        assert m["last_split_brain_ts"] != ""

    def test_counter_increments(self):
        """Multiple divergences increment the counter."""
        for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            record_split_brain_check(
                symbol=sym,
                direction="BUY",
                rehearsal_would_block=False,
                enforcement_denied=True,
            )
        m = get_enforcement_metrics()
        assert m["split_brain_count"] == 3
        assert m["last_split_brain_symbol"] == "SOLUSDT"

    def test_direction_normalised_in_last_symbol(self):
        """Symbol stored upper-cased."""
        record_split_brain_check(
            symbol="btcusdt",
            direction="buy",
            rehearsal_would_block=False,
            enforcement_denied=True,
        )
        m = get_enforcement_metrics()
        assert m["last_split_brain_symbol"] == "BTCUSDT"


# ---------------------------------------------------------------------------
# Test: rehearsal_would_block=None → skip
# ---------------------------------------------------------------------------

class TestRehearsalNone:
    """When rehearsal didn't run (None), split-brain check is a no-op."""

    def test_none_no_count(self):
        record_split_brain_check(
            symbol="XRPUSDT",
            direction="BUY",
            rehearsal_would_block=None,
            enforcement_denied=True,
        )
        m = get_enforcement_metrics()
        assert m["split_brain_count"] == 0
        assert m["last_split_brain_ts"] == ""

    def test_none_enforcement_allows(self):
        record_split_brain_check(
            symbol="XRPUSDT",
            direction="BUY",
            rehearsal_would_block=None,
            enforcement_denied=False,
        )
        m = get_enforcement_metrics()
        assert m["split_brain_count"] == 0


# ---------------------------------------------------------------------------
# Test: to_dict serialisation
# ---------------------------------------------------------------------------

class TestMetricsSerialization:
    """EnforcementMetrics.to_dict() includes split-brain fields."""

    def test_default_fields_present(self):
        m = EnforcementMetrics()
        d = m.to_dict()
        assert "split_brain_count" in d
        assert "last_split_brain_ts" in d
        assert "last_split_brain_symbol" in d
        assert d["split_brain_count"] == 0
        assert d["last_split_brain_ts"] == ""
        assert d["last_split_brain_symbol"] == ""

    def test_after_divergence(self):
        record_split_brain_check(
            symbol="AVAXUSDT",
            direction="SELL",
            rehearsal_would_block=False,
            enforcement_denied=True,
        )
        d = get_enforcement_metrics()
        assert d["split_brain_count"] == 1
        assert d["last_split_brain_symbol"] == "AVAXUSDT"


# ---------------------------------------------------------------------------
# Test: reset clears split-brain counters
# ---------------------------------------------------------------------------

class TestResetClearsSplitBrain:
    """reset_rehearsal() zeros the split-brain state."""

    def test_reset_clears(self):
        record_split_brain_check(
            symbol="DOTUSDT",
            direction="BUY",
            rehearsal_would_block=False,
            enforcement_denied=True,
        )
        m = get_enforcement_metrics()
        assert m["split_brain_count"] == 1

        reset_rehearsal()

        m = get_enforcement_metrics()
        assert m["split_brain_count"] == 0
        assert m["last_split_brain_ts"] == ""
        assert m["last_split_brain_symbol"] == ""


# ---------------------------------------------------------------------------
# Test: fail-open — exceptions don't propagate
# ---------------------------------------------------------------------------

class TestFailOpen:
    """record_split_brain_check never raises."""

    def test_bad_symbol_type(self):
        """Even weird inputs don't raise."""
        # Should not raise — fail-open catches everything
        record_split_brain_check(
            symbol=12345,  # type: ignore
            direction="BUY",
            rehearsal_would_block=False,
            enforcement_denied=True,
        )
        # If it raised, test would fail. If it swallowed, count may or may not
        # increment (depends on whether .upper() works on int), but no crash.
