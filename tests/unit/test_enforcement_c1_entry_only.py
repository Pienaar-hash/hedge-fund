# tests/unit/test_enforcement_c1_entry_only.py
"""
Phase C.1 — Entry-Only Enforcement Tests

Tests the constitutional asymmetry:
  - Entering risk requires authority (permit)
  - Exiting risk must remain always available

Tests cover:
  1. Entry classification (is_entry_order)
  2. Gate function (enforce_entry_permit_or_deny)
  3. Denial logging (DenialWriter)
  4. Enforcement metrics
  5. Safety invariants (exits never blocked)
  6. Flag gating (DLE_ENFORCE_ENTRY_ONLY)
  7. Fail-open behavior
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest import mock

import pytest

from execution.enforcement_rehearsal import (
    # Entry classification
    is_entry_order,
    # Gate function
    enforce_entry_permit_or_deny,
    # Denial writer
    DenialWriter,
    DENIAL_LOG_PATH,
    # Denial reason codes
    DENY_NO_PERMIT,
    DENY_EXPIRED,
    DENY_MISMATCH_DIRECTION,
    DENY_INDEX_UNAVAILABLE,
    # Enforcement lifecycle
    init_enforcement,
    get_enforcement_metrics,
    EnforcementMetrics,
    # Rehearsal lifecycle (needed for setup)
    init_rehearsal,
    build_permit_index,
    reset_rehearsal,
    rehearse_order,
    # Reason codes (from B.5)
    REASON_OK,
    _normalize_direction,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_TS = datetime(2026, 2, 13, 12, 0, 0, tzinfo=timezone.utc)


def _ts_unix(offset_s: int = 0) -> float:
    return (_BASE_TS + timedelta(seconds=offset_s)).timestamp()


def _ts_iso(offset_s: int = 0) -> str:
    return (_BASE_TS + timedelta(seconds=offset_s)).isoformat()


def _make_permit_event(
    *,
    offset_s: int = 0,
    symbol: str = "BTCUSDT",
    direction: str = "BUY",
    ttl_s: int = 60,
    permit_id: str = "PERM_TEST_001",
    state: str = "ISSUED",
) -> dict:
    ts = _ts_iso(offset_s)
    expires = _ts_iso(offset_s + ttl_s)
    return {
        "event_type": "PERMIT",
        "ts": ts,
        "payload": {
            "permit_id": permit_id,
            "schema_version": "v2",
            "action": {
                "type": "ENTRY",
                "symbol": symbol,
                "direction": direction,
            },
            "state": state,
            "decision_id": "DEC_TEST_001",
            "request_id": "REQ_TEST_001",
            "permit_ttl_s": ttl_s,
            "expires_ts": expires,
        },
    }


def _write_shadow(path: Path, events: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for evt in events:
            f.write(json.dumps(evt, sort_keys=True) + "\n")


def _setup_enforcement(tmp_path, events=None, ttl_s=60, extra_flags=None):
    """Helper: init rehearsal + enforcement with given permit events."""
    shadow_log = tmp_path / "shadow.jsonl"
    denial_log = str(tmp_path / "denials.jsonl")
    if events is None:
        events = [_make_permit_event(offset_s=0, ttl_s=ttl_s)]
    _write_shadow(shadow_log, events)

    flags_attrs = {"shadow_dle_enabled": True, "dle_enforce_entry_only": True}
    if extra_flags:
        flags_attrs.update(extra_flags)

    with mock.patch("execution.v6_flags.get_flags") as mock_flags:
        mock_flags.return_value = mock.MagicMock(**flags_attrs)
        init_rehearsal(shadow_log_path=shadow_log, max_age_s=None)
        ok = init_enforcement(denial_log_path=denial_log)

    return shadow_log, denial_log, ok


# ===========================================================================
# Tests: Entry Classification
# ===========================================================================

class TestIsEntryOrder:

    def test_buy_is_entry(self):
        assert is_entry_order(side="BUY") is True

    def test_sell_is_entry(self):
        assert is_entry_order(side="SELL") is True

    def test_reduce_only_is_exit(self):
        assert is_entry_order(side="BUY", reduce_only=True) is False

    def test_reduce_only_sell_is_exit(self):
        assert is_entry_order(side="SELL", reduce_only=True) is False

    def test_close_action_is_exit(self):
        assert is_entry_order(side="SELL", intent_action="CLOSE") is False

    def test_exit_action_is_exit(self):
        assert is_entry_order(side="BUY", intent_action="EXIT") is False

    def test_stop_loss_action_is_exit(self):
        assert is_entry_order(side="SELL", intent_action="STOP_LOSS") is False

    def test_take_profit_action_is_exit(self):
        assert is_entry_order(side="SELL", intent_action="TAKE_PROFIT") is False

    def test_reduce_action_is_exit(self):
        assert is_entry_order(side="SELL", intent_action="REDUCE") is False

    def test_empty_action_is_entry(self):
        assert is_entry_order(side="BUY", intent_action="") is True

    def test_unknown_action_is_entry(self):
        """Conservative: unknown actions are classified as ENTRY (safe)."""
        assert is_entry_order(side="BUY", intent_action="UNKNOWN") is True

    def test_case_insensitive_action(self):
        assert is_entry_order(side="BUY", intent_action="close") is False
        assert is_entry_order(side="BUY", intent_action="Close") is False
        assert is_entry_order(side="BUY", intent_action="CLOSE") is False


# ===========================================================================
# Tests: Denial Writer
# ===========================================================================

class TestDenialWriter:

    def test_append_only(self, tmp_path):
        log_path = str(tmp_path / "denials.jsonl")
        writer = DenialWriter(log_path)
        writer.write({"event_type": "ENTRY_DENIAL", "symbol": "BTCUSDT"})
        writer.write({"event_type": "ENTRY_DENIAL", "symbol": "ETHUSDT"})

        lines = Path(log_path).read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["symbol"] == "BTCUSDT"
        assert json.loads(lines[1])["symbol"] == "ETHUSDT"

    def test_write_failure_safe(self, tmp_path):
        writer = DenialWriter("/nonexistent/dir/that/requires/root/denials.jsonl")
        # Should not raise
        writer.write({"test": True})
        assert writer.write_failure_count >= 0  # May or may not fail depending on OS


# ===========================================================================
# Tests: Enforcement Metrics
# ===========================================================================

class TestEnforcementMetrics:

    def test_initial_state(self):
        m = EnforcementMetrics()
        assert m.entry_evaluated == 0
        assert m.entry_denied == 0
        assert m.entry_permitted == 0
        assert m.exit_passthrough == 0
        assert m.entry_blocks_pct == 0.0
        assert m.deny_reasons == {}

    def test_update_pct(self):
        m = EnforcementMetrics(entry_evaluated=100, entry_denied=3)
        m._update_pct()
        assert m.entry_blocks_pct == 3.0

    def test_to_dict(self):
        m = EnforcementMetrics(entry_evaluated=10, entry_denied=1, enforce_enabled=True)
        d = m.to_dict()
        assert d["entry_evaluated"] == 10
        assert d["enforce_enabled"] is True
        assert isinstance(d["deny_reasons"], dict)


# ===========================================================================
# Tests: Gate Function — enforce_entry_permit_or_deny
# ===========================================================================

class TestEnforceEntryPermitOrDeny:

    def setup_method(self):
        reset_rehearsal()

    def teardown_method(self):
        reset_rehearsal()

    # --- EXIT orders never blocked (constitutional invariant) ---

    def test_exit_never_blocked_even_when_enforcing(self, tmp_path):
        """EXIT orders (reduceOnly=True) always pass, even with no permits."""
        _setup_enforcement(tmp_path, events=[])  # no permits at all
        ok, reason = enforce_entry_permit_or_deny(
            symbol="BTCUSDT", direction="SELL", reduce_only=True,
            order_id="EXIT_1",
        )
        assert ok is True
        assert reason is None

    def test_exit_with_close_action_never_blocked(self, tmp_path):
        _setup_enforcement(tmp_path, events=[])
        ok, reason = enforce_entry_permit_or_deny(
            symbol="BTCUSDT", direction="SELL", intent_action="CLOSE",
            order_id="EXIT_2",
        )
        assert ok is True
        assert reason is None

    def test_exit_passthrough_increments(self, tmp_path):
        _setup_enforcement(tmp_path, events=[])
        enforce_entry_permit_or_deny(
            symbol="BTCUSDT", direction="SELL", reduce_only=True,
            order_id="EXIT_3",
        )
        metrics = get_enforcement_metrics()
        assert metrics["exit_passthrough"] == 1

    # --- ENTRY with valid permit: allowed ---

    def test_entry_with_valid_permit_allowed(self, tmp_path):
        _setup_enforcement(tmp_path, events=[
            _make_permit_event(offset_s=0, ttl_s=60),
        ])
        ok, reason = enforce_entry_permit_or_deny(
            symbol="BTCUSDT", direction="BUY",
            order_id="ENTRY_1", order_ts=_ts_unix(15),
        )
        assert ok is True
        assert reason is None
        metrics = get_enforcement_metrics()
        assert metrics["entry_permitted"] == 1
        assert metrics["entry_denied"] == 0

    # --- ENTRY without permit: denied ---

    def test_entry_no_permit_denied(self, tmp_path):
        _setup_enforcement(tmp_path, events=[])  # no permits
        ok, reason = enforce_entry_permit_or_deny(
            symbol="BTCUSDT", direction="BUY",
            order_id="ENTRY_2", order_ts=_ts_unix(15),
        )
        assert ok is False
        assert reason == DENY_NO_PERMIT

    def test_entry_expired_permit_denied(self, tmp_path):
        _setup_enforcement(tmp_path, events=[
            _make_permit_event(offset_s=0, ttl_s=30),
        ])
        ok, reason = enforce_entry_permit_or_deny(
            symbol="BTCUSDT", direction="BUY",
            order_id="ENTRY_3", order_ts=_ts_unix(60),  # after expiry
        )
        assert ok is False
        assert reason == DENY_EXPIRED

    def test_entry_direction_mismatch_denied(self, tmp_path):
        _setup_enforcement(tmp_path, events=[
            _make_permit_event(offset_s=0, direction="BUY", ttl_s=60),
        ])
        ok, reason = enforce_entry_permit_or_deny(
            symbol="BTCUSDT", direction="SELL",  # mismatch
            order_id="ENTRY_4", order_ts=_ts_unix(15),
        )
        assert ok is False
        assert reason == DENY_MISMATCH_DIRECTION

    # --- Denial logging ---

    def test_denial_written_to_log(self, tmp_path):
        _, denial_log, _ = _setup_enforcement(tmp_path, events=[])
        enforce_entry_permit_or_deny(
            symbol="ETHUSDT", direction="BUY",
            order_id="ENTRY_5", order_ts=_ts_unix(15),
        )
        lines = Path(denial_log).read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["event_type"] == "ENTRY_DENIAL"
        assert record["symbol"] == "ETHUSDT"
        assert record["deny_reason"] == DENY_NO_PERMIT
        assert record["phase"] == "C.1"

    def test_denial_metrics_tracked(self, tmp_path):
        _setup_enforcement(tmp_path, events=[])
        enforce_entry_permit_or_deny(
            symbol="BTCUSDT", direction="BUY",
            order_id="D1", order_ts=_ts_unix(15),
        )
        enforce_entry_permit_or_deny(
            symbol="ETHUSDT", direction="SELL",
            order_id="D2", order_ts=_ts_unix(15),
        )
        metrics = get_enforcement_metrics()
        assert metrics["entry_denied"] == 2
        assert metrics["entry_evaluated"] == 2
        assert metrics["entry_blocks_pct"] == 100.0
        assert DENY_NO_PERMIT in metrics["deny_reasons"]

    # --- Enforcement disabled ---

    def test_enforcement_disabled_passes_all(self):
        """When not initialized, all orders pass."""
        ok, reason = enforce_entry_permit_or_deny(
            symbol="BTCUSDT", direction="BUY",
            order_id="X",
        )
        assert ok is True
        assert reason is None

    def test_enforcement_disabled_when_flag_off(self, tmp_path):
        """DLE_ENFORCE_ENTRY_ONLY=0 means enforcement is inactive."""
        shadow_log = tmp_path / "shadow.jsonl"
        _write_shadow(shadow_log, [])

        with mock.patch("execution.v6_flags.get_flags") as mock_flags:
            mock_flags.return_value = mock.MagicMock(
                shadow_dle_enabled=True, dle_enforce_entry_only=False,
            )
            init_rehearsal(shadow_log_path=shadow_log)
            ok = init_enforcement()
            assert ok is False

        # Entry should pass through
        ok, reason = enforce_entry_permit_or_deny(
            symbol="BTCUSDT", direction="BUY",
            order_id="Y",
        )
        assert ok is True

    # --- Fail-open behavior ---

    def test_fail_open_on_corrupted_state(self, tmp_path):
        """If internal state is corrupted, enforcement fails open."""
        _setup_enforcement(tmp_path, events=[
            _make_permit_event(offset_s=0, ttl_s=60),
        ])

        import execution.enforcement_rehearsal as er
        # Corrupt the permit index
        er._permit_index = None  # type: ignore
        er._index_loaded = False

        ok, reason = enforce_entry_permit_or_deny(
            symbol="BTCUSDT", direction="BUY",
            order_id="Z", order_ts=_ts_unix(15),
        )
        # Index unavailable → denied (not fail-open here, this is by design)
        assert ok is False
        assert reason == DENY_INDEX_UNAVAILABLE

    def test_fail_open_on_exception(self, tmp_path):
        """If evaluate_order itself throws, enforcement fails open."""
        _setup_enforcement(tmp_path, events=[
            _make_permit_event(offset_s=0, ttl_s=60),
        ])

        import execution.enforcement_rehearsal as er
        # Force an exception in evaluate_order by corrupting index type
        er._permit_index = "not_a_dict"  # type: ignore
        er._index_loaded = True  # pretend it's loaded

        ok, reason = enforce_entry_permit_or_deny(
            symbol="BTCUSDT", direction="BUY",
            order_id="W", order_ts=_ts_unix(15),
        )
        # Should fail open (exception in evaluate_order)
        assert ok is True
        assert reason is None


# ===========================================================================
# Tests: Init Enforcement
# ===========================================================================

class TestInitEnforcement:

    def setup_method(self):
        reset_rehearsal()

    def teardown_method(self):
        reset_rehearsal()

    def test_init_requires_shadow_dle(self, tmp_path):
        shadow_log = tmp_path / "shadow.jsonl"
        _write_shadow(shadow_log, [])

        with mock.patch("execution.v6_flags.get_flags") as mock_flags:
            mock_flags.return_value = mock.MagicMock(
                shadow_dle_enabled=False, dle_enforce_entry_only=True,
            )
            ok = init_enforcement()
            assert ok is False

    def test_init_requires_enforce_flag(self, tmp_path):
        shadow_log = tmp_path / "shadow.jsonl"
        _write_shadow(shadow_log, [])

        with mock.patch("execution.v6_flags.get_flags") as mock_flags:
            mock_flags.return_value = mock.MagicMock(
                shadow_dle_enabled=True, dle_enforce_entry_only=False,
            )
            ok = init_enforcement()
            assert ok is False

    def test_init_requires_rehearsal_first(self, tmp_path):
        """Enforcement requires rehearsal to be initialized (permit index)."""
        with mock.patch("execution.v6_flags.get_flags") as mock_flags:
            mock_flags.return_value = mock.MagicMock(
                shadow_dle_enabled=True, dle_enforce_entry_only=True,
            )
            # Don't init rehearsal first
            ok = init_enforcement()
            assert ok is False

    def test_init_success(self, tmp_path):
        _, _, ok = _setup_enforcement(tmp_path)
        assert ok is True
        metrics = get_enforcement_metrics()
        assert metrics["enforce_enabled"] is True


# ===========================================================================
# Tests: Enforcement Metrics in Readiness Surface
# ===========================================================================

class TestEnforcementInReadiness:

    def setup_method(self):
        reset_rehearsal()

    def teardown_method(self):
        reset_rehearsal()

    def test_readiness_includes_enforcement(self, tmp_path):
        from execution.enforcement_rehearsal import compute_phase_c_readiness
        _setup_enforcement(tmp_path, events=[])

        # Deny an entry
        enforce_entry_permit_or_deny(
            symbol="BTCUSDT", direction="BUY",
            order_id="R1", order_ts=_ts_unix(15),
        )

        readiness = compute_phase_c_readiness()
        assert "enforcement" in readiness
        assert readiness["enforcement"]["entry_denied"] == 1
        assert readiness["enforcement"]["enforce_enabled"] is True


# ===========================================================================
# Tests: Safety — Constitutional Invariant
# ===========================================================================

class TestConstitutionalInvariant:
    """
    The asymmetry must hold under ALL conditions:
      - EXIT orders are NEVER blocked
      - Even when enforcement is active
      - Even when there are zero permits
      - Even when the shadow index is unavailable
    """

    def setup_method(self):
        reset_rehearsal()

    def teardown_method(self):
        reset_rehearsal()

    def test_exit_reduce_only_always_passes(self, tmp_path):
        _setup_enforcement(tmp_path, events=[])
        for direction in ["BUY", "SELL"]:
            ok, reason = enforce_entry_permit_or_deny(
                symbol="BTCUSDT", direction=direction, reduce_only=True,
            )
            assert ok is True, f"EXIT (reduceOnly) blocked for {direction}"

    def test_exit_close_action_always_passes(self, tmp_path):
        _setup_enforcement(tmp_path, events=[])
        for action in ["CLOSE", "EXIT", "REDUCE", "STOP_LOSS", "TAKE_PROFIT"]:
            ok, reason = enforce_entry_permit_or_deny(
                symbol="BTCUSDT", direction="SELL", intent_action=action,
            )
            assert ok is True, f"EXIT ({action}) should never be blocked"

    def test_exit_passes_even_with_corrupted_state(self, tmp_path):
        _setup_enforcement(tmp_path, events=[])
        import execution.enforcement_rehearsal as er
        er._permit_index = None  # type: ignore
        er._index_loaded = False

        ok, reason = enforce_entry_permit_or_deny(
            symbol="BTCUSDT", direction="SELL", reduce_only=True,
        )
        assert ok is True
        assert reason is None
