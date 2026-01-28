"""
Unit tests for DLE Shadow Gate (Phase A).

CI-safe: no file I/O required when write_logs=False.
"""

import pytest
from execution.dle_shadow import (
    shadow_build_chain,
    derive_request_id,
    derive_decision_id,
    derive_permit_id,
    hash_snapshot,
    DLEShadowWriter,
    reset_shadow_writer,
)


class TestDeterministicIds:
    """IDs must be deterministic for replay."""

    def test_request_id_with_attempt_id(self):
        """When attempt_id provided, use it directly."""
        rid = derive_request_id(
            attempt_id="sig_abc123",
            request_payload={"symbol": "ETHUSDT"},
        )
        assert rid == "sig_abc123"

    def test_request_id_without_attempt_id(self):
        """When no attempt_id, derive deterministically from payload."""
        payload = {"symbol": "ETHUSDT", "side": "LONG", "qty": 0.1}
        rid1 = derive_request_id(attempt_id=None, request_payload=payload)
        rid2 = derive_request_id(attempt_id=None, request_payload=payload)
        assert rid1 == rid2
        assert rid1.startswith("REQ_")

    def test_decision_id_deterministic(self):
        """Same inputs produce same decision_id."""
        id1 = derive_decision_id(
            phase_id="CYCLE_004",
            action_class="ENTRY",
            constraints={"verdict": "ALLOW"},
            policy_version="v1.0",
        )
        id2 = derive_decision_id(
            phase_id="CYCLE_004",
            action_class="ENTRY",
            constraints={"verdict": "ALLOW"},
            policy_version="v1.0",
        )
        assert id1 == id2
        assert id1.startswith("DEC_")

    def test_decision_id_differs_on_constraints(self):
        """Different constraints produce different decision_id."""
        id1 = derive_decision_id(
            phase_id="CYCLE_004",
            action_class="ENTRY",
            constraints={"verdict": "ALLOW"},
            policy_version="v1.0",
        )
        id2 = derive_decision_id(
            phase_id="CYCLE_004",
            action_class="ENTRY",
            constraints={"verdict": "DENY", "denial_code": "REGIME_STALE"},
            policy_version="v1.0",
        )
        assert id1 != id2

    def test_permit_id_deterministic(self):
        """Same inputs produce same permit_id."""
        id1 = derive_permit_id(
            decision_id="DEC_abc123",
            request_id="sig_xyz789",
            issued_at_iso="2026-01-28T12:00:00+00:00",
        )
        id2 = derive_permit_id(
            decision_id="DEC_abc123",
            request_id="sig_xyz789",
            issued_at_iso="2026-01-28T12:00:00+00:00",
        )
        assert id1 == id2
        assert id1.startswith("PERM_")


class TestSnapshotHashes:
    """Snapshot hashes must handle missing state."""

    def test_hash_none_returns_sentinel(self):
        """Missing state hashes to sentinel."""
        h = hash_snapshot("positions_state", None)
        assert "MISSING" in h
        assert h.startswith("positions_state_MISSING_")

    def test_hash_dict_deterministic(self):
        """Same dict produces same hash."""
        state = {"symbol": "ETHUSDT", "qty": 0.08}
        h1 = hash_snapshot("test", state)
        h2 = hash_snapshot("test", state)
        assert h1 == h2

    def test_hash_dict_order_independent(self):
        """Dict key order doesn't affect hash (sorted keys)."""
        state1 = {"a": 1, "b": 2}
        state2 = {"b": 2, "a": 1}
        h1 = hash_snapshot("test", state1)
        h2 = hash_snapshot("test", state2)
        assert h1 == h2


class TestShadowBuildChainDisabled:
    """When disabled, shadow_build_chain returns None."""

    def test_disabled_returns_none(self):
        """When enabled=False, all IDs are None."""
        rid, did, pid = shadow_build_chain(
            enabled=False,
            write_logs=False,
            writer=None,
            attempt_id="sig_test",
            requested_action="OPEN_POSITION",
            symbol="ETHUSDT",
            side="LONG",
            strategy="vol_target",
            qty_intent=0.1,
            context={},
            phase_id="CYCLE_004",
            action_class="ENTRY",
            policy_version="v1.0",
            scope={},
            constraints={},
            risk={},
            authority_source="DOCTRINE",
        )
        assert rid is None
        assert did is None
        assert pid is None


class TestShadowBuildChainEnabled:
    """When enabled, shadow_build_chain generates IDs."""

    def test_enabled_returns_ids(self):
        """When enabled=True, IDs are generated."""
        rid, did, pid = shadow_build_chain(
            enabled=True,
            write_logs=False,
            writer=None,
            attempt_id="sig_test123",
            requested_action="OPEN_POSITION",
            symbol="ETHUSDT",
            side="LONG",
            strategy="vol_target",
            qty_intent=0.08,
            context={},
            phase_id="CYCLE_004",
            action_class="ENTRY",
            policy_version="v1.0",
            scope={"symbols": ["ETHUSDT"]},
            constraints={"verdict": "ALLOW"},
            risk={"mode": "shadow"},
            authority_source="DOCTRINE",
        )
        assert rid == "sig_test123"
        assert did.startswith("DEC_")
        assert pid.startswith("PERM_")

    def test_deterministic_decision_id(self):
        """Same decision inputs produce same decision_id across calls."""
        ctx = {"positions_state_hash": "pos_aaa", "regime_state_hash": "reg_bbb"}
        
        rid1, did1, pid1 = shadow_build_chain(
            enabled=True,
            write_logs=False,
            writer=None,
            attempt_id=None,
            requested_action="OPEN_POSITION",
            symbol="ETHUSDT",
            side="LONG",
            strategy="vol_target",
            qty_intent=0.1,
            context=ctx,
            phase_id="CYCLE_004",
            action_class="ENTRY",
            policy_version="v1.0",
            scope={"symbols": ["ETHUSDT"]},
            constraints={"verdict": "ALLOW"},
            risk={"mode": "shadow"},
            authority_source="DOCTRINE",
        )
        
        rid2, did2, pid2 = shadow_build_chain(
            enabled=True,
            write_logs=False,
            writer=None,
            attempt_id=None,
            requested_action="OPEN_POSITION",
            symbol="ETHUSDT",
            side="LONG",
            strategy="vol_target",
            qty_intent=0.1,
            context=ctx,
            phase_id="CYCLE_004",
            action_class="ENTRY",
            policy_version="v1.0",
            scope={"symbols": ["ETHUSDT"]},
            constraints={"verdict": "ALLOW"},
            risk={"mode": "shadow"},
            authority_source="DOCTRINE",
        )
        
        # Request IDs match (same payload, no attempt_id)
        assert rid1 == rid2
        # Decision IDs match (same constraints + phase + policy)
        assert did1 == did2
        # Permit IDs differ (include timestamp)
        assert pid1.startswith("PERM_")
        assert pid2.startswith("PERM_")


class TestFailOpen:
    """Shadow gate must fail open on errors."""

    def test_log_write_failure_does_not_raise(self):
        """Log write failure warns but doesn't crash."""

        class BadWriter:
            def write(self, event):
                raise PermissionError("nope")

        # Should not raise
        rid, did, pid = shadow_build_chain(
            enabled=True,
            write_logs=True,
            writer=BadWriter(),
            attempt_id="sig_failopen",
            requested_action="CLOSE_POSITION",
            symbol="ETHUSDT",
            side="LONG",
            strategy="vol_target",
            qty_intent=0.08,
            context={},
            phase_id="CYCLE_004",
            action_class="EXIT",
            policy_version="v1.0",
            scope={},
            constraints={"exit_reason": "REGIME_FLIP"},
            risk={},
            authority_source="DOCTRINE",
        )
        
        # IDs still returned even when write fails
        assert rid == "sig_failopen"
        assert did.startswith("DEC_")
        assert pid.startswith("PERM_")

    def test_writer_tracks_failure_count(self):
        """Writer tracks write failures for monitoring."""
        writer = DLEShadowWriter()
        # Can't easily trigger failure without file system, but check property exists
        assert writer.write_failure_count == 0


class TestExitPath:
    """Exit path observations work correctly."""

    def test_exit_action_class(self):
        """Exit observations use EXIT action class."""
        rid, did, pid = shadow_build_chain(
            enabled=True,
            write_logs=False,
            writer=None,
            attempt_id="sig_exit_test",
            requested_action="CLOSE_POSITION",
            symbol="ETHUSDT",
            side="LONG",
            strategy="vol_target",
            qty_intent=0.08,
            context={"exit_reason": "REGIME_FLIP"},
            phase_id="CYCLE_004",
            action_class="EXIT",
            policy_version="v1.0",
            scope={"symbols": ["ETHUSDT"]},
            constraints={"exit_reason": "REGIME_FLIP"},
            risk={"mode": "shadow"},
            authority_source="DOCTRINE",
        )
        
        assert rid == "sig_exit_test"
        assert did.startswith("DEC_")
        assert pid.startswith("PERM_")


class TestGlobalWriter:
    """Global writer instance management."""

    def test_reset_shadow_writer(self):
        """reset_shadow_writer clears global instance."""
        from execution.dle_shadow import get_shadow_writer, reset_shadow_writer
        
        w1 = get_shadow_writer()
        reset_shadow_writer()
        w2 = get_shadow_writer()
        
        # Different instances after reset
        assert w1 is not w2
