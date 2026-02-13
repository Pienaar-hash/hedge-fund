"""
Phase B.3 tests: DLE shadow permit enrichment.

Covers:
- PERMIT v2 emitted on ALLOW paths (action, TTL, scope_snapshot, provenance)
- No PERMIT emitted on DENY paths
- LINK.permit_id is null on DENY
- Backward compat (no permit enrichment → v1 schema)
- Deterministic permit IDs with enrichment
- TTL computation (expires_ts)
- Append-only property
- Fail-open on enrichment errors
"""

import json
from datetime import datetime, timezone, timedelta

import pytest
from execution.dle_shadow import (
    SCHEMA_VERSION,
    SCHEMA_VERSION_V2,
    DEFAULT_ENTRY_PERMIT_TTL_S,
    DEFAULT_EXIT_PERMIT_TTL_S,
    DLEShadowEvent,
    shadow_build_chain,
    _compute_expires_ts,
    _stable_json,
)


# ─── Helpers ────────────────────────────────────────────────────────────


def _base_kwargs(**overrides):
    """Minimal valid kwargs for shadow_build_chain()."""
    base = dict(
        enabled=True,
        write_logs=True,
        writer=None,
        attempt_id="sig_test_b3",
        requested_action="ENTRY",
        symbol="ETHUSDT",
        side="LONG",
        strategy="vol_target",
        qty_intent=100.0,
        context={},
        phase_id="CYCLE_004",
        action_class="ENTRY_ALLOW",
        policy_version="v7.9",
        scope={"symbol": "ETHUSDT"},
        constraints={"verdict": "ALLOW"},
        risk={},
        authority_source="DOCTRINE",
    )
    base.update(overrides)
    return base


class _CapturingWriter:
    """In-memory shadow writer that captures events for assertion."""

    def __init__(self):
        self.events: list[dict] = []

    def write(self, event: DLEShadowEvent):
        self.events.append(json.loads(_stable_json({
            "schema_version": event.schema_version,
            "event_type": event.event_type,
            "ts": event.ts,
            "payload": event.payload,
        })))


# Full B.3 enrichment for an ALLOW path
PERMIT_ENRICHMENT = dict(
    verdict="PERMIT",
    doctrine_verdict="DOCTRINE_ALLOW",
    context_snapshot={
        "regime": "TREND_UP",
        "regime_confidence": 0.85,
        "nav_usd": 1234.56,
        "positions_hash": "pos_abc123",
        "scores_hash": "scores_def456",
    },
    provenance={
        "engine_version": "v7.9",
        "git_sha": "abc123",
        "docs_version": "v7.9",
    },
    permit_action={
        "type": "ENTRY",
        "symbol": "ETHUSDT",
        "direction": "BUY",
        "qty": 100.0,
    },
    permit_ttl_s=30.0,
)

# Deny enrichment — no permit should be emitted
DENY_ENRICHMENT = dict(
    verdict="DENY",
    deny_reason="VETO_DIRECTION_MISMATCH",
    doctrine_verdict="VETO_DIRECTION_MISMATCH",
    context_snapshot={"regime": "TREND_UP", "positions_hash": "pos_abc"},
    provenance={"engine_version": "v7.9", "git_sha": "abc123", "docs_version": "v7.9"},
    permit_action={"type": "ENTRY", "symbol": "ETHUSDT", "direction": "BUY"},
    permit_ttl_s=30.0,
)

# Exit enrichment
EXIT_ENRICHMENT = dict(
    verdict="PERMIT",
    doctrine_verdict="EXIT_ALLOW",
    context_snapshot={"regime": "CHOPPY", "positions_hash": "pos_xyz"},
    provenance={"engine_version": "v7.9", "git_sha": "abc123", "docs_version": "v7.9"},
    permit_action={
        "type": "EXIT",
        "symbol": "ETHUSDT",
        "direction": "SELL",
    },
    permit_ttl_s=60.0,
)


# ─── PERMIT v2 on ALLOW paths ──────────────────────────────────────────


class TestPermitV2OnAllow:
    """B.3: Enriched PERMIT events use v2 schema on ALLOW paths."""

    def test_permit_v2_schema(self):
        w = _CapturingWriter()
        shadow_build_chain(**_base_kwargs(writer=w, **PERMIT_ENRICHMENT))
        permits = [e for e in w.events if e["event_type"] == "PERMIT"]
        assert len(permits) == 1
        assert permits[0]["schema_version"] == SCHEMA_VERSION_V2

    def test_permit_has_action(self):
        w = _CapturingWriter()
        shadow_build_chain(**_base_kwargs(writer=w, **PERMIT_ENRICHMENT))
        payload = [e for e in w.events if e["event_type"] == "PERMIT"][0]["payload"]
        assert payload["action"]["type"] == "ENTRY"
        assert payload["action"]["symbol"] == "ETHUSDT"
        assert payload["action"]["direction"] == "BUY"
        assert payload["action"]["qty"] == 100.0

    def test_permit_has_ttl(self):
        w = _CapturingWriter()
        shadow_build_chain(**_base_kwargs(writer=w, **PERMIT_ENRICHMENT))
        payload = [e for e in w.events if e["event_type"] == "PERMIT"][0]["payload"]
        assert payload["permit_ttl_s"] == 30.0
        assert "expires_ts" in payload
        # expires_ts must be parseable ISO and ~30s after ts
        issued = datetime.fromisoformat(payload["ts"])
        expires = datetime.fromisoformat(payload["expires_ts"])
        delta = (expires - issued).total_seconds()
        assert 29.0 <= delta <= 31.0

    def test_permit_has_state_and_consumable(self):
        w = _CapturingWriter()
        shadow_build_chain(**_base_kwargs(writer=w, **PERMIT_ENRICHMENT))
        payload = [e for e in w.events if e["event_type"] == "PERMIT"][0]["payload"]
        assert payload["state"] == "ISSUED"
        assert payload["consumable"] is True

    def test_permit_has_scope_snapshot(self):
        w = _CapturingWriter()
        shadow_build_chain(**_base_kwargs(writer=w, **PERMIT_ENRICHMENT))
        payload = [e for e in w.events if e["event_type"] == "PERMIT"][0]["payload"]
        assert payload["scope_snapshot"]["regime"] == "TREND_UP"
        assert payload["scope_snapshot"]["regime_confidence"] == 0.85
        assert payload["scope_snapshot"]["nav_usd"] == 1234.56

    def test_permit_has_provenance(self):
        w = _CapturingWriter()
        shadow_build_chain(**_base_kwargs(writer=w, **PERMIT_ENRICHMENT))
        payload = [e for e in w.events if e["event_type"] == "PERMIT"][0]["payload"]
        assert payload["provenance"]["engine_version"] == "v7.9"
        assert payload["provenance"]["git_sha"] == "abc123"
        assert payload["provenance"]["docs_version"] == "v7.9"

    def test_permit_preserves_existing_fields(self):
        """V1 fields (permit_id, decision_id, request_id, single_use, snapshots) still present."""
        w = _CapturingWriter()
        shadow_build_chain(**_base_kwargs(writer=w, **PERMIT_ENRICHMENT))
        payload = [e for e in w.events if e["event_type"] == "PERMIT"][0]["payload"]
        assert payload["permit_id"].startswith("PERM_")
        assert payload["decision_id"].startswith("DEC_")
        assert payload["request_id"] == "sig_test_b3"
        assert payload["single_use"] is True
        assert "snapshots" in payload


# ─── No PERMIT on DENY paths ───────────────────────────────────────────


class TestNoPermitOnDeny:
    """B.3: Denied outcomes suppress PERMIT event."""

    def test_deny_emits_no_permit(self):
        w = _CapturingWriter()
        shadow_build_chain(**_base_kwargs(
            writer=w,
            action_class="ENTRY_DENY",
            constraints={"verdict": "DENY", "reason": "VETO_DIRECTION_MISMATCH"},
            **DENY_ENRICHMENT,
        ))
        types = [e["event_type"] for e in w.events]
        assert "PERMIT" not in types

    def test_deny_emits_three_events(self):
        """DENY path: REQUEST → DECISION → LINK (no PERMIT)."""
        w = _CapturingWriter()
        shadow_build_chain(**_base_kwargs(
            writer=w,
            action_class="ENTRY_DENY",
            constraints={"verdict": "DENY", "reason": "VETO_CRISIS"},
            **DENY_ENRICHMENT,
        ))
        types = [e["event_type"] for e in w.events]
        assert types == ["REQUEST", "DECISION", "LINK"]

    def test_deny_link_has_null_permit_id(self):
        """LINK event on DENY has permit_id=None."""
        w = _CapturingWriter()
        shadow_build_chain(**_base_kwargs(
            writer=w,
            action_class="ENTRY_DENY",
            constraints={"verdict": "DENY", "reason": "VETO_CRISIS"},
            **DENY_ENRICHMENT,
        ))
        link = [e for e in w.events if e["event_type"] == "LINK"][0]
        assert link["payload"]["permit_id"] is None


# ─── ALLOW paths still emit 4 events ───────────────────────────────────


class TestAllowEventChain:
    """ALLOW paths still produce the full 4-event chain."""

    def test_allow_emits_four_events(self):
        w = _CapturingWriter()
        shadow_build_chain(**_base_kwargs(writer=w, **PERMIT_ENRICHMENT))
        types = [e["event_type"] for e in w.events]
        assert types == ["REQUEST", "DECISION", "PERMIT", "LINK"]

    def test_allow_link_has_permit_id(self):
        w = _CapturingWriter()
        shadow_build_chain(**_base_kwargs(writer=w, **PERMIT_ENRICHMENT))
        link = [e for e in w.events if e["event_type"] == "LINK"][0]
        assert link["payload"]["permit_id"] is not None
        assert link["payload"]["permit_id"].startswith("PERM_")


# ─── Exit permits ──────────────────────────────────────────────────────


class TestExitPermit:
    """Exit permits use EXIT TTL and correct action type."""

    def test_exit_permit_ttl(self):
        w = _CapturingWriter()
        shadow_build_chain(**_base_kwargs(
            writer=w,
            requested_action="EXIT",
            action_class="EXIT_ALLOW",
            **EXIT_ENRICHMENT,
        ))
        payload = [e for e in w.events if e["event_type"] == "PERMIT"][0]["payload"]
        assert payload["permit_ttl_s"] == 60.0

    def test_exit_permit_action_type(self):
        w = _CapturingWriter()
        shadow_build_chain(**_base_kwargs(
            writer=w,
            requested_action="EXIT",
            action_class="EXIT_ALLOW",
            **EXIT_ENRICHMENT,
        ))
        payload = [e for e in w.events if e["event_type"] == "PERMIT"][0]["payload"]
        assert payload["action"]["type"] == "EXIT"
        assert payload["action"]["direction"] == "SELL"


# ─── Backward compat (no permit enrichment → v1) ───────────────────────


class TestPermitBackwardCompat:
    """Without permit enrichment kwargs, PERMIT events use v1."""

    def test_no_permit_enrichment_uses_v1(self):
        """When only B.2 decision enrichment is provided (no permit_action/permit_ttl_s),
        PERMIT stays v1."""
        w = _CapturingWriter()
        shadow_build_chain(**_base_kwargs(
            writer=w,
            verdict="PERMIT",
            context_snapshot={"regime": "TREND_UP"},
            provenance={"engine_version": "v7.9"},
            # No permit_action or permit_ttl_s
        ))
        permits = [e for e in w.events if e["event_type"] == "PERMIT"]
        assert len(permits) == 1
        assert permits[0]["schema_version"] == SCHEMA_VERSION

    def test_no_enrichment_at_all_uses_v1(self):
        """Plain call (no B.2 or B.3 enrichment) → PERMIT v1."""
        w = _CapturingWriter()
        shadow_build_chain(**_base_kwargs(writer=w))
        permits = [e for e in w.events if e["event_type"] == "PERMIT"]
        assert len(permits) == 1
        assert permits[0]["schema_version"] == SCHEMA_VERSION

    def test_no_enrichment_permit_has_no_extra_fields(self):
        w = _CapturingWriter()
        shadow_build_chain(**_base_kwargs(writer=w))
        payload = [e for e in w.events if e["event_type"] == "PERMIT"][0]["payload"]
        for field in ("action", "permit_ttl_s", "expires_ts", "state", "consumable", "scope_snapshot", "provenance"):
            assert field not in payload


# ─── Deterministic IDs ──────────────────────────────────────────────────


class TestPermitIdDeterminism:
    """Permit ID is derived from (decision_id, request_id, ts)."""

    def test_permit_id_links_to_decision(self):
        w = _CapturingWriter()
        shadow_build_chain(**_base_kwargs(writer=w, **PERMIT_ENRICHMENT))
        dec = [e for e in w.events if e["event_type"] == "DECISION"][0]["payload"]
        perm = [e for e in w.events if e["event_type"] == "PERMIT"][0]["payload"]
        assert perm["decision_id"] == dec["decision_id"]

    def test_permit_id_links_to_request(self):
        w = _CapturingWriter()
        shadow_build_chain(**_base_kwargs(writer=w, **PERMIT_ENRICHMENT))
        req = [e for e in w.events if e["event_type"] == "REQUEST"][0]["payload"]
        perm = [e for e in w.events if e["event_type"] == "PERMIT"][0]["payload"]
        assert perm["request_id"] == req["request_id"]


# ─── TTL computation ───────────────────────────────────────────────────


class TestTTLComputation:
    """_compute_expires_ts() smoke tests."""

    def test_30s_ttl(self):
        issued = "2026-02-13T12:00:00+00:00"
        expires = _compute_expires_ts(issued, 30.0)
        dt = datetime.fromisoformat(expires)
        expected = datetime(2026, 2, 13, 12, 0, 30, tzinfo=timezone.utc)
        assert abs((dt - expected).total_seconds()) < 1.0

    def test_60s_ttl(self):
        issued = "2026-02-13T12:00:00+00:00"
        expires = _compute_expires_ts(issued, 60.0)
        dt = datetime.fromisoformat(expires)
        expected = datetime(2026, 2, 13, 12, 1, 0, tzinfo=timezone.utc)
        assert abs((dt - expected).total_seconds()) < 1.0

    def test_default_ttl_constants(self):
        assert DEFAULT_ENTRY_PERMIT_TTL_S == 30.0
        assert DEFAULT_EXIT_PERMIT_TTL_S == 60.0


# ─── Fail-open ──────────────────────────────────────────────────────────


class TestPermitFailOpen:
    """Enrichment errors must not crash execution."""

    def test_bad_writer_with_permit_enrichment(self):
        class _BadWriter:
            def write(self, event):
                raise IOError("disk full")

        # Must not raise
        rid, did, pid = shadow_build_chain(**_base_kwargs(
            writer=_BadWriter(),
            **PERMIT_ENRICHMENT,
        ))
        assert rid is not None
        assert did.startswith("DEC_")
        assert pid.startswith("PERM_")
