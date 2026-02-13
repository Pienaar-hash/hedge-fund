"""
Phase B.2 tests: DLE shadow decision enrichment.

Covers:
- DECISION event v2 enrichment (verdict, deny_reason, context_snapshot, provenance)
- Backward compat (no enrichment → v1 schema)
- Path invariant (verify_shadow_log_path)
- Append-only format
- Field completeness on DENY vs PERMIT
"""

import json
import os
import tempfile
from unittest.mock import patch

import pytest
from execution.dle_shadow import (
    SCHEMA_VERSION,
    SCHEMA_VERSION_V2,
    DEFAULT_LOG_PATH,
    MANIFEST_LOG_PATH,
    DLEShadowWriter,
    DLEShadowEvent,
    shadow_build_chain,
    verify_shadow_log_path,
    _stable_json,
)


# ─── Helpers ────────────────────────────────────────────────────────────


def _base_kwargs(**overrides):
    """Minimal valid kwargs for shadow_build_chain()."""
    base = dict(
        enabled=True,
        write_logs=True,
        writer=None,
        attempt_id="sig_test_b2",
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
        self.events.append(json.loads(_stable_json({"schema_version": event.schema_version,
                                                     "event_type": event.event_type,
                                                     "ts": event.ts,
                                                     "payload": event.payload})))


# ─── Path invariant ─────────────────────────────────────────────────────


class TestPathInvariant:
    """B.2c: verify_shadow_log_path()."""

    def test_path_matches_manifest(self):
        """DEFAULT_LOG_PATH == MANIFEST_LOG_PATH (should pass after B.2a fix)."""
        assert DEFAULT_LOG_PATH == MANIFEST_LOG_PATH
        verify_shadow_log_path()  # must not raise

    def test_path_mismatch_raises(self):
        """If paths diverge, ValueError raised."""
        with patch("execution.dle_shadow.DEFAULT_LOG_PATH", "logs/dle/wrong.jsonl"):
            with pytest.raises(ValueError, match="mismatch"):
                verify_shadow_log_path()


# ─── Backward compat (no enrichment → v1 schema) ───────────────────────


class TestBackwardCompat:
    """Without enrichment kwargs, DECISION events use dle_shadow_v1."""

    def test_no_enrichment_uses_v1(self):
        w = _CapturingWriter()
        shadow_build_chain(**_base_kwargs(writer=w))
        decision_events = [e for e in w.events if e["event_type"] == "DECISION"]
        assert len(decision_events) == 1
        assert decision_events[0]["schema_version"] == SCHEMA_VERSION

    def test_no_enrichment_no_extra_fields(self):
        w = _CapturingWriter()
        shadow_build_chain(**_base_kwargs(writer=w))
        dec = [e for e in w.events if e["event_type"] == "DECISION"][0]
        payload = dec["payload"]
        # v1 payload should NOT contain enrichment fields
        for field in ("verdict", "deny_reason", "doctrine_verdict", "context_snapshot", "provenance"):
            assert field not in payload


# ─── Enriched DECISION events (v2) ─────────────────────────────────────


class TestEnrichedDecision:
    """B.2b: DECISION events with enrichment use v2 schema."""

    ENRICHMENT_PERMIT = dict(
        verdict="PERMIT",
        deny_reason=None,
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
            "git_sha": "38dfbcfe",
            "docs_version": "v7.9",
        },
    )

    ENRICHMENT_DENY = dict(
        verdict="DENY",
        deny_reason="VETO_DIRECTION_MISMATCH",
        doctrine_verdict="VETO_DIRECTION_MISMATCH",
        context_snapshot={
            "regime": "TREND_UP",
            "regime_confidence": 0.72,
            "nav_usd": 1234.56,
            "positions_hash": "pos_abc123",
            "scores_hash": "scores_def456",
        },
        provenance={
            "engine_version": "v7.9",
            "git_sha": "38dfbcfe",
            "docs_version": "v7.9",
        },
    )

    def test_enriched_permit_uses_v2(self):
        w = _CapturingWriter()
        shadow_build_chain(**_base_kwargs(writer=w, **self.ENRICHMENT_PERMIT))
        dec = [e for e in w.events if e["event_type"] == "DECISION"][0]
        assert dec["schema_version"] == SCHEMA_VERSION_V2

    def test_enriched_deny_uses_v2(self):
        w = _CapturingWriter()
        shadow_build_chain(**_base_kwargs(
            writer=w,
            action_class="ENTRY_DENY",
            constraints={"verdict": "DENY", "reason": "VETO_DIRECTION_MISMATCH"},
            **self.ENRICHMENT_DENY,
        ))
        dec = [e for e in w.events if e["event_type"] == "DECISION"][0]
        assert dec["schema_version"] == SCHEMA_VERSION_V2

    def test_permit_payload_has_all_fields(self):
        w = _CapturingWriter()
        shadow_build_chain(**_base_kwargs(writer=w, **self.ENRICHMENT_PERMIT))
        payload = [e for e in w.events if e["event_type"] == "DECISION"][0]["payload"]
        # All enrichment fields present
        assert payload["verdict"] == "PERMIT"
        assert payload["doctrine_verdict"] == "DOCTRINE_ALLOW"
        assert payload["context_snapshot"]["regime"] == "TREND_UP"
        assert payload["context_snapshot"]["regime_confidence"] == 0.85
        assert payload["context_snapshot"]["nav_usd"] == 1234.56
        assert payload["provenance"]["engine_version"] == "v7.9"
        assert payload["provenance"]["git_sha"] == "38dfbcfe"
        assert payload["provenance"]["docs_version"] == "v7.9"
        # deny_reason should NOT be in payload (it was None → not written)
        assert "deny_reason" not in payload

    def test_deny_payload_has_deny_reason(self):
        w = _CapturingWriter()
        shadow_build_chain(**_base_kwargs(
            writer=w,
            action_class="ENTRY_DENY",
            constraints={"verdict": "DENY", "reason": "VETO_DIRECTION_MISMATCH"},
            **self.ENRICHMENT_DENY,
        ))
        payload = [e for e in w.events if e["event_type"] == "DECISION"][0]["payload"]
        assert payload["verdict"] == "DENY"
        assert payload["deny_reason"] == "VETO_DIRECTION_MISMATCH"
        assert payload["doctrine_verdict"] == "VETO_DIRECTION_MISMATCH"

    def test_non_decision_events_stay_v1(self):
        """REQUEST, PERMIT, LINK events remain v1 even when DECISION is enriched."""
        w = _CapturingWriter()
        shadow_build_chain(**_base_kwargs(writer=w, **self.ENRICHMENT_PERMIT))
        for e in w.events:
            if e["event_type"] != "DECISION":
                assert e["schema_version"] == SCHEMA_VERSION, \
                    f"{e['event_type']} should use v1, got {e['schema_version']}"

    def test_four_events_emitted(self):
        """Chain always emits exactly 4 events: REQUEST → DECISION → PERMIT → LINK."""
        w = _CapturingWriter()
        shadow_build_chain(**_base_kwargs(writer=w, **self.ENRICHMENT_PERMIT))
        types = [e["event_type"] for e in w.events]
        assert types == ["REQUEST", "DECISION", "PERMIT", "LINK"]


# ─── Deterministic IDs with enrichment ──────────────────────────────────


class TestDeterministicWithEnrichment:
    """Enrichment does not affect ID determinism."""

    def test_same_inputs_same_decision_id(self):
        """Decision ID depends only on (phase_id, action_class, constraints, policy_version)."""
        w1 = _CapturingWriter()
        w2 = _CapturingWriter()
        enrichment = dict(
            verdict="PERMIT",
            context_snapshot={"regime": "TREND_UP"},
            provenance={"engine_version": "v7.9"},
        )
        shadow_build_chain(**_base_kwargs(writer=w1, **enrichment))
        shadow_build_chain(**_base_kwargs(writer=w2, **enrichment))
        did1 = [e for e in w1.events if e["event_type"] == "DECISION"][0]["payload"]["decision_id"]
        did2 = [e for e in w2.events if e["event_type"] == "DECISION"][0]["payload"]["decision_id"]
        assert did1 == did2

    def test_enrichment_does_not_change_decision_id(self):
        """Adding enrichment doesn't alter decision_id (only payload is extended)."""
        w_plain = _CapturingWriter()
        w_enriched = _CapturingWriter()
        shadow_build_chain(**_base_kwargs(writer=w_plain))
        shadow_build_chain(**_base_kwargs(
            writer=w_enriched,
            verdict="PERMIT",
            context_snapshot={"regime": "TREND_UP"},
        ))
        did_plain = [e for e in w_plain.events if e["event_type"] == "DECISION"][0]["payload"]["decision_id"]
        did_enriched = [e for e in w_enriched.events if e["event_type"] == "DECISION"][0]["payload"]["decision_id"]
        assert did_plain == did_enriched


# ─── Append-only format ─────────────────────────────────────────────────


class TestAppendOnly:
    """Shadow writer appends, never overwrites."""

    def test_multiple_writes_append(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "shadow.jsonl")
            writer = DLEShadowWriter(log_path=path)
            # Write two chains
            shadow_build_chain(**_base_kwargs(writer=writer))
            shadow_build_chain(**_base_kwargs(writer=writer, attempt_id="sig_second"))
            lines = open(path).readlines()
            # 4 events per chain × 2 chains = 8 lines
            assert len(lines) == 8
            # Each line is valid JSON
            for line in lines:
                parsed = json.loads(line)
                assert "schema_version" in parsed
                assert "event_type" in parsed

    def test_writer_failure_count(self):
        writer = DLEShadowWriter(log_path="/dev/null/impossible/path.jsonl")
        writer.write(DLEShadowEvent(
            schema_version=SCHEMA_VERSION,
            event_type="TEST",
            ts="2026-01-01T00:00:00+00:00",
            payload={},
        ))
        assert writer.write_failure_count >= 1


# ─── Context snapshot field coverage ────────────────────────────────────


class TestContextSnapshotFields:
    """Validate context_snapshot structure matches manifest."""

    REQUIRED_SNAPSHOT_KEYS = {"regime", "positions_hash"}

    def test_minimal_snapshot_accepted(self):
        """Minimum viable context_snapshot has regime + positions_hash."""
        w = _CapturingWriter()
        snapshot = {"regime": "CHOPPY", "positions_hash": "pos_xyz"}
        shadow_build_chain(**_base_kwargs(
            writer=w,
            verdict="PERMIT",
            context_snapshot=snapshot,
        ))
        dec = [e for e in w.events if e["event_type"] == "DECISION"][0]
        assert dec["payload"]["context_snapshot"]["regime"] == "CHOPPY"
        assert dec["payload"]["context_snapshot"]["positions_hash"] == "pos_xyz"

    def test_full_snapshot_fields(self):
        """Full context_snapshot includes all optional fields."""
        w = _CapturingWriter()
        snapshot = {
            "regime": "TREND_UP",
            "regime_confidence": 0.92,
            "nav_usd": 5000.0,
            "positions_hash": "pos_123",
            "scores_hash": "scores_456",
        }
        shadow_build_chain(**_base_kwargs(
            writer=w,
            verdict="PERMIT",
            context_snapshot=snapshot,
        ))
        dec = [e for e in w.events if e["event_type"] == "DECISION"][0]
        cs = dec["payload"]["context_snapshot"]
        assert cs["regime"] == "TREND_UP"
        assert cs["regime_confidence"] == 0.92
        assert cs["nav_usd"] == 5000.0
        assert cs["positions_hash"] == "pos_123"
        assert cs["scores_hash"] == "scores_456"


# ─── Provenance field coverage ──────────────────────────────────────────


class TestProvenanceFields:
    """Validate provenance structure matches manifest."""

    def test_provenance_fields_present(self):
        w = _CapturingWriter()
        prov = {
            "engine_version": "v7.9",
            "git_sha": "abc123",
            "docs_version": "v7.9",
        }
        shadow_build_chain(**_base_kwargs(
            writer=w,
            verdict="PERMIT",
            provenance=prov,
        ))
        dec = [e for e in w.events if e["event_type"] == "DECISION"][0]
        assert dec["payload"]["provenance"]["engine_version"] == "v7.9"
        assert dec["payload"]["provenance"]["git_sha"] == "abc123"
        assert dec["payload"]["provenance"]["docs_version"] == "v7.9"


# ─── Fail-open with enrichment ──────────────────────────────────────────


class TestFailOpenEnriched:
    """Enrichment errors must not crash execution."""

    def test_bad_writer_with_enrichment_no_raise(self):
        class _BadWriter:
            def write(self, event):
                raise IOError("disk full")

        # Must not raise
        rid, did, pid = shadow_build_chain(**_base_kwargs(
            writer=_BadWriter(),
            verdict="DENY",
            deny_reason="VETO_CRISIS",
            doctrine_verdict="VETO_CRISIS",
            context_snapshot={"regime": "CRISIS"},
            provenance={"engine_version": "v7.9"},
        ))
        assert rid is not None
        assert did.startswith("DEC_")
        assert pid.startswith("PERM_")
