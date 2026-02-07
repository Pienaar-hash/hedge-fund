# tests/unit/test_prediction_rollback_triggers.py
"""
Tests for prediction/rollback_triggers.py — rollback clause enforcement hooks.

Covers:
    - Temporal restatement detection
    - Replay divergence detection
    - Latency breach detection
    - Silent gap detection
    - Regime corruption flagging
    - Batch health checks
    - JSONL logging (append-only)
    - P0 enforcement=False invariant
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from prediction.rollback_triggers import (
    RollbackTriggerEvent,
    SuggestedAction,
    TriggerType,
    check_latency_breach,
    check_regime_corruption,
    check_replay_divergence,
    check_silent_gap,
    check_temporal_restatement,
    run_health_checks,
)


# ---------------------------------------------------------------------------
# Temporal restatement
# ---------------------------------------------------------------------------

class TestTemporalRestatement:
    def test_restatement_detected(self, tmp_path):
        log = tmp_path / "rollback.jsonl"
        event = check_temporal_restatement(
            dataset_id="poly",
            original_ts="2026-01-01T12:00:00Z",
            restated_ts="2026-01-01T12:00:00Z",
            original_value=0.65,
            restated_value=0.80,
            log_path=log,
        )
        assert event is not None
        assert event.trigger_type == TriggerType.TEMPORAL_RESTATEMENT.value
        assert event.suggested_action == SuggestedAction.REVOKE.value
        assert event.severity == "HIGH"
        assert event.enforced is False  # P0
        assert log.exists()

    def test_no_restatement_same_value(self, tmp_path):
        log = tmp_path / "rollback.jsonl"
        event = check_temporal_restatement(
            dataset_id="poly",
            original_ts="2026-01-01T12:00:00Z",
            restated_ts="2026-01-01T12:00:00Z",
            original_value=0.65,
            restated_value=0.65,
            log_path=log,
        )
        assert event is None
        assert not log.exists()

    def test_different_timestamps_not_restatement(self, tmp_path):
        log = tmp_path / "rollback.jsonl"
        event = check_temporal_restatement(
            dataset_id="poly",
            original_ts="2026-01-01T12:00:00Z",
            restated_ts="2026-01-01T13:00:00Z",
            original_value=0.65,
            restated_value=0.80,
            log_path=log,
        )
        assert event is None


# ---------------------------------------------------------------------------
# Replay divergence
# ---------------------------------------------------------------------------

class TestReplayDivergence:
    def test_divergence_detected(self, tmp_path):
        log = tmp_path / "rollback.jsonl"
        event = check_replay_divergence(
            dataset_id="model_x",
            expected_hash="sha256:aaaa1111",
            actual_hash="sha256:bbbb2222",
            log_path=log,
        )
        assert event is not None
        assert event.trigger_type == TriggerType.REPLAY_DIVERGENCE.value
        assert event.suggested_action == SuggestedAction.REVOKE.value

    def test_no_divergence(self, tmp_path):
        log = tmp_path / "rollback.jsonl"
        event = check_replay_divergence(
            dataset_id="model_x",
            expected_hash="sha256:aaaa1111",
            actual_hash="sha256:aaaa1111",
            log_path=log,
        )
        assert event is None


# ---------------------------------------------------------------------------
# Latency breach
# ---------------------------------------------------------------------------

class TestLatencyBreach:
    def test_breach_detected(self, tmp_path):
        log = tmp_path / "rollback.jsonl"
        event = check_latency_breach(
            dataset_id="poly",
            latency_ms=8000,
            threshold_ms=5000,
            log_path=log,
        )
        assert event is not None
        assert event.trigger_type == TriggerType.LATENCY_BREACH.value
        assert event.suggested_action == SuggestedAction.DOWNGRADE.value
        assert event.severity == "MEDIUM"

    def test_within_threshold(self, tmp_path):
        log = tmp_path / "rollback.jsonl"
        event = check_latency_breach(
            dataset_id="poly",
            latency_ms=3000,
            threshold_ms=5000,
            log_path=log,
        )
        assert event is None

    def test_exactly_at_threshold(self, tmp_path):
        log = tmp_path / "rollback.jsonl"
        event = check_latency_breach(
            dataset_id="poly",
            latency_ms=5000,
            threshold_ms=5000,
            log_path=log,
        )
        assert event is None  # <= threshold = OK


# ---------------------------------------------------------------------------
# Silent gap
# ---------------------------------------------------------------------------

class TestSilentGap:
    def test_gap_detected(self, tmp_path):
        log = tmp_path / "rollback.jsonl"
        old_ts = time.time() - 7200  # 2 hours ago
        event = check_silent_gap(
            dataset_id="poly",
            last_event_ts=old_ts,
            gap_threshold_seconds=3600,
            log_path=log,
        )
        assert event is not None
        assert event.trigger_type == TriggerType.SILENT_GAP.value
        assert event.suggested_action == SuggestedAction.DOWNGRADE.value

    def test_no_gap(self, tmp_path):
        log = tmp_path / "rollback.jsonl"
        recent_ts = time.time() - 60  # 1 minute ago
        event = check_silent_gap(
            dataset_id="poly",
            last_event_ts=recent_ts,
            gap_threshold_seconds=3600,
            log_path=log,
        )
        assert event is None

    def test_custom_now(self, tmp_path):
        log = tmp_path / "rollback.jsonl"
        event = check_silent_gap(
            dataset_id="poly",
            last_event_ts=1000.0,
            now_ts=5000.0,
            gap_threshold_seconds=3600,
            log_path=log,
        )
        assert event is not None
        assert event.evidence["gap_seconds"] == 4000.0


# ---------------------------------------------------------------------------
# Regime corruption
# ---------------------------------------------------------------------------

class TestRegimeCorruption:
    def test_always_logged(self, tmp_path):
        log = tmp_path / "rollback.jsonl"
        event = check_regime_corruption(
            dataset_id="bad_feed",
            evidence={"regime_before": "TREND_UP", "regime_after": "CRISIS"},
            log_path=log,
        )
        assert event.trigger_type == TriggerType.REGIME_CORRUPTION.value
        assert event.suggested_action == SuggestedAction.REVOKE.value
        assert event.severity == "HIGH"
        assert log.exists()


# ---------------------------------------------------------------------------
# Batch health checks
# ---------------------------------------------------------------------------

class TestRunHealthChecks:
    def test_healthy(self, tmp_path):
        log = tmp_path / "rollback.jsonl"
        results = run_health_checks(
            dataset_id="poly",
            last_event_ts=time.time() - 60,
            last_latency_ms=200,
            log_path=log,
        )
        assert len(results) == 0

    def test_multiple_triggers(self, tmp_path):
        log = tmp_path / "rollback.jsonl"
        results = run_health_checks(
            dataset_id="poly",
            last_event_ts=time.time() - 7200,
            last_latency_ms=10000,
            log_path=log,
        )
        assert len(results) == 2  # gap + latency
        types = {r.trigger_type for r in results}
        assert TriggerType.SILENT_GAP.value in types
        assert TriggerType.LATENCY_BREACH.value in types


# ---------------------------------------------------------------------------
# P0 invariant: enforced is always False
# ---------------------------------------------------------------------------

class TestP0Invariant:
    def test_all_triggers_not_enforced(self, tmp_path):
        log = tmp_path / "rollback.jsonl"
        events = [
            check_temporal_restatement(
                dataset_id="d", original_ts="t", restated_ts="t",
                original_value=0.5, restated_value=0.9, log_path=log,
            ),
            check_replay_divergence(
                dataset_id="d", expected_hash="a", actual_hash="b", log_path=log,
            ),
            check_latency_breach(
                dataset_id="d", latency_ms=99999, log_path=log,
            ),
            check_silent_gap(
                dataset_id="d", last_event_ts=0, now_ts=999999, log_path=log,
            ),
            check_regime_corruption(
                dataset_id="d", evidence={"test": True}, log_path=log,
            ),
        ]
        for ev in events:
            assert ev is not None
            assert ev.enforced is False


# ---------------------------------------------------------------------------
# JSONL logging
# ---------------------------------------------------------------------------

class TestLogging:
    def test_append_only(self, tmp_path):
        log = tmp_path / "rollback.jsonl"
        check_latency_breach(dataset_id="d1", latency_ms=9000, log_path=log)
        check_latency_breach(dataset_id="d2", latency_ms=8000, log_path=log)
        lines = log.read_text().strip().split("\n")
        assert len(lines) == 2
        r1 = json.loads(lines[0])
        r2 = json.loads(lines[1])
        assert r1["dataset_id"] == "d1"
        assert r2["dataset_id"] == "d2"
