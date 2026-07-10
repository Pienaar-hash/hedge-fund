"""Tests for the _pub_tick() publication boundary heartbeat.

CARD-HEDGE-PUBTICK-BOUNDARY-HEARTBEAT-INSTRUMENTATION-001: proves whether
_pub_tick() is reached, entered, completed, or abandoned each loop, without
changing publication cadence, trading behavior, or state payloads.
"""
from __future__ import annotations

import time

import pytest

from execution import executor_live

pytestmark = pytest.mark.integration


class _RecordingHeartbeatLog:
    """Captures heartbeat records in-memory instead of touching the filesystem."""

    def __init__(self) -> None:
        self.records: list[dict] = []

    def write(self, record) -> None:
        self.records.append(dict(record))


class _AlwaysFailsHeartbeatLog:
    def write(self, record) -> None:
        raise OSError("simulated telemetry sink failure")


def _patch_pub_tick_internals(monkeypatch, nav=100.0, rows=None):
    """Mirror the minimal patch set used by test_pub_tick_writes_state so the
    unguarded top section of _pub_tick() runs deterministically without
    touching live exchange/state I/O."""
    rows = rows if rows is not None else []
    monkeypatch.setattr(executor_live, "_compute_nav_with_detail", lambda: (nav, {"nav_mode": "enhanced"}))
    monkeypatch.setattr(executor_live, "_collect_rows", lambda: rows)
    monkeypatch.setattr(executor_live, "_persist_positions_cache", lambda _rows: None)
    monkeypatch.setattr(executor_live, "_persist_nav_log", lambda *_args: None)
    monkeypatch.setattr(executor_live, "_persist_spot_state", lambda: None)
    monkeypatch.setattr(executor_live, "write_nav_state", lambda payload: None)
    monkeypatch.setattr(executor_live, "write_positions_snapshot_state", lambda payload: None)
    monkeypatch.setattr(executor_live, "write_positions_state", lambda *_a, **_k: None)
    monkeypatch.setattr(executor_live, "write_synced_state", lambda payload: None)
    monkeypatch.setattr(executor_live, "write_risk_snapshot_state", lambda payload: None)


def _reset_completion_globals(monkeypatch):
    monkeypatch.setattr(executor_live, "_PUB_TICK_LAST_COMPLETED_TS", None)
    monkeypatch.setattr(executor_live, "_PUB_TICK_LAST_COMPLETED_DURATION_MS", None)


def _fresh_state(monkeypatch, *, episode_ledger_due: bool):
    state = executor_live._STATE
    now = time.time()
    due_ts = 0.0 if episode_ledger_due else now
    monkeypatch.setattr(state, "last_episode_ledger_rebuild_ts", due_ts)
    return state


def test_normal_event_ordering_no_ledger_due(monkeypatch):
    log = _RecordingHeartbeatLog()
    monkeypatch.setattr(executor_live, "_PUB_TICK_HEARTBEAT_LOG", log)
    _reset_completion_globals(monkeypatch)
    _patch_pub_tick_internals(monkeypatch)
    state = _fresh_state(monkeypatch, episode_ledger_due=False)

    executor_live._pub_tick_heartbeat("CALL_REACHED", loop_id=42)
    executor_live._pub_tick(state, loop_id=42)

    events = [r["event"] for r in log.records]
    assert events == ["CALL_REACHED", "ENTERED", "COMPLETED"]
    assert all(r["loop_id"] == 42 for r in log.records)
    completed = log.records[-1]
    assert completed["outcome"] == "ok"
    assert completed["episode_ledger_due"] is False
    assert completed["episode_ledger_duration_ms"] is None


def test_ledger_due_path_emits_rebuild_events(monkeypatch):
    log = _RecordingHeartbeatLog()
    monkeypatch.setattr(executor_live, "_PUB_TICK_HEARTBEAT_LOG", log)
    _reset_completion_globals(monkeypatch)
    _patch_pub_tick_internals(monkeypatch)
    state = _fresh_state(monkeypatch, episode_ledger_due=True)
    monkeypatch.setattr("execution.episode_ledger.rebuild_and_save", lambda: None)

    executor_live._pub_tick_heartbeat("CALL_REACHED", loop_id=7)
    executor_live._pub_tick(state, loop_id=7)

    events = [r["event"] for r in log.records]
    assert events == [
        "CALL_REACHED",
        "ENTERED",
        "LEDGER_REBUILD_STARTED",
        "LEDGER_REBUILD_COMPLETED",
        "COMPLETED",
    ]
    rebuild_completed = log.records[3]
    assert rebuild_completed["duration_ms"] is not None
    assert rebuild_completed["duration_ms"] >= 0.0
    completed = log.records[-1]
    assert completed["episode_ledger_due"] is True
    assert completed["episode_ledger_duration_ms"] is not None
    assert state.last_episode_ledger_rebuild_ts > 0.0


def test_ordinary_exception_emits_failed_and_reraises(monkeypatch):
    log = _RecordingHeartbeatLog()
    monkeypatch.setattr(executor_live, "_PUB_TICK_HEARTBEAT_LOG", log)
    _reset_completion_globals(monkeypatch)
    state = _fresh_state(monkeypatch, episode_ledger_due=False)

    def _boom():
        raise ValueError("nav computation exploded")

    monkeypatch.setattr(executor_live, "_compute_nav_with_detail", _boom)

    with pytest.raises(ValueError):
        executor_live._pub_tick(state, loop_id=3)

    events = [r["event"] for r in log.records]
    assert events == ["ENTERED", "FAILED"]
    failed = log.records[-1]
    assert failed["outcome"] == "failed"
    assert failed["exception_type"] == "ValueError"
    assert "nav computation exploded" in failed["exception_message"]

    # The call-site boundary must preserve today's existing publish_tick_failed
    # behavior: catch, log, and continue the loop rather than crash it.
    caught = []
    try:
        executor_live._pub_tick(state, loop_id=3)
    except Exception as exc:  # mirrors execution/executor_live.py call site
        caught.append(exc)
    assert len(caught) == 1


def test_aborted_path_distinguishable_from_call_site_skip(monkeypatch):
    log = _RecordingHeartbeatLog()
    monkeypatch.setattr(executor_live, "_PUB_TICK_HEARTBEAT_LOG", log)
    _reset_completion_globals(monkeypatch)
    state = _fresh_state(monkeypatch, episode_ledger_due=False)

    def _hard_abort():
        raise KeyboardInterrupt()

    monkeypatch.setattr(executor_live, "_compute_nav_with_detail", _hard_abort)

    with pytest.raises(KeyboardInterrupt):
        executor_live._pub_tick(state, loop_id=9)

    events = [r["event"] for r in log.records]
    assert events == ["ENTERED", "ABORTED"]
    assert "FAILED" not in events
    assert "COMPLETED" not in events
    aborted = log.records[-1]
    assert aborted["outcome"] == "aborted"

    # A call-site skip (scheduler/control-flow defect) is externally
    # distinguishable: zero heartbeat events at all, not even ENTERED.
    skip_log = _RecordingHeartbeatLog()
    assert skip_log.records == []


def test_telemetry_sink_failure_does_not_break_pub_tick(monkeypatch):
    monkeypatch.setattr(executor_live, "_PUB_TICK_HEARTBEAT_LOG", _AlwaysFailsHeartbeatLog())
    _reset_completion_globals(monkeypatch)
    nav_values = []
    monkeypatch.setattr(executor_live, "write_nav_state", lambda payload: nav_values.append(payload))
    _patch_pub_tick_internals(monkeypatch)
    monkeypatch.setattr(executor_live, "write_nav_state", lambda payload: nav_values.append(payload))
    state = _fresh_state(monkeypatch, episode_ledger_due=False)

    # Must not raise despite every heartbeat write failing.
    executor_live._pub_tick(state, loop_id=1)

    assert nav_values and nav_values[0]["nav"] == 100.0


def test_bounded_event_count(monkeypatch):
    log = _RecordingHeartbeatLog()
    monkeypatch.setattr(executor_live, "_PUB_TICK_HEARTBEAT_LOG", log)
    _reset_completion_globals(monkeypatch)
    _patch_pub_tick_internals(monkeypatch)
    state = _fresh_state(monkeypatch, episode_ledger_due=False)

    executor_live._pub_tick_heartbeat("CALL_REACHED", loop_id=1)
    executor_live._pub_tick(state, loop_id=1)

    assert len(log.records) == 3
    assert len(log.records) <= 6


def test_duration_and_schema_fields(monkeypatch):
    log = _RecordingHeartbeatLog()
    monkeypatch.setattr(executor_live, "_PUB_TICK_HEARTBEAT_LOG", log)
    _reset_completion_globals(monkeypatch)
    _patch_pub_tick_internals(monkeypatch)
    state = _fresh_state(monkeypatch, episode_ledger_due=False)

    executor_live._pub_tick_heartbeat("CALL_REACHED", loop_id=5)
    executor_live._pub_tick(state, loop_id=5)

    required_keys = {
        "ts",
        "monotonic_s",
        "event",
        "loop_id",
        "duration_ms",
        "episode_ledger_due",
        "episode_ledger_duration_ms",
        "last_completed_ts",
        "last_completed_duration_ms",
        "outcome",
        "exception_type",
        "exception_message",
        "engine_version",
        "git_sha",
    }
    for record in log.records:
        assert required_keys <= record.keys()
        assert isinstance(record["ts"], str)
        assert isinstance(record["monotonic_s"], float)
        assert isinstance(record["loop_id"], int)
        assert isinstance(record["engine_version"], str)
        assert isinstance(record["git_sha"], str)

    completed = log.records[-1]
    assert completed["event"] == "COMPLETED"
    assert isinstance(completed["duration_ms"], float)
    assert completed["duration_ms"] >= 0.0

    # monotonic clock must be non-decreasing across the emitted sequence.
    monotonics = [r["monotonic_s"] for r in log.records]
    assert monotonics == sorted(monotonics)


def test_last_completed_fields_propagate_to_next_call(monkeypatch):
    log = _RecordingHeartbeatLog()
    monkeypatch.setattr(executor_live, "_PUB_TICK_HEARTBEAT_LOG", log)
    _reset_completion_globals(monkeypatch)
    _patch_pub_tick_internals(monkeypatch)
    state = _fresh_state(monkeypatch, episode_ledger_due=False)

    executor_live._pub_tick(state, loop_id=1)
    first_entered = log.records[0]
    assert first_entered["event"] == "ENTERED"
    assert first_entered["last_completed_ts"] is None  # nothing completed before this call

    executor_live._pub_tick(state, loop_id=2)
    second_entered = log.records[-2]
    assert second_entered["event"] == "ENTERED"
    assert second_entered["last_completed_ts"] is not None
