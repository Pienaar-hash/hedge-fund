# tests/unit/test_prediction_state_surface.py
"""
Tests for prediction/state_surface.py — dashboard state file builder.

Covers:
    - Empty‑log baseline
    - JSONL event counting
    - Timestamp extraction (last)
    - Dataset state section (via dataset_admission)
    - Solver status aggregation
    - Atomic write via write_prediction_state
    - Schema stability for dashboard readers
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from prediction.state_surface import build_prediction_state, write_prediction_state


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# Empty / missing logs
# ---------------------------------------------------------------------------

class TestEmptyState:
    def test_baseline_all_zeros(self, tmp_path):
        state = build_prediction_state(log_dir=tmp_path)
        assert state["belief_events"]["count"] == 0
        assert state["belief_events"]["last_ts"] is None
        assert state["episodes"]["count"] == 0
        assert state["dle_events"] == {}  # empty event-type dict
        assert state["firewall"]["denial_count"] == 0
        assert state["rollback_triggers"]["count"] == 0
        assert state["dataset_states"] == {}

    def test_solver_statuses_empty(self, tmp_path):
        state = build_prediction_state(log_dir=tmp_path)
        assert state["aggregates"]["solver_statuses"] == {}

    def test_updated_ts_present(self, tmp_path):
        state = build_prediction_state(log_dir=tmp_path)
        assert "updated_ts" in state
        assert isinstance(state["updated_ts"], str)


# ---------------------------------------------------------------------------
# JSONL counting
# ---------------------------------------------------------------------------

class TestEventCounting:
    def test_belief_event_count(self, tmp_path):
        pred_dir = tmp_path / "prediction"
        _write_jsonl(pred_dir / "belief_events.jsonl", [
            {"event_id": "BEL_001", "ts": 1000},
            {"event_id": "BEL_002", "ts": 2000},
            {"event_id": "BEL_003", "ts": 3000},
        ])
        state = build_prediction_state(log_dir=tmp_path)
        assert state["belief_events"]["count"] == 3

    def test_multiple_log_types(self, tmp_path):
        pred_dir = tmp_path / "prediction"
        _write_jsonl(pred_dir / "belief_events.jsonl", [
            {"ts": 1000}, {"ts": 2000},
        ])
        _write_jsonl(pred_dir / "prediction_episodes.jsonl", [
            {"ts": 3000},
        ])
        _write_jsonl(pred_dir / "dle_prediction_events.jsonl", [
            {"ts": 4000, "event_type": "DECISION"},
            {"ts": 5000, "event_type": "PERMIT"},
            {"ts": 6000, "event_type": "DECISION"},
            {"ts": 7000, "event_type": "VETO"},
        ])
        state = build_prediction_state(log_dir=tmp_path)
        assert state["belief_events"]["count"] == 2
        assert state["episodes"]["count"] == 1
        # dle_events is a dict of event_type → count
        total_dle = sum(state["dle_events"].values())
        assert total_dle == 4


# ---------------------------------------------------------------------------
# Timestamp extraction
# ---------------------------------------------------------------------------

class TestTimestamps:
    def test_last_ts(self, tmp_path):
        pred_dir = tmp_path / "prediction"
        _write_jsonl(pred_dir / "belief_events.jsonl", [
            {"ts": 1000},
            {"ts": 2000},
            {"ts": 3000},
        ])
        state = build_prediction_state(log_dir=tmp_path)
        assert state["belief_events"]["last_ts"] == 3000

    def test_single_entry(self, tmp_path):
        pred_dir = tmp_path / "prediction"
        _write_jsonl(pred_dir / "belief_events.jsonl", [
            {"ts": 5000},
        ])
        state = build_prediction_state(log_dir=tmp_path)
        assert state["belief_events"]["last_ts"] == 5000


# ---------------------------------------------------------------------------
# Dataset state extraction (from dataset_admission config)
# ---------------------------------------------------------------------------

class TestDatasetStates:
    def test_extracts_prediction_datasets(self, tmp_path):
        admission = {
            "datasets": {
                "prediction_polymarket": {"state": "OBSERVE_ONLY"},
                "prediction_model_x": {"state": "REJECTED"},
                "binance_klines_1h": {"state": "PRODUCTION_ELIGIBLE"},  # not prediction_*
            }
        }
        state = build_prediction_state(log_dir=tmp_path, dataset_admission=admission)
        assert "prediction_polymarket" in state["dataset_states"]
        assert "prediction_model_x" in state["dataset_states"]
        assert "binance_klines_1h" not in state["dataset_states"]  # filtered
        assert state["dataset_states"]["prediction_polymarket"] == "OBSERVE_ONLY"

    def test_empty_without_admission(self, tmp_path):
        state = build_prediction_state(log_dir=tmp_path)
        assert state["dataset_states"] == {}


# ---------------------------------------------------------------------------
# Solver status aggregation
# ---------------------------------------------------------------------------

class TestSolverStatuses:
    def test_counts_by_status_field(self, tmp_path):
        pred_dir = tmp_path / "prediction"
        _write_jsonl(pred_dir / "aggregate_state.jsonl", [
            {"ts": 1, "status": "OK"},
            {"ts": 2, "status": "OK"},
            {"ts": 3, "status": "CLIPPED"},
        ])
        state = build_prediction_state(log_dir=tmp_path)
        assert state["aggregates"]["solver_statuses"]["OK"] == 2
        assert state["aggregates"]["solver_statuses"]["CLIPPED"] == 1


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------

class TestWritePredictionState:
    def test_writes_valid_json(self, tmp_path):
        state_dir = tmp_path / "state"
        write_prediction_state(
            log_dir=tmp_path,
            state_dir=state_dir,
        )
        out = state_dir / "prediction_state.json"
        assert out.exists()
        data = json.loads(out.read_text())
        assert "belief_events" in data
        assert "updated_ts" in data

    def test_overwrites_existing(self, tmp_path):
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        out = state_dir / "prediction_state.json"
        out.write_text('{"old": true}')
        write_prediction_state(log_dir=tmp_path, state_dir=state_dir)
        data = json.loads(out.read_text())
        assert "old" not in data
        assert "belief_events" in data


# ---------------------------------------------------------------------------
# Schema stability — dashboard contract
# ---------------------------------------------------------------------------

class TestSchemaStability:
    """Every key the dashboard might read must exist."""

    REQUIRED_TOP_KEYS = [
        "belief_events",
        "aggregates",
        "dle_events",
        "episodes",
        "firewall",
        "rollback_triggers",
        "dataset_states",
        "updated_ts",
        "phase",
        "enabled",
    ]

    def test_top_level_keys(self, tmp_path):
        state = build_prediction_state(log_dir=tmp_path)
        for key in self.REQUIRED_TOP_KEYS:
            assert key in state, f"Missing top-level key: {key}"

    def test_belief_section_has_count_and_last_ts(self, tmp_path):
        state = build_prediction_state(log_dir=tmp_path)
        assert "count" in state["belief_events"]
        assert "last_ts" in state["belief_events"]

    def test_rollback_section_has_count_and_by_type(self, tmp_path):
        state = build_prediction_state(log_dir=tmp_path)
        assert "count" in state["rollback_triggers"]
        assert "by_type" in state["rollback_triggers"]

    def test_firewall_section_has_denial_count(self, tmp_path):
        state = build_prediction_state(log_dir=tmp_path)
        assert "denial_count" in state["firewall"]
        assert "by_verdict" in state["firewall"]
