# tests/unit/test_prediction_dle_gate.py
"""
Tests for prediction/dle_prediction_gate.py — DLE authority gate for belief writes.

Covers:
    - Decision building + deterministic IDs
    - Permit issuance
    - Veto reasons (dataset rejected, question missing, outcome missing, delta exceeded)
    - Full gate pipeline (gate_belief_write)
    - Writer fail-open behavior
    - Feature flag disable
    - Combined gate_and_ingest
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from prediction.dle_prediction_gate import (
    PredictionDLEEvent,
    PredictionDLEWriter,
    PredictionDecision,
    PredictionPermit,
    VetoReason,
    build_decision,
    derive_prediction_decision_id,
    derive_prediction_permit_id,
    gate_and_ingest,
    gate_belief_write,
    issue_permit,
    reset_prediction_writer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def question_graph():
    return {
        "questions": {
            "Q_test": {
                "question_id": "Q_test",
                "outcomes": ["O_yes", "O_no"],
                "constraints": [
                    {"type": "SUM_TO_ONE", "outcomes": ["O_yes", "O_no"]},
                ],
                "state": "ACTIVE",
            },
            "Q_closed": {
                "question_id": "Q_closed",
                "outcomes": ["O_a"],
                "constraints": [],
                "state": "CLOSED",
            },
        }
    }


@pytest.fixture
def admission():
    return {
        "datasets": {
            "poly": {"state": "PRODUCTION_ELIGIBLE"},
            "human": {"state": "OBSERVE_ONLY"},
            "model_x": {"state": "REJECTED"},
        }
    }


@pytest.fixture(autouse=True)
def _reset_writer():
    reset_prediction_writer()
    yield
    reset_prediction_writer()


# ---------------------------------------------------------------------------
# Decision IDs
# ---------------------------------------------------------------------------

class TestDecisionId:
    def test_deterministic(self):
        kwargs = dict(
            phase_id="P0_OBSERVE",
            action_class="WRITE_BELIEF",
            constraints={"max_delta": 0.5},
            policy_version="v1.0",
        )
        id1 = derive_prediction_decision_id(**kwargs)
        id2 = derive_prediction_decision_id(**kwargs)
        assert id1 == id2
        assert id1.startswith("DEC_")

    def test_different_policy_different_id(self):
        base = dict(
            phase_id="P0_OBSERVE",
            action_class="WRITE_BELIEF",
            constraints={"max_delta": 0.5},
        )
        id1 = derive_prediction_decision_id(**base, policy_version="v1.0")
        id2 = derive_prediction_decision_id(**base, policy_version="v2.0")
        assert id1 != id2


# ---------------------------------------------------------------------------
# Permit IDs
# ---------------------------------------------------------------------------

class TestPermitId:
    def test_deterministic(self):
        kwargs = dict(
            decision_id="DEC_abc",
            dataset_id="poly",
            question_id="Q_test",
            outcome_id="O_yes",
            issued_at_iso="2026-01-01T00:00:00Z",
        )
        id1 = derive_prediction_permit_id(**kwargs)
        id2 = derive_prediction_permit_id(**kwargs)
        assert id1 == id2
        assert id1.startswith("PRM_")


# ---------------------------------------------------------------------------
# Decision building
# ---------------------------------------------------------------------------

class TestBuildDecision:
    def test_defaults(self):
        dec = build_decision()
        assert dec.decision_id.startswith("DEC_")
        assert dec.phase_id == "P0_OBSERVE"
        assert dec.action_class == "WRITE_BELIEF"
        assert dec.max_delta == 0.50
        assert dec.max_uses == 1

    def test_custom_scope(self):
        dec = build_decision(
            allowed_datasets=["poly"],
            allowed_questions=["Q_test"],
            max_delta=0.1,
        )
        assert "poly" in dec.allowed_datasets
        assert dec.max_delta == 0.1


# ---------------------------------------------------------------------------
# Permit issuance
# ---------------------------------------------------------------------------

class TestIssuePermit:
    def test_basic(self):
        dec = build_decision()
        permit = issue_permit(
            decision=dec,
            dataset_id="poly",
            question_id="Q_test",
            outcome_id="O_yes",
        )
        assert permit.permit_id.startswith("PRM_")
        assert permit.decision_id == dec.decision_id
        assert permit.single_use is True


# ---------------------------------------------------------------------------
# Gate: veto checks
# ---------------------------------------------------------------------------

class TestGateBelief:
    def test_disabled(self, question_graph):
        dec_id, prm_id, veto = gate_belief_write(
            dataset_id="poly",
            question_id="Q_test",
            outcome_id="O_yes",
            p=0.5,
            dataset_state="PRODUCTION_ELIGIBLE",
            question_graph=question_graph,
            enabled=False,
        )
        assert veto == VetoReason.DISABLED
        assert dec_id is None

    def test_dataset_rejected(self, question_graph):
        dec_id, prm_id, veto = gate_belief_write(
            dataset_id="model_x",
            question_id="Q_test",
            outcome_id="O_yes",
            p=0.5,
            dataset_state="REJECTED",
            question_graph=question_graph,
            enabled=True,
            write_logs=False,
        )
        assert veto == VetoReason.DATASET_REJECTED

    def test_question_not_found(self, question_graph):
        _, _, veto = gate_belief_write(
            dataset_id="poly",
            question_id="Q_unknown",
            outcome_id="O_yes",
            p=0.5,
            dataset_state="PRODUCTION_ELIGIBLE",
            question_graph=question_graph,
            enabled=True,
            write_logs=False,
        )
        assert veto == VetoReason.QUESTION_NOT_FOUND

    def test_question_inactive(self, question_graph):
        _, _, veto = gate_belief_write(
            dataset_id="poly",
            question_id="Q_closed",
            outcome_id="O_a",
            p=0.5,
            dataset_state="PRODUCTION_ELIGIBLE",
            question_graph=question_graph,
            enabled=True,
            write_logs=False,
        )
        assert veto == VetoReason.QUESTION_INACTIVE

    def test_outcome_not_found(self, question_graph):
        _, _, veto = gate_belief_write(
            dataset_id="poly",
            question_id="Q_test",
            outcome_id="O_invalid",
            p=0.5,
            dataset_state="PRODUCTION_ELIGIBLE",
            question_graph=question_graph,
            enabled=True,
            write_logs=False,
        )
        assert veto == VetoReason.OUTCOME_NOT_FOUND

    def test_delta_exceeded(self, question_graph):
        dec = build_decision(max_delta=0.1)
        _, _, veto = gate_belief_write(
            dataset_id="poly",
            question_id="Q_test",
            outcome_id="O_yes",
            p=0.8,
            prior_p=0.3,
            dataset_state="PRODUCTION_ELIGIBLE",
            question_graph=question_graph,
            decision=dec,
            enabled=True,
            write_logs=False,
        )
        assert veto == VetoReason.DELTA_EXCEEDED

    def test_dataset_not_allowed(self, question_graph):
        dec = build_decision(allowed_datasets=["other_src"])
        _, _, veto = gate_belief_write(
            dataset_id="poly",
            question_id="Q_test",
            outcome_id="O_yes",
            p=0.5,
            dataset_state="PRODUCTION_ELIGIBLE",
            question_graph=question_graph,
            decision=dec,
            enabled=True,
            write_logs=False,
        )
        assert veto == VetoReason.DATASET_NOT_ALLOWED

    def test_all_checks_pass(self, question_graph):
        dec_id, prm_id, veto = gate_belief_write(
            dataset_id="poly",
            question_id="Q_test",
            outcome_id="O_yes",
            p=0.6,
            prior_p=0.5,
            dataset_state="PRODUCTION_ELIGIBLE",
            question_graph=question_graph,
            enabled=True,
            write_logs=False,
        )
        assert veto is None
        assert dec_id is not None
        assert prm_id is not None


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class TestWriter:
    def test_writes_to_file(self, tmp_path):
        log = tmp_path / "test.jsonl"
        w = PredictionDLEWriter(str(log))
        w.write(PredictionDLEEvent(
            schema_version="test_v1",
            event_type="DECISION",
            ts="2026-01-01T00:00:00Z",
            payload={"test": True},
        ))
        assert log.exists()
        data = json.loads(log.read_text().strip())
        assert data["event_type"] == "DECISION"

    def test_fail_open(self):
        """Writer errors never raise — fail-open pattern."""
        w = PredictionDLEWriter("/nonexistent/deeply/nested/impossible.jsonl")
        # Should not raise
        w.write(PredictionDLEEvent(
            schema_version="test_v1",
            event_type="TEST",
            ts="now",
            payload={},
        ))
        assert w.write_failure_count >= 0  # may or may not fail depending on OS


# ---------------------------------------------------------------------------
# Combined gate + ingest
# ---------------------------------------------------------------------------

class TestGateAndIngest:
    def test_full_pipeline(self, question_graph, admission, tmp_path):
        log_path = tmp_path / "beliefs.jsonl"
        dec_id, prm_id, veto = gate_and_ingest(
            dataset_id="poly",
            question_id="Q_test",
            outcome_id="O_yes",
            p=0.7,
            confidence=0.9,
            question_graph=question_graph,
            admission=admission,
            enabled=True,
            write_logs=False,
            belief_log_path=log_path,
        )
        assert veto is None
        assert dec_id is not None
        assert prm_id is not None
        assert log_path.exists()

    def test_rejected_dataset_vetoed(self, question_graph, admission, tmp_path):
        log_path = tmp_path / "beliefs.jsonl"
        _, _, veto = gate_and_ingest(
            dataset_id="model_x",
            question_id="Q_test",
            outcome_id="O_yes",
            p=0.5,
            confidence=0.5,
            question_graph=question_graph,
            admission=admission,
            enabled=True,
            write_logs=False,
            belief_log_path=log_path,
        )
        assert veto == VetoReason.DATASET_REJECTED
