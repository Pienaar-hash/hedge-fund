# tests/unit/test_prediction_belief_ingest.py
"""
Tests for prediction/belief_ingest.py — belief event construction, ingestion,
and aggregate computation.

Covers:
    - BeliefEvent construction + deterministic IDs
    - Validation (bounds, question/outcome checks)
    - Dataset admission gating
    - Aggregate computation with constraint solving
    - JSONL logging
    - Full ingest pipeline
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from prediction.belief_ingest import (
    AggregateState,
    BeliefEvent,
    build_belief_event,
    compute_aggregate,
    derive_aggregate_hash,
    derive_belief_event_id,
    get_dataset_state,
    get_dataset_trust,
    ingest_belief,
    log_aggregate_state,
    log_belief_event,
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
                "title": "Test binary",
                "outcomes": ["O_yes", "O_no"],
                "constraints": [
                    {"type": "SUM_TO_ONE", "outcomes": ["O_yes", "O_no"]},
                    {"type": "BOUNDS", "outcomes": ["O_yes", "O_no"], "min": 0.0, "max": 1.0},
                ],
                "state": "ACTIVE",
            },
            "Q_inactive": {
                "question_id": "Q_inactive",
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
            "polymarket": {"state": "PRODUCTION_ELIGIBLE"},
            "human": {"state": "RESEARCH_ONLY"},
            "observe_src": {"state": "OBSERVE_ONLY"},
            "rejected_src": {"state": "REJECTED"},
        }
    }


# ---------------------------------------------------------------------------
# BeliefEvent ID derivation
# ---------------------------------------------------------------------------

class TestDeriveBeliefEventId:
    def test_deterministic(self):
        kwargs = dict(dataset_id="d1", question_id="Q1", outcome_id="O1", ts="2026-01-01T00:00:00Z", p=0.5)
        id1 = derive_belief_event_id(**kwargs)
        id2 = derive_belief_event_id(**kwargs)
        assert id1 == id2
        assert id1.startswith("BEL_")
        assert len(id1) == 4 + 12  # "BEL_" + 12 hex chars

    def test_different_inputs_different_id(self):
        base = dict(dataset_id="d1", question_id="Q1", outcome_id="O1", ts="2026-01-01T00:00:00Z", p=0.5)
        id1 = derive_belief_event_id(**base)
        id2 = derive_belief_event_id(**{**base, "p": 0.6})
        assert id1 != id2


# ---------------------------------------------------------------------------
# Event construction
# ---------------------------------------------------------------------------

class TestBuildBeliefEvent:
    def test_valid(self):
        ev = build_belief_event(
            dataset_id="polymarket",
            question_id="Q_test",
            outcome_id="O_yes",
            p=0.7,
            confidence=0.8,
            decision_id="DEC_abc",
            permit_id="PRM_xyz",
            ts="2026-01-01T00:00:00Z",
        )
        assert ev.belief_event_id.startswith("BEL_")
        assert ev.p == 0.7
        assert ev.confidence == 0.8
        assert ev.dle["decision_id"] == "DEC_abc"
        assert ev.dle["permit_id"] == "PRM_xyz"

    def test_invalid_probability(self):
        with pytest.raises(ValueError, match="Probability"):
            build_belief_event(
                dataset_id="d1", question_id="Q1", outcome_id="O1",
                p=1.5, confidence=0.5,
                decision_id="D", permit_id="P",
            )

    def test_invalid_confidence(self):
        with pytest.raises(ValueError, match="Confidence"):
            build_belief_event(
                dataset_id="d1", question_id="Q1", outcome_id="O1",
                p=0.5, confidence=-0.1,
                decision_id="D", permit_id="P",
            )

    def test_to_dict_roundtrip(self):
        ev = build_belief_event(
            dataset_id="d1", question_id="Q1", outcome_id="O1",
            p=0.5, confidence=0.5,
            decision_id="D", permit_id="P",
            ts="2026-01-01T00:00:00Z",
        )
        d = ev.to_dict()
        assert d["p"] == 0.5
        assert d["dle"]["decision_id"] == "D"


# ---------------------------------------------------------------------------
# Dataset admission
# ---------------------------------------------------------------------------

class TestDatasetAdmission:
    def test_known_states(self, admission):
        assert get_dataset_state("polymarket", admission) == "PRODUCTION_ELIGIBLE"
        assert get_dataset_state("human", admission) == "RESEARCH_ONLY"
        assert get_dataset_state("observe_src", admission) == "OBSERVE_ONLY"
        assert get_dataset_state("rejected_src", admission) == "REJECTED"

    def test_unknown_dataset_rejected(self, admission):
        assert get_dataset_state("unknown", admission) == "REJECTED"

    def test_trust_weights(self, admission):
        assert get_dataset_trust("polymarket", admission) == 1.0
        assert get_dataset_trust("human", admission) == 0.5
        assert get_dataset_trust("observe_src", admission) == 0.25
        assert get_dataset_trust("rejected_src", admission) == 0.0
        assert get_dataset_trust("unknown", admission) == 0.0


# ---------------------------------------------------------------------------
# Aggregate computation
# ---------------------------------------------------------------------------

class TestComputeAggregate:
    def test_single_source(self, question_graph, admission):
        events = [
            build_belief_event(
                dataset_id="polymarket", question_id="Q_test", outcome_id="O_yes",
                p=0.7, confidence=1.0, decision_id="D", permit_id="P",
                ts="2026-01-01T00:00:00Z",
            ),
            build_belief_event(
                dataset_id="polymarket", question_id="Q_test", outcome_id="O_no",
                p=0.3, confidence=1.0, decision_id="D", permit_id="P",
                ts="2026-01-01T00:00:00Z",
            ),
        ]
        agg = compute_aggregate(events, "Q_test", question_graph, admission)
        assert agg is not None
        assert abs(agg.probs["O_yes"] - 0.7) < 1e-8
        assert abs(agg.probs["O_no"] - 0.3) < 1e-8
        assert abs(sum(agg.probs.values()) - 1.0) < 1e-8

    def test_rejected_source_excluded(self, question_graph, admission):
        events = [
            build_belief_event(
                dataset_id="rejected_src", question_id="Q_test", outcome_id="O_yes",
                p=0.9, confidence=1.0, decision_id="D", permit_id="P",
                ts="2026-01-01T00:00:00Z",
            ),
        ]
        agg = compute_aggregate(events, "Q_test", question_graph, admission)
        assert agg is None  # Only source was rejected → no valid sources

    def test_unknown_question_returns_none(self, question_graph, admission):
        events = [
            build_belief_event(
                dataset_id="polymarket", question_id="Q_nonexistent", outcome_id="O_x",
                p=0.5, confidence=1.0, decision_id="D", permit_id="P",
                ts="2026-01-01T00:00:00Z",
            ),
        ]
        assert compute_aggregate(events, "Q_nonexistent", question_graph, admission) is None

    def test_aggregate_hash_deterministic(self):
        h1 = derive_aggregate_hash("Q1", {"A": 0.5, "B": 0.5})
        h2 = derive_aggregate_hash("Q1", {"A": 0.5, "B": 0.5})
        assert h1 == h2
        assert h1.startswith("sha256:")


# ---------------------------------------------------------------------------
# JSONL logging
# ---------------------------------------------------------------------------

class TestLogging:
    def test_log_belief_event(self, tmp_path):
        ev = build_belief_event(
            dataset_id="d1", question_id="Q1", outcome_id="O1",
            p=0.5, confidence=0.5, decision_id="D", permit_id="P",
            ts="2026-01-01T00:00:00Z",
        )
        log_path = tmp_path / "test_beliefs.jsonl"
        log_belief_event(ev, log_path)
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["belief_event_id"].startswith("BEL_")

    def test_log_aggregate_state(self, tmp_path):
        agg = AggregateState(
            ts="2026-01-01T00:00:00Z",
            question_id="Q1",
            probs={"O_a": 0.5, "O_b": 0.5},
            aggregate_hash="sha256:abcd1234",
            inputs_window={"from": "2026-01-01", "to": "2026-01-01"},
            solver={"name": "proj_simplex_v1", "status": "OK", "residual": 0.0},
        )
        log_path = tmp_path / "test_agg.jsonl"
        log_aggregate_state(agg, log_path)
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1


# ---------------------------------------------------------------------------
# Full ingest pipeline
# ---------------------------------------------------------------------------

class TestIngestBelief:
    def test_admitted_source(self, question_graph, admission, tmp_path):
        log_path = tmp_path / "beliefs.jsonl"
        ev = ingest_belief(
            dataset_id="polymarket",
            question_id="Q_test",
            outcome_id="O_yes",
            p=0.6,
            confidence=0.9,
            decision_id="DEC_test",
            permit_id="PRM_test",
            question_graph=question_graph,
            admission=admission,
            belief_log_path=log_path,
        )
        assert ev is not None
        assert ev.p == 0.6
        assert log_path.exists()

    def test_rejected_source_drops(self, question_graph, admission, tmp_path):
        log_path = tmp_path / "beliefs.jsonl"
        ev = ingest_belief(
            dataset_id="rejected_src",
            question_id="Q_test",
            outcome_id="O_yes",
            p=0.6,
            confidence=0.9,
            decision_id="D",
            permit_id="P",
            question_graph=question_graph,
            admission=admission,
            belief_log_path=log_path,
        )
        assert ev is None

    def test_unknown_question_drops(self, question_graph, admission, tmp_path):
        log_path = tmp_path / "beliefs.jsonl"
        ev = ingest_belief(
            dataset_id="polymarket",
            question_id="Q_unknown",
            outcome_id="O_yes",
            p=0.5,
            confidence=0.5,
            decision_id="D",
            permit_id="P",
            question_graph=question_graph,
            admission=admission,
            belief_log_path=log_path,
        )
        assert ev is None

    def test_invalid_outcome_drops(self, question_graph, admission, tmp_path):
        log_path = tmp_path / "beliefs.jsonl"
        ev = ingest_belief(
            dataset_id="polymarket",
            question_id="Q_test",
            outcome_id="O_invalid",
            p=0.5,
            confidence=0.5,
            decision_id="D",
            permit_id="P",
            question_graph=question_graph,
            admission=admission,
            belief_log_path=log_path,
        )
        assert ev is None

    def test_inactive_question_drops(self, question_graph, admission, tmp_path):
        log_path = tmp_path / "beliefs.jsonl"
        ev = ingest_belief(
            dataset_id="polymarket",
            question_id="Q_inactive",
            outcome_id="O_a",
            p=0.5,
            confidence=0.5,
            decision_id="D",
            permit_id="P",
            question_graph=question_graph,
            admission=admission,
            belief_log_path=log_path,
        )
        assert ev is None
