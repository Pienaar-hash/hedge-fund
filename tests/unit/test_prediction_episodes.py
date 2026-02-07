# tests/unit/test_prediction_episodes.py
"""
Tests for prediction/prediction_episodes.py — episode lifecycle and scoring.

Covers:
    - Brier and log scoring rules
    - Episode lifecycle (open → record → resolve/expire)
    - Deterministic episode IDs
    - Per-source attribution scoring
    - Trust weight advisory computation
    - JSONL logging
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from prediction.prediction_episodes import (
    PredictionEpisode,
    brier_score,
    compute_trust_advisory,
    derive_episode_id,
    expire_episode,
    log_episode,
    log_score,
    open_episode,
    record_belief,
    resolve_episode,
    score_forecast,
)


# ---------------------------------------------------------------------------
# Scoring rules
# ---------------------------------------------------------------------------

class TestBrierScore:
    def test_perfect_prediction(self):
        assert brier_score(1.0, 1.0) == 0.0

    def test_worst_prediction(self):
        assert brier_score(0.0, 1.0) == 1.0

    def test_random_prediction(self):
        assert abs(brier_score(0.5, 1.0) - 0.25) < 1e-10

    def test_symmetry(self):
        assert abs(brier_score(0.3, 0.0) - brier_score(0.7, 1.0)) < 1e-10


class TestLogScore:
    def test_confident_correct(self):
        score = log_score(0.99, 1.0)
        assert score < 0.02  # very low (good)

    def test_confident_wrong(self):
        score = log_score(0.01, 1.0)
        assert score > 4.0  # very high (bad)

    def test_random(self):
        score = log_score(0.5, 1.0)
        assert abs(score - math.log(2)) < 1e-10


class TestScoreForecast:
    def test_binary_brier(self):
        probs = {"O_yes": 0.8, "O_no": 0.2}
        result = score_forecast(probs, "O_yes", "brier")
        assert result["scoring_rule"] == "brier"
        assert result["resolved_outcome"] == "O_yes"
        # O_yes: (0.8 - 1.0)^2 = 0.04, O_no: (0.2 - 0.0)^2 = 0.04
        assert abs(result["per_outcome"]["O_yes"] - 0.04) < 1e-10
        assert abs(result["per_outcome"]["O_no"] - 0.04) < 1e-10
        assert abs(result["aggregate_score"] - 0.04) < 1e-10

    def test_three_way(self):
        probs = {"A": 0.5, "B": 0.3, "C": 0.2}
        result = score_forecast(probs, "B")
        assert result["resolved_outcome"] == "B"
        assert "A" in result["per_outcome"]


# ---------------------------------------------------------------------------
# Episode lifecycle
# ---------------------------------------------------------------------------

class TestEpisodeLifecycle:
    def test_open(self):
        ep = open_episode("Q_test", initial_probs={"O_yes": 0.5, "O_no": 0.5})
        assert ep.episode_id.startswith("PEP_")
        assert ep.state == "OPEN"
        assert ep.question_id == "Q_test"
        assert ep.belief_count == 0

    def test_record_belief(self):
        ep = open_episode("Q_test")
        ep = record_belief(ep, belief_event_id="BEL_abc", decision_id="DEC_1", permit_id="PRM_1")
        assert ep.belief_count == 1
        assert "BEL_abc" in ep.belief_event_ids
        assert "DEC_1" in ep.decision_ids

    def test_record_multiple_deduplicates_decisions(self):
        ep = open_episode("Q_test")
        ep = record_belief(ep, belief_event_id="BEL_1", decision_id="DEC_1", permit_id="PRM_1")
        ep = record_belief(ep, belief_event_id="BEL_2", decision_id="DEC_1", permit_id="PRM_2")
        assert ep.belief_count == 2
        assert ep.decision_ids.count("DEC_1") == 1  # deduplicated

    def test_resolve(self):
        ep = open_episode("Q_test")
        ep = record_belief(ep, belief_event_id="BEL_1", decision_id="D", permit_id="P")
        ep = resolve_episode(
            ep,
            resolved_outcome="O_yes",
            final_probs={"O_yes": 0.8, "O_no": 0.2},
        )
        assert ep.state == "RESOLVED"
        assert ep.resolved_outcome == "O_yes"
        assert ep.scoring is not None
        assert ep.resolved_ts is not None

    def test_resolve_with_source_attribution(self):
        ep = open_episode("Q_test")
        ep = resolve_episode(
            ep,
            resolved_outcome="O_yes",
            final_probs={"O_yes": 0.8, "O_no": 0.2},
            source_beliefs={
                "polymarket": [{"outcome_id": "O_yes", "p": 0.9, "confidence": 1.0}],
                "human": [{"outcome_id": "O_yes", "p": 0.6, "confidence": 0.8}],
            },
        )
        assert ep.source_scores is not None
        assert "polymarket" in ep.source_scores
        assert "human" in ep.source_scores
        # Polymarket closer to truth, should score better (lower brier)
        poly_score = ep.source_scores["polymarket"]["aggregate_score"]
        human_score = ep.source_scores["human"]["aggregate_score"]
        assert poly_score < human_score

    def test_expire(self):
        ep = open_episode("Q_test")
        ep = expire_episode(ep)
        assert ep.state == "EXPIRED"
        assert ep.resolved_ts is not None


# ---------------------------------------------------------------------------
# Episode ID
# ---------------------------------------------------------------------------

class TestEpisodeId:
    def test_deterministic(self):
        id1 = derive_episode_id("Q_test", "2026-01-01T00:00:00Z")
        id2 = derive_episode_id("Q_test", "2026-01-01T00:00:00Z")
        assert id1 == id2
        assert id1.startswith("PEP_")

    def test_different_question_different_id(self):
        id1 = derive_episode_id("Q_test", "2026-01-01T00:00:00Z")
        id2 = derive_episode_id("Q_other", "2026-01-01T00:00:00Z")
        assert id1 != id2


# ---------------------------------------------------------------------------
# Trust advisory
# ---------------------------------------------------------------------------

class TestTrustAdvisory:
    def test_good_forecaster_positive_delta(self):
        scores = {
            "polymarket": {
                "scoring_rule": "brier",
                "aggregate_score": 0.05,  # much better than random (0.25)
            }
        }
        adv = compute_trust_advisory(scores)
        assert adv["polymarket"] > 0  # positive trust delta

    def test_bad_forecaster_negative_delta(self):
        scores = {
            "bad_model": {
                "scoring_rule": "brier",
                "aggregate_score": 0.45,  # worse than random
            }
        }
        adv = compute_trust_advisory(scores)
        assert adv["bad_model"] < 0  # negative trust delta

    def test_random_forecaster_near_zero(self):
        scores = {
            "random": {
                "scoring_rule": "brier",
                "aggregate_score": 0.25,  # exactly random for binary
            }
        }
        adv = compute_trust_advisory(scores)
        assert abs(adv["random"]) < 1e-6


# ---------------------------------------------------------------------------
# JSONL logging
# ---------------------------------------------------------------------------

class TestEpisodeLogging:
    def test_log_episode(self, tmp_path):
        ep = open_episode("Q_test")
        log_path = tmp_path / "episodes.jsonl"
        log_episode(ep, log_path)
        assert log_path.exists()
        data = json.loads(log_path.read_text().strip())
        assert data["episode_id"].startswith("PEP_")
        assert data["state"] == "OPEN"
