"""Tests for percentile_normalize_scores in execution/hydra_engine.py."""
from __future__ import annotations

import pytest

from execution.hydra_engine import percentile_normalize_scores


class TestPercentileNormalize:
    def test_basic_ordering_preserved(self):
        intents = [
            {"symbol": "A", "score": 0.54},
            {"symbol": "B", "score": 0.56},
            {"symbol": "C", "score": 0.58},
            {"symbol": "D", "score": 0.52},
            {"symbol": "E", "score": 0.50},
        ]
        percentile_normalize_scores(intents)
        # Original order: E(0.50) < D(0.52) < A(0.54) < B(0.56) < C(0.58)
        # After normalize: E < D < A < B < C (same ordering)
        assert intents[4]["score"] < intents[3]["score"]  # E < D
        assert intents[3]["score"] < intents[0]["score"]  # D < A
        assert intents[0]["score"] < intents[1]["score"]  # A < B
        assert intents[1]["score"] < intents[2]["score"]  # B < C

    def test_spread_expanded(self):
        intents = [
            {"score": 0.54},
            {"score": 0.55},
            {"score": 0.56},
            {"score": 0.57},
            {"score": 0.58},
        ]
        raw_spread = 0.58 - 0.54
        percentile_normalize_scores(intents)
        scores = sorted(i["score"] for i in intents)
        new_spread = scores[-1] - scores[0]
        assert new_spread > raw_spread * 5  # Should be dramatically wider

    def test_clamp_boundaries(self):
        intents = [
            {"score": 0.10},
            {"score": 0.50},
            {"score": 0.90},
        ]
        percentile_normalize_scores(intents, clamp_lo=0.02, clamp_hi=0.98)
        scores = sorted(i["score"] for i in intents)
        assert scores[0] == pytest.approx(0.02, abs=0.001)
        assert scores[-1] == pytest.approx(0.98, abs=0.001)

    def test_single_intent_unchanged(self):
        intents = [{"score": 0.75}]
        percentile_normalize_scores(intents)
        assert intents[0]["score"] == 0.75

    def test_empty_list_no_crash(self):
        intents: list = []
        percentile_normalize_scores(intents)
        assert intents == []

    def test_missing_score_skipped(self):
        intents = [
            {"score": 0.50},
            {"symbol": "X"},  # no score
            {"score": 0.60},
        ]
        percentile_normalize_scores(intents)
        # Only two scored intents → lo and hi
        assert intents[0]["score"] == pytest.approx(0.02, abs=0.001)
        assert intents[2]["score"] == pytest.approx(0.98, abs=0.001)
        # Unscored intent unchanged
        assert "score" not in intents[1] or intents[1].get("score") is None

    def test_hybrid_score_stamped(self):
        intents = [
            {"score": 0.40},
            {"score": 0.60},
            {"score": 0.80},
        ]
        percentile_normalize_scores(intents)
        for intent in intents:
            assert "hybrid_score" in intent
            assert intent["hybrid_score"] == intent["score"]

    def test_ties_handled(self):
        intents = [
            {"score": 0.50},
            {"score": 0.50},
            {"score": 0.70},
        ]
        percentile_normalize_scores(intents)
        # Two tied scores should get the same percentile
        assert intents[0]["score"] == intents[1]["score"]
        assert intents[0]["score"] < intents[2]["score"]

    def test_custom_clamp_range(self):
        intents = [
            {"score": 0.10},
            {"score": 0.90},
        ]
        percentile_normalize_scores(intents, clamp_lo=0.10, clamp_hi=0.90)
        scores = sorted(i["score"] for i in intents)
        assert scores[0] == pytest.approx(0.10, abs=0.001)
        assert scores[-1] == pytest.approx(0.90, abs=0.001)

    def test_five_equal_scores(self):
        """All identical scores should all get the midpoint."""
        intents = [{"score": 0.55} for _ in range(5)]
        percentile_normalize_scores(intents)
        # All tied → average rank = 2.0 out of 4 → pct = 0.5
        mid = 0.02 + 0.96 * 0.5
        for intent in intents:
            assert intent["score"] == pytest.approx(mid, abs=0.001)

    def test_non_dict_items_skipped(self):
        intents = [
            {"score": 0.40},
            "not a dict",  # type: ignore[list-item]
            {"score": 0.60},
        ]
        percentile_normalize_scores(intents)
        assert intents[0]["score"] == pytest.approx(0.02, abs=0.001)
        assert intents[2]["score"] == pytest.approx(0.98, abs=0.001)
        assert intents[1] == "not a dict"
