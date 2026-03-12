"""Tests for percentile_normalize_scores in execution/hydra_engine.py.

After the score-field split, percentile values go to ``score_ranked`` /
``hybrid_score_ranked`` (diagnostic only).  The original ``score`` field
is preserved unchanged for cross-engine ECS arbitration.
"""
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
        # Ranked scores preserve ordering: E < D < A < B < C
        assert intents[4]["score_ranked"] < intents[3]["score_ranked"]  # E < D
        assert intents[3]["score_ranked"] < intents[0]["score_ranked"]  # D < A
        assert intents[0]["score_ranked"] < intents[1]["score_ranked"]  # A < B
        assert intents[1]["score_ranked"] < intents[2]["score_ranked"]  # B < C

    def test_absolute_score_preserved(self):
        """score field must NOT be overwritten by percentile values."""
        intents = [
            {"symbol": "BTC", "score": 0.54},
            {"symbol": "ETH", "score": 0.48},
            {"symbol": "SOL", "score": 0.36},
        ]
        percentile_normalize_scores(intents)
        # Absolute scores unchanged
        assert intents[0]["score"] == pytest.approx(0.54, abs=0.001)
        assert intents[1]["score"] == pytest.approx(0.48, abs=0.001)
        assert intents[2]["score"] == pytest.approx(0.36, abs=0.001)
        # Ranked scores exist and span the range
        ranked = sorted(i["score_ranked"] for i in intents)
        assert ranked[0] == pytest.approx(0.02, abs=0.001)
        assert ranked[-1] == pytest.approx(0.98, abs=0.001)

    def test_spread_expanded_in_ranked(self):
        intents = [
            {"score": 0.54},
            {"score": 0.55},
            {"score": 0.56},
            {"score": 0.57},
            {"score": 0.58},
        ]
        raw_spread = 0.58 - 0.54
        percentile_normalize_scores(intents)
        ranked = sorted(i["score_ranked"] for i in intents)
        new_spread = ranked[-1] - ranked[0]
        assert new_spread > raw_spread * 5  # Ranked scores dramatically wider

    def test_clamp_boundaries(self):
        intents = [
            {"score": 0.10},
            {"score": 0.50},
            {"score": 0.90},
        ]
        percentile_normalize_scores(intents, clamp_lo=0.02, clamp_hi=0.98)
        ranked = sorted(i["score_ranked"] for i in intents)
        assert ranked[0] == pytest.approx(0.02, abs=0.001)
        assert ranked[-1] == pytest.approx(0.98, abs=0.001)

    def test_single_intent_midpoint(self):
        intents = [{"score": 0.75}]
        percentile_normalize_scores(intents)
        # Ranked diagnostic gets midpoint
        assert intents[0]["score_ranked"] == pytest.approx(0.5, abs=0.001)
        assert intents[0]["hybrid_score_ranked"] == pytest.approx(0.5, abs=0.001)
        # Absolute score preserved
        assert intents[0].get("score") == pytest.approx(0.75, abs=0.001)
        assert intents[0]["hybrid_score"] == pytest.approx(0.75, abs=0.001)

    def test_raw_score_preserved_before_normalization(self):
        intents = [
            {"score": 0.40},
            {"score": 0.60},
        ]
        percentile_normalize_scores(intents)
        assert intents[0]["raw_score"] == pytest.approx(0.40, abs=0.001)
        assert intents[1]["raw_score"] == pytest.approx(0.60, abs=0.001)

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
        # Ranked scores for the two scored intents
        assert intents[0]["score_ranked"] == pytest.approx(0.02, abs=0.001)
        assert intents[2]["score_ranked"] == pytest.approx(0.98, abs=0.001)
        # Absolute scores preserved
        assert intents[0]["score"] == pytest.approx(0.50, abs=0.001)
        assert intents[2]["score"] == pytest.approx(0.60, abs=0.001)
        # Unscored intent unchanged
        assert "score_ranked" not in intents[1]

    def test_hybrid_score_ranked_stamped(self):
        intents = [
            {"score": 0.40},
            {"score": 0.60},
            {"score": 0.80},
        ]
        percentile_normalize_scores(intents)
        for intent in intents:
            assert "hybrid_score_ranked" in intent
            assert intent["hybrid_score_ranked"] == intent["score_ranked"]

    def test_ties_handled(self):
        intents = [
            {"score": 0.50},
            {"score": 0.50},
            {"score": 0.70},
        ]
        percentile_normalize_scores(intents)
        # Two tied ranked scores should be equal
        assert intents[0]["score_ranked"] == intents[1]["score_ranked"]
        assert intents[0]["score_ranked"] < intents[2]["score_ranked"]

    def test_custom_clamp_range(self):
        intents = [
            {"score": 0.10},
            {"score": 0.90},
        ]
        percentile_normalize_scores(intents, clamp_lo=0.10, clamp_hi=0.90)
        ranked = sorted(i["score_ranked"] for i in intents)
        assert ranked[0] == pytest.approx(0.10, abs=0.001)
        assert ranked[-1] == pytest.approx(0.90, abs=0.001)

    def test_five_equal_scores(self):
        """All identical scores should all get the midpoint."""
        intents = [{"score": 0.55} for _ in range(5)]
        percentile_normalize_scores(intents)
        # All tied → average rank = 2.0 out of 4 → pct = 0.5
        mid = 0.02 + 0.96 * 0.5
        for intent in intents:
            assert intent["score_ranked"] == pytest.approx(mid, abs=0.001)
            # Absolute score untouched
            assert intent["score"] == pytest.approx(0.55, abs=0.001)

    def test_non_dict_items_skipped(self):
        intents = [
            {"score": 0.40},
            "not a dict",  # type: ignore[list-item]
            {"score": 0.60},
        ]
        percentile_normalize_scores(intents)
        assert intents[0]["score_ranked"] == pytest.approx(0.02, abs=0.001)
        assert intents[2]["score_ranked"] == pytest.approx(0.98, abs=0.001)
        assert intents[1] == "not a dict"

    def test_sol_not_crushed_in_three_symbol_batch(self):
        """Regression: SOL absolute score must not collapse to 0.02."""
        intents = [
            {"symbol": "BTCUSDT", "score": 0.54},
            {"symbol": "ETHUSDT", "score": 0.48},
            {"symbol": "SOLUSDT", "score": 0.36},
        ]
        percentile_normalize_scores(intents)
        sol = next(i for i in intents if i["symbol"] == "SOLUSDT")
        # Absolute score preserved (was being crushed to 0.02 before fix)
        assert sol["score"] == pytest.approx(0.36, abs=0.001)
        assert sol["hybrid_score"] == pytest.approx(0.36, abs=0.001)
        # Ranked diagnostic correctly assigns lowest percentile
        assert sol["score_ranked"] == pytest.approx(0.02, abs=0.001)

    def test_singleton_consistency(self):
        """SOL alone vs SOL-with-others must produce same absolute score."""
        sol_alone = [{"symbol": "SOLUSDT", "score": 0.36}]
        sol_with_others = [
            {"symbol": "BTCUSDT", "score": 0.54},
            {"symbol": "ETHUSDT", "score": 0.48},
            {"symbol": "SOLUSDT", "score": 0.36},
        ]
        percentile_normalize_scores(sol_alone)
        percentile_normalize_scores(sol_with_others)
        sol_a = sol_alone[0]
        sol_b = next(i for i in sol_with_others if i["symbol"] == "SOLUSDT")
        # Both must have same absolute score
        assert sol_a["score"] == sol_b["score"]
        assert sol_a["hybrid_score"] == sol_b["hybrid_score"]
