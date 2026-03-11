"""Tests for execution.candidate_selector — Phase 4 Commit 1."""
import pytest
from execution.candidate_selector import (
    build_candidates,
    select_executable_candidate,
)


def _make_intent(symbol="BTCUSDT", score=0.6, source="hydra", **extra):
    d = {"symbol": symbol, "hybrid_score": score, "source": source}
    d.update(extra)
    return d


class TestBuildCandidates:
    def test_single_hydra(self):
        candidates = build_candidates("BTCUSDT", hydra_intent=_make_intent(score=0.7))
        assert len(candidates) == 1
        assert candidates[0]["_selector_source"] == "hydra"
        assert candidates[0]["_selector_score"] == 0.7

    def test_single_legacy(self):
        candidates = build_candidates("BTCUSDT", legacy_intent=_make_intent(score=0.5, source="legacy"))
        assert len(candidates) == 1
        assert candidates[0]["_selector_source"] == "legacy"

    def test_both_engines_sorted_by_score(self):
        h = _make_intent(score=0.8, source="hydra")
        l = _make_intent(score=0.6, source="legacy")
        candidates = build_candidates("BTCUSDT", hydra_intent=h, legacy_intent=l)
        assert len(candidates) == 2
        assert candidates[0]["_selector_source"] == "hydra"
        assert candidates[1]["_selector_source"] == "legacy"

    def test_legacy_higher_score_sorts_first(self):
        h = _make_intent(score=0.3, source="hydra")
        l = _make_intent(score=0.9, source="legacy")
        candidates = build_candidates("BTCUSDT", hydra_intent=h, legacy_intent=l)
        assert candidates[0]["_selector_source"] == "legacy"
        assert candidates[1]["_selector_source"] == "hydra"

    def test_no_intents(self):
        candidates = build_candidates("BTCUSDT")
        assert candidates == []

    def test_none_intents_ignored(self):
        candidates = build_candidates("BTCUSDT", hydra_intent=None, legacy_intent=None)
        assert candidates == []

    def test_symbol_stamped(self):
        candidates = build_candidates("ETHUSDT", hydra_intent={"score": 0.5})
        assert candidates[0]["symbol"] == "ETHUSDT"

    def test_shallow_copy_does_not_mutate_original(self):
        orig = _make_intent(score=0.5)
        build_candidates("BTCUSDT", hydra_intent=orig)
        assert "_selector_score" not in orig


class TestSelectExecutableCandidate:
    def test_single_candidate_no_band_gate(self):
        candidates = [_make_intent(score=0.7, conviction_band="high")]
        candidates[0]["_selector_source"] = "hydra"
        candidates[0]["_selector_score"] = 0.7
        result = select_executable_candidate(candidates)
        assert result["selected"] is not None
        assert result["winner_engine"] == "hydra"
        assert result["loser_engine"] is None
        assert result["selection_reason"] == "highest_score"

    def test_hydra_wins_higher_score(self):
        h = _make_intent(score=0.8, source="hydra", conviction_band="high")
        h["_selector_source"] = "hydra"
        h["_selector_score"] = 0.8
        l = _make_intent(score=0.5, source="legacy", conviction_band="medium")
        l["_selector_source"] = "legacy"
        l["_selector_score"] = 0.5
        result = select_executable_candidate([h, l], min_conviction_band="medium")
        assert result["selected"] is h
        assert result["winner_engine"] == "hydra"
        assert result["loser_engine"] == "legacy"
        assert result["selection_reason"] == "band_pass"

    def test_legacy_wins_when_hydra_fails_band(self):
        h = _make_intent(score=0.8, source="hydra", conviction_band="low")
        h["_selector_source"] = "hydra"
        h["_selector_score"] = 0.8
        l = _make_intent(score=0.5, source="legacy", conviction_band="high")
        l["_selector_source"] = "legacy"
        l["_selector_score"] = 0.5
        result = select_executable_candidate([h, l], min_conviction_band="medium")
        assert result["selected"] is l
        assert result["winner_engine"] == "legacy"
        assert result["loser_engine"] == "hydra"
        assert result["selection_reason"] == "band_pass"

    def test_both_fail_conviction(self):
        h = _make_intent(score=0.8, source="hydra", conviction_band="very_low")
        h["_selector_source"] = "hydra"
        h["_selector_score"] = 0.8
        l = _make_intent(score=0.5, source="legacy", conviction_band="low")
        l["_selector_source"] = "legacy"
        l["_selector_score"] = 0.5
        result = select_executable_candidate([h, l], min_conviction_band="medium")
        assert result["selected"] is None
        assert result["winner_engine"] == "none"
        assert result["selection_reason"] == "all_rejected_by_band_gate"

    def test_no_candidates(self):
        result = select_executable_candidate([])
        assert result["selected"] is None
        assert result["winner_engine"] == "none"
        assert result["loser_engine"] is None
        assert result["selection_reason"] == "no_candidates"

    def test_band_gate_disabled_when_empty_string(self):
        h = _make_intent(score=0.5, source="hydra")
        h["_selector_source"] = "hydra"
        h["_selector_score"] = 0.5
        # No conviction_band field at all
        result = select_executable_candidate([h], min_conviction_band="")
        assert result["selected"] is h
        assert result["selection_reason"] == "highest_score"

    def test_candidates_list_preserved_in_result(self):
        h = _make_intent(score=0.8, source="hydra", conviction_band="high")
        h["_selector_source"] = "hydra"
        h["_selector_score"] = 0.8
        l = _make_intent(score=0.5, source="legacy", conviction_band="medium")
        l["_selector_source"] = "legacy"
        l["_selector_score"] = 0.5
        result = select_executable_candidate([h, l], min_conviction_band="low")
        assert len(result["candidates"]) == 2


class TestSelectorEndToEnd:
    """Integration-style tests using build_candidates + select."""

    def test_full_pipeline_hydra_wins(self):
        h = _make_intent(score=0.8, source="hydra", conviction_band="high")
        l = _make_intent(score=0.4, source="legacy", conviction_band="medium")
        candidates = build_candidates("BTCUSDT", hydra_intent=h, legacy_intent=l)
        result = select_executable_candidate(candidates, min_conviction_band="medium")
        assert result["selected"]["_selector_source"] == "hydra"

    def test_full_pipeline_fallback_to_legacy(self):
        h = _make_intent(score=0.9, source="hydra", conviction_band="very_low")
        l = _make_intent(score=0.3, source="legacy", conviction_band="high")
        candidates = build_candidates("BTCUSDT", hydra_intent=h, legacy_intent=l)
        result = select_executable_candidate(candidates, min_conviction_band="medium")
        assert result["selected"]["_selector_source"] == "legacy"

    def test_single_engine_passes(self):
        h = _make_intent(score=0.6, source="hydra", conviction_band="medium")
        candidates = build_candidates("BTCUSDT", hydra_intent=h)
        result = select_executable_candidate(candidates, min_conviction_band="medium")
        assert result["selected"] is not None
        assert result["winner_engine"] == "hydra"
        assert result["loser_engine"] is None

    def test_single_engine_fails_band(self):
        h = _make_intent(score=0.6, source="hydra", conviction_band="low")
        candidates = build_candidates("BTCUSDT", hydra_intent=h)
        result = select_executable_candidate(candidates, min_conviction_band="high")
        assert result["selected"] is None
