"""Tests for execution.shadow_selector — Phase 4 Commit 2."""
import json
import os
import time
from pathlib import Path
from unittest import mock

import pytest

from execution.shadow_selector import (
    _shadow_enabled,
    run_shadow_comparison,
)


def _make_intent(symbol="BTCUSDT", score=0.6, source="hydra", **extra):
    d = {"symbol": symbol, "hybrid_score": score, "source": source}
    d.update(extra)
    return d


class TestShadowEnabled:
    def test_disabled_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            # Remove ECS_SHADOW_ENABLED if present
            os.environ.pop("ECS_SHADOW_ENABLED", None)
            assert _shadow_enabled() is False

    def test_enabled_when_set(self):
        with mock.patch.dict(os.environ, {"ECS_SHADOW_ENABLED": "1"}):
            assert _shadow_enabled() is True

    def test_enabled_true_string(self):
        with mock.patch.dict(os.environ, {"ECS_SHADOW_ENABLED": "true"}):
            assert _shadow_enabled() is True

    def test_disabled_zero(self):
        with mock.patch.dict(os.environ, {"ECS_SHADOW_ENABLED": "0"}):
            assert _shadow_enabled() is False


class TestRunShadowComparison:
    def test_returns_none_when_disabled(self):
        with mock.patch.dict(os.environ, {"ECS_SHADOW_ENABLED": "0"}):
            result = run_shadow_comparison(
                symbol="BTCUSDT",
                raw_intent=_make_intent(),
                executor_winner_source="hydra",
                executor_used_fallback=False,
            )
            assert result is None

    def test_agreement_single_hydra(self, tmp_path):
        log_path = tmp_path / "ecs_shadow_events.jsonl"
        with mock.patch.dict(os.environ, {"ECS_SHADOW_ENABLED": "1"}):
            with mock.patch("execution.shadow_selector._LOG_PATH", log_path):
                result = run_shadow_comparison(
                    symbol="BTCUSDT",
                    raw_intent=_make_intent(score=0.7, source="hydra", conviction_band="high"),
                    executor_winner_source="hydra",
                    executor_used_fallback=False,
                    cycle=42,
                )
        assert result is not None
        assert result["agreement"] is True
        assert result["selector_winner"] == "hydra"
        assert result["executor_winner"] == "hydra"
        assert result["cycle"] == 42
        assert result["schema"] == "ecs_shadow_v1"
        # Check JSONL was written
        assert log_path.exists()
        events = [json.loads(l) for l in log_path.read_text().splitlines()]
        assert len(events) == 1
        assert events[0]["agreement"] is True

    def test_agreement_with_fallback_both_engines(self, tmp_path):
        """Executor swapped to legacy; selector should also pick legacy."""
        log_path = tmp_path / "ecs_shadow_events.jsonl"
        raw = _make_intent(
            score=0.8, source="hydra", conviction_band="low",
            _fallback=_make_intent(score=0.4, source="legacy", conviction_band="high"),
        )
        with mock.patch.dict(os.environ, {"ECS_SHADOW_ENABLED": "1"}):
            with mock.patch("execution.shadow_selector._LOG_PATH", log_path):
                result = run_shadow_comparison(
                    symbol="BTCUSDT",
                    raw_intent=raw,
                    executor_winner_source="legacy",
                    executor_used_fallback=True,
                    min_conviction_band="medium",
                )
        assert result is not None
        assert result["agreement"] is True
        assert result["selector_winner"] == "legacy"
        assert result["candidates_count"] == 2

    def test_divergence_detected(self, tmp_path):
        """Force a divergence: executor picked hydra, but selector picks legacy."""
        log_path = tmp_path / "ecs_shadow_events.jsonl"
        # Hydra has high score but low conviction; legacy has lower score but high conviction
        raw = _make_intent(
            score=0.9, source="hydra", conviction_band="very_low",
            _fallback=_make_intent(score=0.3, source="legacy", conviction_band="high"),
        )
        with mock.patch.dict(os.environ, {"ECS_SHADOW_ENABLED": "1"}):
            with mock.patch("execution.shadow_selector._LOG_PATH", log_path):
                result = run_shadow_comparison(
                    symbol="BTCUSDT",
                    raw_intent=raw,
                    executor_winner_source="hydra",  # executor didn't swap
                    executor_used_fallback=False,
                    min_conviction_band="medium",
                )
        assert result is not None
        assert result["agreement"] is False
        assert result["selector_winner"] == "legacy"
        assert result["executor_winner"] == "hydra"

    def test_no_mutation_of_original(self):
        """Shadow must not mutate the live intent objects."""
        raw = _make_intent(score=0.7, source="hydra", conviction_band="high")
        fallback = _make_intent(score=0.4, source="legacy", conviction_band="medium")
        raw["_fallback"] = fallback
        original_keys = set(raw.keys())
        original_fallback_keys = set(fallback.keys())
        with mock.patch.dict(os.environ, {"ECS_SHADOW_ENABLED": "1"}):
            with mock.patch("execution.shadow_selector._LOG_PATH", Path("/dev/null")):
                run_shadow_comparison(
                    symbol="BTCUSDT",
                    raw_intent=raw,
                    executor_winner_source="hydra",
                    executor_used_fallback=False,
                )
        # Original intent should not gain selector metadata
        assert "_selector_score" not in raw
        assert "_selector_source" not in raw
        assert "_selector_score" not in fallback
        # Keys should be unchanged
        assert set(raw.keys()) == original_keys
        assert set(fallback.keys()) == original_fallback_keys

    def test_no_band_gate_all_agree(self, tmp_path):
        """Without band gate, highest score wins — should agree with executor."""
        log_path = tmp_path / "ecs_shadow_events.jsonl"
        raw = _make_intent(score=0.9, source="hydra")
        with mock.patch.dict(os.environ, {"ECS_SHADOW_ENABLED": "1"}):
            with mock.patch("execution.shadow_selector._LOG_PATH", log_path):
                result = run_shadow_comparison(
                    symbol="BTCUSDT",
                    raw_intent=raw,
                    executor_winner_source="hydra",
                    executor_used_fallback=False,
                    min_conviction_band="",
                )
        assert result["agreement"] is True
        assert result["min_conviction_band"] == "none"

    def test_legacy_primary_hydra_fallback(self, tmp_path):
        """When legacy is primary (higher score) with hydra as fallback."""
        log_path = tmp_path / "ecs_shadow_events.jsonl"
        raw = _make_intent(
            score=0.9, source="legacy", conviction_band="high",
            _fallback=_make_intent(score=0.5, source="hydra", conviction_band="medium"),
        )
        with mock.patch.dict(os.environ, {"ECS_SHADOW_ENABLED": "1"}):
            with mock.patch("execution.shadow_selector._LOG_PATH", log_path):
                result = run_shadow_comparison(
                    symbol="BTCUSDT",
                    raw_intent=raw,
                    executor_winner_source="legacy",
                    executor_used_fallback=False,
                    min_conviction_band="medium",
                )
        assert result["agreement"] is True
        assert result["selector_winner"] == "legacy"

    def test_exception_returns_none(self):
        """Internal errors must not propagate — fail-open."""
        with mock.patch.dict(os.environ, {"ECS_SHADOW_ENABLED": "1"}):
            with mock.patch(
                "execution.candidate_selector.build_candidates",
                side_effect=RuntimeError("boom"),
            ):
                result = run_shadow_comparison(
                    symbol="BTCUSDT",
                    raw_intent=_make_intent(),
                    executor_winner_source="hydra",
                    executor_used_fallback=False,
                )
        assert result is None

    def test_multiple_cycles_append(self, tmp_path):
        """Multiple calls should append, not overwrite."""
        log_path = tmp_path / "ecs_shadow_events.jsonl"
        with mock.patch.dict(os.environ, {"ECS_SHADOW_ENABLED": "1"}):
            with mock.patch("execution.shadow_selector._LOG_PATH", log_path):
                for cycle in range(3):
                    run_shadow_comparison(
                        symbol="BTCUSDT",
                        raw_intent=_make_intent(source="hydra", conviction_band="high"),
                        executor_winner_source="hydra",
                        executor_used_fallback=False,
                        cycle=cycle,
                    )
        events = [json.loads(l) for l in log_path.read_text().splitlines()]
        assert len(events) == 3
        assert [e["cycle"] for e in events] == [0, 1, 2]
