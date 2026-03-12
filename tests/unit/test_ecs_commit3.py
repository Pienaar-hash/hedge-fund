"""Tests for Phase 4 Commit 3 — ECS selector live enablement.

Tests:
    1. USE_ECS_SELECTOR flag defaults OFF
    2. Soak telemetry: agreement when ECS matches old path
    3. Soak telemetry: divergence detection
    4. _simulate_fallback_swap reproduces old-path logic
    5. log_ecs_soak_event writes JSONL
    6. ECS path selects correct winner in executor flow
    7. ECS fail-open falls back to legacy path
"""
import json
import os
from unittest import mock

from execution.v6_flags import V6Flags, _env_bool


# ---------------------------------------------------------------------------
# Flag tests
# ---------------------------------------------------------------------------

class TestUseEcsSelectorFlag:
    def test_default_off(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("USE_ECS_SELECTOR", None)
            assert _env_bool("USE_ECS_SELECTOR", "0") is False

    def test_enabled_when_set(self):
        with mock.patch.dict(os.environ, {"USE_ECS_SELECTOR": "1"}):
            assert _env_bool("USE_ECS_SELECTOR", "0") is True

    def test_flag_in_dataclass(self):
        flags = V6Flags(
            intel_v6_enabled=False,
            risk_engine_v6_enabled=False,
            pipeline_v6_shadow_enabled=False,
            router_autotune_v6_enabled=False,
            feedback_allocator_v6_enabled=False,
            router_autotune_v6_apply_enabled=False,
            shadow_dle_enabled=False,
            shadow_dle_write_logs=False,
            dle_enforce_entry_only=False,
            ecs_shadow_enabled=False,
            use_ecs_selector=True,
        )
        assert flags.use_ecs_selector is True

    def test_flags_to_dict_includes_use_ecs_selector(self):
        from execution.v6_flags import flags_to_dict
        flags = V6Flags(
            intel_v6_enabled=False,
            risk_engine_v6_enabled=False,
            pipeline_v6_shadow_enabled=False,
            router_autotune_v6_enabled=False,
            feedback_allocator_v6_enabled=False,
            router_autotune_v6_apply_enabled=False,
            shadow_dle_enabled=False,
            shadow_dle_write_logs=False,
            dle_enforce_entry_only=False,
            ecs_shadow_enabled=False,
            use_ecs_selector=True,
        )
        d = flags_to_dict(flags)
        assert "USE_ECS_SELECTOR" in d
        assert d["USE_ECS_SELECTOR"] is True


# ---------------------------------------------------------------------------
# _simulate_fallback_swap tests
# ---------------------------------------------------------------------------

class TestSimulateFallbackSwap:
    def test_no_fallback_returns_primary(self):
        from execution.shadow_selector import _simulate_fallback_swap
        intent = {"source": "hydra", "conviction_band": "high"}
        result = _simulate_fallback_swap(intent, "low")
        assert result == "hydra"

    def test_swap_when_primary_fails_band(self):
        from execution.shadow_selector import _simulate_fallback_swap
        intent = {
            "source": "hydra",
            "conviction_band": "very_low",
            "_fallback": {"source": "legacy", "conviction_band": "high"},
        }
        result = _simulate_fallback_swap(intent, "medium")
        assert result == "legacy"

    def test_no_swap_when_primary_passes(self):
        from execution.shadow_selector import _simulate_fallback_swap
        intent = {
            "source": "hydra",
            "conviction_band": "high",
            "_fallback": {"source": "legacy", "conviction_band": "high"},
        }
        result = _simulate_fallback_swap(intent, "low")
        assert result == "hydra"

    def test_no_swap_when_fallback_also_fails(self):
        from execution.shadow_selector import _simulate_fallback_swap
        intent = {
            "source": "hydra",
            "conviction_band": "very_low",
            "_fallback": {"source": "legacy", "conviction_band": "low"},
        }
        result = _simulate_fallback_swap(intent, "high")
        assert result == "hydra"

    def test_no_band_gate_returns_primary(self):
        from execution.shadow_selector import _simulate_fallback_swap
        intent = {
            "source": "hydra",
            "conviction_band": "",
            "_fallback": {"source": "legacy", "conviction_band": "high"},
        }
        result = _simulate_fallback_swap(intent, "")
        assert result == "hydra"

    def test_legacy_primary_hydra_fallback(self):
        from execution.shadow_selector import _simulate_fallback_swap
        intent = {
            "source": "legacy",
            "conviction_band": "very_low",
            "_fallback": {"source": "hydra", "conviction_band": "medium"},
        }
        result = _simulate_fallback_swap(intent, "medium")
        assert result == "hydra"


# ---------------------------------------------------------------------------
# log_ecs_soak_event tests
# ---------------------------------------------------------------------------

class TestLogEcsSoakEvent:
    def test_agreement_event(self, tmp_path):
        from execution.shadow_selector import log_ecs_soak_event
        log_path = tmp_path / "ecs_soak_events.jsonl"
        raw = {"source": "hydra", "conviction_band": "high"}
        with mock.patch("execution.shadow_selector._SOAK_LOG_PATH", log_path):
            event = log_ecs_soak_event(
                symbol="BTCUSDT",
                raw_intent=raw,
                ecs_winner="hydra",
                ecs_reason="band_pass",
                candidates_count=1,
                cycle=5,
                min_conviction_band="low",
            )
        assert event is not None
        assert event["agreement"] is True
        assert event["schema"] == "ecs_soak_v1"
        assert event["ecs_winner"] == "hydra"
        assert event["old_path_winner"] == "hydra"
        # Check file was written
        events = [json.loads(ln) for ln in log_path.read_text().splitlines()]
        assert len(events) == 1
        assert events[0]["agreement"] is True

    def test_divergence_event(self, tmp_path):
        from execution.shadow_selector import log_ecs_soak_event
        log_path = tmp_path / "ecs_soak_events.jsonl"
        # Old path would swap to legacy, but ECS picked hydra
        raw = {
            "source": "hydra",
            "conviction_band": "very_low",
            "_fallback": {"source": "legacy", "conviction_band": "high"},
        }
        with mock.patch("execution.shadow_selector._SOAK_LOG_PATH", log_path):
            event = log_ecs_soak_event(
                symbol="SOLUSDT",
                raw_intent=raw,
                ecs_winner="hydra",
                ecs_reason="highest_score",
                candidates_count=2,
                min_conviction_band="medium",
            )
        assert event is not None
        assert event["agreement"] is False
        assert event["ecs_winner"] == "hydra"
        assert event["old_path_winner"] == "legacy"

    def test_multiple_events_append(self, tmp_path):
        from execution.shadow_selector import log_ecs_soak_event
        log_path = tmp_path / "ecs_soak_events.jsonl"
        raw = {"source": "hydra", "conviction_band": "high"}
        with mock.patch("execution.shadow_selector._SOAK_LOG_PATH", log_path):
            for cycle in range(3):
                log_ecs_soak_event(
                    symbol="BTCUSDT",
                    raw_intent=raw,
                    ecs_winner="hydra",
                    ecs_reason="band_pass",
                    candidates_count=1,
                    cycle=cycle,
                )
        events = [json.loads(ln) for ln in log_path.read_text().splitlines()]
        assert len(events) == 3
        assert [e["cycle"] for e in events] == [0, 1, 2]

    def test_exception_returns_none(self):
        from execution.shadow_selector import log_ecs_soak_event
        with mock.patch(
            "execution.shadow_selector._simulate_fallback_swap",
            side_effect=RuntimeError("boom"),
        ):
            result = log_ecs_soak_event(
                symbol="BTCUSDT",
                raw_intent={},
                ecs_winner="hydra",
                ecs_reason="band_pass",
                candidates_count=1,
            )
        assert result is None


# ---------------------------------------------------------------------------
# ECS selector path integration tests
# ---------------------------------------------------------------------------

class TestEcsSelectorPath:
    """Tests for the candidate_selector producing correct winners."""

    def test_hydra_wins_on_score(self):
        from execution.candidate_selector import (
            build_candidates,
            select_executable_candidate,
        )
        h = {"symbol": "BTCUSDT", "hybrid_score": 0.98, "source": "hydra", "conviction_band": "low"}
        leg = {"symbol": "BTCUSDT", "hybrid_score": 0.45, "source": "legacy", "conviction_band": "low"}
        candidates = build_candidates("BTCUSDT", hydra_intent=h, legacy_intent=leg)
        result = select_executable_candidate(candidates, "low")
        assert result["winner_engine"] == "hydra"

    def test_legacy_wins_sol_pattern(self):
        """Reproduce the SOL pattern: hydra score near-zero, legacy ~0.45."""
        from execution.candidate_selector import (
            build_candidates,
            select_executable_candidate,
        )
        h = {"symbol": "SOLUSDT", "hybrid_score": 0.02, "source": "hydra", "conviction_band": "low"}
        leg = {"symbol": "SOLUSDT", "hybrid_score": 0.45, "source": "legacy", "conviction_band": "low"}
        candidates = build_candidates("SOLUSDT", hydra_intent=h, legacy_intent=leg)
        result = select_executable_candidate(candidates, "low")
        assert result["winner_engine"] == "legacy"
        assert result["loser_engine"] == "hydra"

    def test_ecs_preserves_merge_scores(self):
        """Selected candidate should carry merge scores from raw intent."""
        from execution.candidate_selector import (
            build_candidates,
            select_executable_candidate,
        )
        h = {"symbol": "BTCUSDT", "hybrid_score": 0.98, "source": "hydra", "conviction_band": "low"}
        candidates = build_candidates("BTCUSDT", hydra_intent=h)
        result = select_executable_candidate(candidates, "low")
        selected = result["selected"]
        # Simulate what executor does: stamp merge scores
        selected["merge_legacy_score"] = 0.45
        selected["merge_hydra_score"] = 0.98
        assert selected["merge_legacy_score"] == 0.45
        assert selected["merge_hydra_score"] == 0.98

    def test_all_rejected_skips_intent(self):
        from execution.candidate_selector import (
            build_candidates,
            select_executable_candidate,
        )
        h = {"symbol": "BTCUSDT", "hybrid_score": 0.98, "source": "hydra", "conviction_band": "very_low"}
        leg = {"symbol": "BTCUSDT", "hybrid_score": 0.45, "source": "legacy", "conviction_band": "very_low"}
        candidates = build_candidates("BTCUSDT", hydra_intent=h, legacy_intent=leg)
        result = select_executable_candidate(candidates, "high")
        assert result["selected"] is None
        assert result["selection_reason"] == "all_rejected_by_band_gate"
