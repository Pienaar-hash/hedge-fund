"""Unit tests for fallback swap telemetry accumulator."""

import time

import pytest

from execution.fallback_telemetry import FallbackTelemetry


def _intent(score=0.5, source="legacy", fallback_used=False,
            merge_conflict=False, merge_primary_score=None,
            regime=None, merge_legacy_score=None, merge_hydra_score=None):
    d = {"score": score, "source": source, "symbol": "BTCUSDT", "signal": "BUY"}
    if fallback_used:
        d["fallback_used"] = True
    if merge_conflict:
        d["merge_conflict"] = True
    if merge_primary_score is not None:
        d["merge_primary_score"] = merge_primary_score
    if merge_legacy_score is not None:
        d["merge_legacy_score"] = merge_legacy_score
    if merge_hydra_score is not None:
        d["merge_hydra_score"] = merge_hydra_score
    if regime:
        d["metadata"] = {"entry_regime": regime}
    return d


@pytest.mark.unit
class TestFallbackTelemetry:

    def test_empty_snapshot(self):
        t = FallbackTelemetry()
        snap = t.snapshot()
        assert snap["normal_count"] == 0
        assert snap["fallback_count"] == 0
        assert snap["fallback_rate"] == 0.0
        assert snap["fallback_edge_delta"] == 0.0
        assert snap["primary_rejection_gap"] == 0.0

    def test_normal_attempts_only(self):
        t = FallbackTelemetry()
        t.record_attempt(_intent(score=0.6))
        t.record_attempt(_intent(score=0.8))
        snap = t.snapshot()
        assert snap["normal_count"] == 2
        assert snap["fallback_count"] == 0
        assert snap["fallback_edge_delta"] == pytest.approx(-0.7, abs=1e-4)  # 0 - avg_normal

    def test_fallback_edge_delta(self):
        t = FallbackTelemetry()
        # 2 normal at 0.7 avg
        t.record_attempt(_intent(score=0.7))
        t.record_attempt(_intent(score=0.7))
        # 1 fallback at 0.5
        t.record_attempt(_intent(
            score=0.5, fallback_used=True, merge_conflict=True,
            merge_primary_score=0.8,
        ))
        snap = t.snapshot()
        assert snap["normal_count"] == 2
        assert snap["fallback_count"] == 1
        assert snap["conflict_count"] == 1
        assert snap["fallback_rate"] == 1.0  # 1 fallback / 1 conflict
        assert snap["fallback_edge_delta"] == pytest.approx(0.5 - 0.7, abs=1e-4)
        assert snap["primary_rejection_gap"] == pytest.approx(0.8 - 0.5, abs=1e-4)

    def test_fallback_rate_fraction(self):
        t = FallbackTelemetry()
        # 3 conflicts, 1 with fallback
        t.record_attempt(_intent(score=0.6, merge_conflict=True))
        t.record_attempt(_intent(score=0.6, merge_conflict=True))
        t.record_attempt(_intent(
            score=0.4, fallback_used=True, merge_conflict=True,
            merge_primary_score=0.7,
        ))
        snap = t.snapshot()
        assert snap["conflict_count"] == 3
        assert snap["fallback_count"] == 1
        assert snap["fallback_rate"] == pytest.approx(1 / 3, abs=1e-4)

    def test_window_reset(self):
        t = FallbackTelemetry()
        t.record_attempt(_intent(score=0.5))
        assert t.normal_count == 1
        # Force window expiry
        t.last_reset_ts = time.time() - 100_000
        t.record_attempt(_intent(score=0.9))
        assert t.normal_count == 1  # reset cleared the first, then recorded new

    def test_persist_snapshot(self, tmp_path):
        t = FallbackTelemetry()
        t.record_attempt(_intent(score=0.6))
        dest = str(tmp_path / "fallback_metrics.json")
        t.persist_snapshot(dest)
        import json
        with open(dest) as fh:
            data = json.load(fh)
        assert data["normal_count"] == 1
        assert "ts" in data

    def test_hybrid_score_fallback(self):
        """Intent with hybrid_score instead of score is handled."""
        t = FallbackTelemetry()
        t.record_attempt({"hybrid_score": 0.8, "symbol": "BTCUSDT"})
        snap = t.snapshot()
        assert snap["normal_count"] == 1

    def test_none_score_safe(self):
        """None scores don't crash — treated as 0.0."""
        t = FallbackTelemetry()
        t.record_attempt({"symbol": "BTCUSDT"})
        snap = t.snapshot()
        assert snap["normal_count"] == 1

    def test_hydra_quality_diff(self):
        """HQD positive when hydra executes higher-scoring trades."""
        t = FallbackTelemetry()
        # 2 hydra executions at 0.7
        t.record_attempt(_intent(score=0.7, source="hydra"))
        t.record_attempt(_intent(score=0.7, source="hydra"))
        # 2 legacy executions at 0.5
        t.record_attempt(_intent(score=0.5, source="legacy"))
        t.record_attempt(_intent(score=0.5, source="legacy"))
        snap = t.snapshot()
        assert snap["hydra_quality_diff"] == pytest.approx(0.2, abs=1e-4)
        assert snap["hydra_exec_count"] == 2
        assert snap["legacy_exec_count"] == 2
        assert snap["hydra_participation"] == pytest.approx(0.5, abs=1e-4)

    def test_hydra_participation_skewed(self):
        """Participation reflects execution share, not merge share."""
        t = FallbackTelemetry()
        t.record_attempt(_intent(score=0.6, source="hydra"))
        t.record_attempt(_intent(score=0.5, source="legacy"))
        t.record_attempt(_intent(score=0.5, source="legacy"))
        t.record_attempt(_intent(score=0.5, source="legacy"))
        snap = t.snapshot()
        assert snap["hydra_participation"] == pytest.approx(0.25, abs=1e-4)

    def test_hydra_rescue_rate(self):
        """Hydra rescue rate = hydra fallback executions / conflicts."""
        t = FallbackTelemetry()
        # 3 conflicts total
        t.record_attempt(_intent(score=0.6, source="legacy", merge_conflict=True))
        t.record_attempt(_intent(score=0.5, source="legacy", merge_conflict=True))
        # 1 rescue: hydra executed via fallback
        t.record_attempt(_intent(
            score=0.4, source="hydra",
            fallback_used=True, merge_conflict=True,
            merge_primary_score=0.6,
        ))
        snap = t.snapshot()
        assert snap["hydra_rescue_rate"] == pytest.approx(1 / 3, abs=1e-4)
        assert snap["hydra_rescue_count" if "hydra_rescue_count" in snap else "hydra_exec_count"] > 0

    def test_hydra_overconfidence(self):
        """Overconfidence = hydra won merge but fallback fired / hydra primary count."""
        t = FallbackTelemetry()
        # hydra won merge, executed normally (no fallback)
        t.record_attempt(_intent(
            score=0.7, source="hydra", merge_conflict=True,
        ))
        # hydra won merge, but fallback fired (legacy rescued)
        t.record_attempt({
            "score": 0.5, "source": "legacy", "symbol": "BTCUSDT",
            "fallback_used": True, "merge_conflict": True,
            "merge_primary_engine": "hydra", "merge_primary_score": 0.8,
        })
        snap = t.snapshot()
        # 2 hydra primaries, 1 rejected → 50%
        assert snap["hydra_overconfidence"] == pytest.approx(0.5, abs=1e-4)

    def test_hydra_metrics_empty(self):
        """All hydra metrics zero when no attempts recorded."""
        t = FallbackTelemetry()
        snap = t.snapshot()
        assert snap["hydra_quality_diff"] == 0.0
        assert snap["hydra_rescue_rate"] == 0.0
        assert snap["hydra_overconfidence"] == 0.0

    # ---- Regime Score Differential tests ----

    def test_regime_rsd_basic(self):
        """RSD computed per-regime when both engines present."""
        t = FallbackTelemetry()
        t.record_attempt(_intent(score=0.8, source="hydra", regime="TREND_UP"))
        t.record_attempt(_intent(score=0.5, source="legacy", regime="TREND_UP"))
        snap = t.snapshot()
        entry = snap["regime_rsd"]["TREND_UP"]
        assert entry["rsd"] == pytest.approx(0.3, abs=1e-4)
        assert entry["hydra_n"] == 1
        assert entry["legacy_n"] == 1

    def test_regime_rsd_multiple_regimes(self):
        """Separate RSD for each regime."""
        t = FallbackTelemetry()
        t.record_attempt(_intent(score=0.9, source="hydra", regime="TREND_UP"))
        t.record_attempt(_intent(score=0.6, source="legacy", regime="TREND_UP"))
        t.record_attempt(_intent(score=0.4, source="hydra", regime="MEAN_REVERT"))
        t.record_attempt(_intent(score=0.7, source="legacy", regime="MEAN_REVERT"))
        snap = t.snapshot()
        assert snap["regime_rsd"]["TREND_UP"]["rsd"] == pytest.approx(0.3, abs=1e-4)
        assert snap["regime_rsd"]["MEAN_REVERT"]["rsd"] == pytest.approx(-0.3, abs=1e-4)
        assert snap["regime_rsd"]["MEAN_REVERT"]["hydra_n"] == 1
        assert snap["regime_rsd"]["MEAN_REVERT"]["legacy_n"] == 1

    def test_regime_rsd_single_engine_excluded(self):
        """Regimes with only one engine are excluded from RSD."""
        t = FallbackTelemetry()
        t.record_attempt(_intent(score=0.7, source="hydra", regime="TREND_UP"))
        t.record_attempt(_intent(score=0.5, source="hydra", regime="TREND_UP"))
        snap = t.snapshot()
        assert "TREND_UP" not in snap["regime_rsd"]

    def test_regime_rsd_empty(self):
        """No regime data → empty RSD dict."""
        t = FallbackTelemetry()
        snap = t.snapshot()
        assert snap["regime_rsd"] == {}

    def test_regime_rsd_reset(self):
        """Window reset clears regime scores."""
        t = FallbackTelemetry()
        t.record_attempt(_intent(score=0.8, source="hydra", regime="TREND_UP"))
        t.record_attempt(_intent(score=0.5, source="legacy", regime="TREND_UP"))
        assert t._regime_scores  # populated
        t.last_reset_ts = time.time() - 100_000
        t.record_attempt(_intent(score=0.3, source="legacy", regime="CHOPPY"))
        # Only CHOPPY should remain (TREND_UP cleared on reset), and
        # CHOPPY has only one engine so not in RSD
        snap = t.snapshot()
        assert "TREND_UP" not in snap["regime_rsd"]

    def test_regime_from_top_level(self):
        """Falls back to top-level entry_regime when no metadata."""
        t = FallbackTelemetry()
        intent = {"score": 0.6, "source": "hydra", "symbol": "X",
                  "entry_regime": "BREAKOUT"}
        t.record_attempt(intent)
        intent2 = {"score": 0.5, "source": "legacy", "symbol": "X",
                    "entry_regime": "BREAKOUT"}
        t.record_attempt(intent2)
        snap = t.snapshot()
        assert "BREAKOUT" in snap["regime_rsd"]
        assert snap["regime_rsd"]["BREAKOUT"]["hydra_n"] == 1

    # ---- Regime Dependence Drift (RDD) tests ----

    def test_rdd_from_two_regimes(self):
        """RDD = max(rsd) - min(rsd) across regimes."""
        t = FallbackTelemetry()
        # TREND_UP: hydra +0.8, legacy +0.5 → rsd=+0.3
        t.record_attempt(_intent(score=0.8, source="hydra", regime="TREND_UP"))
        t.record_attempt(_intent(score=0.5, source="legacy", regime="TREND_UP"))
        # MEAN_REVERT: hydra +0.3, legacy +0.6 → rsd=-0.3
        t.record_attempt(_intent(score=0.3, source="hydra", regime="MEAN_REVERT"))
        t.record_attempt(_intent(score=0.6, source="legacy", regime="MEAN_REVERT"))
        snap = t.snapshot()
        # +0.3 - (-0.3) = 0.6
        assert snap["regime_dependence_spread"] == pytest.approx(0.6, abs=1e-4)

    def test_rdd_single_regime_is_zero(self):
        """With only one regime, RDD is 0."""
        t = FallbackTelemetry()
        t.record_attempt(_intent(score=0.8, source="hydra", regime="TREND_UP"))
        t.record_attempt(_intent(score=0.5, source="legacy", regime="TREND_UP"))
        snap = t.snapshot()
        assert snap["regime_dependence_spread"] == 0.0

    def test_rdd_empty(self):
        """No regime data → RDD is 0."""
        t = FallbackTelemetry()
        snap = t.snapshot()
        assert snap["regime_dependence_spread"] == 0.0

    # ---- Conflict Edge Lift (CEL) tests ----

    def test_cel_positive_hydra_beats_legacy(self):
        """CEL positive when Hydra executes with higher score than legacy baseline."""
        t = FallbackTelemetry()
        # Hydra executed at 0.8, legacy baseline was 0.5
        t.record_attempt(_intent(
            score=0.8, source="hydra", merge_conflict=True,
            merge_legacy_score=0.5,
        ))
        snap = t.snapshot()
        assert snap["conflict_edge_lift"] == pytest.approx(0.3, abs=1e-4)
        assert snap["cel_count"] == 1

    def test_cel_zero_when_legacy_executes(self):
        """CEL contribution is 0 when legacy wins the conflict."""
        t = FallbackTelemetry()
        t.record_attempt(_intent(
            score=0.7, source="legacy", merge_conflict=True,
            merge_legacy_score=0.7,
        ))
        snap = t.snapshot()
        assert snap["conflict_edge_lift"] == pytest.approx(0.0, abs=1e-4)
        assert snap["cel_count"] == 1

    def test_cel_negative_hydra_worse(self):
        """CEL negative when Hydra takes slot from better legacy trade."""
        t = FallbackTelemetry()
        t.record_attempt(_intent(
            score=0.4, source="hydra", merge_conflict=True,
            merge_legacy_score=0.6,
        ))
        snap = t.snapshot()
        assert snap["conflict_edge_lift"] == pytest.approx(-0.2, abs=1e-4)

    def test_cel_averaged_over_conflicts(self):
        """CEL is average lift across all conflicted executions."""
        t = FallbackTelemetry()
        # Hydra +0.3
        t.record_attempt(_intent(
            score=0.8, source="hydra", merge_conflict=True,
            merge_legacy_score=0.5,
        ))
        # Legacy → 0 contribution
        t.record_attempt(_intent(
            score=0.6, source="legacy", merge_conflict=True,
            merge_legacy_score=0.6,
        ))
        # Hydra -0.1
        t.record_attempt(_intent(
            score=0.4, source="hydra", merge_conflict=True,
            merge_legacy_score=0.5,
        ))
        snap = t.snapshot()
        # (0.3 + 0.0 + -0.1) / 3 = 0.0667
        assert snap["conflict_edge_lift"] == pytest.approx(0.0667, abs=1e-3)
        assert snap["cel_count"] == 3

    def test_cel_empty(self):
        """No conflicts → CEL is 0."""
        t = FallbackTelemetry()
        snap = t.snapshot()
        assert snap["conflict_edge_lift"] == 0.0
        assert snap["cel_count"] == 0

    def test_cel_reset(self):
        """Window reset clears CEL counters."""
        t = FallbackTelemetry()
        t.record_attempt(_intent(
            score=0.8, source="hydra", merge_conflict=True,
            merge_legacy_score=0.5,
        ))
        assert t.cel_count == 1
        t.last_reset_ts = time.time() - 100_000
        t.record_attempt(_intent(score=0.5))  # triggers reset
        assert t.cel_count == 0

    # ---- Conflict Rate test ----

    def test_conflict_rate(self):
        """conflict_rate = conflicts / total executions."""
        t = FallbackTelemetry()
        # 2 conflicted hydra executions
        t.record_attempt(_intent(score=0.7, source="hydra", merge_conflict=True,
                                 merge_legacy_score=0.5))
        t.record_attempt(_intent(score=0.6, source="hydra", merge_conflict=True,
                                 merge_legacy_score=0.4))
        # 3 non-conflict legacy
        t.record_attempt(_intent(score=0.5, source="legacy"))
        t.record_attempt(_intent(score=0.5, source="legacy"))
        t.record_attempt(_intent(score=0.5, source="legacy"))
        snap = t.snapshot()
        # 2 conflicts / 5 total executions = 0.4
        assert snap["conflict_rate"] == pytest.approx(0.4, abs=1e-4)

    # ---- Score Distribution Divergence (SDD) tests ----

    def test_sdd_aligned_scales(self):
        """SDD near zero when both engines score similarly."""
        t = FallbackTelemetry()
        t.record_attempt(_intent(
            score=0.60, source="hydra", merge_conflict=True,
            merge_hydra_score=0.60, merge_legacy_score=0.58,
        ))
        t.record_attempt(_intent(
            score=0.55, source="legacy", merge_conflict=True,
            merge_hydra_score=0.53, merge_legacy_score=0.55,
        ))
        snap = t.snapshot()
        # hydra_mean=(0.60+0.53)/2=0.565, legacy_mean=(0.58+0.55)/2=0.565
        assert snap["score_scale_delta"] == pytest.approx(0.0, abs=1e-3)
        assert snap["sdd_count"] == 2

    def test_sdd_hydra_inflated(self):
        """SDD positive when Hydra scores are systematically higher."""
        t = FallbackTelemetry()
        t.record_attempt(_intent(
            score=0.75, source="hydra", merge_conflict=True,
            merge_hydra_score=0.75, merge_legacy_score=0.55,
        ))
        t.record_attempt(_intent(
            score=0.70, source="hydra", merge_conflict=True,
            merge_hydra_score=0.70, merge_legacy_score=0.50,
        ))
        snap = t.snapshot()
        # hydra_mean=0.725, legacy_mean=0.525, delta=+0.20
        assert snap["score_scale_delta"] == pytest.approx(0.20, abs=1e-3)
        assert snap["sdd_hydra_mean"] == pytest.approx(0.725, abs=1e-3)
        assert snap["sdd_legacy_mean"] == pytest.approx(0.525, abs=1e-3)

    def test_sdd_empty(self):
        """No conflicts → SDD is 0."""
        t = FallbackTelemetry()
        snap = t.snapshot()
        assert snap["score_scale_delta"] == 0.0
        assert snap["sdd_count"] == 0

    def test_sdd_reset(self):
        """Window reset clears SDD counters."""
        t = FallbackTelemetry()
        t.record_attempt(_intent(
            score=0.7, source="hydra", merge_conflict=True,
            merge_hydra_score=0.7, merge_legacy_score=0.5,
        ))
        assert t.sdd_count == 1
        t.last_reset_ts = time.time() - 100_000
        t.record_attempt(_intent(score=0.5))  # triggers reset
        assert t.sdd_count == 0

    # ---- MRI (Migration Readiness Index) ----

    def test_mri_all_conditions_met(self):
        """When all four conditions are satisfied, ecs_ready=True."""
        t = FallbackTelemetry()
        # 300+ conflicts with positive CEL and aligned SDD (hydra=0.51, legacy=0.50 → SDD=0.01)
        for _ in range(310):
            t.record_attempt(_intent(
                score=0.51, source="hydra", merge_conflict=True,
                merge_legacy_score=0.50, merge_hydra_score=0.51,
            ))
        snap = t.snapshot()
        assert snap["ecs_ready"] is True
        assert snap["ecs_ready_trades"] is True
        assert snap["ecs_stable_recovery"] is True
        assert snap["ecs_positive_edge"] is True
        assert snap["ecs_score_calibrated"] is True
        assert snap["ecs_readiness_score"] > 0.8

    def test_mri_insufficient_trades(self):
        """cel_count < 300 → ecs_ready=False, ready_trades=False."""
        t = FallbackTelemetry()
        for _ in range(50):
            t.record_attempt(_intent(
                score=0.51, source="hydra", merge_conflict=True,
                merge_legacy_score=0.50, merge_hydra_score=0.51,
            ))
        snap = t.snapshot()
        assert snap["ecs_ready"] is False
        assert snap["ecs_ready_trades"] is False
        # Other conditions should still be true
        assert snap["ecs_positive_edge"] is True

    def test_mri_high_fallback_rate(self):
        """fallback_rate >= 5% → ecs_ready=False."""
        t = FallbackTelemetry()
        # 300 conflicts, 20 fallback (rate = 20/300 = 6.7%)
        for _ in range(280):
            t.record_attempt(_intent(
                score=0.6, source="hydra", merge_conflict=True,
                merge_legacy_score=0.5, merge_hydra_score=0.6,
            ))
        for _ in range(20):
            t.record_attempt(_intent(
                score=0.55, source="hydra", merge_conflict=True,
                merge_legacy_score=0.5, merge_hydra_score=0.55,
                fallback_used=True, merge_primary_score=0.6,
            ))
        snap = t.snapshot()
        assert snap["ecs_ready"] is False
        assert snap["ecs_stable_recovery"] is False

    def test_mri_negative_cel(self):
        """CEL <= 0 → ecs_ready=False."""
        t = FallbackTelemetry()
        # Legacy wins all conflicts → CEL = 0
        for _ in range(310):
            t.record_attempt(_intent(
                score=0.5, source="legacy", merge_conflict=True,
                merge_legacy_score=0.5, merge_hydra_score=0.4,
            ))
        snap = t.snapshot()
        assert snap["ecs_ready"] is False
        assert snap["ecs_positive_edge"] is False

    def test_mri_score_drift(self):
        """|SDD| > 0.02 → ecs_ready=False."""
        t = FallbackTelemetry()
        # Large score gap → SDD drift
        for _ in range(310):
            t.record_attempt(_intent(
                score=0.8, source="hydra", merge_conflict=True,
                merge_legacy_score=0.5, merge_hydra_score=0.8,
            ))
        snap = t.snapshot()
        assert snap["ecs_ready"] is False
        assert snap["ecs_score_calibrated"] is False

    def test_mri_empty(self):
        """Empty telemetry → ecs_ready=False, score reflects only trivially-satisfied components."""
        t = FallbackTelemetry()
        snap = t.snapshot()
        assert snap["ecs_ready"] is False
        # fallback_rate=0 (0.3) + |sdd|=0 (0.1) = 0.4 — no trades or edge
        assert snap["ecs_readiness_score"] == pytest.approx(0.4, abs=0.01)

    def test_mri_score_partial(self):
        """Partial conditions give intermediate score."""
        t = FallbackTelemetry()
        # 150 conflicts = 50% of 300 trades → 0.5 * 0.4 weight = 0.20
        for _ in range(150):
            t.record_attempt(_intent(
                score=0.51, source="hydra", merge_conflict=True,
                merge_legacy_score=0.50, merge_hydra_score=0.51,
            ))
        snap = t.snapshot()
        assert snap["ecs_ready"] is False
        # Should have trade contribution (0.5*0.4=0.2) + fallback (0.3) + edge + calibration
        assert 0.3 < snap["ecs_readiness_score"] < 0.9

    # ---- MRI score history persistence ----

    def test_persist_builds_score_history(self, tmp_path):
        """persist_snapshot appends ecs_readiness_score to history array."""
        t = FallbackTelemetry()
        dest = str(tmp_path / "fb.json")

        t.persist_snapshot(dest)
        t.persist_snapshot(dest)
        t.persist_snapshot(dest)

        import json
        data = json.loads(open(dest).read())
        assert "ecs_score_history" in data
        assert len(data["ecs_score_history"]) == 3

    def test_persist_history_capped_at_50(self, tmp_path):
        """History array never exceeds 50 entries."""
        t = FallbackTelemetry()
        dest = str(tmp_path / "fb.json")

        # Seed with 48 existing entries
        import json
        seed = {"ecs_score_history": [0.1 * (i % 10) for i in range(48)]}
        with open(dest, "w") as fh:
            json.dump(seed, fh)

        t.persist_snapshot(dest)
        t.persist_snapshot(dest)
        t.persist_snapshot(dest)  # 48 + 3 = 51 → capped to 50

        data = json.loads(open(dest).read())
        assert len(data["ecs_score_history"]) == 50
