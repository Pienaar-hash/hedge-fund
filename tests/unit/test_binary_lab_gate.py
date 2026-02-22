"""
Tests for Binary Lab decision-sheet extensions.

Covers:
  - Within-symbol dispersion (compute_within_symbol_stats)
  - Conviction band distribution (compute_conviction_distribution)
  - Persistence sub-window analysis (compute_persistence)
  - Decision-sheet JSON structure (build_decision_sheet_json)
  - Gate evaluator (evaluate_binary_lab_gate.evaluate)
  - End-to-end pipeline: synthetic audit → gate evaluator
"""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from hybrid_variance_audit import (  # noqa: E402
    COMPONENT_NAMES,
    CONVICTION_MEDIUM_PLUS_FLOOR,
    CONVICTION_NON_UNSCORED_FLOOR,
    DEFAULT_WEIGHTS,
    DISPERSION_FLOOR_STD,
    PERSISTENCE_MIN_PER_WINDOW,
    PERSISTENCE_MIN_WINDOWS,
    WEIGHTS_MIN_RECORDS,
    WITHIN_SYMBOL_MIN_PASSING,
    WITHIN_SYMBOL_STD_FLOOR,
    _percentile_band,
    build_decision_sheet_json,
    check_conditions,
    compute_conviction_distribution,
    compute_decomposition,
    compute_persistence,
    compute_within_symbol_stats,
    extract_vectors,
    generate_synthetic_records,
    run_audit,
)
from evaluate_binary_lab_gate import GATES, evaluate  # noqa: E402


# ── Helpers ─────────────────────────────────────────────────────────────

def _synthetic(n: int = 500, seed: int = 42) -> List[Dict[str, Any]]:
    return generate_synthetic_records(n=n, seed=seed)


def _degenerate_records(n: int = 200) -> List[Dict[str, Any]]:
    """T/C/E fixed at 0.5, only router varies → router-dominant, degenerate."""
    import random
    rng = random.Random(99)
    records: List[Dict[str, Any]] = []
    weights = dict(DEFAULT_WEIGHTS)
    for i in range(n):
        comps = {
            "trend": 0.5,
            "carry": 0.5,
            "expectancy": 0.5,
            "router": max(0, min(1, rng.gauss(0.6, 0.15))),
        }
        weighted = {k: weights[k] * comps[k] for k in COMPONENT_NAMES}
        base = sum(weighted.values())
        records.append({
            "ts": f"2026-02-19T{i // 60:02d}:{i % 60:02d}:00+00:00",
            "symbol": rng.choice(["BTCUSDT", "ETHUSDT"]),
            "direction": "LONG",
            "hybrid_score": round(base, 6),
            "components": comps,
            "weighted": weighted,
            "weights_used": dict(weights),
            "conviction_band": "",
            "rq_score": 0.8,
            "rv_score": 0.0,
        })
    return records


# ═══════════════════════════════════════════════════════════════════════
#  Within-Symbol Dispersion Tests
# ═══════════════════════════════════════════════════════════════════════

class TestWithinSymbolStats:
    """Tests for compute_within_symbol_stats."""

    def test_diverse_symbols_pass(self) -> None:
        recs = _synthetic(500)
        result = compute_within_symbol_stats(recs)
        assert result["pass"] is True
        assert result["symbols_above_floor_count"] >= WITHIN_SYMBOL_MIN_PASSING

    def test_single_symbol_fails_minimum(self) -> None:
        """One symbol can have high dispersion but need ≥2."""
        recs = _synthetic(500)
        # Override all symbols to one
        for r in recs:
            r["symbol"] = "BTCUSDT"
        result = compute_within_symbol_stats(recs)
        # 1 symbol above floor < min_passing (2)
        assert result["symbols_above_floor_count"] == 1
        assert result["pass"] is False

    def test_flat_scores_fail_floor(self) -> None:
        """All hybrid scores identical → zero within-symbol std."""
        recs = _synthetic(100)
        for r in recs:
            r["hybrid_score"] = 0.5
        result = compute_within_symbol_stats(recs)
        assert result["symbols_above_floor_count"] == 0
        assert result["pass"] is False

    def test_empty_records(self) -> None:
        result = compute_within_symbol_stats([])
        assert result["pass"] is False
        assert result["symbols_above_floor_count"] == 0

    def test_threshold_constant(self) -> None:
        assert WITHIN_SYMBOL_STD_FLOOR == 0.0030
        assert WITHIN_SYMBOL_MIN_PASSING == 2


# ═══════════════════════════════════════════════════════════════════════
#  Conviction Distribution Tests (Percentile Banding)
# ═══════════════════════════════════════════════════════════════════════

class TestConvictionDistribution:
    """Tests for compute_conviction_distribution with percentile banding."""

    def test_synthetic_conviction_pass(self) -> None:
        """Synthetic data with varied scores should pass percentile banding."""
        recs = _synthetic(500)
        result = compute_conviction_distribution(recs)
        assert result["pass"] is True
        assert result["non_unscored_pct"] == 1.0  # percentile banding is always scored
        assert result["medium_plus_pct"] >= CONVICTION_MEDIUM_PLUS_FLOOR
        assert result["band_spread_ok"] is True
        assert result["method"] == "percentile"

    def test_percentile_distribution_bands(self) -> None:
        """Percentile banding produces 4 bands: very_low, low, medium, high."""
        recs = _synthetic(500)
        result = compute_conviction_distribution(recs)
        dist = result["distribution"]
        # With sufficient score variance, expect all 4 bands
        for band in ["very_low", "low", "medium", "high"]:
            assert band in dist, f"Missing band: {band}"
        # Approximate percentile mass checks (20/40/30/10 split)
        assert 0.15 <= dist.get("very_low", 0) <= 0.25
        assert 0.35 <= dist.get("low", 0) <= 0.45
        assert 0.25 <= dist.get("medium", 0) <= 0.35
        assert 0.05 <= dist.get("high", 0) <= 0.15

    def test_percentiles_reported(self) -> None:
        """Result includes computed percentile thresholds."""
        recs = _synthetic(500)
        result = compute_conviction_distribution(recs)
        assert "percentiles" in result
        p = result["percentiles"]
        assert "p20" in p and "p60" in p and "p90" in p
        assert p["p20"] < p["p60"] < p["p90"]

    def test_constant_scores_degrades_gracefully(self) -> None:
        """All identical hybrid_scores → all same band, spread fails."""
        recs = _synthetic(200)
        for r in recs:
            r["hybrid_score"] = 0.45
        result = compute_conviction_distribution(recs)
        # All records at same score → P20=P60=P90=0.45 → all "high" (>= p90)
        assert result["band_spread_ok"] is False
        assert result["pass"] is False

    def test_two_score_clusters_fails_spread(self) -> None:
        """Only 2 distinct score levels → at most 2 bands."""
        recs = _synthetic(200)
        for i, r in enumerate(recs):
            r["hybrid_score"] = 0.40 if i % 2 == 0 else 0.50
        result = compute_conviction_distribution(recs)
        non_empty = [b for b, v in result["distribution"].items() if v > 0]
        assert len(non_empty) <= 2
        assert result["band_spread_ok"] is False
        assert result["pass"] is False

    def test_empty_records(self) -> None:
        result = compute_conviction_distribution([])
        assert result["pass"] is False

    def test_distribution_sums_to_one(self) -> None:
        recs = _synthetic(500)
        result = compute_conviction_distribution(recs)
        total = sum(result["distribution"].values())
        assert abs(total - 1.0) < 1e-9

    def test_thresholds_unchanged(self) -> None:
        assert CONVICTION_NON_UNSCORED_FLOOR == 0.30
        assert CONVICTION_MEDIUM_PLUS_FLOOR == 0.08

    def test_medium_plus_is_top_40(self) -> None:
        """With well-spread scores, medium + high ≈ 40% (P60-P100)."""
        recs = _synthetic(1000)
        result = compute_conviction_distribution(recs)
        # medium = 60-90th pct (30%), high = 90-100th pct (10%) → ~40%
        assert 0.35 <= result["medium_plus_pct"] <= 0.45


# ═══════════════════════════════════════════════════════════════════════
#  Persistence Tests
# ═══════════════════════════════════════════════════════════════════════

class TestPersistence:
    """Tests for compute_persistence."""

    def test_synthetic_persistence_pass(self) -> None:
        """Synthetic data should have all windows passing."""
        recs = _synthetic(500)
        weights = dict(DEFAULT_WEIGHTS)
        result = compute_persistence(recs, weights)
        assert result["pass"] is True
        assert result["windows_passed"] >= PERSISTENCE_MIN_WINDOWS
        assert result["total_windows"] == PERSISTENCE_MIN_WINDOWS

    def test_empty_records_fail(self) -> None:
        result = compute_persistence([], dict(DEFAULT_WEIGHTS))
        assert result["pass"] is False
        assert result["windows_passed"] == 0

    def test_degenerate_fails_persistence(self) -> None:
        """Degenerate records fail structural conditions → persistence fails."""
        recs = _degenerate_records(200)
        weights = dict(DEFAULT_WEIGHTS)
        result = compute_persistence(recs, weights)
        assert result["pass"] is False

    def test_window_results_count(self) -> None:
        recs = _synthetic(500)
        weights = dict(DEFAULT_WEIGHTS)
        result = compute_persistence(recs, weights, n_windows=5)
        assert len(result["window_results"]) == 5
        assert result["total_windows"] == 5

    def test_persistence_threshold_constant(self) -> None:
        assert PERSISTENCE_MIN_WINDOWS == 3


# ═══════════════════════════════════════════════════════════════════════
#  Decision Sheet JSON Structure Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDecisionSheetJSON:
    """Tests for build_decision_sheet_json structure."""

    @pytest.fixture()
    def sheet(self) -> Dict[str, Any]:
        recs = _synthetic(500)
        vecs = extract_vectors(recs)
        weights = dict(DEFAULT_WEIGHTS)
        decomp = compute_decomposition(vecs, weights)
        conditions = check_conditions(decomp)
        return build_decision_sheet_json(recs, decomp, conditions, weights, "test-24h")

    def test_top_level_keys(self, sheet: Dict[str, Any]) -> None:
        required = {
            "version", "phase", "window", "reconstruction_error_pct",
            "flags", "variance", "contribution", "conditions",
            "stddev", "dispersion_floor", "within_symbol",
            "conviction", "persistence", "data_sufficiency",
        }
        assert required.issubset(set(sheet.keys()))

    def test_window_metadata(self, sheet: Dict[str, Any]) -> None:
        w = sheet["window"]
        assert w["label"] == "test-24h"
        assert w["record_count"] == 500
        assert w["start_ts"] != ""
        assert w["end_ts"] != ""

    def test_variance_keys(self, sheet: Dict[str, Any]) -> None:
        v = sheet["variance"]
        for name in COMPONENT_NAMES:
            assert name in v
        assert "hybrid" in v
        assert all(isinstance(v[k], float) for k in v)

    def test_contribution_keys(self, sheet: Dict[str, Any]) -> None:
        c = sheet["contribution"]
        for name in COMPONENT_NAMES:
            assert name in c
        assert "signal_share" in c
        # Signal share = T + C + E
        expected = c["trend"] + c["carry"] + c["expectancy"]
        assert abs(c["signal_share"] - expected) < 1e-9

    def test_conditions_all_9_evaluable(self, sheet: Dict[str, Any]) -> None:
        """Every gate referenced in the evaluator resolves from this JSON."""
        for gate_name, _ in GATES:
            # This is what the evaluator does — verify it won't crash
            from evaluate_binary_lab_gate import _resolve
            result = _resolve(sheet, gate_name)
            assert isinstance(result, bool), f"Gate {gate_name} did not resolve to bool"

    def test_stddev_within_keys(self, sheet: Dict[str, Any]) -> None:
        s = sheet["stddev"]
        assert "hybrid" in s
        assert "within" in s
        # within is a dict of symbol → float
        assert isinstance(s["within"], dict)

    def test_flags_keys(self, sheet: Dict[str, Any]) -> None:
        f = sheet["flags"]
        assert "degenerate" in f
        assert "zero_variance_components" in f
        assert "weight_consistency" in f
        assert "weights_data_sufficient" in f

    def test_reconstruction_error_percentage(self, sheet: Dict[str, Any]) -> None:
        # Synthetic data → reconstruction error ≈ 0.0000%
        assert sheet["reconstruction_error_pct"] < 0.01


# ═══════════════════════════════════════════════════════════════════════
#  Gate Evaluator Tests
# ═══════════════════════════════════════════════════════════════════════

class TestGateEvaluator:
    """Tests for evaluate_binary_lab_gate.evaluate."""

    def test_all_gates_pass_on_synthetic(self) -> None:
        result = run_audit(
            _synthetic(500), window_label="test", output_json=False, verbose=False,
        )
        sheet = result["decision_sheet"]
        verdict = evaluate(sheet)
        assert verdict["binary_lab_activation"] == "GO"
        assert verdict["gates_passed"] == 9
        assert verdict["failed_conditions"] == []

    def test_degenerate_data_no_go(self) -> None:
        """Degenerate records → at least condition_a and condition_b fail."""
        result = run_audit(
            _degenerate_records(200), window_label="degen", output_json=False, verbose=False,
        )
        sheet = result["decision_sheet"]
        verdict = evaluate(sheet)
        assert verdict["binary_lab_activation"] == "NO_GO"
        assert "condition_a_router" in verdict["failed_conditions"]

    def test_evaluate_returns_correct_count(self) -> None:
        result = run_audit(
            _synthetic(500), window_label="test", output_json=False, verbose=False,
        )
        sheet = result["decision_sheet"]
        verdict = evaluate(sheet)
        assert verdict["gates_evaluated"] == len(GATES)
        assert verdict["gates_evaluated"] == 9

    def test_single_gate_knockout(self) -> None:
        """Knock out just dispersion_floor → NO_GO with exactly 1 failure."""
        result = run_audit(
            _synthetic(500), window_label="test", output_json=False, verbose=False,
        )
        sheet = copy.deepcopy(result["decision_sheet"])
        sheet["dispersion_floor"]["pass"] = False
        verdict = evaluate(sheet)
        assert verdict["binary_lab_activation"] == "NO_GO"
        assert verdict["failed_conditions"] == ["dispersion_floor"]
        assert verdict["gates_passed"] == 8

    def test_empty_sheet_no_go(self) -> None:
        """Empty dict → all gates fail."""
        verdict = evaluate({})
        assert verdict["binary_lab_activation"] == "NO_GO"
        assert verdict["gates_passed"] == 0

    def test_gate_names_match_sheet(self) -> None:
        """All gate names in GATES list have resolution paths in the decision sheet."""
        result = run_audit(
            _synthetic(500), window_label="test", output_json=False, verbose=False,
        )
        sheet = result["decision_sheet"]
        from evaluate_binary_lab_gate import _resolve
        for gate_name, _ in GATES:
            val = _resolve(sheet, gate_name)
            # Synthetic should have all True
            assert val is True, f"Gate {gate_name} unexpectedly False on synthetic data"


# ═══════════════════════════════════════════════════════════════════════
#  run_audit integration — decision_sheet key present
# ═══════════════════════════════════════════════════════════════════════

class TestRunAuditDecisionSheet:
    """Ensure run_audit now emits decision_sheet key in results."""

    def test_decision_sheet_key_present(self) -> None:
        result = run_audit(
            _synthetic(500), window_label="test", output_json=False, verbose=False,
        )
        assert "decision_sheet" in result
        assert result["decision_sheet"]["version"] == "v1.0"

    def test_insufficient_data_no_sheet(self) -> None:
        """Below MIN_RECORDS → no decision_sheet (early exit)."""
        result = run_audit(
            _synthetic(5), window_label="tiny", output_json=False, verbose=False,
        )
        assert "decision_sheet" not in result
        assert result["sufficient_data"] is False

    def test_synthetic_conviction_bands_present(self) -> None:
        """generate_synthetic_records now produces varied conviction bands."""
        recs = _synthetic(500)
        bands = set(r.get("conviction_band", "") for r in recs)
        # Should have at least 4 distinct bands (including possibly "")
        assert len(bands) >= 4


# ═══════════════════════════════════════════════════════════════════════
#  CI Contract: Synthetic Always Passes 9/9
# ═══════════════════════════════════════════════════════════════════════

class TestSyntheticContract:
    """
    CI invariant: synthetic dataset MUST always pass all 9 gates.

    If this test fails it means either:
    - Gate thresholds were tightened beyond achievability, or
    - Synthetic generator is producing degenerate data, or
    - A code change broke the structural condition logic.
    """

    @pytest.mark.parametrize("n", [200, 500, 1000])
    def test_synthetic_always_go(self, n: int) -> None:
        """Synthetic data at varying sizes must produce GO."""
        recs = generate_synthetic_records(n=n, seed=42)
        result = run_audit(recs, window_label=f"synthetic-{n}", output_json=False, verbose=False)
        assert result["sufficient_data"] is True
        sheet = result["decision_sheet"]
        verdict = evaluate(sheet)
        assert verdict["binary_lab_activation"] == "GO", (
            f"Synthetic ({n} records) expected GO but got NO_GO. "
            f"Failed: {verdict['failed_conditions']}"
        )
        assert verdict["gates_passed"] == 9

    def test_synthetic_reconstruction_below_alert(self) -> None:
        """Reconstruction error must stay below 2% alert threshold."""
        from hybrid_variance_audit import RECONSTRUCTION_ALERT_THRESHOLD  # noqa: E402
        recs = _synthetic(500)
        result = run_audit(recs, window_label="test", output_json=False, verbose=False)
        sheet = result["decision_sheet"]
        assert sheet["reconstruction_error_pct"] < RECONSTRUCTION_ALERT_THRESHOLD * 100

    def test_synthetic_seed_independence(self) -> None:
        """Different seeds still produce GO — not seed-specific."""
        for seed in [1, 7, 42, 99, 314]:
            recs = generate_synthetic_records(n=500, seed=seed)
            result = run_audit(recs, window_label=f"seed-{seed}", output_json=False, verbose=False)
            sheet = result["decision_sheet"]
            verdict = evaluate(sheet)
            assert verdict["binary_lab_activation"] == "GO", (
                f"Seed {seed} failed: {verdict['failed_conditions']}"
            )


# ═══════════════════════════════════════════════════════════════════════
#  Data Sufficiency Flags (Task B)
# ═══════════════════════════════════════════════════════════════════════

class TestDataSufficiency:
    """Tests for epistemic data-sufficiency annotations."""

    def test_data_sufficiency_section_present(self) -> None:
        result = run_audit(_synthetic(500), window_label="test", output_json=False, verbose=False)
        sheet = result["decision_sheet"]
        assert "data_sufficiency" in sheet
        ds = sheet["data_sufficiency"]
        assert "weights_sufficient" in ds
        assert "persistence_sufficient" in ds
        assert "weights_n_records" in ds
        assert "weights_min_required" in ds
        assert "persistence_min_per_window" in ds

    def test_weights_insufficient_below_threshold(self) -> None:
        """500 records < WEIGHTS_MIN_RECORDS (1000) → weights_sufficient=False."""
        result = run_audit(_synthetic(500), window_label="test", output_json=False, verbose=False)
        sheet = result["decision_sheet"]
        assert sheet["data_sufficiency"]["weights_sufficient"] is False
        assert sheet["data_sufficiency"]["weights_n_records"] == 500

    def test_weights_sufficient_above_threshold(self) -> None:
        """1500 records > WEIGHTS_MIN_RECORDS → weights_sufficient=True."""
        recs = generate_synthetic_records(n=1500, seed=42)
        result = run_audit(recs, window_label="test", output_json=False, verbose=False)
        sheet = result["decision_sheet"]
        assert sheet["data_sufficiency"]["weights_sufficient"] is True

    def test_flags_weights_data_sufficient(self) -> None:
        """flags.weights_data_sufficient reflects the same threshold."""
        result = run_audit(_synthetic(500), window_label="test", output_json=False, verbose=False)
        sheet = result["decision_sheet"]
        assert "weights_data_sufficient" in sheet["flags"]
        assert sheet["flags"]["weights_data_sufficient"] == sheet["data_sufficiency"]["weights_sufficient"]

    def test_persistence_data_sufficient_in_result(self) -> None:
        """persistence.data_sufficient reflects sub-window record counts."""
        recs = _synthetic(500)
        weights = dict(DEFAULT_WEIGHTS)
        result = compute_persistence(recs, weights)
        assert "data_sufficient" in result
        # 500 / 3 ≈ 166 per window; >= 200 required → False
        assert result["data_sufficient"] is False

    def test_persistence_data_sufficient_with_enough(self) -> None:
        """Large dataset → data_sufficient=True."""
        recs = generate_synthetic_records(n=1500, seed=42)
        weights = dict(DEFAULT_WEIGHTS)
        result = compute_persistence(recs, weights)
        # 1500 / 3 = 500 per window; >= 200 → True
        assert result["data_sufficient"] is True

    def test_evaluator_insufficient_data_field(self) -> None:
        """Evaluator surfaces insufficient_data when condition_c fails with small n."""
        result = run_audit(_degenerate_records(200), window_label="degen", output_json=False, verbose=False)
        sheet = result["decision_sheet"]
        # Force weights_sufficient to False (degenerate 200 records)
        assert sheet["data_sufficiency"]["weights_sufficient"] is False
        verdict = evaluate(sheet)
        assert "insufficient_data" in verdict

    def test_evaluator_no_insufficient_data_on_go(self) -> None:
        """When all pass, insufficient_data is empty."""
        result = run_audit(_synthetic(500), window_label="test", output_json=False, verbose=False)
        sheet = result["decision_sheet"]
        verdict = evaluate(sheet)
        assert verdict["insufficient_data"] == []

    def test_constants_values(self) -> None:
        assert WEIGHTS_MIN_RECORDS == 1000
        assert PERSISTENCE_MIN_PER_WINDOW == 200


# ═══════════════════════════════════════════════════════════════════════
#  Shadow Conviction Band Tests (Task A)
# ═══════════════════════════════════════════════════════════════════════

class TestShadowConvictionBand:
    """
    Tests for the shadow conviction band logic.

    When conviction_band is empty, the score decomposition writer
    derives a band from hybrid_score using the standard thresholds.
    Test the threshold logic directly.
    """

    @staticmethod
    def _shadow_band(hybrid_score: float) -> str:
        """Replicate the shadow band logic from signal_screener."""
        if hybrid_score >= 0.92:
            return "very_high"
        elif hybrid_score >= 0.80:
            return "high"
        elif hybrid_score >= 0.60:
            return "medium"
        elif hybrid_score >= 0.40:
            return "low"
        else:
            return "very_low"

    @pytest.mark.parametrize("score,expected", [
        (0.95, "very_high"),
        (0.92, "very_high"),
        (0.85, "high"),
        (0.80, "high"),
        (0.70, "medium"),
        (0.60, "medium"),
        (0.50, "low"),
        (0.40, "low"),
        (0.30, "very_low"),
        (0.10, "very_low"),
        (0.0, "very_low"),
    ])
    def test_shadow_band_thresholds(self, score: float, expected: str) -> None:
        assert self._shadow_band(score) == expected

    def test_thresholds_match_conviction_engine(self) -> None:
        """Shadow thresholds are identical to conviction_engine defaults."""
        from execution.conviction_engine import ConvictionConfig
        cfg = ConvictionConfig()
        # Verify the threshold values match
        assert cfg.thresholds["very_high"] == 0.92
        assert cfg.thresholds["high"] == 0.80
        assert cfg.thresholds["medium"] == 0.60
        assert cfg.thresholds["low"] == 0.40


# ═══════════════════════════════════════════════════════════════════════
#  Percentile Band Function Tests
# ═══════════════════════════════════════════════════════════════════════

class TestPercentileBand:
    """Tests for _percentile_band used in audit conviction distribution."""

    @pytest.mark.parametrize("score,p20,p60,p90,expected", [
        (0.41, 0.42, 0.44, 0.47, "very_low"),  # below P20
        (0.42, 0.42, 0.44, 0.47, "low"),        # at P20
        (0.43, 0.42, 0.44, 0.47, "low"),        # between P20 and P60
        (0.44, 0.42, 0.44, 0.47, "medium"),     # at P60
        (0.45, 0.42, 0.44, 0.47, "medium"),     # between P60 and P90
        (0.47, 0.42, 0.44, 0.47, "high"),       # at P90
        (0.50, 0.42, 0.44, 0.47, "high"),       # above P90
    ])
    def test_band_assignment(self, score: float, p20: float, p60: float, p90: float, expected: str) -> None:
        assert _percentile_band(score, p20, p60, p90) == expected

    def test_equal_thresholds_all_high(self) -> None:
        """When all percentiles collapse, everything is >= p90 → high."""
        assert _percentile_band(0.45, 0.45, 0.45, 0.45) == "high"

    def test_bands_are_exhaustive(self) -> None:
        """Every possible score maps to exactly one of 4 bands."""
        valid = {"very_low", "low", "medium", "high"}
        for s in [0.0, 0.1, 0.3, 0.42, 0.44, 0.47, 0.6, 0.9, 1.0]:
            b = _percentile_band(s, 0.42, 0.44, 0.47)
            assert b in valid
