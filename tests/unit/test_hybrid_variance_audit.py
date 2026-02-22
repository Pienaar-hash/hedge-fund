"""
Tests for ``scripts/hybrid_variance_audit.py``.

Validates variance decomposition math, condition flags, edge cases,
and calibration continuity — all on deterministic synthetic data.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from hybrid_variance_audit import (  # noqa: E402
    COMPONENT_NAMES,
    DEFAULT_WEIGHTS,
    MIN_RECORDS,
    RECONSTRUCTION_TOLERANCE,
    ROUTER_DOMINANCE_CEILING,
    SIGNAL_SUFFICIENCY_FLOOR,
    WEIGHT_CONSISTENCY_TOLERANCE,
    check_conditions,
    compute_covariance_matrix,
    compute_decomposition,
    extract_vectors,
    generate_synthetic_records,
    parse_window,
    run_audit,
)


# ── Helpers ─────────────────────────────────────────────────────────────

def _make_records(
    components_list: List[Dict[str, float]],
    weights: Dict[str, float] | None = None,
) -> List[Dict[str, Any]]:
    """Build minimal JSONL-shaped records from a list of component dicts."""
    if weights is None:
        weights = dict(DEFAULT_WEIGHTS)
    records: List[Dict[str, Any]] = []
    for comps in components_list:
        weighted = {k: weights[k] * comps[k] for k in COMPONENT_NAMES}
        base = sum(weighted.values())
        records.append({
            "ts": "2026-02-19T00:00:00+00:00",
            "symbol": "BTCUSDT",
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


def _uniform_records(n: int = 100) -> List[Dict[str, Any]]:
    """All components identical across records → zero variance."""
    comps = {"trend": 0.5, "carry": 0.5, "expectancy": 0.5, "router": 0.5}
    return _make_records([comps] * n)


def _router_dominant_records(n: int = 200) -> List[Dict[str, Any]]:
    """T/C/E fixed, only router varies → router = 100%."""
    import random
    rng = random.Random(99)
    rows: List[Dict[str, float]] = []
    for _ in range(n):
        rows.append({
            "trend": 0.5,
            "carry": 0.5,
            "expectancy": 0.5,
            "router": max(0.0, min(1.0, rng.gauss(0.6, 0.15))),
        })
    return _make_records(rows)


def _balanced_records(n: int = 500) -> List[Dict[str, Any]]:
    """All components vary (independent) → balanced contributions."""
    import random
    rng = random.Random(42)
    rows: List[Dict[str, float]] = []
    dists = {
        "trend": (0.55, 0.18),
        "carry": (0.48, 0.14),
        "expectancy": (0.52, 0.12),
        "router": (0.60, 0.10),
    }
    for _ in range(n):
        row: Dict[str, float] = {}
        for name in COMPONENT_NAMES:
            mu, sigma = dists[name]
            row[name] = max(0.0, min(1.0, rng.gauss(mu, sigma)))
        rows.append(row)
    return _make_records(rows)


# ── Test: parse_window ──────────────────────────────────────────────────

class TestParseWindow:
    def test_hours(self) -> None:
        from datetime import timedelta
        assert parse_window("24h") == timedelta(hours=24)

    def test_days(self) -> None:
        from datetime import timedelta
        assert parse_window("7d") == timedelta(days=7)

    def test_minutes(self) -> None:
        from datetime import timedelta
        assert parse_window("30m") == timedelta(minutes=30)

    def test_invalid(self) -> None:
        with pytest.raises(ValueError):
            parse_window("foo")


# ── Test: Reconstruction accuracy ───────────────────────────────────────

class TestReconstruction:
    """Var(H)_theoretical must match Var(H)_empirical for pure linear sums."""

    def test_synthetic_reconstruction_exact(self) -> None:
        """On synthetic data without RV/RQ/decay, reconstruction must be ~0%."""
        records = generate_synthetic_records(n=1000, seed=7)
        vecs = extract_vectors(records)
        decomp = compute_decomposition(vecs, DEFAULT_WEIGHTS)
        assert decomp["reconstruction_error"] < 0.001  # <0.1%

    def test_balanced_reconstruction(self) -> None:
        records = _balanced_records(500)
        vecs = extract_vectors(records)
        decomp = compute_decomposition(vecs, DEFAULT_WEIGHTS)
        assert decomp["reconstruction_error"] < RECONSTRUCTION_TOLERANCE

    def test_router_dominant_reconstruction(self) -> None:
        records = _router_dominant_records(300)
        vecs = extract_vectors(records)
        decomp = compute_decomposition(vecs, DEFAULT_WEIGHTS)
        assert decomp["reconstruction_error"] < RECONSTRUCTION_TOLERANCE

    def test_contributions_sum_to_one(self) -> None:
        """Contribution fractions must sum to 1.0 when variance > 0."""
        records = _balanced_records(500)
        vecs = extract_vectors(records)
        decomp = compute_decomposition(vecs, DEFAULT_WEIGHTS)
        total = sum(decomp["contribution"].values())
        assert abs(total - 1.0) < 1e-10


# ── Test: Condition A — Router Dominance ─────────────────────────────────

class TestConditionA:
    def test_router_dominant_fails(self) -> None:
        records = _router_dominant_records(200)
        vecs = extract_vectors(records)
        decomp = compute_decomposition(vecs, DEFAULT_WEIGHTS)
        conds = check_conditions(decomp)
        assert conds["condition_a"]["pass"] is False
        assert conds["condition_a"]["router_contribution"] > ROUTER_DOMINANCE_CEILING

    def test_balanced_passes(self) -> None:
        records = _balanced_records(500)
        vecs = extract_vectors(records)
        decomp = compute_decomposition(vecs, DEFAULT_WEIGHTS)
        conds = check_conditions(decomp)
        assert conds["condition_a"]["pass"] is True


# ── Test: Condition B — Signal Sufficiency ───────────────────────────────

class TestConditionB:
    def test_router_only_fails(self) -> None:
        records = _router_dominant_records(200)
        vecs = extract_vectors(records)
        decomp = compute_decomposition(vecs, DEFAULT_WEIGHTS)
        conds = check_conditions(decomp)
        assert conds["condition_b"]["pass"] is False

    def test_balanced_passes(self) -> None:
        records = _balanced_records(500)
        vecs = extract_vectors(records)
        decomp = compute_decomposition(vecs, DEFAULT_WEIGHTS)
        conds = check_conditions(decomp)
        assert conds["condition_b"]["pass"] is True
        assert conds["condition_b"]["signal_contribution"] >= SIGNAL_SUFFICIENCY_FLOOR


# ── Test: Condition C — Weight Consistency ───────────────────────────────

class TestConditionC:
    def test_balanced_consistent(self) -> None:
        """Balanced independent components should be weight-consistent."""
        records = _balanced_records(500)
        vecs = extract_vectors(records)
        decomp = compute_decomposition(vecs, DEFAULT_WEIGHTS)
        conds = check_conditions(decomp)
        assert conds["condition_c"]["pass"] is True
        assert conds["condition_c"]["max_deviation"] <= WEIGHT_CONSISTENCY_TOLERANCE

    def test_extreme_skew_detected(self) -> None:
        """One component with huge variance, others tiny → still weight-consistent
        since expected share accounts for component variance."""
        import random
        rng = random.Random(55)
        rows: List[Dict[str, float]] = []
        for _ in range(300):
            rows.append({
                "trend": max(0.0, min(1.0, rng.gauss(0.5, 0.40))),  # huge
                "carry": max(0.0, min(1.0, rng.gauss(0.5, 0.01))),  # tiny
                "expectancy": max(0.0, min(1.0, rng.gauss(0.5, 0.01))),
                "router": max(0.0, min(1.0, rng.gauss(0.5, 0.01))),
            })
        records = _make_records(rows)
        vecs = extract_vectors(records)
        decomp = compute_decomposition(vecs, DEFAULT_WEIGHTS)
        # Expected share is dominated by trend (high variance × high weight)
        # Actual contribution should also be dominated by trend
        # So weight consistency should pass since actuals track expected
        conds = check_conditions(decomp)
        assert conds["condition_c"]["pass"] is True


# ── Test: Degeneracy detection ──────────────────────────────────────────

class TestDegeneracy:
    def test_all_constant_is_degenerate(self) -> None:
        records = _uniform_records(50)
        vecs = extract_vectors(records)
        decomp = compute_decomposition(vecs, DEFAULT_WEIGHTS)
        conds = check_conditions(decomp)
        assert conds["degeneracy_warning"]["degenerate"] is True
        assert len(conds["degeneracy_warning"]["zero_variance_components"]) == 4

    def test_only_router_varies_is_degenerate(self) -> None:
        records = _router_dominant_records(100)
        vecs = extract_vectors(records)
        decomp = compute_decomposition(vecs, DEFAULT_WEIGHTS)
        conds = check_conditions(decomp)
        assert conds["degeneracy_warning"]["degenerate"] is True
        assert "trend" in conds["degeneracy_warning"]["zero_variance_components"]

    def test_balanced_not_degenerate(self) -> None:
        records = _balanced_records(200)
        vecs = extract_vectors(records)
        decomp = compute_decomposition(vecs, DEFAULT_WEIGHTS)
        conds = check_conditions(decomp)
        assert conds["degeneracy_warning"]["degenerate"] is False


# ── Test: run_audit integration ─────────────────────────────────────────

class TestRunAudit:
    def test_insufficient_data(self) -> None:
        records = _balanced_records(5)  # below MIN_RECORDS
        result = run_audit(records, window_label="test", verbose=False)
        assert result["sufficient_data"] is False
        assert "error" in result

    def test_synthetic_all_clear(self) -> None:
        records = generate_synthetic_records(n=500, seed=42)
        result = run_audit(records, window_label="synthetic", verbose=False)
        assert result["sufficient_data"] is True
        overall = result["conditions"]["overall"]
        assert overall["all_clear"] is True
        assert overall["structural_pass"] is True
        assert overall["reconstruction_pass"] is True
        assert overall["degenerate"] is False

    def test_router_dominant_not_all_clear(self) -> None:
        records = _router_dominant_records(200)
        result = run_audit(records, window_label="test", verbose=False)
        overall = result["conditions"]["overall"]
        assert overall["all_clear"] is False
        assert overall["structural_pass"] is False


# ── Test: Covariance matrix symmetry ────────────────────────────────────

class TestCovarianceMatrix:
    def test_symmetric(self) -> None:
        records = _balanced_records(200)
        vecs = extract_vectors(records)
        cov_mat = compute_covariance_matrix(vecs)
        for a in COMPONENT_NAMES:
            for b in COMPONENT_NAMES:
                assert abs(cov_mat[(a, b)] - cov_mat[(b, a)]) < 1e-15

    def test_diagonal_is_variance(self) -> None:
        records = _balanced_records(200)
        vecs = extract_vectors(records)
        cov_mat = compute_covariance_matrix(vecs)
        decomp = compute_decomposition(vecs, DEFAULT_WEIGHTS)
        for name in COMPONENT_NAMES:
            assert abs(cov_mat[(name, name)] - decomp["component_var"][name]) < 1e-15


# ── Test: Known analytic case ───────────────────────────────────────────

class TestAnalyticCase:
    """
    Construct data where we know the exact answer.

    Two records: trend = [0, 1], everything else = 0.5 (constant).
    Var(trend) = 0.25, all others = 0, no covariance.
    Var(H) = (0.4)^2 * 0.25 = 0.04.
    Contribution: trend = 100%.
    """

    def test_two_point_decomposition(self) -> None:
        rows = [
            {"trend": 0.0, "carry": 0.5, "expectancy": 0.5, "router": 0.5},
            {"trend": 1.0, "carry": 0.5, "expectancy": 0.5, "router": 0.5},
        ]
        # Need ≥ MIN_RECORDS, so replicate
        rows_full = rows * (MIN_RECORDS // 2 + 1)
        records = _make_records(rows_full)
        vecs = extract_vectors(records)
        decomp = compute_decomposition(vecs, DEFAULT_WEIGHTS)

        # Var(trend) = 0.25
        assert abs(decomp["component_var"]["trend"] - 0.25) < 1e-10
        # Others zero
        for name in ("carry", "expectancy", "router"):
            assert abs(decomp["component_var"][name]) < 1e-10
        # Var(H) = 0.4^2 * 0.25 = 0.04
        assert abs(decomp["var_theoretical"] - 0.04) < 1e-10
        # Trend = 100%
        assert abs(decomp["contribution"]["trend"] - 1.0) < 1e-10


# ── Test: Calibration boundary simulation ───────────────────────────────

class TestCalibrationBoundary:
    """
    Simulate pre/post calibration by adjusting sensitivity multipliers.

    Calibration scales factor deviations from 0.5 and sharpens/flattens weights.
    Verify that Var(H) doesn't jump >25% when calibration activates.
    """

    def _apply_mock_calibration(
        self,
        records: List[Dict[str, Any]],
        sensitivity: float = 1.2,
        temperature: float = 0.9,
    ) -> List[Dict[str, Any]]:
        """Simulate calibration: scale deviations, sharpen weights."""
        calibrated: List[Dict[str, Any]] = []
        for rec in records:
            comps = dict(rec["components"])
            weights = dict(rec["weights_used"])

            # Scale deviations from 0.5
            for name in ("trend", "carry", "expectancy"):
                raw = comps[name]
                comps[name] = max(0.0, min(1.0, 0.5 + (raw - 0.5) * sensitivity))

            # Sharpen weights (raise to 1/temp, renormalize)
            powered = {k: v ** (1.0 / temperature) for k, v in weights.items()}
            total = sum(powered.values())
            weights = {k: v / total for k, v in powered.items()}

            weighted = {k: weights[k] * comps[k] for k in COMPONENT_NAMES}
            base = sum(weighted.values())

            new_rec = dict(rec)
            new_rec["components"] = comps
            new_rec["weighted"] = weighted
            new_rec["weights_used"] = weights
            new_rec["hybrid_score"] = round(base, 6)
            calibrated.append(new_rec)

        return calibrated

    def test_mild_calibration_within_tolerance(self) -> None:
        """Mild sensitivity/temperature should produce <25% variance change."""
        base_records = _balanced_records(300)
        cal_records = self._apply_mock_calibration(base_records, sensitivity=1.08, temperature=0.95)

        base_vecs = extract_vectors(base_records)
        cal_vecs = extract_vectors(cal_records)

        base_decomp = compute_decomposition(base_vecs, DEFAULT_WEIGHTS)
        cal_decomp = compute_decomposition(cal_vecs, DEFAULT_WEIGHTS)

        var_pre = base_decomp["var_empirical_base"]
        var_post = cal_decomp["var_empirical_base"]

        if var_pre > 1e-12:
            change = abs(var_post - var_pre) / var_pre
            assert change < 0.25, f"Calibration jump {change:.1%} exceeds 25%"

    def test_extreme_calibration_detected(self) -> None:
        """Extreme sensitivity creates a >25% variance jump."""
        base_records = _balanced_records(300)
        cal_records = self._apply_mock_calibration(base_records, sensitivity=2.5, temperature=0.5)

        base_vecs = extract_vectors(base_records)
        cal_vecs = extract_vectors(cal_records)

        base_decomp = compute_decomposition(base_vecs, DEFAULT_WEIGHTS)
        cal_decomp = compute_decomposition(cal_vecs, DEFAULT_WEIGHTS)

        var_pre = base_decomp["var_empirical_base"]
        var_post = cal_decomp["var_empirical_base"]

        if var_pre > 1e-12:
            change = abs(var_post - var_pre) / var_pre
            assert change > 0.25, f"Expected jump but got only {change:.1%}"
