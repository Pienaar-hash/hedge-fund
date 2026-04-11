"""Tests for factor_ablation_experiment.py — Factor Ablation Grid."""
import json
import math
import os
import tempfile

import pytest

from scripts.factor_ablation_experiment import (
    ABLATION_VARIANTS,
    CARRY_FIX_REFERENCE,
    DEFAULT_WEIGHTS,
    INTERACTION_THRESHOLD,
    MASK_MIDPOINT,
    RMSE_STRONG,
    RMSE_USABLE,
    VARIANT_LABELS,
    ablate_score,
    compute_ablation_grid,
    estimate_factor_values,
    generate_report,
    load_decomposition_means,
    reconstruct_score,
    reconstruction_quality,
    section_carry_crosscheck,
    section_correlation_grid,
    section_displacement,
    section_economic,
    section_interaction,
    section_reconstruction,
    section_verdict,
)
from scripts.carry_fix_experiment import (
    REFERENCE_MASK,
    classify_region,
    compute_carry_score,
)

# ── Fixtures ────────────────────────────────────────────────────────────────

BTC_LO = REFERENCE_MASK["BTCUSDT"]["lo"]
BTC_HI = REFERENCE_MASK["BTCUSDT"]["hi"]


def _make_episode(
    *,
    hybrid_score: float = 0.48,
    net_pnl: float = 1.0,
    side: str = "LONG",
    episode_id: str = "EP_T001",
    entry_ts: str = "2026-03-20T10:00:00+00:00",
) -> dict:
    return {
        "episode_id": episode_id,
        "symbol": "BTCUSDT",
        "side": side,
        "entry_ts": entry_ts,
        "exit_ts": "2026-03-20T12:00:00+00:00",
        "hybrid_score": hybrid_score,
        "net_pnl": net_pnl,
        "intent_id": "ord_test",
    }


def _make_decomp_line(
    symbol: str = "BTCUSDT",
    trend: float = 0.5,
    carry: float = 0.44,
    expectancy: float = 0.4817,
    router: float = 0.5,
) -> str:
    return json.dumps({
        "ts": "2026-02-24T10:13:12+00:00",
        "symbol": symbol,
        "hybrid_score": 0.43,
        "components": {
            "trend": trend,
            "carry": carry,
            "expectancy": expectancy,
            "router": router,
        },
        "weights_used": {"trend": 0.4, "carry": 0.25, "expectancy": 0.2, "router": 0.15},
    })


def _write_decomp_file(lines: list[str]) -> str:
    """Write decomp lines to temp file, return path."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        for line in lines:
            f.write(line + "\n")
    return path


def _make_grid_row(
    *,
    side: str = "LONG",
    net_pnl: float = 1.0,
    v0_score: float = 0.48,
    episode_id: str = "EP_T001",
    factors: dict | None = None,
) -> dict:
    """Build a grid row with all 7 variant scores set to v0_score by default."""
    f = factors or {"trend": 0.5, "carry": 0.44, "expectancy": 0.48, "router": 0.5}
    row: dict = {
        "episode_id": episode_id,
        "symbol": "BTCUSDT",
        "side": side,
        "net_pnl": net_pnl,
        "entry_ts": "2026-03-20T10:00:00+00:00",
        "factors": f,
    }
    for v in ABLATION_VARIANTS:
        row[f"{v['label']}_score"] = v0_score
        row[f"{v['label']}_region"] = classify_region(v0_score)
    return row


# ── TestLoadDecompositionMeans ──────────────────────────────────────────────

class TestLoadDecompositionMeans:
    def test_returns_means_from_jsonl(self, tmp_path):
        p = tmp_path / "decomp.jsonl"
        lines = [
            _make_decomp_line(trend=0.4, carry=0.3, expectancy=0.5, router=0.6),
            _make_decomp_line(trend=0.6, carry=0.5, expectancy=0.5, router=0.4),
        ]
        p.write_text("\n".join(lines) + "\n")
        result = load_decomposition_means(p)
        assert result["trend"] == pytest.approx(0.5, abs=1e-6)
        assert result["carry"] == pytest.approx(0.4, abs=1e-6)
        assert result["expectancy"] == pytest.approx(0.5, abs=1e-6)
        assert result["router"] == pytest.approx(0.5, abs=1e-6)

    def test_filters_by_symbol(self, tmp_path):
        p = tmp_path / "decomp.jsonl"
        lines = [
            _make_decomp_line(symbol="BTCUSDT", trend=0.6),
            _make_decomp_line(symbol="ETHUSDT", trend=0.2),
        ]
        p.write_text("\n".join(lines) + "\n")
        result = load_decomposition_means(p, symbol="BTCUSDT")
        assert result["trend"] == pytest.approx(0.6, abs=1e-6)

    def test_missing_file_returns_neutral(self, tmp_path):
        result = load_decomposition_means(tmp_path / "nope.jsonl")
        assert result == {"trend": 0.5, "carry": 0.5, "expectancy": 0.5, "router": 0.5}

    def test_empty_file_returns_neutral(self, tmp_path):
        p = tmp_path / "decomp.jsonl"
        p.write_text("")
        result = load_decomposition_means(p)
        assert result == {"trend": 0.5, "carry": 0.5, "expectancy": 0.5, "router": 0.5}

    def test_skips_malformed_lines(self, tmp_path):
        p = tmp_path / "decomp.jsonl"
        lines = [
            "not json",
            _make_decomp_line(trend=0.8, carry=0.6, expectancy=0.4, router=0.2),
        ]
        p.write_text("\n".join(lines) + "\n")
        result = load_decomposition_means(p)
        assert result["trend"] == pytest.approx(0.8, abs=1e-6)


# ── TestEstimateFactorValues ───────────────────────────────────────────────

class TestEstimateFactorValues:
    def test_carry_computed_from_inputs(self):
        decomp = {"trend": 0.5, "carry": 0.44, "expectancy": 0.48, "router": 0.5}
        result = estimate_factor_values("LONG", 0.0001, -0.0002, decomp)
        expected_carry = compute_carry_score(0.0001, -0.0002, "LONG")
        assert result["carry"] == pytest.approx(expected_carry, abs=1e-6)

    def test_trend_from_decomp_means(self):
        decomp = {"trend": 0.42, "carry": 0.44, "expectancy": 0.48, "router": 0.5}
        result = estimate_factor_values("LONG", 0.0, 0.0, decomp)
        assert result["trend"] == 0.42

    def test_expectancy_from_decomp_means(self):
        decomp = {"trend": 0.5, "carry": 0.44, "expectancy": 0.55, "router": 0.5}
        result = estimate_factor_values("SHORT", 0.0, 0.0, decomp)
        assert result["expectancy"] == 0.55

    def test_router_from_decomp_means(self):
        decomp = {"trend": 0.5, "carry": 0.44, "expectancy": 0.48, "router": 0.7}
        result = estimate_factor_values("LONG", 0.0, 0.0, decomp)
        assert result["router"] == 0.7

    def test_direction_affects_carry_only(self):
        decomp = {"trend": 0.5, "carry": 0.44, "expectancy": 0.48, "router": 0.5}
        long_r = estimate_factor_values("LONG", 0.0001, 0.001, decomp)
        short_r = estimate_factor_values("SHORT", 0.0001, 0.001, decomp)
        # Carry differs
        assert long_r["carry"] != short_r["carry"]
        # Others identical
        assert long_r["trend"] == short_r["trend"]
        assert long_r["expectancy"] == short_r["expectancy"]
        assert long_r["router"] == short_r["router"]


# ── TestReconstructScore ───────────────────────────────────────────────────

class TestReconstructScore:
    def test_weighted_sum(self):
        factors = {"trend": 0.5, "carry": 0.5, "expectancy": 0.5, "router": 0.5}
        assert reconstruct_score(factors) == pytest.approx(0.5, abs=1e-6)

    def test_with_custom_weights(self):
        factors = {"trend": 1.0, "carry": 0.0, "expectancy": 0.0, "router": 0.0}
        weights = {"trend": 1.0, "carry": 0.0, "expectancy": 0.0, "router": 0.0}
        assert reconstruct_score(factors, weights) == pytest.approx(1.0, abs=1e-6)

    def test_known_values(self):
        factors = {"trend": 0.5, "carry": 0.4364, "expectancy": 0.4817, "router": 0.5}
        expected = 0.4 * 0.5 + 0.25 * 0.4364 + 0.2 * 0.4817 + 0.15 * 0.5
        assert reconstruct_score(factors) == pytest.approx(expected, abs=1e-6)


# ── TestReconstructionQuality ──────────────────────────────────────────────

class TestReconstructionQuality:
    def test_perfect_reconstruction(self):
        # Build episodes where hybrid_score = exact Σ(w·f)
        decomp = {"trend": 0.5, "carry": 0.5, "expectancy": 0.5, "router": 0.5}
        ep = _make_episode(hybrid_score=0.5)
        result = reconstruction_quality([ep], 0.0, 0.0, decomp)
        assert result["rmse"] < 0.01
        assert result["confidence_tier"] == "STRONG"

    def test_high_rmse_gives_exploratory(self):
        decomp = {"trend": 0.9, "carry": 0.9, "expectancy": 0.9, "router": 0.9}
        ep = _make_episode(hybrid_score=0.1)
        result = reconstruction_quality([ep], 0.0, 0.0, decomp)
        assert result["rmse"] >= RMSE_USABLE
        assert result["confidence_tier"] == "EXPLORATORY"

    def test_usable_tier(self):
        # Produce moderate RMSE
        decomp = {"trend": 0.52, "carry": 0.5, "expectancy": 0.5, "router": 0.5}
        # Actual carry at funding=0 is 0.5, so estimated score ~0.504
        ep = _make_episode(hybrid_score=0.46)
        result = reconstruction_quality([ep], 0.0, 0.0, decomp)
        assert RMSE_STRONG <= result["rmse"] < RMSE_USABLE
        assert result["confidence_tier"] == "USABLE"

    def test_empty_gives_exploratory(self):
        result = reconstruction_quality([], 0.0, 0.0, {"trend": 0.5, "carry": 0.5, "expectancy": 0.5, "router": 0.5})
        assert result["n"] == 0
        assert result["confidence_tier"] == "EXPLORATORY"

    def test_non_btc_filtered(self):
        ep = _make_episode()
        ep["symbol"] = "ETHUSDT"
        decomp = {"trend": 0.5, "carry": 0.5, "expectancy": 0.5, "router": 0.5}
        result = reconstruction_quality([ep], 0.0, 0.0, decomp)
        assert result["n"] == 0


# ── TestAblateScore ────────────────────────────────────────────────────────

class TestAblateScore:
    def test_no_removal_returns_original(self):
        assert ablate_score(0.48, {}, {}, []) == 0.48

    def test_single_factor_removal(self):
        factors = {"carry": 0.44, "trend": 0.5, "expectancy": 0.48, "router": 0.5}
        weights = DEFAULT_WEIGHTS.copy()
        original = 0.48
        result = ablate_score(original, factors, weights, ["carry"])
        # Expected: (0.48 - 0.25*0.44) / (1 - 0.25) = (0.48 - 0.11) / 0.75
        expected = (0.48 - 0.25 * 0.44) / 0.75
        assert result == pytest.approx(expected, abs=1e-6)

    def test_pair_removal(self):
        factors = {"carry": 0.44, "trend": 0.5, "expectancy": 0.48, "router": 0.5}
        weights = DEFAULT_WEIGHTS.copy()
        original = 0.48
        result = ablate_score(original, factors, weights, ["carry", "trend"])
        w_removed = 0.25 + 0.40
        removed_contrib = 0.25 * 0.44 + 0.40 * 0.5
        expected = (0.48 - removed_contrib) / (1.0 - w_removed)
        assert result == pytest.approx(expected, abs=1e-6)

    def test_all_removed_returns_zero(self):
        factors = {"carry": 0.4, "trend": 0.5, "expectancy": 0.5, "router": 0.5}
        weights = DEFAULT_WEIGHTS.copy()
        result = ablate_score(0.48, factors, weights, list(weights.keys()))
        assert result == 0.0

    def test_renormalization_preserves_relative_factors(self):
        # If all factors are identical, ablation shouldn't change score
        factors = {"carry": 0.5, "trend": 0.5, "expectancy": 0.5, "router": 0.5}
        original = 0.5
        result = ablate_score(original, factors, DEFAULT_WEIGHTS, ["carry"])
        assert result == pytest.approx(0.5, abs=1e-6)

    def test_unknown_factor_in_remove_treated_as_zero(self):
        factors = {"carry": 0.44, "trend": 0.5}
        weights = {"carry": 0.5, "trend": 0.5}
        result = ablate_score(0.47, factors, weights, ["unknown"])
        # w_removed = 0, so return original
        assert result == pytest.approx(0.47, abs=1e-6)


# ── TestComputeAblationGrid ───────────────────────────────────────────────

class TestComputeAblationGrid:
    def test_produces_7_variant_scores(self):
        decomp = {"trend": 0.5, "carry": 0.44, "expectancy": 0.48, "router": 0.5}
        ep = _make_episode(hybrid_score=0.48, net_pnl=1.0)
        grid = compute_ablation_grid([ep], 0.0001, -0.0002, decomp)
        assert len(grid) == 1
        row = grid[0]
        for v in ABLATION_VARIANTS:
            assert f"{v['label']}_score" in row
            assert f"{v['label']}_region" in row

    def test_filters_zero_score(self):
        decomp = {"trend": 0.5, "carry": 0.5, "expectancy": 0.5, "router": 0.5}
        ep = _make_episode(hybrid_score=0.0)
        grid = compute_ablation_grid([ep], 0.0, 0.0, decomp)
        assert len(grid) == 0

    def test_filters_non_btc(self):
        decomp = {"trend": 0.5, "carry": 0.5, "expectancy": 0.5, "router": 0.5}
        ep = _make_episode()
        ep["symbol"] = "ETHUSDT"
        grid = compute_ablation_grid([ep], 0.0, 0.0, decomp)
        assert len(grid) == 0

    def test_v0_equals_original(self):
        decomp = {"trend": 0.5, "carry": 0.5, "expectancy": 0.5, "router": 0.5}
        ep = _make_episode(hybrid_score=0.47)
        grid = compute_ablation_grid([ep], 0.0, 0.0, decomp)
        assert grid[0]["v0_score"] == pytest.approx(0.47, abs=1e-6)

    def test_scores_clamped(self):
        # Extreme ablation should clamp to [-1, 1]
        decomp = {"trend": 0.99, "carry": 0.01, "expectancy": 0.99, "router": 0.01}
        ep = _make_episode(hybrid_score=0.99)
        grid = compute_ablation_grid([ep], 0.0, 0.0, decomp)
        for v in ABLATION_VARIANTS:
            assert -1.0 <= grid[0][f"{v['label']}_score"] <= 1.0

    def test_includes_factors_in_output(self):
        decomp = {"trend": 0.5, "carry": 0.44, "expectancy": 0.48, "router": 0.5}
        ep = _make_episode()
        grid = compute_ablation_grid([ep], 0.0001, -0.0002, decomp)
        assert "factors" in grid[0]
        assert set(grid[0]["factors"].keys()) == {"trend", "carry", "expectancy", "router"}

    def test_multiple_episodes(self):
        decomp = {"trend": 0.5, "carry": 0.5, "expectancy": 0.5, "router": 0.5}
        eps = [
            _make_episode(hybrid_score=0.45, net_pnl=2.0, episode_id="EP_001"),
            _make_episode(hybrid_score=0.50, net_pnl=-1.0, episode_id="EP_002"),
            _make_episode(hybrid_score=0.55, net_pnl=0.5, episode_id="EP_003"),
        ]
        grid = compute_ablation_grid(eps, 0.0, 0.0, decomp)
        assert len(grid) == 3


# ── TestSectionCarryCrosscheck ─────────────────────────────────────────────

class TestSectionCarryCrosscheck:
    def _build_grid_with_scores(self, v0_scores, v1_scores, pnls, sides=None):
        """Build grid rows with explicit v0/v1 scores."""
        grid = []
        for i, (v0, v1, pnl) in enumerate(zip(v0_scores, v1_scores, pnls)):
            row = _make_grid_row(
                side=sides[i] if sides else "LONG",
                net_pnl=pnl,
                episode_id=f"EP_T{i:03d}",
            )
            row["v0_score"] = v0
            row["v0_region"] = classify_region(v0)
            row["v1_score"] = v1
            row["v1_region"] = classify_region(v1)
            grid.append(row)
        return grid

    def test_crosscheck_pass_when_directions_match(self):
        """If our −carry ablation shows same directions as carry-fix, pass."""
        # Carry-fix reference says: τ worsened, midpoint worsened, spillover worsened
        # So build data where v1 indeed has worse τ, worse midpoint, worse spillover
        v0_scores = [0.43, 0.45, 0.47, 0.49, 0.50]
        # V1 shifts scores higher (further from mask midpoint → worsened midpoint)
        v1_scores = [0.46, 0.48, 0.50, 0.52, 0.54]
        # PnL inversely correlated with scores (τ worsens when scores shift up)
        pnls = [2.0, 1.0, 0.0, -1.0, -2.0]
        grid = self._build_grid_with_scores(v0_scores, v1_scores, pnls)
        result = section_carry_crosscheck(grid)
        assert result["status"] == "CROSSCHECK_PASS"

    def test_crosscheck_fail_on_disagreement(self):
        """If our −carry ablation disagrees with carry-fix, fail."""
        v0_scores = [0.50, 0.52, 0.54]
        # V1 improves midpoint (moves down toward mask) — disagrees with carry-fix
        v1_scores = [0.44, 0.46, 0.48]
        pnls = [-1.0, 0.0, 1.0]
        grid = self._build_grid_with_scores(v0_scores, v1_scores, pnls)
        result = section_carry_crosscheck(grid)
        # midpoint moved closer to mask → midpoint_worsened=False
        # carry-fix says midpoint_worsened=True → disagrees
        assert result["midpoint_agrees"] is False

    def test_output_has_required_fields(self):
        grid = [_make_grid_row()]
        result = section_carry_crosscheck(grid)
        assert "status" in result
        assert "tau_agrees" in result
        assert "midpoint_agrees" in result
        assert "spillover_agrees" in result


# ── TestSectionCorrelationGrid ─────────────────────────────────────────────

class TestSectionCorrelationGrid:
    def test_all_7_variants_present(self):
        grid = [
            _make_grid_row(net_pnl=1.0, v0_score=0.45, episode_id=f"EP_{i}")
            for i in range(5)
        ]
        result = section_correlation_grid(grid)
        for v in ABLATION_VARIANTS:
            assert v["label"] in result
            assert "kendall_tau" in result[v["label"]]
            assert "spearman_rho" in result[v["label"]]
            assert "q5_minus_q1" in result[v["label"]]

    def test_includes_variant_names(self):
        grid = [_make_grid_row(net_pnl=1.0, episode_id=f"EP_{i}") for i in range(3)]
        result = section_correlation_grid(grid)
        assert result["v0"]["name"] == "baseline"
        assert result["v1"]["name"] == "−carry"


# ── TestSectionDisplacement ────────────────────────────────────────────────

class TestSectionDisplacement:
    def test_midpoint_distance_sign(self):
        """Scores above mask midpoint should have positive midpoint_distance."""
        grid = [_make_grid_row(v0_score=0.55, episode_id=f"EP_{i}") for i in range(3)]
        result = section_displacement(grid)
        assert result["v0"]["midpoint_distance"] > 0

    def test_scores_below_midpoint(self):
        grid = [_make_grid_row(v0_score=0.30, episode_id=f"EP_{i}") for i in range(3)]
        result = section_displacement(grid)
        assert result["v0"]["midpoint_distance"] < 0

    def test_spillover_pressure_computation(self):
        row1 = _make_grid_row(v0_score=0.45, episode_id="EP_1")
        row1["v0_region"] = "mask_interior"
        row2 = _make_grid_row(v0_score=0.50, episode_id="EP_2")
        row2["v0_region"] = "near_miss"
        grid = [row1, row2]
        result = section_displacement(grid)
        # 1 near_miss / 1 in_mask = 1.0
        assert result["v0"]["spillover_pressure"] == pytest.approx(1.0, abs=1e-4)

    def test_all_variants_present(self):
        grid = [_make_grid_row(episode_id=f"EP_{i}") for i in range(3)]
        result = section_displacement(grid)
        for v in ABLATION_VARIANTS:
            assert v["label"] in result


# ── TestSectionEconomic ────────────────────────────────────────────────────

class TestSectionEconomic:
    def test_direction_split_present(self):
        grid = [
            _make_grid_row(side="LONG", net_pnl=2.0, episode_id="EP_1"),
            _make_grid_row(side="SHORT", net_pnl=3.0, episode_id="EP_2"),
        ]
        # Force both into mask_interior
        for r in grid:
            r["v0_region"] = "mask_interior"
        result = section_economic(grid)
        assert "long_mask" in result["v0"]
        assert "short_mask" in result["v0"]
        assert result["v0"]["long_mask"]["count"] == 1
        assert result["v0"]["short_mask"]["count"] == 1
        assert result["v0"]["long_mask"]["mean"] == pytest.approx(2.0)
        assert result["v0"]["short_mask"]["mean"] == pytest.approx(3.0)

    def test_abstention_savings(self):
        row = _make_grid_row(net_pnl=-5.0, episode_id="EP_1")
        row["v0_region"] = "near_miss"
        result = section_economic([row])
        assert result["v0"]["abstention_savings"] == pytest.approx(5.0, abs=1e-4)

    def test_empty_region_has_zero_count(self):
        row = _make_grid_row(episode_id="EP_1")
        row["v0_region"] = "outside"
        result = section_economic([row])
        assert result["v0"]["regions"]["mask_interior"]["count"] == 0
        assert result["v0"]["regions"]["near_miss"]["count"] == 0

    def test_all_variants_present(self):
        grid = [_make_grid_row(episode_id=f"EP_{i}") for i in range(3)]
        result = section_economic(grid)
        for v in ABLATION_VARIANTS:
            assert v["label"] in result


# ── TestSectionInteraction ─────────────────────────────────────────────────

class TestSectionInteraction:
    def test_independent_factors(self):
        """When pair ablation effect equals sum of singles, interaction ≈ 0."""
        # Build grid where ablation is perfectly additive
        decomp = {"trend": 0.5, "carry": 0.5, "expectancy": 0.5, "router": 0.5}
        eps = [
            _make_episode(hybrid_score=0.5, net_pnl=float(i), episode_id=f"EP_{i}")
            for i in range(10)
        ]
        grid = compute_ablation_grid(eps, 0.0, 0.0, decomp)
        result = section_interaction(grid)
        # When all factors equal, ablation is perfectly additive
        for pair_label, data in result["interactions"].items():
            for m, mdata in data["metrics"].items():
                # Should be near-zero interaction
                assert abs(mdata["interaction_score"]) < 0.01

    def test_interaction_structure(self):
        grid = [_make_grid_row(episode_id=f"EP_{i}") for i in range(5)]
        result = section_interaction(grid)
        assert "interactions" in result
        assert "strongest" in result
        assert "v4" in result["interactions"]
        assert "v5" in result["interactions"]
        assert "v6" in result["interactions"]
        for m in ["tau", "midpoint_distance", "spillover"]:
            assert m in result["interactions"]["v4"]["metrics"]

    def test_classification_labels(self):
        grid = [_make_grid_row(episode_id=f"EP_{i}") for i in range(5)]
        result = section_interaction(grid)
        valid_labels = {"synergistic", "antagonistic", "independent"}
        for pair_label, data in result["interactions"].items():
            for m, mdata in data["metrics"].items():
                assert mdata["classification"] in valid_labels

    def test_strongest_reported(self):
        grid = [_make_grid_row(episode_id=f"EP_{i}") for i in range(5)]
        result = section_interaction(grid)
        assert "pair" in result["strongest"]
        assert "metric" in result["strongest"]
        assert "score" in result["strongest"]
        assert "significant" in result["strongest"]


# ── TestSectionVerdict ─────────────────────────────────────────────────────

class TestSectionVerdict:
    def _make_sections(
        self,
        *,
        tier: str = "STRONG",
        crosscheck_status: str = "CROSSCHECK_PASS",
        v0_tau: float = -0.3,
        v1_tau: float = -0.2,
        v2_tau: float = -0.25,
        v3_tau: float = -0.29,
        v4_tau: float = 0.0,
        v5_tau: float = -0.15,
        v6_tau: float = -0.20,
        interaction_significant: bool = True,
        short_count: int = 2,
        short_mean: float = 1.5,
    ):
        recon = {"confidence_tier": tier, "n": 10, "rmse": 0.02}
        crosscheck = {"status": crosscheck_status}
        corr: dict = {"n": 10}
        for label, tau in [("v0", v0_tau), ("v1", v1_tau), ("v2", v2_tau),
                           ("v3", v3_tau), ("v4", v4_tau), ("v5", v5_tau),
                           ("v6", v6_tau)]:
            corr[label] = {"kendall_tau": tau, "spearman_rho": 0.0, "q5_minus_q1": 0.0, "name": label}
        displacement = {"n": 10}
        economic: dict = {}
        for v in ABLATION_VARIANTS:
            economic[v["label"]] = {
                "name": v["name"],
                "regions": {},
                "long_mask": {"count": 5, "mean": 1.0, "total": 5.0, "win_rate": 0.8},
                "short_mask": {"count": short_count, "mean": short_mean, "total": short_count * short_mean, "win_rate": 0.7},
            }
        interaction = {
            "strongest": {
                "pair": "v4",
                "metric": "tau",
                "score": 0.05 if interaction_significant else 0.01,
                "significant": interaction_significant,
            },
        }
        return recon, crosscheck, corr, displacement, economic, interaction

    def test_single_factor_dominant(self):
        sections = self._make_sections(
            v1_tau=0.0,   # huge improvement from v0=-0.3 → Δ=+0.3
            v4_tau=0.05,  # pair doesn't beat single by 0.02
            interaction_significant=False,
        )
        result = section_verdict(*sections)
        assert result["recommendation"] == "SINGLE_FACTOR_DOMINANT"
        assert result["confidence_tier"] == "STRONG"

    def test_interaction_dominant(self):
        sections = self._make_sections(
            v1_tau=-0.25,  # weak single improvement (Δ=+0.05)
            v2_tau=-0.27,
            v3_tau=-0.28,
            v4_tau=0.05,   # big pair improvement (Δ=+0.35)
            interaction_significant=True,
        )
        result = section_verdict(*sections)
        assert result["recommendation"] == "INTERACTION_DOMINANT"

    def test_inconclusive_from_exploratory_tier(self):
        sections = self._make_sections(tier="EXPLORATORY")
        result = section_verdict(*sections)
        assert result["recommendation"] == "INCONCLUSIVE"
        assert "RMSE" in result["reason"]

    def test_inconclusive_from_crosscheck_fail(self):
        sections = self._make_sections(crosscheck_status="CROSSCHECK_FAIL")
        result = section_verdict(*sections)
        assert result["recommendation"] == "INCONCLUSIVE"
        assert "crosscheck" in result["reason"].lower()

    def test_inconclusive_when_no_improvement(self):
        sections = self._make_sections(
            v1_tau=-0.31, v2_tau=-0.32, v3_tau=-0.33,
            v4_tau=-0.35, v5_tau=-0.36, v6_tau=-0.37,
            interaction_significant=False,
        )
        result = section_verdict(*sections)
        assert result["recommendation"] == "INCONCLUSIVE"

    def test_short_preservation_flag(self):
        sections = self._make_sections(short_count=0, short_mean=0.0)
        result = section_verdict(*sections)
        assert result["short_preserved"] is False

    def test_verdict_has_required_fields(self):
        sections = self._make_sections()
        result = section_verdict(*sections)
        for key in ["confidence_tier", "crosscheck_ok", "best_single", "best_pair",
                     "interaction_detected", "short_preserved", "recommendation", "reason"]:
            assert key in result


# ── TestGenerateReport ─────────────────────────────────────────────────────

class TestGenerateReport:
    def test_end_to_end_with_mock_data(self, tmp_path):
        # Decomposition file
        decomp_path = tmp_path / "decomp.jsonl"
        decomp_path.write_text(
            _make_decomp_line(trend=0.5, carry=0.44, expectancy=0.48, router=0.5) + "\n"
        )

        # Episode file
        ep_path = tmp_path / "episodes.json"
        episodes = [
            _make_episode(hybrid_score=0.45, net_pnl=2.0, side="LONG", episode_id="EP_001"),
            _make_episode(hybrid_score=0.48, net_pnl=-1.0, side="SHORT", episode_id="EP_002"),
            _make_episode(hybrid_score=0.50, net_pnl=0.5, side="LONG", episode_id="EP_003"),
        ]
        ep_path.write_text(json.dumps({"episodes": episodes}))

        # Funding/basis files (empty → defaults)
        fund_path = tmp_path / "funding.json"
        fund_path.write_text("{}")
        basis_path = tmp_path / "basis.json"
        basis_path.write_text("{}")

        report = generate_report(
            episode_path=ep_path,
            funding_path=fund_path,
            basis_path=basis_path,
            decomp_path=decomp_path,
            funding_rate_override=0.0001,
            basis_pct_override=-0.0002,
        )

        assert report["experiment"] == "Factor Ablation Grid"
        assert report["symbol"] == "BTCUSDT"
        assert "Reconstruction Quality" in report["sections"]
        assert "Carry-Only Crosscheck" in report["sections"]
        assert "Correlation Grid" in report["sections"]
        assert "Displacement Analysis" in report["sections"]
        assert "Economic Impact" in report["sections"]
        assert "Interaction Detection" in report["sections"]
        assert "Verdict" in report["sections"]

    def test_json_output_mode(self, tmp_path, capsys):
        decomp_path = tmp_path / "decomp.jsonl"
        decomp_path.write_text(_make_decomp_line() + "\n")
        ep_path = tmp_path / "episodes.json"
        ep_path.write_text(json.dumps({"episodes": [
            _make_episode(hybrid_score=0.48, net_pnl=1.0),
        ]}))
        fund_path = tmp_path / "funding.json"
        fund_path.write_text("{}")
        basis_path = tmp_path / "basis.json"
        basis_path.write_text("{}")

        generate_report(
            json_output=True,
            episode_path=ep_path,
            funding_path=fund_path,
            basis_path=basis_path,
            decomp_path=decomp_path,
            funding_rate_override=0.0,
            basis_pct_override=0.0,
        )
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["experiment"] == "Factor Ablation Grid"

    def test_text_output_mode(self, tmp_path, capsys):
        decomp_path = tmp_path / "decomp.jsonl"
        decomp_path.write_text(_make_decomp_line() + "\n")
        ep_path = tmp_path / "episodes.json"
        ep_path.write_text(json.dumps({"episodes": [
            _make_episode(hybrid_score=0.48, net_pnl=1.0),
        ]}))
        fund_path = tmp_path / "funding.json"
        fund_path.write_text("{}")
        basis_path = tmp_path / "basis.json"
        basis_path.write_text("{}")

        generate_report(
            json_output=False,
            episode_path=ep_path,
            funding_path=fund_path,
            basis_path=basis_path,
            decomp_path=decomp_path,
            funding_rate_override=0.0,
            basis_pct_override=0.0,
        )
        captured = capsys.readouterr()
        assert "FACTOR ABLATION GRID" in captured.out
        assert "Reconstruction Quality" in captured.out
        assert "Verdict" in captured.out
