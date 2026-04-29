"""Tests for carry_fix_experiment.py — Layer 0 BTC carry-direction bias experiment."""
import json


from scripts.carry_fix_experiment import (
    NEAR_MISS_BAND,
    REFERENCE_MASK,
    classify_region,
    compute_carry_score,
    compute_carry_score_direction_neutral,
    compute_variants,
    generate_report,
    recompute_hybrid_score,
    scale_basis,
    scale_funding_rate,
    section_correlation,
    section_economic_validation,
    section_mask_alignment,
    section_pass_fail,
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


# ── scale_funding_rate ──────────────────────────────────────────────────────

class TestScaleFundingRate:
    def test_zero_funding_is_neutral(self):
        assert scale_funding_rate(0.0, "LONG") == 0.5
        assert scale_funding_rate(0.0, "SHORT") == 0.5

    def test_positive_funding_penalizes_long(self):
        score = scale_funding_rate(0.0001, "LONG")
        assert score < 0.5

    def test_positive_funding_favors_short(self):
        score = scale_funding_rate(0.0001, "SHORT")
        assert score > 0.5

    def test_negative_funding_favors_long(self):
        score = scale_funding_rate(-0.0001, "LONG")
        assert score > 0.5

    def test_bounded_0_1(self):
        for rate in [-0.01, -0.001, 0, 0.001, 0.01]:
            for d in ["LONG", "SHORT"]:
                s = scale_funding_rate(rate, d)
                assert 0.0 <= s <= 1.0


# ── scale_basis ─────────────────────────────────────────────────────────────

class TestScaleBasis:
    def test_zero_basis_is_neutral(self):
        assert scale_basis(0.0, "LONG") == 0.5
        assert scale_basis(0.0, "SHORT") == 0.5

    def test_positive_basis_penalizes_long(self):
        assert scale_basis(0.01, "LONG") < 0.5

    def test_positive_basis_favors_short(self):
        assert scale_basis(0.01, "SHORT") > 0.5


# ── compute_carry_score ─────────────────────────────────────────────────────

class TestComputeCarryScore:
    def test_neutral_at_zero(self):
        score = compute_carry_score(0.0, 0.0, "LONG")
        assert score == 0.5

    def test_direction_asymmetry(self):
        long_score = compute_carry_score(0.0001, 0.001, "LONG")
        short_score = compute_carry_score(0.0001, 0.001, "SHORT")
        assert long_score < 0.5  # penalized
        assert short_score > 0.5  # favored

    def test_bounded(self):
        for d in ["LONG", "SHORT"]:
            s = compute_carry_score(0.01, 0.05, d)
            assert 0.0 <= s <= 1.0


# ── compute_carry_score_direction_neutral ───────────────────────────────────

class TestDirectionNeutralCarry:
    def test_always_above_neutral(self):
        """Direction-neutral carry uses |magnitude|, so always >= 0.5."""
        s = compute_carry_score_direction_neutral(0.0001, 0.001)
        assert s >= 0.5

    def test_zero_inputs_neutral(self):
        s = compute_carry_score_direction_neutral(0.0, 0.0)
        assert s == 0.5

    def test_same_for_pos_neg_funding(self):
        s1 = compute_carry_score_direction_neutral(0.0001, 0.0)
        s2 = compute_carry_score_direction_neutral(-0.0001, 0.0)
        assert abs(s1 - s2) < 1e-9


# ── recompute_hybrid_score ──────────────────────────────────────────────────

class TestRecomputeHybridScore:
    def test_same_carry_no_change(self):
        """Replacing carry with the same value should give original score."""
        score = recompute_hybrid_score(0.48, 0.35, 0.35, 0.25)
        assert abs(score - 0.48) < 1e-9

    def test_neutral_carry_shifts_toward_non_carry_mean(self):
        """Neutralizing carry should pull score toward non-carry center."""
        original = 0.48
        low_carry = 0.3  # below neutral
        new = recompute_hybrid_score(original, low_carry, 0.5, 0.25)
        assert new > original  # pushing carry from 0.3 to 0.5 raises score

    def test_algebraic_identity(self):
        """Verify: new_score = old_score + w * (new_carry - old_carry)."""
        old = 0.50
        old_carry = 0.3
        new_carry = 0.5
        w = 0.25
        expected = old + w * (new_carry - old_carry)
        actual = recompute_hybrid_score(old, old_carry, new_carry, w)
        assert abs(actual - expected) < 1e-9


# ── classify_region ─────────────────────────────────────────────────────────

class TestClassifyRegion:
    def test_inside_mask(self):
        assert classify_region(0.45) == "mask_interior"

    def test_near_miss(self):
        assert classify_region(BTC_HI + 0.01) == "near_miss"

    def test_outside(self):
        assert classify_region(BTC_HI + NEAR_MISS_BAND + 0.01) == "outside"
        assert classify_region(BTC_LO - 0.01) == "outside"


# ── compute_variants ────────────────────────────────────────────────────────

class TestComputeVariants:
    def test_basic_variant_output(self):
        episodes = [_make_episode(hybrid_score=0.48, net_pnl=1.0)]
        variants = compute_variants(episodes, funding_rate=0.0001, basis_pct=0.001)
        assert len(variants) == 1
        v = variants[0]
        assert "v0_score" in v
        assert "v1_score" in v
        assert "v2_score" in v
        assert v["v0_score"] == 0.48

    def test_v1_shifts_toward_neutral(self):
        """V1 should shift score when carry is below neutral."""
        episodes = [_make_episode(hybrid_score=0.48, side="LONG")]
        # Positive funding penalizes LONG → carry < 0.5
        variants = compute_variants(episodes, funding_rate=0.0005, basis_pct=0.005)
        v = variants[0]
        assert v["estimated_carry"] < 0.5
        # V1 (carry→0.5) should raise the score
        assert v["v1_score"] > v["v0_score"]

    def test_filters_non_btc(self):
        episodes = [{
            "episode_id": "EP_X", "symbol": "ETHUSDT", "side": "LONG",
            "entry_ts": "2026-01-01T00:00:00+00:00",
            "exit_ts": "2026-01-01T01:00:00+00:00",
            "hybrid_score": 0.45, "net_pnl": 1.0,
        }]
        variants = compute_variants(episodes, funding_rate=0.0, basis_pct=0.0)
        assert len(variants) == 0

    def test_filters_zero_score(self):
        episodes = [_make_episode(hybrid_score=0.0)]
        variants = compute_variants(episodes, funding_rate=0.0, basis_pct=0.0)
        assert len(variants) == 0

    def test_region_classification_per_variant(self):
        """Variants can produce different region classifications."""
        # Score near mask upper boundary — carry fix might push it inside
        episodes = [_make_episode(hybrid_score=0.50, side="LONG")]
        variants = compute_variants(episodes, funding_rate=0.0005, basis_pct=0.005)
        v = variants[0]
        assert v["v0_region"] == "near_miss"
        # V1 should pull it back since carry was penalizing longs
        # (whether it crosses the boundary depends on magnitude)


# ── section_correlation ─────────────────────────────────────────────────────

class TestSectionCorrelation:
    def test_perfect_positive(self):
        variants = [
            {"v0_score": 0.4, "v1_score": 0.4, "v2_score": 0.4, "net_pnl": -1},
            {"v0_score": 0.5, "v1_score": 0.5, "v2_score": 0.5, "net_pnl": 0},
            {"v0_score": 0.6, "v1_score": 0.6, "v2_score": 0.6, "net_pnl": 1},
        ]
        result = section_correlation(variants)
        assert result["v0"]["kendall_tau"] > 0
        assert result["v0"]["spearman_rho"] > 0

    def test_inverted_correlation(self):
        variants = [
            {"v0_score": 0.6, "v1_score": 0.6, "v2_score": 0.6, "net_pnl": -2},
            {"v0_score": 0.5, "v1_score": 0.5, "v2_score": 0.5, "net_pnl": 0},
            {"v0_score": 0.4, "v1_score": 0.4, "v2_score": 0.4, "net_pnl": 2},
        ]
        result = section_correlation(variants)
        assert result["v0"]["kendall_tau"] < 0


# ── section_mask_alignment ──────────────────────────────────────────────────

class TestSectionMaskAlignment:
    def test_all_inside_mask(self):
        variants = [
            {"v0_score": 0.45, "v1_score": 0.45, "v2_score": 0.45,
             "v0_region": "mask_interior", "v1_region": "mask_interior",
             "v2_region": "mask_interior", "net_pnl": 1.0}
        ]
        result = section_mask_alignment(variants)
        assert result["v0"]["in_mask_pct"] == 1.0
        assert result["v0"]["spillover_pressure"] == 0.0


# ── section_economic_validation ─────────────────────────────────────────────

class TestSectionEconomicValidation:
    def test_abstention_savings_positive(self):
        """Near-miss losers → positive abstention savings."""
        variants = [
            {"v0_score": 0.45, "v1_score": 0.45, "v2_score": 0.45,
             "v0_region": "mask_interior", "v1_region": "mask_interior",
             "v2_region": "mask_interior", "net_pnl": 2.0},
            {"v0_score": 0.50, "v1_score": 0.50, "v2_score": 0.50,
             "v0_region": "near_miss", "v1_region": "near_miss",
             "v2_region": "near_miss", "net_pnl": -3.0},
        ]
        result = section_economic_validation(variants)
        assert result["v0"]["abstention_savings"] == 3.0  # -(-3.0)


# ── section_pass_fail ───────────────────────────────────────────────────────

class TestSectionPassFail:
    def _make_inputs(self, *, v1_tau=0.2, v0_tau=0.1, v1_dist=0.01, v0_dist=0.02,
                     v1_sp=0.1, v0_sp=0.3, v1_mask_ev=1.0, v1_nm_ev=-0.5):
        corr = {
            "v0": {"kendall_tau": v0_tau}, "v1": {"kendall_tau": v1_tau},
            "v2": {"kendall_tau": v1_tau},
        }
        align = {
            "v0": {"midpoint_distance": v0_dist, "spillover_pressure": v0_sp},
            "v1": {"midpoint_distance": v1_dist, "spillover_pressure": v1_sp},
            "v2": {"midpoint_distance": v1_dist, "spillover_pressure": v1_sp},
        }
        econ = {
            "v0": {"regions": {"mask_interior": {"mean": 1.0}, "near_miss": {"mean": -0.5, "count": 2}}},
            "v1": {"regions": {"mask_interior": {"mean": v1_mask_ev}, "near_miss": {"mean": v1_nm_ev, "count": 2}}},
            "v2": {"regions": {"mask_interior": {"mean": v1_mask_ev}, "near_miss": {"mean": v1_nm_ev, "count": 2}}},
        }
        return corr, align, econ

    def test_all_pass(self):
        corr, align, econ = self._make_inputs()
        result = section_pass_fail(corr, align, econ)
        assert result["verdicts"]["v1"]["all_pass"] is True

    def test_fail_on_midpoint_worsens(self):
        corr, align, econ = self._make_inputs(v1_dist=0.05, v0_dist=0.02)
        result = section_pass_fail(corr, align, econ)
        assert result["verdicts"]["v1"]["all_pass"] is False

    def test_fail_on_mask_ev_worse_than_nm(self):
        corr, align, econ = self._make_inputs(v1_mask_ev=-0.5, v1_nm_ev=1.0)
        result = section_pass_fail(corr, align, econ)
        assert result["verdicts"]["v1"]["all_pass"] is False


# ── generate_report ─────────────────────────────────────────────────────────

class TestGenerateReport:
    def test_empty_episodes(self, tmp_path):
        ep_file = tmp_path / "episodes.json"
        ep_file.write_text(json.dumps({"episodes": []}))
        fund_file = tmp_path / "funding.json"
        fund_file.write_text(json.dumps({"symbols": {"BTCUSDT": {"rate": 0.0001}}}))
        basis_file = tmp_path / "basis.json"
        basis_file.write_text(json.dumps({"symbols": {"BTCUSDT": {"basis_pct": 0.001}}}))
        result = generate_report(
            episode_path=ep_file,
            funding_path=fund_file,
            basis_path=basis_file,
        )
        assert result["total_episodes"] == 0
        assert "Score–PnL Correlation" in result["sections"]

    def test_full_structure(self, tmp_path):
        episodes = [
            _make_episode(hybrid_score=0.48, net_pnl=2.0, episode_id="EP_001"),
            _make_episode(hybrid_score=0.50, net_pnl=-1.0, episode_id="EP_002"),
            _make_episode(hybrid_score=0.42, net_pnl=0.5, episode_id="EP_003"),
        ]
        ep_file = tmp_path / "episodes.json"
        ep_file.write_text(json.dumps({"episodes": episodes}))
        fund_file = tmp_path / "funding.json"
        fund_file.write_text(json.dumps({"symbols": {"BTCUSDT": {"rate": 0.0001}}}))
        basis_file = tmp_path / "basis.json"
        basis_file.write_text(json.dumps({"symbols": {"BTCUSDT": {"basis_pct": -0.0004}}}))

        result = generate_report(
            episode_path=ep_file,
            funding_path=fund_file,
            basis_path=basis_file,
        )
        assert result["total_episodes"] == 3
        titles = set(result["sections"].keys())
        assert titles == {
            "Score–PnL Correlation",
            "Mask Alignment",
            "Economic Validation",
            "Selector Relevance",
            "Pass/Fail Assessment",
        }
        assert result["carry_inputs"]["funding_rate"] == 0.0001

    def test_funding_rate_override(self, tmp_path):
        episodes = [_make_episode(hybrid_score=0.48, net_pnl=1.0)]
        ep_file = tmp_path / "episodes.json"
        ep_file.write_text(json.dumps({"episodes": episodes}))
        fund_file = tmp_path / "funding.json"
        fund_file.write_text(json.dumps({"symbols": {"BTCUSDT": {"rate": 0.0001}}}))
        basis_file = tmp_path / "basis.json"
        basis_file.write_text(json.dumps({"symbols": {"BTCUSDT": {"basis_pct": 0.0}}}))

        result = generate_report(
            episode_path=ep_file,
            funding_path=fund_file,
            basis_path=basis_file,
            funding_rate_override=0.0005,
        )
        assert result["carry_inputs"]["funding_rate"] == 0.0005
