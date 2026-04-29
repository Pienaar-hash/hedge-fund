"""Structural invariants for Selector V2 shadow surface.

Locks the constants and behaviours introduced with V2 profit masks
and Candidate D routing. These protect against silent regression
during the shadow observation phase before D activation.
"""

import pytest

from execution.shadow_selector_v2 import (
    PNL_POSITIVE_REGIONS_V2,
    REGIME_BOUNDARIES_V2,
    REGIME_LABELS_V2,
    ZERO_SCORE_POLICY,
    _selector_d,
    in_profit_region,
)


# ── INV-5: mask upper bound within regime boundary ──────────────────────────

class TestInv5MaskWithinBoundary:
    def test_all_masks_within_regime(self):
        for sym, bands in PNL_POSITIVE_REGIONS_V2.items():
            if not bands:
                continue
            hi = max(hi for _, hi in bands)
            regime_bounds = REGIME_BOUNDARIES_V2.get(sym, [])
            assert regime_bounds, f"{sym} has profit mask but no regime boundaries"
            regime_hi = regime_bounds[-1]
            assert hi <= regime_hi, (
                f"INV-5 violation: {sym} mask_hi={hi} > regime_hi={regime_hi}"
            )

    def test_mask_lower_bound_positive(self):
        for sym, bands in PNL_POSITIVE_REGIONS_V2.items():
            for lo, hi in bands:
                assert lo > 0, f"{sym}: mask lower bound must be positive, got {lo}"
                assert lo < hi, f"{sym}: mask lower bound {lo} >= upper bound {hi}"


# ── ETH mask discarded (V2 stability gate) ──────────────────────────────────

class TestEthMaskDiscarded:
    def test_eth_has_no_profit_region(self):
        assert PNL_POSITIVE_REGIONS_V2["ETHUSDT"] == []

    def test_eth_never_routes_hydra_via_d(self):
        for score in [0.44, 0.45, 0.48, 0.50, 0.51]:
            r = _selector_d("ETHUSDT", score)
            assert r["v2_abstain"] is True, (
                f"ETHUSDT score={score}: D should abstain (V2 mask empty)"
            )
            assert r["v2_choice"] == "none"

    def test_sol_has_no_profit_region(self):
        assert PNL_POSITIVE_REGIONS_V2["SOLUSDT"] == []


# ── Zero-score determinism ──────────────────────────────────────────────────

class TestZeroScoreInvariants:
    def test_none_score_abstains(self):
        r = _selector_d("BTCUSDT", None)
        assert r["v2_abstain"] is True
        assert r["v2_choice"] == "none"
        assert r["d_zero_score"] is True
        assert r["rule"] == "D_no_score"

    def test_zero_score_abstains(self):
        r = _selector_d("BTCUSDT", 0.0)
        assert r["v2_abstain"] is True
        assert r["v2_choice"] == "none"
        assert r["d_zero_score"] is True
        assert r["rule"] == "D_zero_score"

    def test_negative_score_abstains(self):
        r = _selector_d("BTCUSDT", -0.1)
        assert r["v2_abstain"] is True
        assert r["d_zero_score"] is True

    def test_policy_action_is_route_to_legacy(self):
        assert ZERO_SCORE_POLICY["action"] == "ROUTE_TO_LEGACY"
        assert ZERO_SCORE_POLICY["candidate_d_verdict"] == "ABSTAIN"


# ── Candidate D profit-region routing ───────────────────────────────────────

class TestCandidateDRouting:
    def test_btc_inside_mask_routes_hydra(self):
        r = _selector_d("BTCUSDT", 0.45)
        assert r["v2_choice"] == "hydra"
        assert r["v2_abstain"] is False
        assert r["rule"] == "D_profit_region"
        assert r["d_zero_score"] is False

    def test_btc_outside_mask_abstains(self):
        r = _selector_d("BTCUSDT", 0.55)
        assert r["v2_choice"] == "none"
        assert r["v2_abstain"] is True
        assert r["rule"] == "D_abstain"

    def test_btc_below_mask_abstains(self):
        r = _selector_d("BTCUSDT", 0.30)
        assert r["v2_abstain"] is True
        assert r["rule"] == "D_abstain"

    def test_unknown_symbol_abstains(self):
        r = _selector_d("XYZUSDT", 0.50)
        assert r["v2_abstain"] is True
        assert r["v2_choice"] == "none"


# ── Regime/label structural consistency ─────────────────────────────────────

class TestRegimeStructure:
    def test_label_count_matches_boundaries(self):
        for sym, bounds in REGIME_BOUNDARIES_V2.items():
            labels = REGIME_LABELS_V2.get(sym, [])
            expected_labels = len(bounds) + 1
            assert len(labels) == expected_labels, (
                f"{sym}: {len(bounds)} boundaries require {expected_labels} labels, "
                f"got {len(labels)}"
            )

    def test_boundaries_monotonically_increasing(self):
        for sym, bounds in REGIME_BOUNDARIES_V2.items():
            for i in range(1, len(bounds)):
                assert bounds[i] > bounds[i - 1], (
                    f"{sym}: boundaries not monotonic at index {i}: "
                    f"{bounds[i - 1]} >= {bounds[i]}"
                )

    def test_all_mask_symbols_have_regime_entry(self):
        for sym in PNL_POSITIVE_REGIONS_V2:
            assert sym in REGIME_BOUNDARIES_V2, (
                f"{sym} has profit mask but no REGIME_BOUNDARIES_V2 entry"
            )


# ── in_profit_region consistency ────────────────────────────────────────────

class TestInProfitRegionConsistency:
    def test_btc_boundary_inclusivity(self):
        bands = PNL_POSITIVE_REGIONS_V2["BTCUSDT"]
        assert len(bands) == 1
        lo, hi = bands[0]
        assert in_profit_region("BTCUSDT", lo) is True
        assert in_profit_region("BTCUSDT", hi) is True
        assert in_profit_region("BTCUSDT", lo - 0.001) is False
        assert in_profit_region("BTCUSDT", hi + 0.001) is False

    def test_empty_symbol_always_false(self):
        assert in_profit_region("ETHUSDT", 0.45) is False
        assert in_profit_region("SOLUSDT", 0.50) is False
        assert in_profit_region("MISSING", 0.50) is False
