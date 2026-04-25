"""Tests for execution/score_edge_diagnostic.py — P3 carry-neutral shadow."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from execution.score_edge_diagnostic import (
    build_diagnostic_record,
    classify_defect,
)


# ── Fixtures ─────────────────────────────────────────────────────────────

def _make_intent(
    *,
    symbol: str = "BTCUSDT",
    side: str = "LONG",
    hybrid_score: float = 0.44,
    carry: float = 0.44,
    trend: float = 0.50,
    expectancy: float = 0.51,
    router: float = 0.50,
    w_carry: float = 0.25,
    w_trend: float = 0.40,
    w_expectancy: float = 0.20,
    w_router: float = 0.15,
    conviction_score: float = 0.42,
    intent_id: str = "ord_test",
) -> dict:
    return {
        "symbol": symbol,
        "positionSide": side,
        "hybrid_score": hybrid_score,
        "hybrid_components": {
            "carry": carry,
            "trend": trend,
            "expectancy": expectancy,
            "router": router,
        },
        "hybrid_weights_used": {
            "carry": w_carry,
            "trend": w_trend,
            "expectancy": w_expectancy,
            "router": w_router,
        },
        "conviction_score": conviction_score,
        "intent_id": intent_id,
    }


def _make_te_result(
    *,
    expected_edge_pct: float = 0.0,
    expected_edge_usd: float = 0.0,
    atr_pct: float = 0.005,
    k_atr: float = 0.6,
    adv: float = 0.0,
    confidence: float = 0.42,
    notional_usd: float = 50.0,
    source: str = "atr_conf_v1",
    fallback_reason: str = "",
):
    m = MagicMock()
    m.expected_edge_pct = expected_edge_pct
    m.expected_edge_usd = expected_edge_usd
    m.atr_pct = atr_pct
    m.k_atr = k_atr
    m.adv = adv
    m.confidence = confidence
    m.notional_usd = notional_usd
    m.source = source
    m.fallback_reason = fallback_reason
    return m


def _make_fg_details(
    *,
    required_edge_usd: float = 0.06,
    round_trip_fee_usd: float = 0.04,
    expected_edge_usd: float = 0.0,
) -> dict:
    return {
        "required_edge_usd": required_edge_usd,
        "round_trip_fee_usd": round_trip_fee_usd,
        "expected_edge_usd": expected_edge_usd,
    }


# ══════════════════════════════════════════════════════════════════════════
# build_diagnostic_record — happy path
# ══════════════════════════════════════════════════════════════════════════

class TestBuildDiagnosticRecord:
    """Core record-building tests."""

    def test_produces_record_with_required_fields(self):
        rec = build_diagnostic_record(
            intent=_make_intent(),
            te_result=_make_te_result(),
            fg_details=_make_fg_details(),
            fg_allowed=False,
        )
        assert rec is not None
        expected_keys = {
            "ts", "symbol", "side", "intent_id",
            "hybrid_score", "hybrid_score_reconstructed", "post_modifier_delta",
            "carry_score", "trend_score", "expectancy_score", "router_score",
            "weights",
            "confidence", "confidence_gap",
            "expected_edge_pct", "fee_required_pct", "fee_required_usd",
            "fee_rt_usd", "notional_usd",
            "edge_source", "adv", "atr_pct", "fallback_reason",
            "carry_neutral_score", "carry_neutral_reconstructed", "carry_delta",
            "veto_reason", "has_decomposition",
        }
        assert expected_keys.issubset(rec.keys())
        assert rec["has_decomposition"] is True

    def test_veto_reason_fee_gate(self):
        rec = build_diagnostic_record(
            intent=_make_intent(),
            te_result=_make_te_result(),
            fg_details=_make_fg_details(),
            fg_allowed=False,
        )
        assert rec["veto_reason"] == "fee_gate"

    def test_veto_reason_pass(self):
        rec = build_diagnostic_record(
            intent=_make_intent(conviction_score=0.65),
            te_result=_make_te_result(expected_edge_pct=0.001, adv=0.15),
            fg_details=_make_fg_details(expected_edge_usd=0.10),
            fg_allowed=True,
        )
        assert rec["veto_reason"] == "pass"

    def test_veto_reason_override(self):
        rec = build_diagnostic_record(
            intent=_make_intent(),
            te_result=_make_te_result(),
            fg_details=_make_fg_details(),
            fg_allowed=False,
            veto_reason="min_notional",
        )
        assert rec["veto_reason"] == "min_notional"

    def test_symbol_and_side_propagated(self):
        rec = build_diagnostic_record(
            intent=_make_intent(symbol="ETHUSDT", side="SHORT"),
            te_result=_make_te_result(),
            fg_details=_make_fg_details(),
            fg_allowed=False,
        )
        assert rec["symbol"] == "ETHUSDT"
        assert rec["side"] == "SHORT"


# ══════════════════════════════════════════════════════════════════════════
# Score decomposition
# ══════════════════════════════════════════════════════════════════════════

class TestScoreDecomposition:
    """Verify score surface fields are computed correctly."""

    def test_reconstructed_matches_weighted_sum(self):
        rec = build_diagnostic_record(
            intent=_make_intent(
                carry=0.44, trend=0.50, expectancy=0.51, router=0.50,
                w_carry=0.25, w_trend=0.40, w_expectancy=0.20, w_router=0.15,
            ),
            te_result=_make_te_result(),
            fg_details=_make_fg_details(),
            fg_allowed=False,
        )
        expected = 0.25 * 0.44 + 0.40 * 0.50 + 0.20 * 0.51 + 0.15 * 0.50
        assert abs(rec["hybrid_score_reconstructed"] - expected) < 1e-6

    def test_post_modifier_delta_computed(self):
        rec = build_diagnostic_record(
            intent=_make_intent(hybrid_score=0.40, carry=0.44, trend=0.50,
                                expectancy=0.51, router=0.50),
            te_result=_make_te_result(),
            fg_details=_make_fg_details(),
            fg_allowed=False,
        )
        expected_recon = 0.25 * 0.44 + 0.40 * 0.50 + 0.20 * 0.51 + 0.15 * 0.50
        assert abs(rec["post_modifier_delta"] - (0.40 - expected_recon)) < 1e-6

    def test_individual_scores_propagated(self):
        rec = build_diagnostic_record(
            intent=_make_intent(carry=0.33, trend=0.66, expectancy=0.77, router=0.88),
            te_result=_make_te_result(),
            fg_details=_make_fg_details(),
            fg_allowed=False,
        )
        assert abs(rec["carry_score"] - 0.33) < 1e-4
        assert abs(rec["trend_score"] - 0.66) < 1e-4
        assert abs(rec["expectancy_score"] - 0.77) < 1e-4
        assert abs(rec["router_score"] - 0.88) < 1e-4


# ══════════════════════════════════════════════════════════════════════════
# Confidence gap
# ══════════════════════════════════════════════════════════════════════════

class TestConfidenceGap:
    """Verify the structural gap measurement."""

    def test_below_threshold_positive_gap(self):
        rec = build_diagnostic_record(
            intent=_make_intent(conviction_score=0.42),
            te_result=_make_te_result(),
            fg_details=_make_fg_details(),
            fg_allowed=False,
        )
        assert rec["confidence_gap"] == pytest.approx(0.08, abs=1e-4)

    def test_above_threshold_negative_gap(self):
        rec = build_diagnostic_record(
            intent=_make_intent(conviction_score=0.65),
            te_result=_make_te_result(adv=0.15),
            fg_details=_make_fg_details(),
            fg_allowed=True,
        )
        assert rec["confidence_gap"] == pytest.approx(-0.15, abs=1e-4)

    def test_exactly_at_threshold(self):
        rec = build_diagnostic_record(
            intent=_make_intent(conviction_score=0.50),
            te_result=_make_te_result(),
            fg_details=_make_fg_details(),
            fg_allowed=False,
        )
        assert rec["confidence_gap"] == pytest.approx(0.0, abs=1e-6)


# ══════════════════════════════════════════════════════════════════════════
# Edge surface
# ══════════════════════════════════════════════════════════════════════════

class TestEdgeSurface:
    """Verify edge decomposition fields."""

    def test_fee_required_pct_computed(self):
        rec = build_diagnostic_record(
            intent=_make_intent(),
            te_result=_make_te_result(notional_usd=100.0),
            fg_details=_make_fg_details(required_edge_usd=0.12),
            fg_allowed=False,
        )
        assert rec["fee_required_pct"] == pytest.approx(0.0012, abs=1e-6)

    def test_fee_required_pct_zero_notional(self):
        rec = build_diagnostic_record(
            intent=_make_intent(),
            te_result=_make_te_result(notional_usd=0.0),
            fg_details=_make_fg_details(),
            fg_allowed=False,
        )
        assert rec["fee_required_pct"] == 0.0

    def test_edge_source_propagated(self):
        rec = build_diagnostic_record(
            intent=_make_intent(),
            te_result=_make_te_result(source="fallback_proxy"),
            fg_details=_make_fg_details(),
            fg_allowed=False,
        )
        assert rec["edge_source"] == "fallback_proxy"

    def test_fallback_reason_propagated(self):
        rec = build_diagnostic_record(
            intent=_make_intent(),
            te_result=_make_te_result(fallback_reason="atr_missing"),
            fg_details=_make_fg_details(),
            fg_allowed=False,
        )
        assert rec["fallback_reason"] == "atr_missing"


# ══════════════════════════════════════════════════════════════════════════
# Carry-neutral shadow
# ══════════════════════════════════════════════════════════════════════════

class TestCarryNeutralShadow:
    """Verify carry-neutral counterfactual computation."""

    def test_carry_neutral_replaces_carry_with_05(self):
        intent = _make_intent(carry=0.30, trend=0.50, expectancy=0.51,
                              router=0.50, hybrid_score=0.44)
        rec = build_diagnostic_record(
            intent=intent,
            te_result=_make_te_result(),
            fg_details=_make_fg_details(),
            fg_allowed=False,
        )
        expected_recon = 0.25 * 0.5 + 0.40 * 0.50 + 0.20 * 0.51 + 0.15 * 0.50
        assert abs(rec["carry_neutral_reconstructed"] - expected_recon) < 1e-6

    def test_carry_neutral_preserves_post_modifier(self):
        intent = _make_intent(
            carry=0.30, trend=0.50, expectancy=0.51,
            router=0.50, hybrid_score=0.40,
        )
        rec = build_diagnostic_record(
            intent=intent,
            te_result=_make_te_result(),
            fg_details=_make_fg_details(),
            fg_allowed=False,
        )
        # carry_neutral_score = carry_neutral_recon + post_modifier_delta
        recon = 0.25 * 0.30 + 0.40 * 0.50 + 0.20 * 0.51 + 0.15 * 0.50
        delta = 0.40 - recon
        cn_recon = 0.25 * 0.5 + 0.40 * 0.50 + 0.20 * 0.51 + 0.15 * 0.50
        expected_cn_score = cn_recon + delta
        assert abs(rec["carry_neutral_score"] - expected_cn_score) < 1e-6

    def test_carry_delta_positive_when_carry_penalizes(self):
        """When carry < 0.5, neutralization raises score → positive delta."""
        rec = build_diagnostic_record(
            intent=_make_intent(carry=0.20, hybrid_score=0.40),
            te_result=_make_te_result(),
            fg_details=_make_fg_details(),
            fg_allowed=False,
        )
        assert rec["carry_delta"] > 0

    def test_carry_delta_negative_when_carry_helps(self):
        """When carry > 0.5, neutralization lowers score → negative delta."""
        rec = build_diagnostic_record(
            intent=_make_intent(carry=0.80, hybrid_score=0.50),
            te_result=_make_te_result(),
            fg_details=_make_fg_details(),
            fg_allowed=False,
        )
        assert rec["carry_delta"] < 0

    def test_carry_delta_zero_when_carry_already_neutral(self):
        """When carry == 0.5, no change."""
        rec = build_diagnostic_record(
            intent=_make_intent(carry=0.50, hybrid_score=0.48),
            te_result=_make_te_result(),
            fg_details=_make_fg_details(),
            fg_allowed=False,
        )
        assert abs(rec["carry_delta"]) < 1e-6

    def test_carry_delta_magnitude_scales_with_weight(self):
        """Higher carry weight → larger delta magnitude."""
        rec_lo = build_diagnostic_record(
            intent=_make_intent(carry=0.20, w_carry=0.10, hybrid_score=0.40),
            te_result=_make_te_result(),
            fg_details=_make_fg_details(),
            fg_allowed=False,
        )
        rec_hi = build_diagnostic_record(
            intent=_make_intent(carry=0.20, w_carry=0.40, hybrid_score=0.40),
            te_result=_make_te_result(),
            fg_details=_make_fg_details(),
            fg_allowed=False,
        )
        assert abs(rec_hi["carry_delta"]) > abs(rec_lo["carry_delta"])


# ══════════════════════════════════════════════════════════════════════════
# Missing data → None
# ══════════════════════════════════════════════════════════════════════════

class TestMissingData:
    """Missing components produce partial records (edge surface only)."""

    def test_missing_components_partial_record(self):
        intent = _make_intent()
        del intent["hybrid_components"]["carry"]
        rec = build_diagnostic_record(
            intent=intent,
            te_result=_make_te_result(),
            fg_details=_make_fg_details(),
            fg_allowed=False,
        )
        assert rec is not None
        assert rec["has_decomposition"] is False
        assert rec["carry_score"] is None
        assert rec["carry_delta"] is None
        # Edge surface still available
        assert rec["confidence_gap"] == pytest.approx(0.08, abs=1e-4)

    def test_missing_weights_partial_record(self):
        intent = _make_intent()
        del intent["hybrid_weights_used"]["trend"]
        rec = build_diagnostic_record(
            intent=intent,
            te_result=_make_te_result(),
            fg_details=_make_fg_details(),
            fg_allowed=False,
        )
        assert rec is not None
        assert rec["has_decomposition"] is False
        assert rec["weights"] is None

    def test_empty_components_partial_record(self):
        intent = _make_intent()
        intent["hybrid_components"] = {}
        rec = build_diagnostic_record(
            intent=intent,
            te_result=_make_te_result(),
            fg_details=_make_fg_details(),
            fg_allowed=False,
        )
        assert rec is not None
        assert rec["has_decomposition"] is False

    def test_no_score_no_confidence_returns_none(self):
        intent = _make_intent(hybrid_score=0.0, conviction_score=0.0)
        rec = build_diagnostic_record(
            intent=intent,
            te_result=_make_te_result(),
            fg_details=_make_fg_details(),
            fg_allowed=False,
        )
        assert rec is None

    def test_hydra_only_intent_produces_edge_record(self):
        """Hydra intents without hybrid_components still get edge diagnostics."""
        intent = {
            "symbol": "SOLUSDT",
            "positionSide": "LONG",
            "hybrid_score": 0.59,
            "conviction_score": 0.44,
            "intent_id": "ord_hydra",
        }
        rec = build_diagnostic_record(
            intent=intent,
            te_result=_make_te_result(),
            fg_details=_make_fg_details(),
            fg_allowed=False,
        )
        assert rec is not None
        assert rec["has_decomposition"] is False
        assert rec["hybrid_score"] == pytest.approx(0.59, abs=1e-4)
        assert rec["confidence_gap"] == pytest.approx(0.06, abs=1e-4)
        assert rec["veto_reason"] == "fee_gate"


# ══════════════════════════════════════════════════════════════════════════
# classify_defect
# ══════════════════════════════════════════════════════════════════════════

class TestClassifyDefect:
    """Defect classification from diagnostic record batches."""

    def _make_records(self, n: int, *, confidence_gap: float = 0.08,
                      carry_delta: float = 0.005,
                      expected_edge_pct: float = 0.0) -> list:
        return [
            {
                "confidence_gap": confidence_gap,
                "carry_delta": carry_delta,
                "expected_edge_pct": expected_edge_pct,
            }
            for _ in range(n)
        ]

    def test_insufficient_data(self):
        result = classify_defect([{"confidence_gap": 0.08}])
        assert result["verdict"] == "INSUFFICIENT_DATA"

    def test_class_b_confidence_below(self):
        """All records have confidence < 0.5, carry not material → Class B."""
        records = self._make_records(10, confidence_gap=0.08, carry_delta=0.002)
        result = classify_defect(records)
        assert result["verdict"] == "CLASS_B"

    def test_class_a_carry_material(self):
        """Confidence above threshold, but carry impact is material → Class A."""
        records = self._make_records(
            10, confidence_gap=-0.05, carry_delta=0.05, expected_edge_pct=0.001,
        )
        result = classify_defect(records)
        assert result["verdict"] == "CLASS_A"

    def test_class_c_dual(self):
        """Both confidence below AND carry material → Class C."""
        records = self._make_records(10, confidence_gap=0.08, carry_delta=0.05)
        result = classify_defect(records)
        assert result["verdict"] == "CLASS_C"

    def test_inconclusive(self):
        """Neither condition met → INCONCLUSIVE."""
        records = self._make_records(
            10, confidence_gap=-0.05, carry_delta=0.002, expected_edge_pct=0.001,
        )
        result = classify_defect(records)
        assert result["verdict"] == "INCONCLUSIVE"

    def test_evidence_fields_present(self):
        records = self._make_records(5, confidence_gap=0.08)
        result = classify_defect(records)
        ev = result["evidence"]
        assert "n_records" in ev
        assert "zero_edge_pct" in ev
        assert "confidence_below_threshold_pct" in ev
        assert "carry_material_pct" in ev
        assert "avg_confidence_gap" in ev
        assert "avg_carry_delta" in ev

    def test_evidence_n_records(self):
        records = self._make_records(7)
        result = classify_defect(records)
        assert result["evidence"]["n_records"] == 7

    def test_zero_edge_pct_correct(self):
        records = self._make_records(5, expected_edge_pct=0.0)
        records.append({"confidence_gap": -0.1, "carry_delta": 0, "expected_edge_pct": 0.003})
        result = classify_defect(records)
        assert result["evidence"]["zero_edge_pct"] == pytest.approx(5 / 6, abs=0.01)


# ══════════════════════════════════════════════════════════════════════════
# Manifest contract
# ══════════════════════════════════════════════════════════════════════════

class TestManifestContract:
    """Verify diagnostic record matches v7_manifest.json schema."""

    def test_record_keys_match_manifest(self):
        import json
        from pathlib import Path

        manifest = json.loads(Path("v7_manifest.json").read_text())
        entry = manifest.get("score_edge_diagnostic", {})
        manifest_fields = set(entry.get("fields", {}).keys())

        rec = build_diagnostic_record(
            intent=_make_intent(),
            te_result=_make_te_result(),
            fg_details=_make_fg_details(),
            fg_allowed=False,
        )
        assert rec is not None
        # Every manifest field must be in the record
        rec_keys = set(rec.keys())
        missing = manifest_fields - rec_keys
        assert not missing, f"Manifest fields missing from record: {missing}"
