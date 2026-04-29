"""Tests for execution.intent_factor_log — per-episode factor decomposition logging."""

from __future__ import annotations

import json
from datetime import datetime

from execution.intent_factor_log import (
    build_factor_log_record,
    REQUIRED_COMPONENTS,
    SCORE_VERSION,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _full_intent(**overrides) -> dict:
    """Return a fully-populated intent dict with all required fields."""
    base = {
        "intent_id": "ord_abc123",
        "symbol": "BTCUSDT",
        "positionSide": "LONG",
        "hybrid_score": 0.72,
        "hybrid_components": {
            "carry": 0.3,
            "trend": 0.8,
            "expectancy": 0.6,
            "router": 0.5,
        },
        "hybrid_weights_used": {
            "carry": 0.25,
            "trend": 0.40,
            "expectancy": 0.20,
            "router": 0.15,
        },
        "hybrid_carry_details": {
            "inputs": {
                "funding_rate": -0.0003,
                "basis_pct": 0.012,
            }
        },
        "conviction_band": "high",
        "conviction_score": 0.85,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# TestBuildFactorLogRecord
# ---------------------------------------------------------------------------

class TestBuildFactorLogRecord:
    """Core build logic."""

    def test_all_required_fields_present(self):
        rec = build_factor_log_record(_full_intent())
        assert rec is not None
        expected_keys = {
            "ts", "intent_id", "symbol", "side",
            "hybrid_score", "hybrid_score_reconstructed",
            "carry_score", "trend_score", "expectancy_score", "router_score",
            "weights", "funding_rate", "basis_pct",
            "conviction_band", "confidence", "score_version",
        }
        assert expected_keys == set(rec.keys())

    def test_intent_id_preserved(self):
        rec = build_factor_log_record(_full_intent(intent_id="ord_xyz789"))
        assert rec["intent_id"] == "ord_xyz789"

    def test_symbol_and_side(self):
        rec = build_factor_log_record(_full_intent(symbol="ETHUSDT", positionSide="SHORT"))
        assert rec["symbol"] == "ETHUSDT"
        assert rec["side"] == "SHORT"

    def test_side_fallback_to_side_key(self):
        intent = _full_intent()
        del intent["positionSide"]
        intent["side"] = "SHORT"
        rec = build_factor_log_record(intent)
        assert rec["side"] == "SHORT"

    def test_component_scores_match(self):
        components = {"carry": 0.1, "trend": 0.9, "expectancy": 0.5, "router": 0.7}
        rec = build_factor_log_record(_full_intent(hybrid_components=components))
        assert rec["carry_score"] == 0.1
        assert rec["trend_score"] == 0.9
        assert rec["expectancy_score"] == 0.5
        assert rec["router_score"] == 0.7

    def test_weights_extracted(self):
        weights = {"carry": 0.30, "trend": 0.35, "expectancy": 0.20, "router": 0.15}
        rec = build_factor_log_record(_full_intent(hybrid_weights_used=weights))
        assert rec["weights"] == {
            "carry": 0.30, "trend": 0.35, "expectancy": 0.20, "router": 0.15,
        }

    def test_score_version(self):
        rec = build_factor_log_record(_full_intent())
        assert rec["score_version"] == SCORE_VERSION

    def test_ts_is_iso8601(self):
        rec = build_factor_log_record(_full_intent())
        # Should parse without error
        dt = datetime.fromisoformat(rec["ts"])
        assert dt.tzinfo is not None  # timezone-aware

    def test_record_is_json_serializable(self):
        rec = build_factor_log_record(_full_intent())
        s = json.dumps(rec)
        assert isinstance(s, str)


# ---------------------------------------------------------------------------
# TestReconstructedScore
# ---------------------------------------------------------------------------

class TestReconstructedScore:
    """Enhancement 1: hybrid_score_reconstructed validation."""

    def test_reconstructed_equals_dot_product(self):
        components = {"carry": 0.3, "trend": 0.8, "expectancy": 0.6, "router": 0.5}
        weights = {"carry": 0.25, "trend": 0.40, "expectancy": 0.20, "router": 0.15}
        expected = (0.25 * 0.3) + (0.40 * 0.8) + (0.20 * 0.6) + (0.15 * 0.5)
        rec = build_factor_log_record(_full_intent(
            hybrid_components=components,
            hybrid_weights_used=weights,
        ))
        assert abs(rec["hybrid_score_reconstructed"] - expected) < 1e-6

    def test_reconstructed_with_zero_components(self):
        components = {"carry": 0.0, "trend": 0.0, "expectancy": 0.0, "router": 0.0}
        rec = build_factor_log_record(_full_intent(hybrid_components=components))
        assert rec["hybrid_score_reconstructed"] == 0.0

    def test_reconstructed_with_unit_weights(self):
        components = {"carry": 1.0, "trend": 1.0, "expectancy": 1.0, "router": 1.0}
        weights = {"carry": 0.25, "trend": 0.25, "expectancy": 0.25, "router": 0.25}
        rec = build_factor_log_record(_full_intent(
            hybrid_components=components,
            hybrid_weights_used=weights,
        ))
        assert abs(rec["hybrid_score_reconstructed"] - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# TestCarryDetailsExtraction
# ---------------------------------------------------------------------------

class TestCarryDetailsExtraction:
    """Carry inputs (funding_rate, basis_pct) from carry_details."""

    def test_funding_rate_extracted(self):
        rec = build_factor_log_record(_full_intent())
        assert rec["funding_rate"] == -0.0003

    def test_basis_pct_extracted(self):
        rec = build_factor_log_record(_full_intent())
        assert rec["basis_pct"] == 0.012

    def test_missing_carry_details_defaults_to_zero(self):
        intent = _full_intent()
        del intent["hybrid_carry_details"]
        rec = build_factor_log_record(intent)
        assert rec["funding_rate"] == 0.0
        assert rec["basis_pct"] == 0.0

    def test_empty_carry_inputs_defaults_to_zero(self):
        rec = build_factor_log_record(_full_intent(
            hybrid_carry_details={"inputs": {}}
        ))
        assert rec["funding_rate"] == 0.0
        assert rec["basis_pct"] == 0.0

    def test_none_carry_details_defaults_to_zero(self):
        rec = build_factor_log_record(_full_intent(hybrid_carry_details=None))
        assert rec["funding_rate"] == 0.0
        assert rec["basis_pct"] == 0.0


# ---------------------------------------------------------------------------
# TestSkipUnscored
# ---------------------------------------------------------------------------

class TestSkipUnscored:
    """Returns None for incomplete / unscored intents."""

    def test_no_intent_id(self):
        intent = _full_intent()
        del intent["intent_id"]
        assert build_factor_log_record(intent) is None

    def test_empty_intent_id(self):
        assert build_factor_log_record(_full_intent(intent_id="")) is None

    def test_no_hybrid_components(self):
        intent = _full_intent()
        del intent["hybrid_components"]
        assert build_factor_log_record(intent) is None

    def test_empty_hybrid_components(self):
        assert build_factor_log_record(_full_intent(hybrid_components={})) is None

    def test_incomplete_components_missing_carry(self):
        components = {"trend": 0.5, "expectancy": 0.4, "router": 0.3}
        assert build_factor_log_record(_full_intent(hybrid_components=components)) is None

    def test_incomplete_components_missing_trend(self):
        components = {"carry": 0.5, "expectancy": 0.4, "router": 0.3}
        assert build_factor_log_record(_full_intent(hybrid_components=components)) is None

    def test_no_weights(self):
        intent = _full_intent()
        del intent["hybrid_weights_used"]
        assert build_factor_log_record(intent) is None

    def test_empty_weights(self):
        assert build_factor_log_record(_full_intent(hybrid_weights_used={})) is None

    def test_incomplete_weights(self):
        weights = {"carry": 0.25, "trend": 0.40}  # missing expectancy, router
        assert build_factor_log_record(_full_intent(hybrid_weights_used=weights)) is None


# ---------------------------------------------------------------------------
# TestConviction
# ---------------------------------------------------------------------------

class TestConviction:
    """Conviction band and confidence extraction."""

    def test_conviction_band(self):
        rec = build_factor_log_record(_full_intent(conviction_band="very_high"))
        assert rec["conviction_band"] == "very_high"

    def test_conviction_score_as_confidence(self):
        rec = build_factor_log_record(_full_intent(conviction_score=0.92))
        assert rec["confidence"] == 0.92

    def test_fallback_to_confidence_key(self):
        intent = _full_intent()
        del intent["conviction_score"]
        intent["confidence"] = 0.77
        rec = build_factor_log_record(intent)
        assert rec["confidence"] == 0.77

    def test_missing_conviction_defaults_zero(self):
        intent = _full_intent()
        del intent["conviction_score"]
        rec = build_factor_log_record(intent)
        assert rec["confidence"] == 0.0

    def test_missing_band_defaults_empty(self):
        intent = _full_intent()
        del intent["conviction_band"]
        rec = build_factor_log_record(intent)
        assert rec["conviction_band"] == ""


# ---------------------------------------------------------------------------
# TestRequiredComponents constant
# ---------------------------------------------------------------------------

class TestConstants:
    def test_required_components_is_four(self):
        assert len(REQUIRED_COMPONENTS) == 4
        assert REQUIRED_COMPONENTS == {"carry", "trend", "expectancy", "router"}
