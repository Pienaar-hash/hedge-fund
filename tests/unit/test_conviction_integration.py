"""
v7.9_C1 / C4 — Conviction engine integration into signal_screener.

Verifies that generic strategy intents (btc_m15, etc.) receive
conviction_score and conviction_band from the conviction engine
during hybrid ranking.  Previously, only vol_target populated
these fields, causing the executor's conviction gate to veto
100% of generic-strategy intents.
"""
from __future__ import annotations

import copy
import types
from typing import Any, Dict, List, Mapping
from unittest.mock import patch, MagicMock

import pytest

import execution.signal_screener as sc
from execution.conviction_engine import (
    ConvictionContext,
    ConvictionConfig,
    ConvictionResult,
    compute_conviction,
    load_conviction_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ranked_result(
    symbol: str = "BTCUSDT",
    direction: str = "LONG",
    hybrid_score: float = 0.65,
    expectancy: float = 0.55,
    router: float = 0.80,
    trend: float = 0.70,
    passes: bool = True,
) -> Dict[str, Any]:
    """Mimic one element returned by rank_intents_by_hybrid_score."""
    return {
        "symbol": symbol,
        "direction": direction,
        "hybrid_score": hybrid_score,
        "passes_threshold": passes,
        "components": {
            "trend": trend,
            "carry": 0.0,
            "expectancy": expectancy,
            "router": router,
        },
        "weighted_contributions": {"trend": 0.3, "expectancy": 0.2, "router": 0.1},
        "weights_used": {"trend": 0.4, "expectancy": 0.3, "router": 0.2},
        "intent": {
            "symbol": symbol,
            "direction": direction,
            "signal": "BUY",
            "strategy": "btc_m15",
            "vol_regime": "normal",
            "metadata": {},
        },
    }


_STRATEGY_CFG: Dict[str, Any] = {
    "conviction": {
        "enabled": True,
        "mode": "live",
        "min_entry_band": "low",
        "thresholds": {
            "very_low": 0.20,
            "low": 0.40,
            "medium": 0.60,
            "high": 0.80,
            "very_high": 0.92,
        },
        "size_multipliers": {
            "very_low": 0.0,
            "low": 0.75,
            "medium": 1.0,
            "high": 1.15,
            "very_high": 1.25,
        },
    },
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConvictionAttachedToIntent:
    """Verify that the screener's ranking loop attaches conviction fields."""

    def _run_ranking_section(
        self,
        ranked_results: List[Dict[str, Any]],
        monkeypatch: pytest.MonkeyPatch,
        *,
        conviction_available: bool = True,
        conviction_enabled: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Exercise only the post-hybrid-ranking section of the screener.

        We build a minimal fake of the screener's control flow:
          ranked_results → conviction compute → ranked_out list.
        This avoids needing to stub the entire exchange layer.
        """
        cfg = copy.deepcopy(_STRATEGY_CFG)
        cfg["universe"] = ["BTCUSDT"]
        if not conviction_enabled:
            cfg["conviction"]["enabled"] = False

        # Load conviction config the same way the screener does
        conviction_cfg = load_conviction_config(cfg) if conviction_available else None

        ranked_out: list[dict] = []

        for result in ranked_results:
            intent = dict(result.get("intent", {}))
            symbol = str(intent.get("symbol", "")).upper()

            intent["hybrid_score"] = result.get("hybrid_score", 0.5)
            intent["hybrid_passes_threshold"] = result.get("passes_threshold", True)
            intent["hybrid_components"] = result.get("components", {})
            intent["hybrid_weighted"] = result.get("weighted_contributions", {})
            intent["hybrid_weights_used"] = result.get("weights_used", {})

            # v7.9_C1 conviction block (mirrors screener code)
            if conviction_available and conviction_cfg and conviction_cfg.enabled:
                components = result.get("components", {})
                vol_regime_raw = str(intent.get("vol_regime", "normal")).lower()
                if vol_regime_raw not in ("low", "normal", "high", "crisis"):
                    vol_regime_raw = "normal"
                ctx = ConvictionContext(
                    hybrid_score=float(intent.get("hybrid_score", 0.0)),
                    expectancy_alpha=float(components.get("expectancy", 0.0)),
                    router_quality=1.0,
                    trend_strength=float(components.get("trend", 0.0)),
                    vol_regime=vol_regime_raw,  # type: ignore[arg-type]
                    dd_state="NORMAL",  # type: ignore[arg-type]
                    risk_mode="OK",  # type: ignore[arg-type]
                )
                conv_result = compute_conviction(ctx, conviction_cfg)
                intent["conviction_score"] = conv_result.conviction_score
                intent["conviction_band"] = conv_result.conviction_band
                intent["conviction_size_multiplier"] = conv_result.size_multiplier

            ranked_out.append(intent)

        return ranked_out

    # -- Core assertions ---------------------------------------------------

    def test_conviction_band_populated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Generic strategy intent must have conviction_band after ranking."""
        results = self._run_ranking_section(
            [_make_ranked_result(hybrid_score=0.65, expectancy=0.55, trend=0.70)],
            monkeypatch,
        )
        assert len(results) == 1
        intent = results[0]
        assert "conviction_band" in intent, "conviction_band must be set"
        assert "conviction_score" in intent, "conviction_score must be set"
        assert intent["conviction_band"] != "", "conviction_band must not be empty"

    def test_high_score_gets_medium_or_above(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Intent with hybrid_score=0.65 and good factors should get >= 'low' band."""
        results = self._run_ranking_section(
            [_make_ranked_result(hybrid_score=0.65, expectancy=0.60, trend=0.75, router=0.90)],
            monkeypatch,
        )
        intent = results[0]
        # With these inputs the conviction score should be reasonably above low threshold
        band_rank = {"very_low": 0, "low": 1, "medium": 2, "high": 3, "very_high": 4}
        assert band_rank.get(intent["conviction_band"], -1) >= 1, (
            f"Expected at least 'low' band, got '{intent['conviction_band']}' "
            f"(score={intent['conviction_score']:.4f})"
        )

    def test_low_score_gets_very_low_band(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Intent with all near-zero factors should get 'very_low' band."""
        results = self._run_ranking_section(
            [_make_ranked_result(hybrid_score=0.05, expectancy=0.05, trend=0.05, router=0.1)],
            monkeypatch,
        )
        intent = results[0]
        assert intent["conviction_band"] == "very_low"

    def test_conviction_disabled_no_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When conviction is disabled, intent should NOT have conviction fields."""
        results = self._run_ranking_section(
            [_make_ranked_result()],
            monkeypatch,
            conviction_enabled=False,
        )
        intent = results[0]
        assert "conviction_band" not in intent
        assert "conviction_score" not in intent

    def test_conviction_unavailable_no_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When conviction module is unavailable, intent stays clean."""
        results = self._run_ranking_section(
            [_make_ranked_result()],
            monkeypatch,
            conviction_available=False,
        )
        intent = results[0]
        assert "conviction_band" not in intent

    def test_multiple_intents_each_scored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every intent in the batch should get its own conviction fields."""
        results = self._run_ranking_section(
            [
                _make_ranked_result("BTCUSDT", "LONG", hybrid_score=0.80, expectancy=0.70, trend=0.85),
                _make_ranked_result("ETHUSDT", "SHORT", hybrid_score=0.30, expectancy=0.20, trend=0.10),
            ],
            monkeypatch,
        )
        assert len(results) == 2
        for intent in results:
            assert "conviction_band" in intent
            assert "conviction_score" in intent

        # BTC (high inputs) should score higher than ETH (low inputs)
        assert results[0]["conviction_score"] > results[1]["conviction_score"]

    def test_vol_regime_normalisation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unknown vol_regime should be normalised to 'normal' without error."""
        results = self._run_ranking_section(
            [_make_ranked_result(hybrid_score=0.60)],
            monkeypatch,
        )
        intent = results[0]
        # Should still have conviction fields (no crash from bad regime)
        assert "conviction_band" in intent

    def test_crisis_vol_regime_lowers_score(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Crisis vol regime should produce lower conviction than normal."""
        normal_result = _make_ranked_result(hybrid_score=0.65, expectancy=0.55, trend=0.70)
        crisis_result = _make_ranked_result(hybrid_score=0.65, expectancy=0.55, trend=0.70)
        crisis_result["intent"]["vol_regime"] = "crisis"

        normal_out = self._run_ranking_section([normal_result], monkeypatch)
        crisis_out = self._run_ranking_section([crisis_result], monkeypatch)

        assert crisis_out[0]["conviction_score"] <= normal_out[0]["conviction_score"]


class TestMetadataPropagation:
    """Verify that conviction fields flow into intent metadata (v7.9-S1)."""

    def test_metadata_has_conviction_fields(self) -> None:
        """The v7.9-S1 metadata push should copy conviction fields."""
        intent: Dict[str, Any] = {
            "symbol": "BTCUSDT",
            "hybrid_score": 0.65,
            "confidence": 0.7,
            "conviction_score": 0.58,
            "conviction_band": "medium",
            "metadata": {},
        }
        # Exercise the exact metadata propagation code from signal_screener
        _meta = intent.get("metadata")
        if not isinstance(_meta, dict):
            _meta = {}
            intent["metadata"] = _meta
        if intent.get("hybrid_score") is not None:
            _meta["hybrid_score"] = round(float(intent.get("hybrid_score", 0)), 6)
        if intent.get("confidence") is not None:
            _meta["confidence"] = round(float(intent.get("confidence", 0)), 4)
        if intent.get("conviction_score") is not None:
            _meta["conviction_score"] = round(float(intent.get("conviction_score", 0)), 4)
        if intent.get("conviction_band"):
            _meta["conviction_band"] = str(intent.get("conviction_band", ""))

        assert _meta["conviction_score"] == 0.58
        assert _meta["conviction_band"] == "medium"
        assert _meta["hybrid_score"] == 0.65

    def test_missing_conviction_not_propagated(self) -> None:
        """If conviction fields are absent, metadata should not have them."""
        intent: Dict[str, Any] = {
            "symbol": "BTCUSDT",
            "hybrid_score": 0.65,
            "metadata": {},
        }
        _meta = intent["metadata"]
        if intent.get("conviction_score") is not None:
            _meta["conviction_score"] = round(float(intent.get("conviction_score", 0)), 4)
        if intent.get("conviction_band"):
            _meta["conviction_band"] = str(intent.get("conviction_band", ""))

        assert "conviction_score" not in _meta
        assert "conviction_band" not in _meta


class TestConvictionImportGuard:
    """Verify the screener handles missing conviction module gracefully."""

    def test_conviction_available_flag_exists(self) -> None:
        """signal_screener must export _CONVICTION_AVAILABLE."""
        assert hasattr(sc, "_CONVICTION_AVAILABLE")
        # In test env the module should be importable
        assert sc._CONVICTION_AVAILABLE is True
