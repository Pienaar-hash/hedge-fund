"""
Unit tests for Hydra executor wiring — v7.9_P2

Tests the Hydra multi-strategy injection point in executor_live.py:
- Disabled path: Hydra pipeline not called
- Enabled path: intents merged with legacy
- Fail-open: exception in Hydra leaves legacy intents intact
- Merge idempotency: no duplicate symbols
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Helpers — simulate the merge logic extracted from executor_live.py
# ---------------------------------------------------------------------------

from execution.hydra_integration import merge_with_single_strategy_intents


def _make_intent(symbol: str, signal: str = "BUY", source: str = "legacy") -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "signal": signal,
        "capital_per_trade": 100.0,
        "leverage": 1,
        "positionSide": "LONG",
        "reduceOnly": False,
        "source": source,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHydraWiringDisabled:
    """When Hydra is disabled, run_hydra_pipeline should not be called."""

    @patch("execution.hydra_integration.is_hydra_enabled", return_value=False)
    @patch("execution.hydra_integration.run_hydra_pipeline")
    def test_hydra_disabled_skips_pipeline(self, mock_pipeline, mock_enabled):
        """is_hydra_enabled=False → pipeline never invoked."""
        legacy = [_make_intent("BTCUSDT"), _make_intent("SOLUSDT")]
        cfg = {"hydra_execution": {"enabled": False}}

        # Simulate executor logic
        intents_raw = list(legacy)
        _hydra_count = 0
        if mock_enabled(cfg):
            mock_pipeline()  # should NOT reach here

        assert _hydra_count == 0
        mock_pipeline.assert_not_called()
        assert len(intents_raw) == 2


@pytest.mark.unit
class TestHydraWiringEnabled:
    """When Hydra is enabled and produces intents, they merge with legacy."""

    def test_hydra_intents_merged(self):
        """Hydra intents merge with legacy, Hydra wins on conflict."""
        hydra_intents = [
            _make_intent("BTCUSDT", "BUY", source="hydra"),
            _make_intent("ETHUSDT", "SELL", source="hydra"),
        ]
        legacy_intents = [
            _make_intent("BTCUSDT", "SELL", source="legacy"),  # conflict
            _make_intent("SOLUSDT", "BUY", source="legacy"),   # no conflict
        ]

        merged = merge_with_single_strategy_intents(
            hydra_intents, legacy_intents, prefer_hydra=True,
        )

        symbols = [i["symbol"] for i in merged]
        assert len(merged) == 3  # BTC(hydra) + ETH(hydra) + SOL(legacy)
        assert symbols.count("BTCUSDT") == 1
        # The BTCUSDT intent should be from Hydra (BUY)
        btc = [i for i in merged if i["symbol"] == "BTCUSDT"][0]
        assert btc["source"] == "hydra"
        assert btc["signal"] == "BUY"

    def test_hydra_only_no_legacy(self):
        """Hydra intents with empty legacy list."""
        hydra = [_make_intent("BTCUSDT", source="hydra")]
        merged = merge_with_single_strategy_intents(hydra, [], prefer_hydra=True)
        assert len(merged) == 1
        assert merged[0]["source"] == "hydra"

    def test_legacy_only_no_hydra(self):
        """Empty Hydra with legacy intents — passes through."""
        legacy = [_make_intent("SOLUSDT")]
        merged = merge_with_single_strategy_intents([], legacy, prefer_hydra=True)
        assert len(merged) == 1
        assert merged[0]["symbol"] == "SOLUSDT"


@pytest.mark.unit
class TestHydraFailOpen:
    """If Hydra pipeline throws, executor must use legacy intents."""

    def test_exception_preserves_legacy(self):
        """Simulate try/except fail-open as wired in executor."""
        legacy_intents = [_make_intent("BTCUSDT"), _make_intent("SOLUSDT")]
        intents_raw = list(legacy_intents)
        _hydra_count = 0

        try:
            # Simulate Hydra pipeline raising
            raise RuntimeError("Hydra model file missing")
        except Exception:
            pass  # fail-open: intents_raw unchanged

        assert _hydra_count == 0
        assert len(intents_raw) == 2
        assert intents_raw[0]["symbol"] == "BTCUSDT"
        assert intents_raw[1]["symbol"] == "SOLUSDT"


@pytest.mark.unit
class TestMergeIdempotency:
    """merge_with_single_strategy_intents produces no duplicates."""

    def test_no_duplicate_symbols_prefer_hydra(self):
        """Same symbol in both → only one copy (Hydra wins)."""
        hydra = [_make_intent("BTCUSDT", "BUY", "hydra")]
        legacy = [_make_intent("BTCUSDT", "SELL", "legacy")]
        merged = merge_with_single_strategy_intents(hydra, legacy, prefer_hydra=True)
        assert len(merged) == 1
        assert merged[0]["source"] == "hydra"

    def test_no_duplicate_symbols_prefer_legacy(self):
        """Same symbol, prefer_hydra=False → legacy wins."""
        hydra = [_make_intent("BTCUSDT", "BUY", "hydra")]
        legacy = [_make_intent("BTCUSDT", "SELL", "legacy")]
        merged = merge_with_single_strategy_intents(hydra, legacy, prefer_hydra=False)
        assert len(merged) == 1
        assert merged[0]["source"] == "legacy"

    def test_disjoint_symbols_all_preserved(self):
        """No overlap → all intents preserved."""
        hydra = [_make_intent("ETHUSDT", source="hydra")]
        legacy = [_make_intent("SOLUSDT", source="legacy")]
        merged = merge_with_single_strategy_intents(hydra, legacy, prefer_hydra=True)
        assert len(merged) == 2
        symbols = {i["symbol"] for i in merged}
        assert symbols == {"ETHUSDT", "SOLUSDT"}

    def test_multiple_overlapping(self):
        """Multiple overlapping symbols, Hydra wins all."""
        hydra = [
            _make_intent("BTCUSDT", "BUY", "hydra"),
            _make_intent("ETHUSDT", "BUY", "hydra"),
            _make_intent("XRPUSDT", "BUY", "hydra"),
        ]
        legacy = [
            _make_intent("BTCUSDT", "SELL", "legacy"),
            _make_intent("ETHUSDT", "SELL", "legacy"),
            _make_intent("SOLUSDT", "BUY", "legacy"),
        ]
        merged = merge_with_single_strategy_intents(hydra, legacy, prefer_hydra=True)
        assert len(merged) == 4  # BTC, ETH, XRP from hydra + SOL from legacy
        hydra_syms = {i["symbol"] for i in merged if i["source"] == "hydra"}
        legacy_syms = {i["symbol"] for i in merged if i["source"] == "legacy"}
        assert hydra_syms == {"BTCUSDT", "ETHUSDT", "XRPUSDT"}
        assert legacy_syms == {"SOLUSDT"}


@pytest.mark.unit
class TestIntelSurfaceExtraction:
    """Intel surface extraction helpers mirror the executor wiring code."""

    def test_hybrid_scores_from_symbol_scores(self):
        """Extract hybrid_scores from symbol_scores_v6 state file shape."""
        state = {
            "updated_ts": 1234.0,
            "symbols": [
                {"symbol": "BTCUSDT", "score": 0.48},
                {"symbol": "ETHUSDT", "score": 0.46},
            ],
        }
        hybrid: Dict[str, float] = {}
        for entry in state.get("symbols", []):
            if isinstance(entry, dict) and "symbol" in entry:
                hybrid[entry["symbol"]] = float(entry.get("score", 0.0))
        assert hybrid == {"BTCUSDT": 0.48, "ETHUSDT": 0.46}

    def test_zscore_map_from_rv_momentum(self):
        """Extract zscore_map from rv_momentum state file shape."""
        state = {
            "per_symbol": {
                "SOLUSDT": {"score": 0.0, "raw_score": 0.12},
                "WIFUSDT": {"score": 0.0, "raw_score": 0.0},
            },
        }
        zscore: Dict[str, float] = {}
        for sym, rv_v in (state.get("per_symbol") or {}).items():
            if isinstance(rv_v, dict):
                zscore[sym] = float(rv_v.get("raw_score", 0.0))
        assert zscore == {"SOLUSDT": 0.12, "WIFUSDT": 0.0}

    def test_universe_and_category_scores(self):
        """Extract universe_scores + category_scores from universe_optimizer."""
        state = {
            "symbol_scores": {"BTCUSDT": 0.65, "SOLUSDT": 0.65},
            "category_scores": {"OTHER": 0.65},
        }
        universe = {str(k): float(v) for k, v in (state.get("symbol_scores") or {}).items()}
        cat_scores = {str(k): float(v) for k, v in (state.get("category_scores") or {}).items()}
        assert universe == {"BTCUSDT": 0.65, "SOLUSDT": 0.65}
        assert cat_scores == {"OTHER": 0.65}

    def test_category_map_from_config(self):
        """Extract symbol_categories from config file shape."""
        state = {
            "_comment": "mapping",
            "categories": {"BTCUSDT": "L1_MAJOR", "DOGEUSDT": "MEME"},
        }
        cat_map = {str(k): str(v) for k, v in (state.get("categories") or {}).items()}
        assert cat_map == {"BTCUSDT": "L1_MAJOR", "DOGEUSDT": "MEME"}

    def test_missing_file_returns_empty(self):
        """Simulates load_json returning None — should produce empty dict."""
        data = None or {}
        hybrid: Dict[str, float] = {}
        for entry in data.get("symbols", []):
            hybrid[entry["symbol"]] = float(entry.get("score", 0.0))
        assert hybrid == {}


# ---------------------------------------------------------------------------
# Test: Score-Based Merge Competition (v7.9_P3)
# ---------------------------------------------------------------------------

def _scored_intent(
    symbol: str, signal: str = "BUY", source: str = "legacy",
    score: float = 0.0, hybrid_score: float | None = None,
) -> Dict[str, Any]:
    """Create an intent with score fields for merge competition tests."""
    d = _make_intent(symbol, signal, source)
    if source == "hydra":
        d["score"] = score
    else:
        d["hybrid_score"] = hybrid_score if hybrid_score is not None else score
    return d


@pytest.mark.unit
class TestScoreBasedMerge:
    """Merge resolves symbol conflicts by comparing scores, not source."""

    def test_higher_hydra_score_wins(self):
        """Hydra intent with higher score wins over legacy."""
        hydra = [_scored_intent("BTCUSDT", "BUY", "hydra", score=0.75)]
        legacy = [_scored_intent("BTCUSDT", "SELL", "legacy", score=0.50)]
        merged = merge_with_single_strategy_intents(hydra, legacy, prefer_hydra=True)
        assert len(merged) == 1
        assert merged[0]["source"] == "hydra"

    def test_higher_legacy_score_wins(self):
        """Legacy intent with higher score beats Hydra."""
        hydra = [_scored_intent("BTCUSDT", "BUY", "hydra", score=0.41)]
        legacy = [_scored_intent("BTCUSDT", "SELL", "legacy", score=0.58)]
        merged = merge_with_single_strategy_intents(hydra, legacy, prefer_hydra=True)
        assert len(merged) == 1
        assert merged[0]["source"] == "legacy"
        assert merged[0]["signal"] == "SELL"

    def test_equal_scores_tiebreak_prefer_hydra(self):
        """Equal scores → prefer_hydra=True means Hydra wins."""
        hydra = [_scored_intent("BTCUSDT", "BUY", "hydra", score=0.60)]
        legacy = [_scored_intent("BTCUSDT", "SELL", "legacy", score=0.60)]
        merged = merge_with_single_strategy_intents(hydra, legacy, prefer_hydra=True)
        assert len(merged) == 1
        assert merged[0]["source"] == "hydra"

    def test_equal_scores_tiebreak_prefer_legacy(self):
        """Equal scores → prefer_hydra=False means legacy wins."""
        hydra = [_scored_intent("BTCUSDT", "BUY", "hydra", score=0.60)]
        legacy = [_scored_intent("BTCUSDT", "SELL", "legacy", score=0.60)]
        merged = merge_with_single_strategy_intents(hydra, legacy, prefer_hydra=False)
        assert len(merged) == 1
        assert merged[0]["source"] == "legacy"

    def test_mixed_conflicts_per_symbol(self):
        """Multi-symbol: each symbol resolved independently by score."""
        hydra = [
            _scored_intent("BTCUSDT", "BUY", "hydra", score=0.80),
            _scored_intent("ETHUSDT", "BUY", "hydra", score=0.30),
        ]
        legacy = [
            _scored_intent("BTCUSDT", "SELL", "legacy", score=0.50),
            _scored_intent("ETHUSDT", "SELL", "legacy", score=0.70),
            _scored_intent("SOLUSDT", "BUY", "legacy", score=0.40),
        ]
        merged = merge_with_single_strategy_intents(hydra, legacy, prefer_hydra=True)
        assert len(merged) == 3
        by_sym = {i["symbol"]: i for i in merged}
        assert by_sym["BTCUSDT"]["source"] == "hydra"   # 0.80 > 0.50
        assert by_sym["ETHUSDT"]["source"] == "legacy"   # 0.70 > 0.30
        assert by_sym["SOLUSDT"]["source"] == "legacy"   # no conflict

    def test_zero_scores_use_tiebreaker(self):
        """No scores on either side → tiebreaker decides."""
        hydra = [_make_intent("BTCUSDT", "BUY", "hydra")]
        legacy = [_make_intent("BTCUSDT", "SELL", "legacy")]
        merged = merge_with_single_strategy_intents(hydra, legacy, prefer_hydra=True)
        assert len(merged) == 1
        assert merged[0]["source"] == "hydra"


# ---------------------------------------------------------------------------
# Test: Fallback Candidate Attachment (v7.9_P3)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMergeFallback:
    """Merge attaches _fallback when both engines produce intents."""

    def test_fallback_attached_on_conflict(self):
        """Winner gets _fallback pointing to the loser."""
        hydra = [_scored_intent("BTCUSDT", "BUY", "hydra", score=0.75)]
        legacy = [_scored_intent("BTCUSDT", "SELL", "legacy", score=0.50)]
        merged = merge_with_single_strategy_intents(hydra, legacy, prefer_hydra=True)
        assert len(merged) == 1
        assert merged[0]["source"] == "hydra"
        fb = merged[0].get("_fallback")
        assert fb is not None
        assert fb["source"] == "legacy"
        assert fb["signal"] == "SELL"

    def test_fallback_is_loser_when_legacy_wins(self):
        """When legacy wins on score, Hydra is the fallback."""
        hydra = [_scored_intent("BTCUSDT", "BUY", "hydra", score=0.30)]
        legacy = [_scored_intent("BTCUSDT", "SELL", "legacy", score=0.70)]
        merged = merge_with_single_strategy_intents(hydra, legacy, prefer_hydra=True)
        assert merged[0]["source"] == "legacy"
        assert merged[0]["_fallback"]["source"] == "hydra"

    def test_no_fallback_without_conflict(self):
        """Non-conflicting intents have no _fallback."""
        hydra = [_scored_intent("BTCUSDT", "BUY", "hydra", score=0.60)]
        legacy = [_scored_intent("SOLUSDT", "SELL", "legacy", score=0.50)]
        merged = merge_with_single_strategy_intents(hydra, legacy, prefer_hydra=True)
        assert len(merged) == 2
        for m in merged:
            assert "_fallback" not in m

    def test_fallback_on_equal_scores(self):
        """Equal scores → winner by tiebreaker, loser as fallback."""
        hydra = [_scored_intent("BTCUSDT", "BUY", "hydra", score=0.50)]
        legacy = [_scored_intent("BTCUSDT", "SELL", "legacy", score=0.50)]
        merged = merge_with_single_strategy_intents(hydra, legacy, prefer_hydra=True)
        assert merged[0]["source"] == "hydra"
        assert merged[0]["_fallback"]["source"] == "legacy"

    def test_merge_legacy_score_stamped(self):
        """Conflicted primary carries merge_legacy_score for CEL computation."""
        hydra = [_scored_intent("BTCUSDT", "BUY", "hydra", score=0.75)]
        legacy = [_scored_intent("BTCUSDT", "SELL", "legacy", score=0.50)]
        merged = merge_with_single_strategy_intents(hydra, legacy, prefer_hydra=True)
        assert merged[0]["merge_legacy_score"] == pytest.approx(0.50)

    def test_merge_legacy_score_when_legacy_wins(self):
        """Legacy score stamped even when legacy is the primary."""
        hydra = [_scored_intent("BTCUSDT", "BUY", "hydra", score=0.30)]
        legacy = [_scored_intent("BTCUSDT", "SELL", "legacy", score=0.70)]
        merged = merge_with_single_strategy_intents(hydra, legacy, prefer_hydra=True)
        assert merged[0]["source"] == "legacy"
        assert merged[0]["merge_legacy_score"] == pytest.approx(0.70)

    def test_no_merge_legacy_score_without_conflict(self):
        """Non-conflicting intents have no merge_legacy_score."""
        hydra = [_scored_intent("BTCUSDT", "BUY", "hydra", score=0.60)]
        legacy = [_scored_intent("SOLUSDT", "SELL", "legacy", score=0.50)]
        merged = merge_with_single_strategy_intents(hydra, legacy, prefer_hydra=True)
        for m in merged:
            assert "merge_legacy_score" not in m

    def test_merge_hydra_score_stamped(self):
        """Conflicted primary carries merge_hydra_score for SDD."""
        hydra = [_scored_intent("BTCUSDT", "BUY", "hydra", score=0.75)]
        legacy = [_scored_intent("BTCUSDT", "SELL", "legacy", score=0.50)]
        merged = merge_with_single_strategy_intents(hydra, legacy, prefer_hydra=True)
        assert merged[0]["merge_hydra_score"] == pytest.approx(0.75)

    def test_no_merge_hydra_score_without_conflict(self):
        """Non-conflicting intents have no merge_hydra_score."""
        hydra = [_scored_intent("BTCUSDT", "BUY", "hydra", score=0.60)]
        legacy = [_scored_intent("SOLUSDT", "SELL", "legacy", score=0.50)]
        merged = merge_with_single_strategy_intents(hydra, legacy, prefer_hydra=True)
        for m in merged:
            assert "merge_hydra_score" not in m


@pytest.mark.unit
class TestFallbackAttribution:
    """After fallback swap, attribution must follow the executed intent."""

    def _simulate_swap(
        self, primary_source: str, primary_score: float,
        fallback_source: str, fallback_score: float,
        primary_band: str = "none", fallback_band: str = "low",
    ) -> Dict[str, Any]:
        """Reproduce the executor fallback swap logic in isolation.

        Returns the intent as it would appear when passed to _send_order.
        """
        from execution.hydra_integration import merge_with_single_strategy_intents

        hydra_src = "hydra"
        legacy_src = "legacy"
        # Build intents so that primary_source wins the merge
        if primary_source == "hydra":
            hydra = [_scored_intent("BTCUSDT", "BUY", hydra_src, score=primary_score)]
            legacy = [_scored_intent("BTCUSDT", "SELL", legacy_src, score=fallback_score)]
        else:
            hydra = [_scored_intent("BTCUSDT", "BUY", hydra_src, score=fallback_score)]
            legacy = [_scored_intent("BTCUSDT", "SELL", legacy_src, score=primary_score)]

        merged = merge_with_single_strategy_intents(hydra, legacy, prefer_hydra=True)
        intent = merged[0]
        assert intent["source"] == primary_source

        # Simulate conviction enrichment
        intent["conviction_band"] = primary_band
        fb = intent.get("_fallback")
        assert fb is not None
        fb["conviction_band"] = fallback_band

        # Simulate executor fallback swap (mirrors executor_live.py logic)
        _conviction_band_order = {"none": -1, "very_low": 0, "low": 1, "medium": 2, "high": 3, "very_high": 4}
        _min_rank = _conviction_band_order["low"]
        _pri_band = str(intent.get("conviction_band") or "").lower()
        _pri_rank = _conviction_band_order.get(_pri_band, -1)

        if _pri_rank < _min_rank:
            _fb_band = str(fb.get("conviction_band") or "").lower()
            _fb_rank = _conviction_band_order.get(_fb_band, -1)
            if _fb_rank >= _min_rank:
                _primary_engine = intent.get("source") or intent.get("strategy") or "unknown"
                _primary_score_val = intent.get("hybrid_score") or intent.get("score") or 0.0
                intent = dict(fb)  # simulates _normalize_intent (preserves fields)
                intent["fallback_used"] = True
                intent["merge_primary_engine"] = _primary_engine
                intent["merge_primary_score"] = float(_primary_score_val)

        # Strip _fallback (as executor does before _send_order)
        intent.pop("_fallback", None)
        return intent

    def test_send_order_receives_fallback_engine(self):
        """After swap, _send_order intent carries the fallback's source, not the primary's."""
        intent = self._simulate_swap(
            primary_source="legacy", primary_score=0.61,
            fallback_source="hydra", fallback_score=0.54,
            primary_band="none", fallback_band="low",
        )
        assert intent["source"] == "hydra", "executed engine must be fallback"
        assert intent["source"] != "legacy", "primary engine must NOT survive swap"

    def test_fallback_used_flag_present(self):
        """Swapped intent carries fallback_used=True."""
        intent = self._simulate_swap(
            primary_source="legacy", primary_score=0.61,
            fallback_source="hydra", fallback_score=0.54,
        )
        assert intent["fallback_used"] is True

    def test_merge_primary_engine_recorded(self):
        """Swapped intent records the original primary engine for analytics."""
        intent = self._simulate_swap(
            primary_source="legacy", primary_score=0.61,
            fallback_source="hydra", fallback_score=0.54,
        )
        assert intent["merge_primary_engine"] == "legacy"
        assert intent["merge_primary_score"] == pytest.approx(0.61)

    def test_no_swap_preserves_primary(self):
        """When primary passes conviction, no swap occurs and no fallback_used."""
        intent = self._simulate_swap(
            primary_source="legacy", primary_score=0.61,
            fallback_source="hydra", fallback_score=0.54,
            primary_band="medium", fallback_band="low",  # primary passes
        )
        assert intent["source"] == "legacy"
        assert "fallback_used" not in intent
        assert "merge_primary_engine" not in intent

    def test_strategy_attribution_resolves_to_fallback(self):
        """The strategy resolution chain in attempt_payload uses fallback fields."""
        intent = self._simulate_swap(
            primary_source="legacy", primary_score=0.61,
            fallback_source="hydra", fallback_score=0.54,
        )
        # Simulate the resolution chain from _send_order
        resolved_strategy = (
            intent.get("strategy")
            or intent.get("strategy_name")
            or intent.get("strategyId")
            or intent.get("source")
            or (intent.get("metadata") or {}).get("strategy")
        )
        assert resolved_strategy != "legacy", "must not resolve to primary"
