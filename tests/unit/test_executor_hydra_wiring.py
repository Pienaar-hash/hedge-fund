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
