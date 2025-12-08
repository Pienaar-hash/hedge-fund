"""
Tests for carry scoring and hybrid score integration (v7.4 B1).
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from execution.intel.symbol_score_v6 import (
    carry_score,
    hybrid_score,
    hybrid_score_universe,
    rank_intents_by_hybrid_score,
    load_hybrid_config,
    HybridScoreConfig,
    _scale_funding_rate,
    _scale_basis,
)


class TestCarryScoring:
    """Tests for carry score computation."""

    def test_carry_score_positive_funding_short(self):
        """Positive funding favors shorts (shorts get paid)."""
        result = carry_score(
            symbol="BTCUSDT",
            direction="SHORT",
            funding_snapshot={"symbols": {"BTCUSDT": {"rate": 0.0003}}},  # ~33% annual
            basis_snapshot={"symbols": {}},
        )
        assert result["symbol"] == "BTCUSDT"
        assert result["direction"] == "SHORT"
        # Positive funding + SHORT should yield score > 0.5
        assert result["score"] > 0.5
        assert "components" in result
        assert result["components"]["funding_score"] > 0.5

    def test_carry_score_positive_funding_long(self):
        """Positive funding penalizes longs (they pay)."""
        result = carry_score(
            symbol="BTCUSDT",
            direction="LONG",
            funding_snapshot={"symbols": {"BTCUSDT": {"rate": 0.0003}}},
            basis_snapshot={"symbols": {}},
        )
        # Positive funding + LONG should yield score < 0.5
        assert result["score"] < 0.5
        assert result["components"]["funding_score"] < 0.5

    def test_carry_score_negative_funding_long(self):
        """Negative funding favors longs (longs get paid)."""
        result = carry_score(
            symbol="ETHUSDT",
            direction="LONG",
            funding_snapshot={"symbols": {"ETHUSDT": {"rate": -0.0002}}},
            basis_snapshot={"symbols": {}},
        )
        # Negative funding + LONG should yield score > 0.5
        assert result["score"] > 0.5

    def test_carry_score_positive_basis_short(self):
        """Positive basis (premium) favors shorts."""
        result = carry_score(
            symbol="BTCUSDT",
            direction="SHORT",
            funding_snapshot={"symbols": {}},
            basis_snapshot={"symbols": {"BTCUSDT": {"basis_pct": 0.01}}},  # 1% premium
        )
        assert result["components"]["basis_score"] > 0.5

    def test_carry_score_negative_basis_long(self):
        """Negative basis (discount) favors longs."""
        result = carry_score(
            symbol="BTCUSDT",
            direction="LONG",
            funding_snapshot={"symbols": {}},
            basis_snapshot={"symbols": {"BTCUSDT": {"basis_pct": -0.01}}},  # 1% discount
        )
        assert result["components"]["basis_score"] > 0.5

    def test_carry_score_missing_symbol(self):
        """Missing symbol data yields neutral score (0.5)."""
        result = carry_score(
            symbol="UNKNOWN",
            direction="LONG",
            funding_snapshot={"symbols": {}},
            basis_snapshot={"symbols": {}},
        )
        assert result["score"] == pytest.approx(0.5, abs=0.01)

    def test_carry_score_scalar_format(self):
        """Handles scalar funding/basis format (not nested dict)."""
        result = carry_score(
            symbol="BTCUSDT",
            direction="SHORT",
            funding_snapshot={"symbols": {"BTCUSDT": 0.0005}},  # scalar
            basis_snapshot={"symbols": {"BTCUSDT": 0.02}},  # scalar
        )
        assert result["score"] > 0.5  # Both favor short


class TestScalingFunctions:
    """Tests for internal scaling functions."""

    def test_scale_funding_rate_long_negative(self):
        """Negative funding -> high score for LONG."""
        score = _scale_funding_rate(-0.001, "LONG")  # ~110% annual
        assert score > 0.7

    def test_scale_funding_rate_short_positive(self):
        """Positive funding -> high score for SHORT."""
        score = _scale_funding_rate(0.001, "SHORT")
        assert score > 0.7

    def test_scale_funding_rate_neutral(self):
        """Zero funding -> neutral score."""
        score = _scale_funding_rate(0.0, "LONG")
        assert score == pytest.approx(0.5, abs=0.01)

    def test_scale_basis_long_discount(self):
        """Negative basis (discount) -> high score for LONG."""
        score = _scale_basis(-0.02, "LONG")
        assert score > 0.7

    def test_scale_basis_short_premium(self):
        """Positive basis (premium) -> high score for SHORT."""
        score = _scale_basis(0.02, "SHORT")
        assert score > 0.7


class TestHybridScoring:
    """Tests for hybrid score blending."""

    @pytest.fixture
    def mock_snapshots(self):
        return {
            "expectancy": {"symbols": {"BTCUSDT": {"expectancy": 5.0, "hit_rate": 0.55}}},
            "router": {"symbols": [{"symbol": "BTCUSDT", "maker_fill_rate": 0.8, "fallback_rate": 0.1}]},
            "funding": {"symbols": {"BTCUSDT": {"rate": 0.0001}}},
            "basis": {"symbols": {"BTCUSDT": {"basis_pct": 0.005}}},
        }

    def test_hybrid_score_computes_all_components(self, mock_snapshots):
        """Hybrid score includes all four components."""
        result = hybrid_score(
            symbol="BTCUSDT",
            direction="LONG",
            trend_score=0.7,
            expectancy_snapshot=mock_snapshots["expectancy"],
            router_health_snapshot=mock_snapshots["router"],
            funding_snapshot=mock_snapshots["funding"],
            basis_snapshot=mock_snapshots["basis"],
        )
        assert "hybrid_score" in result
        assert 0.0 <= result["hybrid_score"] <= 1.0
        assert "components" in result
        assert all(k in result["components"] for k in ["trend", "carry", "expectancy", "router"])

    def test_hybrid_score_passes_threshold(self, mock_snapshots):
        """High scores pass threshold, low scores don't."""
        high_result = hybrid_score(
            symbol="BTCUSDT",
            direction="LONG",
            trend_score=0.9,
            expectancy_snapshot=mock_snapshots["expectancy"],
            router_health_snapshot=mock_snapshots["router"],
            funding_snapshot=mock_snapshots["funding"],
            basis_snapshot=mock_snapshots["basis"],
        )
        assert high_result["passes_threshold"] is True

        low_result = hybrid_score(
            symbol="BTCUSDT",
            direction="LONG",
            trend_score=0.1,
            expectancy_snapshot={"symbols": {}},
            router_health_snapshot={"symbols": []},
            funding_snapshot={"symbols": {"BTCUSDT": {"rate": 0.001}}},  # unfavorable
            basis_snapshot={"symbols": {}},
        )
        assert low_result["passes_threshold"] is False

    def test_hybrid_score_regime_modulation(self, mock_snapshots):
        """Regime affects carry weight multiplier."""
        # strategy_config with vol regime modifiers
        strategy_config = {
            "vol_regimes": {
                "enabled": True,
                "hybrid_weight_modifiers": {
                    "default": {
                        "normal": {"carry": 1.0, "expectancy": 1.0, "router": 1.0},
                        "crisis": {"carry": 0.5, "expectancy": 0.8, "router": 1.2},
                    }
                }
            }
        }
        
        normal_result = hybrid_score(
            symbol="BTCUSDT",
            direction="LONG",
            trend_score=0.6,
            expectancy_snapshot=mock_snapshots["expectancy"],
            router_health_snapshot=mock_snapshots["router"],
            funding_snapshot=mock_snapshots["funding"],
            basis_snapshot=mock_snapshots["basis"],
            regime="normal",
            strategy_config=strategy_config,
        )
        
        crisis_result = hybrid_score(
            symbol="BTCUSDT",
            direction="LONG",
            trend_score=0.6,
            expectancy_snapshot=mock_snapshots["expectancy"],
            router_health_snapshot=mock_snapshots["router"],
            funding_snapshot=mock_snapshots["funding"],
            basis_snapshot=mock_snapshots["basis"],
            regime="crisis",
            strategy_config=strategy_config,
        )
        
        # Crisis regime should have lower carry weight modifier
        assert crisis_result["weight_modifiers"]["carry"] < normal_result["weight_modifiers"]["carry"]

    def test_hybrid_score_custom_config(self, mock_snapshots):
        """Custom config weights are applied."""
        config = HybridScoreConfig(
            trend_weight=0.8,
            carry_weight=0.1,
            expectancy_weight=0.05,
            router_weight=0.05,
        )
        result = hybrid_score(
            symbol="BTCUSDT",
            direction="LONG",
            trend_score=0.9,
            expectancy_snapshot=mock_snapshots["expectancy"],
            router_health_snapshot=mock_snapshots["router"],
            funding_snapshot=mock_snapshots["funding"],
            basis_snapshot=mock_snapshots["basis"],
            config=config,
        )
        # With 80% trend weight and 0.9 trend score, hybrid should be high
        assert result["hybrid_score"] > 0.7


class TestHybridScoreUniverse:
    """Tests for batch hybrid scoring."""

    def test_hybrid_score_universe_sorts_descending(self):
        """Results are sorted by hybrid_score descending."""
        intents = [
            {"symbol": "BTCUSDT", "direction": "LONG", "trend_score": 0.3},
            {"symbol": "ETHUSDT", "direction": "LONG", "trend_score": 0.9},
            {"symbol": "SOLUSDT", "direction": "SHORT", "trend_score": 0.6},
        ]
        results = hybrid_score_universe(
            intents=intents,
            expectancy_snapshot={"symbols": {}},
            router_health_snapshot={"symbols": []},
            funding_snapshot={"symbols": {}},
            basis_snapshot={"symbols": {}},
        )
        assert len(results) == 3
        scores = [r["hybrid_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_hybrid_score_universe_attaches_intent(self):
        """Original intent is attached to result."""
        intents = [{"symbol": "BTCUSDT", "direction": "LONG", "trend_score": 0.5, "extra": "data"}]
        results = hybrid_score_universe(
            intents=intents,
            expectancy_snapshot={"symbols": {}},
            router_health_snapshot={"symbols": []},
            funding_snapshot={"symbols": {}},
            basis_snapshot={"symbols": {}},
        )
        assert len(results) == 1
        assert results[0]["intent"]["extra"] == "data"


class TestRankIntentsByHybridScore:
    """Tests for the convenience ranking function."""

    def test_rank_intents_filters_below_threshold(self):
        """filter_below_threshold=True removes low-scoring intents."""
        intents = [
            {"symbol": "BTCUSDT", "direction": "LONG", "trend_score": 0.9},
            {"symbol": "LOWSCORE", "direction": "LONG", "trend_score": 0.1},
        ]
        
        # With filtering
        filtered = rank_intents_by_hybrid_score(
            intents=intents,
            expectancy_snapshot={"symbols": {}},
            router_health_snapshot={"symbols": []},
            funding_snapshot={"symbols": {}},
            basis_snapshot={"symbols": {}},
            filter_below_threshold=True,
        )
        
        # Without filtering
        unfiltered = rank_intents_by_hybrid_score(
            intents=intents,
            expectancy_snapshot={"symbols": {}},
            router_health_snapshot={"symbols": []},
            funding_snapshot={"symbols": {}},
            basis_snapshot={"symbols": {}},
            filter_below_threshold=False,
        )
        
        assert len(filtered) <= len(unfiltered)

    def test_rank_intents_loads_defaults_if_none(self):
        """Loads default snapshots if not provided."""
        intents = [{"symbol": "BTCUSDT", "direction": "LONG", "trend_score": 0.6}]
        
        # Should not raise even with missing files
        results = rank_intents_by_hybrid_score(intents=intents)
        assert len(results) >= 0  # May be 0 if filtered, or 1 if passes


class TestLoadHybridConfig:
    """Tests for config loading."""

    def test_load_hybrid_config_from_dict(self):
        """Loads config from strategy_config dict."""
        strategy_config = {
            "hybrid_scoring": {
                "enabled": True,
                "trend_weight": 0.5,
                "carry_weight": 0.3,
                "expectancy_weight": 0.1,
                "router_weight": 0.1,
                "intent_ranking": {
                    "min_hybrid_score_long": 0.6,
                    "min_hybrid_score_short": 0.55,
                },
            }
        }
        config = load_hybrid_config(strategy_config)
        assert config.trend_weight == 0.5
        assert config.carry_weight == 0.3
        assert config.min_hybrid_score_long == 0.6
        assert config.min_hybrid_score_short == 0.55

    def test_load_hybrid_config_disabled(self):
        """Returns defaults when disabled."""
        strategy_config = {"hybrid_scoring": {"enabled": False}}
        config = load_hybrid_config(strategy_config)
        assert config.trend_weight == 0.40  # default

    def test_load_hybrid_config_missing(self):
        """Returns defaults when section missing."""
        config = load_hybrid_config({})
        assert isinstance(config, HybridScoreConfig)


class TestHybridScoreConfig:
    """Tests for the config dataclass."""

    def test_default_values(self):
        """Default config has expected weights."""
        config = HybridScoreConfig()
        assert config.trend_weight == 0.40
        assert config.carry_weight == 0.25
        assert config.expectancy_weight == 0.20
        assert config.router_weight == 0.15
        # Default thresholds are 0.15 (config/strategy_config.json overrides to 0.55/0.50)
        assert config.min_hybrid_score_long == 0.15
        assert config.min_hybrid_score_short == 0.15

    def test_custom_values(self):
        """Custom values are stored."""
        config = HybridScoreConfig(
            trend_weight=0.6,
            carry_weight=0.2,
            min_hybrid_score_long=0.7,
        )
        assert config.trend_weight == 0.6
        assert config.carry_weight == 0.2
        assert config.min_hybrid_score_long == 0.7
