"""
Unit tests for Strategy Health Score computation (v7.7_P7).

Tests component health calculations:
- Factor health
- Symbol health
- Category health
- Regime alignment
- Execution quality
- Aggregate health score

PATCHSET_V7.7_P7
"""

import pytest
from typing import Dict, Any

from execution.edge_scanner import (
    StrategyHealth,
    compute_strategy_health,
    _compute_factor_health,
    _compute_symbol_health,
    _compute_category_health,
    _compute_regime_alignment,
    _compute_execution_quality,
)


# ---------------------------------------------------------------------------
# Factor Health Tests
# ---------------------------------------------------------------------------


class TestComputeFactorHealth:
    """Tests for _compute_factor_health."""

    def test_empty_factor_edges(self):
        """Empty factor edges should return default quality."""
        quality, health = _compute_factor_health({})
        
        assert quality == 0.5
        assert health["factor_count"] == 0
        assert health["strength_label"] == "unknown"

    def test_strong_positive_factors(self):
        """All positive edge scores should yield strong health."""
        factor_edges = {
            "momentum": {"edge_score": 1.5, "ir": 0.4},
            "carry": {"edge_score": 1.2, "ir": 0.3},
            "value": {"edge_score": 0.8, "ir": 0.2},
        }
        
        quality, health = _compute_factor_health(factor_edges)
        
        assert quality > 0.7
        assert health["strength_label"] == "strong"
        assert health["pct_negative"] == 0.0
        assert health["factor_count"] == 3

    def test_mixed_factors(self):
        """Mix of positive and negative should yield mixed health."""
        factor_edges = {
            "momentum": {"edge_score": 1.0, "ir": 0.3},
            "carry": {"edge_score": -0.5, "ir": -0.1},
            "value": {"edge_score": 0.3, "ir": 0.1},
            "volatility": {"edge_score": -0.2, "ir": -0.05},
        }
        
        quality, health = _compute_factor_health(factor_edges)
        
        assert 0.3 <= quality <= 0.7
        assert health["strength_label"] == "mixed"
        assert health["pct_negative"] == 0.5  # 2 of 4 negative

    def test_weak_factors_penalty(self):
        """Majority negative factors should be penalized."""
        factor_edges = {
            "momentum": {"edge_score": -0.8, "ir": -0.2},
            "carry": {"edge_score": -1.0, "ir": -0.3},
            "value": {"edge_score": -0.5, "ir": -0.1},
            "trend": {"edge_score": 0.2, "ir": 0.05},
        }
        
        quality, health = _compute_factor_health(factor_edges)
        
        assert quality < 0.5
        assert health["strength_label"] == "weak"
        assert health["pct_negative"] == 0.75

    def test_mean_ir_calculation(self):
        """Mean IR should be correctly calculated."""
        factor_edges = {
            "momentum": {"edge_score": 0.5, "ir": 0.4},
            "carry": {"edge_score": 0.3, "ir": 0.2},
        }
        
        _, health = _compute_factor_health(factor_edges)
        
        assert health["mean_ir"] == pytest.approx(0.3, rel=0.01)


# ---------------------------------------------------------------------------
# Symbol Health Tests
# ---------------------------------------------------------------------------


class TestComputeSymbolHealth:
    """Tests for _compute_symbol_health."""

    def test_empty_symbol_edges(self):
        """Empty symbol edges should return default quality."""
        quality, health = _compute_symbol_health({})
        
        assert quality == 0.5
        assert health["symbol_count"] == 0
        assert health["top_contributors"] == []

    def test_strong_symbols(self):
        """Strong symbols with high conviction should yield high quality."""
        symbol_edges = {
            "BTCUSDT": {"edge_score": 1.5, "conviction": 0.9, "hybrid_score": 0.8},
            "ETHUSDT": {"edge_score": 1.2, "conviction": 0.85, "hybrid_score": 0.75},
            "SOLUSDT": {"edge_score": 0.8, "conviction": 0.7, "hybrid_score": 0.6},
        }
        
        quality, health = _compute_symbol_health(symbol_edges)
        
        assert quality > 0.6
        assert health["symbol_count"] == 3
        assert len(health["top_contributors"]) <= 3
        assert health["top_contributors"][0]["symbol"] in symbol_edges

    def test_conviction_weighted_scoring(self):
        """Higher conviction should weight the score higher."""
        symbol_edges = {
            "LOWCONV": {"edge_score": 1.0, "conviction": 0.1, "hybrid_score": 0.5},
            "HIGHCONV": {"edge_score": 0.8, "conviction": 0.9, "hybrid_score": 0.5},
        }
        
        _, health = _compute_symbol_health(symbol_edges)
        
        # High conviction symbol should be top contributor despite lower raw edge
        assert health["top_contributors"][0]["symbol"] == "HIGHCONV"

    def test_draggers_identification(self):
        """Worst performers should be identified as draggers."""
        symbol_edges = {
            "GOOD": {"edge_score": 1.0, "conviction": 0.8, "hybrid_score": 0.7},
            "BAD": {"edge_score": -1.0, "conviction": 0.3, "hybrid_score": 0.2},
            "MEDIUM": {"edge_score": 0.0, "conviction": 0.5, "hybrid_score": 0.4},
        }
        
        _, health = _compute_symbol_health(symbol_edges, top_n=2)
        
        assert len(health["draggers"]) >= 1
        assert any(d["symbol"] == "BAD" for d in health["draggers"])


# ---------------------------------------------------------------------------
# Category Health Tests
# ---------------------------------------------------------------------------


class TestComputeCategoryHealth:
    """Tests for _compute_category_health."""

    def test_empty_category_edges(self):
        """Empty category edges should return default quality."""
        quality, health = _compute_category_health({})
        
        assert quality == 0.5
        assert health["category_count"] == 0
        assert health["strongest_category"] is None

    def test_category_ranking(self):
        """Categories should be correctly ranked by edge score."""
        category_edges = {
            "DEFI": {"edge_score": 1.5, "momentum": 0.3, "ir": 0.2},
            "L1": {"edge_score": 0.5, "momentum": 0.1, "ir": 0.1},
            "MEME": {"edge_score": -0.5, "momentum": -0.1, "ir": -0.05},
        }
        
        _, health = _compute_category_health(category_edges)
        
        assert health["strongest_category"] == "DEFI"
        assert health["weakest_category"] == "MEME"
        assert health["category_count"] == 3

    def test_single_category(self):
        """Single category should be both strongest and weakest."""
        category_edges = {
            "DEFI": {"edge_score": 0.5, "momentum": 0.2, "ir": 0.1},
        }
        
        _, health = _compute_category_health(category_edges)
        
        assert health["strongest_category"] == "DEFI"
        assert health["weakest_category"] == "DEFI"


# ---------------------------------------------------------------------------
# Regime Alignment Tests
# ---------------------------------------------------------------------------


class TestComputeRegimeAlignment:
    """Tests for _compute_regime_alignment."""

    def test_normal_regime_alignment(self):
        """NORMAL/NORMAL regime should have high alignment with 1.0 multiplier."""
        regime = {"vol_regime": "NORMAL", "dd_state": "NORMAL"}
        regime_adjustments = {
            "conviction": {"combined_multiplier": 1.0},
            "factor_weights": {"combined_multiplier": 1.0},
        }
        factor_health = {"strength_label": "mixed"}
        
        alignment, info = _compute_regime_alignment(regime, regime_adjustments, factor_health)
        
        assert alignment >= 0.8
        assert info["in_expected_range"]

    def test_crisis_with_suppression(self):
        """CRISIS regime with suppressed conviction should be well-aligned."""
        regime = {"vol_regime": "CRISIS", "dd_state": "NORMAL"}
        regime_adjustments = {
            "conviction": {"combined_multiplier": 0.40},
            "factor_weights": {"combined_multiplier": 0.70},
        }
        factor_health = {"strength_label": "mixed"}
        
        alignment, info = _compute_regime_alignment(regime, regime_adjustments, factor_health)
        
        assert alignment >= 0.7
        assert info["vol_regime"] == "CRISIS"

    def test_crisis_without_suppression_misaligned(self):
        """CRISIS regime without suppression should be misaligned."""
        regime = {"vol_regime": "CRISIS", "dd_state": "NORMAL"}
        regime_adjustments = {
            "conviction": {"combined_multiplier": 1.0},  # Should be 0.4
            "factor_weights": {"combined_multiplier": 1.0},
        }
        factor_health = {"strength_label": "mixed"}
        
        alignment, info = _compute_regime_alignment(regime, regime_adjustments, factor_health)
        
        assert alignment < 0.8
        assert not info["in_expected_range"]

    def test_strong_factors_in_crisis_penalty(self):
        """Strong factors in CRISIS should receive alignment penalty."""
        regime = {"vol_regime": "CRISIS", "dd_state": "NORMAL"}
        regime_adjustments = {
            "conviction": {"combined_multiplier": 0.40},
            "factor_weights": {"combined_multiplier": 0.70},
        }
        factor_health = {"strength_label": "strong"}  # Unusual in crisis
        
        alignment, _ = _compute_regime_alignment(regime, regime_adjustments, factor_health)
        
        # Should have penalty for strong factors in crisis
        assert alignment < 0.95


# ---------------------------------------------------------------------------
# Execution Quality Tests
# ---------------------------------------------------------------------------


class TestComputeExecutionQuality:
    """Tests for _compute_execution_quality."""

    def test_excellent_execution(self):
        """High router quality with low slippage should be excellent."""
        regime = {
            "router_quality": 0.95,
            "avg_slippage_bps": 2.0,
            "maker_rate": 0.8,
        }
        
        quality, info = _compute_execution_quality(regime)
        
        assert quality >= 0.85
        assert info["quality_bucket"] == "excellent"

    def test_degraded_execution(self):
        """Medium router quality with high slippage should be degraded."""
        regime = {
            "router_quality": 0.55,
            "avg_slippage_bps": 25.0,
            "maker_rate": 0.4,
        }
        
        quality, info = _compute_execution_quality(regime)
        
        assert 0.4 <= quality < 0.6
        assert info["quality_bucket"] == "degraded"

    def test_poor_execution(self):
        """Low router quality with high slippage should be poor."""
        regime = {
            "router_quality": 0.3,
            "avg_slippage_bps": 50.0,
            "maker_rate": 0.1,
        }
        
        quality, info = _compute_execution_quality(regime)
        
        assert quality < 0.4
        assert info["quality_bucket"] == "poor"

    def test_maker_rate_bonus(self):
        """High maker rate should boost quality."""
        regime_high_maker = {
            "router_quality": 0.7,
            "avg_slippage_bps": 5.0,
            "maker_rate": 0.9,
        }
        regime_low_maker = {
            "router_quality": 0.7,
            "avg_slippage_bps": 5.0,
            "maker_rate": 0.2,
        }
        
        quality_high, _ = _compute_execution_quality(regime_high_maker)
        quality_low, _ = _compute_execution_quality(regime_low_maker)
        
        assert quality_high > quality_low


# ---------------------------------------------------------------------------
# Aggregate Health Score Tests
# ---------------------------------------------------------------------------


class TestComputeStrategyHealth:
    """Tests for compute_strategy_health aggregate function."""

    @pytest.fixture
    def healthy_inputs(self) -> Dict[str, Any]:
        """Return inputs for a healthy system."""
        return {
            "factor_edges": {
                "momentum": {"edge_score": 1.0, "ir": 0.3},
                "carry": {"edge_score": 0.8, "ir": 0.25},
            },
            "symbol_edges": {
                "BTCUSDT": {"edge_score": 1.2, "conviction": 0.9, "hybrid_score": 0.8},
                "ETHUSDT": {"edge_score": 0.9, "conviction": 0.85, "hybrid_score": 0.75},
                "SOLUSDT": {"edge_score": 0.7, "conviction": 0.8, "hybrid_score": 0.7},
                "AVAXUSDT": {"edge_score": 0.6, "conviction": 0.7, "hybrid_score": 0.6},
                "DOGEUSDT": {"edge_score": 0.5, "conviction": 0.6, "hybrid_score": 0.5},
            },
            "category_edges": {
                "DEFI": {"edge_score": 0.8, "momentum": 0.2, "ir": 0.15},
                "L1": {"edge_score": 0.6, "momentum": 0.1, "ir": 0.1},
                "MEME": {"edge_score": 0.3, "momentum": 0.05, "ir": 0.05},
            },
            "regime": {
                "vol_regime": "NORMAL",
                "dd_state": "NORMAL",
                "router_quality": 0.9,
                "avg_slippage_bps": 3.0,
                "maker_rate": 0.75,
            },
            "regime_adjustments": {
                "conviction": {"combined_multiplier": 1.0},
                "factor_weights": {"combined_multiplier": 1.0},
            },
        }

    @pytest.fixture
    def stressed_inputs(self) -> Dict[str, Any]:
        """Return inputs for a stressed system."""
        return {
            "factor_edges": {
                "momentum": {"edge_score": -0.8, "ir": -0.2},
                "carry": {"edge_score": -0.5, "ir": -0.1},
            },
            "symbol_edges": {
                "BTCUSDT": {"edge_score": -0.5, "conviction": 0.3, "hybrid_score": 0.2},
            },
            "category_edges": {
                "DEFI": {"edge_score": -0.3, "momentum": -0.1, "ir": -0.05},
            },
            "regime": {
                "vol_regime": "CRISIS",
                "dd_state": "DRAWDOWN",
                "router_quality": 0.4,
                "avg_slippage_bps": 40.0,
                "maker_rate": 0.2,
            },
            "regime_adjustments": {
                "conviction": {"combined_multiplier": 1.0},  # Misaligned
                "factor_weights": {"combined_multiplier": 1.0},
            },
        }

    def test_healthy_system_high_score(self, healthy_inputs):
        """Healthy system should have high health score."""
        result = compute_strategy_health(**healthy_inputs)
        
        assert result.health_score >= 0.70
        assert "✅ System healthy" in result.notes

    def test_stressed_system_low_score(self, stressed_inputs):
        """Stressed system should have low health score."""
        result = compute_strategy_health(**stressed_inputs)
        
        assert result.health_score < 0.50
        assert any("stressed" in note.lower() or "🚨" in note for note in result.notes)

    def test_health_score_in_range(self, healthy_inputs):
        """Health score should always be in [0, 1]."""
        result = compute_strategy_health(**healthy_inputs)
        
        assert 0.0 <= result.health_score <= 1.0

    def test_strategy_health_to_dict(self, healthy_inputs):
        """StrategyHealth should serialize correctly."""
        result = compute_strategy_health(**healthy_inputs)
        result_dict = result.to_dict()
        
        assert "health_score" in result_dict
        assert "factor_health" in result_dict
        assert "symbol_health" in result_dict
        assert "category_health" in result_dict
        assert "regime_alignment" in result_dict
        assert "execution_quality" in result_dict
        assert "notes" in result_dict
        assert isinstance(result_dict["notes"], list)

    def test_custom_weights(self, healthy_inputs):
        """Custom weights should affect the health score."""
        # Give all weight to factor quality
        weights = {
            "factor_quality": 1.0,
            "symbol_quality": 0.0,
            "category_quality": 0.0,
            "regime_alignment": 0.0,
            "execution_quality": 0.0,
        }
        
        result = compute_strategy_health(**healthy_inputs, weights=weights)
        
        # Score should be close to factor quality score
        factor_quality, _ = _compute_factor_health(healthy_inputs["factor_edges"])
        assert result.health_score == pytest.approx(factor_quality, rel=0.1)

    def test_notes_generated_for_issues(self, stressed_inputs):
        """Notes should be generated for various issues."""
        result = compute_strategy_health(**stressed_inputs)
        
        # Should have multiple notes about issues
        assert len(result.notes) >= 2

    def test_empty_inputs_handled(self):
        """Empty inputs should not crash."""
        result = compute_strategy_health(
            factor_edges={},
            symbol_edges={},
            category_edges={},
            regime={},
            regime_adjustments={},
        )
        
        assert 0.0 <= result.health_score <= 1.0
        assert isinstance(result.notes, list)
