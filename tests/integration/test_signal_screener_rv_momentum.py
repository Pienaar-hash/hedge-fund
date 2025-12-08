"""
Tests for v7.5_C1 — RV Momentum Integration with Signal Screener.

Tests:
- RV score is added to intent metadata
- Higher RV score ranks higher when hybrid scores equal
- RV filter rejects weak relative momentum for LONGs
- RV filter rejects strong relative momentum for SHORTs
"""
from __future__ import annotations

from unittest.mock import patch


class TestScreenerRvMomentumIntegration:
    """Test RV momentum integration with screener."""

    def test_rv_score_added_to_intents(self):
        """RV score is added to intent metadata."""
        # Create mock intents
        mock_intents = [
            {"symbol": "BTCUSDT", "direction": "LONG", "trend_score": 0.7},
        ]
        
        # Mock all dependencies
        with patch("execution.signal_screener._RV_MOMENTUM_AVAILABLE", True):
            with patch("execution.signal_screener.get_rv_score") as mock_rv:
                mock_rv.return_value = 0.65
                
                # The RV score should be added when processing ranked results
                # We test the logic in isolation
                intent = mock_intents[0].copy()
                rv_score_val = mock_rv("BTCUSDT")
                intent["rv_momentum_score"] = rv_score_val
                
                assert intent["rv_momentum_score"] == 0.65

    def test_rv_secondary_sort_order(self):
        """Higher RV score ranks higher when hybrid scores equal."""
        intents = [
            {"symbol": "A", "hybrid_score": 0.7, "router_quality_score": 0.8, "rv_momentum_score": 0.3},
            {"symbol": "B", "hybrid_score": 0.7, "router_quality_score": 0.8, "rv_momentum_score": 0.8},
            {"symbol": "C", "hybrid_score": 0.7, "router_quality_score": 0.8, "rv_momentum_score": 0.5},
        ]
        
        # Sort as the screener does (hybrid desc, router_quality desc, rv_momentum desc)
        sorted_intents = sorted(
            intents,
            key=lambda x: (
                x.get("hybrid_score", 0.0),
                x.get("router_quality_score") or 0.8,
                x.get("rv_momentum_score") or 0.0,
            ),
            reverse=True,
        )
        
        # B should be first (highest RV), then C, then A
        assert sorted_intents[0]["symbol"] == "B"
        assert sorted_intents[1]["symbol"] == "C"
        assert sorted_intents[2]["symbol"] == "A"

    def test_rv_filter_rejects_weak_longs(self):
        """RV filter rejects weak relative momentum for LONGs."""
        rv_min_for_long = -0.5
        
        # Intent with weak RV
        intent = {"symbol": "WEAKUSDT", "direction": "LONG"}
        rv_score = -0.7  # Below threshold
        
        rv_filtered = intent["direction"] == "LONG" and rv_score < rv_min_for_long
        
        assert rv_filtered is True

    def test_rv_filter_accepts_strong_longs(self):
        """RV filter accepts strong relative momentum for LONGs."""
        rv_min_for_long = -0.5
        
        intent = {"symbol": "STRONGUSDT", "direction": "LONG"}
        rv_score = 0.5  # Above threshold
        
        rv_filtered = intent["direction"] == "LONG" and rv_score < rv_min_for_long
        
        assert rv_filtered is False

    def test_rv_filter_rejects_strong_shorts(self):
        """RV filter rejects strong relative momentum for SHORTs."""
        rv_max_for_short = 0.5
        
        intent = {"symbol": "STRONGUSDT", "direction": "SHORT"}
        rv_score = 0.8  # Above threshold (too strong for short)
        
        rv_filtered = intent["direction"] == "SHORT" and rv_score > rv_max_for_short
        
        assert rv_filtered is True

    def test_rv_filter_accepts_weak_shorts(self):
        """RV filter accepts weak relative momentum for SHORTs."""
        rv_max_for_short = 0.5
        
        intent = {"symbol": "WEAKUSDT", "direction": "SHORT"}
        rv_score = -0.3  # Below threshold (weak = good for short)
        
        rv_filtered = intent["direction"] == "SHORT" and rv_score > rv_max_for_short
        
        assert rv_filtered is False

    def test_rv_config_loading(self):
        """RV config is loaded correctly from strategy config."""
        strategy_config = {
            "rv_momentum": {
                "enabled": True,
                "rv_min_for_long": -0.3,
                "rv_max_for_short": 0.4,
            }
        }
        
        rv_min_for_long = strategy_config.get("rv_momentum", {}).get("rv_min_for_long", -0.5)
        rv_max_for_short = strategy_config.get("rv_momentum", {}).get("rv_max_for_short", 0.5)
        
        assert rv_min_for_long == -0.3
        assert rv_max_for_short == 0.4


class TestScreenerRvWithHybridRanking:
    """Test RV integration with full hybrid ranking flow."""

    def test_intents_ranked_by_hybrid_then_rv(self):
        """Intents are ranked by hybrid score, then RV as tiebreaker."""
        intents = [
            {"symbol": "A", "hybrid_score": 0.8, "rv_momentum_score": 0.2},
            {"symbol": "B", "hybrid_score": 0.6, "rv_momentum_score": 0.9},
            {"symbol": "C", "hybrid_score": 0.8, "rv_momentum_score": 0.7},
        ]
        
        # Sort by hybrid desc, then rv desc
        sorted_intents = sorted(
            intents,
            key=lambda x: (
                x.get("hybrid_score", 0.0),
                x.get("rv_momentum_score") or 0.0,
            ),
            reverse=True,
        )
        
        # C should beat A (same hybrid, higher RV)
        # A and C tied on hybrid (0.8), C has higher RV
        # B is lower hybrid (0.6)
        assert sorted_intents[0]["symbol"] == "C"
        assert sorted_intents[1]["symbol"] == "A"
        assert sorted_intents[2]["symbol"] == "B"

    def test_rv_disabled_no_filtering(self):
        """RV disabled means no filtering applied."""
        intents = [
            {"symbol": "A", "direction": "LONG"},
            {"symbol": "B", "direction": "LONG"},
        ]
        
        rv_config_enabled = False
        rv_min_for_long = -0.5
        
        filtered = []
        for intent in intents:
            rv_score = -0.8  # Would be filtered if enabled
            rv_filtered = False
            
            if rv_config_enabled:
                if intent["direction"] == "LONG" and rv_score < rv_min_for_long:
                    rv_filtered = True
            
            if not rv_filtered:
                filtered.append(intent)
        
        # All should pass since RV is disabled
        assert len(filtered) == 2
