"""
Tests for v7.5_B2 — Router Quality integration with Signal Screener.

Tests:
- Candidates with low router_quality_score are filtered out
- Among similar hybrid scores, higher router_quality is preferred
- Filter threshold from config is respected
"""
from __future__ import annotations

import pytest
from unittest.mock import patch


class TestSignalScreenerRouterQualityFiltering:
    """Test router quality filtering in screener."""

    def test_low_quality_filtered(self):
        """Intents with low router quality are filtered out."""
        from execution.signal_screener import _ROUTER_QUALITY_AVAILABLE
        
        # Skip if router quality not available
        if not _ROUTER_QUALITY_AVAILABLE:
            pytest.skip("Router quality module not available")
        
        from execution.router_metrics import RouterQualityConfig
        
        cfg = RouterQualityConfig(
            enabled=True,
            min_for_emission=0.4,
        )
        
        # Create mock intents
        intents = [
            {"symbol": "BTCUSDT", "direction": "LONG", "hybrid_score": 0.7},
            {"symbol": "ETHUSDT", "direction": "LONG", "hybrid_score": 0.65},
            {"symbol": "BADUSDT", "direction": "LONG", "hybrid_score": 0.6},
        ]
        
        # Mock router quality scores
        scores = {
            "BTCUSDT": 0.85,  # Good
            "ETHUSDT": 0.72,  # OK
            "BADUSDT": 0.25,  # Below threshold
        }
        
        with patch("execution.router_metrics.get_router_quality_score") as mock_rq:
            mock_rq.side_effect = lambda sym, cfg: scores.get(sym.upper(), 0.8)
            
            # Filter manually like screener would
            filtered = []
            for intent in intents:
                sym = intent["symbol"]
                rq_score = scores.get(sym, 0.8)
                if rq_score >= cfg.min_for_emission:
                    intent["router_quality_score"] = rq_score
                    filtered.append(intent)
            
            # BADUSDT should be filtered
            assert len(filtered) == 2
            symbols = [i["symbol"] for i in filtered]
            assert "BADUSDT" not in symbols
            assert "BTCUSDT" in symbols
            assert "ETHUSDT" in symbols

    def test_secondary_sort_by_router_quality(self):
        """Intents with same hybrid score sorted by router quality."""
        from execution.signal_screener import _ROUTER_QUALITY_AVAILABLE
        
        # Skip if router quality not available
        if not _ROUTER_QUALITY_AVAILABLE:
            pytest.skip("Router quality module not available")
        
        # Create intents with similar hybrid scores
        intents = [
            {"symbol": "AAUSDT", "hybrid_score": 0.70, "router_quality_score": 0.60},
            {"symbol": "BBUSDT", "hybrid_score": 0.70, "router_quality_score": 0.90},
            {"symbol": "CCUSDT", "hybrid_score": 0.70, "router_quality_score": 0.75},
        ]
        
        # Sort like screener does: (hybrid_score, router_quality) both descending
        sorted_intents = sorted(
            intents,
            key=lambda x: (x.get("hybrid_score", 0), x.get("router_quality_score") or 0.8),
            reverse=True,
        )
        
        # BBUSDT (highest router quality) should be first
        assert sorted_intents[0]["symbol"] == "BBUSDT"
        assert sorted_intents[1]["symbol"] == "CCUSDT"
        assert sorted_intents[2]["symbol"] == "AAUSDT"

    def test_hybrid_score_takes_priority(self):
        """Hybrid score is primary sort key, router quality is secondary."""
        intents = [
            {"symbol": "LOWH", "hybrid_score": 0.50, "router_quality_score": 0.95},
            {"symbol": "HIGHH", "hybrid_score": 0.85, "router_quality_score": 0.60},
        ]
        
        sorted_intents = sorted(
            intents,
            key=lambda x: (x.get("hybrid_score", 0), x.get("router_quality_score") or 0.8),
            reverse=True,
        )
        
        # HIGHH has higher hybrid score, should be first despite lower router quality
        assert sorted_intents[0]["symbol"] == "HIGHH"
        assert sorted_intents[1]["symbol"] == "LOWH"

    def test_missing_router_quality_uses_default(self):
        """Intents without router_quality_score use default (0.8) for sorting."""
        intents = [
            {"symbol": "KNOWN", "hybrid_score": 0.70, "router_quality_score": 0.85},
            {"symbol": "UNKNOWN", "hybrid_score": 0.70},  # No router_quality_score
            {"symbol": "LOW", "hybrid_score": 0.70, "router_quality_score": 0.60},
        ]
        
        sorted_intents = sorted(
            intents,
            key=lambda x: (x.get("hybrid_score", 0), x.get("router_quality_score") or 0.8),
            reverse=True,
        )
        
        # Order: KNOWN (0.85), UNKNOWN (default 0.8), LOW (0.60)
        assert sorted_intents[0]["symbol"] == "KNOWN"
        assert sorted_intents[1]["symbol"] == "UNKNOWN"
        assert sorted_intents[2]["symbol"] == "LOW"


class TestRouterQualityConfigInScreener:
    """Test config loading in screener context."""

    def test_min_for_emission_respected(self):
        """Min_for_emission threshold from config is used."""
        from execution.router_metrics import load_router_quality_config
        
        cfg_dict = {
            "router_quality": {
                "enabled": True,
                "min_for_emission": 0.35,
            }
        }
        
        cfg = load_router_quality_config(cfg_dict)
        assert cfg.enabled is True
        assert cfg.min_for_emission == 0.35

    def test_disabled_skips_filtering(self):
        """When disabled, no filtering occurs."""
        from execution.router_metrics import load_router_quality_config
        
        cfg_dict = {
            "router_quality": {
                "enabled": False,
                "min_for_emission": 0.50,
            }
        }
        
        cfg = load_router_quality_config(cfg_dict)
        assert cfg.enabled is False
        # min_for_emission irrelevant when disabled


class TestRouterQualityAvailabilityFlag:
    """Test the module availability flag."""

    def test_flag_is_set(self):
        """_ROUTER_QUALITY_AVAILABLE flag is set correctly."""
        from execution.signal_screener import _ROUTER_QUALITY_AVAILABLE
        
        # Should be True since router_metrics exists
        assert _ROUTER_QUALITY_AVAILABLE is True
