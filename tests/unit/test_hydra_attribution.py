"""
Unit tests for Hydra head_contributions attribution persistence — v7.9_P2

Tests the persistence and retrieval of head_contributions through the
position_tp_sl registry, which enables fill-time attribution.

Flow:
  Hydra Intent → persist_head_contributions_for_position() 
               → position_tp_sl.json
               → get_head_contributions_for_position() → Fill enrichment
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from execution.position_tp_sl_registry import (
    register_position_tp_sl,
    get_position_tp_sl,
    get_head_contributions,
    get_all_head_contributions,
    unregister_position_tp_sl,
    _make_key,
)

from execution.hydra_integration import (
    persist_head_contributions_for_position,
    get_head_contributions_for_position,
    get_all_position_head_contributions,
    enrich_fill_with_head_contributions,
    build_head_contributions_map_for_positions,
)


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_registry_path(tmp_path):
    """Provide a temp path for registry file."""
    return tmp_path / "position_tp_sl.json"


@pytest.fixture
def mock_registry(temp_registry_path, monkeypatch):
    """Mock the registry path to use temp file."""
    import execution.position_tp_sl_registry as reg
    
    # Reset registry state
    reg._TP_SL_REGISTRY = {}
    reg._REGISTRY_LOADED = False
    
    # Patch the path
    monkeypatch.setattr(reg, "_REGISTRY_PATH", temp_registry_path)
    monkeypatch.setattr(reg, "_STATE_DIR", temp_registry_path.parent)
    
    yield temp_registry_path
    
    # Cleanup
    reg._TP_SL_REGISTRY = {}
    reg._REGISTRY_LOADED = False


# ---------------------------------------------------------------------------
# Registry Level Tests
# ---------------------------------------------------------------------------


class TestRegistryHeadContributions:
    """Tests for head_contributions storage in position_tp_sl_registry."""
    
    def test_register_with_head_contributions(self, mock_registry):
        """Test registering position with head_contributions."""
        head_contribs = {"TREND": 0.7, "CATEGORY": 0.3}
        
        register_position_tp_sl(
            symbol="BTCUSDT",
            position_side="LONG",
            take_profit_price=45000.0,
            stop_loss_price=35000.0,
            head_contributions=head_contribs,
            strategy_heads=["TREND", "CATEGORY"],
        )
        
        entry = get_position_tp_sl("BTCUSDT", "LONG")
        assert entry is not None
        assert entry.get("head_contributions") == head_contribs
        assert entry.get("strategy_heads") == ["TREND", "CATEGORY"]
    
    def test_get_head_contributions(self, mock_registry):
        """Test retrieving head_contributions for a position."""
        head_contribs = {"ZSCORE": 0.5, "CORR_PAIR": 0.5}
        
        register_position_tp_sl(
            symbol="ETHUSDT",
            position_side="SHORT",
            take_profit_price=2000.0,
            stop_loss_price=2800.0,
            head_contributions=head_contribs,
        )
        
        result = get_head_contributions("ETHUSDT", "SHORT")
        assert result == head_contribs
    
    def test_get_head_contributions_not_found(self, mock_registry):
        """Test get_head_contributions returns empty dict if not found."""
        result = get_head_contributions("NOSYMBOL", "LONG")
        assert result == {}
    
    def test_get_head_contributions_no_contributions(self, mock_registry):
        """Test get_head_contributions returns empty dict if no contributions stored."""
        register_position_tp_sl(
            symbol="SOLUSDT",
            position_side="LONG",
            take_profit_price=100.0,
            stop_loss_price=80.0,
            # No head_contributions
        )
        
        result = get_head_contributions("SOLUSDT", "LONG")
        assert result == {}
    
    def test_get_all_head_contributions(self, mock_registry):
        """Test retrieving all head_contributions."""
        register_position_tp_sl(
            symbol="BTCUSDT",
            position_side="LONG",
            take_profit_price=50000.0,
            stop_loss_price=40000.0,
            head_contributions={"TREND": 0.8, "VOL_TARGET": 0.2},
        )
        register_position_tp_sl(
            symbol="ETHUSDT",
            position_side="SHORT",
            take_profit_price=2000.0,
            stop_loss_price=2800.0,
            head_contributions={"ZSCORE": 1.0},
        )
        register_position_tp_sl(
            symbol="SOLUSDT",
            position_side="LONG",
            take_profit_price=100.0,
            stop_loss_price=80.0,
            # No contributions - should not appear
        )
        
        result = get_all_head_contributions()
        assert len(result) == 2
        assert "BTCUSDT:LONG" in result
        assert "ETHUSDT:SHORT" in result
        assert "SOLUSDT:LONG" not in result


# ---------------------------------------------------------------------------
# Integration Level Tests (hydra_integration.py)
# ---------------------------------------------------------------------------


class TestHydraAttributionPersistence:
    """Tests for hydra_integration attribution helpers."""
    
    def test_persist_head_contributions_for_position(self, mock_registry):
        """Test persisting head_contributions via hydra_integration."""
        head_contribs = {"ALPHA_MINER": 0.4, "CATEGORY": 0.6}
        
        result = persist_head_contributions_for_position(
            symbol="BTCUSDT",
            side="LONG",
            head_contributions=head_contribs,
            strategy_heads=["ALPHA_MINER", "CATEGORY"],
        )
        
        assert result is True
        
        # Verify retrieval
        stored = get_head_contributions_for_position("BTCUSDT", "LONG")
        assert stored == head_contribs
    
    def test_get_head_contributions_for_position(self, mock_registry):
        """Test retrieving head_contributions via hydra_integration."""
        register_position_tp_sl(
            symbol="ETHUSDT",
            position_side="LONG",
            take_profit_price=2500.0,
            stop_loss_price=2000.0,
            head_contributions={"TREND": 0.9, "CORR_PAIR": 0.1},
        )
        
        result = get_head_contributions_for_position("ETHUSDT", "LONG")
        assert result == {"TREND": 0.9, "CORR_PAIR": 0.1}
    
    def test_get_all_position_head_contributions(self, mock_registry):
        """Test get_all_position_head_contributions."""
        register_position_tp_sl(
            symbol="BTCUSDT",
            position_side="LONG",
            take_profit_price=50000.0,
            stop_loss_price=40000.0,
            head_contributions={"TREND": 0.6, "VOL_TARGET": 0.4},
        )
        register_position_tp_sl(
            symbol="ETHUSDT",
            position_side="SHORT",
            take_profit_price=2000.0,
            stop_loss_price=2800.0,
            head_contributions={"ZSCORE": 0.7, "CATEGORY": 0.3},
        )
        
        result = get_all_position_head_contributions()
        assert len(result) == 2
        assert result["BTCUSDT:LONG"] == {"TREND": 0.6, "VOL_TARGET": 0.4}
        assert result["ETHUSDT:SHORT"] == {"ZSCORE": 0.7, "CATEGORY": 0.3}


# ---------------------------------------------------------------------------
# Fill Enrichment Tests
# ---------------------------------------------------------------------------


class TestFillEnrichment:
    """Tests for fill enrichment with head_contributions."""
    
    def test_enrich_fill_with_head_contributions(self, mock_registry):
        """Test enriching fill with head_contributions."""
        register_position_tp_sl(
            symbol="BTCUSDT",
            position_side="LONG",
            take_profit_price=50000.0,
            stop_loss_price=38000.0,
            head_contributions={"TREND": 0.7, "ALPHA_MINER": 0.3},
        )
        
        fill = {
            "symbol": "BTCUSDT",
            "side": "LONG",
            "qty": 0.01,
            "price": 42000.0,
            "pnl": 100.0,
        }
        
        enriched = enrich_fill_with_head_contributions(fill)
        
        assert enriched["head_contributions"] == {"TREND": 0.7, "ALPHA_MINER": 0.3}
        assert enriched["source"] == "hydra"
    
    def test_enrich_fill_with_position_side_key(self, mock_registry):
        """Test enrichment using positionSide key."""
        register_position_tp_sl(
            symbol="ETHUSDT",
            position_side="SHORT",
            take_profit_price=2000.0,
            stop_loss_price=2800.0,
            head_contributions={"ZSCORE": 1.0},
        )
        
        fill = {
            "symbol": "ETHUSDT",
            "positionSide": "SHORT",  # Different key
            "qty": 0.5,
            "pnl": -50.0,
        }
        
        enriched = enrich_fill_with_head_contributions(fill)
        
        assert enriched["head_contributions"] == {"ZSCORE": 1.0}
        assert enriched["source"] == "hydra"
    
    def test_enrich_fill_no_contributions(self, mock_registry):
        """Test enrichment when no contributions exist."""
        fill = {
            "symbol": "UNKNOWN",
            "side": "LONG",
            "qty": 0.1,
            "pnl": 10.0,
        }
        
        enriched = enrich_fill_with_head_contributions(fill)
        
        assert enriched["head_contributions"] == {}
        assert enriched["source"] == "legacy"
    
    def test_enrich_fill_missing_symbol(self, mock_registry):
        """Test enrichment with missing symbol returns original fill."""
        fill = {"qty": 0.1, "pnl": 10.0}
        
        enriched = enrich_fill_with_head_contributions(fill)
        
        assert enriched == fill


# ---------------------------------------------------------------------------
# Position Map Builder Tests
# ---------------------------------------------------------------------------


class TestPositionMapBuilder:
    """Tests for build_head_contributions_map_for_positions."""
    
    def test_build_map_for_positions(self, mock_registry):
        """Test building contributions map from position list."""
        register_position_tp_sl(
            symbol="BTCUSDT",
            position_side="LONG",
            take_profit_price=50000.0,
            stop_loss_price=40000.0,
            head_contributions={"TREND": 0.8},
        )
        register_position_tp_sl(
            symbol="ETHUSDT",
            position_side="SHORT",
            take_profit_price=2000.0,
            stop_loss_price=2800.0,
            head_contributions={"ZSCORE": 0.6, "CATEGORY": 0.4},
        )
        
        positions = [
            {"symbol": "BTCUSDT", "positionSide": "LONG", "positionAmt": 0.01},
            {"symbol": "ETHUSDT", "side": "SHORT", "qty": 0.5},
        ]
        
        result = build_head_contributions_map_for_positions(positions)
        
        assert result["BTCUSDT"] == {"TREND": 0.8}
        assert result["ETHUSDT"] == {"ZSCORE": 0.6, "CATEGORY": 0.4}
    
    def test_build_map_infers_side_from_qty(self, mock_registry):
        """Test that side is inferred from qty sign when not provided."""
        register_position_tp_sl(
            symbol="BTCUSDT",
            position_side="LONG",
            take_profit_price=50000.0,
            stop_loss_price=40000.0,
            head_contributions={"TREND": 1.0},
        )
        register_position_tp_sl(
            symbol="ETHUSDT",
            position_side="SHORT",
            take_profit_price=2000.0,
            stop_loss_price=2800.0,
            head_contributions={"ZSCORE": 1.0},
        )
        
        positions = [
            {"symbol": "BTCUSDT", "positionAmt": 0.5},  # Positive = LONG
            {"symbol": "ETHUSDT", "qty": -0.3},         # Negative = SHORT
        ]
        
        result = build_head_contributions_map_for_positions(positions)
        
        assert result["BTCUSDT"] == {"TREND": 1.0}
        assert result["ETHUSDT"] == {"ZSCORE": 1.0}
    
    def test_build_map_empty_positions(self, mock_registry):
        """Test building map with empty positions list."""
        result = build_head_contributions_map_for_positions([])
        assert result == {}


# ---------------------------------------------------------------------------
# Persistence Robustness Tests
# ---------------------------------------------------------------------------


class TestPersistenceRobustness:
    """Tests for persistence edge cases and error handling."""
    
    def test_persist_updates_existing_entry(self, mock_registry):
        """Test that persisting updates an existing entry."""
        # First registration (with TP/SL)
        register_position_tp_sl(
            symbol="BTCUSDT",
            position_side="LONG",
            take_profit_price=50000.0,
            stop_loss_price=38000.0,
        )
        
        # Later, persist head_contributions
        persist_head_contributions_for_position(
            symbol="BTCUSDT",
            side="LONG",
            head_contributions={"TREND": 0.9, "VOL_TARGET": 0.1},
        )
        
        # TP/SL should still be there, plus contributions
        entry = get_position_tp_sl("BTCUSDT", "LONG")
        assert entry is not None
        assert entry.get("take_profit_price") == 50000.0
        assert entry.get("stop_loss_price") == 38000.0
        assert entry.get("head_contributions") == {"TREND": 0.9, "VOL_TARGET": 0.1}
    
    def test_persist_creates_minimal_entry(self, mock_registry):
        """Test that persist creates minimal entry if position doesn't exist."""
        persist_head_contributions_for_position(
            symbol="NEWUSDT",
            side="SHORT",
            head_contributions={"ALPHA_MINER": 1.0},
        )
        
        entry = get_position_tp_sl("NEWUSDT", "SHORT")
        assert entry is not None
        assert entry.get("head_contributions") == {"ALPHA_MINER": 1.0}
        assert entry.get("source") == "hydra"
    
    def test_cleanup_on_position_close(self, mock_registry):
        """Test that head_contributions are cleaned up when position is removed."""
        register_position_tp_sl(
            symbol="BTCUSDT",
            position_side="LONG",
            take_profit_price=50000.0,
            stop_loss_price=38000.0,
            head_contributions={"TREND": 0.5, "ZSCORE": 0.5},
        )
        
        # Position closed
        unregister_position_tp_sl("BTCUSDT", "LONG")
        
        # Should return empty
        result = get_head_contributions("BTCUSDT", "LONG")
        assert result == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
