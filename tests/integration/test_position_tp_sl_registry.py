"""
Unit tests for position_tp_sl_registry.py (PATCHSET V2)
"""

import pytest
import json
import os
import tempfile
import time
from pathlib import Path


# ===========================================================================
# Fixture: Isolated registry with cleared state
# ===========================================================================

@pytest.fixture
def fresh_registry(monkeypatch, tmp_path):
    """Provide a fresh registry instance with cleared state."""
    import execution.position_tp_sl_registry as reg
    
    # Point to temp file
    temp_file = tmp_path / "position_tp_sl.json"
    monkeypatch.setattr(reg, "_REGISTRY_PATH", temp_file)
    monkeypatch.setattr(reg, "_STATE_DIR", tmp_path)
    
    # Clear state
    reg._TP_SL_REGISTRY.clear()
    reg._REGISTRY_LOADED = True  # Skip file load
    
    yield reg
    
    # Cleanup
    reg._TP_SL_REGISTRY.clear()
    reg._REGISTRY_LOADED = False


# ===========================================================================
# Tests: register_position_tp_sl
# ===========================================================================

class TestRegisterPositionTpSl:
    def test_register_entry(self, fresh_registry):
        """Register an entry and verify it's stored."""
        fresh_registry.register_position_tp_sl(
            symbol="BTCUSDT",
            position_side="LONG",
            take_profit_price=55000.0,
            stop_loss_price=48000.0,
        )
        
        result = fresh_registry.get_position_tp_sl("BTCUSDT", "LONG")
        assert result is not None
        assert result["take_profit_price"] == 55000.0
        assert result["stop_loss_price"] == 48000.0
        assert result["strategy"] == "vol_target"

    def test_register_adds_timestamp(self, fresh_registry):
        """Verify timestamp is added to entry."""
        before = time.time()
        fresh_registry.register_position_tp_sl("BTCUSDT", "LONG", 55000.0, 48000.0)
        after = time.time()
        
        result = fresh_registry.get_position_tp_sl("BTCUSDT", "LONG")
        assert "registered_at" in result
        assert before <= result["registered_at"] <= after

    def test_skip_registration_no_levels(self, fresh_registry):
        """Skip registration when no TP/SL levels provided."""
        fresh_registry.register_position_tp_sl("BTCUSDT", "LONG", None, None)
        
        result = fresh_registry.get_position_tp_sl("BTCUSDT", "LONG")
        assert result is None

    def test_register_with_metadata(self, fresh_registry):
        """Register with additional metadata."""
        metadata = {
            "sl_atr_mult": 2.0,
            "tp_atr_mult": 3.0,
            "min_rr": 1.2,
            "reward_risk": 1.5,
        }
        fresh_registry.register_position_tp_sl(
            symbol="BTCUSDT",
            position_side="LONG",
            take_profit_price=55000.0,
            stop_loss_price=48000.0,
            metadata=metadata,
        )
        
        result = fresh_registry.get_position_tp_sl("BTCUSDT", "LONG")
        assert result["sl_atr_mult"] == 2.0
        assert result["tp_atr_mult"] == 3.0

    def test_register_tp_only(self, fresh_registry):
        """Can register with only TP."""
        fresh_registry.register_position_tp_sl("BTCUSDT", "LONG", 55000.0, None)
        
        result = fresh_registry.get_position_tp_sl("BTCUSDT", "LONG")
        assert result is not None
        assert result["take_profit_price"] == 55000.0
        assert result["stop_loss_price"] is None

    def test_register_sl_only(self, fresh_registry):
        """Can register with only SL."""
        fresh_registry.register_position_tp_sl("BTCUSDT", "LONG", None, 48000.0)
        
        result = fresh_registry.get_position_tp_sl("BTCUSDT", "LONG")
        assert result is not None
        assert result["take_profit_price"] is None
        assert result["stop_loss_price"] == 48000.0


# ===========================================================================
# Tests: unregister_position_tp_sl
# ===========================================================================

class TestUnregisterPositionTpSl:
    def test_unregister_entry(self, fresh_registry):
        """Unregister entry and verify it's gone."""
        fresh_registry.register_position_tp_sl("BTCUSDT", "LONG", 55000.0, 48000.0)
        
        fresh_registry.unregister_position_tp_sl("BTCUSDT", "LONG")
        
        result = fresh_registry.get_position_tp_sl("BTCUSDT", "LONG")
        assert result is None

    def test_unregister_nonexistent(self, fresh_registry):
        """Unregistering nonexistent entry doesn't fail."""
        # Should not raise
        fresh_registry.unregister_position_tp_sl("BTCUSDT", "LONG")


# ===========================================================================
# Tests: get_position_tp_sl
# ===========================================================================

class TestGetPositionTpSl:
    def test_get_existing(self, fresh_registry):
        """Get returns entry for existing position."""
        fresh_registry.register_position_tp_sl("BTCUSDT", "LONG", 55000.0, 48000.0)
        
        result = fresh_registry.get_position_tp_sl("BTCUSDT", "LONG")
        assert result is not None
        assert result["symbol"] == "BTCUSDT"

    def test_get_nonexistent(self, fresh_registry):
        """Get returns None for nonexistent position."""
        result = fresh_registry.get_position_tp_sl("BTCUSDT", "LONG")
        assert result is None

    def test_get_case_insensitive_symbol(self, fresh_registry):
        """Symbol lookup is case-insensitive."""
        fresh_registry.register_position_tp_sl("BTCUSDT", "LONG", 55000.0, 48000.0)
        
        result = fresh_registry.get_position_tp_sl("btcusdt", "LONG")
        assert result is not None

    def test_get_case_insensitive_side(self, fresh_registry):
        """Side lookup is case-insensitive."""
        fresh_registry.register_position_tp_sl("BTCUSDT", "LONG", 55000.0, 48000.0)
        
        result = fresh_registry.get_position_tp_sl("BTCUSDT", "long")
        assert result is not None


# ===========================================================================
# Tests: get_all_tp_sl_positions
# ===========================================================================

class TestGetAllTpSlPositions:
    def test_get_all_empty(self, fresh_registry):
        """Get all returns empty dict when no positions."""
        result = fresh_registry.get_all_tp_sl_positions()
        assert result == {}

    def test_get_all_with_entries(self, fresh_registry):
        """Get all returns all registered positions."""
        fresh_registry.register_position_tp_sl("BTCUSDT", "LONG", 55000.0, 48000.0)
        fresh_registry.register_position_tp_sl("ETHUSDT", "SHORT", 2700.0, 3150.0)
        
        result = fresh_registry.get_all_tp_sl_positions()
        assert len(result) == 2
        assert "BTCUSDT:LONG" in result
        assert "ETHUSDT:SHORT" in result


# ===========================================================================
# Tests: enrich_positions_with_tp_sl
# ===========================================================================

class TestEnrichPositionsWithTpSl:
    def test_enrich_single_position(self, fresh_registry):
        """Enrich a single position with TP/SL data."""
        fresh_registry.register_position_tp_sl("BTCUSDT", "LONG", 55000.0, 48000.0)
        
        positions = [{"symbol": "BTCUSDT", "positionSide": "LONG", "qty": 0.5}]
        enriched = fresh_registry.enrich_positions_with_tp_sl(positions)
        
        assert len(enriched) == 1
        assert "tp_sl" in enriched[0]
        assert enriched[0]["tp_sl"]["take_profit_price"] == 55000.0

    def test_enrich_no_tp_sl(self, fresh_registry):
        """Positions without TP/SL are not enriched."""
        positions = [{"symbol": "BTCUSDT", "positionSide": "LONG", "qty": 0.5}]
        enriched = fresh_registry.enrich_positions_with_tp_sl(positions)
        
        assert len(enriched) == 1
        assert "tp_sl" not in enriched[0]

    def test_enrich_infers_side_from_qty(self, fresh_registry):
        """Side is inferred from qty when positionSide not present."""
        fresh_registry.register_position_tp_sl("BTCUSDT", "LONG", 55000.0, 48000.0)
        
        positions = [{"symbol": "BTCUSDT", "qty": 0.5}]  # No positionSide
        enriched = fresh_registry.enrich_positions_with_tp_sl(positions)
        
        assert "tp_sl" in enriched[0]

    def test_enrich_preserves_original(self, fresh_registry):
        """Original position data is preserved."""
        positions = [{"symbol": "BTCUSDT", "positionSide": "LONG", "qty": 0.5, "extra": "data"}]
        enriched = fresh_registry.enrich_positions_with_tp_sl(positions)
        
        assert enriched[0]["extra"] == "data"


# ===========================================================================
# Tests: cleanup_stale_entries
# ===========================================================================

class TestCleanupStaleEntries:
    def test_cleanup_removes_stale(self, fresh_registry):
        """Cleanup removes entries not in active list."""
        fresh_registry.register_position_tp_sl("BTCUSDT", "LONG", 55000.0, 48000.0)
        fresh_registry.register_position_tp_sl("ETHUSDT", "SHORT", 2700.0, 3150.0)
        fresh_registry.register_position_tp_sl("SOLUSDT", "LONG", 150.0, 90.0)
        
        # Only BTC and ETH are still active
        active = [
            {"symbol": "BTCUSDT", "positionSide": "LONG", "qty": 0.5},
            {"symbol": "ETHUSDT", "positionSide": "SHORT", "qty": -2.0},
        ]
        removed = fresh_registry.cleanup_stale_entries(active)
        
        assert removed == 1
        assert fresh_registry.get_position_tp_sl("BTCUSDT", "LONG") is not None
        assert fresh_registry.get_position_tp_sl("ETHUSDT", "SHORT") is not None
        assert fresh_registry.get_position_tp_sl("SOLUSDT", "LONG") is None

    def test_cleanup_all_active(self, fresh_registry):
        """Cleanup removes nothing when all entries are active."""
        fresh_registry.register_position_tp_sl("BTCUSDT", "LONG", 55000.0, 48000.0)
        fresh_registry.register_position_tp_sl("ETHUSDT", "SHORT", 2700.0, 3150.0)
        
        active = [
            {"symbol": "BTCUSDT", "positionSide": "LONG", "qty": 0.5},
            {"symbol": "ETHUSDT", "positionSide": "SHORT", "qty": -2.0},
        ]
        removed = fresh_registry.cleanup_stale_entries(active)
        
        assert removed == 0
        assert len(fresh_registry.get_all_tp_sl_positions()) == 2

    def test_cleanup_infers_side_from_qty(self, fresh_registry):
        """Cleanup can handle positions without positionSide."""
        fresh_registry.register_position_tp_sl("BTCUSDT", "LONG", 55000.0, 48000.0)
        
        active = [{"symbol": "BTCUSDT", "qty": 0.5}]  # qty > 0 = LONG
        removed = fresh_registry.cleanup_stale_entries(active)
        
        assert removed == 0
        assert fresh_registry.get_position_tp_sl("BTCUSDT", "LONG") is not None


# ===========================================================================
# Tests: Same symbol different sides
# ===========================================================================

class TestSameSymbolDifferentSides:
    def test_register_both_sides(self, fresh_registry):
        """Can have same symbol with different sides."""
        fresh_registry.register_position_tp_sl("BTCUSDT", "LONG", 55000.0, 48000.0)
        fresh_registry.register_position_tp_sl("BTCUSDT", "SHORT", 45000.0, 52000.0)
        
        long_entry = fresh_registry.get_position_tp_sl("BTCUSDT", "LONG")
        short_entry = fresh_registry.get_position_tp_sl("BTCUSDT", "SHORT")
        
        assert long_entry["take_profit_price"] == 55000.0
        assert short_entry["take_profit_price"] == 45000.0

    def test_unregister_one_side_only(self, fresh_registry):
        """Unregistering one side doesn't affect other."""
        fresh_registry.register_position_tp_sl("BTCUSDT", "LONG", 55000.0, 48000.0)
        fresh_registry.register_position_tp_sl("BTCUSDT", "SHORT", 45000.0, 52000.0)
        
        fresh_registry.unregister_position_tp_sl("BTCUSDT", "LONG")
        
        assert fresh_registry.get_position_tp_sl("BTCUSDT", "LONG") is None
        assert fresh_registry.get_position_tp_sl("BTCUSDT", "SHORT") is not None


# ===========================================================================
# Tests: Update existing entry
# ===========================================================================

class TestUpdateExisting:
    def test_update_overwrites(self, fresh_registry):
        """Registering with same key updates the entry."""
        fresh_registry.register_position_tp_sl("BTCUSDT", "LONG", 55000.0, 48000.0)
        fresh_registry.register_position_tp_sl("BTCUSDT", "LONG", 60000.0, 50000.0)
        
        result = fresh_registry.get_position_tp_sl("BTCUSDT", "LONG")
        assert result["take_profit_price"] == 60000.0
        assert result["stop_loss_price"] == 50000.0
