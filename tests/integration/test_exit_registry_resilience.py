"""
Tests for TP/SL registry resilience (v7.4_C2).

Validates:
1. seed_missing_entries populates registry for open positions
2. cleanup_stale_entries removes registry entries for closed positions
3. sync_registry_with_positions handles both operations
4. ATR-based TP/SL computation is correct for LONG and SHORT
"""

from __future__ import annotations

import json
import time
import pytest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def temp_registry(tmp_path: Path):
    """Create a temporary registry file and patch the module to use it."""
    registry_path = tmp_path / "position_tp_sl.json"
    
    # Reset module state
    import execution.position_tp_sl_registry as reg
    original_path = reg._REGISTRY_PATH
    original_registry = reg._TP_SL_REGISTRY.copy()
    original_loaded = reg._REGISTRY_LOADED
    
    reg._REGISTRY_PATH = registry_path
    reg._TP_SL_REGISTRY = {}
    reg._REGISTRY_LOADED = False
    
    yield registry_path
    
    # Restore
    reg._REGISTRY_PATH = original_path
    reg._TP_SL_REGISTRY = original_registry
    reg._REGISTRY_LOADED = original_loaded


@pytest.fixture
def sample_long_positions():
    """Sample LONG positions from exchange."""
    return [
        {
            "symbol": "BTCUSDT",
            "positionSide": "LONG",
            "positionAmt": 0.1,
            "entryPrice": 50000.0,
            "unRealizedProfit": -100.0,
        },
        {
            "symbol": "ETHUSDT",
            "positionSide": "LONG",
            "positionAmt": 1.5,
            "entryPrice": 3000.0,
            "unRealizedProfit": 50.0,
        },
    ]


@pytest.fixture
def sample_short_position():
    """Sample SHORT position from exchange."""
    return {
        "symbol": "SOLUSDT",
        "positionSide": "SHORT",
        "positionAmt": -10.0,
        "entryPrice": 150.0,
        "unRealizedProfit": 20.0,
    }


# -----------------------------------------------------------------------------
# Tests: seed_missing_entries
# -----------------------------------------------------------------------------

class TestSeedMissingEntries:
    """Tests for seed_missing_entries function."""
    
    def test_seeds_long_positions(self, temp_registry, sample_long_positions):
        """LONG positions get TP above entry and SL below entry."""
        from execution.position_tp_sl_registry import seed_missing_entries, get_position_tp_sl
        
        seeded = seed_missing_entries(sample_long_positions, sl_atr_mult=2.0, tp_atr_mult=3.0)
        
        assert seeded == 2
        
        # Check BTC entry
        btc = get_position_tp_sl("BTCUSDT", "LONG")
        assert btc is not None
        assert btc["entry_price"] == 50000.0
        # For LONG: TP > entry > SL
        assert btc["take_profit_price"] > btc["entry_price"]
        assert btc["stop_loss_price"] < btc["entry_price"]
        assert btc["source"] == "startup_seed"
        
        # Check ETH entry
        eth = get_position_tp_sl("ETHUSDT", "LONG")
        assert eth is not None
        assert eth["take_profit_price"] > eth["entry_price"]
        assert eth["stop_loss_price"] < eth["entry_price"]
    
    def test_seeds_short_positions(self, temp_registry, sample_short_position):
        """SHORT positions get TP below entry and SL above entry."""
        from execution.position_tp_sl_registry import seed_missing_entries, get_position_tp_sl
        
        seeded = seed_missing_entries([sample_short_position], sl_atr_mult=2.0, tp_atr_mult=3.0)
        
        assert seeded == 1
        
        sol = get_position_tp_sl("SOLUSDT", "SHORT")
        assert sol is not None
        assert sol["entry_price"] == 150.0
        # For SHORT: SL > entry > TP
        assert sol["stop_loss_price"] > sol["entry_price"]
        assert sol["take_profit_price"] < sol["entry_price"]
    
    def test_skips_already_registered(self, temp_registry, sample_long_positions):
        """Positions already in registry are not re-seeded."""
        from execution.position_tp_sl_registry import (
            seed_missing_entries,
            register_position_tp_sl,
            get_position_tp_sl,
        )
        
        # Pre-register one position
        register_position_tp_sl(
            symbol="BTCUSDT",
            position_side="LONG",
            take_profit_price=55000.0,
            stop_loss_price=45000.0,
            metadata={"source": "original"},
        )
        
        # Seed should only add the new one
        seeded = seed_missing_entries(sample_long_positions)
        assert seeded == 1
        
        # BTC should retain original values
        btc = get_position_tp_sl("BTCUSDT", "LONG")
        assert btc["take_profit_price"] == 55000.0
        assert btc["stop_loss_price"] == 45000.0
    
    def test_skips_zero_qty_positions(self, temp_registry):
        """Positions with zero quantity are skipped."""
        from execution.position_tp_sl_registry import seed_missing_entries
        
        positions = [
            {"symbol": "BTCUSDT", "positionSide": "LONG", "positionAmt": 0.0, "entryPrice": 50000.0},
        ]
        
        seeded = seed_missing_entries(positions)
        assert seeded == 0
    
    def test_skips_invalid_entry_price(self, temp_registry):
        """Positions with zero or negative entry price are skipped."""
        from execution.position_tp_sl_registry import seed_missing_entries
        
        positions = [
            {"symbol": "BTCUSDT", "positionSide": "LONG", "positionAmt": 0.1, "entryPrice": 0.0},
            {"symbol": "ETHUSDT", "positionSide": "LONG", "positionAmt": 0.1, "entryPrice": -100.0},
        ]
        
        seeded = seed_missing_entries(positions)
        assert seeded == 0
    
    def test_atr_calculation(self, temp_registry):
        """ATR estimate uses fallback percentage correctly."""
        from execution.position_tp_sl_registry import seed_missing_entries, get_position_tp_sl
        
        positions = [
            {"symbol": "BTCUSDT", "positionSide": "LONG", "positionAmt": 0.1, "entryPrice": 100.0},
        ]
        
        # With 1% ATR fallback, SL mult 2, TP mult 3
        # ATR = 100 * 0.01 = 1.0
        # SL = 100 - (1.0 * 2) = 98.0
        # TP = 100 + (1.0 * 3) = 103.0
        seeded = seed_missing_entries(positions, sl_atr_mult=2.0, tp_atr_mult=3.0, atr_fallback_pct=0.01)
        
        assert seeded == 1
        btc = get_position_tp_sl("BTCUSDT", "LONG")
        assert btc["stop_loss_price"] == pytest.approx(98.0)
        assert btc["take_profit_price"] == pytest.approx(103.0)
        assert btc["metadata"]["atr_estimate"] == pytest.approx(1.0)


# -----------------------------------------------------------------------------
# Tests: cleanup_stale_entries
# -----------------------------------------------------------------------------

class TestCleanupStaleEntries:
    """Tests for cleanup_stale_entries function."""
    
    def test_removes_stale_entries(self, temp_registry, sample_long_positions):
        """Entries for closed positions are removed."""
        from execution.position_tp_sl_registry import (
            register_position_tp_sl,
            cleanup_stale_entries,
            get_position_tp_sl,
        )
        
        # Register positions
        register_position_tp_sl("BTCUSDT", "LONG", 55000.0, 45000.0)
        register_position_tp_sl("ETHUSDT", "LONG", 3500.0, 2500.0)
        register_position_tp_sl("XRPUSDT", "LONG", 0.7, 0.5)  # Not in active positions
        
        # Only BTC and ETH are active
        removed = cleanup_stale_entries(sample_long_positions)
        
        assert removed == 1
        assert get_position_tp_sl("BTCUSDT", "LONG") is not None
        assert get_position_tp_sl("ETHUSDT", "LONG") is not None
        assert get_position_tp_sl("XRPUSDT", "LONG") is None
    
    def test_no_stale_entries(self, temp_registry, sample_long_positions):
        """No removal when all entries are active."""
        from execution.position_tp_sl_registry import (
            register_position_tp_sl,
            cleanup_stale_entries,
        )
        
        register_position_tp_sl("BTCUSDT", "LONG", 55000.0, 45000.0)
        register_position_tp_sl("ETHUSDT", "LONG", 3500.0, 2500.0)
        
        removed = cleanup_stale_entries(sample_long_positions)
        assert removed == 0
    
    def test_clears_all_when_no_positions(self, temp_registry):
        """All entries removed when no active positions."""
        from execution.position_tp_sl_registry import (
            register_position_tp_sl,
            cleanup_stale_entries,
            get_all_tp_sl_positions,
        )
        
        register_position_tp_sl("BTCUSDT", "LONG", 55000.0, 45000.0)
        register_position_tp_sl("ETHUSDT", "LONG", 3500.0, 2500.0)
        
        removed = cleanup_stale_entries([])
        
        assert removed == 2
        assert len(get_all_tp_sl_positions()) == 0


# -----------------------------------------------------------------------------
# Tests: sync_registry_with_positions
# -----------------------------------------------------------------------------

class TestSyncRegistryWithPositions:
    """Tests for sync_registry_with_positions function."""
    
    def test_full_sync_scenario(self, temp_registry, sample_long_positions):
        """Sync removes stale and adds missing entries."""
        from execution.position_tp_sl_registry import (
            register_position_tp_sl,
            sync_registry_with_positions,
            get_position_tp_sl,
        )
        
        # Start with one active (BTC) and one stale (XRP)
        register_position_tp_sl("BTCUSDT", "LONG", 55000.0, 45000.0)
        register_position_tp_sl("XRPUSDT", "LONG", 0.7, 0.5)
        
        result = sync_registry_with_positions(sample_long_positions)
        
        # XRP should be removed, ETH should be added
        assert result["stale_removed"] == 1
        assert result["new_seeded"] == 1
        
        assert get_position_tp_sl("BTCUSDT", "LONG") is not None
        assert get_position_tp_sl("ETHUSDT", "LONG") is not None
        assert get_position_tp_sl("XRPUSDT", "LONG") is None
    
    def test_empty_positions_clears_registry(self, temp_registry):
        """Empty position list clears all registry entries."""
        from execution.position_tp_sl_registry import (
            register_position_tp_sl,
            sync_registry_with_positions,
            get_all_tp_sl_positions,
        )
        
        register_position_tp_sl("BTCUSDT", "LONG", 55000.0, 45000.0)
        register_position_tp_sl("ETHUSDT", "LONG", 3500.0, 2500.0)
        
        result = sync_registry_with_positions([])
        
        assert result["stale_removed"] == 2
        assert result["new_seeded"] == 0
        assert len(get_all_tp_sl_positions()) == 0
    
    def test_already_synced_is_noop(self, temp_registry, sample_long_positions):
        """Synced registry requires no changes."""
        from execution.position_tp_sl_registry import (
            register_position_tp_sl,
            sync_registry_with_positions,
        )
        
        register_position_tp_sl("BTCUSDT", "LONG", 55000.0, 45000.0)
        register_position_tp_sl("ETHUSDT", "LONG", 3500.0, 2500.0)
        
        result = sync_registry_with_positions(sample_long_positions)
        
        assert result["stale_removed"] == 0
        assert result["new_seeded"] == 0


# -----------------------------------------------------------------------------
# Tests: Persistence
# -----------------------------------------------------------------------------

class TestRegistryPersistence:
    """Tests for registry file persistence."""
    
    def test_seed_persists_to_file(self, temp_registry, sample_long_positions):
        """Seeded entries are written to disk."""
        from execution.position_tp_sl_registry import seed_missing_entries
        
        seed_missing_entries(sample_long_positions)
        
        assert temp_registry.exists()
        data = json.loads(temp_registry.read_text())
        assert "entries" in data
        assert len(data["entries"]) == 2
        assert "BTCUSDT:LONG" in data["entries"]
        assert "ETHUSDT:LONG" in data["entries"]
    
    def test_cleanup_persists_to_file(self, temp_registry, sample_long_positions):
        """Cleanup changes are written to disk."""
        from execution.position_tp_sl_registry import (
            register_position_tp_sl,
            cleanup_stale_entries,
        )
        
        register_position_tp_sl("BTCUSDT", "LONG", 55000.0, 45000.0)
        register_position_tp_sl("XRPUSDT", "LONG", 0.7, 0.5)
        
        # Only keep BTC
        cleanup_stale_entries([sample_long_positions[0]])
        
        data = json.loads(temp_registry.read_text())
        assert "BTCUSDT:LONG" in data["entries"]
        assert "XRPUSDT:LONG" not in data["entries"]


# -----------------------------------------------------------------------------
# Tests: Edge cases
# -----------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case tests for registry resilience."""
    
    def test_handles_alternative_position_format(self, temp_registry):
        """Handles positions with 'side' instead of 'positionSide'."""
        from execution.position_tp_sl_registry import seed_missing_entries, get_position_tp_sl
        
        positions = [
            {"symbol": "BTCUSDT", "side": "LONG", "qty": 0.1, "entry_price": 50000.0},
        ]
        
        seeded = seed_missing_entries(positions)
        
        assert seeded == 1
        assert get_position_tp_sl("BTCUSDT", "LONG") is not None
    
    def test_infers_side_from_qty_sign(self, temp_registry):
        """Infers position side from quantity sign when not specified."""
        from execution.position_tp_sl_registry import seed_missing_entries, get_position_tp_sl
        
        positions = [
            {"symbol": "BTCUSDT", "positionAmt": 0.1, "entryPrice": 50000.0},  # Positive = LONG
            {"symbol": "ETHUSDT", "positionAmt": -1.5, "entryPrice": 3000.0},  # Negative = SHORT
        ]
        
        seeded = seed_missing_entries(positions)
        
        assert seeded == 2
        assert get_position_tp_sl("BTCUSDT", "LONG") is not None
        assert get_position_tp_sl("ETHUSDT", "SHORT") is not None
    
    def test_symbol_case_normalization(self, temp_registry):
        """Symbols are normalized to uppercase."""
        from execution.position_tp_sl_registry import seed_missing_entries, get_position_tp_sl
        
        positions = [
            {"symbol": "btcusdt", "positionSide": "long", "positionAmt": 0.1, "entryPrice": 50000.0},
        ]
        
        seeded = seed_missing_entries(positions)
        
        assert seeded == 1
        # Should be accessible with uppercase
        assert get_position_tp_sl("BTCUSDT", "LONG") is not None
