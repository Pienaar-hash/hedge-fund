"""
Unit tests for exit_scanner.py (PATCHSET V2 — Vol-Target Exit Scanner)
"""

import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any, Optional
import json
import os
import tempfile

from execution.exit_scanner import (
    ExitReason,
    ExitCandidate,
    _get_side,
    _get_last_price,
    _build_exit_candidate,
    scan_tp_sl_exits,
    build_exit_intent,
)


# ===========================================================================
# Fixture: Mock registry state
# ===========================================================================

@pytest.fixture
def mock_registry(monkeypatch):
    """Mock the registry module to avoid file I/O."""
    import execution.position_tp_sl_registry as reg
    
    # Clear and reset
    reg._TP_SL_REGISTRY.clear()
    reg._REGISTRY_LOADED = True  # Skip file load
    
    yield reg
    
    # Cleanup
    reg._TP_SL_REGISTRY.clear()
    reg._REGISTRY_LOADED = False


@pytest.fixture
def registry_with_entries(mock_registry, monkeypatch):
    """Pre-populate registry with test entries.
    
    V7.4_C3: Also mocks build_position_ledger to raise ImportError,
    forcing scan_tp_sl_exits to fall back to the legacy registry path.
    """
    # Mock ledger to force fallback to registry
    def mock_build_ledger():
        raise ImportError("Mocked - force fallback to registry")
    
    monkeypatch.setattr(
        "execution.position_ledger.build_position_ledger",
        mock_build_ledger,
    )
    
    mock_registry.register_position_tp_sl(
        symbol="BTCUSDT",
        position_side="LONG",
        take_profit_price=55000.0,
        stop_loss_price=48000.0,
    )
    mock_registry.register_position_tp_sl(
        symbol="ETHUSDT",
        position_side="SHORT",
        take_profit_price=2700.0,
        stop_loss_price=3150.0,
    )
    return mock_registry


# ===========================================================================
# Tests: _get_side
# ===========================================================================

class TestGetSide:
    def test_positionSide_long(self):
        pos = {"positionSide": "LONG", "qty": 1.0}
        assert _get_side(pos) == "LONG"

    def test_positionSide_short(self):
        pos = {"positionSide": "SHORT", "qty": -1.0}
        assert _get_side(pos) == "SHORT"

    def test_side_buy(self):
        pos = {"side": "BUY", "qty": 1.0}
        assert _get_side(pos) == "BUY"

    def test_side_sell(self):
        pos = {"side": "SELL", "qty": -1.0}
        assert _get_side(pos) == "SELL"

    def test_inferred_from_positive_qty(self):
        pos = {"qty": 0.5}
        assert _get_side(pos) == "LONG"

    def test_inferred_from_negative_qty(self):
        pos = {"qty": -0.5}
        assert _get_side(pos) == "SHORT"

    def test_inferred_from_positionAmt(self):
        pos = {"positionAmt": 1.5}
        assert _get_side(pos) == "LONG"

    def test_zero_qty_no_side(self):
        pos = {"qty": 0}
        assert _get_side(pos) is None

    def test_lowercase_normalized(self):
        pos = {"positionSide": "long"}
        assert _get_side(pos) == "LONG"


# ===========================================================================
# Tests: _get_last_price
# ===========================================================================

class TestGetLastPrice:
    def test_price_found(self):
        price_map = {"BTCUSDT": 50000.0, "ETHUSDT": 3000.0}
        assert _get_last_price("BTCUSDT", price_map) == 50000.0

    def test_price_not_found(self):
        price_map = {"BTCUSDT": 50000.0}
        assert _get_last_price("SOLUSDT", price_map) is None

    def test_empty_price_map(self):
        assert _get_last_price("BTCUSDT", {}) is None


# ===========================================================================
# Tests: _build_exit_candidate
# ===========================================================================

class TestBuildExitCandidate:
    def test_long_tp_hit(self):
        pos = {"symbol": "BTCUSDT", "positionSide": "LONG", "qty": 0.5}
        tp_sl = {"take_profit_price": 55000.0, "stop_loss_price": 48000.0, "entry_price": 50000.0}
        candidate = _build_exit_candidate(pos, 56000.0, tp_sl)
        
        assert candidate is not None
        assert candidate.symbol == "BTCUSDT"
        assert candidate.position_side == "LONG"
        assert candidate.qty == 0.5
        assert candidate.exit_reason == ExitReason.TAKE_PROFIT
        assert candidate.trigger_price == 56000.0
        assert candidate.tp_price == 55000.0
        assert candidate.sl_price == 48000.0

    def test_long_sl_hit(self):
        pos = {"symbol": "BTCUSDT", "positionSide": "LONG", "qty": 0.5}
        tp_sl = {"take_profit_price": 55000.0, "stop_loss_price": 48000.0}
        candidate = _build_exit_candidate(pos, 47000.0, tp_sl)
        
        assert candidate is not None
        assert candidate.exit_reason == ExitReason.STOP_LOSS
        assert candidate.trigger_price == 47000.0

    def test_short_tp_hit(self):
        pos = {"symbol": "ETHUSDT", "positionSide": "SHORT", "qty": -2.0}
        tp_sl = {"take_profit_price": 2700.0, "stop_loss_price": 3150.0}
        candidate = _build_exit_candidate(pos, 2600.0, tp_sl)
        
        assert candidate is not None
        assert candidate.exit_reason == ExitReason.TAKE_PROFIT
        assert candidate.position_side == "SHORT"
        assert candidate.qty == 2.0  # abs

    def test_short_sl_hit(self):
        pos = {"symbol": "ETHUSDT", "positionSide": "SHORT", "qty": -2.0}
        tp_sl = {"take_profit_price": 2700.0, "stop_loss_price": 3150.0}
        candidate = _build_exit_candidate(pos, 3200.0, tp_sl)
        
        assert candidate is not None
        assert candidate.exit_reason == ExitReason.STOP_LOSS

    def test_no_exit_within_range(self):
        pos = {"symbol": "BTCUSDT", "positionSide": "LONG", "qty": 0.5}
        tp_sl = {"take_profit_price": 55000.0, "stop_loss_price": 48000.0}
        candidate = _build_exit_candidate(pos, 51000.0, tp_sl)
        
        assert candidate is None

    def test_no_exit_without_tp_sl(self):
        pos = {"symbol": "BTCUSDT", "positionSide": "LONG", "qty": 0.5}
        candidate = _build_exit_candidate(pos, 51000.0, None)
        
        assert candidate is None

    def test_exact_tp_price_triggers(self):
        pos = {"symbol": "BTCUSDT", "positionSide": "LONG", "qty": 0.5}
        tp_sl = {"take_profit_price": 55000.0, "stop_loss_price": 48000.0}
        candidate = _build_exit_candidate(pos, 55000.0, tp_sl)
        
        assert candidate is not None
        assert candidate.exit_reason == ExitReason.TAKE_PROFIT

    def test_exact_sl_price_triggers(self):
        pos = {"symbol": "BTCUSDT", "positionSide": "LONG", "qty": 0.5}
        tp_sl = {"take_profit_price": 55000.0, "stop_loss_price": 48000.0}
        candidate = _build_exit_candidate(pos, 48000.0, tp_sl)
        
        assert candidate is not None
        assert candidate.exit_reason == ExitReason.STOP_LOSS

    def test_zero_qty_no_candidate(self):
        pos = {"symbol": "BTCUSDT", "positionSide": "LONG", "qty": 0.0}
        tp_sl = {"take_profit_price": 55000.0, "stop_loss_price": 48000.0}
        candidate = _build_exit_candidate(pos, 56000.0, tp_sl)
        
        assert candidate is None

    def test_missing_symbol_no_candidate(self):
        pos = {"positionSide": "LONG", "qty": 0.5}
        tp_sl = {"take_profit_price": 55000.0, "stop_loss_price": 48000.0}
        candidate = _build_exit_candidate(pos, 56000.0, tp_sl)
        
        assert candidate is None

    def test_tp_only_hit(self):
        """Only TP set, no SL."""
        pos = {"symbol": "BTCUSDT", "positionSide": "LONG", "qty": 0.5}
        tp_sl = {"take_profit_price": 55000.0}
        candidate = _build_exit_candidate(pos, 56000.0, tp_sl)
        
        assert candidate is not None
        assert candidate.exit_reason == ExitReason.TAKE_PROFIT
        assert candidate.sl_price is None

    def test_sl_only_hit(self):
        """Only SL set, no TP."""
        pos = {"symbol": "BTCUSDT", "positionSide": "LONG", "qty": 0.5}
        tp_sl = {"stop_loss_price": 48000.0}
        candidate = _build_exit_candidate(pos, 47000.0, tp_sl)
        
        assert candidate is not None
        assert candidate.exit_reason == ExitReason.STOP_LOSS
        assert candidate.tp_price is None


# ===========================================================================
# Tests: scan_tp_sl_exits (with registry)
# ===========================================================================

class TestScanTpSlExits:
    def test_finds_tp_hit(self, registry_with_entries):
        """Find TP exit when LONG position price exceeds TP level."""
        positions = [
            {"symbol": "BTCUSDT", "positionSide": "LONG", "qty": 0.5},
        ]
        price_map = {"BTCUSDT": 56000.0}  # Above TP of 55000
        
        exits = scan_tp_sl_exits(positions, price_map)
        
        assert len(exits) == 1
        assert exits[0].symbol == "BTCUSDT"
        assert exits[0].exit_reason == ExitReason.TAKE_PROFIT

    def test_finds_sl_hit(self, registry_with_entries):
        """Find SL exit when LONG position price drops below SL level."""
        positions = [
            {"symbol": "BTCUSDT", "positionSide": "LONG", "qty": 0.5},
        ]
        price_map = {"BTCUSDT": 47000.0}  # Below SL of 48000
        
        exits = scan_tp_sl_exits(positions, price_map)
        
        assert len(exits) == 1
        assert exits[0].exit_reason == ExitReason.STOP_LOSS

    def test_finds_short_tp(self, registry_with_entries):
        """Find TP exit on SHORT position when price drops below TP."""
        positions = [
            {"symbol": "ETHUSDT", "positionSide": "SHORT", "qty": -2.0},
        ]
        price_map = {"ETHUSDT": 2600.0}  # Below TP of 2700
        
        exits = scan_tp_sl_exits(positions, price_map)
        
        assert len(exits) == 1
        assert exits[0].exit_reason == ExitReason.TAKE_PROFIT

    def test_finds_short_sl(self, registry_with_entries):
        """Find SL exit on SHORT position when price rises above SL."""
        positions = [
            {"symbol": "ETHUSDT", "positionSide": "SHORT", "qty": -2.0},
        ]
        price_map = {"ETHUSDT": 3200.0}  # Above SL of 3150
        
        exits = scan_tp_sl_exits(positions, price_map)
        
        assert len(exits) == 1
        assert exits[0].exit_reason == ExitReason.STOP_LOSS

    def test_no_exit_within_range(self, registry_with_entries):
        """No exit when price is within TP/SL range."""
        positions = [
            {"symbol": "BTCUSDT", "positionSide": "LONG", "qty": 0.5},
        ]
        price_map = {"BTCUSDT": 51000.0}  # Between 48000 SL and 55000 TP
        
        exits = scan_tp_sl_exits(positions, price_map)
        
        assert len(exits) == 0

    def test_position_not_in_registry(self, registry_with_entries):
        """No exit for position without registry entry."""
        positions = [
            {"symbol": "SOLUSDT", "positionSide": "LONG", "qty": 10.0},
        ]
        price_map = {"SOLUSDT": 999.0}  # Doesn't matter - no registry entry
        
        exits = scan_tp_sl_exits(positions, price_map)
        
        assert len(exits) == 0

    def test_no_price_available(self, registry_with_entries):
        """No exit when price is not available."""
        positions = [
            {"symbol": "BTCUSDT", "positionSide": "LONG", "qty": 0.5},
        ]
        price_map = {}  # Empty price map
        
        exits = scan_tp_sl_exits(positions, price_map)
        
        assert len(exits) == 0

    def test_multiple_positions(self, registry_with_entries):
        """Scan multiple positions and find multiple exits."""
        positions = [
            {"symbol": "BTCUSDT", "positionSide": "LONG", "qty": 0.5},
            {"symbol": "ETHUSDT", "positionSide": "SHORT", "qty": -2.0},
        ]
        # BTC above TP, ETH below TP (both in profit)
        price_map = {"BTCUSDT": 56000.0, "ETHUSDT": 2600.0}
        
        exits = scan_tp_sl_exits(positions, price_map)
        
        assert len(exits) == 2
        symbols = {e.symbol for e in exits}
        assert "BTCUSDT" in symbols
        assert "ETHUSDT" in symbols


# ===========================================================================
# Tests: build_exit_intent
# ===========================================================================

class TestBuildExitIntent:
    def test_long_exit_intent(self):
        """Exit intent for LONG position should SELL."""
        candidate = ExitCandidate(
            symbol="BTCUSDT",
            position_side="LONG",
            qty=0.5,
            exit_reason=ExitReason.TAKE_PROFIT,
            trigger_price=56000.0,
            tp_price=55000.0,
            sl_price=48000.0,
            entry_price=50000.0,
        )
        
        intent = build_exit_intent(candidate)
        
        assert intent["symbol"] == "BTCUSDT"
        assert intent["signal"] == "SELL"
        assert intent["reduceOnly"] is True
        assert intent["positionSide"] == "LONG"
        assert intent["quantity"] == 0.5
        assert intent["metadata"]["strategy"] == "vol_target_exit"
        assert intent["metadata"]["exit_reason"] == "tp"
        assert intent["metadata"]["trigger_price"] == 56000.0

    def test_short_exit_intent(self):
        """Exit intent for SHORT position should BUY."""
        candidate = ExitCandidate(
            symbol="ETHUSDT",
            position_side="SHORT",
            qty=2.0,
            exit_reason=ExitReason.STOP_LOSS,
            trigger_price=3200.0,
            tp_price=2700.0,
            sl_price=3150.0,
            entry_price=3000.0,
        )
        
        intent = build_exit_intent(candidate)
        
        assert intent["signal"] == "BUY"
        assert intent["positionSide"] == "SHORT"
        assert intent["metadata"]["exit_reason"] == "sl"

    def test_intent_has_timestamp(self):
        candidate = ExitCandidate(
            symbol="BTCUSDT",
            position_side="LONG",
            qty=0.5,
            exit_reason=ExitReason.TAKE_PROFIT,
            trigger_price=56000.0,
            tp_price=55000.0,
            sl_price=48000.0,
        )
        
        intent = build_exit_intent(candidate)
        
        assert "timestamp" in intent
        assert "generated_at" in intent


# ===========================================================================
# Tests: Registry integration
# ===========================================================================

class TestRegistryIntegration:
    def test_register_and_get_entry(self, mock_registry):
        """Test registering and getting registry entries."""
        mock_registry.register_position_tp_sl(
            symbol="BTCUSDT",
            position_side="LONG",
            take_profit_price=55000.0,
            stop_loss_price=48000.0,
        )
        
        entry = mock_registry.get_position_tp_sl("BTCUSDT", "LONG")
        
        assert entry is not None
        assert entry["take_profit_price"] == 55000.0
        assert entry["stop_loss_price"] == 48000.0

    def test_unregister_entry(self, mock_registry):
        """Test unregistering entries."""
        mock_registry.register_position_tp_sl(
            symbol="BTCUSDT",
            position_side="LONG",
            take_profit_price=55000.0,
            stop_loss_price=48000.0,
        )
        
        mock_registry.unregister_position_tp_sl("BTCUSDT", "LONG")
        
        entry = mock_registry.get_position_tp_sl("BTCUSDT", "LONG")
        assert entry is None

    def test_cleanup_stale_entries(self, mock_registry):
        """Test cleanup of stale entries."""
        mock_registry.register_position_tp_sl("BTCUSDT", "LONG", 55000.0, 48000.0)
        mock_registry.register_position_tp_sl("ETHUSDT", "SHORT", 2700.0, 3150.0)
        
        # Only BTC is active
        active = [{"symbol": "BTCUSDT", "positionSide": "LONG", "qty": 0.5}]
        removed = mock_registry.cleanup_stale_entries(active)
        
        assert removed == 1
        assert mock_registry.get_position_tp_sl("BTCUSDT", "LONG") is not None
        assert mock_registry.get_position_tp_sl("ETHUSDT", "SHORT") is None


# ===========================================================================
# Tests: ExitReason enum
# ===========================================================================

class TestExitReason:
    def test_tp_value(self):
        assert ExitReason.TAKE_PROFIT.value == "tp"

    def test_sl_value(self):
        assert ExitReason.STOP_LOSS.value == "sl"

    def test_is_string_enum(self):
        assert str(ExitReason.TAKE_PROFIT) == "ExitReason.TAKE_PROFIT"
        assert ExitReason.TAKE_PROFIT == "tp"
