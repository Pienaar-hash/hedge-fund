"""
Tests for Exit Scanner + Position Ledger Integration (v7.4_C3)

Tests:
1. Exit scanner uses ledger for TP/SL lookup
2. LONG position hit SL -> exit intent
3. LONG position hit TP -> exit intent
4. Position with no TP/SL -> no exit
"""

import json
import pytest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch, MagicMock

from execution.exit_scanner import (
    ExitCandidate,
    ExitReason,
    scan_tp_sl_exits,
    build_exit_intent,
    _build_exit_candidate,
)
from execution.position_ledger import (
    PositionLedgerEntry,
    PositionTP_SL,
)


@pytest.fixture
def temp_state_dir(tmp_path):
    """Create temporary state directory."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


@pytest.fixture
def sample_ledger_entries():
    """Sample ledger entries for testing."""
    return {
        "BTCUSDT:LONG": PositionLedgerEntry(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=Decimal("50000"),
            qty=Decimal("0.5"),
            tp_sl=PositionTP_SL(tp=Decimal("55000"), sl=Decimal("47000")),
        ),
        "ETHUSDT:SHORT": PositionLedgerEntry(
            symbol="ETHUSDT",
            side="SHORT",
            entry_price=Decimal("3000"),
            qty=Decimal("2.0"),
            tp_sl=PositionTP_SL(tp=Decimal("2800"), sl=Decimal("3200")),
        ),
        "SOLUSDT:LONG": PositionLedgerEntry(
            symbol="SOLUSDT",
            side="LONG",
            entry_price=Decimal("100"),
            qty=Decimal("10.0"),
            tp_sl=PositionTP_SL(tp=None, sl=None),  # No TP/SL
        ),
    }


class TestExitScannerLedgerIntegration:
    """Test exit scanner with position ledger."""

    def test_long_position_hits_stop_loss(self, sample_ledger_entries):
        """Test LONG position with mark < SL triggers exit."""
        price_map = {
            "BTCUSDT": 46000.0,  # Below SL of 47000
            "ETHUSDT": 3000.0,
            "SOLUSDT": 100.0,
        }

        with patch("execution.position_ledger.build_position_ledger") as mock_ledger:
            mock_ledger.return_value = sample_ledger_entries

            results = scan_tp_sl_exits([], price_map)

            assert len(results) == 1
            candidate = results[0]
            assert candidate.symbol == "BTCUSDT"
            assert candidate.position_side == "LONG"
            assert candidate.exit_reason == ExitReason.STOP_LOSS
            assert candidate.trigger_price == 46000.0
            assert candidate.sl_price == 47000.0
            assert candidate.qty == 0.5

    def test_long_position_hits_take_profit(self, sample_ledger_entries):
        """Test LONG position with mark > TP triggers exit."""
        price_map = {
            "BTCUSDT": 56000.0,  # Above TP of 55000
            "ETHUSDT": 3000.0,
            "SOLUSDT": 100.0,
        }

        with patch("execution.position_ledger.build_position_ledger") as mock_ledger:
            mock_ledger.return_value = sample_ledger_entries

            results = scan_tp_sl_exits([], price_map)

            assert len(results) == 1
            candidate = results[0]
            assert candidate.symbol == "BTCUSDT"
            assert candidate.exit_reason == ExitReason.TAKE_PROFIT
            assert candidate.trigger_price == 56000.0
            assert candidate.tp_price == 55000.0

    def test_short_position_hits_stop_loss(self, sample_ledger_entries):
        """Test SHORT position with mark > SL triggers exit."""
        price_map = {
            "BTCUSDT": 50000.0,
            "ETHUSDT": 3300.0,  # Above SL of 3200
            "SOLUSDT": 100.0,
        }

        with patch("execution.position_ledger.build_position_ledger") as mock_ledger:
            mock_ledger.return_value = sample_ledger_entries

            results = scan_tp_sl_exits([], price_map)

            assert len(results) == 1
            candidate = results[0]
            assert candidate.symbol == "ETHUSDT"
            assert candidate.position_side == "SHORT"
            assert candidate.exit_reason == ExitReason.STOP_LOSS
            assert candidate.trigger_price == 3300.0
            assert candidate.sl_price == 3200.0

    def test_short_position_hits_take_profit(self, sample_ledger_entries):
        """Test SHORT position with mark < TP triggers exit."""
        price_map = {
            "BTCUSDT": 50000.0,
            "ETHUSDT": 2700.0,  # Below TP of 2800
            "SOLUSDT": 100.0,
        }

        with patch("execution.position_ledger.build_position_ledger") as mock_ledger:
            mock_ledger.return_value = sample_ledger_entries

            results = scan_tp_sl_exits([], price_map)

            assert len(results) == 1
            candidate = results[0]
            assert candidate.symbol == "ETHUSDT"
            assert candidate.exit_reason == ExitReason.TAKE_PROFIT
            assert candidate.trigger_price == 2700.0
            assert candidate.tp_price == 2800.0

    def test_position_without_tp_sl_is_skipped(self, sample_ledger_entries):
        """Test position with no TP/SL does not trigger exit."""
        price_map = {
            "BTCUSDT": 50000.0,
            "ETHUSDT": 3000.0,
            "SOLUSDT": 50.0,  # Way below any reasonable SL, but no SL set
        }

        with patch("execution.position_ledger.build_position_ledger") as mock_ledger:
            mock_ledger.return_value = sample_ledger_entries

            results = scan_tp_sl_exits([], price_map)

            # SOLUSDT should NOT be in results (no TP/SL)
            assert len(results) == 0 or not any(r.symbol == "SOLUSDT" for r in results)

    def test_no_price_available_skips_position(self, sample_ledger_entries):
        """Test positions without price data are skipped."""
        price_map = {
            # Missing BTCUSDT price
            "ETHUSDT": 2700.0,  # This would trigger TP
            "SOLUSDT": 100.0,
        }

        with patch("execution.position_ledger.build_position_ledger") as mock_ledger:
            mock_ledger.return_value = sample_ledger_entries

            results = scan_tp_sl_exits([], price_map)

            # Only ETH should trigger (BTC has no price)
            assert len(results) == 1
            assert results[0].symbol == "ETHUSDT"

    def test_multiple_exits_same_scan(self, sample_ledger_entries):
        """Test multiple positions triggering exits in same scan."""
        price_map = {
            "BTCUSDT": 46000.0,  # SL hit
            "ETHUSDT": 2700.0,   # TP hit
            "SOLUSDT": 100.0,
        }

        with patch("execution.position_ledger.build_position_ledger") as mock_ledger:
            mock_ledger.return_value = sample_ledger_entries

            results = scan_tp_sl_exits([], price_map)

            assert len(results) == 2
            symbols = {r.symbol for r in results}
            assert "BTCUSDT" in symbols
            assert "ETHUSDT" in symbols


class TestBuildExitIntent:
    """Test exit intent construction."""

    def test_build_exit_intent_for_long_sl(self):
        """Test building exit intent for LONG position SL."""
        candidate = ExitCandidate(
            symbol="BTCUSDT",
            position_side="LONG",
            qty=0.5,
            exit_reason=ExitReason.STOP_LOSS,
            trigger_price=46000.0,
            tp_price=55000.0,
            sl_price=47000.0,
            entry_price=50000.0,
        )

        intent = build_exit_intent(candidate)

        assert intent["symbol"] == "BTCUSDT"
        assert intent["signal"] == "SELL"  # Close LONG = SELL
        assert intent["reduceOnly"] is True
        assert intent["positionSide"] == "LONG"
        assert intent["quantity"] == 0.5
        assert intent["metadata"]["exit_reason"] == "sl"
        assert intent["metadata"]["trigger_price"] == 46000.0

    def test_build_exit_intent_for_short_tp(self):
        """Test building exit intent for SHORT position TP."""
        candidate = ExitCandidate(
            symbol="ETHUSDT",
            position_side="SHORT",
            qty=2.0,
            exit_reason=ExitReason.TAKE_PROFIT,
            trigger_price=2700.0,
            tp_price=2800.0,
            sl_price=3200.0,
            entry_price=3000.0,
        )

        intent = build_exit_intent(candidate)

        assert intent["symbol"] == "ETHUSDT"
        assert intent["signal"] == "BUY"  # Close SHORT = BUY
        assert intent["reduceOnly"] is True
        assert intent["positionSide"] == "SHORT"
        assert intent["quantity"] == 2.0
        assert intent["metadata"]["exit_reason"] == "tp"


class TestLegacyFallback:
    """Test fallback to legacy registry when ledger unavailable."""

    def test_fallback_to_registry_on_import_error(self):
        """Test that scanner falls back to registry if ledger import fails."""
        positions = [
            {"symbol": "BTCUSDT", "positionSide": "LONG", "qty": 0.5},
        ]
        price_map = {"BTCUSDT": 46000.0}
        
        mock_registry_entry = {
            "take_profit_price": 55000.0,
            "stop_loss_price": 47000.0,
            "entry_price": 50000.0,
        }

        # Mock ledger import to fail, then registry to work
        with patch("execution.position_ledger.build_position_ledger", side_effect=ImportError):
            with patch("execution.position_tp_sl_registry.get_position_tp_sl", return_value=mock_registry_entry):
                # This should fall back to registry-based scan
                results = scan_tp_sl_exits(positions, price_map)

                # Should still find the exit using legacy path
                assert len(results) == 1
                assert results[0].symbol == "BTCUSDT"
                assert results[0].exit_reason == ExitReason.STOP_LOSS


class TestBuildExitCandidateDirectly:
    """Test the _build_exit_candidate helper directly."""

    def test_long_no_hit(self):
        """LONG position with price between SL and TP returns None."""
        pos = {"symbol": "BTCUSDT", "positionSide": "LONG", "qty": 0.5}
        tp_sl = {"take_profit_price": 55000.0, "stop_loss_price": 47000.0}
        
        result = _build_exit_candidate(pos, 50000.0, tp_sl)
        assert result is None

    def test_short_no_hit(self):
        """SHORT position with price between TP and SL returns None."""
        pos = {"symbol": "ETHUSDT", "positionSide": "SHORT", "qty": -2.0}
        tp_sl = {"take_profit_price": 2800.0, "stop_loss_price": 3200.0}
        
        result = _build_exit_candidate(pos, 3000.0, tp_sl)
        assert result is None

    def test_zero_qty_returns_none(self):
        """Zero qty position returns None."""
        pos = {"symbol": "BTCUSDT", "positionSide": "LONG", "qty": 0}
        tp_sl = {"take_profit_price": 55000.0, "stop_loss_price": 47000.0}
        
        result = _build_exit_candidate(pos, 46000.0, tp_sl)
        assert result is None
