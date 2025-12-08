"""
Tests for Position Ledger (v7.4_C3)

Tests:
1. Basic merge of positions_state + tp_sl registry
2. sync_ledger_with_positions() seeding
3. Stale entry removal
4. Edge formats (alternative key names)
"""

import json
import pytest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from execution.position_ledger import (
    PositionLedgerEntry,
    PositionTP_SL,
    build_position_ledger,
    sync_ledger_with_positions,
    load_positions_state,
    load_tp_sl_registry,
    save_tp_sl_registry,
    upsert_tp_sl,
    delete_tp_sl,
    ledger_to_dict,
    get_ledger_summary,
    _make_key,
    _normalize_side,
    _normalize_entry_price,
    _normalize_qty,
)


@pytest.fixture
def temp_state_dir(tmp_path):
    """Create temporary state directory."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


@pytest.fixture
def sample_positions_state():
    """Sample positions_state.json content."""
    return {
        "positions": [
            {
                "symbol": "BTCUSDT",
                "positionSide": "LONG",
                "qty": "0.5",
                "entryPrice": "50000.00",
                "updateTime": 1700000000000,
            },
            {
                "symbol": "ETHUSDT",
                "positionSide": "SHORT",
                "qty": "-2.0",
                "entryPrice": "3000.00",
                "updateTime": 1700000001000,
            },
        ],
        "updated": 1700000002.0,
    }


@pytest.fixture
def sample_tp_sl_registry():
    """Sample position_tp_sl.json content."""
    return {
        "entries": {
            "BTCUSDT:LONG": {
                "symbol": "BTCUSDT",
                "position_side": "LONG",
                "entry_price": 50000.0,
                "take_profit_price": 55000.0,
                "stop_loss_price": 47000.0,
                "enable_tp_sl": True,
            },
            # Note: ETHUSDT:SHORT is missing - will test seeding
        },
        "updated_at": 1700000000.0,
    }


class TestNormalization:
    """Test field normalization helpers."""

    def test_normalize_side_positionSide(self):
        assert _normalize_side({"positionSide": "LONG"}) == "LONG"
        assert _normalize_side({"positionSide": "SHORT"}) == "SHORT"
        assert _normalize_side({"positionSide": "BUY"}) == "LONG"
        assert _normalize_side({"positionSide": "SELL"}) == "SHORT"

    def test_normalize_side_from_qty(self):
        assert _normalize_side({"qty": 1.0}) == "LONG"
        assert _normalize_side({"qty": -1.0}) == "SHORT"
        assert _normalize_side({"positionAmt": 0.5}) == "LONG"
        assert _normalize_side({"positionAmt": -0.5}) == "SHORT"

    def test_normalize_side_zero_qty(self):
        assert _normalize_side({"qty": 0}) is None
        assert _normalize_side({}) is None

    def test_normalize_entry_price_variants(self):
        assert _normalize_entry_price({"entry_price": "100.5"}) == Decimal("100.5")
        assert _normalize_entry_price({"entryPrice": 100.5}) == Decimal("100.5")
        assert _normalize_entry_price({"avg_entry_price": 100.5}) == Decimal("100.5")
        assert _normalize_entry_price({"avgEntryPrice": "100.5"}) == Decimal("100.5")

    def test_normalize_entry_price_invalid(self):
        assert _normalize_entry_price({}) is None
        assert _normalize_entry_price({"entry_price": 0}) is None
        assert _normalize_entry_price({"entry_price": -10}) is None

    def test_normalize_qty_variants(self):
        assert _normalize_qty({"qty": "1.5"}) == Decimal("1.5")
        assert _normalize_qty({"qty": -1.5}) == Decimal("1.5")  # absolute
        assert _normalize_qty({"positionAmt": "2.0"}) == Decimal("2.0")
        assert _normalize_qty({"amount": 3.0}) == Decimal("3.0")

    def test_make_key(self):
        assert _make_key("btcusdt", "long") == "BTCUSDT:LONG"
        assert _make_key("ETHUSDT", "SHORT") == "ETHUSDT:SHORT"


class TestBasicMerge:
    """Test basic merge of positions + TP/SL registry."""

    def test_build_position_ledger_merges_correctly(self, temp_state_dir, sample_positions_state, sample_tp_sl_registry):
        """Test that ledger merges positions with TP/SL data."""
        # Write test files
        (temp_state_dir / "positions_state.json").write_text(json.dumps(sample_positions_state))
        (temp_state_dir / "position_tp_sl.json").write_text(json.dumps(sample_tp_sl_registry))

        ledger = build_position_ledger(temp_state_dir)

        assert len(ledger) == 2
        
        # BTC has TP/SL
        btc = ledger.get("BTCUSDT:LONG")
        assert btc is not None
        assert btc.symbol == "BTCUSDT"
        assert btc.side == "LONG"
        assert btc.entry_price == Decimal("50000.00")
        assert btc.qty == Decimal("0.5")
        assert btc.tp_sl.tp == Decimal("55000.0")
        assert btc.tp_sl.sl == Decimal("47000.0")

        # ETH has no TP/SL in registry
        eth = ledger.get("ETHUSDT:SHORT")
        assert eth is not None
        assert eth.symbol == "ETHUSDT"
        assert eth.side == "SHORT"
        assert eth.entry_price == Decimal("3000.00")
        assert eth.qty == Decimal("2.0")  # absolute
        assert eth.tp_sl.tp is None
        assert eth.tp_sl.sl is None

    def test_build_ledger_empty_state(self, temp_state_dir):
        """Test ledger with no files."""
        ledger = build_position_ledger(temp_state_dir)
        assert ledger == {}

    def test_build_ledger_skips_zero_qty(self, temp_state_dir):
        """Test that zero-qty positions are excluded."""
        positions = {
            "positions": [
                {"symbol": "BTCUSDT", "positionSide": "LONG", "qty": "0", "entryPrice": "50000"},
            ]
        }
        (temp_state_dir / "positions_state.json").write_text(json.dumps(positions))

        ledger = build_position_ledger(temp_state_dir)
        assert ledger == {}


class TestSyncLedger:
    """Test sync_ledger_with_positions() function."""

    def test_sync_seeds_missing_entries(self, temp_state_dir, sample_positions_state):
        """Test that sync seeds TP/SL for positions without registry entries."""
        (temp_state_dir / "positions_state.json").write_text(json.dumps(sample_positions_state))
        # No tp_sl registry file

        ledger = sync_ledger_with_positions(
            seed_missing=True,
            remove_stale=True,
            state_dir=temp_state_dir,
        )

        assert len(ledger) == 2

        # Both should now have TP/SL
        btc = ledger.get("BTCUSDT:LONG")
        assert btc.tp_sl.tp is not None
        assert btc.tp_sl.sl is not None

        eth = ledger.get("ETHUSDT:SHORT")
        assert eth.tp_sl.tp is not None
        assert eth.tp_sl.sl is not None

        # Verify registry was written
        registry = load_tp_sl_registry(temp_state_dir)
        assert "BTCUSDT:LONG" in registry
        assert "ETHUSDT:SHORT" in registry

    def test_sync_removes_stale_entries(self, temp_state_dir):
        """Test that sync removes registry entries for closed positions."""
        # Position state with only BTC
        positions = {
            "positions": [
                {"symbol": "BTCUSDT", "positionSide": "LONG", "qty": "0.5", "entryPrice": "50000"},
            ]
        }
        # Registry with BTC and SOL (SOL is stale)
        registry = {
            "entries": {
                "BTCUSDT:LONG": {"symbol": "BTCUSDT", "position_side": "LONG", "take_profit_price": 55000, "stop_loss_price": 47000},
                "SOLUSDT:LONG": {"symbol": "SOLUSDT", "position_side": "LONG", "take_profit_price": 200, "stop_loss_price": 100},
            }
        }

        (temp_state_dir / "positions_state.json").write_text(json.dumps(positions))
        (temp_state_dir / "position_tp_sl.json").write_text(json.dumps(registry))

        ledger = sync_ledger_with_positions(
            seed_missing=True,
            remove_stale=True,
            state_dir=temp_state_dir,
        )

        assert len(ledger) == 1
        assert "BTCUSDT:LONG" in ledger

        # Registry should no longer have SOL
        updated_registry = load_tp_sl_registry(temp_state_dir)
        assert "BTCUSDT:LONG" in updated_registry
        assert "SOLUSDT:LONG" not in updated_registry

    def test_sync_no_seed_when_disabled(self, temp_state_dir, sample_positions_state):
        """Test that sync doesn't seed when seed_missing=False."""
        (temp_state_dir / "positions_state.json").write_text(json.dumps(sample_positions_state))

        ledger = sync_ledger_with_positions(
            seed_missing=False,
            remove_stale=True,
            state_dir=temp_state_dir,
        )

        # Both entries should have None TP/SL
        btc = ledger.get("BTCUSDT:LONG")
        assert btc.tp_sl.tp is None
        assert btc.tp_sl.sl is None


class TestEdgeFormats:
    """Test handling of alternative field names."""

    def test_alternative_position_fields(self, temp_state_dir):
        """Test positions with alternative field names."""
        positions = {
            "positions": [
                {
                    "symbol": "BTCUSDT",
                    "position_side": "LONG",  # underscore instead of camelCase
                    "amount": "0.5",  # 'amount' instead of 'qty'
                    "avg_entry_price": "50000.00",  # underscore
                },
                {
                    "symbol": "ETHUSDT",
                    "side": "SHORT",  # 'side' instead of 'positionSide'
                    "positionAmt": "-2.0",
                    "avgEntryPrice": "3000.00",  # camelCase
                },
            ]
        }
        (temp_state_dir / "positions_state.json").write_text(json.dumps(positions))

        ledger = build_position_ledger(temp_state_dir)

        assert len(ledger) == 2
        
        btc = ledger.get("BTCUSDT:LONG")
        assert btc is not None
        assert btc.entry_price == Decimal("50000.00")
        assert btc.qty == Decimal("0.5")

        eth = ledger.get("ETHUSDT:SHORT")
        assert eth is not None
        assert eth.entry_price == Decimal("3000.00")
        assert eth.qty == Decimal("2.0")

    def test_tp_sl_embedded_in_position(self, temp_state_dir):
        """Test TP/SL embedded in position dict (not in registry)."""
        positions = {
            "positions": [
                {
                    "symbol": "BTCUSDT",
                    "positionSide": "LONG",
                    "qty": "0.5",
                    "entryPrice": "50000.00",
                    "tp_sl": {
                        "take_profit_price": 55000.0,
                        "stop_loss_price": 47000.0,
                    }
                },
            ]
        }
        (temp_state_dir / "positions_state.json").write_text(json.dumps(positions))

        ledger = build_position_ledger(temp_state_dir)

        btc = ledger.get("BTCUSDT:LONG")
        assert btc.tp_sl.tp == Decimal("55000.0")
        assert btc.tp_sl.sl == Decimal("47000.0")


class TestTPSLOperations:
    """Test TP/SL upsert and delete operations."""

    def test_upsert_tp_sl(self, temp_state_dir):
        """Test upserting TP/SL for a position."""
        upsert_tp_sl(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=Decimal("50000"),
            tp=Decimal("55000"),
            sl=Decimal("47000"),
            state_dir=temp_state_dir,
        )

        registry = load_tp_sl_registry(temp_state_dir)
        assert "BTCUSDT:LONG" in registry
        entry = registry["BTCUSDT:LONG"]
        assert entry["take_profit_price"] == 55000.0
        assert entry["stop_loss_price"] == 47000.0

    def test_delete_tp_sl_single_side(self, temp_state_dir):
        """Test deleting TP/SL for a specific side."""
        # Create entries
        upsert_tp_sl("BTCUSDT", "LONG", Decimal("50000"), Decimal("55000"), Decimal("47000"), state_dir=temp_state_dir)
        upsert_tp_sl("BTCUSDT", "SHORT", Decimal("50000"), Decimal("45000"), Decimal("53000"), state_dir=temp_state_dir)

        delete_tp_sl("BTCUSDT", side="LONG", state_dir=temp_state_dir)

        registry = load_tp_sl_registry(temp_state_dir)
        assert "BTCUSDT:LONG" not in registry
        assert "BTCUSDT:SHORT" in registry

    def test_delete_tp_sl_both_sides(self, temp_state_dir):
        """Test deleting TP/SL for both sides."""
        upsert_tp_sl("BTCUSDT", "LONG", Decimal("50000"), Decimal("55000"), Decimal("47000"), state_dir=temp_state_dir)
        upsert_tp_sl("BTCUSDT", "SHORT", Decimal("50000"), Decimal("45000"), Decimal("53000"), state_dir=temp_state_dir)

        delete_tp_sl("BTCUSDT", side=None, state_dir=temp_state_dir)

        registry = load_tp_sl_registry(temp_state_dir)
        assert "BTCUSDT:LONG" not in registry
        assert "BTCUSDT:SHORT" not in registry


class TestLedgerToDict:
    """Test ledger serialization."""

    def test_ledger_to_dict(self):
        """Test converting ledger to JSON-serializable dict."""
        ledger = {
            "BTCUSDT:LONG": PositionLedgerEntry(
                symbol="BTCUSDT",
                side="LONG",
                entry_price=Decimal("50000"),
                qty=Decimal("0.5"),
                tp_sl=PositionTP_SL(tp=Decimal("55000"), sl=Decimal("47000")),
                created_ts=1700000000.0,
                updated_ts=1700000001.0,
            ),
        }

        result = ledger_to_dict(ledger)

        assert "BTCUSDT:LONG" in result
        entry = result["BTCUSDT:LONG"]
        assert entry["symbol"] == "BTCUSDT"
        assert entry["side"] == "LONG"
        assert entry["qty"] == 0.5
        assert entry["entry_price"] == 50000.0
        assert entry["tp"] == 55000.0
        assert entry["sl"] == 47000.0
        assert entry["created_ts"] == 1700000000.0

    def test_ledger_to_dict_with_none_tp_sl(self):
        """Test serialization with None TP/SL."""
        ledger = {
            "ETHUSDT:SHORT": PositionLedgerEntry(
                symbol="ETHUSDT",
                side="SHORT",
                entry_price=Decimal("3000"),
                qty=Decimal("2.0"),
                tp_sl=PositionTP_SL(),  # Both None
            ),
        }

        result = ledger_to_dict(ledger)

        entry = result["ETHUSDT:SHORT"]
        assert entry["tp"] is None
        assert entry["sl"] is None


class TestGetLedgerSummary:
    """Test ledger summary for debugging."""

    def test_get_ledger_summary(self, temp_state_dir, sample_positions_state, sample_tp_sl_registry):
        """Test summary generation."""
        (temp_state_dir / "positions_state.json").write_text(json.dumps(sample_positions_state))
        (temp_state_dir / "position_tp_sl.json").write_text(json.dumps(sample_tp_sl_registry))

        summary = get_ledger_summary(temp_state_dir)

        assert summary["num_positions"] == 2
        assert summary["num_with_tp_sl"] == 1  # Only BTC has TP/SL
        assert summary["num_registry_entries"] == 1
        assert summary["consistency"] == "partial"
        assert "BTCUSDT:LONG" in summary["symbols"]
        assert "ETHUSDT:SHORT" in summary["symbols"]
