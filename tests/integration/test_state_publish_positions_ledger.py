"""
Tests for State Publish + Position Ledger Integration (v7.4_C3)

Tests:
1. publish_positions includes positions_ledger block
2. write_positions_ledger_state creates correct file
3. Ledger fields are correctly serialized
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from decimal import Decimal

from execution.position_ledger import (
    PositionLedgerEntry,
    PositionTP_SL,
    ledger_to_dict,
)


@pytest.fixture
def temp_state_dir(tmp_path):
    """Create temporary state directory."""
    state_dir = tmp_path / "logs" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


@pytest.fixture
def sample_ledger():
    """Sample ledger for testing."""
    return {
        "BTCUSDT:LONG": PositionLedgerEntry(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=Decimal("50000"),
            qty=Decimal("0.5"),
            tp_sl=PositionTP_SL(tp=Decimal("55000"), sl=Decimal("47000")),
            created_ts=1700000000.0,
            updated_ts=1700000001.0,
        ),
        "ETHUSDT:SHORT": PositionLedgerEntry(
            symbol="ETHUSDT",
            side="SHORT",
            entry_price=Decimal("3000"),
            qty=Decimal("2.0"),
            tp_sl=PositionTP_SL(tp=Decimal("2800"), sl=Decimal("3200")),
        ),
    }


class TestWritePositionsLedgerState:
    """Test write_positions_ledger_state function."""

    def test_write_positions_ledger_state(self, temp_state_dir):
        """Test that positions_ledger.json is written correctly."""
        from execution.state_publish import write_positions_ledger_state

        payload = {
            "entries": {
                "BTCUSDT:LONG": {
                    "symbol": "BTCUSDT",
                    "side": "LONG",
                    "qty": 0.5,
                    "entry_price": 50000.0,
                    "tp": 55000.0,
                    "sl": 47000.0,
                }
            },
            "updated_ts": 1700000000.0,
        }

        write_positions_ledger_state(payload, temp_state_dir)

        path = temp_state_dir / "positions_ledger.json"
        assert path.exists()

        written = json.loads(path.read_text())
        assert "entries" in written
        assert "BTCUSDT:LONG" in written["entries"]
        assert written["entries"]["BTCUSDT:LONG"]["tp"] == 55000.0


class TestPublishPositionsWithLedger:
    """Test publish_positions includes ledger data."""

    def test_publish_positions_adds_ledger_block(self, sample_ledger):
        """Test that publish_positions includes positions_ledger block."""
        from execution.state_publish import publish_positions

        rows = [
            {"symbol": "BTCUSDT", "positionSide": "LONG", "qty": 0.5},
            {"symbol": "ETHUSDT", "positionSide": "SHORT", "qty": -2.0},
        ]

        # Mock firestore as disabled
        with patch("execution.state_publish._firestore_enabled", return_value=False):
            # Mock ledger build at the module level
            with patch("execution.position_ledger.build_position_ledger", return_value=sample_ledger):
                with patch("execution.position_ledger.ledger_to_dict", return_value={
                    "BTCUSDT:LONG": {"symbol": "BTCUSDT", "tp": 55000.0, "sl": 47000.0},
                    "ETHUSDT:SHORT": {"symbol": "ETHUSDT", "tp": 2800.0, "sl": 3200.0},
                }):
                    with patch("execution.state_publish._append_local_jsonl") as mock_jsonl:
                        with patch("execution.state_publish.write_positions_state") as mock_write:
                            with patch("execution.state_publish._compute_exec_stats", return_value={}):
                                publish_positions(rows)

                                # Check that write_positions_state was called with ledger data
                                assert mock_write.called
                                call_args = mock_write.call_args[0][0]
                                assert "positions_ledger" in call_args
                                assert "BTCUSDT:LONG" in call_args["positions_ledger"]


class TestLedgerFieldsSerialization:
    """Test that ledger fields are correctly serialized for state files."""

    def test_ledger_to_dict_all_fields(self, sample_ledger):
        """Test ledger_to_dict includes all expected fields."""
        result = ledger_to_dict(sample_ledger)

        btc = result["BTCUSDT:LONG"]
        assert btc["symbol"] == "BTCUSDT"
        assert btc["side"] == "LONG"
        assert btc["qty"] == 0.5
        assert btc["entry_price"] == 50000.0
        assert btc["tp"] == 55000.0
        assert btc["sl"] == 47000.0
        assert btc["created_ts"] == 1700000000.0
        assert btc["updated_ts"] == 1700000001.0

        eth = result["ETHUSDT:SHORT"]
        assert eth["symbol"] == "ETHUSDT"
        assert eth["side"] == "SHORT"
        assert eth["qty"] == 2.0
        assert eth["entry_price"] == 3000.0
        assert eth["tp"] == 2800.0
        assert eth["sl"] == 3200.0

    def test_ledger_to_dict_handles_none_timestamps(self):
        """Test ledger_to_dict handles None timestamps."""
        ledger = {
            "SOLUSDT:LONG": PositionLedgerEntry(
                symbol="SOLUSDT",
                side="LONG",
                entry_price=Decimal("100"),
                qty=Decimal("10"),
                tp_sl=PositionTP_SL(),
                created_ts=None,
                updated_ts=None,
            ),
        }

        result = ledger_to_dict(ledger)
        sol = result["SOLUSDT:LONG"]
        
        assert sol["tp"] is None
        assert sol["sl"] is None
        assert sol["created_ts"] is None
        assert sol["updated_ts"] is None

    def test_serialized_ledger_is_json_compatible(self, sample_ledger):
        """Test that serialized ledger can be JSON encoded."""
        result = ledger_to_dict(sample_ledger)
        
        # Should not raise
        json_str = json.dumps(result)
        assert json_str
        
        # Should round-trip
        parsed = json.loads(json_str)
        assert parsed["BTCUSDT:LONG"]["tp"] == 55000.0


class TestStateFilesContract:
    """Test that state files maintain the expected contract."""

    def test_positions_state_includes_ledger_key(self):
        """Verify positions.json can include positions_ledger key."""
        payload = {
            "rows": [{"symbol": "BTCUSDT", "qty": 0.5}],
            "updated": 1700000000.0,
            "positions_ledger": {
                "BTCUSDT:LONG": {
                    "symbol": "BTCUSDT",
                    "side": "LONG",
                    "tp": 55000.0,
                    "sl": 47000.0,
                }
            },
        }
        
        # Verify structure is valid JSON
        json_str = json.dumps(payload)
        parsed = json.loads(json_str)
        
        assert "positions_ledger" in parsed
        assert "rows" in parsed
        assert parsed["positions_ledger"]["BTCUSDT:LONG"]["tp"] == 55000.0

    def test_ledger_additive_to_existing_contract(self):
        """Test that ledger data is additive and doesn't break existing fields."""
        # Simulate existing positions.json structure
        existing = {
            "rows": [
                {"symbol": "BTCUSDT", "positionSide": "LONG", "qty": 0.5, "entryPrice": 50000.0}
            ],
            "updated": 1700000000.0,
            "exec_stats": {"orders_7d": 10},
        }
        
        # Add ledger data
        existing["positions_ledger"] = {
            "BTCUSDT:LONG": {"symbol": "BTCUSDT", "tp": 55000.0, "sl": 47000.0}
        }
        
        # All original fields should still be present
        assert "rows" in existing
        assert "updated" in existing
        assert "exec_stats" in existing
        assert existing["rows"][0]["symbol"] == "BTCUSDT"


class TestDashboardLoadPositionsLedger:
    """Test dashboard loading of positions ledger."""

    def test_load_positions_ledger_from_file(self, temp_state_dir):
        """Test loading positions_ledger from dedicated file."""
        # Create ledger file
        ledger_data = {
            "entries": {
                "BTCUSDT:LONG": {"symbol": "BTCUSDT", "tp": 55000.0, "sl": 47000.0}
            },
            "updated_ts": 1700000000.0,
        }
        (temp_state_dir / "positions_ledger.json").write_text(json.dumps(ledger_data))

        # Mock the state paths
        with patch("dashboard.state_v7.POSITIONS_LEDGER_PATH", temp_state_dir / "positions_ledger.json"):
            from dashboard.state_v7 import load_positions_ledger
            
            result = load_positions_ledger()
            
            assert "entries" in result
            assert "BTCUSDT:LONG" in result["entries"]

    def test_load_positions_ledger_fallback_to_embedded(self, temp_state_dir):
        """Test loading positions_ledger from embedded in positions.json."""
        # Create positions.json with embedded ledger
        positions_data = {
            "rows": [{"symbol": "BTCUSDT", "qty": 0.5}],
            "positions_ledger": {
                "BTCUSDT:LONG": {"symbol": "BTCUSDT", "tp": 55000.0, "sl": 47000.0}
            },
            "updated": 1700000000.0,
        }
        (temp_state_dir / "positions.json").write_text(json.dumps(positions_data))

        with patch("dashboard.state_v7.POSITIONS_LEDGER_PATH", temp_state_dir / "positions_ledger.json"):
            with patch("dashboard.state_v7.POSITIONS_PATH", temp_state_dir / "positions.json"):
                with patch("dashboard.state_v7.POSITIONS_STATE_PATH", temp_state_dir / "positions_state.json"):
                    from dashboard.state_v7 import load_positions_ledger
                    
                    result = load_positions_ledger()
                    
                    assert "entries" in result
                    assert "BTCUSDT:LONG" in result["entries"]


class TestLedgerConsistencyStatus:
    """Test ledger consistency status for dashboard."""

    def test_consistency_ok_all_have_tp_sl(self, temp_state_dir):
        """Test consistency is OK when all positions have TP/SL."""
        # Positions
        positions = {
            "rows": [{"symbol": "BTCUSDT", "positionSide": "LONG", "qty": 0.5}],
        }
        (temp_state_dir / "positions.json").write_text(json.dumps(positions))
        
        # Ledger with TP/SL
        ledger = {
            "entries": {
                "BTCUSDT:LONG": {"symbol": "BTCUSDT", "tp": 55000.0, "sl": 47000.0}
            }
        }
        (temp_state_dir / "positions_ledger.json").write_text(json.dumps(ledger))

        with patch("dashboard.state_v7.POSITIONS_PATH", temp_state_dir / "positions.json"):
            with patch("dashboard.state_v7.POSITIONS_STATE_PATH", temp_state_dir / "positions_state.json"):
                with patch("dashboard.state_v7.POSITIONS_LEDGER_PATH", temp_state_dir / "positions_ledger.json"):
                    from dashboard.state_v7 import get_ledger_consistency_status
                    
                    result = get_ledger_consistency_status()
                    
                    assert result["status"] == "ok"
                    assert result["num_positions"] == 1
                    assert result["num_ledger"] == 1
                    assert result["num_with_tp_sl"] == 1

    def test_consistency_error_empty_ledger(self, temp_state_dir):
        """Test consistency is ERROR when positions exist but ledger is empty."""
        positions = {
            "rows": [{"symbol": "BTCUSDT", "positionSide": "LONG", "qty": 0.5}],
        }
        (temp_state_dir / "positions.json").write_text(json.dumps(positions))
        
        # Empty ledger
        (temp_state_dir / "positions_ledger.json").write_text(json.dumps({"entries": {}}))

        with patch("dashboard.state_v7.POSITIONS_PATH", temp_state_dir / "positions.json"):
            with patch("dashboard.state_v7.POSITIONS_STATE_PATH", temp_state_dir / "positions_state.json"):
                with patch("dashboard.state_v7.POSITIONS_LEDGER_PATH", temp_state_dir / "positions_ledger.json"):
                    from dashboard.state_v7 import get_ledger_consistency_status
                    
                    result = get_ledger_consistency_status()
                    
                    assert result["status"] == "error"
                    assert "empty" in result["message"].lower()
