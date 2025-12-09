from __future__ import annotations

import json
from decimal import Decimal

import pytest

from execution.position_ledger import (
    PositionLedgerEntry,
    PositionTP_SL,
    build_positions_ledger_state,
)
from execution.state_publish import write_positions_ledger_state

pytestmark = pytest.mark.integration


def test_positions_ledger_contract_snapshot(tmp_path):
    state_dir = tmp_path / "logs" / "state"
    ledger = {
        "BTCUSDT:LONG": PositionLedgerEntry(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=Decimal("50000"),
            qty=Decimal("0.25"),
            tp_sl=PositionTP_SL(tp=Decimal("52000"), sl=Decimal("48000")),
            created_ts=1700000000.0,
            updated_ts=1700000001.0,
        ),
        "ETHUSDT:SHORT": PositionLedgerEntry(
            symbol="ETHUSDT",
            side="SHORT",
            entry_price=Decimal("3000"),
            qty=Decimal("1.5"),
            tp_sl=PositionTP_SL(tp=Decimal("2800"), sl=Decimal("3200")),
        ),
    }

    snapshot = build_positions_ledger_state(ledger, updated_at="2024-01-01T00:00:00Z")
    write_positions_ledger_state(snapshot, state_dir=state_dir)

    path = state_dir / "positions_ledger.json"
    assert path.exists()

    payload = json.loads(path.read_text())
    assert payload.get("updated_at") == "2024-01-01T00:00:00Z"
    assert "entries" in payload
    assert isinstance(payload["entries"], list)
    assert payload.get("metadata", {}).get("entry_count") == len(ledger)
    assert "tp_sl_levels" in payload

    entry = payload["entries"][0]
    assert entry["symbol"]
    assert entry["side"]
    assert entry["entry_price"] > 0
    assert entry["qty"] != 0
