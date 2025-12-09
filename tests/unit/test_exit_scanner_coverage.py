from __future__ import annotations

from decimal import Decimal

import pytest

from execution import exit_scanner
from execution.diagnostics_metrics import get_exit_status, reset_diagnostics
from execution.position_ledger import PositionLedgerEntry, PositionTP_SL


class DummyLedgerEntry(PositionLedgerEntry):
    pass


def test_exit_scanner_updates_coverage_with_mismatches(monkeypatch):
    reset_diagnostics()

    # Build synthetic ledger: one with TP/SL, one missing
    ledger = {
        "BTCUSDT:LONG": DummyLedgerEntry(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=Decimal("100"),
            qty=Decimal("1"),
            tp_sl=PositionTP_SL(tp=Decimal("110"), sl=Decimal("95")),
        ),
        "ETHUSDT:LONG": DummyLedgerEntry(
            symbol="ETHUSDT",
            side="LONG",
            entry_price=Decimal("200"),
            qty=Decimal("2"),
            tp_sl=PositionTP_SL(),  # missing TP/SL
        ),
    }

    positions_state = {
        "positions": [
            {"symbol": "BTCUSDT", "qty": 1.0, "positionSide": "LONG", "entry_price": 100.0},
            {"symbol": "ETHUSDT", "qty": 2.0, "positionSide": "LONG", "entry_price": 200.0},
        ]
    }

    registry = {"BTCUSDT:LONG": {"take_profit_price": 110.0, "stop_loss_price": 95.0}}

    monkeypatch.setattr("execution.position_ledger.load_positions_state", lambda state_dir=None: positions_state)
    monkeypatch.setattr("execution.position_ledger.build_position_ledger", lambda state_dir=None: ledger)
    monkeypatch.setattr("execution.position_ledger.load_tp_sl_registry", lambda state_dir=None: registry)

    price_map = {"BTCUSDT": 111.0, "ETHUSDT": 150.0}

    results = exit_scanner.scan_tp_sl_exits([], price_map)

    status = get_exit_status()
    assert status.open_positions_count == 2
    assert status.tp_sl_registered_count == 1
    assert status.tp_sl_missing_count == 1
    assert status.underwater_without_tp_sl_count == 1  # ETH underwater and missing TP/SL
    assert status.ledger_registry_mismatch is True
    assert status.mismatch_breakdown.get("missing_tp_sl_entries") == 1
    assert len(results) == 1  # BTC TP hit
