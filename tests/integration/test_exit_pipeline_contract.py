from __future__ import annotations

from dataclasses import dataclass

import pytest

from execution import exit_scanner
from execution import position_ledger
from execution import position_tp_sl_registry
from execution.diagnostics_metrics import (
    get_exit_status,
    record_router_event,
    reset_diagnostics,
)

pytestmark = pytest.mark.integration


@dataclass
class DummyTP:
    tp: float | None = None
    sl: float | None = None


@dataclass
class DummyEntry:
    symbol: str
    side: str
    qty: float
    entry_price: float
    tp_sl: DummyTP


def test_exit_pipeline_status_updates_and_triggers(monkeypatch):
    reset_diagnostics()

    ledger_entries = {
        "BTCUSDT:LONG": DummyEntry("BTCUSDT", "LONG", 1.0, 100.0, DummyTP(tp=105.0, sl=95.0)),
        "ETHUSDT:LONG": DummyEntry("ETHUSDT", "LONG", 2.0, 200.0, DummyTP()),
    }
    registry_entries = {
        "BTCUSDT:LONG": {"take_profit_price": 105.0, "stop_loss_price": 95.0},
    }
    positions_state = {
        "positions": [
            {"symbol": "BTCUSDT", "qty": 1.0, "positionSide": "LONG", "entry_price": 100.0},
            {"symbol": "ETHUSDT", "qty": 2.0, "positionSide": "LONG", "entry_price": 200.0},
        ]
    }

    monkeypatch.setattr(position_ledger, "build_position_ledger", lambda state_dir=None: ledger_entries)
    monkeypatch.setattr(position_ledger, "load_positions_state", lambda state_dir=None: positions_state)
    monkeypatch.setattr(position_ledger, "load_tp_sl_registry", lambda state_dir=None: registry_entries)
    monkeypatch.setattr(position_tp_sl_registry, "get_all_tp_sl_positions", lambda: registry_entries)

    price_map = {"BTCUSDT": 106.0, "ETHUSDT": 180.0}

    results = exit_scanner.scan_tp_sl_exits([], price_map)

    status = get_exit_status()
    assert status.last_exit_scan_ts is not None
    assert status.last_exit_trigger_ts is not None
    assert status.open_positions_count == 2
    assert status.tp_sl_registered_count == 1
    assert status.tp_sl_missing_count == 1
    assert status.underwater_without_tp_sl_count == 1
    assert status.tp_sl_coverage_pct == 0.5
    assert status.ledger_registry_mismatch is True
    assert status.mismatch_breakdown.get("missing_tp_sl_entries") == 1
    assert len(results) == 1


def test_record_router_event_sets_timestamp():
    reset_diagnostics()
    record_router_event()
    status = get_exit_status()
    assert status.last_router_event_ts is not None
