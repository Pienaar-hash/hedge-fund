from __future__ import annotations

from decimal import Decimal

from execution.position_ledger import (
    LedgerReconciliationReport,
    PositionLedgerEntry,
    PositionTP_SL,
    reconcile_ledger_and_registry,
)


def _positions_state() -> dict:
    return {
        "positions": [
            {"symbol": "BTCUSDT", "qty": 1.0, "positionSide": "LONG", "entry_price": 100.0},
            {"symbol": "ETHUSDT", "qty": -2.0, "positionSide": "SHORT", "entry_price": 200.0},
        ]
    }


def test_reconcile_reports_missing_and_ghost_entries():
    ledger = {
        "BTCUSDT:LONG": PositionLedgerEntry(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=Decimal("100"),
            qty=Decimal("1"),
            tp_sl=PositionTP_SL(tp=Decimal("110"), sl=Decimal("95")),
        ),
        # Ghost ledger entry (no live position)
        "XRPUSDT:LONG": PositionLedgerEntry(
            symbol="XRPUSDT",
            side="LONG",
            entry_price=Decimal("0.5"),
            qty=Decimal("100"),
            tp_sl=PositionTP_SL(),
        ),
    }
    registry = {
        "BTCUSDT:LONG": {"take_profit_price": 110.0, "stop_loss_price": 95.0},
        # Stale registry entry (position closed)
        "ADAUSDT:LONG": {"take_profit_price": 0.4, "stop_loss_price": 0.3},
    }

    report: LedgerReconciliationReport = reconcile_ledger_and_registry(_positions_state(), ledger, registry)

    assert "ETHUSDT:SHORT" in report.missing_ledger_positions
    assert "XRPUSDT:LONG" in report.ghost_ledger_entries
    assert "ADAUSDT:LONG" in report.stale_tp_sl_entries
    # ETH ledger entry missing -> missing TP/SL captured via missing_ledger_positions
    assert "XRPUSDT:LONG" in report.missing_tp_sl_entries
    assert report.has_mismatch is True
    breakdown = report.breakdown_counts()
    assert breakdown["missing_ledger_positions"] == 1
    assert breakdown["ghost_ledger_entries"] == 1
