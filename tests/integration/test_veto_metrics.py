from __future__ import annotations

from execution.diagnostics_metrics import (
    build_runtime_diagnostics_snapshot,
    record_order_placed,
    record_signal_emitted,
    record_veto,
    reset_diagnostics,
)


def test_veto_counters_accumulate() -> None:
    reset_diagnostics()
    record_signal_emitted()
    record_signal_emitted()
    record_order_placed()
    record_veto("max_concurrent")
    record_veto("symbol_cap")

    snap = build_runtime_diagnostics_snapshot()
    vc = snap.veto_counters

    assert vc.total_signals == 2
    assert vc.total_orders == 1
    assert vc.total_vetoes == 2
    assert vc.by_reason.get("max_concurrent") == 1
    assert vc.by_reason.get("symbol_cap") == 1
    assert vc.last_veto_ts is not None
