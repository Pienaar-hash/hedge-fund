import json
from datetime import datetime, timezone

import pandas as pd

import dashboard.router_health as rh


def test_load_router_health_from_order_events(tmp_path, monkeypatch) -> None:
    signal_path = tmp_path / "signal_metrics.jsonl"
    order_metrics_path = tmp_path / "order_metrics.jsonl"
    events_path = tmp_path / "orders_executed.jsonl"

    now = datetime.now(timezone.utc).isoformat()
    attempt_id = "att-test"

    signal_record = {
        "attempt_id": attempt_id,
        "symbol": "BTCUSDT",
        "signal": "BUY",
        "doctor": {"confidence": 0.8, "ok": True},
        "ts": now,
    }
    with signal_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(signal_record) + "\n")

    order_metrics_record = {
        "attempt_id": attempt_id,
        "event": "execution",
        "prices": {"mark": 100.0, "submitted": 100.0, "avg_fill": 101.0},
        "timing_ms": {"decision": 50},
        "slippage_bps": 10.0,
    }
    with order_metrics_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(order_metrics_record) + "\n")

    ack_event = {
        "event_type": "order_ack",
        "symbol": "BTCUSDT",
        "side": "BUY",
        "orderId": 1,
        "clientOrderId": "abc",
        "ts": now,
        "attempt_id": attempt_id,
    }
    fill_event = {
        "event_type": "order_fill",
        "symbol": "BTCUSDT",
        "side": "BUY",
        "orderId": 1,
        "clientOrderId": "abc",
        "executedQty": 1.0,
        "avgPrice": 101.0,
        "fee_total": 0.01,
        "feeAsset": "USDT",
        "latency_ms": 250.0,
        "ts_fill_first": now,
        "ts_fill_last": now,
        "attempt_id": attempt_id,
    }
    close_event = {
        "event_type": "order_close",
        "symbol": "BTCUSDT",
        "orderId": 1,
        "clientOrderId": "abc",
        "realizedPnlUsd": 5.0,
        "fees_total": 0.01,
        "position_size_before": 1.0,
        "position_size_after": 0.0,
        "ts_close": now,
        "attempt_id": attempt_id,
    }
    with events_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(ack_event) + "\n")
        handle.write(json.dumps(fill_event) + "\n")
        handle.write(json.dumps(close_event) + "\n")

    monkeypatch.setattr(rh, "ORDER_EVENTS_PATH", events_path)

    data = rh.load_router_health(window=10, signal_path=signal_path, order_path=order_metrics_path)

    assert not data.trades.empty
    assert not data.per_symbol.empty
    assert "median_latency_ms" in data.per_symbol.columns
    assert "median_slippage_bps" in data.per_symbol.columns
    assert data.summary["fill_rate_pct"] >= 100.0
    assert pd.notna(data.summary["fees_total"])
