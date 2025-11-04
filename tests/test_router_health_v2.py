from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd

import dashboard.router_health as rh


def test_router_health_confidence_and_sharpe(tmp_path, monkeypatch) -> None:
    signal_path = tmp_path / "signal_metrics.jsonl"
    order_metrics_path = tmp_path / "order_metrics.jsonl"
    events_path = tmp_path / "orders_executed.jsonl"

    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    attempts = []
    confidences = [0.9, 0.7, 0.5, 0.8, 0.6]
    pnls = [5.0, -2.0, 3.0, 1.5, -1.0]

    with signal_path.open("w", encoding="utf-8") as sig_handle, order_metrics_path.open("w", encoding="utf-8") as order_handle:
        for idx, (conf, pnl) in enumerate(zip(confidences, pnls), start=1):
            attempt_id = f"att-{idx}"
            attempts.append(attempt_id)
            ts = (now + timedelta(minutes=idx)).isoformat()
            sig_record = {
                "attempt_id": attempt_id,
                "symbol": "ETHUSDT",
                "signal": "BUY",
                "doctor": {"confidence": conf},
                "ts": ts,
            }
            sig_handle.write(json.dumps(sig_record) + "\n")

            order_metrics_record = {
                "attempt_id": attempt_id,
                "event": "position_close",
                "pnl_usd": pnl,
                "pnl_pct": pnl / 100.0,
                "prices": {"mark": 100 + idx},
            }
            order_handle.write(json.dumps(order_metrics_record) + "\n")

    with events_path.open("w", encoding="utf-8") as events_handle:
        for idx, (attempt_id, pnl) in enumerate(zip(attempts, pnls), start=1):
            ts = (now + timedelta(minutes=idx, seconds=5)).isoformat()
            events_handle.write(
                json.dumps(
                    {
                        "event_type": "order_ack",
                        "symbol": "ETHUSDT",
                        "orderId": idx,
                        "clientOrderId": f"{attempt_id}-ack",
                        "attempt_id": attempt_id,
                    }
                )
                + "\n"
            )
            events_handle.write(
                json.dumps(
                    {
                        "event_type": "order_fill",
                        "symbol": "ETHUSDT",
                        "orderId": idx,
                        "clientOrderId": f"{attempt_id}-fill",
                        "attempt_id": attempt_id,
                        "executedQty": 1.0,
                        "avgPrice": 100 + idx,
                        "latency_ms": 200 + idx,
                        "ts_fill_first": ts,
                        "ts_fill_last": ts,
                    }
                )
                + "\n"
            )
            events_handle.write(
                json.dumps(
                    {
                        "event_type": "order_close",
                        "symbol": "ETHUSDT",
                        "orderId": idx,
                        "clientOrderId": f"{attempt_id}-close",
                        "attempt_id": attempt_id,
                        "realizedPnlUsd": pnl,
                        "fees_total": 0.01,
                        "ts_close": ts,
                    }
                )
                + "\n"
            )

    monkeypatch.setattr(rh, "ORDER_EVENTS_PATH", events_path)

    data = rh.load_router_health(window=20, signal_path=signal_path, order_path=order_metrics_path)

    assert "confidence_weighted_cum_pnl" in data.pnl_curve.columns
    assert "rolling_sharpe" in data.pnl_curve.columns
    assert abs(data.summary["avg_confidence"] - sum(confidences) / len(confidences)) < 1e-6

    expected_weighted = sum(p * c for p, c in zip(pnls, confidences))
    assert abs(data.summary["confidence_weighted_cum_pnl"] - expected_weighted) < 1e-6

    assert "normalized_sharpe" in data.summary
    assert "confidence_weighted_pnl" in data.per_symbol.columns
    assert "normalized_sharpe" in data.per_symbol.columns

    assert data.overlays.get("confidence") is not None
    assert not data.overlays["confidence"].empty

    # Ensure per-symbol frame keeps numeric Sharpe content
    sharpe_series = pd.to_numeric(data.per_symbol["normalized_sharpe"], errors="coerce")
    assert sharpe_series.notna().any()
