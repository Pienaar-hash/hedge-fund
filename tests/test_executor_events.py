from typing import Any, Dict, List, Tuple

import pytest

import execution.executor_live as executor_live


def test_executor_ack_and_fill_event_flow(monkeypatch) -> None:
    recorded: List[Tuple[str, Dict[str, Any]]] = []

    def fake_write_event(event_type: str, payload: Dict[str, Any], *, logger: Any | None = None) -> None:
        recorded.append((event_type, dict(payload)))

    monkeypatch.setattr(executor_live, "write_event", fake_write_event)
    monkeypatch.setattr(executor_live, "_POSITION_TRACKER", executor_live.PositionTracker())
    monkeypatch.setattr(executor_live, "_fetch_order_status", lambda *args, **kwargs: {"status": "FILLED"})

    trades = [
        {
            "qty": "0.5",
            "price": "100.0",
            "commission": "0.01",
            "commissionAsset": "USDT",
            "time": 1700000000000,
            "id": 1,
        },
        {
            "qty": "0.5",
            "price": "101.0",
            "commission": "0.01",
            "commissionAsset": "USDT",
            "time": 1700000005000,
            "id": 2,
        },
    ]
    monkeypatch.setattr(executor_live, "_fetch_order_trades", lambda *args, **kwargs: trades)

    ack_resp = {"status": "NEW", "orderId": 111, "clientOrderId": "cli-111"}
    ack_info = executor_live._emit_order_ack(
        symbol="BTCUSDT",
        side="BUY",
        order_type="MARKET",
        request_qty=1.0,
        position_side="LONG",
        reduce_only=False,
        resp=ack_resp,
        latency_ms=25.0,
        attempt_id="att-1",
        intent_id="intent-1",
    )
    assert ack_info is not None

    fill_summary = executor_live._confirm_order_fill(ack_info)
    assert fill_summary is not None
    assert pytest.approx(fill_summary.executed_qty) == pytest.approx(1.0)
    assert pytest.approx(fill_summary.avg_price) == pytest.approx(100.5)

    event_types = [et for et, _ in recorded]
    assert "order_ack" in event_types
    assert "order_fill" in event_types

    fill_payload = next(payload for et, payload in recorded if et == "order_fill")
    assert pytest.approx(fill_payload["executedQty"]) == pytest.approx(1.0)
    assert pytest.approx(fill_payload["avgPrice"]) == pytest.approx(100.5)
