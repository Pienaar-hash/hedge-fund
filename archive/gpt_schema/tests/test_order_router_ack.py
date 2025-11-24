from typing import Any, Dict

from execution import order_router


def test_route_order_ack_preserves_status(monkeypatch) -> None:
    def fake_send_order(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": "NEW",
            "orderId": 123456,
            "clientOrderId": "abc123",
            "executedQty": "0",
            "avgPrice": "0.0",
        }

    monkeypatch.setattr(order_router.ex, "send_order", fake_send_order)
    monkeypatch.setattr(order_router.ex, "get_symbol_filters", lambda symbol: {})

    intent = {"symbol": "BTCUSDT", "quantity": 1.0, "side": "BUY"}
    result = order_router.route_order(intent, {}, dry_run=False)

    assert result["status"] == "NEW"
    assert result["price"] is None
    assert result["qty"] is None
    assert result["accepted"] is True
