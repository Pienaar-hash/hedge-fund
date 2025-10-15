from __future__ import annotations

from typing import Any, Dict

import execution.exchange_utils as ex
from execution.order_router import route_order


def test_route_order_reduce_only(monkeypatch) -> None:
    recorded: Dict[str, Any] = {}

    def fake_set_dry_run(flag: bool) -> None:
        recorded["dry_run"] = bool(flag)

    def fake_send_order(**payload: Any) -> Dict[str, Any]:
        recorded["payload"] = payload
        return {
            "orderId": 321,
            "avgPrice": "123.45",
            "executedQty": payload.get("quantity", "0"),
            "status": "NEW",
        }

    monkeypatch.setattr(ex, "set_dry_run", fake_set_dry_run)
    monkeypatch.setattr(ex, "send_order", fake_send_order)

    intent = {
        "symbol": "BTCUSDT",
        "signal": "SELL",
        "quantity": "0.50",
        "reduceOnly": "true",
        "type": "market",
    }
    result = route_order(intent, {}, False)

    assert result["accepted"] is True
    assert result["order_id"] == 321
    assert recorded["dry_run"] is False
    payload = recorded["payload"]
    assert payload["reduceOnly"] == "true"
    assert payload["side"] == "SELL"
    assert payload["quantity"] == "0.50"


def test_route_order_dry_run(monkeypatch) -> None:
    recorded: Dict[str, Any] = {}

    def fake_set_dry_run(flag: bool) -> None:
        recorded.setdefault("dry_flags", []).append(flag)

    def fake_send_order(**payload: Any) -> Dict[str, Any]:
        recorded["payload"] = payload
        return {
            "dryRun": True,
            "orderId": 0,
            "origQty": payload.get("quantity", "0"),
            "avgPrice": "0.0",
            "status": "DRY",
        }

    monkeypatch.setattr(ex, "set_dry_run", fake_set_dry_run)
    monkeypatch.setattr(ex, "send_order", fake_send_order)

    intent = {
        "symbol": "ETHUSDT",
        "side": "buy",
        "quantity": 1,
    }
    result = route_order(intent, {"price": 1800.0}, True)

    assert result["accepted"] is True
    assert result["reason"] == "dry_run"
    assert result["order_id"] == 0
    assert recorded["dry_flags"][-1] is True
    assert recorded["payload"]["side"] == "BUY"
