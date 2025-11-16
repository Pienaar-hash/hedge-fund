from execution import executor_live


def test_send_order_flips_short_to_long(monkeypatch):
    positions_state = [
        {
            "symbol": "BTCUSDT",
            "positionSide": "SHORT",
            "qty": -0.01,
            "markPrice": 60000.0,
            "entryPrice": 60000.0,
        }
    ]

    def fake_get_positions():
        return [dict(p) for p in positions_state]

    def fake_get_price(symbol: str) -> float:
        assert symbol == "BTCUSDT"
        return 60000.0

    def fake_build_order_payload(
        symbol: str,
        side: str,
        price: float,
        desired_gross_usd: float,
        reduce_only: bool,
        position_side: str,
    ):
        payload = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": "0.010",
        }
        if reduce_only:
            payload["reduceOnly"] = "true"
        if position_side:
            payload["positionSide"] = position_side
        meta = {
            "normalized_price": price if price else 60000.0,
            "normalized_qty": 0.01,
        }
        return payload, meta

    sent_orders = []

    def fake_send_order(**payload):
        sent_orders.append(payload)
        if payload.get("reduceOnly") == "true":
            positions_state.clear()
        return {
            "orderId": len(sent_orders),
            "status": "FILLED",
            "avgPrice": "60000",
            "executedQty": "0.010",
        }

    monkeypatch.setattr(executor_live, "get_positions", fake_get_positions)
    monkeypatch.setattr(executor_live, "get_price", fake_get_price)
    monkeypatch.setattr(executor_live, "build_order_payload", fake_build_order_payload)
    monkeypatch.setattr(executor_live, "send_order", fake_send_order)
    monkeypatch.setattr(executor_live, "publish_intent_audit", lambda *_: None)
    monkeypatch.setattr(executor_live, "publish_order_audit", lambda *_: None)
    monkeypatch.setattr(executor_live, "publish_close_audit", lambda *_: None)
    monkeypatch.setattr(executor_live, "_compute_nav", lambda: 1000.0)
    monkeypatch.setattr(executor_live._PORTFOLIO_SNAPSHOT, "refresh", lambda: None)
    monkeypatch.setattr(executor_live._RISK_GATE, "_daily_loss_pct", lambda: 0.0)
    monkeypatch.setattr(executor_live._RISK_GATE, "allowed_gross_notional", lambda *args, **kwargs: (True, ""))
    monkeypatch.setattr(
        executor_live,
        "check_order",
        lambda **kwargs: (False, {}),
    )
    monkeypatch.setattr(executor_live, "DRY_RUN", False)

    executor_live._send_order(
        {
            "symbol": "BTCUSDT",
            "signal": "BUY",
            "capital_per_trade": 120.0,
            "leverage": 1.0,
            "price": 60000.0,
        }
    )

    assert len(sent_orders) == 2
    first, second = sent_orders
    assert first["reduceOnly"] == "true"
    assert first["positionSide"] == "SHORT"
    assert second.get("reduceOnly") != "true"
    assert second["positionSide"] == "LONG"


def test_opposite_position_detects_long():
    positions = [
        {"symbol": "ETHUSDT", "positionSide": "LONG", "qty": 5, "markPrice": 3000},
        {"symbol": "ETHUSDT", "positionSide": "SHORT", "qty": -2, "markPrice": 3000},
    ]
    side, qty, mark = executor_live._opposite_position("ETHUSDT", "SHORT", positions)
    assert side == "LONG"
    assert qty == 5
    assert mark == 3000
