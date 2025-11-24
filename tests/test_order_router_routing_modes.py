import pytest

from execution.order_router import PlaceOrderResult, route_order
from execution.intel.router_policy import RouterPolicy


@pytest.fixture(autouse=True)
def _stub_filters(monkeypatch):
    monkeypatch.setattr("execution.order_router.ex.get_symbol_filters", lambda _symbol: {})


def _base_intent():
    return {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "type": "LIMIT",
        "price": 100.0,
        "quantity": 1.0,
    }


def test_maker_first_runs_only_with_good_policy(monkeypatch):
    calls = {"submit": 0, "send": 0}
    monkeypatch.setattr(
        "execution.order_router.router_policy",
        lambda symbol: RouterPolicy(maker_first=True, taker_bias="balanced", quality="good", reason="ok", offset_bps=1.5),
    )

    def fake_submit(symbol, px, qty, side, post_only=True, prev=None, place_func=None):
        calls["submit"] += 1
        return PlaceOrderResult(
            order_id="abc",
            side=side,
            price=px,
            qty=qty,
            is_maker=True,
            raw={"orderId": "abc", "status": "FILLED", "avgPrice": px, "executedQty": qty, "origQty": qty},
        )

    def fake_send_order(**kwargs):
        calls["send"] += 1
        return {"orderId": "taker", "status": "NEW", "avgPrice": kwargs.get("price"), "executedQty": kwargs.get("quantity")}

    monkeypatch.setattr("execution.order_router.submit_limit", fake_submit)
    monkeypatch.setattr("execution.order_router.ex.send_order", fake_send_order)

    risk_ctx = {"payload": {}, "maker_first": True, "maker_qty": 1.0, "maker_price": 100.0}
    resp = route_order(_base_intent(), risk_ctx, dry_run=False)

    assert calls["submit"] == 1
    assert calls["send"] == 0
    assert risk_ctx["routed_as"] == "maker"
    assert resp["router_meta"]["is_maker_final"] is True
    assert resp["router_meta"]["decision"]["route"] == "maker"
    assert "policy_quality_not_good" not in resp["router_meta"]["decision"]["reasons"]


def test_broken_quality_forces_taker(monkeypatch):
    monkeypatch.setattr(
        "execution.order_router.router_policy",
        lambda symbol: RouterPolicy(maker_first=True, taker_bias="prefer_taker", quality="broken", reason="bad", offset_bps=1.0),
    )

    def deny_submit(*_args, **_kwargs):
        raise AssertionError("maker path should be bypassed")

    def fake_send_order(**_kwargs):
        return {"orderId": "taker", "status": "FILLED", "avgPrice": 101.0, "executedQty": 1.0}

    monkeypatch.setattr("execution.order_router.submit_limit", deny_submit)
    monkeypatch.setattr("execution.order_router.ex.send_order", fake_send_order)

    risk_ctx = {"payload": {}, "maker_first": True, "maker_qty": 1.0, "maker_price": 100.0}
    resp = route_order(_base_intent(), risk_ctx, dry_run=False)

    assert risk_ctx["routed_as"] == "taker"
    decision = resp["router_meta"]["decision"]
    assert decision["route"] == "taker"
    assert "policy_quality_not_good" in decision["reasons"]
    assert resp["router_meta"]["maker_start"] is False


def test_missing_orderbook_routes_taker_with_reason(monkeypatch):
    monkeypatch.setattr(
        "execution.order_router.router_policy",
        lambda symbol: RouterPolicy(maker_first=True, taker_bias="balanced", quality="good", reason="ok", offset_bps=1.5),
    )

    def deny_submit(*_args, **_kwargs):
        raise AssertionError("maker should not fire")

    monkeypatch.setattr("execution.order_router.submit_limit", deny_submit)
    monkeypatch.setattr(
        "execution.order_router.ex.send_order",
        lambda **_kwargs: {"orderId": "taker", "status": "NEW", "avgPrice": _kwargs.get("price"), "executedQty": _kwargs.get("quantity")},
    )

    risk_ctx = {"payload": {}, "maker_first": True, "maker_qty": 1.0, "maker_price": None}
    resp = route_order(_base_intent(), risk_ctx, dry_run=False)

    decision = resp["router_meta"]["decision"]
    assert decision["route"] == "taker"
    assert "missing_maker_price" in decision["reasons"]
    assert resp["router_meta"]["maker_start"] is False


def test_router_health_snapshot_schema(monkeypatch):
    from execution.executor_live import _build_router_health_snapshot

    monkeypatch.setattr("execution.executor_live.universe_by_symbol", lambda: {"BTCUSDT": {}, "ETHUSDT": {}})
    monkeypatch.setattr(
        "execution.executor_live.router_effectiveness_7d",
        lambda s: {
            "maker_fill_ratio": 0.5,
            "fallback_ratio": 0.1,
            "slip_q50": 1.0,
            "slip_q95": 2.0,
            "latency_p50_ms": 500.0,
        },
    )
    monkeypatch.setattr(
        "execution.executor_live.router_policy",
        lambda s: RouterPolicy(maker_first=True, taker_bias="balanced", quality="good", reason="ok", offset_bps=1.2),
    )

    snapshot = _build_router_health_snapshot()
    assert snapshot["summary"]["count"] == 2
    assert snapshot["summary"]["quality_counts"]["good"] == 2
    assert "per_symbol" in snapshot
    entry_map = {row["symbol"]: row for row in snapshot["symbols"]}
    assert entry_map["BTCUSDT"]["offset_bps"] == 1.2
    assert entry_map["BTCUSDT"]["policy"]["offset_bps"] == 1.2
    assert entry_map["BTCUSDT"]["ack_latency_ms"] == 500.0
