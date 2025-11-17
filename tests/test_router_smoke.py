from __future__ import annotations

from typing import Any, Dict

import execution.exchange_utils as ex
from execution.order_router import PlaceOrderResult, route_order
from execution.intel.router_policy import RouterPolicy


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


def test_router_policy_neutral_keeps_maker_path(monkeypatch) -> None:
    calls: dict[str, int] = {"maker": 0, "taker": 0}

    def fake_set_dry_run(flag: bool) -> None:
        return None

    def fake_send_order(**payload: Any) -> Dict[str, Any]:
        calls["taker"] += 1
        return {
            "orderId": 555,
            "status": "NEW",
            "avgPrice": payload.get("price", "0"),
            "executedQty": payload.get("quantity", "0"),
        }

    def fake_submit_limit(symbol: str, price: float, qty: float, side: str) -> PlaceOrderResult:
        calls["maker"] += 1
        return PlaceOrderResult(
            order_id="maker-1",
            side=side,
            price=price,
            qty=qty,
            is_maker=True,
            raw={
                "orderId": "maker-1",
                "status": "NEW",
                "avgPrice": str(price),
                "executedQty": "0",
                "origQty": str(qty),
            },
        )

    monkeypatch.setattr(ex, "set_dry_run", fake_set_dry_run)
    monkeypatch.setattr(ex, "send_order", fake_send_order)
    monkeypatch.setattr(ex, "get_symbol_filters", lambda _s: {})
    monkeypatch.setattr(
        "execution.order_router.submit_limit",
        fake_submit_limit,
    )
    monkeypatch.setattr(
        "execution.order_router.router_policy",
        lambda _s: RouterPolicy(maker_first=True, taker_bias="neutral", quality="ok", reason="test"),
    )
    monkeypatch.setattr(
        "execution.order_router.suggest_maker_offset_bps",
        lambda _s: 0.0,
    )

    intent = {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "quantity": "0.10",
        "type": "MARKET",
    }
    router_ctx = {"maker_first": True, "maker_qty": "0.10", "maker_price": 100.0}

    route_order(intent, router_ctx, dry_run=False)

    assert calls["maker"] == 1
    assert calls["taker"] == 0
    assert router_ctx.get("routed_as") == "maker"


def test_router_policy_prefer_taker_overrides_maker(monkeypatch) -> None:
    calls: dict[str, int] = {"maker": 0, "taker": 0}

    monkeypatch.setattr(ex, "set_dry_run", lambda flag: None)
    monkeypatch.setattr(ex, "get_symbol_filters", lambda _s: {})

    def fake_send_order(**payload: Any) -> Dict[str, Any]:
        calls["taker"] += 1
        return {"orderId": 42, "status": "NEW", "avgPrice": payload.get("price", "0"), "executedQty": "0"}

    monkeypatch.setattr(ex, "send_order", fake_send_order)
    def fail_submit_limit(*_args, **_kwargs):
        calls["maker"] += 1
        raise AssertionError("maker path should be skipped")

    monkeypatch.setattr("execution.order_router.submit_limit", fail_submit_limit)
    monkeypatch.setattr(
        "execution.order_router.router_policy",
        lambda _s: RouterPolicy(maker_first=True, taker_bias="prefer_taker", quality="ok", reason="test"),
    )

    intent = {
        "symbol": "ETHUSDT",
        "side": "SELL",
        "quantity": "0.25",
        "type": "MARKET",
    }
    router_ctx = {"maker_first": True, "maker_qty": "0.25", "maker_price": 1800.0}

    route_order(intent, router_ctx, dry_run=False)

    assert calls["maker"] == 0
    assert calls["taker"] == 1
    assert router_ctx.get("routed_as") == "taker"


def test_route_order_router_meta_fallback(monkeypatch) -> None:
    calls: dict[str, int] = {"maker": 0, "taker": 0}

    monkeypatch.setattr(ex, "set_dry_run", lambda flag: None)
    monkeypatch.setattr(ex, "get_symbol_filters", lambda _s: {})

    def fake_send_order(**payload: Any) -> Dict[str, Any]:
        calls["taker"] += 1
        return {"orderId": 777, "status": "NEW", "avgPrice": payload.get("price"), "executedQty": payload.get("quantity")}

    def fail_submit_limit(symbol: str, price: float, qty: float, side: str) -> PlaceOrderResult:
        calls["maker"] += 1
        raise RuntimeError("maker failed")

    monkeypatch.setattr(ex, "send_order", fake_send_order)
    monkeypatch.setattr("execution.order_router.submit_limit", fail_submit_limit)
    monkeypatch.setattr(
        "execution.order_router.router_policy",
        lambda _s: RouterPolicy(maker_first=True, taker_bias="balanced", quality="ok", reason="test"),
    )
    monkeypatch.setattr(
        "execution.order_router.suggest_maker_offset_bps",
        lambda _s: 0.0,
    )

    intent = {"symbol": "BTCUSDT", "side": "BUY", "quantity": "0.1", "type": "MARKET"}
    router_ctx = {"maker_first": True, "maker_qty": "0.1", "maker_price": 100.0}

    result = route_order(intent, router_ctx, dry_run=False)

    assert calls["maker"] == 1
    assert calls["taker"] == 1
    meta = result.get("router_meta") or {}
    assert meta.get("maker_start") is True
    assert meta.get("is_maker_final") is False
    assert meta.get("used_fallback") is True
    assert meta.get("router_policy", {}).get("reason") == "test"
