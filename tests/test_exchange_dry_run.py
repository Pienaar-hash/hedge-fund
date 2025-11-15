from __future__ import annotations

import pytest

import execution.exchange_utils as ex


def test_set_dry_run_blocks_signed_order(monkeypatch) -> None:
    calls: list[tuple] = []

    def fake_request(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("dry-run should skip signed requests")

    monkeypatch.setattr(ex._S, "request", fake_request)
    previous = ex.is_dry_run()
    try:
        ex.set_dry_run(True)
        resp = ex.send_order("BTCUSDT", "BUY", "MARKET", "1")
        assert resp["dryRun"] is True
        assert resp["status"] == "DRY_RUN"
        assert resp["symbol"] == "BTCUSDT"
        assert calls == []
    finally:
        ex.set_dry_run(previous)


def test_set_dry_run_false_allows_signed_order(monkeypatch) -> None:
    class DummyResponse:
        def __init__(self) -> None:
            self.status_code = 200
            self._payload = {"orderId": 42, "status": "FILLED"}

        def json(self) -> dict[str, object]:
            return self._payload

        def raise_for_status(self) -> None:
            return None

    captured: dict[str, object] = {}

    def fake_request(method, url, data=None, timeout=None, headers=None):
        captured["method"] = method
        captured["url"] = url
        captured["data"] = data
        captured["timeout"] = timeout
        captured["headers"] = headers
        return DummyResponse()

    monkeypatch.setattr(ex._S, "request", fake_request)
    previous = ex.is_dry_run()
    try:
        ex.set_dry_run(False)
        resp = ex.send_order("ETHUSDT", "SELL", "LIMIT", "1", price="1800")
        assert captured["method"] == "POST"
        assert str(captured["url"]).endswith("/fapi/v1/order")
        payload = str(captured["data"] or "")
        assert "signature=" in payload
        assert resp["orderId"] == 42
    finally:
        ex.set_dry_run(previous)


@pytest.fixture
def capture_signed_order(monkeypatch):
    captured: dict[str, object] = {}
    positions_state = {
        "data": [
            {
                "symbol": "BTCUSDT",
                "positionSide": "LONG",
                "qty": 0.006,
            }
        ]
    }

    class DummyResponse:
        def json(self) -> dict[str, object]:
            return {"orderId": 7, "status": "NEW"}

    def fake_positions(symbol=None):
        return positions_state["data"]

    def fake_req(method, path, *, signed=False, params=None, timeout=None):
        captured["method"] = method
        captured["path"] = path
        captured["signed"] = signed
        captured["params"] = dict(params or {})
        return DummyResponse()

    monkeypatch.setattr(ex, "_req", fake_req)
    monkeypatch.setattr(ex, "get_positions", fake_positions)
    monkeypatch.setattr(ex, "UMFutures", object(), raising=False)
    previous = ex.is_dry_run()
    ex.set_dry_run(False)
    try:
        yield {"captured": captured, "positions": positions_state}
    finally:
        ex.set_dry_run(previous)


def test_market_reduce_only_keeps_quantity_and_reduce_flag(capture_signed_order) -> None:
    ctx = capture_signed_order
    resp = ex.send_order(
        "BTCUSDT",
        "SELL",
        "MARKET",
        "0.006",
        positionSide="LONG",
        reduceOnly=True,
    )
    payload = ctx["captured"]["params"]
    assert resp["orderId"] == 7
    assert payload["type"] == "MARKET"
    assert payload["quantity"] == "0.006"
    assert payload["reduceOnly"] is True
    assert "closePosition" not in payload
    assert set(payload.keys()) == {"symbol", "side", "type", "quantity", "positionSide", "reduceOnly"}


def test_market_reduce_only_usdc_strips_reduce_flag(capture_signed_order) -> None:
    ctx = capture_signed_order
    resp = ex.send_order(
        "BTCUSDC",
        "SELL",
        "MARKET",
        "0.006",
        positionSide="LONG",
        reduceOnly=True,
    )
    payload = ctx["captured"]["params"]
    assert resp["orderId"] == 7
    assert payload["type"] == "MARKET"
    assert payload["quantity"] == "0.006"
    assert "reduceOnly" not in payload
    assert "positionSide" not in payload
    assert set(payload.keys()) == {"symbol", "side", "type", "quantity"}


def test_stop_market_reduce_only_converts_to_close_position(capture_signed_order) -> None:
    ctx = capture_signed_order
    resp = ex.send_order(
        "BTCUSDT",
        "SELL",
        "STOP_MARKET",
        "0.006",
        positionSide="LONG",
        reduceOnly=True,
    )
    payload = ctx["captured"]["params"]
    assert resp["status"] == "NEW"
    assert payload["type"] == "STOP_MARKET"
    assert payload["closePosition"] is True
    assert "quantity" not in payload
    assert "reduceOnly" not in payload
