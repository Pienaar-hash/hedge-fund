from __future__ import annotations

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
