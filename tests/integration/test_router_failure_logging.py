from __future__ import annotations

import pytest
import requests

from execution import order_router
import execution.exchange_utils as ex
import execution.utils.execution_health as eh
from execution.intel.router_policy import RouterPolicy


def test_route_order_failure_records_error(monkeypatch):
    eh.reset_error_registry()
    resp = requests.Response()
    resp.status_code = 503
    resp._content = b'{"code": -1003, "msg": "Service unavailable"}'  # type: ignore[attr-defined]

    def failing_send_order(**kwargs):
        raise requests.HTTPError(response=resp)

    monkeypatch.setattr(order_router.ex, "send_order", failing_send_order)
    monkeypatch.setattr(order_router.ex, "get_symbol_filters", lambda symbol: {})
    monkeypatch.setattr(order_router, "router_policy", lambda symbol: RouterPolicy(False, "prefer_taker", "ok", "test"))
    monkeypatch.setattr(eh, "router_effectiveness_7d", lambda symbol=None: {})
    monkeypatch.setattr(eh, "rolling_sharpe_7d", lambda s: 0.0)
    monkeypatch.setattr(eh, "dd_today_pct", lambda s: 0.0)
    monkeypatch.setattr(eh, "is_symbol_disabled", lambda s: False)
    monkeypatch.setattr(eh, "get_symbol_disable_meta", lambda s: None)

    with pytest.raises(requests.HTTPError):
        order_router.route_order(
            {"symbol": "BTCUSDT", "side": "BUY", "type": "MARKET", "quantity": 1},
            {"payload": {"symbol": "BTCUSDT", "side": "BUY", "type": "MARKET", "quantity": "1"}},
            False,
        )

    health = eh.compute_execution_health("BTCUSDT")
    router_errors = health["errors"].get("router", {})
    assert router_errors["count"] >= 1
    assert router_errors["last_error"]["classification"]["category"] in {"exchange_server", "rate_limit", "http_error"}
