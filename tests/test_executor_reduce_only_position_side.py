import os

import pytest

os.environ.setdefault("ALLOW_PROD_WRITE", "1")
os.environ.setdefault("FIRESTORE_ENABLED", "0")

_DOTENV = None
_ORIG_LOAD = None
try:
    import dotenv as _DOTENV  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _DOTENV = None
else:
    _ORIG_LOAD = _DOTENV.load_dotenv

    def _load_without_override(*args, **kwargs):
        kwargs["override"] = False
        return _ORIG_LOAD(*args, **kwargs)

    _DOTENV.load_dotenv = _load_without_override  # type: ignore[attr-defined]

from execution import executor_live

if _DOTENV is not None and _ORIG_LOAD is not None:
    _DOTENV.load_dotenv = _ORIG_LOAD


class StopSend(Exception):
    """Sentinel for halting _send_order after verifying payload."""


@pytest.fixture
def stub_executor_env(monkeypatch):
    builder_calls = []
    send_calls = []

    def fake_get_positions():
        return []

    def fake_get_price(symbol: str) -> float:
        assert symbol
        return 60000.0

    def fake_build_order_payload(
        symbol: str,
        side: str,
        price: float,
        desired_gross_usd: float,
        reduce_only: bool,
        position_side: str,
    ):
        builder_calls.append(
            {
                "symbol": symbol,
                "side": side,
                "reduce_only": reduce_only,
                "position_side": position_side,
            }
        )
        payload = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": "0.010",
        }
        if reduce_only:
            payload["reduceOnly"] = True
        if position_side:
            payload["positionSide"] = position_side
        meta = {"normalized_price": price or 60000.0, "normalized_qty": 0.01}
        return payload, meta

    def fake_send_order(**payload):
        send_calls.append(payload)
        raise StopSend

    monkeypatch.setattr(executor_live, "get_positions", fake_get_positions)
    monkeypatch.setattr(executor_live, "get_price", fake_get_price)
    monkeypatch.setattr(executor_live, "build_order_payload", fake_build_order_payload)
    monkeypatch.setattr(executor_live, "send_order", fake_send_order)
    monkeypatch.setattr(executor_live, "publish_intent_audit", lambda *_, **__: None)
    monkeypatch.setattr(executor_live, "publish_order_audit", lambda *_, **__: None)
    monkeypatch.setattr(executor_live, "publish_close_audit", lambda *_, **__: None)
    monkeypatch.setattr(executor_live, "_compute_nav", lambda: 1000.0)
    monkeypatch.setattr(executor_live._PORTFOLIO_SNAPSHOT, "refresh", lambda: None, raising=False)
    monkeypatch.setattr(executor_live._PORTFOLIO_SNAPSHOT, "current_nav_usd", lambda: 1000.0, raising=False)
    monkeypatch.setattr(executor_live._PORTFOLIO_SNAPSHOT, "current_gross_usd", lambda: 0.0, raising=False)
    monkeypatch.setattr(executor_live._PORTFOLIO_SNAPSHOT, "symbol_gross_usd", lambda: {}, raising=False)
    monkeypatch.setattr(executor_live._RISK_GATE, "_daily_loss_pct", lambda: 0.0, raising=False)
    monkeypatch.setattr(
        executor_live._RISK_GATE,
        "allowed_gross_notional",
        lambda *args, **kwargs: (True, ""),
        raising=False,
    )
    monkeypatch.setattr(executor_live, "check_order", lambda **kwargs: (True, {"reasons": []}))
    monkeypatch.setattr(executor_live, "_route_intent", None)
    monkeypatch.setattr(executor_live, "_route_order", None)
    monkeypatch.setattr(executor_live, "DRY_RUN", False)

    return {"build_calls": builder_calls, "send_calls": send_calls, "stop_exc": StopSend}


def _base_intent(signal: str) -> dict:
    return {
        "symbol": "BTCUSDT",
        "signal": signal,
        "reduceOnly": True,
        "capital_per_trade": 100.0,
        "leverage": 1.0,
        "price": 60000.0,
    }


def test_reduce_only_buy_sets_short_position(stub_executor_env):
    intent = _base_intent("BUY")

    with pytest.raises(stub_executor_env["stop_exc"]):
        executor_live._send_order(intent, skip_flip=True)

    assert stub_executor_env["build_calls"], "build_order_payload was not invoked"
    assert stub_executor_env["send_calls"], "send_order was not invoked"
    assert intent["positionSide"] == "SHORT"
    assert stub_executor_env["build_calls"][0]["position_side"] == "SHORT"
    assert stub_executor_env["send_calls"][0]["positionSide"] == "SHORT"


def test_reduce_only_sell_sets_long_position(stub_executor_env):
    intent = _base_intent("SELL")

    with pytest.raises(stub_executor_env["stop_exc"]):
        executor_live._send_order(intent, skip_flip=True)

    assert intent["positionSide"] == "LONG"
    assert stub_executor_env["build_calls"][0]["position_side"] == "LONG"
    assert stub_executor_env["send_calls"][0]["positionSide"] == "LONG"


def test_explicit_position_side_is_preserved(stub_executor_env):
    intent = _base_intent("BUY")
    intent["positionSide"] = "LONG"

    with pytest.raises(stub_executor_env["stop_exc"]):
        executor_live._send_order(intent, skip_flip=True)

    assert intent["positionSide"] == "LONG"
    assert stub_executor_env["build_calls"][0]["position_side"] == "LONG"
    assert stub_executor_env["send_calls"][0]["positionSide"] == "LONG"
