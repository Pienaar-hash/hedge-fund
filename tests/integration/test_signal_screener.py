import time

import pytest

from execution import signal_screener as ss


def test_reduce_plan_covers_short_with_mark(monkeypatch):
    now = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    monkeypatch.setattr(time, "strftime", lambda *_args, **_kwargs: now)

    positions = {"SHORT": {"qty": 0.25, "mark": 20000.0, "notional": 5000.0}}
    intents, delta = ss._reduce_plan("BTCUSDT", "BUY", "15m", positions, 21000.0, 10000.0, 10.0)
    assert pytest.approx(delta, rel=1e-6) == 5000.0
    assert len(intents) == 1
    intent = intents[0]
    assert intent["reduceOnly"] is True
    assert intent["signal"] == "BUY"
    assert intent["positionSide"] == "SHORT"
    assert intent["per_trade_nav_pct"] == pytest.approx(0.5)
    assert intent["min_notional"] == pytest.approx(10.0)
    assert intent["timeframe"] == "15m"


def test_reduce_plan_respects_fallback_price(monkeypatch):
    now = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    monkeypatch.setattr(time, "strftime", lambda *_args, **_kwargs: now)

    positions = {"LONG": {"qty": 0.1, "mark": 0.0, "notional": 0.0}}
    intents, delta = ss._reduce_plan("ETHUSDT", "SELL", "1h", positions, 3000.0, 5000.0, 5.0)
    assert pytest.approx(delta, rel=1e-6) == 300.0
    assert len(intents) == 1
    intent = intents[0]
    assert intent["positionSide"] == "LONG"
    assert intent["signal"] == "SELL"
    assert intent["price"] == 3000.0


def test_reduce_plan_noop_when_no_flip():
    intents, delta = ss._reduce_plan("SOLUSDT", "BUY", "15m", {"LONG": {"qty": 0.2}}, 100.0, 0.0, 0.0)
    assert intents == []
    assert delta == 0.0
