from __future__ import annotations

import sys
import types

from execution import signal_generator as sg


def test_signal_gen_smoke(monkeypatch) -> None:
    sg._SEEN_KEYS.clear()

    momentum_mod = types.ModuleType("strategies.momentum")
    relative_mod = types.ModuleType("strategies.relative_value")

    def momentum_signals(**_: object):
        return [
            {"symbol": "BTCUSDT", "timeframe": "1h", "signal": "BUY", "candle_close": 100},
            {"symbol": "BTCUSDT", "timeframe": "1h", "signal": "BUY", "candle_close": 100},
        ]

    def relative_signals(**_: object):
        return [
            {"symbol": "ETHUSDT", "timeframe": "4h", "signal": "SELL", "candle_close": 200},
            {"symbol": "BTCUSDT", "timeframe": "1h", "signal": "BUY", "candle_close": 100},
        ]

    setattr(momentum_mod, "generate_signals", momentum_signals)
    setattr(relative_mod, "generate_signals", relative_signals)

    orig_momentum = sys.modules.get("strategies.momentum")
    orig_relative = sys.modules.get("strategies.relative_value")

    sys.modules["strategies.momentum"] = momentum_mod
    sys.modules["strategies.relative_value"] = relative_mod

    try:
        first = sg.generate_intents(now=1.0, universe=["BTC", "ETH"], cfg={"foo": "bar"})
        assert len(first) == 2
        keys = {(item["symbol"], item["signal"]) for item in first}
        assert ("BTCUSDT", "BUY") in keys
        assert ("ETHUSDT", "SELL") in keys

        second = sg.generate_intents(now=2.0, universe=["BTC", "ETH"], cfg={"foo": "bar"})
        assert second == []
    finally:
        sg._SEEN_KEYS.clear()
        if orig_momentum is not None:
            sys.modules["strategies.momentum"] = orig_momentum
        else:
            sys.modules.pop("strategies.momentum", None)
        if orig_relative is not None:
            sys.modules["strategies.relative_value"] = orig_relative
        else:
            sys.modules.pop("strategies.relative_value", None)
