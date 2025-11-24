from execution import signal_screener as ss
from execution import executor_live as ex


def test_generate_intents_emits_when_signal_cross(monkeypatch):
    monkeypatch.setattr(ss, "resolve_allowed_symbols", lambda: (["BTCUSDT"], {"BTCUSDT": "CORE"}))
    monkeypatch.setattr(ss, "get_positions", lambda: [], raising=False)
    monkeypatch.setattr(ss, "allow_trade", lambda _sym: True)
    monkeypatch.setattr(ss, "symbol_min_gross", lambda _sym: 10.0)
    monkeypatch.setattr(ss, "symbol_min_notional", lambda _sym: 10.0)
    monkeypatch.setattr(ss, "symbol_target_leverage", lambda _sym: 2.0)
    monkeypatch.setattr(ss, "_zscore", lambda *_args, **_kwargs: 1.2)
    monkeypatch.setattr(ss, "_rsi", lambda *_args, **_kwargs: 75.0)
    monkeypatch.setattr(
        ss,
        "get_symbol_filters",
        lambda *_args, **_kwargs: {"MIN_NOTIONAL": {"minNotional": 5}, "LOT_SIZE": {"minQty": 0.001, "stepSize": 0.001}},
    )
    monkeypatch.setattr(ss, "get_klines", lambda *_args, **_kwargs: [[0, 0, 0, 0, 100 + i, 0] for i in range(50)])
    monkeypatch.setattr(ss, "get_price", lambda *_args, **_kwargs: 100.0)

    class _Snap:
        def current_nav_usd(self):
            return 10_000.0

        def current_gross_usd(self):
            return 0.0

    monkeypatch.setattr(ss, "PortfolioSnapshot", lambda *_args, **_kwargs: _Snap())
    intents = ss.generate_intents()
    assert len(intents) >= 1
    first = intents[0]
    assert first.get("symbol") == "BTCUSDT"
    assert first.get("veto", []) == []


def test_internal_screener_submits_with_stubbed_intents(monkeypatch):
    submitted = []
    monkeypatch.setattr(ex, "run_screener_once", lambda: {"attempted": 1, "emitted": 1, "intents": [{"raw": {"symbol": "BTCUSDT", "signal": "BUY", "capital_per_trade": 10.0, "leverage": 1}}]})
    monkeypatch.setattr(ex, "_publish_intent_audit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ex, "_send_order", lambda intent: submitted.append(intent))
    monkeypatch.setattr(ex, "_symbol_on_cooldown", lambda *_args, **_kwargs: False)
    ex._LAST_SCREENER_RUN = 0
    ex.EXTERNAL_SIGNAL = False
    ex._maybe_run_internal_screener()
    assert submitted
    assert submitted[0]["symbol"] == "BTCUSDT"


def test_executor_respects_screener_gross(monkeypatch):
    calls = {}

    gross = ex._clamp_intent_gross("BTCUSDT", 100.0, 10000.0, 15.0)
    assert gross == 100.0
