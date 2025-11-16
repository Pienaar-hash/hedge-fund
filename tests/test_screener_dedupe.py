from __future__ import annotations

import execution.signal_screener as sc


def _setup_stubs(monkeypatch) -> None:
    sc._DEDUP_CACHE.clear()
    monkeypatch.setattr(sc, "is_listed_on_futures", lambda symbol: True)
    monkeypatch.setattr(
        sc,
        "evaluate_entry_gate",
        lambda *args, **kwargs: (False, {"metric": 0.0}),
    )

    def _ok_check_order(**kwargs):
        return False, {}

    monkeypatch.setattr(sc, "check_order", _ok_check_order)


def test_screener_dedupe_same_candle(monkeypatch) -> None:
    _setup_stubs(monkeypatch)
    tf = "15m"
    candle = 1_700_000_000.0

    ok1, reasons1, _ = sc.would_emit(
        "BTCUSDT",
        "BUY",
        notional=10.0,
        lev=20.0,
        nav=1000.0,
        timeframe=tf,
        candle_close_ts=candle,
    )

    assert ok1 is True
    assert reasons1 == []

    ok2, reasons2, _ = sc.would_emit(
        "BTCUSDT",
        "BUY",
        notional=10.0,
        lev=20.0,
        nav=1000.0,
        timeframe=tf,
        candle_close_ts=candle,
    )

    assert ok2 is False
    assert "dedupe" in reasons2


def test_screener_dedupe_next_candle(monkeypatch) -> None:
    _setup_stubs(monkeypatch)
    tf = "1m"
    base_candle = 1_700_000_000.0

    ok1, _, _ = sc.would_emit(
        "ETHUSDT",
        "SELL",
        notional=5.0,
        lev=10.0,
        nav=1000.0,
        timeframe=tf,
        candle_close_ts=base_candle,
    )
    assert ok1 is True

    ok2, reasons2, _ = sc.would_emit(
        "ETHUSDT",
        "SELL",
        notional=5.0,
        lev=10.0,
        nav=1000.0,
        timeframe=tf,
        candle_close_ts=base_candle + 60.0,
    )
    assert ok2 is True
    assert "dedupe" not in reasons2
