import execution.risk_limits as rl


def test_symbol_notional_cap_execution_hardening(monkeypatch):
    monkeypatch.setattr(rl, "total_notional_7d", lambda: 1000.0)
    monkeypatch.setattr(rl, "notional_7d_by_symbol", lambda symbol: 400.0)
    assert rl.symbol_notional_guard("LTCUSDC") is False


def test_symbol_dd_guard_execution_hardening(monkeypatch):
    triggered = {"disabled": False}
    monkeypatch.setattr(rl, "dd_today_pct", lambda symbol: -5.0)

    def fake_disable(symbol, ttl_hours=0, reason=None):
        triggered["disabled"] = True

    monkeypatch.setattr(rl, "disable_symbol_temporarily", fake_disable)
    assert rl.symbol_dd_guard("LTCUSDC") is False
    assert triggered["disabled"] is True
