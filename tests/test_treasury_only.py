from execution import nav as navmod


def test_treasury_only_valuation(monkeypatch):
    monkeypatch.setattr(navmod, "get_price", lambda sym: {"BTCUSDT": 50000.0, "XAUTUSDT": 2400.0}[sym])
    monkeypatch.setattr(navmod, "_load_json", lambda *_: {"BTC": 0.01, "XAUT": 0.5})
    val, detail = navmod.compute_treasury_only()
    assert int(val) == int(0.01 * 50000 + 0.5 * 2400)
    assert "treasury" in detail and set(detail["treasury"].keys()) == {"BTC", "XAUT"}
