from dashboard.live_helpers import get_hourly_expectancy, get_symbol_score


def test_get_symbol_score_delegates_to_intel_execution_intelligence(monkeypatch):
    calls = {}

    def fake_compute(symbol: str):
        calls["symbol"] = symbol
        return {"symbol": symbol, "score": 1.0, "components": {"dummy": 1}}

    monkeypatch.setattr("dashboard.live_helpers.compute_symbol_score", fake_compute)

    result = get_symbol_score("BTCUSDC")
    assert result["symbol"] == "BTCUSDC"
    assert calls["symbol"] == "BTCUSDC"


def test_get_hourly_expectancy_delegates_to_intel_execution_intelligence(monkeypatch):
    def fake_hourly(symbol=None):
        return {9: {"count": 5, "exp_per_notional": 0.001, "slip_bps_avg": 1.0}}

    monkeypatch.setattr("dashboard.live_helpers.intel_hourly_expectancy", fake_hourly)

    result = get_hourly_expectancy("BTCUSDC")
    assert 9 in result
    assert result[9]["count"] == 5
