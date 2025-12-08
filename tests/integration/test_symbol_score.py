from execution.intel.symbol_score import compute_symbol_score, score_to_size_factor, symbol_size_factor


def test_compute_symbol_score_uses_metrics_execution_intelligence(monkeypatch):
    monkeypatch.setattr(
        "execution.intel.symbol_score.rolling_sharpe_7d",
        lambda s: 2.0,
    )
    monkeypatch.setattr(
        "execution.intel.symbol_score.atr_pct",
        lambda s, lookback_bars=50: 1.0 if lookback_bars == 50 else 1.0,
    )
    monkeypatch.setattr(
        "execution.intel.symbol_score.router_effectiveness_7d",
        lambda s: {"maker_fill_ratio": 0.7, "fallback_ratio": 0.2, "slip_q50": 2.0},
    )
    monkeypatch.setattr(
        "execution.intel.symbol_score.dd_today_pct",
        lambda s: -0.5,
    )

    result = compute_symbol_score("BTCUSDC")
    assert result["symbol"] == "BTCUSDC"
    assert isinstance(result["score"], float)
    comps = result["components"]
    assert comps["sharpe"] == 2.0
    assert "router_score" in comps
    assert "atr_ratio" in comps


def test_score_to_size_factor_monotonic_execution_intelligence():
    low = score_to_size_factor(-2.0)
    mid = score_to_size_factor(0.0)
    high = score_to_size_factor(2.0)

    assert low < mid < high
    assert 0.25 <= low <= 1.0
    assert mid == 1.0
    assert high <= 2.0


def test_symbol_size_factor_wraps_score_and_factor_execution_intelligence(monkeypatch):
    def fake_compute(symbol):
        return {"symbol": symbol, "score": 2.0, "components": {}}

    monkeypatch.setattr(
        "execution.intel.symbol_score.compute_symbol_score",
        fake_compute,
    )

    payload = symbol_size_factor("BTCUSDC")
    assert payload["symbol"] == "BTCUSDC"
    assert payload["score"] == 2.0
    assert payload["size_factor"] > 1.0
