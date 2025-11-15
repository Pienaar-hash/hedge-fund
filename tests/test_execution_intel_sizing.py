from execution.signal_generator import size_for


def test_size_for_respects_symbol_size_factor_execution_intelligence(monkeypatch):
    monkeypatch.setattr("execution.signal_generator.inverse_vol_size", lambda symbol, base_size, lookback=50: base_size)
    monkeypatch.setattr("execution.signal_generator.volatility_regime_scale", lambda symbol: 1.0)
    monkeypatch.setattr("execution.signal_generator.size_multiplier", lambda symbol: 1.0)
    monkeypatch.setattr(
        "execution.signal_generator.symbol_size_factor",
        lambda symbol: {"symbol": symbol, "score": -2.0, "size_factor": 0.5, "components": {}},
    )

    low = size_for("BTCUSDC", 100.0)
    assert low == 50.0

    monkeypatch.setattr(
        "execution.signal_generator.symbol_size_factor",
        lambda symbol: {"symbol": symbol, "score": 2.0, "size_factor": 1.5, "components": {}},
    )
    high = size_for("BTCUSDC", 100.0)
    assert high == 150.0
    assert high > low


def test_size_for_preserves_existing_stack_execution_intelligence(monkeypatch):
    monkeypatch.setattr("execution.signal_generator.inverse_vol_size", lambda symbol, base_size, lookback=50: base_size * 0.5)
    monkeypatch.setattr("execution.signal_generator.volatility_regime_scale", lambda symbol: 2.0)
    monkeypatch.setattr("execution.signal_generator.size_multiplier", lambda symbol: 1.2)
    monkeypatch.setattr(
        "execution.signal_generator.symbol_size_factor",
        lambda symbol: {"symbol": symbol, "score": 0.0, "size_factor": 0.8, "components": {}},
    )

    size = size_for("BTCUSDC", 100.0)
    assert size == 96.0
