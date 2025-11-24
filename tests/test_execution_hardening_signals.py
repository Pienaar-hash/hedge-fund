import execution.signal_generator as sg
from execution.position_sizing import inverse_vol_size


def test_inverse_vol_size_execution_hardening(monkeypatch):
    monkeypatch.setattr("execution.position_sizing.rolling_sigma", lambda symbol, lookback=50: 2.0)
    smaller = inverse_vol_size("WIFUSDT", 100.0, 50)
    monkeypatch.setattr("execution.position_sizing.rolling_sigma", lambda symbol, lookback=50: 1.0)
    larger = inverse_vol_size("WIFUSDT", 100.0, 50)
    assert larger > smaller


def test_allow_trade_expectancy_gate_execution_hardening(monkeypatch):
    monkeypatch.setattr("execution.signal_generator.atr_pct", lambda s, lookback_bars=50, median_only=False: 0.2)
    monkeypatch.setattr(
        "execution.signal_generator.rolling_expectancy",
        lambda symbol: -1.0,
    )
    assert sg.allow_trade("BTCUSDT") is False


def test_size_multiplier_shrinks_for_negative_sharpe(monkeypatch):
    from execution.utils.execution_health import size_multiplier

    monkeypatch.setattr("execution.utils.execution_health.rolling_sharpe_7d", lambda s: -0.5)
    multiplier = size_multiplier("BTCUSDC")
    assert multiplier == 1.0


def test_size_multiplier_grows_for_positive_sharpe(monkeypatch):
    from execution.utils.execution_health import size_multiplier

    monkeypatch.setattr("execution.utils.execution_health.rolling_sharpe_7d", lambda s: 2.0)
    multiplier = size_multiplier("BTCUSDC")
    assert multiplier == 1.0


def test_asset_universe_usdc_only_execution_hardening():
    from execution.utils.metrics import is_in_asset_universe

    assert is_in_asset_universe("BTCUSDC")
    assert not is_in_asset_universe("BTCUSDT")
    assert not is_in_asset_universe("ETHBUSD")
