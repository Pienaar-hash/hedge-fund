import execution.utils.execution_health as eh


def test_classify_atr_regime_buckets_execution_hardening(monkeypatch):
    def quiet_atr(symbol, lookback_bars=50):
        return 0.5 if lookback_bars == 50 else 1.0

    monkeypatch.setattr("execution.utils.execution_health.atr_pct", quiet_atr)
    res = eh.classify_atr_regime("BTCUSDC")
    assert res["atr_regime"] == "quiet"

    def panic_atr(symbol, lookback_bars=50):
        return 3.5 if lookback_bars == 50 else 1.0

    monkeypatch.setattr("execution.utils.execution_health.atr_pct", panic_atr)
    res = eh.classify_atr_regime("BTCUSDC")
    assert res["atr_regime"] == "panic"


def test_classify_router_health_flags_execution_hardening():
    stats = {
        "maker_fill_ratio": 0.3,
        "fallback_ratio": 0.75,
        "slip_q25": 0.0,
        "slip_q50": 5.0,
        "slip_q75": 10.0,
    }
    res = eh.classify_router_health(stats)
    assert "high_fallback_ratio" in res["router_warnings"]
    assert "elevated_median_slippage" in res["router_warnings"]


def test_classify_risk_health_sharpe_states_execution_hardening(monkeypatch):
    monkeypatch.setattr("execution.utils.execution_health.dd_today_pct", lambda s: -4.0)
    monkeypatch.setattr("execution.utils.execution_health.is_symbol_disabled", lambda s: True)
    monkeypatch.setattr(
        "execution.utils.execution_health.get_symbol_disable_meta",
        lambda s: {"until": 9999999999.0, "reason": "dd_cap_hit"},
    )

    res = eh.classify_risk_health("BTCUSDC", sharpe=-1.5)
    assert "dd_warning" in res["risk_flags"]
    assert "dd_kill_threshold" in res["risk_flags"]
    assert "symbol_disabled" in res["risk_flags"]
    assert res["sharpe_state"] == "poor"


def test_compute_execution_health_combines_parts_execution_hardening(monkeypatch):
    monkeypatch.setattr(
        "execution.utils.execution_health.router_effectiveness_7d",
        lambda symbol: {
            "maker_fill_ratio": 0.6,
            "fallback_ratio": 0.2,
            "slip_q25": 0.0,
            "slip_q50": 1.0,
            "slip_q75": 2.0,
        },
    )
    monkeypatch.setattr("execution.utils.execution_health.rolling_sharpe_7d", lambda s: 2.0)
    monkeypatch.setattr("execution.utils.execution_health.dd_today_pct", lambda s: -0.5)
    monkeypatch.setattr("execution.utils.execution_health.is_symbol_disabled", lambda s: False)
    monkeypatch.setattr("execution.utils.execution_health.get_symbol_disable_meta", lambda s: None)

    def atr_stub(symbol, lookback_bars=50):
        return 1.0

    monkeypatch.setattr("execution.utils.execution_health.atr_pct", atr_stub)
    monkeypatch.setattr("execution.utils.execution_health.volatility_regime_scale", lambda s: 1.0)
    monkeypatch.setattr("execution.utils.execution_health.size_multiplier", lambda s: 1.5)
    monkeypatch.setattr(
        "execution.utils.execution_health.symbol_size_factor",
        lambda symbol: {"symbol": symbol, "size_factor": 0.8},
    )
    monkeypatch.setattr("execution.utils.execution_health.suggest_maker_offset_bps", lambda s: 3.0)
    from execution.intel.router_policy import RouterPolicy
    monkeypatch.setattr(
        "execution.utils.execution_health.router_policy",
        lambda symbol: RouterPolicy(maker_first=False, taker_bias="prefer_taker", quality="broken", reason="test"),
    )

    snapshot = eh.compute_execution_health("BTCUSDC")
    assert snapshot["symbol"] == "BTCUSDC"
    assert snapshot["router"]["maker_fill_ratio"] == 0.6
    assert snapshot["sizing"]["sharpe_7d"] == 2.0
    assert snapshot["sizing"]["size_mult_combined"] == 1.5
    assert snapshot["sizing"]["intel_size_factor"] == 0.8
    # floating point multiply: 1.5 * 0.8 ~= 1.2000000000000002
    assert abs(snapshot["sizing"]["final_size_factor"] - 1.2) < 1e-8
    assert snapshot["router"]["maker_offset_bps"] == 3.0
    assert snapshot["router"]["policy_quality"] == "broken"
    assert snapshot["router"]["policy_maker_first"] is False
    assert snapshot["router"]["policy_taker_bias"] == "prefer_taker"
