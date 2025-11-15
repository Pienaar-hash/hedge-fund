from execution.intel.maker_offset import suggest_maker_offset_bps


def test_offset_tightens_when_router_good_execution_intelligence(monkeypatch):
    monkeypatch.setattr(
        "execution.intel.maker_offset.router_effectiveness_7d",
        lambda s: {"maker_fill_ratio": 0.9, "fallback_ratio": 0.1, "slip_q50": 0.5},
    )
    monkeypatch.setattr(
        "execution.intel.maker_offset.classify_atr_regime",
        lambda s: "quiet",
    )
    monkeypatch.setattr(
        "execution.intel.maker_offset.hourly_expectancy",
        lambda s: {12: {"exp_per_notional": 0.001}},
    )
    monkeypatch.setattr("execution.intel.maker_offset._current_hour", lambda: 12)

    off = suggest_maker_offset_bps("BTCUSDC")
    assert off < 2.0


def test_offset_widens_in_hot_regime_or_high_fallback_execution_intelligence(monkeypatch):
    monkeypatch.setattr(
        "execution.intel.maker_offset.router_effectiveness_7d",
        lambda s: {"maker_fill_ratio": 0.4, "fallback_ratio": 0.7, "slip_q50": 4.0},
    )
    monkeypatch.setattr(
        "execution.intel.maker_offset.classify_atr_regime",
        lambda s: "hot",
    )
    monkeypatch.setattr(
        "execution.intel.maker_offset.hourly_expectancy",
        lambda s: {15: {"exp_per_notional": -0.002}},
    )
    monkeypatch.setattr("execution.intel.maker_offset._current_hour", lambda: 15)

    off = suggest_maker_offset_bps("BTCUSDC")
    assert off > 2.0
