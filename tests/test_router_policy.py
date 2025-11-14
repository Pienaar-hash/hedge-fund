from execution.intel.router_policy import classify_router_quality, router_policy


def test_classify_router_quality_tiers_execution_intelligence():
    assert classify_router_quality({"maker_fill_ratio": 0.8, "fallback_ratio": 0.2, "slip_q50": 2.0}) == "good"
    assert classify_router_quality({"maker_fill_ratio": 0.4, "fallback_ratio": 0.7, "slip_q50": 6.0}) == "degraded"
    assert classify_router_quality({"maker_fill_ratio": 0.2, "fallback_ratio": 0.95, "slip_q50": 25.0}) == "broken"
    assert classify_router_quality({"maker_fill_ratio": 0.5, "fallback_ratio": 0.3, "slip_q50": 5.0}) == "ok"


def test_router_policy_disables_maker_when_broken_execution_intelligence(monkeypatch):
    monkeypatch.setattr(
        "execution.intel.router_policy.router_effectiveness_7d",
        lambda s: {"maker_fill_ratio": 0.2, "fallback_ratio": 0.95, "slip_q50": 30.0},
    )
    monkeypatch.setattr("execution.intel.router_policy.classify_atr_regime", lambda s: "normal")

    policy = router_policy("BTCUSDC")
    assert policy.quality == "broken"
    assert policy.maker_first is False
    assert policy.taker_bias == "prefer_taker"


def test_router_policy_prefers_maker_when_good_execution_intelligence(monkeypatch):
    monkeypatch.setattr(
        "execution.intel.router_policy.router_effectiveness_7d",
        lambda s: {"maker_fill_ratio": 0.8, "fallback_ratio": 0.2, "slip_q50": 2.0},
    )
    monkeypatch.setattr("execution.intel.router_policy.classify_atr_regime", lambda s: "normal")

    policy = router_policy("ETHUSDC")
    assert policy.quality == "good"
    assert policy.maker_first is True
    assert policy.taker_bias == "prefer_maker"
