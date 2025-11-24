import pytest

from execution.intel.router_policy import (
    classify_router_quality,
    classify_router_regime,
    compute_router_summary,
    router_policy,
)


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


def test_router_policy_carries_offset(monkeypatch):
    monkeypatch.setattr(
        "execution.intel.router_policy.router_effectiveness_7d",
        lambda s: {"maker_fill_ratio": 0.75, "fallback_ratio": 0.2, "slip_q50": 2.0},
    )
    monkeypatch.setattr("execution.intel.router_policy.classify_atr_regime", lambda s: "normal")
    monkeypatch.setattr("execution.intel.router_policy.maker_offset.suggest_maker_offset_bps", lambda _s: 1.25)

    policy = router_policy("BTCUSDC")
    assert policy.offset_bps == 1.25
    assert policy.maker_first is True


def test_classify_router_regime_variants():
    assert (
        classify_router_regime({"maker_fill_rate": 0.8, "fallback_rate": 0.1, "slippage_p95": 3.0})
        == "maker_strong"
    )
    assert (
        classify_router_regime({"maker_fill_rate": 0.4, "fallback_rate": 0.65, "slippage_p95": 4.0})
        == "fallback_heavy"
    )
    assert (
        classify_router_regime({"maker_fill_rate": 0.5, "fallback_rate": 0.2, "slippage_p95": 14.0})
        == "slippage_hot"
    )


def test_compute_router_summary_normalizes_inputs():
    events = [
        {"is_maker_final": True, "maker_start": True, "used_fallback": False, "slippage_bps": 1.0, "latency_ms": 400},
        {"is_maker_final": False, "maker_start": True, "used_fallback": True, "slippage_bps": 12.0, "latency_ms": 800},
    ]
    summary = compute_router_summary(events)
    assert summary["maker_fill_rate"] == 0.5
    assert summary["fallback_rate"] == 0.5
    assert summary["slippage_p50"] == pytest.approx(6.5)
    assert summary["slippage_p95"] == pytest.approx(11.45)
