from __future__ import annotations

from execution.intel import router_autotune_v6


def _router_entry(symbol: str, maker_fill: float, fallback: float, slip_p95: float, policy_bias: str = "balanced"):
    return {
        "symbol": symbol,
        "maker_fill_rate": maker_fill,
        "fallback_rate": fallback,
        "slippage_p50": slip_p95 / 2.0,
        "slippage_p95": slip_p95,
        "policy": {"maker_first": True, "taker_bias": policy_bias},
    }


def test_router_autotune_prefers_maker_on_strong_expectancy(monkeypatch):
    router_health = {"symbols": [_router_entry("BTCUSDT", 0.82, 0.1, 3.0, "balanced")]}
    expectancy = {"symbols": {"BTCUSDT": {"expectancy": 6.0, "hit_rate": 0.65}}}
    scores = {"symbols": [{"symbol": "BTCUSDT", "score": 0.78}]}
    monkeypatch.setattr(router_autotune_v6.maker_offset, "suggest_maker_offset_bps", lambda symbol: 2.0)

    result = router_autotune_v6.build_suggestions(
        expectancy_snapshot=expectancy,
        symbol_scores_snapshot=scores,
        router_health_snapshot=router_health,
        risk_config={"router_autotune_v6": {"max_bias_step": 0.05, "max_offset_step_bps": 1.0}},
        lookback_days=7.0,
    )
    entry = result["symbols"][0]
    assert entry["proposed_policy"]["maker_first"] is True
    assert entry["proposed_policy"]["taker_bias"] <= entry["current_policy"]["taker_bias"]


def test_router_autotune_pushes_taker_when_fallback_heavy(monkeypatch):
    router_health = {"symbols": [_router_entry("ETHUSDT", 0.4, 0.7, 8.0, "balanced")]}
    expectancy = {"symbols": {"ETHUSDT": {"expectancy": -2.0, "hit_rate": 0.4}}}
    scores = {"symbols": [{"symbol": "ETHUSDT", "score": 0.2}]}
    monkeypatch.setattr(router_autotune_v6.maker_offset, "suggest_maker_offset_bps", lambda symbol: 1.5)

    result = router_autotune_v6.build_suggestions(
        expectancy_snapshot=expectancy,
        symbol_scores_snapshot=scores,
        router_health_snapshot=router_health,
        risk_config={"router_autotune_v6": {"max_bias_step": 0.05, "max_offset_step_bps": 1.0}},
        lookback_days=7.0,
    )
    entry = result["symbols"][0]
    assert entry["proposed_policy"]["taker_bias"] > entry["current_policy"]["taker_bias"]
    assert entry["proposed_policy"]["offset_bps"] >= entry["current_policy"]["offset_bps"]


def test_router_autotune_respects_bias_and_offset_bounds(monkeypatch):
    router_health = {"symbols": [_router_entry("SOLUSDT", 0.15, 0.9, 20.0, "prefer_maker")]}
    expectancy = {"symbols": {"SOLUSDT": {"expectancy": -4.0, "hit_rate": 0.3}}}
    scores = {"symbols": [{"symbol": "SOLUSDT", "score": 0.1}]}
    monkeypatch.setattr(router_autotune_v6.maker_offset, "suggest_maker_offset_bps", lambda symbol: 0.6)

    bounds = {"router_autotune_v6": {"max_bias_step": 0.04, "max_offset_step_bps": 0.8, "min_bias": 0.0, "max_bias": 1.0}}
    result = router_autotune_v6.build_suggestions(
        expectancy_snapshot=expectancy,
        symbol_scores_snapshot=scores,
        router_health_snapshot=router_health,
        risk_config=bounds,
        lookback_days=7.0,
    )
    entry = result["symbols"][0]
    current = entry["current_policy"]
    proposed = entry["proposed_policy"]
    assert 0.0 <= proposed["taker_bias"] <= 1.0
    assert proposed["offset_bps"] >= current["offset_bps"]
    assert abs(proposed["taker_bias"] - current["taker_bias"]) <= 0.04 + 1e-9
    assert abs(proposed["offset_bps"] - current["offset_bps"]) <= 0.8 + 1e-9
