from __future__ import annotations

from execution.intel import feedback_allocator_v6 as allocator


def _risk_config():
    return {
        "global": {
            "max_trade_nav_pct": 0.2,
            "symbol_notional_share_cap_pct": 25.0,
        },
        "per_symbol": {
            "BTCUSDT": {"max_nav_pct": 0.25},
            "ETHUSDT": {"max_nav_pct": 0.2},
        },
    }


def _pairs_universe():
    return {
        "universe": [
            {"symbol": "BTCUSDT", "caps": {"max_nav_pct": 0.25, "max_concurrent_positions": 1}, "enabled": True},
            {"symbol": "ETHUSDT", "caps": {"max_nav_pct": 0.2, "max_concurrent_positions": 1}, "enabled": True},
        ]
    }


def _router_snapshot():
    return {
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "regime": "maker_strong",
                "quality": "good",
                "current_policy": {"maker_first": True, "taker_bias": 0.4},
                "proposed_policy": {"maker_first": True, "taker_bias": 0.35},
            },
            {
                "symbol": "ETHUSDT",
                "regime": "fallback_heavy",
                "quality": "degraded",
                "current_policy": {"maker_first": True, "taker_bias": 0.6},
                "proposed_policy": {"maker_first": False, "taker_bias": 0.7},
            },
        ]
    }


def test_feedback_allocator_prefers_high_score_symbol():
    expectancy = {
        "symbols": {
            "BTCUSDT": {"expectancy": 6.0, "hit_rate": 0.65},
            "ETHUSDT": {"expectancy": 1.0, "hit_rate": 0.55},
        },
        "lookback_hours": 168.0,
    }
    scores = {"symbols": [{"symbol": "BTCUSDT", "score": 0.85}, {"symbol": "ETHUSDT", "score": 0.3}]}
    risk_snapshot = {
        "symbols": [
            {"symbol": "BTCUSDT", "risk": {"dd_today_pct": -2.0}},
            {"symbol": "ETHUSDT", "risk": {"dd_today_pct": -1.0}},
        ]
    }

    payload = allocator.build_suggestions(
        expectancy_snapshot=expectancy,
        symbol_scores_snapshot=scores,
        router_policy_snapshot=_router_snapshot(),
        risk_snapshot=risk_snapshot,
        risk_config=_risk_config(),
        pairs_universe=_pairs_universe(),
        lookback_days=7.0,
    )
    btc = next(item for item in payload["symbols"] if item["symbol"] == "BTCUSDT")
    eth = next(item for item in payload["symbols"] if item["symbol"] == "ETHUSDT")
    assert btc["suggested_weight"] > eth["suggested_weight"]
    assert btc["suggested_caps"]["max_nav_pct"] >= eth["suggested_caps"]["max_nav_pct"]
    assert sum(item["suggested_weight"] for item in payload["symbols"]) <= 1.0
    assert payload["global"]["risk_mode"] == "normal"


def test_feedback_allocator_scales_down_in_cautious_mode():
    expectancy = {
        "symbols": {
            "BTCUSDT": {"expectancy": 6.0, "hit_rate": 0.65},
            "ETHUSDT": {"expectancy": 1.0, "hit_rate": 0.55},
        },
        "lookback_hours": 168.0,
    }
    scores = {"symbols": [{"symbol": "BTCUSDT", "score": 0.85}, {"symbol": "ETHUSDT", "score": 0.3}]}
    cautious_risk_snapshot = {
        "symbols": [
            {"symbol": "BTCUSDT", "risk": {"dd_today_pct": -6.0}},
            {"symbol": "ETHUSDT", "risk": {"dd_today_pct": -7.0}},
        ]
    }

    normal = allocator.build_suggestions(
        expectancy_snapshot=expectancy,
        symbol_scores_snapshot=scores,
        router_policy_snapshot=_router_snapshot(),
        risk_snapshot={"symbols": [{"symbol": "BTCUSDT", "risk": {"dd_today_pct": -2.0}}]},
        risk_config=_risk_config(),
        pairs_universe=_pairs_universe(),
    )
    cautious = allocator.build_suggestions(
        expectancy_snapshot=expectancy,
        symbol_scores_snapshot=scores,
        router_policy_snapshot=_router_snapshot(),
        risk_snapshot=cautious_risk_snapshot,
        risk_config=_risk_config(),
        pairs_universe=_pairs_universe(),
    )
    btc_normal = next(item for item in normal["symbols"] if item["symbol"] == "BTCUSDT")
    btc_cautious = next(item for item in cautious["symbols"] if item["symbol"] == "BTCUSDT")
    assert cautious["global"]["risk_mode"] == "cautious"
    assert btc_cautious["suggested_weight"] < btc_normal["suggested_weight"]
    assert btc_cautious["suggested_caps"]["max_nav_pct"] <= btc_normal["suggested_caps"]["max_nav_pct"]


def test_feedback_allocator_handles_missing_intel():
    payload = allocator.build_suggestions(
        expectancy_snapshot={"symbols": {}},
        symbol_scores_snapshot={"symbols": []},
        router_policy_snapshot={"symbols": []},
        risk_snapshot={"symbols": []},
        risk_config=_risk_config(),
        pairs_universe=_pairs_universe(),
    )
    assert payload["symbols"] == []
