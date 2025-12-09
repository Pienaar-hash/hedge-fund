from __future__ import annotations

import json
from pathlib import Path

from execution.intel.hybrid_score_engine import HybridFactorPayload, compute_hybrid_scores


def _strategy_cfg() -> dict:
    cfg = json.loads(Path("config/strategy_config.json").read_text())
    fd = cfg.setdefault("factor_diagnostics", {})
    fd.setdefault("orthogonalization", {})["enabled"] = False
    return cfg


_VOL_REGIME_FACTOR = {
    "low": 0.25,
    "normal": 0.0,
    "high": -0.35,
    "crisis": -0.75,
}


def _payload(
    *,
    trend: float = 0.5,
    carry: float = 0.5,
    expectancy: float = 0.5,
    router_quality: float = 0.8,
    rv_momentum: float = 0.0,
    vol_regime_label: str = "normal",
) -> HybridFactorPayload:
    return HybridFactorPayload(
        symbol="BTCUSDT",
        direction="LONG",
        regime=vol_regime_label,
        vol_regime_label=vol_regime_label,
        router_quality_score=router_quality,
        factors={
            "trend": trend,
            "carry": carry,
            "expectancy": expectancy,
            "router_quality": router_quality,
            "rv_momentum": rv_momentum,
            "vol_regime": _VOL_REGIME_FACTOR.get(vol_regime_label, 0.0),
        },
    )


def test_weights_smoothed_and_clamped():
    raw_weights = {f: 0.9 for f in ["trend", "carry", "expectancy", "router_quality", "rv_momentum", "vol_regime"]}
    prev_weights = {f: 0.0 for f in raw_weights}

    result = compute_hybrid_scores(
        [_payload()],
        factor_weights_raw=raw_weights,
        prev_factor_weights=prev_weights,
        max_abs_score=1.0,
        strategy_config=_strategy_cfg(),
    )[0]

    weights = result["weights"]
    # Smoothing pulls weights down from raw 0.9, then clamp + renormalize
    assert all(0.05 <= w <= 0.40 for w in weights.values())
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_router_quality_multipliers():
    cfg = _strategy_cfg()
    payloads = [_payload(router_quality=0.1), _payload(router_quality=0.95)]
    low, high = compute_hybrid_scores(payloads, strategy_config=cfg)

    assert low["router_quality"]["multiplier"] < 1.0
    assert high["router_quality"]["multiplier"] >= 1.0
    assert high["hybrid_score"] > low["hybrid_score"]


def test_vol_regime_modifier_applies():
    cfg = _strategy_cfg()
    normal, high_vol = compute_hybrid_scores(
        [
            _payload(vol_regime_label="normal"),
            _payload(vol_regime_label="high"),
        ],
        strategy_config=cfg,
    )

    assert high_vol["vol_regime_modifier"] != normal["vol_regime_modifier"]
    assert high_vol["hybrid_score"] < normal["hybrid_score"]


def test_monotonic_trend_value_increases_score():
    payloads = [
        _payload(trend=0.2, carry=0.5, expectancy=0.5),
        _payload(trend=0.8, carry=0.5, expectancy=0.5),
    ]
    results = compute_hybrid_scores(payloads, strategy_config=_strategy_cfg())
    scores = {p.factors["trend"]: res["hybrid_score"] for p, res in zip(payloads, results)}
    assert scores[max(scores.keys())] > scores[min(scores.keys())]
