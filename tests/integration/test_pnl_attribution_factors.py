from __future__ import annotations

from execution.pnl_tracker import build_pnl_attribution_snapshot


def _trade_with_hybrid(score: float | None, trend_strength: float | None, carry_long: float | None, carry_short: float | None, pnl: float) -> dict:
    return {
        "symbol": "BTCUSDT",
        "strategy": "vol_target",
        "realized_pnl": pnl,
        "metadata": {
            "strategy": "vol_target",
            "vol_target": {
                "hybrid": {
                    "hybrid_score": score,
                    "components": {
                        "trend_strength": trend_strength,
                        "carry_long": carry_long,
                        "carry_short": carry_short,
                    },
                },
                "trend": {"strength": trend_strength},
                "carry": {"score_long": carry_long, "score_short": carry_short},
            },
        },
    }


def test_no_hybrid_metadata_produces_no_factors():
    trade = {"symbol": "BTCUSDT", "strategy": "momentum", "realized_pnl": 1.0}
    snapshot = build_pnl_attribution_snapshot([trade], [])
    regimes = snapshot.get("regimes")
    assert not regimes or not regimes.get("factors")


def test_single_decile_bucket():
    trade = _trade_with_hybrid(score=0.02, trend_strength=0.1, carry_long=None, carry_short=None, pnl=5.0)
    snapshot = build_pnl_attribution_snapshot([trade], [])
    factors = snapshot["regimes"]["factors"]
    slot = factors["hybrid_score_decile"]["0"]
    assert slot["trade_count"] == 1
    assert slot["total_pnl"] == 5.0
    assert slot["avg_pnl"] == 5.0


def test_multiple_deciles_and_average():
    trades = [
        _trade_with_hybrid(score=0.05, trend_strength=0.2, carry_long=None, carry_short=None, pnl=2.0),
        _trade_with_hybrid(score=0.55, trend_strength=0.5, carry_long=None, carry_short=None, pnl=4.0),
        _trade_with_hybrid(score=0.95, trend_strength=0.9, carry_long=None, carry_short=None, pnl=-1.0),
    ]
    snapshot = build_pnl_attribution_snapshot(trades, [])
    factors = snapshot["regimes"]["factors"]
    assert factors["hybrid_score_decile"]["0"]["trade_count"] == 1
    assert factors["hybrid_score_decile"]["5"]["trade_count"] == 1
    assert factors["hybrid_score_decile"]["9"]["trade_count"] == 1
    assert factors["hybrid_score_decile"]["5"]["avg_pnl"] == 4.0


def test_trend_strength_buckets():
    trades = [
        _trade_with_hybrid(score=0.1, trend_strength=0.1, carry_long=None, carry_short=None, pnl=1.0),
        _trade_with_hybrid(score=0.2, trend_strength=0.5, carry_long=None, carry_short=None, pnl=2.0),
        _trade_with_hybrid(score=0.3, trend_strength=0.8, carry_long=None, carry_short=None, pnl=-1.0),
    ]
    snapshot = build_pnl_attribution_snapshot(trades, [])
    buckets = snapshot["regimes"]["factors"]["trend_strength_bucket"]
    assert buckets["weak"]["trade_count"] == 1
    assert buckets["medium"]["trade_count"] == 1
    assert buckets["strong"]["trade_count"] == 1


def test_carry_regimes():
    trades = [
        _trade_with_hybrid(score=0.1, trend_strength=0.2, carry_long=0.5, carry_short=0.1, pnl=3.0),
        _trade_with_hybrid(score=0.2, trend_strength=0.2, carry_long=0.0, carry_short=0.4, pnl=-2.0),
        _trade_with_hybrid(score=0.3, trend_strength=0.2, carry_long=0.2, carry_short=0.2, pnl=1.0),
    ]
    snapshot = build_pnl_attribution_snapshot(trades, [])
    carries = snapshot["regimes"]["factors"]["carry_regime"]
    assert carries["long_carry"]["trade_count"] == 1
    assert carries["short_carry"]["trade_count"] == 1
    assert carries["neutral"]["trade_count"] == 1
