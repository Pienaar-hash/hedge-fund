from execution.intel import symbol_score_v6 as scores


def _snapshot(expectancy, hit_rate=0.6):
    return {"symbols": {"BTC": {"expectancy": expectancy, "hit_rate": hit_rate, "max_drawdown": 1.0}}}


def _router(**overrides):
    base = {
        "symbol": "BTC",
        "maker_fill_rate": 0.7,
        "fallback_rate": 0.1,
        "slippage_p50": 1.0,
        "fees_total": 1.0,
        "realized_pnl": 10.0,
        "volatility_scale": 1.0,
    }
    base.update(overrides)
    return {"symbols": [base]}


def test_score_symbol_increase_with_expectancy():
    low = scores.score_universe(_snapshot(2.0), _router())["symbols"][0]["score"]
    high = scores.score_universe(_snapshot(10.0), _router())["symbols"][0]["score"]
    assert high > low
    assert 0.0 <= high <= 1.0


def test_score_symbol_penalizes_slippage():
    good = scores.score_universe(_snapshot(5.0), _router(slippage_p50=1.0))["symbols"][0]["score"]
    bad = scores.score_universe(_snapshot(5.0), _router(slippage_p50=15.0))["symbols"][0]["score"]
    assert good > bad
