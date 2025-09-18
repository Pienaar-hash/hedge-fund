from execution.sizing import determine_position_size, estimate_annualized_volatility


def _mock_closes(start: float = 100.0, step: float = 0.5, n: int = 200):
    vals = []
    for i in range(n):
        drift = step * i
        noise = (step / 2.0) * ((-1) ** i)
        vals.append(start + drift + noise)
    return vals


def test_determine_position_size_respects_nav_cap():
    scfg = {"capital_per_trade": 50.0, "leverage": 5.0}
    sizing_cfg = {
        "default_leverage": 3.0,
        "max_trade_nav_pct": 2.0,
        "vol_target_annual_pct": 0.0,
        "kelly_fraction": 0.5,
    }
    risk_global = {"max_trade_nav_pct": 0.5}
    closes = _mock_closes()
    res = determine_position_size(
        symbol="BTCUSDT",
        strategy_cfg=scfg,
        sizing_cfg=sizing_cfg,
        risk_global_cfg=risk_global,
        nav=1000.0,
        timeframe="15m",
        closes=closes,
        price=closes[-1],
        exchange_min_notional=5.0,
        min_qty_notional=5.0,
        size_floor_usd=5.0,
    )
    assert res.ok is True
    # Nav capped at 0.5% -> 5 USD gross
    assert abs(res.gross - 5.0) < 1e-6


def test_determine_position_size_blocks_below_floor():
    scfg = {"capital_per_trade": 1.0, "leverage": 1.0}
    sizing_cfg = {
        "default_leverage": 1.0,
        "max_trade_nav_pct": 10.0,
        "vol_target_annual_pct": 0.0,
    }
    closes = _mock_closes()
    res = determine_position_size(
        symbol="ETHUSDT",
        strategy_cfg=scfg,
        sizing_cfg=sizing_cfg,
        risk_global_cfg={},
        nav=100.0,
        timeframe="1h",
        closes=closes,
        price=closes[-1],
        exchange_min_notional=25.0,
        min_qty_notional=25.0,
        size_floor_usd=25.0,
    )
    assert res.ok is False
    assert "below_size_floor" in res.reasons


def test_estimate_volatility_handles_window():
    closes = _mock_closes()
    vol = estimate_annualized_volatility(closes, "1h", 50)
    assert vol > 0
