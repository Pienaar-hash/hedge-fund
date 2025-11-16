from execution.risk_limits import RiskState, check_order


def _base_cfg():
    return {
        "global": {
            "min_notional_usdt": 10.0,
            "max_portfolio_gross_nav_pct": 15.0,
            "max_concurrent_positions": 3,
            "tiers": {
                "CORE": {"per_symbol_nav_pct": 8.0},
                "SATELLITE": {"per_symbol_nav_pct": 4.0},
                "TACTICAL": {"per_symbol_nav_pct": 2.0},
                "ALT-EXT": {"per_symbol_nav_pct": 1.0},
            },
        },
        "per_symbol": {"BTCUSDT": {"max_order_notional": 25.0}},
    }


def test_portfolio_cap_blocks():
    st = RiskState()
    cfg = _base_cfg()
    # nav=1000, portfolio cap 15% => 150, current gross 149, request 5 -> block
    veto, details = check_order(
        symbol="BTCUSDT",
        side="BUY",
        requested_notional=5.0,
        price=50000.0,
        nav=1000.0,
        open_qty=0.0,
        now=0.0,
        cfg=cfg,
        state=st,
        current_gross_notional=149.0,
    )
    assert veto is True
    rs = details.get("reasons", [])
    assert "portfolio_cap" in rs or "max_gross_nav_pct" in rs


def test_tier_cap_blocks():
    st = RiskState()
    cfg = _base_cfg()
    # CORE: per-symbol cap 8% of 1000 => 80. current tier gross=79, request 5 -> block
    veto, details = check_order(
        symbol="BTCUSDT",
        side="BUY",
        requested_notional=5.0,
        price=50000.0,
        nav=1000.0,
        open_qty=0.0,
        now=0.0,
        cfg=cfg,
        state=st,
        current_gross_notional=0.0,
        tier_name="CORE",
        current_tier_gross_notional=79.0,
    )
    assert veto is True
    assert "tier_cap" in details.get("reasons", [])


def test_max_concurrent_blocks():
    st = RiskState()
    cfg = _base_cfg()
    veto, details = check_order(
        symbol="BTCUSDT",
        side="BUY",
        requested_notional=10.0,
        price=50000.0,
        nav=1000.0,
        open_qty=0.0,
        now=0.0,
        cfg=cfg,
        state=st,
        open_positions_count=3,
    )
    assert veto is True
    assert "max_concurrent" in details.get("reasons", [])
