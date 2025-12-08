from execution.executor_live import compute_final_gross_for_test


def test_executor_respects_screener_gross_and_qty():
    intent = {
        "symbol": "BTCUSDT",
        "gross_usd": 150.0,
        "qty": 0.005,
        "price": 30000.0,
        "min_notional": 5.0,
        "per_trade_nav_pct": 0.0,
    }
    cfg = {"min_notional_usd": 5.0, "per_symbol_limits": {}}
    gross = compute_final_gross_for_test(intent, nav_usd=10_000.0, size_risk_cfg=cfg)
    assert gross == 150.0


def test_executor_uses_nav_pct_only_when_intent_lacks_gross():
    intent = {
        "symbol": "ETHUSDT",
        "gross_usd": 0.0,
        "min_notional": 5.0,
        "per_trade_nav_pct": 0.02,  # 2%
    }
    cfg = {"min_notional_usd": 5.0, "per_symbol_limits": {}}
    gross = compute_final_gross_for_test(intent, nav_usd=1_000.0, size_risk_cfg=cfg)
    assert gross == 20.0
