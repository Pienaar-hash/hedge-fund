from execution.min_notional_planner import MinNotionalAction, plan_min_notional_action


def test_btcusdt_pass_when_above_min_notional() -> None:
    plan = plan_min_notional_action(
        symbol="BTCUSDT",
        intended_qty=0.002,
        mark_price=65000.0,
        intended_notional=130.0,
        min_notional=100.0,
        nav_usd=10000.0,
        max_nav_pct=0.05,
        leverage=2.0,
        fee_rate=0.0004,
    )
    assert plan.action == MinNotionalAction.PASS
    assert plan.adjusted_notional == 130.0


def test_ethusdt_upsize_to_min_notional() -> None:
    plan = plan_min_notional_action(
        symbol="ETHUSDT",
        intended_qty=0.01,
        mark_price=3000.0,
        intended_notional=30.0,
        min_notional=50.0,
        nav_usd=10000.0,
        max_nav_pct=0.05,
        leverage=3.0,
        fee_rate=0.0004,
    )
    assert plan.action == MinNotionalAction.UPSIZE_TO_MIN_NOTIONAL
    assert plan.min_notional_required == 50.0
    assert plan.adjusted_notional == 50.0


def test_solusdt_abstain_when_upsize_breaches_cap() -> None:
    plan = plan_min_notional_action(
        symbol="SOLUSDT",
        intended_qty=0.2,
        mark_price=150.0,
        intended_notional=30.0,
        min_notional=50.0,
        nav_usd=1000.0,
        max_nav_pct=0.04,
        leverage=1.0,
        fee_rate=0.0004,
    )
    assert plan.action == MinNotionalAction.ABSTAIN_MIN_NOTIONAL
    assert plan.reason == "upsize_breaches_nav_cap"


def test_solusdt_reject_when_upsize_is_uneconomic() -> None:
    plan = plan_min_notional_action(
        symbol="SOLUSDT",
        intended_qty=0.01,
        mark_price=140.0,
        intended_notional=1.4,
        min_notional=10.0,
        nav_usd=10000.0,
        max_nav_pct=0.10,
        leverage=1.0,
        fee_rate=0.002,
    )
    assert plan.action == MinNotionalAction.REJECT_MIN_NOTIONAL_UNECONOMIC
    assert plan.reason == "upsize_uneconomic"
