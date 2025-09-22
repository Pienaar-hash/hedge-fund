from execution.risk_limits import RiskGate

CFG = {
    "sizing": {
        "capital_per_trade_usdt": 10.0,
        "default_leverage": 10,
        "per_symbol_leverage": {"BTC": 20},
        "max_gross_exposure_pct": 120,
        "max_symbol_exposure_pct": 35,
        "min_notional_usdt": 5.5,
    },
    "risk": {
        "daily_loss_limit_pct": 5,
        "cooldown_minutes_after_stop": 60,
        "max_trades_per_symbol_per_hour": 6,
    },
}


class FakeRisk(RiskGate):
    def _portfolio_nav(self) -> float:
        return 1000.0  # fixed nav for determinism

    def _gross_exposure_pct(self) -> float:
        return 0.0

    def _symbol_exposure_pct(self, symbol: str, portfolio=None) -> float:
        return 0.0

    def _daily_loss_pct(self) -> float:
        return 0.0


def test_allows_basic_trade_under_caps():
    r = FakeRisk(CFG)
    allowed, veto = r.allowed_gross_notional("BTCUSDT", gross_usd=200.0)  # 10 * 20
    assert allowed is True
    assert veto == ""


def test_blocks_symbol_cap_when_future_pct_exceeds():
    class R(FakeRisk):
        def _symbol_exposure_pct(self, symbol: str, portfolio=None) -> float:
            return 34.9

    r = R(CFG)
    allowed, veto = r.allowed_gross_notional("BTCUSDT", gross_usd=5.5)  # adds ~0.55%
    assert allowed is False
    assert veto in {"symbol_cap", "tier_cap"}


def test_blocks_portfolio_cap_when_future_pct_exceeds():
    class R(FakeRisk):
        def _gross_exposure_pct(self) -> float:
            return 119.7

    r = R(CFG)
    allowed, veto = r.allowed_gross_notional("ETHUSDT", gross_usd=5.5)
    assert allowed is False
    assert veto == "portfolio_cap"


def test_below_min_notional_blocks():
    r = FakeRisk(CFG)
    allowed, veto = r.allowed_gross_notional("BTCUSDT", gross_usd=1.0)
    assert allowed is False
    assert veto == "below_min_notional"
