"""
Tests for v7 drawdown/daily-loss normalization fix.

Ensures that observed metrics (percent-style, e.g., 1.141 meaning 1.141%)
are correctly compared to config caps (fractional, e.g., 0.30 meaning 30%).
"""

import pytest

from execution import risk_limits as risk_limits_module
from execution.risk_limits import RiskState, check_order, _normalize_observed_pct


def _base_cfg_fractional():
    """Config with fractional caps as used in production (after normalize_percentage)."""
    return {
        "global": {
            "whitelist": ["BTCUSDT", "ETHUSDT"],
            "min_notional_usdt": 25.0,
            "daily_loss_limit_pct": 0.10,  # 10% as fraction
            "max_nav_drawdown_pct": 0.30,  # 30% as fraction
            "max_trade_nav_pct": 0.20,
            "trade_equity_nav_pct": 0.15,
            "nav_freshness_seconds": 1_000_000,
        },
        "per_symbol": {
            "BTCUSDT": {
                "min_notional": 25.0,
                "max_order_notional": 50_000.0,
                "max_nav_pct": 0.25,
                "max_leverage": 4,
            },
            "ETHUSDT": {
                "min_notional": 25.0,
                "max_order_notional": 25_000.0,
                "max_nav_pct": 0.20,
                "max_leverage": 4,
            },
        },
    }


def _fresh_nav(monkeypatch, nav_value: float):
    """Mock fresh NAV state."""
    nav_snapshot = {"age_s": 0.0, "sources_ok": True, "fresh": True, "nav_total": nav_value}
    monkeypatch.setattr(risk_limits_module, "nav_health_snapshot", lambda threshold_s=None: dict(nav_snapshot))
    monkeypatch.setattr(risk_limits_module, "get_nav_freshness_snapshot", lambda: (0.0, True))


def _mock_drawdown(monkeypatch, dd_pct: float, daily_loss_pct: float | None = None):
    """Mock drawdown snapshot with percent-style values."""
    def mock_snapshot(cfg=None):
        return {
            "dd_pct": dd_pct,
            "dd_abs": 0.0,
            "peak": 1000.0,
            "nav": 1000.0 * (1 - dd_pct / 100.0) if dd_pct > 0 else 1000.0,
            "usable": True,
            "stale_flags": {},
            "nav_health": {"age_s": 0.0, "sources_ok": True},
            "peak_state": {},
            "drawdown": {"pct": dd_pct, "peak_nav": 1000.0, "nav": 1000.0},
            "daily_loss": {"pct": daily_loss_pct} if daily_loss_pct is not None else {},
        }
    monkeypatch.setattr(risk_limits_module, "_drawdown_snapshot", mock_snapshot)


class TestNormalizeObservedPct:
    """Unit tests for _normalize_observed_pct helper."""

    def test_zero_returns_zero(self):
        assert _normalize_observed_pct(0.0) == 0.0

    def test_negative_returns_zero(self):
        assert _normalize_observed_pct(-5.0) == 0.0

    def test_percent_style_normalizes_to_fraction(self):
        # 1.141% (percent-style) -> 0.01141 (fraction)
        result = _normalize_observed_pct(1.141)
        assert pytest.approx(result, rel=1e-6) == 0.01141

    def test_large_percent_normalizes_to_fraction(self):
        # 35% (percent-style) -> 0.35 (fraction)
        result = _normalize_observed_pct(35.0)
        assert pytest.approx(result, rel=1e-6) == 0.35

    def test_always_divides_by_100(self):
        """v7: _normalize_observed_pct always divides by 100 (percent-style to fractional)."""
        # 0.05 (percent-style, meaning 0.05%) -> 0.0005 (fractional)
        result = _normalize_observed_pct(0.05)
        assert pytest.approx(result, rel=1e-6) == 0.0005


class TestDrawdownGateNormalization:
    """Tests for nav_drawdown_limit gate with normalized comparisons."""

    def test_small_drawdown_does_not_trip_nav_drawdown_limit(self, monkeypatch):
        """
        Given max_nav_drawdown_pct = 0.30 (30%) and observed drawdown of 1.141%
        (stored as 1.141 in percent-style), the gate should NOT veto.
        """
        _fresh_nav(monkeypatch, 1000.0)
        _mock_drawdown(monkeypatch, dd_pct=1.141)  # 1.141% drawdown (percent-style)

        st = RiskState()
        cfg = _base_cfg_fractional()
        cfg["global"]["max_nav_drawdown_pct"] = 0.30  # 30% cap as fraction

        veto, details = check_order(
            symbol="BTCUSDT",
            side="BUY",
            requested_notional=25.0,
            price=50000.0,
            nav=1000.0,
            open_qty=0.0,
            now=0.0,
            cfg=cfg,
            state=st,
            current_gross_notional=0.0,
        )

        reasons = details.get("reasons", [])
        assert "nav_drawdown_limit" not in reasons, f"Small drawdown should not veto: {details}"

    def test_large_drawdown_trips_nav_drawdown_limit(self, monkeypatch):
        """
        Given max_nav_drawdown_pct = 0.30 (30%) and observed drawdown of 35%
        (stored as 35.0 in percent-style), the gate SHOULD veto.
        """
        _fresh_nav(monkeypatch, 1000.0)
        _mock_drawdown(monkeypatch, dd_pct=35.0)  # 35% drawdown (percent-style)

        st = RiskState()
        cfg = _base_cfg_fractional()
        cfg["global"]["max_nav_drawdown_pct"] = 0.30  # 30% cap as fraction

        veto, details = check_order(
            symbol="BTCUSDT",
            side="BUY",
            requested_notional=25.0,
            price=50000.0,
            nav=1000.0,
            open_qty=0.0,
            now=0.0,
            cfg=cfg,
            state=st,
            current_gross_notional=0.0,
        )

        reasons = details.get("reasons", [])
        assert "nav_drawdown_limit" in reasons, f"Large drawdown should veto: {details}"

        # Verify normalized fractions are logged in observations
        observations = details.get("observations", {})
        drawdown_guard = observations.get("drawdown_guard", {})
        assert "observed_dd_frac" in drawdown_guard, f"Missing drawdown_guard in observations: {details}"
        assert "cap_dd_frac" in drawdown_guard
        assert pytest.approx(drawdown_guard["observed_dd_frac"], rel=1e-2) == 0.35
        assert pytest.approx(drawdown_guard["cap_dd_frac"], rel=1e-6) == 0.30

    def test_drawdown_exactly_at_limit_trips_veto(self, monkeypatch):
        """
        Given max_nav_drawdown_pct = 0.30 (30%) and observed drawdown of exactly 30%
        (stored as 30.0 in percent-style), the gate SHOULD veto.
        """
        _fresh_nav(monkeypatch, 1000.0)
        _mock_drawdown(monkeypatch, dd_pct=30.0)  # Exactly 30% drawdown

        st = RiskState()
        cfg = _base_cfg_fractional()
        cfg["global"]["max_nav_drawdown_pct"] = 0.30  # 30% cap

        veto, details = check_order(
            symbol="BTCUSDT",
            side="BUY",
            requested_notional=25.0,
            price=50000.0,
            nav=1000.0,
            open_qty=0.0,
            now=0.0,
            cfg=cfg,
            state=st,
            current_gross_notional=0.0,
        )

        reasons = details.get("reasons", [])
        assert "nav_drawdown_limit" in reasons


class TestDailyLossGateNormalization:
    """Tests for day_loss_limit gate with normalized comparisons."""

    def test_small_daily_loss_does_not_trip_day_loss_limit(self, monkeypatch):
        """
        Given daily_loss_limit_pct = 0.10 (10%) and observed daily loss of 1.5%
        (stored as 1.5 in daily_pnl_pct), the gate should NOT veto.
        """
        _fresh_nav(monkeypatch, 1000.0)
        _mock_drawdown(monkeypatch, dd_pct=1.5, daily_loss_pct=1.5)

        st = RiskState()
        st.daily_pnl_pct = -1.5  # 1.5% daily loss (percent-style)
        cfg = _base_cfg_fractional()
        cfg["global"]["daily_loss_limit_pct"] = 0.10  # 10% cap as fraction

        veto, details = check_order(
            symbol="BTCUSDT",
            side="BUY",
            requested_notional=25.0,
            price=50000.0,
            nav=1000.0,
            open_qty=0.0,
            now=0.0,
            cfg=cfg,
            state=st,
            current_gross_notional=0.0,
        )

        reasons = details.get("reasons", [])
        assert "day_loss_limit" not in reasons, f"Small daily loss should not veto: {details}"

    def test_large_daily_loss_trips_day_loss_limit(self, monkeypatch):
        """
        Given daily_loss_limit_pct = 0.10 (10%) and observed daily loss of 12%
        (stored as 12.0 in daily_pnl_pct), the gate SHOULD veto.
        """
        _fresh_nav(monkeypatch, 1000.0)
        _mock_drawdown(monkeypatch, dd_pct=12.0, daily_loss_pct=12.0)

        st = RiskState()
        st.daily_pnl_pct = -12.0  # 12% daily loss (percent-style)
        cfg = _base_cfg_fractional()
        cfg["global"]["daily_loss_limit_pct"] = 0.10  # 10% cap as fraction

        veto, details = check_order(
            symbol="BTCUSDT",
            side="BUY",
            requested_notional=25.0,
            price=50000.0,
            nav=1000.0,
            open_qty=0.0,
            now=0.0,
            cfg=cfg,
            state=st,
            current_gross_notional=0.0,
        )

        reasons = details.get("reasons", [])
        assert "day_loss_limit" in reasons, f"Large daily loss should veto: {details}"

        # Verify normalized fractions are logged in observations
        observations = details.get("observations", {})
        daily_loss_guard = observations.get("daily_loss_guard", {})
        assert "observed_day_loss_frac" in daily_loss_guard, f"Missing daily_loss_guard in observations: {details}"
        assert "cap_day_loss_frac" in daily_loss_guard
        assert pytest.approx(daily_loss_guard["observed_day_loss_frac"], rel=1e-2) == 0.12
        assert pytest.approx(daily_loss_guard["cap_day_loss_frac"], rel=1e-6) == 0.10

    def test_daily_loss_exactly_at_limit_trips_veto(self, monkeypatch):
        """
        Given daily_loss_limit_pct = 0.10 (10%) and observed daily loss of exactly 10%
        (stored as 10.0 in daily_pnl_pct), the gate SHOULD veto.
        """
        _fresh_nav(monkeypatch, 1000.0)
        _mock_drawdown(monkeypatch, dd_pct=10.0, daily_loss_pct=10.0)

        st = RiskState()
        st.daily_pnl_pct = -10.0  # Exactly 10% daily loss
        cfg = _base_cfg_fractional()
        cfg["global"]["daily_loss_limit_pct"] = 0.10  # 10% cap

        veto, details = check_order(
            symbol="BTCUSDT",
            side="BUY",
            requested_notional=25.0,
            price=50000.0,
            nav=1000.0,
            open_qty=0.0,
            now=0.0,
            cfg=cfg,
            state=st,
            current_gross_notional=0.0,
        )

        reasons = details.get("reasons", [])
        assert "day_loss_limit" in reasons


class TestBackwardsCompatibility:
    """Ensure existing percent-style configs still work correctly."""

    def test_percent_style_config_also_works(self, monkeypatch):
        """
        If config has daily_loss_limit_pct = 10.0 (percent-style, meaning 10%),
        and observed daily loss is 12% (12.0 percent-style), it should veto.

        Both values get normalized: 10.0 -> 0.10, 12.0 -> 0.12, and 0.12 >= 0.10.
        """
        _fresh_nav(monkeypatch, 1000.0)
        _mock_drawdown(monkeypatch, dd_pct=12.0, daily_loss_pct=12.0)

        st = RiskState()
        st.daily_pnl_pct = -12.0
        cfg = _base_cfg_fractional()
        cfg["global"]["daily_loss_limit_pct"] = 10.0  # Percent-style cap

        veto, details = check_order(
            symbol="BTCUSDT",
            side="BUY",
            requested_notional=25.0,
            price=50000.0,
            nav=1000.0,
            open_qty=0.0,
            now=0.0,
            cfg=cfg,
            state=st,
            current_gross_notional=0.0,
        )

        reasons = details.get("reasons", [])
        assert "day_loss_limit" in reasons
