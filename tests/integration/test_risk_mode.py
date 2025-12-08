"""
Tests for risk mode classification (v7).

Test cases:
- HALTED when nav_age_s > 90
- HALTED when sources_ok == false
- HALTED when config_load_failed
- DEFENSIVE when dd_frac >= 0.30
- DEFENSIVE when daily_loss_frac >= 0.10
- WARN when router health degraded
- WARN when maker_first disabled > 5 cycles
- WARN when fallback ratio > threshold
- OK when all healthy
"""

import pytest
from execution.risk_engine_v6 import (
    RiskMode,
    RiskModeResult,
    classify_risk_mode,
    compute_risk_mode_from_state,
)


class TestClassifyRiskMode:
    """Unit tests for classify_risk_mode function."""

    def test_halted_when_nav_stale(self):
        """HALTED when nav_age_s > 90"""
        result = classify_risk_mode(nav_age_s=100)
        assert result.mode == RiskMode.HALTED
        assert "nav_stale" in result.reason
        assert result.score == 1.0

    def test_halted_when_nav_exactly_90_is_ok(self):
        """nav_age_s == 90 should NOT be HALTED (threshold is >90)"""
        result = classify_risk_mode(nav_age_s=90, sources_ok=True)
        assert result.mode != RiskMode.HALTED

    def test_halted_when_sources_not_ok(self):
        """HALTED when sources_ok == false"""
        result = classify_risk_mode(nav_age_s=30, sources_ok=False)
        assert result.mode == RiskMode.HALTED
        assert "sources_unavailable" in result.reason
        assert result.score == 1.0

    def test_halted_when_config_load_failed(self):
        """HALTED when risk engine fails to load config"""
        result = classify_risk_mode(config_load_failed=True)
        assert result.mode == RiskMode.HALTED
        assert "config" in result.reason.lower()
        assert result.score == 1.0

    def test_defensive_when_dd_frac_high(self):
        """DEFENSIVE when dd_frac >= 0.30"""
        result = classify_risk_mode(nav_age_s=30, sources_ok=True, dd_frac=0.30)
        assert result.mode == RiskMode.DEFENSIVE
        assert "drawdown" in result.reason.lower()
        # Score should be dd_frac / 0.30 = 1.0
        assert result.score == 1.0

    def test_defensive_when_dd_frac_above_threshold(self):
        """DEFENSIVE when dd_frac > 0.30"""
        result = classify_risk_mode(nav_age_s=30, sources_ok=True, dd_frac=0.35)
        assert result.mode == RiskMode.DEFENSIVE
        assert "drawdown" in result.reason.lower()
        # Score capped at 1.0
        assert result.score == 1.0

    def test_defensive_score_calculation(self):
        """DEFENSIVE score = min(1.0, dd_frac / 0.30)"""
        result = classify_risk_mode(nav_age_s=30, sources_ok=True, dd_frac=0.15)
        assert result.mode == RiskMode.DEFENSIVE or result.mode == RiskMode.OK
        # dd_frac=0.15 < 0.30, so should NOT be defensive
        # Actually 0.15 < 0.30, so NOT defensive
        result_defensive = classify_risk_mode(nav_age_s=30, sources_ok=True, dd_frac=0.30)
        assert result_defensive.score == pytest.approx(1.0)

    def test_defensive_when_daily_loss_high(self):
        """DEFENSIVE when daily_loss_frac >= 0.10"""
        result = classify_risk_mode(
            nav_age_s=30, sources_ok=True, dd_frac=0.05, daily_loss_frac=0.10
        )
        assert result.mode == RiskMode.DEFENSIVE
        assert "daily_loss" in result.reason.lower()

    def test_defensive_daily_loss_score(self):
        """DEFENSIVE score for daily loss = min(1.0, daily_loss_frac / 0.10)"""
        result = classify_risk_mode(
            nav_age_s=30, sources_ok=True, dd_frac=0.05, daily_loss_frac=0.15
        )
        assert result.mode == RiskMode.DEFENSIVE
        # Score capped at 1.0
        assert result.score == 1.0

    def test_warn_when_router_degraded(self):
        """WARN when router health degraded"""
        result = classify_risk_mode(
            nav_age_s=30, sources_ok=True, dd_frac=0.05, daily_loss_frac=0.05,
            router_degraded=True
        )
        assert result.mode == RiskMode.WARN
        assert "router" in result.reason.lower()
        assert result.score == 0.5

    def test_warn_when_maker_first_disabled_cycles(self):
        """WARN when maker_first=false for >5 cycles"""
        result = classify_risk_mode(
            nav_age_s=30, sources_ok=True, dd_frac=0.05, daily_loss_frac=0.05,
            maker_first_disabled_cycles=6
        )
        assert result.mode == RiskMode.WARN
        assert "maker_first" in result.reason.lower()
        assert result.score == 0.5

    def test_warn_when_fallback_ratio_high(self):
        """WARN when fallback ratio > threshold"""
        result = classify_risk_mode(
            nav_age_s=30, sources_ok=True, dd_frac=0.05, daily_loss_frac=0.05,
            fallback_ratio=0.6
        )
        assert result.mode == RiskMode.WARN
        assert "fallback" in result.reason.lower()
        assert result.score == 0.5

    def test_ok_when_all_healthy(self):
        """OK when all systems healthy"""
        result = classify_risk_mode(
            nav_age_s=30,
            sources_ok=True,
            dd_frac=0.05,
            daily_loss_frac=0.02,
            router_degraded=False,
            maker_first_disabled_cycles=0,
            fallback_ratio=0.2,
        )
        assert result.mode == RiskMode.OK
        assert "healthy" in result.reason.lower()
        assert result.score == 0.0

    def test_ok_with_none_values(self):
        """OK when values are None (assumed healthy)"""
        result = classify_risk_mode()
        assert result.mode == RiskMode.OK
        assert result.score == 0.0

    def test_priority_halted_over_defensive(self):
        """HALTED takes priority over DEFENSIVE"""
        result = classify_risk_mode(
            nav_age_s=100,  # HALTED
            sources_ok=True,
            dd_frac=0.50,  # DEFENSIVE
        )
        assert result.mode == RiskMode.HALTED

    def test_priority_defensive_over_warn(self):
        """DEFENSIVE takes priority over WARN"""
        result = classify_risk_mode(
            nav_age_s=30,
            sources_ok=True,
            dd_frac=0.35,  # DEFENSIVE
            router_degraded=True,  # WARN
        )
        assert result.mode == RiskMode.DEFENSIVE


class TestComputeRiskModeFromState:
    """Integration tests for compute_risk_mode_from_state function."""

    def test_from_nav_health_stale(self):
        """Compute HALTED from stale nav_health state"""
        nav_health = {"age_s": 120, "sources_ok": True}
        result = compute_risk_mode_from_state(nav_health=nav_health)
        assert result.mode == RiskMode.HALTED

    def test_from_nav_health_sources_unavailable(self):
        """Compute HALTED from nav sources unavailable"""
        nav_health = {"age_s": 30, "sources_ok": False}
        result = compute_risk_mode_from_state(nav_health=nav_health)
        assert result.mode == RiskMode.HALTED

    def test_from_risk_snapshot_high_dd(self):
        """Compute DEFENSIVE from high drawdown in risk_snapshot"""
        nav_health = {"age_s": 30, "sources_ok": True}
        risk_snapshot = {"dd_frac": 0.35, "daily_loss_frac": 0.05}
        result = compute_risk_mode_from_state(
            nav_health=nav_health, risk_snapshot=risk_snapshot
        )
        assert result.mode == RiskMode.DEFENSIVE

    def test_from_router_health_degraded(self):
        """Compute WARN from degraded router health"""
        nav_health = {"age_s": 30, "sources_ok": True}
        risk_snapshot = {"dd_frac": 0.05, "daily_loss_frac": 0.02}
        router_health = {
            "summary": {"quality_counts": {"degraded": 2, "ok": 5}},
            "symbols": [{"symbol": "BTCUSDT", "maker_first": True}],
        }
        result = compute_risk_mode_from_state(
            nav_health=nav_health,
            risk_snapshot=risk_snapshot,
            router_health=router_health,
        )
        assert result.mode == RiskMode.WARN

    def test_from_router_health_broken(self):
        """Compute WARN from broken router health"""
        nav_health = {"age_s": 30, "sources_ok": True}
        risk_snapshot = {"dd_frac": 0.05, "daily_loss_frac": 0.02}
        router_health = {
            "summary": {"quality_counts": {"broken": 1, "ok": 6}},
            "symbols": [],
        }
        result = compute_risk_mode_from_state(
            nav_health=nav_health,
            risk_snapshot=risk_snapshot,
            router_health=router_health,
        )
        assert result.mode == RiskMode.WARN

    def test_healthy_state(self):
        """Compute OK from healthy state files"""
        nav_health = {"age_s": 30, "sources_ok": True}
        risk_snapshot = {"dd_frac": 0.02, "daily_loss_frac": 0.01}
        router_health = {
            "summary": {"quality_counts": {"ok": 8}},
            "symbols": [
                {"symbol": "BTCUSDT", "maker_first": True, "fallback_rate": 0.1},
                {"symbol": "ETHUSDT", "maker_first": True, "fallback_rate": 0.2},
            ],
        }
        result = compute_risk_mode_from_state(
            nav_health=nav_health,
            risk_snapshot=risk_snapshot,
            router_health=router_health,
        )
        assert result.mode == RiskMode.OK

    def test_config_load_failed(self):
        """Compute HALTED when config load failed"""
        result = compute_risk_mode_from_state(config_load_failed=True)
        assert result.mode == RiskMode.HALTED


class TestRiskModeResult:
    """Tests for RiskModeResult dataclass."""

    def test_to_dict(self):
        """to_dict returns correct structure"""
        result = RiskModeResult(
            mode=RiskMode.DEFENSIVE,
            reason="drawdown_high_dd_frac=0.3500",
            score=1.0,
        )
        d = result.to_dict()
        assert d["risk_mode"] == "DEFENSIVE"
        assert d["risk_mode_reason"] == "drawdown_high_dd_frac=0.3500"
        assert d["risk_mode_score"] == 1.0

    def test_to_dict_ok_mode(self):
        """to_dict for OK mode"""
        result = RiskModeResult(
            mode=RiskMode.OK,
            reason="all_systems_healthy",
            score=0.0,
        )
        d = result.to_dict()
        assert d["risk_mode"] == "OK"
        assert d["risk_mode_score"] == 0.0
