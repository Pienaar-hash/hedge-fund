from __future__ import annotations

from dashboard.state_v7 import (
    compute_dd_regime_badge,
    compute_risk_mode_badge,
    compute_router_quality_badge,
    compute_vol_regime_badge,
    get_regime_badges,
)


def test_vol_regime_badge_from_state():
    assert compute_vol_regime_badge({"current_regime": "high"}) == "VOL_HIGH"
    assert compute_vol_regime_badge({}) == "VOL_NORMAL"


def test_dd_regime_badge_mapping():
    assert compute_dd_regime_badge({"dd_state": {"state": "recovery"}}) == "DD_RECOVERY"
    assert compute_dd_regime_badge({"dd_state": "drawdown"}) == "DD_DRAWDOWN"


def test_router_quality_badge_thresholds():
    rh = {"global": {"quality_score": 0.95}}
    assert compute_router_quality_badge(rh, high_threshold=0.9) == "ROUTER_STRONG"
    rh_low = {"global": {"quality_score": 0.4}}
    assert compute_router_quality_badge(rh_low, low_threshold=0.5) == "ROUTER_WEAK"


def test_risk_mode_badge_with_breach():
    snap = {"risk_mode": "defensive", "anomalies": {"var_limit_breach": True}}
    assert compute_risk_mode_badge(snap) == "RISK_BREACH"
    assert compute_risk_mode_badge({"risk_mode": "OK"}) == "RISK_NORMAL"


def test_get_regime_badges_combines_inputs():
    badges = get_regime_badges(
        vol_state={"current_regime": "low"},
        risk_snapshot={"dd_state": "normal", "risk_mode": "CRISIS"},
        router_health={"global": {"quality_score": 0.8}},
    )
    assert badges["vol"] == "VOL_LOW"
    assert badges["risk"] in {"RISK_CRITICAL", "RISK_BREACH"}
