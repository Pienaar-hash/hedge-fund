from typing import Any, Dict, List

from execution import risk_limits as risk_limits_module
from execution.risk_limits import RiskState, check_order


def _fresh_nav(monkeypatch, nav_value: float) -> None:
    nav_snapshot = {
        "age_s": 0.0,
        "sources_ok": True,
        "fresh": True,
        "nav_total": nav_value,
    }
    monkeypatch.setattr(
        risk_limits_module,
        "nav_health_snapshot",
        lambda threshold_s=None: dict(nav_snapshot),
    )
    monkeypatch.setattr(risk_limits_module, "get_nav_freshness_snapshot", lambda: (0.0, True))


def test_veto_geometry_survives_multi_reason(monkeypatch, mock_clean_drawdown_state) -> None:
    """
    Regression test: constraint_geometry must survive detail_payload reassignment
    even when symbol_cap is NOT the primary veto reason.
    
    This tests the SATURATED case (budget <= 0, overshoot_pct is None).
    
    Scenario:
    - NAV = 100, max_nav_pct = 0.01 (1%) → cap = 1 USDT
    - current_gross = 2 USDT → already over cap, available_budget <= 0
    - Primary veto is min_notional (5 < 10 min)
    - Secondary reason is symbol_cap
    
    The bug would wipe constraint_geometry when detail_payload was reassigned.
    """
    _fresh_nav(monkeypatch, 100.0)
    monkeypatch.setattr(risk_limits_module, "_nav_history_from_log", lambda limit=200: [])
    monkeypatch.setattr(risk_limits_module, "universe_by_symbol", lambda: {"BTCUSDT": {}})
    monkeypatch.setattr(risk_limits_module, "load_symbol_caps", lambda: {})
    monkeypatch.setattr(risk_limits_module, "get_confirmed_nav", lambda: {})

    events: List[Dict[str, Any]] = []

    def _capture_log_event(_logger, event_type: str, payload: Dict[str, Any]) -> None:
        if event_type == "risk_veto":
            events.append(payload)

    monkeypatch.setattr(risk_limits_module, "log_event", _capture_log_event)

    cfg = {
        "global": {
            "min_notional_usdt": 10.0,
            "nav_freshness_seconds": 10_000,
        },
        "per_symbol": {
            "BTCUSDT": {
                "min_notional": 10.0,
                "max_nav_pct": 0.01,
            }
        },
    }

    veto, detail = check_order(
        symbol="BTCUSDT",
        side="BUY",
        requested_notional=5.0,
        price=1.0,
        nav=100.0,
        open_qty=0.0,
        now=0.0,
        cfg=cfg,
        state=RiskState(),
        current_gross_notional=2.0,
        lev=1.0,
        open_positions_count=0,
    )

    assert veto is True
    reasons = detail.get("reasons") or []
    assert reasons[0] == "min_notional"
    assert "symbol_cap" in reasons
    assert len(events) == 1

    veto_detail = events[0].get("veto_detail", {})
    geometry = veto_detail.get("constraint_geometry")
    assert geometry is not None
    # Budget saturated: current_gross (2.0) > cap (1.0), so available_budget <= 0
    assert geometry.get("budget_saturated") is True
    # overshoot_pct is None when saturated (division by zero guard)
    assert geometry.get("overshoot_pct") is None


def test_veto_geometry_survives_non_primary_reason_unsaturated(
    monkeypatch, mock_clean_drawdown_state
) -> None:
    """
    Regression test: constraint_geometry must survive detail_payload reassignment
    even when symbol_cap is NOT the primary veto reason.
    
    This tests the NON-SATURATED case where overshoot_pct is computed.
    
    Scenario:
    - NAV = 1000, max_nav_pct = 0.10 (10%) → cap = 100 USDT
    - current_gross = 50 USDT → available_budget = 50 USDT
    - requested_notional = 80 USDT → excess = 30 USDT
    - overshoot_pct = 30/50 * 100 = 60%
    - But min_notional = 100 USDT, so primary veto is "min_notional"
    
    The bug would wipe constraint_geometry when detail_payload was reassigned
    for a non-symbol_cap primary reason.
    """
    _fresh_nav(monkeypatch, 1000.0)
    monkeypatch.setattr(risk_limits_module, "_nav_history_from_log", lambda limit=200: [])
    monkeypatch.setattr(risk_limits_module, "universe_by_symbol", lambda: {"ETHUSDT": {}})
    monkeypatch.setattr(risk_limits_module, "load_symbol_caps", lambda: {})
    monkeypatch.setattr(risk_limits_module, "get_confirmed_nav", lambda: {})

    events: List[Dict[str, Any]] = []

    def _capture_log_event(_logger, event_type: str, payload: Dict[str, Any]) -> None:
        if event_type == "risk_veto":
            events.append(payload)

    monkeypatch.setattr(risk_limits_module, "log_event", _capture_log_event)

    cfg = {
        "global": {
            "min_notional_usdt": 100.0,  # High floor forces min_notional veto
            "nav_freshness_seconds": 10_000,
        },
        "per_symbol": {
            "ETHUSDT": {
                "min_notional": 100.0,
                "max_nav_pct": 0.10,  # 10% of 1000 = 100 USDT cap
            }
        },
    }

    veto, detail = check_order(
        symbol="ETHUSDT",
        side="BUY",
        requested_notional=80.0,  # Below min_notional (100), but exceeds cap budget
        price=1.0,
        nav=1000.0,
        open_qty=0.0,
        now=0.0,
        cfg=cfg,
        state=RiskState(),
        current_gross_notional=50.0,  # Already at 50, budget = 100-50 = 50
        lev=1.0,
        open_positions_count=0,
    )

    assert veto is True
    reasons = detail.get("reasons") or []
    # Primary reason should be min_notional, NOT symbol_cap
    assert reasons[0] == "min_notional", f"Expected min_notional as primary, got {reasons}"
    assert "symbol_cap" in reasons, "symbol_cap should be in reasons list"
    assert len(events) == 1

    veto_detail = events[0].get("veto_detail", {})
    geometry = veto_detail.get("constraint_geometry")
    
    # THE CRITICAL ASSERTION: geometry must survive even with non-primary reason
    assert geometry is not None, (
        "constraint_geometry was wiped! This is the regression bug from Phase A.3"
    )
    
    # Verify geometry semantics
    assert geometry.get("budget_saturated") is False, "Budget should NOT be saturated"
    assert geometry.get("available_budget") == 50.0, "available_budget = cap - current = 100 - 50"
    assert geometry.get("excess_notional") == 30.0, "excess = requested - available = 80 - 50"
    
    # overshoot_pct is computed when NOT saturated
    overshoot = geometry.get("overshoot_pct")
    assert overshoot is not None, "overshoot_pct should be computed when not saturated"
    assert abs(overshoot - 60.0) < 0.01, f"Expected 60%, got {overshoot}%"
