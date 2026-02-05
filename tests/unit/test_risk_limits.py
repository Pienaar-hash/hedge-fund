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
    assert geometry.get("budget_saturated") is True
    assert geometry.get("overshoot_pct") is None
