"""Investor-defence contract: min-notional planning must never become a bypass.

Given:
    intended_notional < min_notional
    upsized_notional (== min_notional) would exceed max_nav_pct * nav_usd

Then:
    risk_limits.check_order() must veto with veto == True
    min_notional_action in veto detail == ABSTAIN_MIN_NOTIONAL
    No order attempt may be emitted (caller receives veto=True and stops).
"""

from __future__ import annotations

from typing import Any, Dict, List

from execution import risk_limits as risk_limits_module
from execution.risk_limits import RiskState, check_order


def _patch_nav(monkeypatch, nav_value: float) -> None:
    snap = {
        "age_s": 0.0,
        "sources_ok": True,
        "fresh": True,
        "nav_total": nav_value,
    }
    monkeypatch.setattr(risk_limits_module, "nav_health_snapshot", lambda threshold_s=None: dict(snap))
    monkeypatch.setattr(risk_limits_module, "get_nav_freshness_snapshot", lambda: (0.0, True))
    monkeypatch.setattr(risk_limits_module, "_nav_history_from_log", lambda limit=200: [])


def test_upsize_exceeds_nav_cap_results_in_abstain(monkeypatch, mock_clean_drawdown_state) -> None:
    """
    Primary investor-defence test.

    Scenario (SOLUSDT)
    ------------------
    NAV = 1 000 USDT
    max_nav_pct = 4 % → implied cap = 40 USDT
    min_notional = 50 USDT  (exchange floor)
    intended_notional = 20 USDT  (< min, triggers planner)
    upsized_notional = 50 USDT  (> 40 USDT cap)

    Expected outcome
    ----------------
    Planner action = ABSTAIN_MIN_NOTIONAL ("upsize_breaches_nav_cap")
    risk_limits.check_order() → veto = True
    detail["min_notional_action"] == "ABSTAIN_MIN_NOTIONAL"
    detail["reasons"][0] == "min_notional"
    """
    nav = 1000.0
    _patch_nav(monkeypatch, nav)
    monkeypatch.setattr(risk_limits_module, "universe_by_symbol", lambda: {"SOLUSDT": {}})
    monkeypatch.setattr(risk_limits_module, "load_symbol_caps", lambda: {})
    monkeypatch.setattr(risk_limits_module, "get_confirmed_nav", lambda: {})

    emitted_vetoes: List[Dict[str, Any]] = []

    def _capture_log_event(_logger, event_type: str, payload: Dict[str, Any]) -> None:
        if event_type == "risk_veto":
            emitted_vetoes.append(payload)

    monkeypatch.setattr(risk_limits_module, "log_event", _capture_log_event)

    cfg = {
        "global": {
            "min_notional_usdt": 50.0,
            "max_trade_nav_pct": 0.04,
            "nav_freshness_seconds": 100_000,
        },
        "per_symbol": {
            "SOLUSDT": {
                "min_notional": 50.0,
                "max_nav_pct": 0.04,  # 4 % of 1 000 = 40 USDT cap → below min_notional of 50
            },
        },
    }

    veto, detail = check_order(
        symbol="SOLUSDT",
        side="BUY",
        requested_notional=20.0,   # intended < min_notional
        price=150.0,
        nav=nav,
        open_qty=0.0,
        now=0.0,
        cfg=cfg,
        state=RiskState(),
        current_gross_notional=0.0,
        lev=1.0,
    )

    # Must be vetoed — no order may proceed
    assert veto is True, "Expected veto=True when upsize breaches cap"

    reasons = detail.get("reasons", [])
    assert "min_notional" in reasons, f"Expected 'min_notional' in reasons, got {reasons}"

    action = detail.get("min_notional_action")
    assert action == "ABSTAIN_MIN_NOTIONAL", (
        f"Expected min_notional_action=ABSTAIN_MIN_NOTIONAL, got {action!r}"
    )

    # Exactly one veto event must have been logged; no order attempt beyond that
    assert len(emitted_vetoes) == 1, f"Expected exactly 1 veto log entry, got {len(emitted_vetoes)}"
    veto_detail = emitted_vetoes[0].get("veto_detail", {})
    assert veto_detail.get("min_notional_action") == "ABSTAIN_MIN_NOTIONAL"


def test_normal_below_min_notional_vetos_when_cap_allows_upsize(monkeypatch, mock_clean_drawdown_state) -> None:
    """
    Counterpart to the main bypass test.

    When the upsized notional fits within the NAV cap,
    the planner returns UPSIZE_TO_MIN_NOTIONAL and the veto is still
    emitted (because the *original* requested_notional < min_notional —
    risk_limits blocks it and lets the caller decide whether to re-submit
    at the adjusted size).

    This confirms the veto path is intact and the planner does not
    silently alter the notional being evaluated by check_order.
    """
    nav = 10_000.0
    _patch_nav(monkeypatch, nav)
    monkeypatch.setattr(risk_limits_module, "universe_by_symbol", lambda: {"ETHUSDT": {}})
    monkeypatch.setattr(risk_limits_module, "load_symbol_caps", lambda: {})
    monkeypatch.setattr(risk_limits_module, "get_confirmed_nav", lambda: {})

    emitted_vetoes: List[Dict[str, Any]] = []

    def _capture(logger, event_type: str, payload: Dict[str, Any]) -> None:
        if event_type == "risk_veto":
            emitted_vetoes.append(payload)

    monkeypatch.setattr(risk_limits_module, "log_event", _capture)

    cfg = {
        "global": {
            "min_notional_usdt": 50.0,
            "max_trade_nav_pct": 0.10,   # 10 % of 10 000 = 1 000 USDT → upsize of 50 fits
            "nav_freshness_seconds": 100_000,
        },
        "per_symbol": {
            "ETHUSDT": {
                "min_notional": 50.0,
                "max_nav_pct": 0.10,
            },
        },
    }

    veto, detail = check_order(
        symbol="ETHUSDT",
        side="BUY",
        requested_notional=20.0,   # intended < min_notional = 50
        price=3000.0,
        nav=nav,
        open_qty=0.0,
        now=0.0,
        cfg=cfg,
        state=RiskState(),
        current_gross_notional=0.0,
        lev=1.0,
    )

    # Still vetoed because requested_notional < min_notional;
    # planner just classifies the action.
    assert veto is True
    action = detail.get("min_notional_action")
    assert action == "UPSIZE_TO_MIN_NOTIONAL", (
        f"Expected UPSIZE_TO_MIN_NOTIONAL when upsize fits cap, got {action!r}"
    )
    assert len(emitted_vetoes) == 1
