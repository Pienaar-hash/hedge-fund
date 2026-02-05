from __future__ import annotations

from typing import Any, Dict, List

import execution.executor_live as executor
from execution import risk_limits as risk_limits_module
from execution.risk_limits import RiskState


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


def test_source_head_parity_v6_vs_fallback(monkeypatch, mock_clean_drawdown_state) -> None:
    _fresh_nav(monkeypatch, 100.0)
    monkeypatch.setattr(risk_limits_module, "universe_by_symbol", lambda: {"BTCUSDT": {}})
    monkeypatch.setattr(risk_limits_module, "load_symbol_caps", lambda: {})
    monkeypatch.setattr(risk_limits_module, "get_confirmed_nav", lambda: {})
    monkeypatch.setattr(risk_limits_module, "_nav_history_from_log", lambda limit=200: [])

    events: List[Dict[str, Any]] = []

    def _capture_log_event(_logger, event_type: str, payload: Dict[str, Any]) -> None:
        if event_type == "risk_veto":
            events.append(payload)

    monkeypatch.setattr(risk_limits_module, "log_event", _capture_log_event)

    cfg = {
        "global": {
            "min_notional_usdt": 0.0,
            "nav_freshness_seconds": 10_000,
        },
        "per_symbol": {
            "BTCUSDT": {
                "min_notional": 0.0,
                "max_nav_pct": 0.01,
            }
        },
    }
    intent = {
        "price": 1.0,
        "metadata": {"strategy": "vol_target"},
    }

    def _run_path(use_v6: bool) -> str:
        events.clear()
        monkeypatch.setattr(executor, "RISK_ENGINE_V6_ENABLED", use_v6)
        monkeypatch.setattr(executor, "_load_pairs_cfg", lambda: {"universe": [{"symbol": "BTCUSDT"}]})
        executor._RISK_CFG = cfg
        executor._RISK_STATE = RiskState()
        executor._RISK_ENGINE_V6 = None
        executor._RISK_ENGINE_V6_CFG_DIGEST = None
        veto, _detail = executor._evaluate_order_risk(
            symbol="BTCUSDT",
            side="BUY",
            gross_target=5.0,
            nav=100.0,
            sym_open_qty=0.0,
            current_gross=0.0,
            current_symbol_gross=0.0,
            open_positions_count=0,
            tier_name=None,
            current_tier_gross=0.0,
            lev=1.0,
            reduce_only=False,
            intent=intent,
        )
        assert veto is True
        assert len(events) == 1
        return str(events[0].get("source_head"))

    source_v6 = _run_path(True)
    source_fallback = _run_path(False)
    assert source_v6 == source_fallback == "vol_target"
