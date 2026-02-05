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

def test_source_head_parity_alternate_locations(monkeypatch, mock_clean_drawdown_state) -> None:
    """
    Test source_head attribution parity when strategy is in different locations:
    1. intent.strategy_id (direct)
    2. intent.strategy (direct) 
    3. intent.metadata.strategy (nested)
    4. intent.metadata.head (nested alternate name)
    
    Both v6 and fallback paths must extract the same value.
    """
    _fresh_nav(monkeypatch, 100.0)
    monkeypatch.setattr(risk_limits_module, "universe_by_symbol", lambda: {"ETHUSDT": {}})
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
            "ETHUSDT": {
                "min_notional": 0.0,
                "max_nav_pct": 0.01,  # Forces symbol_cap veto
            }
        },
    }

    def _run_both_paths(intent: Dict[str, Any], expected_head: str) -> None:
        """Run v6 and fallback paths, verify source_head parity."""
        for use_v6 in [True, False]:
            events.clear()
            monkeypatch.setattr(executor, "RISK_ENGINE_V6_ENABLED", use_v6)
            monkeypatch.setattr(executor, "_load_pairs_cfg", lambda: {"universe": [{"symbol": "ETHUSDT"}]})
            executor._RISK_CFG = cfg
            executor._RISK_STATE = RiskState()
            executor._RISK_ENGINE_V6 = None
            executor._RISK_ENGINE_V6_CFG_DIGEST = None
            
            veto, _detail = executor._evaluate_order_risk(
                symbol="ETHUSDT",
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
            actual_head = events[0].get("source_head")
            path_name = "v6" if use_v6 else "fallback"
            assert actual_head == expected_head, (
                f"{path_name} path: expected source_head='{expected_head}', got '{actual_head}'"
            )

    # Case 1: strategy in intent.metadata.strategy (most common)
    _run_both_paths(
        intent={"price": 1.0, "metadata": {"strategy": "TREND"}},
        expected_head="TREND",
    )

    # Case 2: strategy_id at top level
    _run_both_paths(
        intent={"price": 1.0, "strategy_id": "MEAN_REVERT"},
        expected_head="MEAN_REVERT",
    )

    # Case 3: strategy at top level (alternate field name)
    _run_both_paths(
        intent={"price": 1.0, "strategy": "VOL_HARVEST"},
        expected_head="VOL_HARVEST",
    )

    # Case 4: metadata.head (alternate nested field)
    _run_both_paths(
        intent={"price": 1.0, "metadata": {"head": "EMERGENT_ALPHA"}},
        expected_head="EMERGENT_ALPHA",
    )


def test_source_head_parity_with_constraint_geometry(monkeypatch, mock_clean_drawdown_state) -> None:
    """
    Test that both paths emit identical source_head AND preserve constraint_geometry.
    This is the full integration test combining attribution + geometry preservation.
    """
    _fresh_nav(monkeypatch, 1000.0)
    monkeypatch.setattr(risk_limits_module, "universe_by_symbol", lambda: {"SOLUSDT": {}})
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
            "min_notional_usdt": 100.0,  # Forces min_notional as primary reason
            "nav_freshness_seconds": 10_000,
        },
        "per_symbol": {
            "SOLUSDT": {
                "min_notional": 100.0,
                "max_nav_pct": 0.10,  # 10% of 1000 = 100 cap
            }
        },
    }
    
    intent = {
        "price": 1.0,
        "metadata": {"strategy": "vol_target"},
    }

    results: Dict[str, Dict[str, Any]] = {}

    for use_v6 in [True, False]:
        events.clear()
        monkeypatch.setattr(executor, "RISK_ENGINE_V6_ENABLED", use_v6)
        monkeypatch.setattr(executor, "_load_pairs_cfg", lambda: {"universe": [{"symbol": "SOLUSDT"}]})
        executor._RISK_CFG = cfg
        executor._RISK_STATE = RiskState()
        executor._RISK_ENGINE_V6 = None
        executor._RISK_ENGINE_V6_CFG_DIGEST = None
        
        veto, _detail = executor._evaluate_order_risk(
            symbol="SOLUSDT",
            side="BUY",
            gross_target=80.0,  # Below min_notional, but would exceed cap budget
            nav=1000.0,
            sym_open_qty=0.0,
            current_gross=0.0,
            current_symbol_gross=50.0,  # Already 50, leaves 50 budget
            open_positions_count=0,
            tier_name=None,
            current_tier_gross=0.0,
            lev=1.0,
            reduce_only=False,
            intent=intent,
        )
        assert veto is True
        assert len(events) == 1
        
        path_name = "v6" if use_v6 else "fallback"
        results[path_name] = {
            "source_head": events[0].get("source_head"),
            "veto_detail": events[0].get("veto_detail", {}),
        }

    # Verify source_head parity
    assert results["v6"]["source_head"] == results["fallback"]["source_head"] == "vol_target"
    
    # Verify both paths preserve constraint_geometry
    for path_name in ["v6", "fallback"]:
        geometry = results[path_name]["veto_detail"].get("constraint_geometry")
        assert geometry is not None, f"{path_name} path: constraint_geometry was wiped!"
        assert geometry.get("budget_saturated") is False
        assert geometry.get("available_budget") == 50.0