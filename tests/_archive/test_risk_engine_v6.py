from __future__ import annotations

import json
import time
from typing import Any, Dict

import pytest

from execution.risk_engine_v6 import OrderIntent, RiskDecision, RiskEngineV6
from execution.risk_limits import RiskState, check_order
import execution.signal_screener as screener
import execution.executor_live as executor


@pytest.fixture(autouse=True)
def _fresh_nav(monkeypatch):
    monkeypatch.setattr("execution.risk_limits.get_nav_freshness_snapshot", lambda: (0.0, True))
    monkeypatch.setattr("execution.risk_limits.enforce_nav_freshness_or_veto", lambda *_args, **_kwargs: True)
    monkeypatch.setattr("execution.risk_limits._emit_veto", lambda *args, **kwargs: None)


def _decision_pair(risk_cfg: Dict[str, Any], intent: OrderIntent, state_setup=None):
    state_a = RiskState()
    state_b = RiskState()
    if state_setup:
        state_setup(state_a)
        state_setup(state_b)
    veto, details = check_order(
        symbol=intent.symbol,
        side=intent.side,
        requested_notional=intent.quote_notional,
        price=intent.price,
        nav=intent.nav_usd,
        open_qty=intent.symbol_open_qty or 0.0,
        now=time.time(),
        cfg=risk_cfg,
        state=state_a,
        current_gross_notional=intent.current_gross_notional or 0.0,
        lev=intent.leverage,
        open_positions_count=intent.open_positions_count,
        tier_name=intent.tier_name,
        current_tier_gross_notional=intent.tier_gross_notional or 0.0,
    )
    engine = RiskEngineV6.from_configs(risk_cfg, {"universe": [{"symbol": intent.symbol}]})
    decision = engine.check_order(intent, state_b)
    return veto, details, decision


def test_risk_engine_matches_trade_cap():
    risk_cfg = {"global": {"max_trade_nav_pct": 5.0}}
    intent = OrderIntent(
        symbol="BTCUSDT",
        side="BUY",
        qty=1.0,
        quote_notional=120.0,
        nav_usd=1000.0,
        leverage=1.0,
    )
    veto, details, decision = _decision_pair(risk_cfg, intent)
    assert decision.allowed == (not veto)
    assert decision.reasons == details.get("reasons")


def test_risk_engine_matches_tier_cap():
    risk_cfg = {"global": {"tiers": {"CORE": {"per_symbol_nav_pct": 10.0}}}}
    intent = OrderIntent(
        symbol="ETHUSDT",
        side="BUY",
        qty=1.0,
        quote_notional=150.0,
        nav_usd=1000.0,
        tier_name="CORE",
        tier_gross_notional=950.0,
    )
    veto, details, decision = _decision_pair(risk_cfg, intent)
    assert decision.allowed == (not veto)
    assert decision.reasons == details.get("reasons")


def test_risk_engine_matches_daily_loss_limit():
    risk_cfg = {"global": {"daily_loss_limit_pct": 5.0}}

    def _state_setup(state: RiskState) -> None:
        state.daily_pnl_pct = -6.0

    intent = OrderIntent(
        symbol="SOLUSDT",
        side="SELL",
        qty=1.0,
        quote_notional=10.0,
        nav_usd=1000.0,
    )
    veto, details, decision = _decision_pair(risk_cfg, intent, state_setup=_state_setup)
    assert decision.allowed == (not veto)
    assert decision.reasons == details.get("reasons")


def test_risk_engine_builds_snapshot(monkeypatch):
    monkeypatch.setattr("execution.risk_engine_v6.universe_by_symbol", lambda: {"BTCUSDT": {}, "ETHUSDT": {}})

    def fake_health(sym: str) -> Dict[str, Any]:
        return {"symbol": sym, "router": {}, "risk": {}}

    monkeypatch.setattr("execution.risk_engine_v6.compute_execution_health", fake_health)
    engine = RiskEngineV6.from_configs({}, {})
    snapshot = engine.build_risk_snapshot()
    assert len(snapshot["symbols"]) == 2
    assert snapshot["symbols"][0]["symbol"] in {"BTCUSDT", "ETHUSDT"}


def test_screener_would_emit_identical_with_risk_engine(monkeypatch):
    monkeypatch.setattr(screener, "RISK_ENGINE_V6_ENABLED", False)
    monkeypatch.setattr(screener, "is_listed_on_futures", lambda _s: True)
    monkeypatch.setattr(screener, "symbol_tier", lambda _s: "CORE")
    monkeypatch.setattr(screener, "symbol_target_leverage", lambda _s: 1.0)
    monkeypatch.setattr(screener, "_strategy_concurrency_budget", lambda: 0)
    monkeypatch.setattr(screener, "_load_risk_cfg", lambda: {"global": {}})
    monkeypatch.setattr(screener, "symbol_min_gross", lambda _s: 0.0)
    monkeypatch.setattr(screener, "symbol_min_notional", lambda _s: 0.0)
    monkeypatch.setattr(screener, "_entry_gate_result", lambda *args, **kwargs: (False, {"metric": 0.0}))

    def fake_check_order(*_args, **_kwargs):
        return True, {"reasons": ["portfolio_cap"], "thresholds": {"cap": 1.0}}

    monkeypatch.setattr(screener, "check_order", fake_check_order)
    ok_old, reasons_old, _ = screener.would_emit(
        "BTCUSDT",
        "BUY",
        notional=10.0,
        lev=1.0,
        nav=1000.0,
        open_positions_count=0,
        current_gross_notional=0.0,
        current_tier_gross_notional=0.0,
        orderbook_gate=False,
        timeframe=None,
        candle_close_ts=None,
    )

    class FakeEngine:
        def check_order(self, *_args, **_kwargs):
            return RiskDecision(
                allowed=False,
                clamped_qty=0.0,
                reasons=["portfolio_cap"],
                hit_caps={"portfolio_cap": True},
                diagnostics={"reasons": ["portfolio_cap"], "thresholds": {"cap": 1.0}},
            )

    fake_engine = FakeEngine()
    monkeypatch.setattr(screener, "_RISK_ENGINE_V6", fake_engine)
    monkeypatch.setattr(screener, "_get_risk_engine_v6", lambda _cfg: fake_engine)
    monkeypatch.setattr(screener, "RISK_ENGINE_V6_ENABLED", True)
    ok_new, reasons_new, _ = screener.would_emit(
        "BTCUSDT",
        "BUY",
        notional=10.0,
        lev=1.0,
        nav=1000.0,
        open_positions_count=0,
        current_gross_notional=0.0,
        current_tier_gross_notional=0.0,
        orderbook_gate=False,
        timeframe=None,
        candle_close_ts=None,
    )
    assert ok_old == ok_new
    assert reasons_old == reasons_new


def test_executor_risk_helper_parity(monkeypatch):
    def fake_check_order(*_args, **_kwargs):
        return True, {"reasons": ["max_concurrent"], "thresholds": {"max_concurrent_positions": 1}}

    monkeypatch.setattr(executor, "check_order", fake_check_order)
    intent = {"price": 0.0}
    executor.RISK_ENGINE_V6_ENABLED = False
    risk_veto_old, details_old = executor._evaluate_order_risk(
        "BTCUSDT",
        "BUY",
        gross_target=100.0,
        nav=1000.0,
        sym_open_qty=0.0,
        current_gross=0.0,
        open_positions_count=0,
        tier_name=None,
        current_tier_gross=0.0,
        lev=1.0,
        reduce_only=False,
        intent=intent,
    )

    class FakeEngine:
        def check_order(self, *_args, **_kwargs):
            return RiskDecision(
                allowed=False,
                clamped_qty=0.0,
                reasons=["max_concurrent"],
                hit_caps={"max_concurrent": True},
                diagnostics={"reasons": ["max_concurrent"], "thresholds": {"max_concurrent_positions": 1}},
            )

    fake_engine = FakeEngine()
    monkeypatch.setattr(executor, "_RISK_ENGINE_V6", fake_engine)
    monkeypatch.setattr(executor, "_get_risk_engine_v6", lambda: fake_engine)
    monkeypatch.setattr(executor, "RISK_ENGINE_V6_ENABLED", True)
    risk_veto_new, details_new = executor._evaluate_order_risk(
        "BTCUSDT",
        "BUY",
        gross_target=100.0,
        nav=1000.0,
        sym_open_qty=0.0,
        current_gross=0.0,
        open_positions_count=0,
        tier_name=None,
        current_tier_gross=0.0,
        lev=1.0,
        reduce_only=False,
        intent=intent,
    )
    assert risk_veto_old == risk_veto_new
    assert json.dumps(details_old, sort_keys=True) == json.dumps(details_new, sort_keys=True)
