from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import json

from execution.pipeline_v6_shadow import (
    PIPELINE_SHADOW_LOG,
    append_shadow_decision,
    build_shadow_summary,
    load_shadow_decisions,
    run_pipeline_v6_shadow,
)
from execution.risk_engine_v6 import RiskDecision, RiskEngineV6


class FakeRiskEngine:
    def __init__(self, allowed: bool = True):
        self.allowed = allowed
        self.calls = 0

    def check_order(self, intent, state):
        self.calls += 1
        if not self.allowed:
            return RiskDecision(allowed=False, clamped_qty=0.0, reasons=["portfolio_cap"], hit_caps={"portfolio_cap": True})
        return RiskDecision(allowed=True, clamped_qty=intent.qty, reasons=[], hit_caps={})


def _base_signal():
    return {"side": "BUY", "notional": 100.0, "price": 10.0, "signal_strength": 1.0}


def _nav_state():
    return {"nav_usd": 1000.0, "portfolio_gross_usd": 0.0, "symbol_open_qty": 0.0}


def test_pipeline_shadow_allowed(monkeypatch, tmp_path):
    tmp_log = tmp_path / "shadow.jsonl"
    monkeypatch.setattr("execution.pipeline_v6_shadow.PIPELINE_SHADOW_LOG", tmp_log)
    monkeypatch.setattr("execution.pipeline_v6_shadow.router_policy", lambda _s: type("P", (), {"maker_first": True, "taker_bias": "balanced", "quality": "ok"})())
    monkeypatch.setattr("execution.pipeline_v6_shadow.suggest_maker_offset_bps", lambda _s: 1.0)
    monkeypatch.setattr("execution.pipeline_v6_shadow.effective_px", lambda px, side, is_maker=True: px)
    engine = FakeRiskEngine(allowed=True)
    result = run_pipeline_v6_shadow(
        "BTCUSDT",
        _base_signal(),
        _nav_state(),
        {},
        {},
        {},
        {},
        risk_engine=engine,
    )
    assert result["risk_decision"]["allowed"] is True
    assert "router_decision" in result
    assert result["size_decision"]["gross_usd"] == 100.0
    append_shadow_decision(result)
    entries = load_shadow_decisions(limit=10)
    assert len(entries) == 1
    summary = build_shadow_summary(entries)
    assert summary["allowed"] == 1


def test_pipeline_shadow_veto(monkeypatch):
    monkeypatch.setattr("execution.pipeline_v6_shadow.router_policy", lambda _s: type("P", (), {"maker_first": True, "taker_bias": "balanced", "quality": "ok"})())
    engine = FakeRiskEngine(allowed=False)
    result = run_pipeline_v6_shadow(
        "ETHUSDT",
        _base_signal(),
        _nav_state(),
        {},
        {},
        {},
        {},
        risk_engine=engine,
    )
    assert result["risk_decision"]["allowed"] is False
    assert "router_decision" not in result


def test_shadow_summary(tmp_path, monkeypatch):
    tmp_log = tmp_path / "shadow.jsonl"
    monkeypatch.setattr("execution.pipeline_v6_shadow.PIPELINE_SHADOW_LOG", tmp_log)
    append_shadow_decision({"risk_decision": {"allowed": True}})
    append_shadow_decision({"risk_decision": {"allowed": False}})
    entries = load_shadow_decisions(limit=5)
    summary = build_shadow_summary(entries)
    assert summary["total"] == 2
    assert summary["allowed"] == 1
