from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from execution.position_sizing import compute_adaptive_weight, compute_strategy_performance_factor


def test_risk_mode_factors():
    cfg = {}
    regimes = {"atr_regime": 0, "dd_regime": 0}
    pnl = {}
    assert compute_adaptive_weight("s", regimes, {"risk_mode": "OK"}, pnl, cfg) == pytest.approx(1.0)
    assert compute_adaptive_weight("s", regimes, {"risk_mode": "WARN"}, pnl, cfg) == pytest.approx(0.8)
    assert compute_adaptive_weight("s", regimes, {"risk_mode": "DEFENSIVE"}, pnl, cfg) == pytest.approx(0.5)
    assert compute_adaptive_weight("s", regimes, {"risk_mode": "HALTED"}, pnl, cfg) == pytest.approx(0.0)


def test_atr_rules_application():
    cfg = {}
    pnl = {}
    regimes = {"atr_regime": 3, "dd_regime": 0}
    weight = compute_adaptive_weight("s", regimes, {"risk_mode": "OK"}, pnl, cfg)
    assert weight == pytest.approx(0.5)


def test_dd_rules_application():
    cfg = {}
    pnl = {}
    regimes = {"atr_regime": 0, "dd_regime": 3}
    weight = compute_adaptive_weight("s", regimes, {"risk_mode": "OK"}, pnl, cfg)
    assert weight == pytest.approx(0.0)


def test_performance_factor_low_high():
    pnl = {"per_strategy": {"mom": {"wins": 2, "trade_count": 10}}}
    perf_cfg = {"performance_rules": {"winrate_low_threshold": 0.3, "winrate_high_threshold": 0.6, "low_factor": 0.5, "high_factor": 1.2}}
    low_factor = compute_strategy_performance_factor("mom", pnl, perf_cfg["performance_rules"])
    assert low_factor == pytest.approx(0.5)
    pnl_high = {"per_strategy": {"mom": {"wins": 8, "trade_count": 10}}}
    high_factor = compute_strategy_performance_factor("mom", pnl_high, perf_cfg["performance_rules"])
    assert high_factor == pytest.approx(1.2)


def test_missing_performance_defaults_to_1():
    assert compute_strategy_performance_factor("x", {}, {}) == 1.0


def test_weight_clamping():
    cfg = {"base_weight": 1.0, "risk_mode_rules": {"OK": 3.0}, "max_weight": 1.0}
    weight = compute_adaptive_weight("s", {"atr_regime": 0, "dd_regime": 0}, {"risk_mode": "OK"}, {}, cfg)
    assert weight == pytest.approx(1.0)


def test_final_weight_combination_all_factors():
    cfg = {
        "performance_rules": {"winrate_high_threshold": 0.6, "high_factor": 1.1, "winrate_low_threshold": 0.2, "low_factor": 0.7},
    }
    pnl = {"per_strategy": {"s": {"wins": 7, "trade_count": 10}}}
    regimes = {"atr_regime": 3, "dd_regime": 2}
    risk = {"risk_mode": "WARN"}
    weight = compute_adaptive_weight("s", regimes, risk, pnl, cfg)
    expected = 1.0 * 0.8 * 0.5 * 0.5 * 1.1
    assert weight == pytest.approx(expected)


def test_strategy_disabled_in_halted_returns_zero():
    weight = compute_adaptive_weight("s", {"atr_regime": 0, "dd_regime": 0}, {"risk_mode": "HALTED"}, {}, {})
    assert weight == 0.0


def test_strategy_disabled_in_dd_critical_returns_zero():
    weight = compute_adaptive_weight("s", {"atr_regime": 0, "dd_regime": 3}, {"risk_mode": "OK"}, {}, {})
    assert weight == 0.0


def test_missing_regimes_defaults_to_factor_1():
    weight = compute_adaptive_weight("s", {}, {"risk_mode": "OK"}, {}, {})
    assert weight == pytest.approx(1.0)


def test_missing_risk_snapshot_defaults_to_ok():
    weight = compute_adaptive_weight("s", {"atr_regime": 0, "dd_regime": 0}, {}, {}, {})
    assert weight == pytest.approx(1.0)


def _mock_strategy():
    return [{"id": "T1", "symbol": "BTCUSDT", "timeframe": "1h", "params": {"per_trade_nav_pct": 0.01}}]


def test_qty_calculation_correct_with_weight_applied(monkeypatch):
    import execution.signal_screener as ss
    import execution.exchange_utils as eu

    monkeypatch.setattr(ss, "_load_strategy_list", _mock_strategy)
    monkeypatch.setattr(ss, "resolve_allowed_symbols", lambda: (["BTCUSDT"], {"BTCUSDT": "CORE"}))
    monkeypatch.setattr(eu, "get_positions", lambda: [])  # Patch at source module
    monkeypatch.setattr(ss, "allow_trade", lambda _sym: True)
    monkeypatch.setattr(ss, "symbol_min_gross", lambda _sym: 0.0)
    monkeypatch.setattr(ss, "symbol_min_notional", lambda _sym: 0.0)
    monkeypatch.setattr(ss, "symbol_target_leverage", lambda _sym: 1.0)
    monkeypatch.setattr(ss, "_zscore", lambda *_args, **_kwargs: 1.0)
    monkeypatch.setattr(ss, "_rsi", lambda *_args, **_kwargs: 80.0)
    monkeypatch.setattr(ss, "_trend_filter", lambda *_args, **_kwargs: "BEAR")
    monkeypatch.setattr(
        ss,
        "get_symbol_filters",
        lambda *_args, **_kwargs: {"MIN_NOTIONAL": {"minNotional": 0}, "LOT_SIZE": {"minQty": 0.001, "stepSize": 0.001}},
    )
    monkeypatch.setattr(ss, "get_klines", lambda *_args, **_kwargs: [[0, 0, 0, 0, 100, 0] for _ in range(50)])
    monkeypatch.setattr(ss, "get_price", lambda *_args, **_kwargs: 100.0)
    monkeypatch.setattr(ss, "PortfolioSnapshot", lambda *_args, **_kwargs: SimpleNamespace(current_gross_usd=lambda: 0.0))
    monkeypatch.setattr(ss, "nav_health_snapshot", lambda: {"nav_total": 10000, "sources_ok": True, "age_s": 0})
    monkeypatch.setattr(ss, "adaptive_sizing", lambda *a, **k: (a[1], 1.0))
    monkeypatch.setattr(ss, "strategy_enablement", lambda *a, **k: True)
    monkeypatch.setattr(ss, "_load_pnl_attribution", lambda: {"per_strategy": {"T1": {"wins": 1, "trade_count": 2}}})
    monkeypatch.setattr(ss, "load_regime_snapshot", lambda: {"atr_regime": 0, "dd_regime": 0})
    monkeypatch.setattr(ss, "load_risk_snapshot", lambda: {"risk_mode": "OK"})

    base_cfg = {
        "strategy_weighting": {"T1": {"base_weight": 0.5}},
        "strategies": _mock_strategy(),
        "vol_target": {"enabled": False},
    }
    monkeypatch.setattr(ss.json, "load", lambda *_args, **_kwargs: base_cfg)

    intents = ss.generate_signals_from_config()
    assert len(intents) == 1
    intent = intents[0]
    assert intent["gross_usd"] == pytest.approx(10000 * 0.01 * 0.5)
    assert intent["qty"] == pytest.approx(intent["gross_usd"] / 100.0)
    assert intent["metadata"]["adaptive_weight"]["final_weight"] == pytest.approx(0.5)
