from __future__ import annotations

import pytest

from execution.strategies.vol_target import (
    CarryConfig,
    TrendConfig,
    VolTargetConfig,
    compute_carry_bias,
    compute_trend_bias,
    decide_hybrid_side,
    generate_vol_target_intent,
    load_carry_inputs,
    load_htf_trend_data,
)


@pytest.fixture
def basic_cfg() -> dict:
    return {
        "enabled": True,
        "base_per_trade_nav_pct": 0.015,
        "min_per_trade_nav_pct": 0.005,
        "max_per_trade_nav_pct": 0.03,
        "target_vol": 0.015,
        "min_vol": 0.003,
        "max_vol": 0.08,
        "min_vol_factor": 0.25,
        "max_vol_factor": 2.0,
        "atr_lookback": 14,
        "use_atr_percentiles": True,
        "require_trend_alignment": True,
        "max_dd_regime": 2,
        "max_risk_mode": "DEFENSIVE",
        "min_signal_score": 0.0,
        "sl_atr_mult": 2.0,
        "tp_atr_mult": 3.0,
        "min_rr": 1.2,
        "side_mode": "trend",
        "enable_tp_sl": True,
        "trend": {
            "htf_tf": "1h",
            "fast_ema": 5,
            "slow_ema": 10,
            "min_trend_strength": 0.05,
            "use_htf_rsi_filter": True,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
        },
        "carry": {
            "use_funding": True,
            "min_funding_annualized": 0.0,
            "max_funding_annualized": 0.5,
            "funding_weight": 0.3,
            "use_basis": False,
            "max_basis_pct": 0.1,
            "basis_weight": 0.2,
        },
    }


def test_trend_bias_uptrend():
    closes = [10, 11, 12, 13, 14, 15]
    cfg = TrendConfig(fast_ema=3, slow_ema=5, min_trend_strength=0.05, use_htf_rsi_filter=False, htf_tf="1h", rsi_overbought=70, rsi_oversold=30)
    trend_info = compute_trend_bias(closes, htf_rsi=None, cfg=cfg)
    assert trend_info["direction"] == "LONG"
    assert trend_info["strength"] > cfg.min_trend_strength


def test_trend_bias_downtrend():
    closes = [15, 14, 13, 12, 11, 10]
    cfg = TrendConfig(fast_ema=3, slow_ema=5, min_trend_strength=0.05, use_htf_rsi_filter=False, htf_tf="1h", rsi_overbought=70, rsi_oversold=30)
    trend_info = compute_trend_bias(closes, htf_rsi=None, cfg=cfg)
    assert trend_info["direction"] == "SHORT"


def test_carry_bias_positive_and_negative():
    cfg = CarryConfig(use_funding=True, min_funding_annualized=0.0, max_funding_annualized=0.5, funding_weight=0.3, use_basis=False, max_basis_pct=0.1, basis_weight=0.2)
    pos = compute_carry_bias(funding_annualized=0.1, basis_pct=None, cfg=cfg)
    neg = compute_carry_bias(funding_annualized=-0.1, basis_pct=None, cfg=cfg)
    assert pos["score_short"] > 0
    assert neg["score_long"] > 0


def test_hybrid_confluence_long():
    cfg = VolTargetConfig(
        enabled=True,
        base_per_trade_nav_pct=0.01,
        min_per_trade_nav_pct=0.005,
        max_per_trade_nav_pct=0.03,
        target_vol=0.015,
        min_vol=0.003,
        max_vol=0.08,
        min_vol_factor=0.25,
        max_vol_factor=2.0,
        atr_lookback=14,
        use_atr_percentiles=True,
        require_trend_alignment=True,
        max_dd_regime=2,
        max_risk_mode="DEFENSIVE",
        min_signal_score=0.0,
        sl_atr_mult=2.0,
        tp_atr_mult=3.0,
        min_rr=1.2,
        side_mode="trend",
        enable_tp_sl=True,
    )
    trend_info = {"direction": "LONG", "strength": 0.6}
    carry_info = {"score_long": 0.2, "score_short": 0.0}
    result = decide_hybrid_side(trend_info, carry_info, cfg)
    assert result["side"] == "BUY"
    assert result["hybrid_score"] > 0.5


def test_rsi_filter_can_flatten_trend():
    closes = [100, 100.1, 100.2, 100.3, 100.4, 100.5]
    cfg = TrendConfig(fast_ema=3, slow_ema=4, min_trend_strength=0.2, use_htf_rsi_filter=True, rsi_overbought=50, rsi_oversold=30, htf_tf="1h")
    trend_info = compute_trend_bias(closes, htf_rsi=80.0, cfg=cfg)
    assert trend_info["direction"] == "FLAT"


def test_side_mode_long_only_blocks_sell():
    cfg = VolTargetConfig(
        enabled=True,
        base_per_trade_nav_pct=0.01,
        min_per_trade_nav_pct=0.005,
        max_per_trade_nav_pct=0.03,
        target_vol=0.015,
        min_vol=0.003,
        max_vol=0.08,
        min_vol_factor=0.25,
        max_vol_factor=2.0,
        atr_lookback=14,
        use_atr_percentiles=True,
        require_trend_alignment=False,
        max_dd_regime=2,
        max_risk_mode="DEFENSIVE",
        min_signal_score=0.0,
        sl_atr_mult=2.0,
        tp_atr_mult=3.0,
        min_rr=1.2,
        side_mode="long_only",
        enable_tp_sl=True,
    )
    trend_info = {"direction": "SHORT", "strength": 0.8}
    carry_info = {"score_long": 0.0, "score_short": 0.0}
    result = decide_hybrid_side(trend_info, carry_info, cfg)
    assert result["side"] == "NONE"


def test_generate_intent_includes_hybrid_metadata(monkeypatch, basic_cfg):
    def fake_htf(symbol, tf, fast, slow):
        closes = [10, 11, 12, 13, 14, 15]
        return closes, 55.0

    def fake_carry(symbol):
        return -0.1, None

    monkeypatch.setattr("execution.strategies.vol_target.load_htf_trend_data", fake_htf)
    monkeypatch.setattr("execution.strategies.vol_target.load_carry_inputs", fake_carry)

    intent = generate_vol_target_intent(
        symbol="BTCUSDT",
        timeframe="15m",
        price=100.0,
        nav=10000.0,
        atr_value=1.0,
        regimes_snapshot={"atr_regime": 1, "dd_regime": 0},
        risk_snapshot={"risk_mode": "OK"},
        trend="BULL",
        trend_aligned=True,
        strategy_cfg=basic_cfg,
    )
    assert intent is not None
    meta = intent["metadata"]["vol_target"]
    assert "trend" in meta and "carry" in meta and "hybrid" in meta
    assert meta["hybrid"]["side"] == "BUY"


def test_generate_intent_returns_none_when_hybrid_none(monkeypatch, basic_cfg):
    def fake_htf(symbol, tf, fast, slow):
        return [], None

    def fake_carry(symbol):
        return None, None

    monkeypatch.setattr("execution.strategies.vol_target.load_htf_trend_data", fake_htf)
    monkeypatch.setattr("execution.strategies.vol_target.load_carry_inputs", fake_carry)

    intent = generate_vol_target_intent(
        symbol="BTCUSDT",
        timeframe="15m",
        price=100.0,
        nav=10000.0,
        atr_value=1.0,
        regimes_snapshot={"atr_regime": 1, "dd_regime": 0},
        risk_snapshot={"risk_mode": "OK"},
        trend="NEUTRAL",
        trend_aligned=False,
        strategy_cfg=basic_cfg,
    )
    assert intent is None
