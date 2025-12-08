"""
Tests for Volatility Target Strategy (v7.3-alpha)

Tests cover:
1. Low volatility → larger gross_usd (within max clamp)
2. High volatility → smaller gross_usd (within min clamp)
3. Zero or missing ATR → no intent
4. Risk mode gating
5. DD regime gating
6. Trend alignment gating
7. Metadata correctness
8. NAV/price sanity
"""

import pytest

from execution.strategies.vol_target import (
    VolTargetConfig,
    RISK_MODE_ORDER,
    _risk_mode_allowed,
    compute_vol_factor,
    compute_per_trade_nav_pct,
    compute_tp_sl_prices,
    generate_vol_target_intent,
)


@pytest.fixture(autouse=True)
def _stub_vol_target_fetches(monkeypatch):
    """Prevent network calls during tests by stubbing HTF and carry fetchers."""
    def fake_htf(symbol, tf, fast, slow):
        closes = [100.0] * max(fast, slow, 5)
        return closes, 50.0

    def fake_carry(symbol):
        return None, None

    monkeypatch.setattr("execution.strategies.vol_target.load_htf_trend_data", fake_htf)
    monkeypatch.setattr("execution.strategies.vol_target.load_carry_inputs", fake_carry)
@pytest.fixture
def base_config() -> dict:
    """Base vol_target configuration for tests."""
    return {
        "enabled": True,
        "base_per_trade_nav_pct": 0.015,  # 1.5%
        "min_per_trade_nav_pct": 0.005,   # 0.5%
        "max_per_trade_nav_pct": 0.03,    # 3%
        "target_vol": 0.015,              # 1.5% target vol
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
        # TP/SL fields (v7.3-alpha1)
        "sl_atr_mult": 2.0,
        "tp_atr_mult": 3.0,
        "min_rr": 1.2,
        "side_mode": "trend",
        "enable_tp_sl": True,
        "trend": {
            "htf_tf": "1h",
            "fast_ema": 21,
            "slow_ema": 50,
            "min_trend_strength": 0.1,
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


@pytest.fixture
def base_regimes() -> dict:
    """Base regime snapshot."""
    return {
        "atr_regime": 1,
        "dd_regime": 0,
    }


@pytest.fixture
def base_risk() -> dict:
    """Base risk snapshot."""
    return {
        "risk_mode": "OK",
    }


class TestRiskModeAllowed:
    """Test risk mode ordering and validation."""

    def test_ok_below_defensive(self):
        assert _risk_mode_allowed("OK", "DEFENSIVE") is True

    def test_warn_below_defensive(self):
        assert _risk_mode_allowed("WARN", "DEFENSIVE") is True

    def test_defensive_at_defensive(self):
        assert _risk_mode_allowed("DEFENSIVE", "DEFENSIVE") is True

    def test_halted_above_defensive(self):
        assert _risk_mode_allowed("HALTED", "DEFENSIVE") is False

    def test_halted_at_halted(self):
        assert _risk_mode_allowed("HALTED", "HALTED") is True

    def test_unknown_mode_is_conservative(self):
        assert _risk_mode_allowed("UNKNOWN", "DEFENSIVE") is False

    def test_case_insensitive(self):
        assert _risk_mode_allowed("ok", "DEFENSIVE") is True
        assert _risk_mode_allowed("OK", "defensive") is True


class TestComputeVolFactor:
    """Test volatility factor computation."""

    def test_low_vol_produces_factor_above_one(self, base_config):
        """Low vol → vol_factor > 1 (scale up)."""
        cfg = VolTargetConfig(**base_config)
        price = 100.0
        # ATR = 0.5, vol = 0.5/100 = 0.005 (below target 0.015)
        atr_value = 0.5
        factor = compute_vol_factor(atr_value, price, cfg)
        assert factor is not None
        assert factor > 1.0  # Should scale up for low vol
        # vol = 0.005, target = 0.015, factor = 0.015/0.005 = 3.0, clamped to max 2.0
        assert factor == pytest.approx(2.0)

    def test_high_vol_produces_factor_below_one(self, base_config):
        """High vol → vol_factor < 1 (scale down)."""
        cfg = VolTargetConfig(**base_config)
        price = 100.0
        # ATR = 6.0, vol = 6.0/100 = 0.06 (above target 0.015)
        atr_value = 6.0
        factor = compute_vol_factor(atr_value, price, cfg)
        assert factor is not None
        assert factor < 1.0  # Should scale down for high vol
        # vol = 0.06, target = 0.015, factor = 0.015/0.06 = 0.25
        assert factor == pytest.approx(0.25)

    def test_target_vol_produces_factor_near_one(self, base_config):
        """Vol at target → factor ≈ 1."""
        cfg = VolTargetConfig(**base_config)
        price = 100.0
        # ATR = 1.5, vol = 1.5/100 = 0.015 (at target)
        atr_value = 1.5
        factor = compute_vol_factor(atr_value, price, cfg)
        assert factor is not None
        assert factor == pytest.approx(1.0)

    def test_zero_atr_returns_none(self, base_config):
        """Zero ATR → no intent."""
        cfg = VolTargetConfig(**base_config)
        factor = compute_vol_factor(0.0, 100.0, cfg)
        assert factor is None

    def test_negative_atr_returns_none(self, base_config):
        """Negative ATR → no intent."""
        cfg = VolTargetConfig(**base_config)
        factor = compute_vol_factor(-1.0, 100.0, cfg)
        assert factor is None

    def test_zero_price_returns_none(self, base_config):
        """Zero price → no intent."""
        cfg = VolTargetConfig(**base_config)
        factor = compute_vol_factor(1.5, 0.0, cfg)
        assert factor is None

    def test_factor_clamped_to_min(self, base_config):
        """Very high vol → factor clamped to min."""
        cfg = VolTargetConfig(**base_config)
        price = 100.0
        # ATR = 8.0, vol = 8.0/100 = 0.08 (at max_vol)
        # factor = 0.015/0.08 = 0.1875, but min_vol_factor = 0.25
        atr_value = 8.0
        factor = compute_vol_factor(atr_value, price, cfg)
        assert factor is not None
        assert factor == pytest.approx(0.25)

    def test_factor_clamped_to_max(self, base_config):
        """Very low vol → factor clamped to max."""
        cfg = VolTargetConfig(**base_config)
        price = 100.0
        # ATR = 0.3, vol = 0.3/100 = 0.003 (at min_vol)
        # factor = 0.015/0.003 = 5.0, but max_vol_factor = 2.0
        atr_value = 0.3
        factor = compute_vol_factor(atr_value, price, cfg)
        assert factor is not None
        assert factor == pytest.approx(2.0)


class TestComputePerTradeNavPct:
    """Test per-trade NAV percentage computation."""

    def test_scaled_and_clamped_to_max(self, base_config):
        """Factor > 1 scaled up but clamped."""
        cfg = VolTargetConfig(**base_config)
        # base = 0.015, factor = 2.0 → raw = 0.03, max = 0.03
        pct = compute_per_trade_nav_pct(0.015, 2.0, cfg)
        assert pct == pytest.approx(0.03)

    def test_scaled_and_clamped_to_min(self, base_config):
        """Factor < 1 scaled down but clamped."""
        cfg = VolTargetConfig(**base_config)
        # base = 0.015, factor = 0.25 → raw = 0.00375, min = 0.005
        pct = compute_per_trade_nav_pct(0.015, 0.25, cfg)
        assert pct == pytest.approx(0.005)

    def test_factor_one_equals_base(self, base_config):
        """Factor = 1 → base pct."""
        cfg = VolTargetConfig(**base_config)
        pct = compute_per_trade_nav_pct(0.015, 1.0, cfg)
        assert pct == pytest.approx(0.015)


class TestGenerateVolTargetIntent:
    """Test full intent generation."""

    def test_low_vol_larger_gross_usd(self, base_config, base_regimes, base_risk):
        """Low volatility → larger gross_usd (within max clamp)."""
        intent = generate_vol_target_intent(
            symbol="BTCUSDT",
            timeframe="15m",
            price=100.0,
            nav=10000.0,
            atr_value=0.5,  # Low vol
            regimes_snapshot=base_regimes,
            risk_snapshot=base_risk,
            trend="BULL",
            trend_aligned=True,
            strategy_cfg=base_config,
        )
        assert intent is not None
        # vol_factor = 2.0 (clamped), per_trade_nav_pct = 0.03 (max)
        assert intent["gross_usd"] == pytest.approx(10000.0 * 0.03)
        assert intent["per_trade_nav_pct"] == pytest.approx(0.03)

    def test_high_vol_smaller_gross_usd(self, base_config, base_regimes, base_risk):
        """High volatility → smaller gross_usd (within min clamp)."""
        intent = generate_vol_target_intent(
            symbol="BTCUSDT",
            timeframe="15m",
            price=100.0,
            nav=10000.0,
            atr_value=6.0,  # High vol
            regimes_snapshot=base_regimes,
            risk_snapshot=base_risk,
            trend="BULL",
            trend_aligned=True,
            strategy_cfg=base_config,
        )
        assert intent is not None
        # vol_factor = 0.25 (clamped), per_trade_nav_pct = 0.005 (min)
        assert intent["gross_usd"] == pytest.approx(10000.0 * 0.005)
        assert intent["per_trade_nav_pct"] == pytest.approx(0.005)

    def test_zero_atr_no_intent(self, base_config, base_regimes, base_risk):
        """Zero ATR → no intent."""
        intent = generate_vol_target_intent(
            symbol="BTCUSDT",
            timeframe="15m",
            price=100.0,
            nav=10000.0,
            atr_value=0.0,  # No ATR data
            regimes_snapshot=base_regimes,
            risk_snapshot=base_risk,
            trend="BULL",
            trend_aligned=True,
            strategy_cfg=base_config,
        )
        assert intent is None

    def test_negative_atr_no_intent(self, base_config, base_regimes, base_risk):
        """Negative ATR → no intent."""
        intent = generate_vol_target_intent(
            symbol="BTCUSDT",
            timeframe="15m",
            price=100.0,
            nav=10000.0,
            atr_value=-1.0,
            regimes_snapshot=base_regimes,
            risk_snapshot=base_risk,
            trend="BULL",
            trend_aligned=True,
            strategy_cfg=base_config,
        )
        assert intent is None

    def test_risk_mode_halted_no_intent(self, base_config, base_regimes):
        """Risk mode HALTED → no intent."""
        risk = {"risk_mode": "HALTED"}
        intent = generate_vol_target_intent(
            symbol="BTCUSDT",
            timeframe="15m",
            price=100.0,
            nav=10000.0,
            atr_value=1.5,
            regimes_snapshot=base_regimes,
            risk_snapshot=risk,
            trend="BULL",
            trend_aligned=True,
            strategy_cfg=base_config,
        )
        assert intent is None

    def test_risk_mode_defensive_allowed(self, base_config, base_regimes):
        """Risk mode DEFENSIVE with max_risk_mode=DEFENSIVE → intent allowed."""
        risk = {"risk_mode": "DEFENSIVE"}
        intent = generate_vol_target_intent(
            symbol="BTCUSDT",
            timeframe="15m",
            price=100.0,
            nav=10000.0,
            atr_value=1.5,
            regimes_snapshot=base_regimes,
            risk_snapshot=risk,
            trend="BULL",
            trend_aligned=True,
            strategy_cfg=base_config,
        )
        assert intent is not None

    def test_dd_regime_critical_no_intent(self, base_config, base_risk):
        """DD regime = 3 (CRITICAL) when max_dd_regime = 2 → no intent."""
        regimes = {"atr_regime": 1, "dd_regime": 3}
        intent = generate_vol_target_intent(
            symbol="BTCUSDT",
            timeframe="15m",
            price=100.0,
            nav=10000.0,
            atr_value=1.5,
            regimes_snapshot=regimes,
            risk_snapshot=base_risk,
            trend="BULL",
            trend_aligned=True,
            strategy_cfg=base_config,
        )
        assert intent is None

    def test_dd_regime_high_allowed(self, base_config, base_risk):
        """DD regime = 2 (HIGH) when max_dd_regime = 2 → intent allowed."""
        regimes = {"atr_regime": 1, "dd_regime": 2}
        intent = generate_vol_target_intent(
            symbol="BTCUSDT",
            timeframe="15m",
            price=100.0,
            nav=10000.0,
            atr_value=1.5,
            regimes_snapshot=regimes,
            risk_snapshot=base_risk,
            trend="BULL",
            trend_aligned=True,
            strategy_cfg=base_config,
        )
        assert intent is not None

    def test_trend_alignment_required_not_aligned(self, base_config, base_regimes, base_risk):
        """require_trend_alignment=True and trend_aligned=False → no intent."""
        intent = generate_vol_target_intent(
            symbol="BTCUSDT",
            timeframe="15m",
            price=100.0,
            nav=10000.0,
            atr_value=1.5,
            regimes_snapshot=base_regimes,
            risk_snapshot=base_risk,
            trend="BEAR",
            trend_aligned=False,  # Not aligned
            strategy_cfg=base_config,
        )
        assert intent is None

    def test_trend_alignment_not_required(self, base_config, base_regimes, base_risk):
        """require_trend_alignment=False → intent allowed regardless of trend."""
        cfg = dict(base_config)
        cfg["require_trend_alignment"] = False
        intent = generate_vol_target_intent(
            symbol="BTCUSDT",
            timeframe="15m",
            price=100.0,
            nav=10000.0,
            atr_value=1.5,
            regimes_snapshot=base_regimes,
            risk_snapshot=base_risk,
            trend="BEAR",
            trend_aligned=False,
            strategy_cfg=cfg,
        )
        assert intent is not None

    def test_metadata_correctness(self, base_config, base_regimes, base_risk):
        """Verify metadata contains all expected vol_target fields."""
        intent = generate_vol_target_intent(
            symbol="BTCUSDT",
            timeframe="15m",
            price=100.0,
            nav=10000.0,
            atr_value=1.5,
            regimes_snapshot=base_regimes,
            risk_snapshot=base_risk,
            trend="BULL",
            trend_aligned=True,
            strategy_cfg=base_config,
        )
        assert intent is not None
        assert "metadata" in intent
        meta = intent["metadata"]
        assert meta.get("strategy") == "vol_target"
        assert "vol_target" in meta
        vol_meta = meta["vol_target"]
        assert "atr_value" in vol_meta
        assert vol_meta["atr_value"] == pytest.approx(1.5)
        assert "vol_factor" in vol_meta
        assert "target_vol" in vol_meta
        assert vol_meta["target_vol"] == pytest.approx(0.015)
        assert "base_per_trade_nav_pct" in vol_meta
        assert "computed_per_trade_nav_pct" in vol_meta
        assert "atr_regime" in vol_meta
        assert "dd_regime" in vol_meta
        assert "risk_mode" in vol_meta

    def test_zero_nav_no_intent(self, base_config, base_regimes, base_risk):
        """NAV <= 0 → no intent."""
        intent = generate_vol_target_intent(
            symbol="BTCUSDT",
            timeframe="15m",
            price=100.0,
            nav=0.0,
            atr_value=1.5,
            regimes_snapshot=base_regimes,
            risk_snapshot=base_risk,
            trend="BULL",
            trend_aligned=True,
            strategy_cfg=base_config,
        )
        assert intent is None

    def test_negative_nav_no_intent(self, base_config, base_regimes, base_risk):
        """Negative NAV → no intent."""
        intent = generate_vol_target_intent(
            symbol="BTCUSDT",
            timeframe="15m",
            price=100.0,
            nav=-1000.0,
            atr_value=1.5,
            regimes_snapshot=base_regimes,
            risk_snapshot=base_risk,
            trend="BULL",
            trend_aligned=True,
            strategy_cfg=base_config,
        )
        assert intent is None

    def test_zero_price_no_intent(self, base_config, base_regimes, base_risk):
        """Price <= 0 → no intent."""
        intent = generate_vol_target_intent(
            symbol="BTCUSDT",
            timeframe="15m",
            price=0.0,
            nav=10000.0,
            atr_value=1.5,
            regimes_snapshot=base_regimes,
            risk_snapshot=base_risk,
            trend="BULL",
            trend_aligned=True,
            strategy_cfg=base_config,
        )
        assert intent is None

    def test_negative_price_no_intent(self, base_config, base_regimes, base_risk):
        """Negative price → no intent."""
        intent = generate_vol_target_intent(
            symbol="BTCUSDT",
            timeframe="15m",
            price=-100.0,
            nav=10000.0,
            atr_value=1.5,
            regimes_snapshot=base_regimes,
            risk_snapshot=base_risk,
            trend="BULL",
            trend_aligned=True,
            strategy_cfg=base_config,
        )
        assert intent is None

    def test_disabled_strategy_no_intent(self, base_config, base_regimes, base_risk):
        """Strategy disabled → no intent."""
        cfg = dict(base_config)
        cfg["enabled"] = False
        intent = generate_vol_target_intent(
            symbol="BTCUSDT",
            timeframe="15m",
            price=100.0,
            nav=10000.0,
            atr_value=1.5,
            regimes_snapshot=base_regimes,
            risk_snapshot=base_risk,
            trend="BULL",
            trend_aligned=True,
            strategy_cfg=cfg,
        )
        assert intent is None

    def test_intent_has_required_fields(self, base_config, base_regimes, base_risk):
        """Verify intent has all required fields for downstream processing."""
        intent = generate_vol_target_intent(
            symbol="BTCUSDT",
            timeframe="15m",
            price=100.0,
            nav=10000.0,
            atr_value=1.5,
            regimes_snapshot=base_regimes,
            risk_snapshot=base_risk,
            trend="BULL",
            trend_aligned=True,
            strategy_cfg=base_config,
        )
        assert intent is not None
        # Required fields
        assert "timestamp" in intent
        assert "symbol" in intent
        assert intent["symbol"] == "BTCUSDT"
        assert "timeframe" in intent
        assert intent["timeframe"] == "15m"
        assert "signal" in intent
        assert intent["signal"] == "BUY"
        assert "reduceOnly" in intent
        assert intent["reduceOnly"] is False
        assert "price" in intent
        assert intent["price"] == pytest.approx(100.0)
        assert "per_trade_nav_pct" in intent
        assert "gross_usd" in intent
        assert intent["gross_usd"] > 0


class TestComputeTpSlPrices:
    """Test TP/SL price computation (v7.3-alpha1)."""

    def test_basic_long_side_tp_sl(self, base_config):
        """BUY side → TP above entry, SL below entry."""
        cfg = VolTargetConfig(**base_config)
        result = compute_tp_sl_prices(price=100.0, atr_value=2.0, side="BUY", cfg=cfg)
        assert result is not None
        tp_price, sl_price, rr = result
        # SL = price - sl_atr_mult * ATR = 100 - 2*2 = 96
        # TP = price + tp_atr_mult * ATR = 100 + 3*2 = 106
        assert sl_price == pytest.approx(96.0)
        assert tp_price == pytest.approx(106.0)
        # RR = tp_dist / sl_dist = 6 / 4 = 1.5
        assert rr == pytest.approx(1.5)

    def test_basic_short_side_tp_sl(self, base_config):
        """SELL side → TP below entry, SL above entry."""
        cfg = VolTargetConfig(**base_config)
        result = compute_tp_sl_prices(price=100.0, atr_value=2.0, side="SELL", cfg=cfg)
        assert result is not None
        tp_price, sl_price, rr = result
        # SL = price + sl_atr_mult * ATR = 100 + 2*2 = 104
        # TP = price - tp_atr_mult * ATR = 100 - 3*2 = 94
        assert sl_price == pytest.approx(104.0)
        assert tp_price == pytest.approx(94.0)
        assert rr == pytest.approx(1.5)

    def test_tp_sl_disabled_returns_none(self, base_config):
        """enable_tp_sl=False → returns None."""
        cfg_dict = dict(base_config)
        cfg_dict["enable_tp_sl"] = False
        cfg = VolTargetConfig(**cfg_dict)
        result = compute_tp_sl_prices(price=100.0, atr_value=2.0, side="BUY", cfg=cfg)
        assert result is None

    def test_zero_atr_returns_none(self, base_config):
        """ATR = 0 → returns None."""
        cfg = VolTargetConfig(**base_config)
        result = compute_tp_sl_prices(price=100.0, atr_value=0.0, side="BUY", cfg=cfg)
        assert result is None

    def test_negative_atr_returns_none(self, base_config):
        """Negative ATR → returns None."""
        cfg = VolTargetConfig(**base_config)
        result = compute_tp_sl_prices(price=100.0, atr_value=-1.0, side="BUY", cfg=cfg)
        assert result is None

    def test_none_atr_returns_none(self, base_config):
        """None ATR → returns None."""
        cfg = VolTargetConfig(**base_config)
        result = compute_tp_sl_prices(price=100.0, atr_value=None, side="BUY", cfg=cfg)
        assert result is None

    def test_zero_price_returns_none(self, base_config):
        """Price = 0 → returns None."""
        cfg = VolTargetConfig(**base_config)
        result = compute_tp_sl_prices(price=0.0, atr_value=2.0, side="BUY", cfg=cfg)
        assert result is None

    def test_negative_price_returns_none(self, base_config):
        """Negative price → returns None."""
        cfg = VolTargetConfig(**base_config)
        result = compute_tp_sl_prices(price=-100.0, atr_value=2.0, side="BUY", cfg=cfg)
        assert result is None

    def test_invalid_side_returns_none(self, base_config):
        """Unknown side → returns None."""
        cfg = VolTargetConfig(**base_config)
        result = compute_tp_sl_prices(price=100.0, atr_value=2.0, side="UNKNOWN", cfg=cfg)
        assert result is None

    def test_rr_below_min_rr_returns_none(self, base_config):
        """RR below min_rr → returns None."""
        cfg_dict = dict(base_config)
        cfg_dict["sl_atr_mult"] = 3.0  # SL = 3*ATR
        cfg_dict["tp_atr_mult"] = 3.0  # TP = 3*ATR
        cfg_dict["min_rr"] = 2.0       # Require RR >= 2.0
        cfg = VolTargetConfig(**cfg_dict)
        # RR = 3/3 = 1.0 < 2.0
        result = compute_tp_sl_prices(price=100.0, atr_value=2.0, side="BUY", cfg=cfg)
        assert result is None

    def test_rr_equal_min_rr_allowed(self, base_config):
        """RR = min_rr → allowed."""
        cfg_dict = dict(base_config)
        cfg_dict["sl_atr_mult"] = 2.0  # SL = 2*ATR
        cfg_dict["tp_atr_mult"] = 2.4  # TP = 2.4*ATR
        cfg_dict["min_rr"] = 1.2       # Require RR >= 1.2
        cfg = VolTargetConfig(**cfg_dict)
        # RR = 2.4/2.0 = 1.2
        result = compute_tp_sl_prices(price=100.0, atr_value=2.0, side="BUY", cfg=cfg)
        assert result is not None
        _, _, rr = result
        assert rr == pytest.approx(1.2)

    def test_sl_price_at_zero_returns_none(self, base_config):
        """SL price would be <= 0 for BUY → returns None."""
        cfg_dict = dict(base_config)
        cfg_dict["sl_atr_mult"] = 50.0  # Large multiplier
        cfg = VolTargetConfig(**cfg_dict)
        # SL = 100 - 50*2 = 0
        result = compute_tp_sl_prices(price=100.0, atr_value=2.0, side="BUY", cfg=cfg)
        assert result is None

    def test_tp_price_negative_for_sell_returns_none(self, base_config):
        """TP price would be <= 0 for SELL → returns None."""
        cfg_dict = dict(base_config)
        cfg_dict["tp_atr_mult"] = 100.0  # Very large multiplier
        cfg = VolTargetConfig(**cfg_dict)
        # TP = 100 - 100*2 = -100
        result = compute_tp_sl_prices(price=100.0, atr_value=2.0, side="SELL", cfg=cfg)
        assert result is None

    def test_side_case_insensitive(self, base_config):
        """Side is case-insensitive."""
        cfg = VolTargetConfig(**base_config)
        result_lower = compute_tp_sl_prices(price=100.0, atr_value=2.0, side="buy", cfg=cfg)
        result_upper = compute_tp_sl_prices(price=100.0, atr_value=2.0, side="BUY", cfg=cfg)
        assert result_lower is not None
        assert result_upper is not None
        assert result_lower == result_upper


class TestGenerateVolTargetIntentTpSl:
    """Test TP/SL integration in generate_vol_target_intent (v7.3-alpha1)."""

    def test_intent_includes_tp_sl_prices(self, base_config, base_regimes, base_risk):
        """Intent includes take_profit_price and stop_loss_price at top level."""
        intent = generate_vol_target_intent(
            symbol="BTCUSDT",
            timeframe="15m",
            price=100.0,
            nav=10000.0,
            atr_value=2.0,
            regimes_snapshot=base_regimes,
            risk_snapshot=base_risk,
            trend="BULL",
            trend_aligned=True,
            strategy_cfg=base_config,
        )
        assert intent is not None
        assert "take_profit_price" in intent
        assert "stop_loss_price" in intent
        # For BUY: SL = 100 - 2*2 = 96, TP = 100 + 3*2 = 106
        assert intent["stop_loss_price"] == pytest.approx(96.0)
        assert intent["take_profit_price"] == pytest.approx(106.0)

    def test_intent_metadata_includes_tp_sl_block(self, base_config, base_regimes, base_risk):
        """Intent metadata includes tp_sl block with all fields."""
        intent = generate_vol_target_intent(
            symbol="BTCUSDT",
            timeframe="15m",
            price=100.0,
            nav=10000.0,
            atr_value=2.0,
            regimes_snapshot=base_regimes,
            risk_snapshot=base_risk,
            trend="BULL",
            trend_aligned=True,
            strategy_cfg=base_config,
        )
        assert intent is not None
        meta = intent["metadata"]["vol_target"]
        assert "tp_sl" in meta
        tp_sl = meta["tp_sl"]
        assert tp_sl["enable_tp_sl"] is True
        assert tp_sl["sl_atr_mult"] == pytest.approx(2.0)
        assert tp_sl["tp_atr_mult"] == pytest.approx(3.0)
        assert tp_sl["min_rr"] == pytest.approx(1.2)
        assert tp_sl["reward_risk"] == pytest.approx(1.5)  # 6/4
        assert tp_sl["take_profit_price"] == pytest.approx(106.0)
        assert tp_sl["stop_loss_price"] == pytest.approx(96.0)

    def test_tp_sl_disabled_intent_has_none_values(self, base_config, base_regimes, base_risk):
        """When enable_tp_sl=False, intent has None for TP/SL values."""
        cfg = dict(base_config)
        cfg["enable_tp_sl"] = False
        intent = generate_vol_target_intent(
            symbol="BTCUSDT",
            timeframe="15m",
            price=100.0,
            nav=10000.0,
            atr_value=2.0,
            regimes_snapshot=base_regimes,
            risk_snapshot=base_risk,
            trend="BULL",
            trend_aligned=True,
            strategy_cfg=cfg,
        )
        assert intent is not None
        assert intent["take_profit_price"] is None
        assert intent["stop_loss_price"] is None
        tp_sl = intent["metadata"]["vol_target"]["tp_sl"]
        assert tp_sl["take_profit_price"] is None
        assert tp_sl["stop_loss_price"] is None
        assert tp_sl["reward_risk"] is None

    def test_bad_rr_intent_has_none_tp_sl(self, base_config, base_regimes, base_risk):
        """When RR < min_rr, intent has None for TP/SL but is still generated."""
        cfg = dict(base_config)
        cfg["sl_atr_mult"] = 3.0
        cfg["tp_atr_mult"] = 3.0
        cfg["min_rr"] = 2.0  # RR = 1.0 < 2.0
        intent = generate_vol_target_intent(
            symbol="BTCUSDT",
            timeframe="15m",
            price=100.0,
            nav=10000.0,
            atr_value=2.0,
            regimes_snapshot=base_regimes,
            risk_snapshot=base_risk,
            trend="BULL",
            trend_aligned=True,
            strategy_cfg=cfg,
        )
        assert intent is not None
        assert intent["take_profit_price"] is None
        assert intent["stop_loss_price"] is None

    def test_zero_atr_no_tp_sl_but_no_intent(self, base_config, base_regimes, base_risk):
        """Zero ATR → no intent at all (not just missing TP/SL)."""
        intent = generate_vol_target_intent(
            symbol="BTCUSDT",
            timeframe="15m",
            price=100.0,
            nav=10000.0,
            atr_value=0.0,
            regimes_snapshot=base_regimes,
            risk_snapshot=base_risk,
            trend="BULL",
            trend_aligned=True,
            strategy_cfg=base_config,
        )
        assert intent is None

    def test_tp_sl_with_high_atr(self, base_config, base_regimes, base_risk):
        """High ATR → TP/SL still computed correctly."""
        intent = generate_vol_target_intent(
            symbol="BTCUSDT",
            timeframe="15m",
            price=100.0,
            nav=10000.0,
            atr_value=10.0,  # High ATR
            regimes_snapshot=base_regimes,
            risk_snapshot=base_risk,
            trend="BULL",
            trend_aligned=True,
            strategy_cfg=base_config,
        )
        assert intent is not None
        # SL = 100 - 2*10 = 80, TP = 100 + 3*10 = 130
        assert intent["stop_loss_price"] == pytest.approx(80.0)
        assert intent["take_profit_price"] == pytest.approx(130.0)

    def test_tp_sl_defaults_when_config_missing_fields(self, base_regimes, base_risk):
        """TP/SL uses defaults when config fields are missing."""
        minimal_cfg = {
            "enabled": True,
            "base_per_trade_nav_pct": 0.015,
            # Missing TP/SL fields → should use defaults
        }
        intent = generate_vol_target_intent(
            symbol="BTCUSDT",
            timeframe="15m",
            price=100.0,
            nav=10000.0,
            atr_value=2.0,
            regimes_snapshot=base_regimes,
            risk_snapshot=base_risk,
            trend="BULL",
            trend_aligned=True,
            strategy_cfg=minimal_cfg,
        )
        assert intent is not None
        # Default sl_atr_mult=2.0, tp_atr_mult=3.0
        assert intent["stop_loss_price"] == pytest.approx(96.0)
        assert intent["take_profit_price"] == pytest.approx(106.0)
