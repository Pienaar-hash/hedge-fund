"""Tests for P6 Candidate 1 — Regime-Permissioned Simple Rules."""

import pytest
import time

from execution.p6_simple_rules import (
    C1Config,
    DEFAULT_CONFIG,
    P6Signal,
    compute_ema,
    compute_mr_conviction,
    compute_trend_conviction,
    compute_zscore,
    generate_simple_rule_signals,
    _trend_signals,
    _mr_signals,
    _suppress_per_symbol,
)

# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def bullish_closes():
    """Closes that produce EMA-fast > EMA-slow (trending up)."""
    # Start flat then ramp up
    flat = [100.0] * 60
    ramp = [100.0 + i * 2.0 for i in range(1, 41)]
    return flat + ramp  # 100 bars total

@pytest.fixture
def bearish_closes():
    """Closes that produce EMA-fast < EMA-slow (trending down)."""
    flat = [100.0] * 60
    ramp = [100.0 - i * 2.0 for i in range(1, 41)]
    return flat + ramp

@pytest.fixture
def mr_closes_oversold():
    """Closes that produce negative z-score (price below mean)."""
    # Flat then sharp dip
    flat = [100.0] * 50
    dip = [100.0 - i * 3.0 for i in range(1, 11)]
    return flat + dip

@pytest.fixture
def mr_closes_overbought():
    """Closes that produce positive z-score (price above mean)."""
    flat = [100.0] * 50
    spike = [100.0 + i * 3.0 for i in range(1, 11)]
    return flat + spike

@pytest.fixture
def trend_up_features():
    return {"trend_r2": 0.8, "trend_slope": 0.001, "range_position": 0.6}

@pytest.fixture
def trend_down_features():
    return {"trend_r2": 0.7, "trend_slope": -0.001, "range_position": 0.4}

@pytest.fixture
def mr_features_low():
    return {"range_position": 0.1, "mean_reversion_score": 0.6}

@pytest.fixture
def mr_features_high():
    return {"range_position": 0.9, "mean_reversion_score": 0.6}

TS = 1700000000.0


# ── Config tests ─────────────────────────────────────────────────────────

class TestC1Config:
    def test_frozen(self):
        cfg = C1Config()
        with pytest.raises(AttributeError):
            cfg.ema_fast_period = 20  # type: ignore[misc]

    def test_defaults(self):
        cfg = DEFAULT_CONFIG
        assert cfg.ema_fast_period == 15
        assert cfg.ema_slow_period == 50
        assert cfg.trend_r2_min == 0.5
        assert cfg.zscore_lookback == 48
        assert cfg.zscore_entry == 1.5
        assert cfg.range_pos_long_max == 0.2
        assert cfg.range_pos_short_min == 0.8

    def test_to_dict_roundtrip(self):
        d = DEFAULT_CONFIG.to_dict()
        assert isinstance(d, dict)
        assert d["ema_fast_period"] == 15
        assert d["zscore_entry"] == 1.5

    def test_conviction_coefficients_present(self):
        cfg = DEFAULT_CONFIG
        assert cfg.conv_k1 == 0.30
        assert cfg.conv_k2 == 0.20
        assert cfg.conv_k3 == 0.25
        assert cfg.conv_k4 == 0.20
        assert cfg.conv_z_cap == 4.0


# ── Conviction proxy tests ───────────────────────────────────────────────

class TestTrendConviction:
    def test_baseline_at_r2_half_no_spread(self):
        """r2=0.5, no EMA spread → conviction = 0.5."""
        c = compute_trend_conviction(0.5, 100.0, 100.0, 100.0, DEFAULT_CONFIG)
        assert c == pytest.approx(0.5, abs=1e-6)

    def test_high_r2_increases_conviction(self):
        c = compute_trend_conviction(0.9, 100.0, 100.0, 100.0, DEFAULT_CONFIG)
        assert c > 0.5

    def test_low_r2_decreases_conviction(self):
        c = compute_trend_conviction(0.1, 100.0, 100.0, 100.0, DEFAULT_CONFIG)
        assert c < 0.5

    def test_ema_spread_increases_conviction(self):
        c = compute_trend_conviction(0.5, 110.0, 100.0, 100.0, DEFAULT_CONFIG)
        assert c > 0.5

    def test_clipped_at_zero(self):
        c = compute_trend_conviction(0.0, 50.0, 100.0, 100.0, DEFAULT_CONFIG)
        assert c >= 0.0

    def test_clipped_at_one(self):
        c = compute_trend_conviction(1.0, 200.0, 100.0, 100.0, DEFAULT_CONFIG)
        assert c <= 1.0

    def test_zero_price_returns_half(self):
        c = compute_trend_conviction(0.8, 110.0, 100.0, 0.0, DEFAULT_CONFIG)
        assert c == 0.5

    def test_deterministic(self):
        args = (0.8, 105.0, 100.0, 100.0, DEFAULT_CONFIG)
        assert compute_trend_conviction(*args) == compute_trend_conviction(*args)


class TestMRConviction:
    def test_baseline_zero_zscore_half_rp(self):
        c = compute_mr_conviction(0.0, 0.5, DEFAULT_CONFIG)
        assert c == pytest.approx(0.5, abs=1e-6)

    def test_high_zscore_increases_conviction(self):
        c = compute_mr_conviction(3.0, 0.5, DEFAULT_CONFIG)
        assert c > 0.5

    def test_extreme_rp_increases_conviction(self):
        c = compute_mr_conviction(0.0, 0.1, DEFAULT_CONFIG)
        assert c > 0.5

    def test_z_capped(self):
        """z-score beyond z_cap should not keep increasing."""
        c1 = compute_mr_conviction(4.0, 0.5, DEFAULT_CONFIG)
        c2 = compute_mr_conviction(10.0, 0.5, DEFAULT_CONFIG)
        assert c1 == pytest.approx(c2, abs=1e-6)

    def test_clipped_bounds(self):
        c = compute_mr_conviction(10.0, 0.0, DEFAULT_CONFIG)
        assert 0.0 <= c <= 1.0

    def test_deterministic(self):
        args = (-2.0, 0.1, DEFAULT_CONFIG)
        assert compute_mr_conviction(*args) == compute_mr_conviction(*args)


# ── EMA tests ────────────────────────────────────────────────────────────

class TestComputeEMA:
    def test_empty(self):
        assert compute_ema([], 10) == []

    def test_single(self):
        assert compute_ema([42.0], 10) == [42.0]

    def test_length_preserved(self):
        prices = [float(i) for i in range(100)]
        ema = compute_ema(prices, 15)
        assert len(ema) == len(prices)

    def test_ema_trails_in_uptrend(self):
        prices = [float(i) for i in range(100)]
        ema = compute_ema(prices, 15)
        # EMA should lag behind price in uptrend
        assert ema[-1] < prices[-1]

    def test_ema_trails_in_downtrend(self):
        prices = [100.0 - i for i in range(100)]
        ema = compute_ema(prices, 15)
        assert ema[-1] > prices[-1]


# ── Z-score tests ────────────────────────────────────────────────────────

class TestComputeZscore:
    def test_flat_prices(self):
        assert compute_zscore([100.0] * 50, 48) == 0.0

    def test_single_price(self):
        assert compute_zscore([100.0], 48) == 0.0

    def test_positive_spike(self):
        prices = [100.0] * 49 + [200.0]
        z = compute_zscore(prices, 48)
        assert z > 1.0

    def test_negative_dip(self):
        prices = [100.0] * 49 + [50.0]
        z = compute_zscore(prices, 48)
        assert z < -1.0

    def test_short_window(self):
        z = compute_zscore([1.0, 2.0, 3.0], 48)
        # Should use all 3 prices since < lookback
        assert isinstance(z, float)


# ── Trend signal tests ───────────────────────────────────────────────────

class TestTrendSignals:
    def test_trend_up_bullish_produces_normal_long(self, bullish_closes, trend_up_features):
        sigs = _trend_signals("BTCUSDT", bullish_closes, "TREND_UP", trend_up_features, DEFAULT_CONFIG, TS)
        normal = [s for s in sigs if s.candidate_id == "C1_TREND_NORMAL"]
        assert len(normal) == 1
        assert normal[0].side == "LONG"
        assert normal[0].polarity == "normal"

    def test_trend_up_bullish_produces_inverted_short(self, bullish_closes, trend_up_features):
        sigs = _trend_signals("BTCUSDT", bullish_closes, "TREND_UP", trend_up_features, DEFAULT_CONFIG, TS)
        inv = [s for s in sigs if s.candidate_id == "C1_TREND_INVERTED"]
        assert len(inv) == 1
        assert inv[0].side == "SHORT"
        assert inv[0].polarity == "inverted"

    def test_trend_down_bearish_produces_normal_short(self, bearish_closes, trend_down_features):
        sigs = _trend_signals("BTCUSDT", bearish_closes, "TREND_DOWN", trend_down_features, DEFAULT_CONFIG, TS)
        normal = [s for s in sigs if s.candidate_id == "C1_TREND_NORMAL"]
        assert len(normal) == 1
        assert normal[0].side == "SHORT"

    def test_trend_down_bearish_produces_inverted_long(self, bearish_closes, trend_down_features):
        sigs = _trend_signals("BTCUSDT", bearish_closes, "TREND_DOWN", trend_down_features, DEFAULT_CONFIG, TS)
        inv = [s for s in sigs if s.candidate_id == "C1_TREND_INVERTED"]
        assert len(inv) == 1
        assert inv[0].side == "LONG"

    def test_wrong_regime_no_signal(self, bullish_closes, trend_up_features):
        sigs = _trend_signals("BTCUSDT", bullish_closes, "MEAN_REVERT", trend_up_features, DEFAULT_CONFIG, TS)
        assert len(sigs) == 0

    def test_low_r2_no_signal(self, bullish_closes):
        features = {"trend_r2": 0.3, "trend_slope": 0.001}
        sigs = _trend_signals("BTCUSDT", bullish_closes, "TREND_UP", features, DEFAULT_CONFIG, TS)
        assert len(sigs) == 0

    def test_insufficient_data(self, trend_up_features):
        sigs = _trend_signals("BTCUSDT", [100.0] * 10, "TREND_UP", trend_up_features, DEFAULT_CONFIG, TS)
        assert len(sigs) == 0

    def test_feature_snapshot_present(self, bullish_closes, trend_up_features):
        sigs = _trend_signals("BTCUSDT", bullish_closes, "TREND_UP", trend_up_features, DEFAULT_CONFIG, TS)
        for s in sigs:
            assert "ema_fast" in s.feature_snapshot
            assert "ema_slow" in s.feature_snapshot
            assert "trend_r2" in s.feature_snapshot
            assert "conviction_proxy" in s.feature_snapshot
            assert s.feature_snapshot["regime"] == "TREND_UP"

    def test_conviction_proxy_set(self, bullish_closes, trend_up_features):
        sigs = _trend_signals("BTCUSDT", bullish_closes, "TREND_UP", trend_up_features, DEFAULT_CONFIG, TS)
        for s in sigs:
            assert s.conviction > 0.5  # r2=0.8 + positive spread → >0.5
            assert s.conviction <= 1.0

    def test_all_signals_have_correct_metadata(self, bullish_closes, trend_up_features):
        sigs = _trend_signals("BTCUSDT", bullish_closes, "TREND_UP", trend_up_features, DEFAULT_CONFIG, TS)
        for s in sigs:
            assert s.candidate_family == "C1"
            assert s.rule_name == "trend_ema_crossover"
            assert s.symbol == "BTCUSDT"
            assert s.ts == TS


# ── MR signal tests ──────────────────────────────────────────────────────

class TestMRSignals:
    def test_mr_oversold_produces_normal_long(self, mr_closes_oversold, mr_features_low):
        sigs = _mr_signals("ETHUSDT", mr_closes_oversold, "MEAN_REVERT", mr_features_low, DEFAULT_CONFIG, TS)
        normal = [s for s in sigs if s.candidate_id == "C1_MR_NORMAL"]
        assert len(normal) == 1
        assert normal[0].side == "LONG"

    def test_mr_oversold_produces_inverted_short(self, mr_closes_oversold, mr_features_low):
        sigs = _mr_signals("ETHUSDT", mr_closes_oversold, "MEAN_REVERT", mr_features_low, DEFAULT_CONFIG, TS)
        inv = [s for s in sigs if s.candidate_id == "C1_MR_INVERTED"]
        assert len(inv) == 1
        assert inv[0].side == "SHORT"

    def test_mr_overbought_produces_normal_short(self, mr_closes_overbought, mr_features_high):
        sigs = _mr_signals("ETHUSDT", mr_closes_overbought, "MEAN_REVERT", mr_features_high, DEFAULT_CONFIG, TS)
        normal = [s for s in sigs if s.candidate_id == "C1_MR_NORMAL"]
        assert len(normal) == 1
        assert normal[0].side == "SHORT"

    def test_mr_overbought_produces_inverted_long(self, mr_closes_overbought, mr_features_high):
        sigs = _mr_signals("ETHUSDT", mr_closes_overbought, "MEAN_REVERT", mr_features_high, DEFAULT_CONFIG, TS)
        inv = [s for s in sigs if s.candidate_id == "C1_MR_INVERTED"]
        assert len(inv) == 1
        assert inv[0].side == "LONG"

    def test_non_mr_regime_no_signal(self, mr_closes_oversold, mr_features_low):
        sigs = _mr_signals("ETHUSDT", mr_closes_oversold, "TREND_UP", mr_features_low, DEFAULT_CONFIG, TS)
        assert len(sigs) == 0

    def test_range_position_too_high_for_long(self, mr_closes_oversold):
        features = {"range_position": 0.5, "mean_reversion_score": 0.6}
        sigs = _mr_signals("ETHUSDT", mr_closes_oversold, "MEAN_REVERT", features, DEFAULT_CONFIG, TS)
        # z-score is negative but range_position > 0.2 → no long trigger
        assert len(sigs) == 0

    def test_feature_snapshot_present(self, mr_closes_oversold, mr_features_low):
        sigs = _mr_signals("ETHUSDT", mr_closes_oversold, "MEAN_REVERT", mr_features_low, DEFAULT_CONFIG, TS)
        for s in sigs:
            assert "zscore" in s.feature_snapshot
            assert "range_position" in s.feature_snapshot
            assert "conviction_proxy" in s.feature_snapshot

    def test_conviction_proxy_set(self, mr_closes_oversold, mr_features_low):
        sigs = _mr_signals("ETHUSDT", mr_closes_oversold, "MEAN_REVERT", mr_features_low, DEFAULT_CONFIG, TS)
        for s in sigs:
            assert s.conviction > 0.5  # extreme zscore + low range → >0.5
            assert s.conviction <= 1.0


# ── Suppression tests ────────────────────────────────────────────────────

class TestSuppression:
    def test_single_signal_passes(self):
        sig = P6Signal(
            candidate_id="C1_TREND_NORMAL", candidate_family="C1",
            rule_name="trend_ema_crossover", symbol="BTCUSDT",
            side="LONG", polarity="normal", regime="TREND_UP",
            feature_snapshot={}, ts=TS,
        )
        selected, suppressed = _suppress_per_symbol([sig])
        assert len(selected) == 1
        assert len(suppressed) == 0
        assert selected[0].selected_for_eval

    def test_trend_beats_mr_same_symbol_same_candidate(self):
        trend = P6Signal(
            candidate_id="C1_MR_NORMAL", candidate_family="C1",
            rule_name="trend_ema_crossover", symbol="BTCUSDT",
            side="LONG", polarity="normal", regime="TREND_UP",
            feature_snapshot={}, ts=TS,
        )
        mr = P6Signal(
            candidate_id="C1_MR_NORMAL", candidate_family="C1",
            rule_name="mr_zscore_range", symbol="BTCUSDT",
            side="LONG", polarity="normal", regime="TREND_UP",
            feature_snapshot={}, ts=TS,
        )
        selected, suppressed = _suppress_per_symbol([mr, trend])
        assert len(selected) == 1
        assert selected[0].rule_name == "trend_ema_crossover"
        assert len(suppressed) == 1
        assert not suppressed[0].selected_for_eval

    def test_different_candidate_ids_not_suppressed(self):
        s1 = P6Signal(
            candidate_id="C1_TREND_NORMAL", candidate_family="C1",
            rule_name="trend_ema_crossover", symbol="BTCUSDT",
            side="LONG", polarity="normal", regime="TREND_UP",
            feature_snapshot={}, ts=TS,
        )
        s2 = P6Signal(
            candidate_id="C1_TREND_INVERTED", candidate_family="C1",
            rule_name="trend_ema_crossover", symbol="BTCUSDT",
            side="SHORT", polarity="inverted", regime="TREND_UP",
            feature_snapshot={}, ts=TS,
        )
        selected, suppressed = _suppress_per_symbol([s1, s2])
        assert len(selected) == 2
        assert len(suppressed) == 0

    def test_different_symbols_not_suppressed(self):
        s1 = P6Signal(
            candidate_id="C1_TREND_NORMAL", candidate_family="C1",
            rule_name="trend_ema_crossover", symbol="BTCUSDT",
            side="LONG", polarity="normal", regime="TREND_UP",
            feature_snapshot={}, ts=TS,
        )
        s2 = P6Signal(
            candidate_id="C1_TREND_NORMAL", candidate_family="C1",
            rule_name="trend_ema_crossover", symbol="ETHUSDT",
            side="LONG", polarity="normal", regime="TREND_UP",
            feature_snapshot={}, ts=TS,
        )
        selected, suppressed = _suppress_per_symbol([s1, s2])
        assert len(selected) == 2
        assert len(suppressed) == 0

    def test_empty_input(self):
        selected, suppressed = _suppress_per_symbol([])
        assert selected == []
        assert suppressed == []


# ── Integration: generate_simple_rule_signals ────────────────────────────

class TestGenerateSimpleRuleSignals:
    def test_trend_up_produces_two_signals(self, bullish_closes, trend_up_features):
        selected, suppressed = generate_simple_rule_signals(
            bullish_closes, trend_up_features, "TREND_UP", "BTCUSDT", ts=TS,
        )
        assert len(selected) == 2  # NORMAL + INVERTED
        ids = {s.candidate_id for s in selected}
        assert "C1_TREND_NORMAL" in ids
        assert "C1_TREND_INVERTED" in ids

    def test_mean_revert_produces_two_signals(self, mr_closes_oversold, mr_features_low):
        selected, suppressed = generate_simple_rule_signals(
            mr_closes_oversold, mr_features_low, "MEAN_REVERT", "ETHUSDT", ts=TS,
        )
        assert len(selected) == 2
        ids = {s.candidate_id for s in selected}
        assert "C1_MR_NORMAL" in ids
        assert "C1_MR_INVERTED" in ids

    def test_choppy_regime_no_signals(self, bullish_closes, trend_up_features):
        selected, suppressed = generate_simple_rule_signals(
            bullish_closes, trend_up_features, "CHOPPY", "BTCUSDT", ts=TS,
        )
        assert len(selected) == 0
        assert len(suppressed) == 0

    def test_crisis_regime_no_signals(self, bullish_closes, trend_up_features):
        selected, suppressed = generate_simple_rule_signals(
            bullish_closes, trend_up_features, "CRISIS", "BTCUSDT", ts=TS,
        )
        assert len(selected) == 0

    def test_signal_to_dict_roundtrip(self, bullish_closes, trend_up_features):
        selected, _ = generate_simple_rule_signals(
            bullish_closes, trend_up_features, "TREND_UP", "BTCUSDT", ts=TS,
        )
        for s in selected:
            d = s.to_dict()
            assert isinstance(d, dict)
            assert d["candidate_id"].startswith("C1_")
            assert d["ts"] == TS

    def test_default_config_used_when_none(self, bullish_closes, trend_up_features):
        s1, _ = generate_simple_rule_signals(
            bullish_closes, trend_up_features, "TREND_UP", "BTCUSDT", ts=TS,
        )
        s2, _ = generate_simple_rule_signals(
            bullish_closes, trend_up_features, "TREND_UP", "BTCUSDT",
            config=DEFAULT_CONFIG, ts=TS,
        )
        # Same config → same signals
        assert len(s1) == len(s2)

    def test_custom_config(self, bullish_closes, trend_up_features):
        # Very high R2 threshold → no signals
        strict = C1Config(trend_r2_min=0.99)
        selected, _ = generate_simple_rule_signals(
            bullish_closes, trend_up_features, "TREND_UP", "BTCUSDT",
            config=strict, ts=TS,
        )
        assert len(selected) == 0

    def test_timestamp_auto_generated(self, bullish_closes, trend_up_features):
        before = time.time()
        selected, _ = generate_simple_rule_signals(
            bullish_closes, trend_up_features, "TREND_UP", "BTCUSDT",
        )
        after = time.time()
        for s in selected:
            assert before <= s.ts <= after

    def test_polarity_flip_correctness(self, bullish_closes, trend_up_features):
        """Normal and inverted must have opposite sides for same trigger."""
        selected, _ = generate_simple_rule_signals(
            bullish_closes, trend_up_features, "TREND_UP", "BTCUSDT", ts=TS,
        )
        normal = [s for s in selected if s.polarity == "normal"][0]
        inverted = [s for s in selected if s.polarity == "inverted"][0]
        assert normal.side != inverted.side
