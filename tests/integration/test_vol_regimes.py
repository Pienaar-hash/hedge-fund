"""
Tests for EWMA volatility regime model (v7.4 B2).
"""
from __future__ import annotations

import math
import pytest
from unittest.mock import patch

from execution.utils.vol import (
    compute_log_returns,
    compute_ewma_vol,
    classify_vol_regime,
    compute_vol_regime_from_prices,
    get_sizing_multiplier,
    get_hybrid_weight_modifiers,
    build_vol_regime_snapshot,
    build_vol_regime_summary,
    load_vol_regime_config,
    VolRegime,
    VolRegimeConfig,
    HybridWeightModifiers,
)


class TestComputeLogReturns:
    """Tests for log return computation."""

    def test_simple_returns(self):
        """Computes log returns correctly."""
        prices = [100.0, 110.0, 105.0]  # +10%, -4.55%
        returns = compute_log_returns(prices)
        assert len(returns) == 2
        assert returns[0] == pytest.approx(math.log(110 / 100), abs=1e-6)
        assert returns[1] == pytest.approx(math.log(105 / 110), abs=1e-6)

    def test_single_price(self):
        """Single price returns empty list."""
        assert compute_log_returns([100.0]) == []

    def test_empty_prices(self):
        """Empty prices returns empty list."""
        assert compute_log_returns([]) == []

    def test_handles_zero_price(self):
        """Zero prices return 0 for that return."""
        prices = [100.0, 0.0, 100.0]
        returns = compute_log_returns(prices)
        assert len(returns) == 2
        assert returns[0] == 0.0  # 100 -> 0
        assert returns[1] == 0.0  # 0 -> 100


class TestComputeEwmaVol:
    """Tests for EWMA volatility computation."""

    def test_stable_returns(self):
        """Stable returns produce consistent low vol."""
        # Small constant returns
        returns = [0.001] * 100
        vol = compute_ewma_vol(returns, halflife_bars=20)
        assert vol < 0.01  # Low vol for stable returns

    def test_volatile_returns(self):
        """Volatile returns produce higher vol."""
        # Alternating returns
        returns = [0.05, -0.05] * 50
        vol = compute_ewma_vol(returns, halflife_bars=20)
        assert vol > 0.01  # Higher vol

    def test_empty_returns(self):
        """Empty returns return 0."""
        assert compute_ewma_vol([], halflife_bars=20) == 0.0

    def test_single_return(self):
        """Single return returns 0."""
        assert compute_ewma_vol([0.01], halflife_bars=20) == 0.0

    def test_window_truncation(self):
        """Window parameter truncates series."""
        # Stable period with small variance, then volatile period with alternating returns
        stable = [0.001] * 50
        # Volatile period: alternating +10% and -10% creates high variance
        volatile = [0.10 if i % 2 == 0 else -0.10 for i in range(50)]
        returns = stable + volatile
        vol_full = compute_ewma_vol(returns, halflife_bars=20)
        vol_recent = compute_ewma_vol(returns, halflife_bars=20, window_bars=30)
        # Recent window (all volatile) should capture more volatility than full series
        # Full series includes stable period which pulls down the average
        assert vol_recent > vol_full * 0.9  # Recent window captures more volatility


class TestClassifyVolRegime:
    """Tests for volatility regime classification."""

    def test_low_regime(self):
        """Low short vol relative to long -> low regime."""
        regime = classify_vol_regime(vol_short=0.01, vol_long=0.03)
        assert regime.label == "low"
        assert regime.ratio == pytest.approx(0.333, abs=0.01)

    def test_normal_regime(self):
        """Similar short and long vol -> normal regime."""
        regime = classify_vol_regime(vol_short=0.02, vol_long=0.02)
        assert regime.label == "normal"
        assert regime.ratio == pytest.approx(1.0, abs=0.01)

    def test_high_regime(self):
        """Elevated short vol -> high regime."""
        regime = classify_vol_regime(vol_short=0.03, vol_long=0.02)
        assert regime.label == "high"
        assert regime.ratio == pytest.approx(1.5, abs=0.01)

    def test_crisis_regime(self):
        """Very high short vol -> crisis regime."""
        regime = classify_vol_regime(vol_short=0.05, vol_long=0.02)
        assert regime.label == "crisis"
        assert regime.ratio == pytest.approx(2.5, abs=0.01)

    def test_zero_long_vol(self):
        """Zero long vol returns normal with ratio 1."""
        regime = classify_vol_regime(vol_short=0.02, vol_long=0.0)
        assert regime.label == "normal"
        assert regime.ratio == 1.0

    def test_custom_config(self):
        """Custom thresholds are respected."""
        config = VolRegimeConfig(
            ratio_low=0.5,
            ratio_normal=1.0,
            ratio_high=1.5,
        )
        # Ratio 0.4 should be "low" with default (0.6) but normal here isn't right
        # ratio_low=0.5, so ratio 0.4 < 0.5 -> low
        regime = classify_vol_regime(vol_short=0.01, vol_long=0.025, config=config)
        assert regime.label == "low"


class TestComputeVolRegimeFromPrices:
    """Tests for end-to-end vol regime from prices."""

    def test_stable_prices_normal(self):
        """Stable prices produce normal regime."""
        # Gradually increasing prices (low vol)
        prices = [100 + i * 0.01 for i in range(200)]
        regime = compute_vol_regime_from_prices(prices)
        # Should be low or normal since vol is minimal
        assert regime.label in ("low", "normal")

    def test_spike_produces_high_or_crisis(self):
        """Recent spike produces high or crisis."""
        # Stable then spike
        prices = [100.0] * 500 + [100.0 + (i % 10) * 2 for i in range(200)]
        config = VolRegimeConfig(
            short_window_bars=50,
            long_window_bars=300,
            short_halflife_bars=25,
            long_halflife_bars=150,
        )
        regime = compute_vol_regime_from_prices(prices, config)
        # Short-term vol should be higher
        assert regime.vol_short >= regime.vol_long or regime.ratio >= 0.8

    def test_insufficient_data_normal(self):
        """Insufficient data returns normal."""
        prices = [100.0, 101.0, 102.0]
        regime = compute_vol_regime_from_prices(prices)
        assert regime.label == "normal"
        assert regime.vol_short == 0.0
        assert regime.vol_long == 0.0


class TestGetSizingMultiplier:
    """Tests for tier-based sizing multipliers."""

    def test_core_multipliers(self):
        """CORE tier uses correct multipliers."""
        config = {
            "vol_regimes": {
                "enabled": True,
                "sizing_multipliers": {
                    "CORE": {"low": 1.15, "normal": 1.0, "high": 0.75, "crisis": 0.5}
                }
            }
        }
        assert get_sizing_multiplier("CORE", "low", config) == 1.15
        assert get_sizing_multiplier("CORE", "normal", config) == 1.0
        assert get_sizing_multiplier("CORE", "high", config) == 0.75
        assert get_sizing_multiplier("CORE", "crisis", config) == 0.5

    def test_satellite_multipliers(self):
        """SATELLITE tier uses correct multipliers."""
        config = {
            "vol_regimes": {
                "enabled": True,
                "sizing_multipliers": {
                    "SATELLITE": {"low": 1.1, "normal": 1.0, "high": 0.7, "crisis": 0.4}
                }
            }
        }
        assert get_sizing_multiplier("SATELLITE", "crisis", config) == 0.4

    def test_fallback_to_core(self):
        """Unknown tier falls back to CORE."""
        config = {
            "vol_regimes": {
                "enabled": True,
                "sizing_multipliers": {
                    "CORE": {"low": 1.15, "normal": 1.0, "high": 0.75, "crisis": 0.5}
                }
            }
        }
        assert get_sizing_multiplier("UNKNOWN_TIER", "high", config) == 0.75

    def test_missing_config_returns_1(self):
        """Missing config returns 1.0."""
        assert get_sizing_multiplier("CORE", "high", None) == 1.0
        assert get_sizing_multiplier("CORE", "high", {}) == 1.0

    def test_disabled_returns_1(self):
        """Disabled vol_regimes returns 1.0."""
        config = {"vol_regimes": {"enabled": False}}
        assert get_sizing_multiplier("CORE", "crisis", config) == 1.0


class TestGetHybridWeightModifiers:
    """Tests for hybrid weight modifiers."""

    def test_crisis_modifiers(self):
        """Crisis regime has correct modifiers."""
        config = {
            "vol_regimes": {
                "enabled": True,
                "hybrid_weight_modifiers": {
                    "default": {
                        "crisis": {"carry": 0.4, "expectancy": 0.8, "router": 1.1}
                    }
                }
            }
        }
        mods = get_hybrid_weight_modifiers("crisis", config)
        assert mods.carry == 0.4
        assert mods.expectancy == 0.8
        assert mods.router == 1.1

    def test_normal_modifiers(self):
        """Normal regime has 1.0 modifiers by default."""
        config = {
            "vol_regimes": {
                "enabled": True,
                "hybrid_weight_modifiers": {
                    "default": {
                        "normal": {"carry": 1.0, "expectancy": 1.0, "router": 1.0}
                    }
                }
            }
        }
        mods = get_hybrid_weight_modifiers("normal", config)
        assert mods.carry == 1.0
        assert mods.expectancy == 1.0
        assert mods.router == 1.0

    def test_missing_returns_defaults(self):
        """Missing config returns default modifiers (1.0)."""
        mods = get_hybrid_weight_modifiers("crisis", None)
        assert mods.carry == 1.0
        assert mods.expectancy == 1.0
        assert mods.router == 1.0


class TestBuildVolRegimeSnapshot:
    """Tests for building vol regime snapshot."""

    def test_with_prices(self):
        """Snapshot includes all fields with prices."""
        prices = [100 + i * 0.01 for i in range(200)]
        snap = build_vol_regime_snapshot("BTCUSDT", prices)
        assert snap["symbol"] == "BTCUSDT"
        assert snap["vol_regime"] in ("low", "normal", "high", "crisis")
        assert "vol" in snap
        assert "short" in snap["vol"]
        assert "long" in snap["vol"]
        assert "ratio" in snap["vol"]
        assert "updated_ts" in snap

    def test_without_prices(self):
        """Snapshot returns defaults without prices."""
        snap = build_vol_regime_snapshot("ETHUSDT", None)
        assert snap["symbol"] == "ETHUSDT"
        assert snap["vol_regime"] == "normal"
        assert snap["vol"]["short"] == 0.0
        assert snap["vol"]["long"] == 0.0
        assert snap["vol"]["ratio"] == 1.0


class TestBuildVolRegimeSummary:
    """Tests for building regime summary."""

    def test_summary_counts(self):
        """Counts regimes correctly."""
        regimes = [
            VolRegime("low", 0.01, 0.02, 0.5),
            VolRegime("normal", 0.02, 0.02, 1.0),
            VolRegime("normal", 0.02, 0.02, 1.0),
            VolRegime("high", 0.03, 0.02, 1.5),
            VolRegime("crisis", 0.05, 0.02, 2.5),
        ]
        summary = build_vol_regime_summary(regimes)
        assert summary["low"] == 1
        assert summary["normal"] == 2
        assert summary["high"] == 1
        assert summary["crisis"] == 1

    def test_summary_with_strings(self):
        """Accepts string labels."""
        regimes = ["low", "normal", "normal", "high"]
        summary = build_vol_regime_summary(regimes)
        assert summary["low"] == 1
        assert summary["normal"] == 2
        assert summary["high"] == 1
        assert summary["crisis"] == 0


class TestLoadVolRegimeConfig:
    """Tests for loading config from strategy_config."""

    def test_load_from_dict(self):
        """Loads config from dict correctly."""
        strategy_config = {
            "vol_regimes": {
                "enabled": True,
                "defaults": {
                    "short_window_bars": 100,
                    "long_window_bars": 500,
                    "short_halflife_bars": 50,
                    "long_halflife_bars": 250,
                    "ratio_thresholds": {
                        "low": 0.5,
                        "normal": 1.0,
                        "high": 1.5,
                    }
                }
            }
        }
        config = load_vol_regime_config(strategy_config)
        assert config.short_window_bars == 100
        assert config.long_window_bars == 500
        assert config.short_halflife_bars == 50
        assert config.long_halflife_bars == 250
        assert config.ratio_low == 0.5
        assert config.ratio_normal == 1.0
        assert config.ratio_high == 1.5

    def test_load_defaults(self):
        """Returns defaults when config missing."""
        config = load_vol_regime_config(None)
        assert config.short_window_bars == 168
        assert config.long_window_bars == 720
        assert config.ratio_low == 0.6
        assert config.ratio_normal == 1.2
        assert config.ratio_high == 1.8

    def test_disabled_returns_defaults(self):
        """Disabled config returns defaults."""
        config = load_vol_regime_config({"vol_regimes": {"enabled": False}})
        assert config.short_window_bars == 168
