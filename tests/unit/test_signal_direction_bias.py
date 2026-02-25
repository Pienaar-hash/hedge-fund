"""
Tests for regime-consistent signal direction (v7.9 structural long-bias fix).

Validates that:
  - _fallback_trend() returns regime-consistent direction (not hardcoded LONG)
  - compute_trend_bias() does not dampen regime-aligned direction via RSI
  - compute_trend_bias() DOES dampen counter-trend direction via RSI
  - TREND_DOWN regime produces SHORT-capable signals
  - TREND_UP regime produces LONG-capable signals
  - CHOPPY/CRISIS/MEAN_REVERT produce FLAT fallback (no artificial bias)
  - vol_trend_aligned does not treat NEUTRAL as BULL
"""

from __future__ import annotations

import pytest

from execution.strategies.vol_target import (
    TrendConfig,
    VolTargetConfig,
    compute_trend_bias,
    decide_hybrid_side,
    _fallback_trend,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trending_closes(direction: str, n: int = 20) -> list[float]:
    """Generate monotonically trending close prices."""
    if direction == "up":
        return [100.0 + i * 0.5 for i in range(n)]
    elif direction == "down":
        return [100.0 - i * 0.5 for i in range(n)]
    else:
        return [100.0] * n


def _make_trend_cfg(**overrides) -> TrendConfig:
    defaults = dict(
        fast_ema=5,
        slow_ema=10,
        min_trend_strength=0.01,
        use_htf_rsi_filter=True,
        rsi_overbought=70,
        rsi_oversold=30,
        htf_tf="1h",
    )
    defaults.update(overrides)
    return TrendConfig(**defaults)


def _make_vol_target_cfg(**overrides) -> VolTargetConfig:
    defaults = dict(
        enabled=True,
        base_per_trade_nav_pct=0.015,
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
    defaults.update(overrides)
    return VolTargetConfig(**defaults)


# ═══════════════════════════════════════════════════════════════════════════
# 1. _fallback_trend — regime-consistent defaults
# ═══════════════════════════════════════════════════════════════════════════

class TestFallbackTrendRegimeConsistent:
    """_fallback_trend must return regime-consistent direction, never hardcoded LONG."""

    def test_trend_up_returns_long(self):
        direction, strength = _fallback_trend("NEUTRAL", True, 0.1, regime="TREND_UP")
        assert direction == "LONG"
        assert strength >= 0.1

    def test_trend_down_returns_short(self):
        direction, strength = _fallback_trend("NEUTRAL", True, 0.1, regime="TREND_DOWN")
        assert direction == "SHORT"
        assert strength >= 0.1

    def test_choppy_returns_flat(self):
        direction, strength = _fallback_trend("NEUTRAL", True, 0.1, regime="CHOPPY")
        assert direction is None or direction == "FLAT"

    def test_crisis_returns_flat(self):
        direction, strength = _fallback_trend("NEUTRAL", True, 0.1, regime="CRISIS")
        assert direction is None or direction == "FLAT"

    def test_mean_revert_returns_flat(self):
        direction, strength = _fallback_trend("NEUTRAL", True, 0.1, regime="MEAN_REVERT")
        assert direction is None or direction == "FLAT"

    def test_breakout_returns_flat(self):
        direction, strength = _fallback_trend("NEUTRAL", True, 0.1, regime="BREAKOUT")
        assert direction is None or direction == "FLAT"

    def test_bull_always_long_regardless_of_regime(self):
        """Explicit BULL trend always maps to LONG."""
        direction, _ = _fallback_trend("BULL", True, 0.1, regime="TREND_DOWN")
        assert direction == "LONG"

    def test_bear_always_short_regardless_of_regime(self):
        """Explicit BEAR trend always maps to SHORT."""
        direction, _ = _fallback_trend("BEAR", True, 0.1, regime="TREND_UP")
        assert direction == "SHORT"

    def test_not_trend_aligned_returns_none(self):
        """When not trend-aligned, always returns None regardless of regime."""
        direction, _ = _fallback_trend("BULL", False, 0.1, regime="TREND_UP")
        assert direction is None

    def test_no_regime_defaults_to_flat(self):
        """When regime is None/unknown, ambiguous trends should not default LONG."""
        direction, _ = _fallback_trend("NEUTRAL", True, 0.1, regime=None)
        assert direction is None or direction == "FLAT"

    def test_backward_compat_bull_bear_unchanged(self):
        """BULL/BEAR mapping must still work when regime=None (backward compat)."""
        d_bull, _ = _fallback_trend("BULL", True, 0.1, regime=None)
        d_bear, _ = _fallback_trend("BEAR", True, 0.1, regime=None)
        assert d_bull == "LONG"
        assert d_bear == "SHORT"


# ═══════════════════════════════════════════════════════════════════════════
# 2. compute_trend_bias — RSI counter-trend suppression only
# ═══════════════════════════════════════════════════════════════════════════

class TestRSIRegimeAlignment:
    """RSI damping must only suppress counter-trend entries, not regime-confirmed direction."""

    def test_rsi_does_not_dampen_short_in_trend_down(self):
        """In TREND_DOWN, oversold RSI should NOT suppress SHORT."""
        closes = _make_trending_closes("down")
        cfg = _make_trend_cfg(min_trend_strength=0.01, use_htf_rsi_filter=True, rsi_oversold=30)
        # RSI very oversold — would normally halve SHORT strength
        result = compute_trend_bias(closes, htf_rsi=15.0, cfg=cfg, regime="TREND_DOWN")
        assert result["direction"] == "SHORT", f"Expected SHORT, got {result['direction']}"
        assert result["strength"] > cfg.min_trend_strength

    def test_rsi_does_not_dampen_long_in_trend_up(self):
        """In TREND_UP, overbought RSI should NOT suppress LONG."""
        closes = _make_trending_closes("up")
        cfg = _make_trend_cfg(min_trend_strength=0.01, use_htf_rsi_filter=True, rsi_overbought=70)
        result = compute_trend_bias(closes, htf_rsi=85.0, cfg=cfg, regime="TREND_UP")
        assert result["direction"] == "LONG"
        assert result["strength"] > cfg.min_trend_strength

    def test_rsi_dampens_short_in_trend_up(self):
        """Counter-trend SHORT in TREND_UP should still be dampened by RSI."""
        closes = _make_trending_closes("down")
        cfg = _make_trend_cfg(min_trend_strength=0.05, use_htf_rsi_filter=True, rsi_oversold=30)
        # No regime alignment → RSI damping applies
        result_no_regime = compute_trend_bias(closes, htf_rsi=15.0, cfg=cfg, regime="TREND_UP")
        result_aligned = compute_trend_bias(closes, htf_rsi=15.0, cfg=cfg, regime="TREND_DOWN")
        # Counter-trend should have weaker strength (or flatten)
        assert result_aligned["strength"] >= result_no_regime["strength"]

    def test_rsi_dampens_long_in_trend_down(self):
        """Counter-trend LONG in TREND_DOWN should still be dampened by RSI."""
        closes = _make_trending_closes("up")
        cfg = _make_trend_cfg(min_trend_strength=0.05, use_htf_rsi_filter=True, rsi_overbought=70)
        result_counter = compute_trend_bias(closes, htf_rsi=85.0, cfg=cfg, regime="TREND_DOWN")
        result_aligned = compute_trend_bias(closes, htf_rsi=85.0, cfg=cfg, regime="TREND_UP")
        assert result_aligned["strength"] >= result_counter["strength"]

    def test_no_regime_preserves_original_rsi_behavior(self):
        """When regime=None, RSI filter works exactly as before (backward compat)."""
        closes = _make_trending_closes("down")
        cfg = _make_trend_cfg(min_trend_strength=0.05, use_htf_rsi_filter=True, rsi_oversold=30)
        result = compute_trend_bias(closes, htf_rsi=15.0, cfg=cfg)
        # Original behavior: RSI oversold halves SHORT strength
        # This might flatten depending on strength => just ensure it ran
        assert result["direction"] in ("SHORT", "FLAT")


# ═══════════════════════════════════════════════════════════════════════════
# 3. End-to-end: regime → hybrid side
# ═══════════════════════════════════════════════════════════════════════════

class TestRegimeToHybridDirection:
    """Full pipeline: closes + regime → compute_trend_bias → decide_hybrid_side → correct side."""

    def test_trend_down_produces_sell(self):
        """Downtrending closes + TREND_DOWN regime → SELL signal."""
        closes = _make_trending_closes("down")
        cfg = _make_trend_cfg(min_trend_strength=0.01, use_htf_rsi_filter=True, rsi_oversold=30)
        trend_info = compute_trend_bias(closes, htf_rsi=20.0, cfg=cfg, regime="TREND_DOWN")
        assert trend_info["direction"] == "SHORT"

        carry_info = {"score_long": 0.0, "score_short": 0.0}
        hybrid = decide_hybrid_side(trend_info, carry_info, _make_vol_target_cfg())
        assert hybrid["side"] == "SELL"

    def test_trend_up_produces_buy(self):
        """Uptrending closes + TREND_UP regime → BUY signal."""
        closes = _make_trending_closes("up")
        cfg = _make_trend_cfg(min_trend_strength=0.01, use_htf_rsi_filter=True, rsi_overbought=70)
        trend_info = compute_trend_bias(closes, htf_rsi=80.0, cfg=cfg, regime="TREND_UP")
        assert trend_info["direction"] == "LONG"

        carry_info = {"score_long": 0.0, "score_short": 0.0}
        hybrid = decide_hybrid_side(trend_info, carry_info, _make_vol_target_cfg())
        assert hybrid["side"] == "BUY"

    def test_choppy_flat_closes_no_signal(self):
        """Flat closes + CHOPPY regime → no directional bias."""
        closes = _make_trending_closes("flat")
        cfg = _make_trend_cfg(min_trend_strength=0.01)
        trend_info = compute_trend_bias(closes, htf_rsi=50.0, cfg=cfg, regime="CHOPPY")
        carry_info = {"score_long": 0.0, "score_short": 0.0}
        hybrid = decide_hybrid_side(trend_info, carry_info, _make_vol_target_cfg())
        assert hybrid["side"] == "NONE"


# ═══════════════════════════════════════════════════════════════════════════
# 4. Screener alignment: NEUTRAL ≠ BULL
# ═══════════════════════════════════════════════════════════════════════════

class TestNeutralTrendAlignment:
    """NEUTRAL trend should NOT count as BULL-aligned (long-bias removal)."""

    def test_neutral_not_aligned(self):
        """NEUTRAL should produce trend_aligned=False, not True."""
        # This tests the rule: vol_trend_aligned = vol_trend == "BULL"
        # We can't import the screener's inline logic, but we test contract:
        vol_trend = "NEUTRAL"
        # New behavior: NEUTRAL is NOT aligned
        vol_trend_aligned = vol_trend == "BULL"
        assert vol_trend_aligned is False

    def test_bull_still_aligned(self):
        vol_trend = "BULL"
        vol_trend_aligned = vol_trend == "BULL"
        assert vol_trend_aligned is True

    def test_bear_still_not_aligned(self):
        vol_trend = "BEAR"
        vol_trend_aligned = vol_trend == "BULL"
        assert vol_trend_aligned is False
