"""
Unit tests for conviction_engine.py (v7.7)

Tests conviction score computation, band mapping, size multipliers,
and DD/risk_mode overrides.
"""

from __future__ import annotations

import pytest

from execution.conviction_engine import (
    ConvictionConfig,
    ConvictionContext,
    ConvictionResult,
    compute_conviction_score,
    get_conviction_band,
    compute_size_multiplier,
    compute_conviction,
    apply_conviction_to_nav_pct,
    load_conviction_config,
    get_global_router_quality,
    get_dd_state,
    get_risk_mode,
)


# ---------------------------------------------------------------------------
# Test: compute_conviction_score
# ---------------------------------------------------------------------------

class TestComputeConvictionScore:
    """Test raw conviction score computation."""

    def test_all_zeros_returns_zero(self):
        ctx = ConvictionContext(
            hybrid_score=0.0,
            expectancy_alpha=0.0,
            router_quality=0.0,
            trend_strength=0.0,
            vol_regime="normal",
        )
        score = compute_conviction_score(ctx)
        assert score == 0.0

    def test_all_ones_returns_one(self):
        ctx = ConvictionContext(
            hybrid_score=1.0,
            expectancy_alpha=1.0,
            router_quality=1.0,
            trend_strength=1.0,
            vol_regime="normal",
        )
        score = compute_conviction_score(ctx)
        assert score == 1.0

    def test_medium_inputs_return_medium_score(self):
        ctx = ConvictionContext(
            hybrid_score=0.5,
            expectancy_alpha=0.5,
            router_quality=0.5,
            trend_strength=0.5,
            vol_regime="normal",
        )
        score = compute_conviction_score(ctx)
        # 0.4*0.5 + 0.25*0.5 + 0.2*0.5 + 0.15*0.5 = 0.5
        assert score == pytest.approx(0.5, abs=0.01)

    def test_vol_regime_high_applies_penalty(self):
        ctx = ConvictionContext(
            hybrid_score=1.0,
            expectancy_alpha=1.0,
            router_quality=1.0,
            trend_strength=1.0,
            vol_regime="high",
        )
        score = compute_conviction_score(ctx)
        # 1.0 * 0.85 = 0.85
        assert score == pytest.approx(0.85, abs=0.01)

    def test_vol_regime_crisis_applies_heavy_penalty(self):
        ctx = ConvictionContext(
            hybrid_score=1.0,
            expectancy_alpha=1.0,
            router_quality=1.0,
            trend_strength=1.0,
            vol_regime="crisis",
        )
        score = compute_conviction_score(ctx)
        # 1.0 * 0.6 = 0.6
        assert score == pytest.approx(0.6, abs=0.01)

    def test_score_clamped_to_zero_one(self):
        ctx = ConvictionContext(
            hybrid_score=2.0,  # exceeds 1.0
            expectancy_alpha=2.0,
            router_quality=2.0,
            trend_strength=2.0,
            vol_regime="normal",
        )
        score = compute_conviction_score(ctx)
        assert 0.0 <= score <= 1.0

    def test_negative_inputs_clamped_to_zero(self):
        ctx = ConvictionContext(
            hybrid_score=-0.5,
            expectancy_alpha=-0.5,
            router_quality=-0.5,
            trend_strength=-0.5,
            vol_regime="normal",
        )
        score = compute_conviction_score(ctx)
        assert score == 0.0


# ---------------------------------------------------------------------------
# Test: get_conviction_band
# ---------------------------------------------------------------------------

class TestGetConvictionBand:
    """Test conviction band mapping.
    
    Thresholds define lower bound for each band (checked from highest to lowest):
    - score >= 0.92 → very_high
    - score >= 0.80 → high
    - score >= 0.60 → medium
    - score >= 0.40 → low
    - score < 0.40 → very_low
    """

    def test_very_low_band(self):
        cfg = ConvictionConfig()
        assert get_conviction_band(0.10, cfg) == "very_low"
        assert get_conviction_band(0.39, cfg) == "very_low"

    def test_low_band(self):
        cfg = ConvictionConfig()
        # Score must be >= 0.40 but < 0.60 for "low"
        assert get_conviction_band(0.40, cfg) == "low"
        assert get_conviction_band(0.59, cfg) == "low"

    def test_medium_band(self):
        cfg = ConvictionConfig()
        # Score must be >= 0.60 but < 0.80 for "medium"
        assert get_conviction_band(0.60, cfg) == "medium"
        assert get_conviction_band(0.79, cfg) == "medium"

    def test_high_band(self):
        cfg = ConvictionConfig()
        # Score must be >= 0.80 but < 0.92 for "high"
        assert get_conviction_band(0.80, cfg) == "high"
        assert get_conviction_band(0.91, cfg) == "high"

    def test_very_high_band(self):
        cfg = ConvictionConfig()
        # Score must be >= 0.92 for "very_high"
        assert get_conviction_band(0.92, cfg) == "very_high"
        assert get_conviction_band(1.0, cfg) == "very_high"

    def test_edge_case_at_thresholds(self):
        cfg = ConvictionConfig()
        # At exactly threshold, score belongs to that band
        assert get_conviction_band(0.40, cfg) == "low"
        assert get_conviction_band(0.60, cfg) == "medium"
        assert get_conviction_band(0.80, cfg) == "high"
        assert get_conviction_band(0.92, cfg) == "very_high"


# ---------------------------------------------------------------------------
# Test: compute_size_multiplier
# ---------------------------------------------------------------------------

class TestComputeSizeMultiplier:
    """Test size multiplier computation with overrides."""

    def test_base_multiplier_from_band(self):
        cfg = ConvictionConfig()
        ctx = ConvictionContext(router_quality=1.0)
        
        mult, veto = compute_size_multiplier(0.5, "medium", ctx, cfg)
        assert mult == 1.0
        assert veto is None

    def test_very_high_band_multiplier(self):
        cfg = ConvictionConfig()
        ctx = ConvictionContext(router_quality=1.0)
        
        mult, veto = compute_size_multiplier(0.95, "very_high", ctx, cfg)
        assert mult == 2.2
        assert veto is None

    def test_router_quality_below_min_for_any_vetoes(self):
        cfg = ConvictionConfig()
        ctx = ConvictionContext(router_quality=0.40)  # < 0.50
        
        mult, veto = compute_size_multiplier(0.5, "medium", ctx, cfg)
        assert mult == 0.0
        assert veto is not None
        assert "router_quality" in veto

    def test_router_quality_below_min_for_full_size_clamps(self):
        cfg = ConvictionConfig()
        ctx = ConvictionContext(router_quality=0.65)  # between 0.50 and 0.80
        
        mult, veto = compute_size_multiplier(0.5, "medium", ctx, cfg)
        # Should be scaled down but not vetoed
        assert 0.0 < mult < 1.0
        assert veto is None

    def test_dd_state_drawdown_override(self):
        cfg = ConvictionConfig()
        ctx = ConvictionContext(
            router_quality=1.0,
            dd_state="DRAWDOWN",
        )
        
        mult, veto = compute_size_multiplier(0.95, "very_high", ctx, cfg)
        # very_high = 2.2, but DRAWDOWN caps at 0.5
        assert mult == 0.5
        assert veto is None

    def test_dd_state_recovery_override(self):
        cfg = ConvictionConfig()
        ctx = ConvictionContext(
            router_quality=1.0,
            dd_state="RECOVERY",
        )
        
        mult, veto = compute_size_multiplier(0.95, "very_high", ctx, cfg)
        # very_high = 2.2, but RECOVERY caps at 0.8
        assert mult == 0.8
        assert veto is None

    def test_risk_mode_defensive_override(self):
        cfg = ConvictionConfig()
        ctx = ConvictionContext(
            router_quality=1.0,
            risk_mode="DEFENSIVE",
        )
        
        mult, veto = compute_size_multiplier(0.7, "high", ctx, cfg)
        # high = 1.6, but DEFENSIVE caps at 0.5
        assert mult == 0.5
        assert veto is None

    def test_risk_mode_extreme_vetoes(self):
        cfg = ConvictionConfig()
        ctx = ConvictionContext(
            router_quality=1.0,
            risk_mode="EXTREME",
        )
        
        mult, veto = compute_size_multiplier(0.5, "medium", ctx, cfg)
        assert mult == 0.0
        assert veto is not None
        assert "EXTREME" in veto


# ---------------------------------------------------------------------------
# Test: compute_conviction (main entry point)
# ---------------------------------------------------------------------------

class TestComputeConviction:
    """Test the main compute_conviction function."""

    def test_returns_conviction_result(self):
        ctx = ConvictionContext(
            hybrid_score=0.7,
            expectancy_alpha=0.6,
            router_quality=0.9,
            trend_strength=0.8,
            vol_regime="normal",
        )
        
        result = compute_conviction(ctx)
        
        assert isinstance(result, ConvictionResult)
        assert 0.0 <= result.conviction_score <= 1.0
        assert result.conviction_band in ("very_low", "low", "medium", "high", "very_high")
        assert result.size_multiplier >= 0.0

    def test_low_quality_vetoes_trade(self):
        ctx = ConvictionContext(
            hybrid_score=0.9,
            expectancy_alpha=0.9,
            router_quality=0.30,  # Below min_for_any_trade
            trend_strength=0.9,
        )
        
        result = compute_conviction(ctx)
        
        assert result.vetoed is True
        assert result.size_multiplier == 0.0
        assert result.veto_reason is not None

    def test_normal_case_not_vetoed(self):
        ctx = ConvictionContext(
            hybrid_score=0.6,
            expectancy_alpha=0.5,
            router_quality=0.85,
            trend_strength=0.5,
            vol_regime="normal",
            dd_state="NORMAL",
            risk_mode="OK",
        )
        
        result = compute_conviction(ctx)
        
        assert result.vetoed is False
        assert result.size_multiplier > 0.0


# ---------------------------------------------------------------------------
# Test: apply_conviction_to_nav_pct
# ---------------------------------------------------------------------------

class TestApplyConvictionToNavPct:
    """Test NAV pct application with clamping."""

    def test_basic_scaling(self):
        result = ConvictionResult(
            conviction_score=0.7,
            conviction_band="high",
            size_multiplier=1.6,
            vetoed=False,
        )
        
        final = apply_conviction_to_nav_pct(
            base_nav_pct=0.02,
            conviction_result=result,
            min_nav_pct=0.005,
            max_nav_pct=0.04,
        )
        
        # 0.02 * 1.6 = 0.032, within bounds
        assert final == pytest.approx(0.032, abs=0.001)

    def test_clamped_to_max(self):
        result = ConvictionResult(
            conviction_score=0.95,
            conviction_band="very_high",
            size_multiplier=2.2,
            vetoed=False,
        )
        
        final = apply_conviction_to_nav_pct(
            base_nav_pct=0.03,
            conviction_result=result,
            min_nav_pct=0.005,
            max_nav_pct=0.04,
        )
        
        # 0.03 * 2.2 = 0.066, clamped to max 0.04
        assert final == 0.04

    def test_clamped_to_min(self):
        result = ConvictionResult(
            conviction_score=0.1,
            conviction_band="very_low",
            size_multiplier=0.3,
            vetoed=False,
        )
        
        final = apply_conviction_to_nav_pct(
            base_nav_pct=0.01,
            conviction_result=result,
            min_nav_pct=0.005,
            max_nav_pct=0.04,
        )
        
        # 0.01 * 0.3 = 0.003, clamped to min 0.005
        assert final == 0.005

    def test_vetoed_returns_zero(self):
        result = ConvictionResult(
            conviction_score=0.5,
            conviction_band="medium",
            size_multiplier=0.0,
            vetoed=True,
            veto_reason="test veto",
        )
        
        final = apply_conviction_to_nav_pct(
            base_nav_pct=0.02,
            conviction_result=result,
            min_nav_pct=0.005,
            max_nav_pct=0.04,
        )
        
        assert final == 0.0


# ---------------------------------------------------------------------------
# Test: State extraction helpers
# ---------------------------------------------------------------------------

class TestStateExtraction:
    """Test state extraction from snapshots."""

    def test_get_global_router_quality_nested(self):
        router_health = {
            "router_health": {
                "global": {
                    "quality_score": 0.85,
                },
            },
        }
        assert get_global_router_quality(router_health) == pytest.approx(0.85)

    def test_get_global_router_quality_flat(self):
        router_health = {
            "global": {
                "quality_score": 0.75,
            },
        }
        assert get_global_router_quality(router_health) == pytest.approx(0.75)

    def test_get_global_router_quality_missing_defaults_to_one(self):
        assert get_global_router_quality({}) == 1.0

    def test_get_dd_state_valid(self):
        assert get_dd_state({"dd_state": "DRAWDOWN"}) == "DRAWDOWN"
        assert get_dd_state({"dd_state": "RECOVERY"}) == "RECOVERY"
        assert get_dd_state({"dd_state": "NORMAL"}) == "NORMAL"

    def test_get_dd_state_invalid_defaults_to_normal(self):
        assert get_dd_state({"dd_state": "UNKNOWN"}) == "NORMAL"
        assert get_dd_state({}) == "NORMAL"

    def test_get_risk_mode_valid(self):
        assert get_risk_mode({"risk_mode": "OK"}) == "OK"
        assert get_risk_mode({"risk_mode": "DEFENSIVE"}) == "DEFENSIVE"
        assert get_risk_mode({"risk_mode": "EXTREME"}) == "EXTREME"

    def test_get_risk_mode_invalid_defaults_to_ok(self):
        assert get_risk_mode({"risk_mode": "UNKNOWN"}) == "OK"
        assert get_risk_mode({}) == "OK"


# ---------------------------------------------------------------------------
# Test: load_conviction_config
# ---------------------------------------------------------------------------

class TestLoadConvictionConfig:
    """Test config loading from strategy_config."""

    def test_load_default_when_none(self):
        cfg = load_conviction_config(None)
        assert cfg.enabled is False  # Default to disabled for backward compatibility
        assert cfg.size_multipliers["medium"] == 1.0

    def test_load_from_strategy_config(self):
        strategy_cfg = {
            "conviction": {
                "enabled": False,
                "size_multipliers": {
                    "medium": 0.8,
                },
            },
        }
        cfg = load_conviction_config(strategy_cfg)
        assert cfg.enabled is False
        assert cfg.size_multipliers["medium"] == 0.8

    def test_load_missing_conviction_key_uses_defaults(self):
        strategy_cfg = {"other_key": "value"}
        cfg = load_conviction_config(strategy_cfg)
        assert cfg.enabled is False  # Default to disabled for backward compatibility
