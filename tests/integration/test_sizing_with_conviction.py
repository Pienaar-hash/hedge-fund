"""
Integration tests for conviction sizing with vol_target (v7.7)

Tests the full conviction sizing flow with mock state surfaces.
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from typing import Any, Dict

from execution.conviction_engine import (
    ConvictionConfig,
    ConvictionContext,
    compute_conviction,
    apply_conviction_to_nav_pct,
    load_conviction_config,
)
from execution.strategies.vol_target import apply_conviction_sizing


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_strategy_config() -> Dict[str, Any]:
    """Return a strategy config with conviction enabled."""
    return {
        "conviction": {
            "enabled": True,
            "size_multipliers": {
                "very_low": 0.3,
                "low": 0.5,
                "medium": 1.0,
                "high": 1.6,
                "very_high": 2.2,
            },
            "thresholds": {
                "very_low": 0.20,
                "low": 0.40,
                "medium": 0.60,
                "high": 0.80,
                "very_high": 0.92,
            },
            "dd_overrides": {
                "DRAWDOWN": 0.5,
                "RECOVERY": 0.8,
            },
            "risk_mode_overrides": {
                "DEFENSIVE": 0.5,
                "EXTREME": 0.0,
            },
            "router_quality_thresholds": {
                "min_for_any_trade": 0.50,
                "min_for_full_size": 0.80,
            },
            # v7.7_P6: Disable smoothing for unit tests
            "regime_curves": {
                "smoothing_alpha": 1.0,  # No smoothing in tests
            },
        },
    }


@pytest.fixture
def mock_risk_snapshot_file(tmp_path: Path) -> Path:
    """Create a mock risk_snapshot.json file."""
    risk_snapshot = {
        "dd_state": "NORMAL",
        "risk_mode": "OK",
        "updated_ts": 1700000000,
    }
    path = tmp_path / "risk_snapshot.json"
    path.write_text(json.dumps(risk_snapshot))
    return path


@pytest.fixture
def mock_router_health_file(tmp_path: Path) -> Path:
    """Create a mock router_health.json file."""
    router_health = {
        "router_health": {
            "global": {
                "quality_score": 0.85,
            },
        },
        "updated_ts": 1700000000,
    }
    path = tmp_path / "router_health.json"
    path.write_text(json.dumps(router_health))
    return path


# ---------------------------------------------------------------------------
# Integration Tests: Sizing Clamping
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSizingClamping:
    """Test that sizing is properly clamped to min/max bounds."""

    def test_size_clamped_to_max_nav_pct(self, mock_strategy_config: Dict[str, Any]):
        """High conviction should not exceed max_nav_pct."""
        ctx = ConvictionContext(
            hybrid_score=1.0,
            expectancy_alpha=1.0,
            router_quality=1.0,
            trend_strength=1.0,
            vol_regime="normal",
            dd_state="NORMAL",
            risk_mode="OK",
        )
        cfg = load_conviction_config(mock_strategy_config)
        result = compute_conviction(ctx, cfg)
        
        # Base 3% * 2.2 = 6.6%, should clamp to 4%
        final = apply_conviction_to_nav_pct(
            base_nav_pct=0.03,
            conviction_result=result,
            min_nav_pct=0.005,
            max_nav_pct=0.04,
        )
        
        assert final == 0.04
        assert final <= 0.04

    def test_size_clamped_to_min_nav_pct(self, mock_strategy_config: Dict[str, Any]):
        """Low conviction should not go below min_nav_pct."""
        ctx = ConvictionContext(
            hybrid_score=0.1,
            expectancy_alpha=0.1,
            router_quality=0.6,
            trend_strength=0.1,
            vol_regime="crisis",  # Extra penalty
            dd_state="NORMAL",
            risk_mode="OK",
        )
        cfg = load_conviction_config(mock_strategy_config)
        result = compute_conviction(ctx, cfg)
        
        final = apply_conviction_to_nav_pct(
            base_nav_pct=0.01,
            conviction_result=result,
            min_nav_pct=0.005,
            max_nav_pct=0.04,
        )
        
        assert final >= 0.005

    def test_size_within_bounds(self, mock_strategy_config: Dict[str, Any]):
        """Medium conviction should stay within bounds."""
        ctx = ConvictionContext(
            hybrid_score=0.5,
            expectancy_alpha=0.5,
            router_quality=0.9,
            trend_strength=0.5,
            vol_regime="normal",
            dd_state="NORMAL",
            risk_mode="OK",
        )
        cfg = load_conviction_config(mock_strategy_config)
        result = compute_conviction(ctx, cfg)
        
        final = apply_conviction_to_nav_pct(
            base_nav_pct=0.02,
            conviction_result=result,
            min_nav_pct=0.005,
            max_nav_pct=0.04,
        )
        
        assert 0.005 <= final <= 0.04


# ---------------------------------------------------------------------------
# Integration Tests: DD State Variations
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestDDStateVariations:
    """Test sizing behavior under different DD states."""

    def test_drawdown_state_caps_size(self, mock_strategy_config: Dict[str, Any]):
        """DRAWDOWN state should cap multiplier at 0.5."""
        ctx = ConvictionContext(
            hybrid_score=1.0,
            expectancy_alpha=1.0,
            router_quality=1.0,
            trend_strength=1.0,
            vol_regime="normal",
            dd_state="DRAWDOWN",
            risk_mode="OK",
        )
        cfg = load_conviction_config(mock_strategy_config)
        result = compute_conviction(ctx, cfg)
        
        # Even with very_high band (2.2x), DD caps at 0.5x
        assert result.size_multiplier == 0.5

    def test_recovery_state_caps_size(self, mock_strategy_config: Dict[str, Any]):
        """RECOVERY state should cap multiplier at 0.8."""
        ctx = ConvictionContext(
            hybrid_score=1.0,
            expectancy_alpha=1.0,
            router_quality=1.0,
            trend_strength=1.0,
            vol_regime="normal",
            dd_state="RECOVERY",
            risk_mode="OK",
        )
        cfg = load_conviction_config(mock_strategy_config)
        result = compute_conviction(ctx, cfg)
        
        assert result.size_multiplier == 0.8

    def test_normal_state_allows_full_size(self, mock_strategy_config: Dict[str, Any]):
        """NORMAL state should not cap multiplier."""
        ctx = ConvictionContext(
            hybrid_score=1.0,
            expectancy_alpha=1.0,
            router_quality=1.0,
            trend_strength=1.0,
            vol_regime="normal",
            dd_state="NORMAL",
            risk_mode="OK",
        )
        cfg = load_conviction_config(mock_strategy_config)
        result = compute_conviction(ctx, cfg)
        
        # very_high band = 2.2x, no cap
        assert result.size_multiplier == 2.2


# ---------------------------------------------------------------------------
# Integration Tests: Router Quality Gating
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestRouterQualityGating:
    """Test router quality threshold enforcement."""

    def test_quality_below_min_for_any_vetoes(self, mock_strategy_config: Dict[str, Any]):
        """Router quality < 0.50 should veto the trade entirely."""
        ctx = ConvictionContext(
            hybrid_score=1.0,
            expectancy_alpha=1.0,
            router_quality=0.45,  # Below 0.50
            trend_strength=1.0,
            vol_regime="normal",
            dd_state="NORMAL",
            risk_mode="OK",
        )
        cfg = load_conviction_config(mock_strategy_config)
        result = compute_conviction(ctx, cfg)
        
        assert result.vetoed is True
        assert result.size_multiplier == 0.0
        assert "router_quality" in (result.veto_reason or "")

    def test_quality_between_thresholds_scales_down(self, mock_strategy_config: Dict[str, Any]):
        """Router quality between min_for_any and min_for_full should scale down."""
        ctx = ConvictionContext(
            hybrid_score=1.0,
            expectancy_alpha=1.0,
            router_quality=0.65,  # Between 0.50 and 0.80
            trend_strength=1.0,
            vol_regime="normal",
            dd_state="NORMAL",
            risk_mode="OK",
        )
        cfg = load_conviction_config(mock_strategy_config)
        result = compute_conviction(ctx, cfg)
        
        assert result.vetoed is False
        # Multiplier should be reduced but not zero
        assert 0.0 < result.size_multiplier < 2.2

    def test_quality_above_min_for_full_no_scaling(self, mock_strategy_config: Dict[str, Any]):
        """Router quality >= 0.80 should not scale down."""
        ctx = ConvictionContext(
            hybrid_score=1.0,
            expectancy_alpha=1.0,
            router_quality=0.85,  # Above 0.80
            trend_strength=1.0,
            vol_regime="normal",
            dd_state="NORMAL",
            risk_mode="OK",
        )
        cfg = load_conviction_config(mock_strategy_config)
        result = compute_conviction(ctx, cfg)
        
        assert result.vetoed is False
        # Full very_high multiplier
        assert result.size_multiplier == 2.2


# ---------------------------------------------------------------------------
# Integration Tests: Risk Mode Variations
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestRiskModeVariations:
    """Test sizing behavior under different risk modes."""

    def test_extreme_risk_mode_vetoes(self, mock_strategy_config: Dict[str, Any]):
        """EXTREME risk mode should veto all trades."""
        ctx = ConvictionContext(
            hybrid_score=1.0,
            expectancy_alpha=1.0,
            router_quality=1.0,
            trend_strength=1.0,
            vol_regime="normal",
            dd_state="NORMAL",
            risk_mode="EXTREME",
        )
        cfg = load_conviction_config(mock_strategy_config)
        result = compute_conviction(ctx, cfg)
        
        assert result.vetoed is True
        assert result.size_multiplier == 0.0
        assert "EXTREME" in (result.veto_reason or "")

    def test_defensive_risk_mode_caps_size(self, mock_strategy_config: Dict[str, Any]):
        """DEFENSIVE risk mode should cap multiplier at 0.5."""
        ctx = ConvictionContext(
            hybrid_score=1.0,
            expectancy_alpha=1.0,
            router_quality=1.0,
            trend_strength=1.0,
            vol_regime="normal",
            dd_state="NORMAL",
            risk_mode="DEFENSIVE",
        )
        cfg = load_conviction_config(mock_strategy_config)
        result = compute_conviction(ctx, cfg)
        
        assert result.vetoed is False
        assert result.size_multiplier == 0.5

    def test_ok_risk_mode_no_cap(self, mock_strategy_config: Dict[str, Any]):
        """OK risk mode should not cap multiplier."""
        ctx = ConvictionContext(
            hybrid_score=1.0,
            expectancy_alpha=1.0,
            router_quality=1.0,
            trend_strength=1.0,
            vol_regime="normal",
            dd_state="NORMAL",
            risk_mode="OK",
        )
        cfg = load_conviction_config(mock_strategy_config)
        result = compute_conviction(ctx, cfg)
        
        assert result.vetoed is False
        assert result.size_multiplier == 2.2


# ---------------------------------------------------------------------------
# Integration Tests: Vol Regime Effects
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestVolRegimeEffects:
    """Test conviction score changes under different vol regimes."""

    def test_crisis_regime_reduces_conviction(self, mock_strategy_config: Dict[str, Any]):
        """Crisis vol regime should apply 0.4 penalty (v7.7_P6 regime curves)."""
        ctx_normal = ConvictionContext(
            hybrid_score=0.8,
            expectancy_alpha=0.8,
            router_quality=1.0,
            trend_strength=0.8,
            vol_regime="normal",
        )
        ctx_crisis = ConvictionContext(
            hybrid_score=0.8,
            expectancy_alpha=0.8,
            router_quality=1.0,
            trend_strength=0.8,
            vol_regime="crisis",
        )
        cfg = load_conviction_config(mock_strategy_config)
        
        result_normal = compute_conviction(ctx_normal, cfg)
        result_crisis = compute_conviction(ctx_crisis, cfg)
        
        # Crisis should have lower conviction score
        assert result_crisis.conviction_score < result_normal.conviction_score
        # v7.7_P6: Approximately 0.4x the normal score (was 0.6x legacy)
        ratio = result_crisis.conviction_score / result_normal.conviction_score
        assert 0.35 <= ratio <= 0.45


# ---------------------------------------------------------------------------
# Integration Tests: apply_conviction_sizing helper
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestApplyConvictionSizingHelper:
    """Test the vol_target integration helper."""

    def test_disabled_returns_base_nav_pct(self):
        """When conviction disabled, returns base nav_pct unchanged."""
        strategy_cfg = {"conviction": {"enabled": False}}
        
        final_pct, metadata = apply_conviction_sizing(
            base_nav_pct=0.02,
            hybrid_score=0.8,
            trend_strength=0.7,
            vol_regime="normal",
            strategy_config=strategy_cfg,
            min_nav_pct=0.005,
            max_nav_pct=0.04,
        )
        
        assert final_pct == 0.02
        assert metadata is None

    def test_enabled_returns_scaled_nav_pct(self, mock_strategy_config: Dict[str, Any]):
        """When conviction enabled, returns scaled nav_pct with metadata."""
        # Note: This will use default state files which may not exist in tests.
        # The function handles missing state gracefully (defaults to 1.0 quality, etc.)
        final_pct, metadata = apply_conviction_sizing(
            base_nav_pct=0.02,
            hybrid_score=0.7,
            trend_strength=0.6,
            vol_regime="normal",
            strategy_config=mock_strategy_config,
            min_nav_pct=0.005,
            max_nav_pct=0.04,
        )
        
        # Should return some value (may differ based on default state loading)
        assert final_pct >= 0.0
        assert metadata is not None
        assert "conviction_score" in metadata
        assert "conviction_band" in metadata
        assert "size_multiplier" in metadata


# ---------------------------------------------------------------------------
# Integration Tests: Conviction Score Bounds
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestConvictionScoreBounds:
    """Test that conviction score is always in [0, 1]."""

    def test_score_always_in_bounds(self, mock_strategy_config: Dict[str, Any]):
        """Conviction score should always be between 0 and 1."""
        test_cases = [
            (0.0, 0.0, 0.0, 0.0, "low"),
            (1.0, 1.0, 1.0, 1.0, "normal"),
            (0.5, 0.5, 0.5, 0.5, "high"),
            (0.0, 0.0, 1.0, 0.0, "crisis"),
            (1.0, 0.0, 0.0, 1.0, "normal"),
        ]
        
        cfg = load_conviction_config(mock_strategy_config)
        
        for hybrid, exp, router, trend, regime in test_cases:
            ctx = ConvictionContext(
                hybrid_score=hybrid,
                expectancy_alpha=exp,
                router_quality=router,
                trend_strength=trend,
                vol_regime=regime,  # type: ignore[arg-type]
                dd_state="NORMAL",
                risk_mode="OK",
            )
            result = compute_conviction(ctx, cfg)
            
            assert 0.0 <= result.conviction_score <= 1.0, (
                f"Score {result.conviction_score} out of bounds for inputs: "
                f"hybrid={hybrid}, exp={exp}, router={router}, trend={trend}, regime={regime}"
            )
