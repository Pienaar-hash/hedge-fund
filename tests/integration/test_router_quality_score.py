"""
Tests for v7.5_B2 — Router Quality Score computation.

Tests:
1. Ideal conditions → score near max_score
2. Mediocre conditions → score ~ base_score - small penalties
3. Bad conditions → score near min_score
4. Clamp behavior
"""
from __future__ import annotations

import pytest

from execution.router_metrics import (
    RouterQualityConfig,
    RouterQualitySnapshot,
    compute_router_quality_score,
    build_router_quality_snapshot,
    load_router_quality_config,
)


class TestRouterQualityConfig:
    """Test router quality configuration loading."""

    def test_default_config(self):
        """Default config has expected values."""
        cfg = RouterQualityConfig()
        assert cfg.enabled is True
        assert cfg.base_score == 0.8
        assert cfg.min_score == 0.2
        assert cfg.max_score == 1.0
        assert cfg.slippage_drift_green_bps == 2.0
        assert cfg.slippage_drift_yellow_bps == 6.0
        assert cfg.twap_skip_penalty == 0.10
        assert cfg.low_quality_threshold == 0.5
        assert cfg.high_quality_threshold == 0.9

    def test_load_from_strategy_config(self):
        """Config loads from strategy config dict."""
        strategy_cfg = {
            "router_quality": {
                "enabled": True,
                "base_score": 0.75,
                "min_score": 0.15,
                "slippage_drift_bps_thresholds": {
                    "green": 1.5,
                    "yellow": 5.0,
                },
                "bucket_penalties": {
                    "A_HIGH": 0.0,
                    "B_MEDIUM": -0.10,
                    "C_LOW": -0.20,
                },
            }
        }
        cfg = load_router_quality_config(strategy_cfg)
        assert cfg.base_score == 0.75
        assert cfg.min_score == 0.15
        assert cfg.slippage_drift_green_bps == 1.5
        assert cfg.slippage_drift_yellow_bps == 5.0
        assert cfg.bucket_penalty_b_medium == -0.10
        assert cfg.bucket_penalty_c_low == -0.20

    def test_disabled_config(self):
        """Disabled config is respected."""
        strategy_cfg = {
            "router_quality": {
                "enabled": False,
            }
        }
        cfg = load_router_quality_config(strategy_cfg)
        assert cfg.enabled is False


class TestComputeRouterQualityScore:
    """Test router quality score computation."""

    @pytest.fixture
    def default_cfg(self):
        """Default configuration."""
        return RouterQualityConfig()

    def test_ideal_conditions(self, default_cfg):
        """Ideal: A_HIGH bucket, low drift, no TWAP skips → near max_score."""
        score = compute_router_quality_score(
            bucket="A_HIGH",
            ewma_expected_bps=3.0,
            ewma_realized_bps=2.5,  # Negative drift (better than expected)
            twap_skip_ratio=0.0,
            cfg=default_cfg,
        )
        # base=0.8, bucket=0, drift=-0.5 (green)=-0.02, twap=0
        # 0.8 + 0.0 - 0.02 - 0.0 = 0.78
        assert score >= 0.75
        assert score <= default_cfg.max_score

    def test_mediocre_conditions(self, default_cfg):
        """Mediocre: B_MEDIUM bucket, medium drift → mid-range score."""
        score = compute_router_quality_score(
            bucket="B_MEDIUM",
            ewma_expected_bps=3.0,
            ewma_realized_bps=7.0,  # 4 bps drift (between green and yellow)
            twap_skip_ratio=0.1,
            cfg=default_cfg,
        )
        # base=0.8, bucket=-0.05, drift=4 (yellow)=-0.08, twap=-0.01
        # 0.8 - 0.05 - 0.08 - 0.01 = 0.66
        assert 0.5 <= score <= 0.75

    def test_bad_conditions(self, default_cfg):
        """Bad: C_LOW bucket, high drift, high TWAP skip → near min_score."""
        score = compute_router_quality_score(
            bucket="C_LOW",
            ewma_expected_bps=5.0,
            ewma_realized_bps=15.0,  # 10 bps drift (above yellow)
            twap_skip_ratio=0.5,
            cfg=default_cfg,
        )
        # base=0.8, bucket=-0.15, drift=10 (red)=-0.18, twap=-0.05
        # 0.8 - 0.15 - 0.18 - 0.05 = 0.42
        assert score <= 0.5
        assert score >= default_cfg.min_score

    def test_clamp_to_max(self, default_cfg):
        """Score never exceeds max_score."""
        # Start with very positive conditions
        cfg = RouterQualityConfig(base_score=1.2, max_score=1.0)
        score = compute_router_quality_score(
            bucket="A_HIGH",
            ewma_expected_bps=5.0,
            ewma_realized_bps=4.0,  # Negative drift
            twap_skip_ratio=0.0,
            cfg=cfg,
        )
        assert score <= 1.0

    def test_clamp_to_min(self, default_cfg):
        """Score never falls below min_score."""
        # Worst possible conditions
        cfg = RouterQualityConfig(min_score=0.2)
        score = compute_router_quality_score(
            bucket="C_LOW",
            ewma_expected_bps=2.0,
            ewma_realized_bps=30.0,  # Huge drift
            twap_skip_ratio=1.0,  # All slices skipped
            cfg=cfg,
        )
        assert score >= 0.2

    def test_generic_bucket_no_penalty(self, default_cfg):
        """GENERIC bucket has no penalty."""
        score_generic = compute_router_quality_score(
            bucket="GENERIC",
            ewma_expected_bps=3.0,
            ewma_realized_bps=3.0,
            twap_skip_ratio=0.0,
            cfg=default_cfg,
        )
        score_a_high = compute_router_quality_score(
            bucket="A_HIGH",
            ewma_expected_bps=3.0,
            ewma_realized_bps=3.0,
            twap_skip_ratio=0.0,
            cfg=default_cfg,
        )
        # Both should be similar (no bucket penalty for either)
        assert abs(score_generic - score_a_high) < 0.01

    def test_disabled_returns_base_score(self):
        """When disabled, returns base_score unchanged."""
        cfg = RouterQualityConfig(enabled=False, base_score=0.75)
        score = compute_router_quality_score(
            bucket="C_LOW",
            ewma_expected_bps=1.0,
            ewma_realized_bps=20.0,  # Would be huge penalty
            twap_skip_ratio=1.0,
            cfg=cfg,
        )
        assert score == 0.75


class TestBuildRouterQualitySnapshot:
    """Test snapshot building."""

    def test_builds_snapshot_with_computed_score(self):
        """Snapshot includes computed score and all metrics."""
        cfg = RouterQualityConfig()
        snapshot = build_router_quality_snapshot(
            symbol="BTCUSDT",
            ewma_expected_bps=2.5,
            ewma_realized_bps=3.0,
            bucket="A_HIGH",
            twap_skip_ratio=0.05,
            trade_count=100,
            cfg=cfg,
        )
        
        assert isinstance(snapshot, RouterQualitySnapshot)
        assert snapshot.symbol == "BTCUSDT"
        assert snapshot.bucket == "A_HIGH"
        assert snapshot.ewma_expected_bps == 2.5
        assert snapshot.ewma_realized_bps == 3.0
        assert snapshot.slippage_drift_bps == 0.5
        assert snapshot.twap_skip_ratio == 0.05
        assert snapshot.trade_count == 100
        assert 0.2 <= snapshot.score <= 1.0

    def test_snapshot_to_dict(self):
        """Snapshot serializes to dict correctly."""
        snapshot = RouterQualitySnapshot(
            symbol="ETHUSDT",
            score=0.72,
            bucket="B_MEDIUM",
            ewma_expected_bps=3.5,
            ewma_realized_bps=5.0,
            slippage_drift_bps=1.5,
            twap_skip_ratio=0.1,
            trade_count=50,
        )
        d = snapshot.to_dict()
        
        assert d["score"] == 0.72
        assert d["bucket"] == "B_MEDIUM"
        assert d["ewma_expected_bps"] == 3.5
        assert d["ewma_realized_bps"] == 5.0
        assert d["slippage_drift_bps"] == 1.5
        assert d["twap_skip_ratio"] == 0.1
        assert d["trade_count"] == 50

    def test_lowercase_symbol_uppercased(self):
        """Symbol is normalized to uppercase."""
        cfg = RouterQualityConfig()
        snapshot = build_router_quality_snapshot(
            symbol="btcusdt",
            ewma_expected_bps=2.0,
            ewma_realized_bps=2.5,
            bucket="A_HIGH",
            twap_skip_ratio=0.0,
            trade_count=10,
            cfg=cfg,
        )
        assert snapshot.symbol == "BTCUSDT"


class TestDriftZones:
    """Test slippage drift zone handling."""

    @pytest.fixture
    def cfg(self):
        return RouterQualityConfig(
            base_score=0.8,
            slippage_drift_green_bps=2.0,
            slippage_drift_yellow_bps=6.0,
        )

    def test_green_zone_drift(self, cfg):
        """Drift <= green threshold has minimal penalty."""
        score = compute_router_quality_score(
            bucket="A_HIGH",
            ewma_expected_bps=5.0,
            ewma_realized_bps=6.5,  # 1.5 bps drift (green zone)
            twap_skip_ratio=0.0,
            cfg=cfg,
        )
        # base=0.8, bucket=0, drift(green)=-0.02, twap=0
        assert score == pytest.approx(0.78, abs=0.01)

    def test_yellow_zone_drift(self, cfg):
        """Drift between green and yellow has moderate penalty."""
        score = compute_router_quality_score(
            bucket="A_HIGH",
            ewma_expected_bps=5.0,
            ewma_realized_bps=10.0,  # 5 bps drift (yellow zone)
            twap_skip_ratio=0.0,
            cfg=cfg,
        )
        # base=0.8, bucket=0, drift(yellow)=-0.08, twap=0
        assert score == pytest.approx(0.72, abs=0.01)

    def test_red_zone_drift(self, cfg):
        """Drift above yellow has strong penalty."""
        score = compute_router_quality_score(
            bucket="A_HIGH",
            ewma_expected_bps=5.0,
            ewma_realized_bps=15.0,  # 10 bps drift (red zone)
            twap_skip_ratio=0.0,
            cfg=cfg,
        )
        # base=0.8, bucket=0, drift(red)=-0.18, twap=0
        assert score == pytest.approx(0.62, abs=0.01)

    def test_negative_drift_is_green(self, cfg):
        """Negative drift (better than expected) is in green zone."""
        score = compute_router_quality_score(
            bucket="A_HIGH",
            ewma_expected_bps=5.0,
            ewma_realized_bps=3.0,  # -2 bps drift (better than expected)
            twap_skip_ratio=0.0,
            cfg=cfg,
        )
        # base=0.8, bucket=0, drift(green)=-0.02, twap=0
        assert score == pytest.approx(0.78, abs=0.01)
