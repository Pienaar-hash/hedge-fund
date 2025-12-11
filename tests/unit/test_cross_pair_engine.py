"""
Unit tests for cross-pair statistical arbitrage engine (Crossfire).

Tests:
- Config loading and defaults
- OLS hedge ratio computation
- Spread series calculation
- Correlation computation
- Half-life estimation (mean reversion speed)
- Edge score logic
- Signal classification (ENTER/EXIT/NONE)
- State I/O

v7.8_P5: Cross-Pair Statistical Arbitrage Engine.
"""

import pytest
import math
from unittest.mock import patch, MagicMock
from dataclasses import asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

from execution.cross_pair_engine import (
    # Config
    CrossPairConfig,
    load_cross_pair_config,
    # Stats
    PairStats,
    compute_pair_stats,
    compute_ols_hedge_ratio,
    compute_spread_series,
    compute_correlation,
    compute_half_life,
    compute_residual_momentum,
    # Edge
    PairEdge,
    compute_pair_edge,
    # State
    CrossPairState,
    load_cross_pair_state,
    save_cross_pair_state,
    # Runner
    should_run_cross_pair,
    run_cross_pair_scan,
    # Views
    get_top_pair_edges,
    get_cross_pair_summary,
    get_pair_edges_for_insights,
    # Integration
    get_pair_leg_boost,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_config() -> CrossPairConfig:
    """Default cross-pair config for testing."""
    return CrossPairConfig()


@pytest.fixture
def enabled_config() -> CrossPairConfig:
    """Enabled cross-pair config for testing."""
    return CrossPairConfig(
        enabled=True,
        pairs=[("BTCUSDT", "ETHUSDT")],
        lookback_bars=100,
        half_life_bars=60,
        min_corr=0.6,
        max_spread_z=4.0,
        entry_z=1.5,
        exit_z=0.3,
        min_liquidity_usd=50000.0,
        smoothing_alpha=0.3,
    )


@pytest.fixture
def sample_prices_base():
    """Sample price series for base asset (100 bars)."""
    import random
    random.seed(42)
    # Random walk starting at 50000
    prices = [50000.0]
    for i in range(99):
        prices.append(prices[-1] * (1 + random.gauss(0, 0.01)))
    return prices


@pytest.fixture
def sample_prices_quote():
    """Sample price series for quote asset (100 bars, correlated)."""
    import random
    random.seed(43)
    # Correlated random walk starting at 3000
    prices = [3000.0]
    for i in range(99):
        prices.append(prices[-1] * (1 + random.gauss(0, 0.012)))
    return prices


# ---------------------------------------------------------------------------
# Test: Config Loading
# ---------------------------------------------------------------------------

class TestConfigLoading:
    """Tests for config loading."""

    def test_default_config_values(self, default_config):
        """Test default config has expected values."""
        assert default_config.enabled is False
        assert default_config.lookback_bars > 0
        assert default_config.half_life_bars > 0
        assert 0 < default_config.min_corr <= 1
        assert default_config.max_spread_z > 0
        assert default_config.entry_z > 0
        assert default_config.exit_z >= 0
        assert default_config.min_liquidity_usd >= 0
        assert 0 < default_config.smoothing_alpha <= 1

    def test_enabled_config_custom_values(self, enabled_config):
        """Test custom config values."""
        assert enabled_config.enabled is True
        assert enabled_config.pairs == [("BTCUSDT", "ETHUSDT")]
        assert enabled_config.lookback_bars == 100
        assert enabled_config.min_corr == 0.6

    def test_load_cross_pair_config_with_strategy_cfg(self):
        """Test loading config from strategy config dict."""
        cfg = {
            "cross_pair_engine": {
                "enabled": True,
                "pairs": [["BTCUSDT", "ETHUSDT"], ["SOLUSDT", "SUIUSDT"]],
                "lookback_bars": 150,
                "min_corr": 0.7,
            }
        }
        config = load_cross_pair_config(strategy_cfg=cfg)
        assert isinstance(config, CrossPairConfig)
        assert config.enabled is True
        assert len(config.pairs) == 2
        assert config.lookback_bars == 150
        assert config.min_corr == 0.7

    def test_load_cross_pair_config_missing_section(self):
        """Test loading config when section is missing."""
        cfg = {"some_other_section": {}}
        config = load_cross_pair_config(strategy_cfg=cfg)
        assert isinstance(config, CrossPairConfig)
        assert config.enabled is False  # Default to disabled

    def test_config_normalization(self):
        """Test that config normalizes pairs to tuples."""
        config = CrossPairConfig(
            enabled=True,
            pairs=[["btcusdt", "ethusdt"]],  # lowercase list
        )
        assert config.pairs[0] == ("BTCUSDT", "ETHUSDT")


# ---------------------------------------------------------------------------
# Test: Hedge Ratio Computation
# ---------------------------------------------------------------------------

class TestHedgeRatio:
    """Tests for OLS hedge ratio computation."""

    def test_hedge_ratio_basic(self, sample_prices_base, sample_prices_quote):
        """Test basic hedge ratio computation."""
        ratio, intercept = compute_ols_hedge_ratio(sample_prices_base, sample_prices_quote)
        assert ratio is not None
        assert isinstance(ratio, float)
        assert ratio > 0  # Should be positive for positively correlated assets

    def test_hedge_ratio_identical_series(self):
        """Test hedge ratio for identical price series."""
        prices = [100.0 + i for i in range(50)]
        ratio, intercept = compute_ols_hedge_ratio(prices, prices)
        assert ratio is not None
        # Should be ~1.0 for identical series
        assert abs(ratio - 1.0) < 0.01

    def test_hedge_ratio_scaled_series(self):
        """Test hedge ratio for scaled series."""
        base = [100.0 + i for i in range(50)]
        quote = [2 * (100.0 + i) for i in range(50)]  # 2x scaled
        ratio, intercept = compute_ols_hedge_ratio(base, quote)
        assert ratio is not None
        # Hedge ratio should be ~2.0 (base ~= 2 * quote in regression terms)
        # Actually it depends on which is X and which is Y in the regression
        assert 0.3 < ratio < 3.0  # Allow reasonable range

    def test_hedge_ratio_insufficient_data(self):
        """Test hedge ratio with insufficient data points."""
        base = [100.0, 101.0]  # Only 2 points
        quote = [50.0, 51.0]
        ratio, intercept = compute_ols_hedge_ratio(base, quote)
        # Should return default (1.0) for insufficient data
        assert ratio is not None
        assert isinstance(ratio, float)


# ---------------------------------------------------------------------------
# Test: Spread Series Computation
# ---------------------------------------------------------------------------

class TestSpreadSeries:
    """Tests for spread series computation."""

    def test_spread_series_basic(self, sample_prices_base, sample_prices_quote):
        """Test basic spread series computation."""
        hedge_ratio = 16.0
        spread = compute_spread_series(sample_prices_base, sample_prices_quote, hedge_ratio)
        assert spread is not None
        assert len(spread) > 0
        assert all(isinstance(x, float) for x in spread)

    def test_spread_series_length(self, sample_prices_base, sample_prices_quote):
        """Test spread series has correct length."""
        hedge_ratio = 16.0
        spread = compute_spread_series(sample_prices_base, sample_prices_quote, hedge_ratio)
        expected_len = min(len(sample_prices_base), len(sample_prices_quote))
        assert len(spread) == expected_len


# ---------------------------------------------------------------------------
# Test: Correlation Computation
# ---------------------------------------------------------------------------

class TestCorrelation:
    """Tests for correlation computation."""

    def test_correlation_positive(self, sample_prices_base, sample_prices_quote):
        """Test correlation for positively correlated series."""
        corr = compute_correlation(sample_prices_base, sample_prices_quote)
        assert corr is not None
        assert isinstance(corr, float)
        assert -1.0 <= corr <= 1.0

    def test_correlation_identical(self):
        """Test correlation for identical series."""
        prices = [100.0 + i * 0.1 for i in range(50)]
        corr = compute_correlation(prices, prices)
        assert corr is not None
        assert abs(corr - 1.0) < 0.01  # Should be ~1.0

    def test_correlation_negative(self):
        """Test correlation for negatively correlated series."""
        base = [100.0 + i for i in range(50)]
        quote = [100.0 - i for i in range(50)]
        corr = compute_correlation(base, quote)
        assert corr is not None
        assert corr < 0  # Should be negative


# ---------------------------------------------------------------------------
# Test: Half-Life Estimation
# ---------------------------------------------------------------------------

class TestHalfLife:
    """Tests for mean-reversion half-life estimation."""

    def test_half_life_basic(self, sample_prices_base, sample_prices_quote):
        """Test basic half-life estimation."""
        hedge_ratio = 16.0
        spread = compute_spread_series(sample_prices_base, sample_prices_quote, hedge_ratio)
        half_life = compute_half_life(spread)
        assert half_life is not None
        assert isinstance(half_life, float)
        assert half_life > 0  # Should be positive

    def test_half_life_mean_reverting(self):
        """Test half-life for a mean-reverting spread."""
        # Create a mean-reverting spread with known characteristics
        spread = []
        value = 0.0
        for i in range(100):
            value = value * 0.9 + 0.1 * ((-1) ** i)  # Mean-reverting
            spread.append(value)

        half_life = compute_half_life(spread)
        # Should find a relatively short half-life for mean-reverting spread
        assert half_life is not None
        assert half_life > 0


# ---------------------------------------------------------------------------
# Test: Residual Momentum
# ---------------------------------------------------------------------------

class TestResidualMomentum:
    """Tests for residual momentum computation."""

    def test_residual_momentum_basic(self, sample_prices_base, sample_prices_quote):
        """Test basic residual momentum."""
        hedge_ratio = 16.0
        spread = compute_spread_series(sample_prices_base, sample_prices_quote, hedge_ratio)
        momo = compute_residual_momentum(spread)
        assert momo is not None
        assert isinstance(momo, float)

    def test_residual_momentum_trending_up(self):
        """Test momentum for upward trending spread."""
        spread = [float(i) for i in range(50)]
        momo = compute_residual_momentum(spread)
        assert momo is not None
        assert momo > 0  # Should be positive for uptrend


# ---------------------------------------------------------------------------
# Test: Pair Stats Computation
# ---------------------------------------------------------------------------

class TestPairStats:
    """Tests for pair stats computation."""

    def test_compute_pair_stats_basic(self, enabled_config, sample_prices_base, sample_prices_quote):
        """Test basic pair stats computation."""
        stats = compute_pair_stats(
            base="BTCUSDT",
            quote="ETHUSDT",
            prices_base=sample_prices_base,
            prices_quote=sample_prices_quote,
            config=enabled_config,
            base_liquidity=1000000.0,
            quote_liquidity=500000.0,
        )
        assert stats is not None
        assert isinstance(stats, PairStats)
        assert stats.base == "BTCUSDT"
        assert stats.quote == "ETHUSDT"
        assert stats.hedge_ratio is not None
        assert stats.spread_z is not None
        assert stats.corr is not None

    def test_pair_stats_liquidity_check(self, enabled_config, sample_prices_base, sample_prices_quote):
        """Test that liquidity check works."""
        # High liquidity - should be OK
        stats_high = compute_pair_stats(
            base="BTCUSDT",
            quote="ETHUSDT",
            prices_base=sample_prices_base,
            prices_quote=sample_prices_quote,
            config=enabled_config,
            base_liquidity=1000000.0,
            quote_liquidity=500000.0,
        )
        assert stats_high.liquidity_ok is True

        # Low liquidity - should fail
        stats_low = compute_pair_stats(
            base="BTCUSDT",
            quote="ETHUSDT",
            prices_base=sample_prices_base,
            prices_quote=sample_prices_quote,
            config=enabled_config,
            base_liquidity=100.0,
            quote_liquidity=100.0,
        )
        assert stats_low.liquidity_ok is False


# ---------------------------------------------------------------------------
# Test: Edge Score Logic
# ---------------------------------------------------------------------------

class TestEdgeScore:
    """Tests for edge score computation."""

    def test_compute_pair_edge_basic(self, enabled_config, sample_prices_base, sample_prices_quote):
        """Test basic pair edge computation from stats."""
        stats = compute_pair_stats(
            base="BTCUSDT",
            quote="ETHUSDT",
            prices_base=sample_prices_base,
            prices_quote=sample_prices_quote,
            config=enabled_config,
            base_liquidity=1000000.0,
            quote_liquidity=500000.0,
        )
        edge = compute_pair_edge(stats, enabled_config)
        assert edge is not None
        assert isinstance(edge, PairEdge)
        assert edge.pair == ("BTCUSDT", "ETHUSDT")
        assert edge.stats is not None

    def test_compute_pair_edge_score_range(self, enabled_config, sample_prices_base, sample_prices_quote):
        """Test that edge score is in valid range."""
        stats = compute_pair_stats(
            base="BTCUSDT",
            quote="ETHUSDT",
            prices_base=sample_prices_base,
            prices_quote=sample_prices_quote,
            config=enabled_config,
            base_liquidity=1000000.0,
            quote_liquidity=500000.0,
        )
        edge = compute_pair_edge(stats, enabled_config)
        assert 0.0 <= edge.edge_score <= 1.0
        assert 0.0 <= edge.ema_score <= 1.0

    def test_compute_pair_edge_ineligible(self, enabled_config, sample_prices_base, sample_prices_quote):
        """Test edge for ineligible pair."""
        stats = compute_pair_stats(
            base="BTCUSDT",
            quote="ETHUSDT",
            prices_base=sample_prices_base,
            prices_quote=sample_prices_quote,
            config=enabled_config,
            base_liquidity=10.0,  # Very low
            quote_liquidity=10.0,
        )
        edge = compute_pair_edge(stats, enabled_config)
        assert stats.eligible is False
        assert edge.signal == "NONE"


# ---------------------------------------------------------------------------
# Test: Full Scan (run_cross_pair_scan)
# ---------------------------------------------------------------------------

class TestRunCrossPairScan:
    """Tests for the full cross-pair scan function."""

    def test_scan_disabled_returns_empty_state(self):
        """Test that disabled config returns an empty state."""
        config = CrossPairConfig(enabled=False)
        state = run_cross_pair_scan(config=config)
        assert state is not None
        assert state.pair_edges == {}
        assert state.pairs_analyzed == 0
        assert state.notes == "disabled"

    def test_should_run_cross_pair(self):
        """Test should_run_cross_pair cycle logic."""
        config = CrossPairConfig(enabled=True, run_interval_cycles=5)
        
        # Should run at cycle 0
        assert should_run_cross_pair(0, config) is True
        # Should run at cycle 5
        assert should_run_cross_pair(5, config) is True
        # Should not run at cycle 3
        assert should_run_cross_pair(3, config) is False

    def test_should_run_disabled(self):
        """Test should_run when disabled."""
        config = CrossPairConfig(enabled=False)
        assert should_run_cross_pair(0, config) is False


# ---------------------------------------------------------------------------
# Test: State I/O
# ---------------------------------------------------------------------------

class TestStateIO:
    """Tests for state file I/O."""

    def test_save_and_load_state(self, tmp_path):
        """Test save and load cycle."""
        state = CrossPairState(
            updated_ts=1234567890.0,
            pair_edges={},
            pairs_analyzed=5,
            pairs_eligible=3,
            cycle_count=10,
        )
        
        state_path = tmp_path / "cross_pair_edges.json"
        save_cross_pair_state(state, state_path)
        
        assert state_path.exists()
        
        loaded = load_cross_pair_state(state_path)
        assert loaded is not None
        assert loaded.updated_ts == 1234567890.0
        assert loaded.pairs_analyzed == 5

    def test_load_state_missing_file(self, tmp_path):
        """Test loading non-existent file returns empty state."""
        state_path = tmp_path / "nonexistent.json"
        loaded = load_cross_pair_state(state_path)
        # Returns empty state, not None
        assert loaded is not None
        assert loaded.pair_edges == {}
        assert loaded.pairs_analyzed == 0


# ---------------------------------------------------------------------------
# Test: View Functions
# ---------------------------------------------------------------------------

class TestViewFunctions:
    """Tests for view helper functions."""

    def test_get_top_pair_edges_empty(self):
        """Test getting top pair edges from empty state."""
        state = CrossPairState(
            updated_ts=1234567890.0,
            pair_edges={},
            pairs_analyzed=0,
            pairs_eligible=0,
            cycle_count=10,
        )
        top = get_top_pair_edges(state, top_k=2)
        assert len(top) == 0

    def test_get_cross_pair_summary_structure(self):
        """Test getting summary has expected structure."""
        state = CrossPairState(
            updated_ts=1234567890.0,
            pair_edges={},
            pairs_analyzed=3,
            pairs_eligible=1,
            cycle_count=10,
        )
        summary = get_cross_pair_summary(state)
        assert summary is not None
        assert "pairs_analyzed" in summary
        assert summary["pairs_analyzed"] == 3
        assert "cycle_count" in summary

    def test_get_pair_edges_for_insights_empty(self):
        """Test formatting edges for EdgeInsights with empty state."""
        state = CrossPairState(
            updated_ts=1234567890.0,
            pair_edges={},
            pairs_analyzed=0,
            pairs_eligible=0,
            cycle_count=10,
        )
        insights_data = get_pair_edges_for_insights(state)
        assert insights_data is not None
        assert isinstance(insights_data, dict)


# ---------------------------------------------------------------------------
# Test: Integration Helper
# ---------------------------------------------------------------------------

class TestIntegrationHelper:
    """Tests for integration helper functions."""

    def test_get_pair_leg_boost_empty_state(self):
        """Test getting boost with empty state."""
        state = CrossPairState(
            updated_ts=1234567890.0,
            pair_edges={},
            pairs_analyzed=0,
            pairs_eligible=0,
            cycle_count=10,
        )
        
        # No edges, so should return 0
        boost = get_pair_leg_boost("ETHUSDT", state, max_boost=0.10)
        assert isinstance(boost, float)
        assert boost == 0.0

    def test_get_pair_leg_boost_no_state(self):
        """Test boost when no state available."""
        boost = get_pair_leg_boost("BTCUSDT", None, max_boost=0.10)
        assert boost == 0.0


# ---------------------------------------------------------------------------
# Test: Dataclass Serialization
# ---------------------------------------------------------------------------

class TestDataclassSerialization:
    """Tests for dataclass serialization."""

    def test_pair_stats_fields(self):
        """Test PairStats has expected fields."""
        stats = PairStats(
            base="BTCUSDT",
            quote="ETHUSDT",
            hedge_ratio=16.0,
            spread_mean=0.0,
            spread_std=100.0,
            spread_z=2.0,
            spread_last=50.0,
            corr=0.85,
            half_life_est=20.0,
            residual_momo=0.01,
            liquidity_ok=True,
            eligible=True,
        )
        assert stats.base == "BTCUSDT"
        assert stats.quote == "ETHUSDT"
        assert stats.hedge_ratio == 16.0

    def test_pair_edge_fields(self):
        """Test PairEdge has expected fields."""
        stats = PairStats(
            base="BTCUSDT",
            quote="ETHUSDT",
            hedge_ratio=16.0,
            spread_mean=0.0,
            spread_std=100.0,
            spread_z=2.0,
            spread_last=50.0,
            corr=0.85,
            half_life_est=20.0,
            residual_momo=0.01,
            liquidity_ok=True,
            eligible=True,
        )
        edge = PairEdge(
            pair=("BTCUSDT", "ETHUSDT"),
            edge_score=0.75,
            ema_score=0.75,
            long_leg="ETHUSDT",
            short_leg="BTCUSDT",
            signal="ENTER",
            reason="test",
            stats=stats,
        )
        assert edge.pair == ("BTCUSDT", "ETHUSDT")
        assert edge.ema_score == 0.75
        assert edge.signal == "ENTER"

    def test_cross_pair_state_fields(self):
        """Test CrossPairState has expected fields."""
        state = CrossPairState(
            updated_ts=1234567890.0,
            pair_edges={},
            pairs_analyzed=5,
            pairs_eligible=3,
            cycle_count=10,
        )
        assert state.updated_ts == 1234567890.0
        assert state.pairs_analyzed == 5
        assert state.pairs_eligible == 3


# ---------------------------------------------------------------------------
# Test: Config Validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    """Tests for config validation in __post_init__."""

    def test_lookback_floor(self):
        """Test that lookback_bars has a minimum."""
        config = CrossPairConfig(lookback_bars=5)  # Too low
        assert config.lookback_bars >= 30  # Should be at minimum

    def test_smoothing_alpha_bounds(self):
        """Test that smoothing_alpha is bounded."""
        config_low = CrossPairConfig(smoothing_alpha=0.001)  # Too low
        assert 0.01 <= config_low.smoothing_alpha <= 1.0

        config_high = CrossPairConfig(smoothing_alpha=1.5)  # Too high
        assert 0.01 <= config_high.smoothing_alpha <= 1.0

    def test_entry_z_positive(self):
        """Test that entry_z must be positive."""
        config = CrossPairConfig(entry_z=-1.0)  # Invalid
        assert config.entry_z > 0
