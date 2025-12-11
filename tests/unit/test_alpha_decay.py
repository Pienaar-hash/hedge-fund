"""
Unit tests for Alpha Decay & Survival Curves (v7.8_P7).

Tests the core decay estimation, half-life computation, and survival probabilities.
"""
import math
import pytest
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_config():
    """Sample AlphaDecayConfig for testing."""
    from execution.alpha_decay import AlphaDecayConfig
    return AlphaDecayConfig(
        enabled=True,
        lookback_days=120,
        min_samples=10,
        smoothing_alpha=0.15,
        symbol_half_life_floor=5,
        symbol_half_life_ceiling=90,
        category_half_life_floor=10,
        factor_decay_floor=0.95,
        decay_penalty_strength=0.20,
        sentinel_x_integration=True,
        crisis_half_life_reduction=0.30,
        choppy_half_life_reduction=0.15,
    )


@pytest.fixture
def sample_history():
    """Sample AlphaDecayHistory for testing."""
    from execution.alpha_decay import AlphaDecayHistory
    history = AlphaDecayHistory()
    
    # Add some symbol edge scores over time
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()
    for i in range(30):
        # Decaying edge score: 0.8 * exp(-i/50)
        edge = 0.8 * math.exp(-i / 50)
        timestamp = base_time + i * 86400  # 1 day apart
        history.add_symbol_edge("BTCUSDT", edge, timestamp)
        history.add_symbol_edge("ETHUSDT", 0.6 + 0.01 * i, timestamp)  # Improving
    
    # Add factor PnL
    for i in range(30):
        pnl = 100 * math.exp(-i / 30)  # Decaying
        timestamp = base_time + i * 86400
        history.add_factor_pnl("trend", pnl, timestamp)
        history.add_factor_pnl("momentum", 50 + 5 * i, timestamp)  # Improving
    
    return history


# ---------------------------------------------------------------------------
# Configuration Tests
# ---------------------------------------------------------------------------


class TestAlphaDecayConfig:
    """Tests for AlphaDecayConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from execution.alpha_decay import AlphaDecayConfig
        cfg = AlphaDecayConfig()
        
        assert cfg.enabled is False
        assert cfg.lookback_days == 120
        assert cfg.min_samples == 30
        assert cfg.smoothing_alpha == 0.15
        assert cfg.symbol_half_life_floor == 5
        assert cfg.symbol_half_life_ceiling == 90
        assert cfg.decay_penalty_strength == 0.20

    def test_from_dict(self):
        """Test config creation from dict."""
        from execution.alpha_decay import AlphaDecayConfig
        
        d = {
            "enabled": True,
            "lookback_days": 60,
            "min_samples": 20,
            "decay_penalty_strength": 0.30,
        }
        cfg = AlphaDecayConfig.from_dict(d)
        
        assert cfg.enabled is True
        assert cfg.lookback_days == 60
        assert cfg.min_samples == 20
        assert cfg.decay_penalty_strength == 0.30

    def test_to_dict(self, sample_config):
        """Test config serialization."""
        d = sample_config.to_dict()
        
        assert d["enabled"] is True
        assert d["lookback_days"] == 120
        assert d["min_samples"] == 10
        assert "decay_penalty_strength" in d


# ---------------------------------------------------------------------------
# Decay Rate Computation Tests
# ---------------------------------------------------------------------------


class TestDecayRateComputation:
    """Tests for decay rate computation."""

    def test_compute_decay_rate_decaying(self):
        """Test decay rate for decaying values."""
        from execution.alpha_decay import compute_decay_rate
        
        # Exponentially decaying values
        values = [math.exp(-i / 20) for i in range(30)]
        times = [i * 86400 for i in range(30)]  # 1 day apart
        
        rate = compute_decay_rate(values, times, min_samples=10)
        
        # Rate should be negative (decaying)
        assert rate < 0
        # Should be approximately -1/20 = -0.05 (per day)
        assert abs(rate + 0.05) < 0.02

    def test_compute_decay_rate_improving(self):
        """Test decay rate for improving values."""
        from execution.alpha_decay import compute_decay_rate
        
        # Exponentially growing values
        values = [0.1 * math.exp(i / 30) for i in range(30)]
        times = [i * 86400 for i in range(30)]
        
        rate = compute_decay_rate(values, times, min_samples=10)
        
        # Rate should be positive (improving)
        assert rate > 0

    def test_compute_decay_rate_stable(self):
        """Test decay rate for stable values."""
        from execution.alpha_decay import compute_decay_rate
        
        # Constant values
        values = [0.5] * 30
        times = [i * 86400 for i in range(30)]
        
        rate = compute_decay_rate(values, times, min_samples=10)
        
        # Rate should be approximately zero
        assert abs(rate) < 0.01

    def test_compute_decay_rate_insufficient_samples(self):
        """Test decay rate with insufficient samples."""
        from execution.alpha_decay import compute_decay_rate
        
        values = [0.5, 0.4, 0.3]
        times = [0, 86400, 172800]
        
        rate = compute_decay_rate(values, times, min_samples=10)
        
        # Should return 0 for insufficient data
        assert rate == 0.0

    def test_compute_decay_rate_filters_non_positive(self):
        """Test that non-positive values are filtered."""
        from execution.alpha_decay import compute_decay_rate
        
        values = [0.5, -0.1, 0.4, 0, 0.3, 0.2] + [0.1] * 10
        times = list(range(16))
        
        # Should not raise, should compute from positive values
        rate = compute_decay_rate(values, times, min_samples=10)
        assert isinstance(rate, float)


# ---------------------------------------------------------------------------
# Half-Life Computation Tests
# ---------------------------------------------------------------------------


class TestHalfLifeComputation:
    """Tests for half-life computation."""

    def test_compute_half_life_normal(self):
        """Test half-life computation with normal decay rate."""
        from execution.alpha_decay import compute_half_life
        
        # decay_rate = -0.05 per day
        # half_life = ln(2) / 0.05 ≈ 13.86 days
        half_life = compute_half_life(-0.05, floor=5, ceiling=90)
        
        assert abs(half_life - 13.86) < 0.5

    def test_compute_half_life_floor_clamp(self):
        """Test half-life is clamped to floor."""
        from execution.alpha_decay import compute_half_life
        
        # Very fast decay
        half_life = compute_half_life(-1.0, floor=5, ceiling=90)
        
        assert half_life == 5

    def test_compute_half_life_ceiling_clamp(self):
        """Test half-life is clamped to ceiling."""
        from execution.alpha_decay import compute_half_life
        
        # Very slow decay
        half_life = compute_half_life(-0.001, floor=5, ceiling=90)
        
        assert half_life == 90

    def test_compute_half_life_zero_decay(self):
        """Test half-life with zero decay rate."""
        from execution.alpha_decay import compute_half_life
        
        half_life = compute_half_life(0.0, floor=5, ceiling=90)
        
        # Should return ceiling (infinite half-life)
        assert half_life == 90


# ---------------------------------------------------------------------------
# Survival Probability Tests
# ---------------------------------------------------------------------------


class TestSurvivalProbability:
    """Tests for survival probability computation."""

    def test_compute_survival_probability_at_half_life(self):
        """Test survival prob at exactly one half-life."""
        from execution.alpha_decay import compute_survival_probability
        
        prob = compute_survival_probability(time_elapsed=20, half_life=20)
        
        # At time = half_life, exp(-1) ≈ 0.368
        import math
        expected = math.exp(-1)
        assert abs(prob - expected) < 0.01

    def test_compute_survival_probability_at_zero(self):
        """Test survival prob at time zero."""
        from execution.alpha_decay import compute_survival_probability
        
        prob = compute_survival_probability(time_elapsed=0, half_life=20)
        
        assert prob == pytest.approx(1.0)

    def test_compute_survival_probability_at_two_half_lives(self):
        """Test survival prob at two half-lives."""
        from execution.alpha_decay import compute_survival_probability
        
        prob = compute_survival_probability(time_elapsed=40, half_life=20)
        
        # At time = 2*half_life, exp(-2) ≈ 0.135
        import math
        expected = math.exp(-2)
        assert abs(prob - expected) < 0.01

    def test_compute_survival_probability_bounds(self):
        """Test survival probability is bounded [0, 1]."""
        from execution.alpha_decay import compute_survival_probability
        
        prob1 = compute_survival_probability(time_elapsed=1000, half_life=10)
        prob2 = compute_survival_probability(time_elapsed=0, half_life=10)
        
        assert 0 <= prob1 <= 1
        assert 0 <= prob2 <= 1


# ---------------------------------------------------------------------------
# Sentinel-X Acceleration Tests
# ---------------------------------------------------------------------------


class TestSentinelXAcceleration:
    """Tests for Sentinel-X half-life acceleration."""

    def test_crisis_acceleration(self, sample_config):
        """Test half-life reduction in CRISIS regime."""
        from execution.alpha_decay import apply_sentinel_x_acceleration
        
        base_half_life = 30
        adjusted = apply_sentinel_x_acceleration(base_half_life, "CRISIS", sample_config)
        
        # Should be reduced by 30%
        expected = 30 * (1 - 0.30)
        assert abs(adjusted - expected) < 0.1

    def test_choppy_acceleration(self, sample_config):
        """Test half-life reduction in CHOPPY regime."""
        from execution.alpha_decay import apply_sentinel_x_acceleration
        
        base_half_life = 30
        adjusted = apply_sentinel_x_acceleration(base_half_life, "CHOPPY", sample_config)
        
        # Should be reduced by 15%
        expected = 30 * (1 - 0.15)
        assert abs(adjusted - expected) < 0.1

    def test_trend_up_no_change(self, sample_config):
        """Test no half-life change in TREND_UP regime."""
        from execution.alpha_decay import apply_sentinel_x_acceleration
        
        base_half_life = 30
        adjusted = apply_sentinel_x_acceleration(base_half_life, "TREND_UP", sample_config)
        
        assert adjusted == base_half_life

    def test_disabled_integration(self, sample_config):
        """Test no change when sentinel_x_integration is disabled."""
        from execution.alpha_decay import apply_sentinel_x_acceleration
        
        sample_config.sentinel_x_integration = False
        base_half_life = 30
        adjusted = apply_sentinel_x_acceleration(base_half_life, "CRISIS", sample_config)
        
        assert adjusted == base_half_life


# ---------------------------------------------------------------------------
# Factor Weight Multiplier Tests
# ---------------------------------------------------------------------------


class TestFactorWeightMultiplier:
    """Tests for factor weight multiplier computation."""

    def test_high_survival_no_penalty(self, sample_config):
        """Test no penalty with high survival probability."""
        from execution.alpha_decay import compute_factor_weight_multiplier
        
        mult = compute_factor_weight_multiplier(
            decay_rate=-0.01,
            survival_prob=1.0,
            config=sample_config,
        )
        
        # High survival = no penalty
        assert mult == pytest.approx(1.0)

    def test_low_survival_penalty(self, sample_config):
        """Test penalty with low survival probability."""
        from execution.alpha_decay import compute_factor_weight_multiplier
        
        mult = compute_factor_weight_multiplier(
            decay_rate=-0.1,
            survival_prob=0.3,
            config=sample_config,
        )
        
        # deterioration = 0.7, penalty = 0.20 * 0.7 = 0.14
        # multiplier = 1 - 0.14 = 0.86, but floor is 0.95
        # So result should be clamped to floor
        assert mult == sample_config.factor_decay_floor

    def test_floor_enforcement(self, sample_config):
        """Test floor is enforced."""
        from execution.alpha_decay import compute_factor_weight_multiplier
        
        mult = compute_factor_weight_multiplier(
            decay_rate=-1.0,
            survival_prob=0.0,  # 100% deterioration
            config=sample_config,
        )
        
        # Should be clamped to floor (0.95)
        assert mult == sample_config.factor_decay_floor


# ---------------------------------------------------------------------------
# Symbol Decay Stats Tests
# ---------------------------------------------------------------------------


class TestSymbolDecayStats:
    """Tests for symbol decay stats computation."""

    def test_compute_symbol_decay_stats_decaying(self, sample_config, sample_history):
        """Test stats for decaying symbol."""
        from execution.alpha_decay import compute_symbol_decay_stats
        
        edges, times = sample_history.get_symbol_series("BTCUSDT")
        stats = compute_symbol_decay_stats(
            symbol="BTCUSDT",
            edge_scores=edges,
            timestamps=times,
            config=sample_config,
            regime="NORMAL",
        )
        
        assert stats.symbol == "BTCUSDT"
        assert stats.decay_rate < 0  # Decaying
        assert 5 <= stats.half_life <= 90  # Within bounds
        assert 0 <= stats.survival_prob <= 1
        assert stats.deterioration_prob == pytest.approx(1 - stats.survival_prob)

    def test_compute_symbol_decay_stats_improving(self, sample_config, sample_history):
        """Test stats for improving symbol."""
        from execution.alpha_decay import compute_symbol_decay_stats
        
        edges, times = sample_history.get_symbol_series("ETHUSDT")
        stats = compute_symbol_decay_stats(
            symbol="ETHUSDT",
            edge_scores=edges,
            timestamps=times,
            config=sample_config,
            regime="NORMAL",
        )
        
        assert stats.symbol == "ETHUSDT"
        assert stats.decay_rate > 0  # Improving
        assert stats.trend_direction == "improving"

    def test_compute_symbol_decay_stats_insufficient_data(self, sample_config):
        """Test stats with insufficient data."""
        from execution.alpha_decay import compute_symbol_decay_stats
        
        stats = compute_symbol_decay_stats(
            symbol="NEWUSDT",
            edge_scores=[0.5, 0.4],
            timestamps=[0, 86400],
            config=sample_config,
            regime="NORMAL",
        )
        
        # Should return neutral stats
        assert stats.decay_rate == 0.0
        assert stats.half_life == sample_config.symbol_half_life_ceiling


# ---------------------------------------------------------------------------
# Category Decay Stats Tests
# ---------------------------------------------------------------------------


class TestCategoryDecayStats:
    """Tests for category decay stats computation."""

    def test_compute_category_decay_stats(self, sample_config):
        """Test category aggregation."""
        from execution.alpha_decay import (
            compute_category_decay_stats,
            SymbolDecayStats,
        )
        
        symbol_stats = {
            "BTCUSDT": SymbolDecayStats(
                symbol="BTCUSDT",
                decay_rate=-0.02,
                half_life=35,
                survival_prob=0.6,
                deterioration_prob=0.4,
                ema_edge_score=0.5,
            ),
            "ETHUSDT": SymbolDecayStats(
                symbol="ETHUSDT",
                decay_rate=-0.01,
                half_life=70,
                survival_prob=0.8,
                deterioration_prob=0.2,
                ema_edge_score=0.6,
            ),
        }
        
        cat_stats = compute_category_decay_stats(
            category="L1",
            symbol_stats=symbol_stats,
            category_symbols=["BTCUSDT", "ETHUSDT"],
            config=sample_config,
        )
        
        assert cat_stats.category == "L1"
        assert cat_stats.symbol_count == 2
        assert cat_stats.avg_symbol_survival == pytest.approx(0.7)  # (0.6 + 0.8) / 2
        assert cat_stats.weakest_symbol == "BTCUSDT"
        assert cat_stats.strongest_symbol == "ETHUSDT"


# ---------------------------------------------------------------------------
# Factor Decay Stats Tests
# ---------------------------------------------------------------------------


class TestFactorDecayStats:
    """Tests for factor decay stats computation."""

    def test_compute_factor_decay_stats(self, sample_config, sample_history):
        """Test factor decay computation."""
        from execution.alpha_decay import compute_factor_decay_stats
        
        pnl, times = sample_history.get_factor_series("trend")
        stats = compute_factor_decay_stats(
            factor="trend",
            pnl_contributions=pnl,
            timestamps=times,
            config=sample_config,
            ir_rolling=0.5,
        )
        
        assert stats.factor == "trend"
        assert stats.ir_rolling == 0.5
        assert 0 <= stats.adjusted_factor_weight_multiplier <= 1


# ---------------------------------------------------------------------------
# History Management Tests
# ---------------------------------------------------------------------------


class TestAlphaDecayHistory:
    """Tests for history management."""

    def test_add_and_get_symbol_edge(self):
        """Test adding and retrieving symbol edges."""
        from execution.alpha_decay import AlphaDecayHistory
        
        history = AlphaDecayHistory()
        history.add_symbol_edge("BTCUSDT", 0.5, 1000)
        history.add_symbol_edge("BTCUSDT", 0.6, 2000)
        
        edges, times = history.get_symbol_series("BTCUSDT")
        
        assert edges == [0.5, 0.6]
        assert times == [1000, 2000]

    def test_prune_old_samples(self):
        """Test pruning old samples."""
        from execution.alpha_decay import AlphaDecayHistory
        from datetime import datetime, timezone
        
        history = AlphaDecayHistory()
        now = datetime.now(timezone.utc).timestamp()
        
        # Add old and new samples
        history.add_symbol_edge("BTCUSDT", 0.5, now - 200000)  # Old
        history.add_symbol_edge("BTCUSDT", 0.6, now - 100)     # New
        
        # Prune samples older than 1 day
        history.prune_old_samples(86400)
        
        edges, _ = history.get_symbol_series("BTCUSDT")
        assert len(edges) == 1
        assert edges[0] == 0.6


# ---------------------------------------------------------------------------
# State I/O Tests
# ---------------------------------------------------------------------------


class TestStateIO:
    """Tests for state file I/O."""

    def test_alpha_decay_state_to_dict(self):
        """Test state serialization."""
        from execution.alpha_decay import (
            AlphaDecayState,
            SymbolDecayStats,
            CategoryDecayStats,
            FactorDecayStats,
        )
        
        state = AlphaDecayState(
            updated_ts="2024-01-01T00:00:00+00:00",
            cycle_count=5,
            symbols={
                "BTCUSDT": SymbolDecayStats(
                    symbol="BTCUSDT",
                    decay_rate=-0.02,
                    half_life=35,
                    survival_prob=0.6,
                    deterioration_prob=0.4,
                    ema_edge_score=0.5,
                )
            },
            categories={"L1": CategoryDecayStats(
                category="L1",
                decay_rate=-0.01,
                half_life=70,
                survival_prob=0.8,
                deterioration_prob=0.2,
            )},
            factors={"trend": FactorDecayStats(
                factor="trend",
                decay_rate=-0.01,
                survival_prob=0.7,
                adjusted_factor_weight_multiplier=0.95,
            )},
            avg_symbol_survival=0.6,
            overall_alpha_health=0.65,
            weakest_symbols=["BTCUSDT"],
            strongest_symbols=["ETHUSDT"],
        )
        
        d = state.to_dict()
        
        assert d["updated_ts"] == "2024-01-01T00:00:00+00:00"
        assert d["cycle_count"] == 5
        assert "BTCUSDT" in d["symbols"]
        assert d["symbols"]["BTCUSDT"]["decay_rate"] == -0.02
        assert d["avg_symbol_survival"] == 0.6

    def test_alpha_decay_state_from_dict(self):
        """Test state deserialization."""
        from execution.alpha_decay import AlphaDecayState
        
        d = {
            "updated_ts": "2024-01-01T00:00:00+00:00",
            "cycle_count": 5,
            "symbols": {
                "BTCUSDT": {
                    "symbol": "BTCUSDT",
                    "decay_rate": -0.02,
                    "half_life": 35,
                    "survival_prob": 0.6,
                    "deterioration_prob": 0.4,
                    "ema_edge_score": 0.5,
                }
            },
            "categories": {},
            "factors": {},
            "avg_symbol_survival": 0.6,
            "overall_alpha_health": 0.65,
            "weakest_symbols": ["BTCUSDT"],
            "strongest_symbols": ["ETHUSDT"],
            "meta": {},
        }
        
        state = AlphaDecayState.from_dict(d)
        
        assert state.updated_ts == "2024-01-01T00:00:00+00:00"
        assert state.cycle_count == 5
        assert "BTCUSDT" in state.symbols
        assert state.symbols["BTCUSDT"].decay_rate == -0.02


# ---------------------------------------------------------------------------
# Integration Helper Tests
# ---------------------------------------------------------------------------


class TestIntegrationHelpers:
    """Tests for downstream integration helpers."""

    def test_get_symbol_decay_penalty(self, sample_config):
        """Test symbol decay penalty computation."""
        from execution.alpha_decay import (
            get_symbol_decay_penalty,
            AlphaDecayState,
            SymbolDecayStats,
        )
        
        state = AlphaDecayState(
            symbols={
                "BTCUSDT": SymbolDecayStats(
                    symbol="BTCUSDT",
                    decay_rate=-0.02,
                    half_life=35,
                    survival_prob=0.6,
                    deterioration_prob=0.4,
                    ema_edge_score=0.5,
                )
            }
        )
        
        penalty = get_symbol_decay_penalty("BTCUSDT", state, sample_config)
        
        # penalty = 1 - 0.20 * 0.4 = 0.92
        assert penalty == pytest.approx(0.92, abs=0.01)

    def test_get_symbol_decay_penalty_disabled(self, sample_config):
        """Test no penalty when disabled."""
        from execution.alpha_decay import get_symbol_decay_penalty
        
        sample_config.enabled = False
        penalty = get_symbol_decay_penalty("BTCUSDT", None, sample_config)
        
        assert penalty == 1.0

    def test_get_alpha_router_adjustment(self, sample_config):
        """Test alpha router adjustment."""
        from execution.alpha_decay import (
            get_alpha_router_adjustment,
            AlphaDecayState,
        )
        
        state = AlphaDecayState(avg_symbol_survival=0.7)
        
        adj = get_alpha_router_adjustment(state, sample_config)
        
        # adj = 0.8 + 0.2 * 0.7 = 0.94
        assert adj == pytest.approx(0.94)

    def test_get_factor_decay_multipliers(self, sample_config):
        """Test factor decay multipliers."""
        from execution.alpha_decay import (
            get_factor_decay_multipliers,
            AlphaDecayState,
            FactorDecayStats,
        )
        
        state = AlphaDecayState(
            factors={
                "trend": FactorDecayStats(
                    factor="trend",
                    decay_rate=-0.01,
                    survival_prob=0.7,
                    adjusted_factor_weight_multiplier=0.95,
                ),
                "momentum": FactorDecayStats(
                    factor="momentum",
                    decay_rate=0.02,
                    survival_prob=0.9,
                    adjusted_factor_weight_multiplier=1.0,
                ),
            }
        )
        
        mults = get_factor_decay_multipliers(state, sample_config)
        
        assert mults["trend"] == 0.95
        assert mults["momentum"] == 1.0

    def test_get_conviction_decay_adjustment(self, sample_config):
        """Test conviction decay adjustment."""
        from execution.alpha_decay import (
            get_conviction_decay_adjustment,
            AlphaDecayState,
        )
        
        state = AlphaDecayState(overall_alpha_health=0.8)
        
        adj = get_conviction_decay_adjustment(state, sample_config)
        
        # adj = 0.85 + 0.15 * 0.8 = 0.97
        assert adj == pytest.approx(0.97)

    def test_get_alpha_decay_summary(self):
        """Test summary generation for EdgeInsights."""
        from execution.alpha_decay import (
            get_alpha_decay_summary,
            AlphaDecayState,
        )
        
        state = AlphaDecayState(
            avg_symbol_survival=0.65,
            overall_alpha_health=0.70,
            weakest_symbols=["XYZUSDT"],
            strongest_symbols=["BTCUSDT"],
            weakest_categories=["MEME"],
            weakest_factors=["momentum"],
        )
        
        summary = get_alpha_decay_summary(state)
        
        assert summary["overall_alpha_health"] == 0.70
        assert summary["avg_symbol_survival"] == 0.65
        assert "XYZUSDT" in summary["weakest_symbols"]
        assert "BTCUSDT" in summary["strongest_symbols"]


# ---------------------------------------------------------------------------
# EMA Smoothing Tests
# ---------------------------------------------------------------------------


class TestEMASmoothing:
    """Tests for EMA smoothing."""

    def test_ema_smooth(self):
        """Test EMA smoothing computation."""
        from execution.alpha_decay import ema_smooth
        
        # With alpha=0.5, current=10, prev=6
        result = ema_smooth(current=10, previous=6, alpha=0.5)
        
        # 0.5 * 10 + 0.5 * 6 = 8
        assert result == 8.0

    def test_ema_smooth_full_weight_current(self):
        """Test EMA with alpha=1 (full current weight)."""
        from execution.alpha_decay import ema_smooth
        
        result = ema_smooth(current=10, previous=6, alpha=1.0)
        
        assert result == 10.0

    def test_ema_smooth_full_weight_previous(self):
        """Test EMA with alpha=0 (full previous weight)."""
        from execution.alpha_decay import ema_smooth
        
        result = ema_smooth(current=10, previous=6, alpha=0.0)
        
        assert result == 6.0


# ---------------------------------------------------------------------------
# Trend Direction Classification Tests
# ---------------------------------------------------------------------------


class TestTrendDirection:
    """Tests for trend direction classification."""

    def test_classify_improving(self):
        """Test improving classification."""
        from execution.alpha_decay import classify_trend_direction
        
        assert classify_trend_direction(0.05) == "improving"

    def test_classify_declining(self):
        """Test declining classification."""
        from execution.alpha_decay import classify_trend_direction
        
        assert classify_trend_direction(-0.05) == "declining"

    def test_classify_stable(self):
        """Test stable classification."""
        from execution.alpha_decay import classify_trend_direction
        
        assert classify_trend_direction(0.0005) == "stable"
        assert classify_trend_direction(-0.0005) == "stable"
