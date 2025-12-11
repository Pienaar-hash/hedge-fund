"""
Unit tests for alpha_miner.py (v7.8_P4 — Autonomous Alpha Miner).

Tests cover:
- Config loading
- Feature extraction
- Scoring functions
- Candidate selection
- State I/O
- View functions
"""
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch, MagicMock

import pytest

from execution.alpha_miner import (
    # Config
    AlphaMinerConfig,
    load_alpha_miner_config,
    DEFAULT_WEIGHTS,
    # Features
    SymbolAlphaFeatures,
    extract_symbol_features,
    infer_category,
    compute_momentum,
    compute_volatility,
    compute_trend_consistency,
    compute_liquidity_score,
    compute_spread_quality,
    # Scoring
    compute_alpha_score,
    apply_ema_smoothing,
    # Candidates
    AlphaMinerCandidate,
    select_candidates,
    generate_candidate_reason,
    # State
    AlphaMinerState,
    load_alpha_miner_state,
    save_alpha_miner_state,
    # Runner
    should_run_miner,
    # Views
    get_top_candidates,
    get_miner_summary,
    get_candidates_by_category,
    # Helpers
    filter_excluded_symbols,
    load_current_universe,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_config() -> AlphaMinerConfig:
    """Default configuration for testing."""
    return AlphaMinerConfig(
        enabled=True,
        min_liquidity_usd=100_000.0,
        max_spread_pct=0.50,
        score_threshold=0.40,
        top_k=10,
        smoothing_alpha=0.20,
        lookback_bars=50,
    )


@pytest.fixture
def sample_features() -> SymbolAlphaFeatures:
    """Sample features for testing."""
    return SymbolAlphaFeatures(
        symbol="TESTUSDT",
        short_momo=0.05,
        long_momo=0.15,
        volatility=0.80,
        trend_consistency=0.65,
        liquidity_score=0.70,
        spread_quality=0.85,
        router_score=0.60,
        category_hint="L1_ALT",
        volume_24h=500_000.0,
        price=10.0,
        data_quality=1.0,
    )


@pytest.fixture
def sample_klines() -> List[List[float]]:
    """Generate sample OHLCV data (100 bars)."""
    # Create uptrending price data
    klines = []
    base_price = 100.0
    for i in range(100):
        ts = 1000000 + i * 14400  # 4h bars
        price = base_price * (1 + 0.002 * i)  # Gradual uptrend
        noise = 1 + (0.01 if i % 3 == 0 else -0.005)
        open_p = price * 0.999
        high_p = price * 1.01 * noise
        low_p = price * 0.99 * noise
        close_p = price * noise
        volume = 1000.0
        klines.append([ts, open_p, high_p, low_p, close_p, volume])
    return klines


@pytest.fixture
def sample_state() -> AlphaMinerState:
    """Sample state for testing."""
    feat = SymbolAlphaFeatures(
        symbol="NEWUSDT",
        short_momo=0.08,
        long_momo=0.12,
        volatility=0.75,
        trend_consistency=0.60,
        liquidity_score=0.65,
        spread_quality=0.80,
        router_score=0.55,
        category_hint="MEME",
        volume_24h=300_000.0,
        price=0.50,
        data_quality=1.0,
    )
    cand = AlphaMinerCandidate(
        symbol="NEWUSDT",
        score=0.55,
        ema_score=0.52,
        features=feat,
        reason="positive momentum; consistent trend",
        in_universe=False,
        first_seen_ts=1000000.0,
    )
    return AlphaMinerState(
        updated_ts=1000000.0,
        cycle_count=5,
        symbols_scanned=100,
        symbols_passed_filter=20,
        candidates=[cand],
        ema_scores={"NEWUSDT": 0.52, "OLDUSDT": 0.45},
        notes="test state",
        errors=[],
    )


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------


class TestAlphaMinerConfig:
    """Tests for AlphaMinerConfig dataclass."""

    def test_default_values(self):
        cfg = AlphaMinerConfig()
        assert cfg.enabled is False
        assert cfg.min_liquidity_usd == 500_000.0
        assert cfg.top_k == 20
        assert cfg.smoothing_alpha == 0.15

    def test_custom_values(self):
        cfg = AlphaMinerConfig(
            enabled=True,
            min_liquidity_usd=250_000.0,
            score_threshold=0.60,
        )
        assert cfg.enabled is True
        assert cfg.min_liquidity_usd == 250_000.0
        assert cfg.score_threshold == 0.60

    def test_post_init_validation(self):
        # Invalid smoothing_alpha gets reset
        cfg = AlphaMinerConfig(smoothing_alpha=2.0)
        assert cfg.smoothing_alpha == 0.15

        cfg2 = AlphaMinerConfig(smoothing_alpha=-0.5)
        assert cfg2.smoothing_alpha == 0.15

    def test_top_k_validation(self):
        cfg = AlphaMinerConfig(top_k=0)
        assert cfg.top_k == 20

    def test_score_threshold_validation(self):
        cfg = AlphaMinerConfig(score_threshold=1.5)
        assert cfg.score_threshold == 0.50

    def test_weights_default(self):
        cfg = AlphaMinerConfig()
        assert cfg.weights == DEFAULT_WEIGHTS


class TestLoadAlphaMinerConfig:
    """Tests for load_alpha_miner_config function."""

    def test_load_from_file(self, tmp_path):
        config_file = tmp_path / "strategy_config.json"
        config_file.write_text(json.dumps({
            "alpha_miner": {
                "enabled": True,
                "min_liquidity_usd": 300000.0,
                "score_threshold": 0.55,
            }
        }))

        cfg = load_alpha_miner_config(config_file)
        assert cfg.enabled is True
        assert cfg.min_liquidity_usd == 300000.0
        assert cfg.score_threshold == 0.55

    def test_missing_file_returns_disabled(self, tmp_path):
        missing = tmp_path / "missing.json"
        cfg = load_alpha_miner_config(missing)
        assert cfg.enabled is False

    def test_empty_section_returns_disabled(self, tmp_path):
        config_file = tmp_path / "strategy_config.json"
        config_file.write_text(json.dumps({"alpha_miner": {}}))

        cfg = load_alpha_miner_config(config_file)
        assert cfg.enabled is False

    def test_invalid_json_returns_disabled(self, tmp_path):
        config_file = tmp_path / "strategy_config.json"
        config_file.write_text("not json")

        cfg = load_alpha_miner_config(config_file)
        assert cfg.enabled is False


# ---------------------------------------------------------------------------
# Feature Extraction Tests
# ---------------------------------------------------------------------------


class TestInferCategory:
    """Tests for category inference heuristics."""

    def test_btc_category(self):
        assert infer_category("BTCUSDT") == "L1_MAJOR"

    def test_eth_category(self):
        assert infer_category("ETHUSDT") == "L1_MAJOR"

    def test_meme_category(self):
        assert infer_category("DOGEUSDT") == "MEME"
        assert infer_category("SHIBUSDT") == "MEME"
        assert infer_category("PEPEUSDT") == "MEME"

    def test_l1_alt_category(self):
        assert infer_category("SOLUSDT") == "L1_ALT"
        assert infer_category("AVAXUSDT") == "L1_ALT"

    def test_defi_category(self):
        assert infer_category("LINKUSDT") == "DEFI"
        assert infer_category("UNIUSDT") == "DEFI"

    def test_l2_category(self):
        assert infer_category("ARBUSDT") == "L2"
        assert infer_category("OPUSDT") == "L2"

    def test_unknown_defaults_to_other(self):
        assert infer_category("UNKNOWNUSDT") == "OTHER"
        assert infer_category("RANDUSDT") == "OTHER"


class TestComputeMomentum:
    """Tests for momentum calculation."""

    def test_positive_momentum(self):
        closes = [100, 101, 102, 103, 104, 105, 106, 107, 110]
        momo = compute_momentum(closes, 7)
        assert momo > 0

    def test_negative_momentum(self):
        closes = [110, 109, 108, 107, 106, 105, 104, 103, 100]
        momo = compute_momentum(closes, 7)
        assert momo < 0

    def test_insufficient_data(self):
        closes = [100, 101, 102]
        momo = compute_momentum(closes, 7)
        assert momo == 0.0

    def test_clamping(self):
        # Huge momentum gets clamped
        closes = [10, 10, 10, 10, 10, 10, 10, 10, 100]  # 10x increase
        momo = compute_momentum(closes, 7)
        assert momo == 1.0  # Clamped to max

    def test_zero_older_price(self):
        closes = [0, 0, 0, 0, 0, 0, 0, 0, 100]
        momo = compute_momentum(closes, 7)
        assert momo == 0.0


class TestComputeVolatility:
    """Tests for volatility calculation."""

    def test_volatility_calculation(self, sample_klines):
        closes = [row[4] for row in sample_klines]
        vol = compute_volatility(closes)
        assert vol > 0
        assert vol < 5.0  # Should be reasonable annualized vol

    def test_insufficient_data(self):
        closes = [100, 101, 102]
        vol = compute_volatility(closes)
        assert vol == 0.0

    def test_constant_prices_low_vol(self):
        closes = [100.0] * 50
        vol = compute_volatility(closes)
        # Very low vol due to no variance
        assert vol < 0.1  # Should be very low but annualization inflates it


class TestComputeTrendConsistency:
    """Tests for trend consistency calculation."""

    def test_uptrend_consistency(self):
        # Perfect uptrend
        closes = list(range(100, 140))  # 40 bars going up
        consistency = compute_trend_consistency(closes, 30)
        assert consistency > 0.8

    def test_downtrend_consistency(self):
        # Perfect downtrend
        closes = list(range(140, 100, -1))
        consistency = compute_trend_consistency(closes, 30)
        assert consistency < 0.2

    def test_choppy_market(self):
        # Alternating up/down
        closes = [100 + (i % 2) for i in range(40)]
        consistency = compute_trend_consistency(closes, 30)
        assert 0.3 < consistency < 0.7

    def test_insufficient_data(self):
        closes = [100, 101, 102]
        consistency = compute_trend_consistency(closes, 30)
        assert consistency == 0.5


class TestComputeLiquidityScore:
    """Tests for liquidity score calculation."""

    def test_zero_volume(self):
        score = compute_liquidity_score(0, 500_000)
        assert score == 0.0

    def test_below_min_volume(self):
        score = compute_liquidity_score(250_000, 500_000)
        assert score == 0.5  # Half of min

    def test_at_min_volume(self):
        score = compute_liquidity_score(500_000, 500_000)
        assert 0.5 <= score <= 0.6

    def test_high_volume(self):
        score = compute_liquidity_score(50_000_000, 500_000)  # 100x min
        assert score > 0.8

    def test_capped_at_one(self):
        score = compute_liquidity_score(1_000_000_000, 500_000)
        assert score <= 1.0


class TestComputeSpreadQuality:
    """Tests for spread quality calculation."""

    def test_zero_spread(self):
        quality = compute_spread_quality(0, 0.30)
        assert quality == 1.0

    def test_max_spread(self):
        quality = compute_spread_quality(0.30, 0.30)
        assert quality == 0.0

    def test_mid_spread(self):
        quality = compute_spread_quality(0.15, 0.30)
        assert quality == 0.5

    def test_above_max_spread(self):
        quality = compute_spread_quality(0.50, 0.30)
        assert quality == 0.0


class TestExtractSymbolFeatures:
    """Tests for full feature extraction."""

    def test_extract_with_klines(self, sample_config, sample_klines):
        feat = extract_symbol_features(
            symbol="TESTUSDT",
            config=sample_config,
            klines=sample_klines,
            orderbook=None,
            volume_24h=500_000.0,
            price=100.0,
        )
        assert feat is not None
        assert feat.symbol == "TESTUSDT"
        assert feat.long_momo != 0
        assert feat.volatility > 0
        assert feat.liquidity_score > 0

    def test_extract_insufficient_klines(self, sample_config):
        short_klines = [[1000, 100, 101, 99, 100, 1000]] * 10
        feat = extract_symbol_features(
            symbol="TESTUSDT",
            config=sample_config,
            klines=short_klines,
            orderbook=None,
            volume_24h=500_000.0,
            price=100.0,
        )
        assert feat is None  # Not enough data

    def test_extract_with_orderbook(self, sample_config, sample_klines):
        # Create tight spread orderbook (0.1% spread < max_spread_pct of 0.50%)
        orderbook = {
            "bids": [[99.95, 100]],
            "asks": [[100.05, 100]],
        }
        # Increase max_spread_pct to handle percentage calculations
        sample_config.max_spread_pct = 1.0  # Allow 1% spread
        feat = extract_symbol_features(
            symbol="TESTUSDT",
            config=sample_config,
            klines=sample_klines,
            orderbook=orderbook,
            volume_24h=500_000.0,
            price=100.0,
        )
        assert feat is not None
        # Spread is ~0.1%, max is 1%, so quality should be high
        assert feat.spread_quality > 0.8


# ---------------------------------------------------------------------------
# Scoring Tests
# ---------------------------------------------------------------------------


class TestComputeAlphaScore:
    """Tests for composite alpha score calculation."""

    def test_score_range(self, sample_features):
        score = compute_alpha_score(sample_features, DEFAULT_WEIGHTS)
        assert 0.0 <= score <= 1.0

    def test_high_quality_features(self):
        feat = SymbolAlphaFeatures(
            symbol="TEST",
            short_momo=0.5,
            long_momo=0.5,
            volatility=0.5,  # Low vol = good
            trend_consistency=0.9,
            liquidity_score=0.9,
            spread_quality=0.9,
            router_score=0.9,
            category_hint="L1_MAJOR",
            data_quality=1.0,
        )
        score = compute_alpha_score(feat, DEFAULT_WEIGHTS)
        assert score > 0.7

    def test_low_quality_features(self):
        feat = SymbolAlphaFeatures(
            symbol="TEST",
            short_momo=-0.5,
            long_momo=-0.5,
            volatility=2.0,  # High vol = bad
            trend_consistency=0.2,
            liquidity_score=0.1,
            spread_quality=0.1,
            router_score=0.1,
            category_hint="OTHER",
            data_quality=1.0,
        )
        score = compute_alpha_score(feat, DEFAULT_WEIGHTS)
        assert score < 0.4

    def test_data_quality_discount(self, sample_features):
        full_score = compute_alpha_score(sample_features, DEFAULT_WEIGHTS)
        
        # Same features but 50% data quality
        sample_features.data_quality = 0.5
        discounted_score = compute_alpha_score(sample_features, DEFAULT_WEIGHTS)
        
        assert discounted_score < full_score
        assert abs(discounted_score - full_score * 0.5) < 0.01


class TestApplyEmaSmoothing:
    """Tests for EMA smoothing."""

    def test_first_observation(self):
        current = {"A": 0.5, "B": 0.7}
        previous = {}
        ema = apply_ema_smoothing(current, previous, 0.2)
        assert ema["A"] == 0.5  # No smoothing on first
        assert ema["B"] == 0.7

    def test_smoothing_effect(self):
        current = {"A": 1.0}
        previous = {"A": 0.0}
        ema = apply_ema_smoothing(current, previous, 0.2)
        # EMA = 0.2 * 1.0 + 0.8 * 0.0 = 0.2
        assert abs(ema["A"] - 0.2) < 0.01

    def test_high_alpha_less_smoothing(self):
        current = {"A": 1.0}
        previous = {"A": 0.0}
        ema = apply_ema_smoothing(current, previous, 0.8)
        # EMA = 0.8 * 1.0 + 0.2 * 0.0 = 0.8
        assert abs(ema["A"] - 0.8) < 0.01


# ---------------------------------------------------------------------------
# Candidate Selection Tests
# ---------------------------------------------------------------------------


class TestGenerateCandidateReason:
    """Tests for reason generation."""

    def test_strong_momentum_reason(self, sample_features):
        sample_features.long_momo = 0.20
        reason = generate_candidate_reason(sample_features)
        assert "momentum" in reason.lower()

    def test_consistent_trend_reason(self, sample_features):
        sample_features.trend_consistency = 0.70
        reason = generate_candidate_reason(sample_features)
        assert "trend" in reason.lower()

    def test_good_liquidity_reason(self, sample_features):
        sample_features.liquidity_score = 0.80
        reason = generate_candidate_reason(sample_features)
        assert "liquidity" in reason.lower()

    def test_default_reason(self):
        feat = SymbolAlphaFeatures(symbol="TEST")
        reason = generate_candidate_reason(feat)
        assert "threshold" in reason.lower()


class TestSelectCandidates:
    """Tests for candidate selection."""

    def test_select_above_threshold(self, sample_config):
        sample_config.score_threshold = 0.40
        sample_config.top_k = 5

        scores = {"A": 0.6, "B": 0.5, "C": 0.3, "D": 0.7}
        ema_scores = {"A": 0.55, "B": 0.48, "C": 0.35, "D": 0.65}
        features = {
            sym: SymbolAlphaFeatures(symbol=sym, long_momo=0.1)
            for sym in ["A", "B", "C", "D"]
        }

        candidates = select_candidates(
            scores=scores,
            ema_scores=ema_scores,
            features_map=features,
            config=sample_config,
            current_universe=["C"],  # C already in universe
            previous_candidates=[],
        )

        # Only D, A, B should be candidates (ema >= 0.40, C is 0.35)
        syms = [c.symbol for c in candidates]
        assert "D" in syms
        assert "A" in syms
        assert "B" in syms
        assert "C" not in syms  # Below threshold

    def test_respects_top_k(self, sample_config):
        sample_config.top_k = 2
        sample_config.score_threshold = 0.30

        scores = {"A": 0.6, "B": 0.5, "C": 0.4, "D": 0.7}
        ema_scores = {"A": 0.55, "B": 0.48, "C": 0.42, "D": 0.65}
        features = {
            sym: SymbolAlphaFeatures(symbol=sym)
            for sym in scores
        }

        candidates = select_candidates(
            scores=scores,
            ema_scores=ema_scores,
            features_map=features,
            config=sample_config,
            current_universe=[],
            previous_candidates=[],
        )

        assert len(candidates) == 2
        assert candidates[0].symbol == "D"  # Highest score
        assert candidates[1].symbol == "A"

    def test_tracks_first_seen(self, sample_config):
        sample_config.top_k = 5
        sample_config.score_threshold = 0.30

        prev_cand = AlphaMinerCandidate(
            symbol="A",
            score=0.5,
            ema_score=0.5,
            features=SymbolAlphaFeatures(symbol="A"),
            reason="test",
            first_seen_ts=12345.0,
        )

        scores = {"A": 0.6, "B": 0.5}
        ema_scores = {"A": 0.55, "B": 0.48}
        features = {
            sym: SymbolAlphaFeatures(symbol=sym)
            for sym in scores
        }

        candidates = select_candidates(
            scores=scores,
            ema_scores=ema_scores,
            features_map=features,
            config=sample_config,
            current_universe=[],
            previous_candidates=[prev_cand],
        )

        a_cand = next(c for c in candidates if c.symbol == "A")
        b_cand = next(c for c in candidates if c.symbol == "B")
        
        assert a_cand.first_seen_ts == 12345.0  # Preserved
        assert b_cand.first_seen_ts > 0  # New timestamp


# ---------------------------------------------------------------------------
# State I/O Tests
# ---------------------------------------------------------------------------


class TestStateIO:
    """Tests for state persistence."""

    def test_save_and_load_state(self, sample_state, tmp_path):
        state_file = tmp_path / "alpha_miner.json"
        
        assert save_alpha_miner_state(sample_state, state_file)
        
        loaded = load_alpha_miner_state(state_file)
        assert loaded.cycle_count == sample_state.cycle_count
        assert loaded.symbols_scanned == sample_state.symbols_scanned
        assert len(loaded.candidates) == 1
        assert loaded.candidates[0].symbol == "NEWUSDT"

    def test_load_missing_file(self, tmp_path):
        missing = tmp_path / "missing.json"
        state = load_alpha_miner_state(missing)
        assert state.cycle_count == 0
        assert len(state.candidates) == 0

    def test_state_to_dict(self, sample_state):
        d = sample_state.to_dict()
        assert d["cycle_count"] == 5
        assert len(d["candidates"]) == 1
        assert d["candidates"][0]["symbol"] == "NEWUSDT"


# ---------------------------------------------------------------------------
# Runner Tests
# ---------------------------------------------------------------------------


class TestShouldRunMiner:
    """Tests for run scheduling."""

    def test_disabled_returns_false(self):
        cfg = AlphaMinerConfig(enabled=False, run_interval_cycles=10)
        state = AlphaMinerState()
        assert should_run_miner(50, cfg, state) is False

    def test_respects_interval(self):
        cfg = AlphaMinerConfig(enabled=True, run_interval_cycles=10)
        state = AlphaMinerState()
        
        assert should_run_miner(10, cfg, state) is True
        assert should_run_miner(11, cfg, state) is False
        assert should_run_miner(20, cfg, state) is True
        assert should_run_miner(25, cfg, state) is False


# ---------------------------------------------------------------------------
# Helper Function Tests
# ---------------------------------------------------------------------------


class TestFilterExcludedSymbols:
    """Tests for symbol filtering."""

    def test_excludes_patterns(self):
        symbols = ["BTCUSDT", "BUSDUSDT", "ETHUSDT", "TUSDUSDT"]
        exclude = ["BUSD", "TUSD"]
        result = filter_excluded_symbols(symbols, exclude, [])
        assert result == ["BTCUSDT", "ETHUSDT"]

    def test_excludes_universe(self):
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        exclude = []
        universe = ["BTCUSDT", "ETHUSDT"]
        result = filter_excluded_symbols(symbols, exclude, universe)
        assert result == ["SOLUSDT"]

    def test_combined_filters(self):
        symbols = ["BTCUSDT", "BUSDUSDT", "ETHUSDT", "SOLUSDT"]
        exclude = ["BUSD"]
        universe = ["BTCUSDT"]
        result = filter_excluded_symbols(symbols, exclude, universe)
        assert result == ["ETHUSDT", "SOLUSDT"]


class TestLoadCurrentUniverse:
    """Tests for universe loading."""

    def test_load_valid_universe(self, tmp_path):
        universe_file = tmp_path / "pairs_universe.json"
        universe_file.write_text(json.dumps({
            "BTCUSDT": {"tier": "CORE"},
            "ETHUSDT": {"tier": "CORE"},
        }))
        
        result = load_current_universe(universe_file)
        assert "BTCUSDT" in result
        assert "ETHUSDT" in result

    def test_load_missing_file(self, tmp_path):
        missing = tmp_path / "missing.json"
        result = load_current_universe(missing)
        assert result == []


# ---------------------------------------------------------------------------
# View Function Tests
# ---------------------------------------------------------------------------


class TestViewFunctions:
    """Tests for dashboard view functions."""

    def test_get_top_candidates(self, sample_state, tmp_path):
        state_file = tmp_path / "alpha_miner.json"
        save_alpha_miner_state(sample_state, state_file)
        
        with patch("execution.alpha_miner.DEFAULT_STATE_PATH", state_file):
            candidates = get_top_candidates(sample_state, top_k=5)
        
        assert len(candidates) == 1
        assert candidates[0]["symbol"] == "NEWUSDT"

    def test_get_miner_summary(self, sample_state):
        summary = get_miner_summary(sample_state)
        assert summary["cycle_count"] == 5
        assert summary["symbols_scanned"] == 100
        assert summary["num_candidates"] == 1
        assert summary["top_candidate"] == "NEWUSDT"

    def test_get_candidates_by_category(self, sample_state):
        by_cat = get_candidates_by_category(sample_state)
        assert "MEME" in by_cat
        assert len(by_cat["MEME"]) == 1
        assert by_cat["MEME"][0]["symbol"] == "NEWUSDT"
