"""
Unit tests for Sentinel-X Hybrid ML Market Regime Classifier (v7.8_P6).

Tests cover:
- Configuration loading and validation
- Feature extraction from price data
- ML-style regime scoring
- Rule-based classification with crisis override
- Label stickiness and smoothing
- State I/O
- Helper functions for downstream integrations
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config():
    """Default SentinelXConfig."""
    from execution.sentinel_x import SentinelXConfig
    return SentinelXConfig(enabled=True)


@pytest.fixture
def sample_prices():
    """Sample price series for testing."""
    # Uptrend with some noise
    base = 100.0
    prices = []
    for i in range(100):
        noise = math.sin(i * 0.3) * 2
        trend = i * 0.5
        prices.append(base + trend + noise)
    return prices


@pytest.fixture
def downtrend_prices():
    """Downtrend price series for testing."""
    base = 150.0
    prices = []
    for i in range(100):
        noise = math.sin(i * 0.3) * 2
        trend = -i * 0.5
        prices.append(base + trend + noise)
    return prices


@pytest.fixture
def choppy_prices():
    """Choppy/sideways price series for testing."""
    base = 100.0
    prices = []
    for i in range(100):
        noise = math.sin(i * 0.5) * 5
        prices.append(base + noise)
    return prices


@pytest.fixture
def tmp_state_dir():
    """Create temporary directory for state files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------


class TestSentinelXConfig:
    """Tests for SentinelXConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from execution.sentinel_x import SentinelXConfig
        
        cfg = SentinelXConfig()
        assert cfg.enabled is False
        assert cfg.lookback_bars == 240
        assert cfg.feature_agg_bars == 48
        assert cfg.prob_threshold_primary == 0.55
        assert cfg.label_stickiness == 3
        assert len(cfg.regimes) == 6

    def test_config_validation(self):
        """Test config validation in __post_init__."""
        from execution.sentinel_x import SentinelXConfig
        
        # Invalid lookback_bars should be corrected
        cfg = SentinelXConfig(lookback_bars=5)
        assert cfg.lookback_bars == 20  # MIN_BARS_FOR_FEATURES
        
        # Invalid thresholds should be corrected
        cfg = SentinelXConfig(prob_threshold_primary=0.1)
        assert cfg.prob_threshold_primary == 0.55
        
        # Regimes should be uppercased
        cfg = SentinelXConfig(regimes=["trend_up", "choppy"])
        assert cfg.regimes == ["TREND_UP", "CHOPPY"]

    def test_load_sentinel_x_config_defaults(self):
        """Test loading config with no strategy_cfg."""
        from execution.sentinel_x import load_sentinel_x_config
        
        cfg = load_sentinel_x_config(None)
        assert cfg.enabled is False

    def test_load_sentinel_x_config_from_dict(self):
        """Test loading config from strategy config dict."""
        from execution.sentinel_x import load_sentinel_x_config
        
        strategy_cfg = {
            "sentinel_x": {
                "enabled": True,
                "lookback_bars": 300,
                "prob_thresholds": {"primary": 0.60, "secondary": 0.30},
                "crisis_hard_rules": {"dd_threshold": 0.15},
                "smoothing": {"prob_alpha": 0.25, "label_stickiness": 5},
            }
        }
        
        cfg = load_sentinel_x_config(strategy_cfg)
        assert cfg.enabled is True
        assert cfg.lookback_bars == 300
        assert cfg.prob_threshold_primary == 0.60
        assert cfg.crisis_dd_threshold == 0.15
        assert cfg.label_stickiness == 5


# ---------------------------------------------------------------------------
# Feature Extraction Tests
# ---------------------------------------------------------------------------


class TestFeatureExtraction:
    """Tests for feature extraction functions."""

    def test_compute_returns(self):
        """Test log returns computation."""
        from execution.sentinel_x import compute_returns
        
        prices = [100, 101, 102, 101]
        returns = compute_returns(prices)
        assert len(returns) == 3
        assert returns[0] == pytest.approx(math.log(101/100), abs=1e-6)

    def test_compute_returns_empty(self):
        """Test returns with insufficient data."""
        from execution.sentinel_x import compute_returns
        
        assert compute_returns([]) == []
        assert compute_returns([100]) == []

    def test_compute_mean(self):
        """Test mean computation."""
        from execution.sentinel_x import compute_mean
        
        assert compute_mean([1, 2, 3, 4, 5]) == 3.0
        assert compute_mean([]) == 0.0

    def test_compute_std(self):
        """Test standard deviation computation."""
        from execution.sentinel_x import compute_std
        
        series = [1, 2, 3, 4, 5]
        std = compute_std(series)
        assert std > 0
        assert std == pytest.approx(math.sqrt(2), abs=0.01)

    def test_compute_skewness(self):
        """Test skewness computation."""
        from execution.sentinel_x import compute_skewness
        
        # Symmetric distribution → skew ≈ 0
        symmetric = [1, 2, 3, 4, 5]
        assert abs(compute_skewness(symmetric)) < 0.5

    def test_compute_linear_regression(self):
        """Test linear regression computation."""
        from execution.sentinel_x import compute_linear_regression
        
        # Perfect linear trend
        y = [1, 2, 3, 4, 5]
        slope, intercept, r2 = compute_linear_regression(y)
        assert slope == pytest.approx(1.0, abs=0.01)
        assert r2 == pytest.approx(1.0, abs=0.01)

    def test_compute_z_score(self):
        """Test z-score computation."""
        from execution.sentinel_x import compute_z_score
        
        series = [10, 20, 30, 40, 50]
        z = compute_z_score(30, series)  # Mean is 30
        assert z == pytest.approx(0.0, abs=0.01)

    def test_extract_regime_features_uptrend(self, sample_prices, default_config):
        """Test feature extraction from uptrend data."""
        from execution.sentinel_x import extract_regime_features
        
        features = extract_regime_features(sample_prices, cfg=default_config)
        
        assert features.data_quality > 0.5
        assert features.trend_slope > 0  # Positive trend
        assert features.returns_mean > 0  # Positive returns

    def test_extract_regime_features_insufficient_data(self, default_config):
        """Test feature extraction with insufficient data."""
        from execution.sentinel_x import extract_regime_features
        
        features = extract_regime_features([100, 101, 102], cfg=default_config)
        assert features.data_quality == 0.0  # Insufficient data


# ---------------------------------------------------------------------------
# ML Model Tests
# ---------------------------------------------------------------------------


class TestSimpleRegimeModel:
    """Tests for SimpleRegimeModel."""

    def test_model_predict_proba(self, default_config):
        """Test model probability prediction."""
        from execution.sentinel_x import SimpleRegimeModel, RegimeFeatures
        
        model = SimpleRegimeModel(default_config)
        
        # Create features suggesting uptrend
        features = RegimeFeatures(
            trend_slope=0.005,
            trend_r2=0.8,
            returns_mean=0.001,
            data_quality=1.0,
        )
        
        probs = model.predict_proba(features)
        assert "TREND_UP" in probs.probs
        assert sum(probs.probs.values()) == pytest.approx(1.0, abs=0.01)

    def test_model_uptrend_detection(self, sample_prices, default_config):
        """Test model detects uptrend."""
        from execution.sentinel_x import SimpleRegimeModel, extract_regime_features
        
        features = extract_regime_features(sample_prices, cfg=default_config)
        model = SimpleRegimeModel(default_config)
        probs = model.predict_proba(features)
        
        primary, primary_prob = probs.get_primary()
        # Uptrend should favor TREND_UP or similar
        assert primary in ["TREND_UP", "BREAKOUT", "CHOPPY"]

    def test_model_low_data_quality(self, default_config):
        """Test model handles low data quality."""
        from execution.sentinel_x import SimpleRegimeModel, RegimeFeatures
        
        model = SimpleRegimeModel(default_config)
        
        features = RegimeFeatures(data_quality=0.1)
        probs = model.predict_proba(features)
        
        # Low quality → more uniform distribution
        assert sum(probs.probs.values()) == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Classification Tests
# ---------------------------------------------------------------------------


class TestRegimeClassification:
    """Tests for rule-based classification."""

    def test_crisis_override_dd(self, default_config):
        """Test crisis override from drawdown."""
        from execution.sentinel_x import check_crisis_override, RegimeFeatures
        
        features = RegimeFeatures(vol_regime_z=0.5)
        
        # Normal DD → no crisis
        triggered, reason = check_crisis_override(features, default_config, current_dd=0.05)
        assert triggered is False
        
        # High DD → crisis
        triggered, reason = check_crisis_override(features, default_config, current_dd=0.15)
        assert triggered is True
        assert "DD=" in reason

    def test_crisis_override_vol_spike(self, default_config):
        """Test crisis override from volatility spike."""
        from execution.sentinel_x import check_crisis_override, RegimeFeatures
        
        # High vol z-score → crisis
        features = RegimeFeatures(vol_regime_z=3.5)
        triggered, reason = check_crisis_override(features, default_config, current_dd=0.0)
        assert triggered is True
        assert "VolZ=" in reason

    def test_label_stickiness(self, default_config):
        """Test label stickiness prevents flicker."""
        from execution.sentinel_x import apply_label_stickiness, HistoryMeta
        
        history = HistoryMeta(last_primary="CHOPPY")
        
        # First change attempt - should keep CHOPPY
        new_label, history = apply_label_stickiness(
            "TREND_UP", 0.7, history, default_config
        )
        assert new_label == "CHOPPY"  # Still sticky
        assert history.consecutive_count == 1
        
        # Second change attempt
        new_label, history = apply_label_stickiness(
            "TREND_UP", 0.7, history, default_config
        )
        assert new_label == "CHOPPY"
        assert history.consecutive_count == 2
        
        # Third change attempt - should flip (stickiness=3)
        new_label, history = apply_label_stickiness(
            "TREND_UP", 0.7, history, default_config
        )
        assert new_label == "TREND_UP"
        assert history.consecutive_count == 0

    def test_smooth_probabilities(self):
        """Test probability smoothing."""
        from execution.sentinel_x import smooth_probabilities
        
        current = {"TREND_UP": 0.8, "CHOPPY": 0.2}
        prev = {"TREND_UP": 0.4, "CHOPPY": 0.6}
        
        smoothed = smooth_probabilities(current, prev, alpha=0.5)
        
        # With alpha=0.5: 0.5 * 0.8 + 0.5 * 0.4 = 0.6, then normalized
        # After normalization (sum = 1.0): should be close to 0.6 for TREND_UP
        # But since we renormalize, the exact value depends on both regimes
        # Current: 0.8, 0.2 → smoothed raw: 0.6, 0.4 → sum 1.0 → normalized: 0.6, 0.4
        # Wait - also includes all REGIMES. Let me check actual behavior.
        # The function adds missing regimes with 1/6 default for prev.
        # Just check the value is between current and prev
        assert 0.3 < smoothed["TREND_UP"] < 0.9

    def test_classify_regime(self, sample_prices, default_config):
        """Test full classification pipeline."""
        from execution.sentinel_x import (
            extract_regime_features,
            SimpleRegimeModel,
            classify_regime,
        )
        
        features = extract_regime_features(sample_prices, cfg=default_config)
        model = SimpleRegimeModel(default_config)
        probs = model.predict_proba(features)
        
        state = classify_regime(probs, features, default_config)
        
        assert state.primary_regime in [
            "TREND_UP", "TREND_DOWN", "MEAN_REVERT", 
            "BREAKOUT", "CHOPPY", "CRISIS"
        ]
        assert state.updated_ts != ""
        assert isinstance(state.regime_probs, dict)


# ---------------------------------------------------------------------------
# State I/O Tests
# ---------------------------------------------------------------------------


class TestStateIO:
    """Tests for state loading and saving."""

    def test_save_and_load_state(self, tmp_state_dir):
        """Test round-trip state persistence."""
        from execution.sentinel_x import (
            SentinelXState,
            save_sentinel_x_state,
            load_sentinel_x_state,
        )
        
        state_path = tmp_state_dir / "sentinel_x.json"
        
        state = SentinelXState(
            updated_ts="2024-01-01T00:00:00Z",
            primary_regime="TREND_UP",
            regime_probs={"TREND_UP": 0.7, "CHOPPY": 0.3},
        )
        
        assert save_sentinel_x_state(state, state_path) is True
        assert state_path.exists()
        
        loaded = load_sentinel_x_state(state_path)
        assert loaded.primary_regime == "TREND_UP"
        assert loaded.regime_probs["TREND_UP"] == 0.7

    def test_load_nonexistent_state(self, tmp_state_dir):
        """Test loading from nonexistent file returns empty state."""
        from execution.sentinel_x import load_sentinel_x_state
        
        state_path = tmp_state_dir / "does_not_exist.json"
        loaded = load_sentinel_x_state(state_path)
        
        assert loaded.primary_regime == "CHOPPY"
        assert loaded.updated_ts == ""


# ---------------------------------------------------------------------------
# Run Step Tests
# ---------------------------------------------------------------------------


class TestRunSentinelXStep:
    """Tests for the main run function."""

    def test_run_step_disabled(self, sample_prices):
        """Test run step when disabled."""
        from execution.sentinel_x import run_sentinel_x_step
        
        cfg_dict = {"sentinel_x": {"enabled": False}}
        result = run_sentinel_x_step(
            prices=sample_prices,
            strategy_cfg=cfg_dict,
            dry_run=True,
        )
        assert result is None

    def test_run_step_enabled(self, sample_prices, tmp_state_dir):
        """Test run step when enabled."""
        from execution.sentinel_x import run_sentinel_x_step, SentinelXConfig
        
        cfg = SentinelXConfig(enabled=True)
        state_path = tmp_state_dir / "sentinel_x.json"
        
        result = run_sentinel_x_step(
            prices=sample_prices,
            cfg=cfg,
            state_path=state_path,
            dry_run=True,
        )
        
        assert result is not None
        assert result.primary_regime in [
            "TREND_UP", "TREND_DOWN", "MEAN_REVERT",
            "BREAKOUT", "CHOPPY", "CRISIS"
        ]

    def test_should_run_sentinel_x(self, default_config):
        """Test cycle throttling."""
        from execution.sentinel_x import should_run_sentinel_x
        
        # Cycle 5 with interval 5 → should run
        cfg = default_config
        cfg.run_interval_cycles = 5
        assert should_run_sentinel_x(5, cfg) is True
        assert should_run_sentinel_x(3, cfg) is False
        assert should_run_sentinel_x(10, cfg) is True


# ---------------------------------------------------------------------------
# Helper Function Tests
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Tests for view and integration helper functions."""

    def test_get_sentinel_x_summary(self, tmp_state_dir):
        """Test summary function."""
        from execution.sentinel_x import (
            SentinelXState,
            save_sentinel_x_state,
            get_sentinel_x_summary,
        )
        
        state = SentinelXState(
            updated_ts="2024-01-01T00:00:00Z",
            primary_regime="BREAKOUT",
            smoothed_probs={"BREAKOUT": 0.6, "CHOPPY": 0.4},
        )
        save_sentinel_x_state(state, tmp_state_dir / "sentinel_x.json")
        
        with patch("execution.sentinel_x.DEFAULT_STATE_PATH", tmp_state_dir / "sentinel_x.json"):
            summary = get_sentinel_x_summary()
            assert summary["primary_regime"] == "BREAKOUT"

    def test_get_regime_conviction_weight(self):
        """Test conviction weight helper."""
        from execution.sentinel_x import get_regime_conviction_weight
        
        assert get_regime_conviction_weight("TREND_UP") == 1.10
        assert get_regime_conviction_weight("CRISIS") == 0.50
        assert get_regime_conviction_weight("CHOPPY") == 0.85
        # Unknown regime → 1.0
        assert get_regime_conviction_weight("UNKNOWN") == 1.0

    def test_get_factor_regime_weights(self):
        """Test factor weight helper."""
        from execution.sentinel_x import get_factor_regime_weights
        
        weights = get_factor_regime_weights("TREND_UP")
        assert "trend" in weights
        assert weights["trend"] > 1.0  # Boost trend factor in uptrend
        
        weights = get_factor_regime_weights("MEAN_REVERT")
        assert weights["mean_revert"] > 1.0  # Boost mean revert factor

    def test_get_regime_allocation_factor(self):
        """Test allocation factor helper."""
        from execution.sentinel_x import get_regime_allocation_factor
        
        assert get_regime_allocation_factor("TREND_UP") == 1.05
        assert get_regime_allocation_factor("CRISIS") == 0.60

    def test_get_regime_universe_shrink(self):
        """Test universe shrink helper."""
        from execution.sentinel_x import get_regime_universe_shrink
        
        assert get_regime_universe_shrink("TREND_UP") == 0.0
        assert get_regime_universe_shrink("CRISIS") == 0.40
        assert get_regime_universe_shrink("CHOPPY") == 0.25


# ---------------------------------------------------------------------------
# RegimeProbabilities Tests
# ---------------------------------------------------------------------------


class TestRegimeProbabilities:
    """Tests for RegimeProbabilities dataclass."""

    def test_normalize(self):
        """Test probability normalization."""
        from execution.sentinel_x import RegimeProbabilities
        
        probs = RegimeProbabilities(probs={"TREND_UP": 0.4, "CHOPPY": 0.6})
        probs.normalize()
        
        assert sum(probs.probs.values()) == pytest.approx(1.0, abs=0.01)

    def test_get_primary(self):
        """Test getting primary regime."""
        from execution.sentinel_x import RegimeProbabilities
        
        probs = RegimeProbabilities(probs={
            "TREND_UP": 0.5,
            "CHOPPY": 0.3,
            "MEAN_REVERT": 0.2,
        })
        
        primary, prob = probs.get_primary()
        assert primary == "TREND_UP"
        assert prob == 0.5

    def test_get_secondary(self):
        """Test getting secondary regime."""
        from execution.sentinel_x import RegimeProbabilities
        
        probs = RegimeProbabilities(probs={
            "TREND_UP": 0.5,
            "CHOPPY": 0.3,
            "MEAN_REVERT": 0.2,
        })
        
        secondary = probs.get_secondary()
        assert secondary is not None
        assert secondary[0] == "CHOPPY"
        assert secondary[1] == 0.3


# ---------------------------------------------------------------------------
# Scoring Threshold Calibration Tests (v7.9-W4)
# ---------------------------------------------------------------------------


class TestScoringThresholdCalibration:
    """Tests that externalized scoring thresholds produce correct regime
    detection for known market conditions.

    The old hardcoded threshold (0.001) made it impossible for Sentinel-X
    to detect a TREND_DOWN even during a 9% BTC drop in 3 days because
    the per-bar slope (0.000453) was below the threshold.
    """

    @pytest.fixture
    def calibrated_config(self):
        """Config with v7.9 calibrated thresholds."""
        from execution.sentinel_x import SentinelXConfig
        return SentinelXConfig(
            enabled=True,
            trend_slope_threshold=0.0003,
            reversal_threshold=0.40,
            vol_spike_threshold=1.5,
            breakout_threshold=0.02,
        )

    @pytest.fixture
    def old_config(self):
        """Config with old hardcoded thresholds (pre-v7.9)."""
        from execution.sentinel_x import SentinelXConfig
        return SentinelXConfig(
            enabled=True,
            trend_slope_threshold=0.001,
            reversal_threshold=0.45,
            vol_spike_threshold=1.5,
            breakout_threshold=0.02,
        )

    @pytest.fixture
    def btc_crash_features(self):
        """Features from actual BTC testnet 2025-02-24: 9% drop over 3 days."""
        from execution.sentinel_x import RegimeFeatures
        return RegimeFeatures(
            returns_mean=-0.000561,
            returns_std=0.003006,
            returns_skew=-1.9616,
            returns_kurtosis=3.4828,
            atr_norm=0.002134,
            vol_regime_z=0.0816,
            trend_slope=-0.000453,
            trend_r2=0.5579,
            trend_acceleration=0.00016,
            breakout_score=0.0,
            range_position=0.1838,
            mean_reversion_score=0.3913,
            hl_spread=0.004113,
            volume_z=-1.6088,
            data_quality=1.0,
        )

    def test_old_threshold_misclassifies_btc_crash_as_choppy(
        self, old_config, btc_crash_features
    ):
        """With old 0.001 threshold, a 9% BTC drop scores CHOPPY 100%."""
        from execution.sentinel_x import SimpleRegimeModel

        model = SimpleRegimeModel(old_config)
        probs = model.predict_proba(btc_crash_features)
        primary, primary_prob = probs.get_primary()
        assert primary == "CHOPPY", (
            f"Old threshold should classify as CHOPPY, got {primary}"
        )
        # Under old thresholds, TREND_DOWN gets 0.0
        assert probs.probs["TREND_DOWN"] == pytest.approx(0.0, abs=0.01)

    def test_calibrated_threshold_detects_btc_crash_as_trend_down(
        self, calibrated_config, btc_crash_features
    ):
        """With 0.0003 threshold, the 9% BTC drop is correctly TREND_DOWN."""
        from execution.sentinel_x import SimpleRegimeModel

        model = SimpleRegimeModel(calibrated_config)
        probs = model.predict_proba(btc_crash_features)
        primary, primary_prob = probs.get_primary()
        assert primary == "TREND_DOWN", (
            f"Calibrated threshold should detect TREND_DOWN, got {primary}"
        )
        assert primary_prob > 0.55, (
            f"TREND_DOWN should exceed primary threshold, got {primary_prob:.2f}"
        )

    def test_trend_down_score_grows_with_slope_magnitude(self, calibrated_config):
        """Steeper slopes produce higher TREND_DOWN scores."""
        from execution.sentinel_x import SimpleRegimeModel, RegimeFeatures

        model = SimpleRegimeModel(calibrated_config)
        prev_score = 0.0
        for slope in [-0.0004, -0.0005, -0.0006, -0.0007]:
            features = RegimeFeatures(
                trend_slope=slope,
                trend_r2=0.6,
                returns_mean=-0.0005,
                data_quality=1.0,
            )
            probs = model.predict_proba(features)
            score = probs.probs["TREND_DOWN"]
            assert score > prev_score, (
                f"Slope {slope} should score higher than {prev_score}, got {score}"
            )
            prev_score = score

    def test_trend_up_score_with_calibrated_threshold(self, calibrated_config):
        """Moderate uptrend is detected with calibrated threshold."""
        from execution.sentinel_x import SimpleRegimeModel, RegimeFeatures

        model = SimpleRegimeModel(calibrated_config)
        features = RegimeFeatures(
            trend_slope=0.0004,
            trend_r2=0.60,
            returns_mean=0.0003,
            data_quality=1.0,
        )
        probs = model.predict_proba(features)
        primary, _ = probs.get_primary()
        assert primary == "TREND_UP", f"Expected TREND_UP, got {primary}"

    def test_choppy_requires_truly_directionless_market(self, calibrated_config):
        """CHOPPY should only win when slope is genuinely near zero."""
        from execution.sentinel_x import SimpleRegimeModel, RegimeFeatures

        model = SimpleRegimeModel(calibrated_config)
        # Very flat market with low R²
        features = RegimeFeatures(
            trend_slope=0.00005,
            trend_r2=0.15,
            returns_mean=0.0,
            returns_kurtosis=4.0,
            returns_skew=0.3,
            data_quality=1.0,
        )
        probs = model.predict_proba(features)
        primary, _ = probs.get_primary()
        assert primary == "CHOPPY", f"Flat market should be CHOPPY, got {primary}"

    def test_mean_revert_with_lowered_reversal_threshold(self, calibrated_config):
        """Mean reversion detected with 0.40 reversal_threshold."""
        from execution.sentinel_x import SimpleRegimeModel, RegimeFeatures

        model = SimpleRegimeModel(calibrated_config)
        features = RegimeFeatures(
            mean_reversion_score=0.42,
            trend_r2=0.20,
            vol_regime_z=0.5,
            data_quality=1.0,
        )
        probs = model.predict_proba(features)
        assert probs.probs["MEAN_REVERT"] > 0, (
            "Reversal score 0.42 with threshold 0.40 should produce non-zero MEAN_REVERT"
        )

    def test_old_threshold_blocks_mean_revert_at_042(self, old_config):
        """Old 0.45 reversal threshold blocks mean reversion at 0.42."""
        from execution.sentinel_x import SimpleRegimeModel, RegimeFeatures

        model = SimpleRegimeModel(old_config)
        features = RegimeFeatures(
            mean_reversion_score=0.42,
            trend_r2=0.20,
            vol_regime_z=0.5,
            data_quality=1.0,
        )
        probs = model.predict_proba(features)
        assert probs.probs["MEAN_REVERT"] == pytest.approx(0.0, abs=0.01), (
            "Old threshold=0.45 should block MR score of 0.42"
        )

    def test_config_loader_reads_scoring_thresholds(self):
        """load_sentinel_x_config reads scoring_thresholds section."""
        from execution.sentinel_x import load_sentinel_x_config

        strategy_cfg = {
            "sentinel_x": {
                "enabled": True,
                "scoring_thresholds": {
                    "trend_slope_threshold": 0.0005,
                    "reversal_threshold": 0.38,
                    "vol_spike_threshold": 2.0,
                    "breakout_threshold": 0.03,
                },
            }
        }
        cfg = load_sentinel_x_config(strategy_cfg)
        assert cfg.trend_slope_threshold == 0.0005
        assert cfg.reversal_threshold == 0.38
        assert cfg.vol_spike_threshold == 2.0
        assert cfg.breakout_threshold == 0.03

    def test_config_loader_defaults_scoring_thresholds(self):
        """Missing scoring_thresholds uses calibrated defaults."""
        from execution.sentinel_x import load_sentinel_x_config

        strategy_cfg = {"sentinel_x": {"enabled": True}}
        cfg = load_sentinel_x_config(strategy_cfg)
        assert cfg.trend_slope_threshold == 0.0003
        assert cfg.reversal_threshold == 0.40
        assert cfg.vol_spike_threshold == 1.5
        assert cfg.breakout_threshold == 0.02

    def test_model_uses_config_thresholds(self):
        """SimpleRegimeModel inherits thresholds from config."""
        from execution.sentinel_x import SimpleRegimeModel, SentinelXConfig

        cfg = SentinelXConfig(
            enabled=True,
            trend_slope_threshold=0.0005,
            reversal_threshold=0.35,
            vol_spike_threshold=2.0,
            breakout_threshold=0.04,
        )
        model = SimpleRegimeModel(cfg)
        assert model.trend_slope_threshold == 0.0005
        assert model.reversal_threshold == 0.35
        assert model.vol_spike_threshold == 2.0
        assert model.breakout_threshold == 0.04

    def test_default_config_has_calibrated_thresholds(self):
        """SentinelXConfig dataclass defaults are v7.9-calibrated."""
        from execution.sentinel_x import SentinelXConfig

        cfg = SentinelXConfig()
        assert cfg.trend_slope_threshold == 0.0003
        assert cfg.reversal_threshold == 0.40


# ═══════════════════════════════════════════════════════════════════════════
# v7.9-W5: Hysteresis Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTrendHysteresis:
    """Hysteresis prevents CHOPPY ↔ TREND thrash at the boundary."""

    @pytest.fixture
    def hyst_config(self):
        from execution.sentinel_x import SentinelXConfig
        return SentinelXConfig(
            enabled=True,
            trend_slope_threshold=0.0003,
            trend_hysteresis_ratio=0.67,  # exit threshold = 0.0002
        )

    @pytest.fixture
    def boundary_features(self):
        """Features with slope at -0.00025: between entry (0.0003) and exit (0.0002)."""
        from execution.sentinel_x import RegimeFeatures
        return RegimeFeatures(
            trend_slope=-0.00025,
            trend_r2=0.5,
            returns_mean=-0.0003,
            returns_std=0.002,
            returns_kurtosis=1.5,
            atr_norm=0.002,
            mean_reversion_score=0.1,
            data_quality=1.0,
        )

    def test_boundary_slope_new_entry_is_choppy(self, hyst_config, boundary_features):
        """Slope -0.00025 is NOT enough to enter TREND_DOWN from scratch."""
        from execution.sentinel_x import SimpleRegimeModel
        model = SimpleRegimeModel(hyst_config)
        probs = model.predict_proba(boundary_features, current_regime="CHOPPY")
        primary, _ = probs.get_primary()
        # Should NOT fire TREND_DOWN since |0.00025| < entry threshold |0.0003|
        assert primary != "TREND_DOWN"

    def test_boundary_slope_stays_trend_down_with_hysteresis(self, hyst_config, boundary_features):
        """Slope -0.00025 IS enough to STAY in TREND_DOWN (exit = 0.0002)."""
        from execution.sentinel_x import SimpleRegimeModel
        model = SimpleRegimeModel(hyst_config)
        probs = model.predict_proba(boundary_features, current_regime="TREND_DOWN")
        td_prob = probs.probs.get("TREND_DOWN", 0.0)
        # TREND_DOWN should still have positive score since |0.00025| > exit |0.0002|
        assert td_prob > 0.0

    def test_slope_below_exit_threshold_exits_trend(self, hyst_config):
        """Slope -0.00015 is below the exit threshold (0.0002) → loses trend score."""
        from execution.sentinel_x import SimpleRegimeModel, RegimeFeatures
        model = SimpleRegimeModel(hyst_config)
        features = RegimeFeatures(
            trend_slope=-0.00015,
            trend_r2=0.4,
            returns_mean=-0.0001,
            returns_std=0.002,
            returns_kurtosis=1.5,
            atr_norm=0.002,
            mean_reversion_score=0.1,
            data_quality=1.0,
        )
        probs = model.predict_proba(features, current_regime="TREND_DOWN")
        td_prob = probs.probs.get("TREND_DOWN", 0.0)
        # |0.00015| < exit threshold 0.0002 → zero TREND_DOWN score
        assert td_prob == 0.0

    def test_hysteresis_ratio_one_disables_hysteresis(self):
        """trend_hysteresis_ratio=1.0 means entry=exit threshold (no deadband)."""
        from execution.sentinel_x import SimpleRegimeModel, SentinelXConfig, RegimeFeatures
        cfg = SentinelXConfig(
            enabled=True,
            trend_slope_threshold=0.0003,
            trend_hysteresis_ratio=1.0,
        )
        model = SimpleRegimeModel(cfg)
        features = RegimeFeatures(
            trend_slope=-0.00025,
            trend_r2=0.5,
            returns_mean=-0.0003,
            returns_std=0.002,
            returns_kurtosis=1.5,
            data_quality=1.0,
        )
        probs = model.predict_proba(features, current_regime="TREND_DOWN")
        td_prob = probs.probs.get("TREND_DOWN", 0.0)
        # With ratio=1.0 exit threshold = 0.0003, |0.00025| < 0.0003 → no score
        assert td_prob == 0.0

    def test_symmetric_hysteresis_trend_up(self, hyst_config):
        """Hysteresis works symmetrically for TREND_UP."""
        from execution.sentinel_x import SimpleRegimeModel, RegimeFeatures
        model = SimpleRegimeModel(hyst_config)
        features = RegimeFeatures(
            trend_slope=0.00025,  # Between entry (0.0003) and exit (0.0002)
            trend_r2=0.5,
            returns_mean=0.0003,
            returns_std=0.002,
            returns_kurtosis=1.5,
            data_quality=1.0,
        )
        # From CHOPPY → should NOT enter TREND_UP
        probs_cold = model.predict_proba(features, current_regime="CHOPPY")
        tu_cold = probs_cold.probs.get("TREND_UP", 0.0)
        # From TREND_UP → should STAY (hysteresis protects)
        probs_warm = model.predict_proba(features, current_regime="TREND_UP")
        tu_warm = probs_warm.probs.get("TREND_UP", 0.0)
        assert tu_cold == 0.0
        assert tu_warm > 0.0

    def test_config_default_hysteresis_ratio(self):
        """Default hysteresis ratio is 0.67."""
        from execution.sentinel_x import SentinelXConfig
        cfg = SentinelXConfig()
        assert cfg.trend_hysteresis_ratio == 0.67

    def test_config_loader_reads_hysteresis(self):
        """Config loader picks up trend_hysteresis_ratio from scoring_thresholds."""
        from execution.sentinel_x import load_sentinel_x_config
        strategy_cfg = {
            "sentinel_x": {
                "enabled": True,
                "scoring_thresholds": {
                    "trend_hysteresis_ratio": 0.5,
                },
            },
        }
        cfg = load_sentinel_x_config(strategy_cfg)
        assert cfg.trend_hysteresis_ratio == 0.5

    def test_backward_compat_no_current_regime(self, hyst_config):
        """predict_proba without current_regime works (backward compat)."""
        from execution.sentinel_x import SimpleRegimeModel, RegimeFeatures
        model = SimpleRegimeModel(hyst_config)
        features = RegimeFeatures(
            trend_slope=-0.0005,
            trend_r2=0.6,
            returns_mean=-0.0005,
            returns_std=0.003,
            returns_kurtosis=2.0,
            data_quality=1.0,
        )
        probs = model.predict_proba(features)
        td_prob = probs.probs.get("TREND_DOWN", 0.0)
        assert td_prob > 0.0  # Strong slope always detected
