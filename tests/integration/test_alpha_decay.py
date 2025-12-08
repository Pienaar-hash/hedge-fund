"""
Tests for Alpha Decay Engine (v7.5_A1)
"""

import math
import time
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import json

from execution.intel.symbol_score_v6 import (
    AlphaDecayConfig,
    load_alpha_decay_config,
    compute_alpha_decay,
    apply_alpha_decay,
    get_signal_age_minutes,
    load_signal_timestamps,
    save_signal_timestamps,
    update_signal_timestamp,
    build_alpha_decay_snapshot,
)


# ===========================================================================
# Tests: Alpha Decay Config
# ===========================================================================

class TestAlphaDecayConfig:
    def test_default_config_values(self):
        """Default config should have expected values."""
        config = AlphaDecayConfig()
        
        assert config.enabled == True
        assert config.half_life_minutes == 45.0
        assert config.min_decay_multiplier == 0.35

    def test_load_config_from_strategy_config(self):
        """Should load config from strategy config dict."""
        strategy_cfg = {
            "alpha_decay": {
                "enabled": True,
                "half_life_minutes": 60,
                "min_decay_multiplier": 0.25,
            }
        }
        
        config = load_alpha_decay_config(strategy_cfg)
        
        assert config.enabled == True
        assert config.half_life_minutes == 60.0
        assert config.min_decay_multiplier == 0.25

    def test_load_config_disabled(self):
        """Should correctly load disabled config."""
        strategy_cfg = {
            "alpha_decay": {
                "enabled": False,
            }
        }
        
        config = load_alpha_decay_config(strategy_cfg)
        
        assert config.enabled == False


# ===========================================================================
# Tests: Compute Alpha Decay
# ===========================================================================

class TestComputeAlphaDecay:
    def test_fresh_signal_no_decay(self):
        """Fresh signal (age=0) should have decay=1.0."""
        config = AlphaDecayConfig(enabled=True, half_life_minutes=45, min_decay_multiplier=0.35)
        
        decay = compute_alpha_decay(age_minutes=0.0, config=config)
        
        assert decay == 1.0

    def test_half_life_decay(self):
        """Signal at half-life age should have decay=0.5."""
        config = AlphaDecayConfig(enabled=True, half_life_minutes=45, min_decay_multiplier=0.0)
        
        decay = compute_alpha_decay(age_minutes=45.0, config=config)
        
        assert pytest.approx(decay, rel=0.01) == 0.5

    def test_double_half_life_decay(self):
        """Signal at 2x half-life should have decay=0.25."""
        config = AlphaDecayConfig(enabled=True, half_life_minutes=30, min_decay_multiplier=0.0)
        
        decay = compute_alpha_decay(age_minutes=60.0, config=config)
        
        assert pytest.approx(decay, rel=0.01) == 0.25

    def test_minimum_decay_clamp(self):
        """Decay should be clamped to minimum multiplier."""
        config = AlphaDecayConfig(enabled=True, half_life_minutes=45, min_decay_multiplier=0.35)
        
        # Very old signal that would decay below minimum
        decay = compute_alpha_decay(age_minutes=500.0, config=config)
        
        assert decay == 0.35

    def test_disabled_returns_one(self):
        """Disabled config should return decay=1.0."""
        config = AlphaDecayConfig(enabled=False, half_life_minutes=45, min_decay_multiplier=0.35)
        
        decay = compute_alpha_decay(age_minutes=100.0, config=config)
        
        assert decay == 1.0

    def test_zero_half_life_returns_one(self):
        """Zero half-life should return decay=1.0."""
        config = AlphaDecayConfig(enabled=True, half_life_minutes=0, min_decay_multiplier=0.35)
        
        decay = compute_alpha_decay(age_minutes=100.0, config=config)
        
        assert decay == 1.0

    def test_decay_is_monotonic(self):
        """Decay should decrease monotonically with age."""
        config = AlphaDecayConfig(enabled=True, half_life_minutes=45, min_decay_multiplier=0.0)
        
        ages = [0, 15, 30, 45, 60, 90, 120]
        decays = [compute_alpha_decay(age, config) for age in ages]
        
        # Each subsequent decay should be less than or equal to previous
        for i in range(1, len(decays)):
            assert decays[i] <= decays[i-1]


# ===========================================================================
# Tests: Signal Timestamps
# ===========================================================================

class TestSignalTimestamps:
    def test_save_and_load_timestamps(self):
        """Should save and load timestamps correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "signal_timestamps.json"
            
            timestamps = {
                "BTCUSDT:LONG": 1700000000.0,
                "ETHUSDT:SHORT": 1700001000.0,
            }
            
            save_signal_timestamps(timestamps, path)
            loaded = load_signal_timestamps(path)
            
            assert loaded["BTCUSDT:LONG"] == 1700000000.0
            assert loaded["ETHUSDT:SHORT"] == 1700001000.0

    def test_load_nonexistent_returns_empty(self):
        """Loading nonexistent file should return empty dict."""
        result = load_signal_timestamps("/nonexistent/path.json")
        
        assert result == {}

    def test_update_signal_timestamp(self):
        """Should update individual signal timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "signal_timestamps.json"
            
            update_signal_timestamp("BTCUSDT", "LONG", 1700000000.0, path)
            update_signal_timestamp("ETHUSDT", "SHORT", 1700001000.0, path)
            
            loaded = load_signal_timestamps(path)
            
            assert "BTCUSDT:LONG" in loaded
            assert "ETHUSDT:SHORT" in loaded


class TestGetSignalAgeMinutes:
    def test_returns_zero_for_fresh_signal(self):
        """Fresh signal with no timestamp should return 0."""
        timestamps = {}
        
        age = get_signal_age_minutes("BTCUSDT", "LONG", timestamps, now=1700000000.0)
        
        assert age == 0.0

    def test_calculates_age_correctly(self):
        """Should calculate age in minutes correctly."""
        timestamps = {"BTCUSDT:LONG": 1700000000.0}
        now = 1700000000.0 + (30 * 60)  # 30 minutes later
        
        age = get_signal_age_minutes("BTCUSDT", "LONG", timestamps, now)
        
        assert age == 30.0

    def test_handles_case_insensitive_keys(self):
        """Should handle uppercase key lookup."""
        timestamps = {"BTCUSDT:LONG": 1700000000.0}
        now = 1700000000.0 + (15 * 60)
        
        # Try lowercase input
        age = get_signal_age_minutes("btcusdt", "long", timestamps, now)
        
        # Key is normalized to uppercase
        assert age == 15.0


# ===========================================================================
# Tests: Apply Alpha Decay
# ===========================================================================

class TestApplyAlphaDecay:
    def test_applies_decay_to_hybrid_score(self):
        """Should apply decay multiplier to hybrid score."""
        timestamps = {"BTCUSDT:LONG": 1700000000.0}
        now = 1700000000.0 + (45 * 60)  # 45 minutes = 1 half-life
        
        config = AlphaDecayConfig(enabled=True, half_life_minutes=45, min_decay_multiplier=0.0)
        
        result = apply_alpha_decay(
            hybrid_score=0.8,
            symbol="BTCUSDT",
            direction="LONG",
            config=config,
            timestamps=timestamps,
            now=now,
        )
        
        # At 1 half-life, decay should be ~0.5
        assert pytest.approx(result["decayed_score"], rel=0.05) == 0.4  # 0.8 * 0.5

    def test_returns_decay_metadata(self):
        """Should return decay metadata."""
        result = apply_alpha_decay(
            hybrid_score=0.7,
            symbol="ETHUSDT",
            direction="SHORT",
        )
        
        assert "decayed_score" in result
        assert "decay_multiplier" in result
        assert "age_minutes" in result
        assert "decay_enabled" in result

    def test_disabled_returns_unchanged_score(self):
        """Disabled decay should return unchanged score."""
        config = AlphaDecayConfig(enabled=False)
        
        result = apply_alpha_decay(
            hybrid_score=0.75,
            symbol="BTCUSDT",
            direction="LONG",
            config=config,
        )
        
        assert result["decayed_score"] == 0.75
        assert result["decay_multiplier"] == 1.0


# ===========================================================================
# Tests: Build Alpha Decay Snapshot
# ===========================================================================

class TestBuildAlphaDecaySnapshot:
    def test_builds_snapshot_for_symbols(self):
        """Should build decay snapshot for all symbols."""
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        
        with patch("execution.intel.symbol_score_v6.load_signal_timestamps") as mock_load:
            mock_load.return_value = {}  # All fresh signals
            
            snapshot = build_alpha_decay_snapshot(symbols)
        
        assert "updated_ts" in snapshot
        assert "config" in snapshot
        assert "symbols" in snapshot
        assert "BTCUSDT" in snapshot["symbols"]
        assert "ETHUSDT" in snapshot["symbols"]
        assert "SOLUSDT" in snapshot["symbols"]

    def test_snapshot_includes_both_directions(self):
        """Snapshot should include both LONG and SHORT directions."""
        with patch("execution.intel.symbol_score_v6.load_signal_timestamps") as mock_load:
            mock_load.return_value = {}
            
            snapshot = build_alpha_decay_snapshot(["BTCUSDT"])
        
        btc_data = snapshot["symbols"]["BTCUSDT"]
        assert "long" in btc_data
        assert "short" in btc_data

    def test_snapshot_respects_config(self):
        """Snapshot should use provided config."""
        config = AlphaDecayConfig(enabled=True, half_life_minutes=60, min_decay_multiplier=0.25)
        
        with patch("execution.intel.symbol_score_v6.load_signal_timestamps") as mock_load:
            mock_load.return_value = {}
            
            snapshot = build_alpha_decay_snapshot(["BTCUSDT"], config=config)
        
        assert snapshot["config"]["half_life_minutes"] == 60
        assert snapshot["config"]["min_decay_multiplier"] == 0.25


# ===========================================================================
# Tests: Integration with Hybrid Score
# ===========================================================================

class TestAlphaDecayHybridIntegration:
    """Test that alpha decay integrates correctly with hybrid scoring."""
    
    def test_decay_affects_threshold_comparison(self):
        """Decayed score should be used for threshold comparison."""
        # A score of 0.6 that decays to 0.3 should fail a 0.5 threshold
        config = AlphaDecayConfig(enabled=True, half_life_minutes=45, min_decay_multiplier=0.0)
        timestamps = {"BTCUSDT:LONG": 1700000000.0}
        now = 1700000000.0 + (45 * 60)  # 1 half-life
        
        result = apply_alpha_decay(
            hybrid_score=0.6,
            symbol="BTCUSDT",
            direction="LONG",
            config=config,
            timestamps=timestamps,
            now=now,
        )
        
        # Original score 0.6, decayed to ~0.3
        assert result["decayed_score"] < 0.5  # Would fail 0.5 threshold

    def test_at_minimum_flag_set_correctly(self):
        """at_minimum flag should be set when decay reaches floor."""
        config = AlphaDecayConfig(enabled=True, half_life_minutes=45, min_decay_multiplier=0.35)
        timestamps = {"BTCUSDT:LONG": 1700000000.0}
        now = 1700000000.0 + (500 * 60)  # Very old signal
        
        result = apply_alpha_decay(
            hybrid_score=0.8,
            symbol="BTCUSDT",
            direction="LONG",
            config=config,
            timestamps=timestamps,
            now=now,
        )
        
        assert result["at_minimum"] == True
        assert result["decay_multiplier"] == 0.35
