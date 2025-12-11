"""
Integration tests for alpha_miner.py (v7.8_P4 — Autonomous Alpha Miner).

Tests cover:
- Full miner step execution
- Edge scanner integration
- State file contract
- Dashboard panel integration
- Manifest alignment
"""
import json
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_state_dir(tmp_path) -> Path:
    """Create temp state directory."""
    state_dir = tmp_path / "logs" / "state"
    state_dir.mkdir(parents=True)
    return state_dir


@pytest.fixture
def tmp_config_dir(tmp_path) -> Path:
    """Create temp config directory."""
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    return config_dir


@pytest.fixture
def sample_strategy_config(tmp_config_dir) -> Path:
    """Create sample strategy_config.json with alpha_miner enabled."""
    config_file = tmp_config_dir / "strategy_config.json"
    config_file.write_text(json.dumps({
        "alpha_miner": {
            "enabled": True,
            "min_liquidity_usd": 100000.0,
            "max_spread_pct": 0.50,
            "score_threshold": 0.40,
            "top_k": 10,
            "smoothing_alpha": 0.20,
            "lookback_bars": 50,
            "run_interval_cycles": 5,
            "exclude_patterns": ["BUSD", "TUSD"],
        }
    }))
    return config_file


@pytest.fixture
def sample_universe(tmp_config_dir) -> Path:
    """Create sample pairs_universe.json."""
    universe_file = tmp_config_dir / "pairs_universe.json"
    universe_file.write_text(json.dumps({
        "BTCUSDT": {"tier": "CORE"},
        "ETHUSDT": {"tier": "CORE"},
    }))
    return universe_file


@pytest.fixture
def mock_exchange_symbols() -> List[str]:
    """Mock list of exchange symbols."""
    return [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT",
        "LINKUSDT", "AVAXUSDT", "MATICUSDT", "ARBUSDT",
        "PEPEUSDT", "SHIBUSDT", "WIFUSDT",
        "BUSDUSDT", "TUSDUSDT",  # Stablecoins to filter
    ]


@pytest.fixture
def mock_klines() -> List[List[float]]:
    """Generate mock OHLCV data (100 bars, uptrend)."""
    klines = []
    base_price = 100.0
    for i in range(100):
        ts = 1000000 + i * 14400
        price = base_price * (1 + 0.002 * i)
        klines.append([ts, price * 0.99, price * 1.01, price * 0.98, price, 10000.0])
    return klines


# ---------------------------------------------------------------------------
# Full Miner Step Integration Tests
# ---------------------------------------------------------------------------


class TestAlphaMinerStepIntegration:
    """Integration tests for run_alpha_miner_step."""

    def test_full_miner_step(
        self,
        tmp_state_dir,
        sample_strategy_config,
        sample_universe,
        mock_exchange_symbols,
        mock_klines,
    ):
        """Test complete miner step with mocked exchange data."""
        from execution.alpha_miner import (
            run_alpha_miner_step,
            load_alpha_miner_config,
            AlphaMinerConfig,
        )

        state_file = tmp_state_dir / "alpha_miner.json"

        # Create config
        cfg = AlphaMinerConfig(
            enabled=True,
            min_liquidity_usd=10000.0,  # Low threshold for test
            score_threshold=0.30,
            top_k=5,
            lookback_bars=50,
        )

        # Mock exchange calls
        with patch("execution.alpha_miner.get_all_exchange_symbols") as mock_syms, \
             patch("execution.alpha_miner.load_current_universe") as mock_univ, \
             patch("execution.exchange_utils.get_klines") as mock_kl:
            
            mock_syms.return_value = mock_exchange_symbols
            mock_univ.return_value = ["BTCUSDT", "ETHUSDT"]
            mock_kl.return_value = mock_klines

            state = run_alpha_miner_step(
                config=cfg,
                state_path=state_file,
                fetch_orderbook=False,
                dry_run=False,
            )

        # Verify state was created
        assert state.cycle_count == 1
        assert state.symbols_scanned > 0
        assert state.updated_ts > 0

        # Verify state file was written
        assert state_file.exists()
        saved = json.loads(state_file.read_text())
        assert "candidates" in saved
        assert "ema_scores" in saved

    def test_miner_step_disabled(self, tmp_state_dir):
        """Test that disabled miner returns empty state."""
        from execution.alpha_miner import run_alpha_miner_step, AlphaMinerConfig

        cfg = AlphaMinerConfig(enabled=False)
        state = run_alpha_miner_step(config=cfg)

        assert state.notes == "disabled"
        assert len(state.candidates) == 0

    def test_miner_step_dry_run(
        self,
        tmp_state_dir,
        mock_exchange_symbols,
        mock_klines,
    ):
        """Test dry_run mode doesn't write state."""
        from execution.alpha_miner import run_alpha_miner_step, AlphaMinerConfig

        state_file = tmp_state_dir / "alpha_miner.json"
        cfg = AlphaMinerConfig(
            enabled=True,
            min_liquidity_usd=10000.0,
            lookback_bars=50,
        )

        with patch("execution.alpha_miner.get_all_exchange_symbols") as mock_syms, \
             patch("execution.alpha_miner.load_current_universe") as mock_univ, \
             patch("execution.exchange_utils.get_klines") as mock_kl:
            
            mock_syms.return_value = mock_exchange_symbols
            mock_univ.return_value = []
            mock_kl.return_value = mock_klines

            state = run_alpha_miner_step(
                config=cfg,
                state_path=state_file,
                dry_run=True,  # Don't save
            )

        # State returned but file not written
        assert state.symbols_scanned > 0
        assert not state_file.exists()


# ---------------------------------------------------------------------------
# Edge Scanner Integration Tests
# ---------------------------------------------------------------------------


class TestEdgeScannerIntegration:
    """Tests for edge_scanner.py integration."""

    def test_update_alpha_miner_function_exists(self):
        """Verify update_alpha_miner is exported from edge_scanner."""
        from execution.edge_scanner import update_alpha_miner
        assert callable(update_alpha_miner)

    def test_update_alpha_miner_disabled(self, tmp_state_dir):
        """Test that disabled config skips miner update."""
        from execution.edge_scanner import update_alpha_miner, EdgeInsights

        # Create disabled config
        with patch("execution.alpha_miner.load_alpha_miner_config") as mock_cfg:
            from execution.alpha_miner import AlphaMinerConfig
            mock_cfg.return_value = AlphaMinerConfig(enabled=False)

            # Should not raise, just return
            edge_insights = EdgeInsights()
            update_alpha_miner(edge_insights, strategy_config=None)

    def test_update_alpha_miner_respects_cycle_interval(self, tmp_state_dir):
        """Test that miner only runs every N cycles."""
        from execution.edge_scanner import update_alpha_miner, EdgeInsights
        from execution import edge_scanner

        # Reset cycle counter
        edge_scanner._ALPHA_MINER_CYCLE_COUNT = 0

        with patch("execution.alpha_miner.load_alpha_miner_config") as mock_cfg, \
             patch("execution.alpha_miner.run_alpha_miner_step") as mock_run:
            
            from execution.alpha_miner import AlphaMinerConfig, AlphaMinerState
            mock_cfg.return_value = AlphaMinerConfig(
                enabled=True,
                run_interval_cycles=5,
            )
            mock_run.return_value = AlphaMinerState()

            edge_insights = EdgeInsights()
            
            # Run 4 times - should not trigger miner
            for _ in range(4):
                update_alpha_miner(edge_insights)
            
            assert mock_run.call_count == 0

            # 5th time should trigger
            update_alpha_miner(edge_insights)
            assert mock_run.call_count == 1


# ---------------------------------------------------------------------------
# State File Contract Tests
# ---------------------------------------------------------------------------


class TestStateFileContract:
    """Tests verifying state file adheres to v7_manifest.json contract."""

    def test_state_file_schema(self, tmp_state_dir):
        """Verify state file has required fields from manifest."""
        from execution.alpha_miner import (
            AlphaMinerState,
            AlphaMinerCandidate,
            SymbolAlphaFeatures,
            save_alpha_miner_state,
        )

        state_file = tmp_state_dir / "alpha_miner.json"

        feat = SymbolAlphaFeatures(
            symbol="TESTUSDT",
            short_momo=0.05,
            long_momo=0.10,
            volatility=0.80,
            trend_consistency=0.65,
        )
        cand = AlphaMinerCandidate(
            symbol="TESTUSDT",
            score=0.55,
            ema_score=0.52,
            features=feat,
            reason="test",
        )
        state = AlphaMinerState(
            updated_ts=time.time(),
            cycle_count=3,
            symbols_scanned=50,
            symbols_passed_filter=10,
            candidates=[cand],
            ema_scores={"TESTUSDT": 0.52},
            notes="test notes",
            errors=["error1"],
        )

        save_alpha_miner_state(state, state_file)
        saved = json.loads(state_file.read_text())

        # Required fields from v7_manifest.json
        assert "updated_ts" in saved
        assert "cycle_count" in saved
        assert "symbols_scanned" in saved
        assert "symbols_passed_filter" in saved
        assert "candidates" in saved
        assert "ema_scores" in saved
        assert "notes" in saved
        assert "errors" in saved

        # Candidate structure
        assert len(saved["candidates"]) == 1
        cand_dict = saved["candidates"][0]
        assert "symbol" in cand_dict
        assert "score" in cand_dict
        assert "ema_score" in cand_dict
        assert "features" in cand_dict
        assert "reason" in cand_dict

    def test_manifest_alignment(self):
        """Verify alpha_miner is listed in v7_manifest.json."""
        manifest_path = Path("v7_manifest.json")
        if not manifest_path.exists():
            pytest.skip("v7_manifest.json not found")

        manifest = json.loads(manifest_path.read_text())
        state_files = manifest.get("state_files", {})

        assert "alpha_miner" in state_files
        miner_entry = state_files["alpha_miner"]
        assert miner_entry["path"] == "logs/state/alpha_miner.json"
        assert miner_entry["owner"] == "executor"
        assert "optional" in miner_entry and miner_entry["optional"] is True


# ---------------------------------------------------------------------------
# Dashboard Panel Integration Tests
# ---------------------------------------------------------------------------


class TestDashboardPanelIntegration:
    """Tests for dashboard panel integration."""

    def test_load_alpha_miner_state_exists(self):
        """Verify load function is available in dashboard."""
        from dashboard.edge_discovery_panel import load_alpha_miner_state
        assert callable(load_alpha_miner_state)

    def test_load_alpha_miner_state_returns_dict(self, tmp_path):
        """Test loading state from file."""
        from dashboard.edge_discovery_panel import load_alpha_miner_state

        # Just verify function returns dict type (may be empty if no file)
        result = load_alpha_miner_state()
        assert isinstance(result, dict)

    def test_render_alpha_miner_exists(self):
        """Verify render function is available."""
        from dashboard.edge_discovery_panel import render_alpha_miner
        assert callable(render_alpha_miner)


# ---------------------------------------------------------------------------
# Config Integration Tests
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    """Tests for config file integration."""

    def test_strategy_config_has_alpha_miner_section(self):
        """Verify strategy_config.json has alpha_miner section."""
        config_path = Path("config/strategy_config.json")
        if not config_path.exists():
            pytest.skip("strategy_config.json not found")

        config = json.loads(config_path.read_text())
        assert "alpha_miner" in config

        miner_cfg = config["alpha_miner"]
        assert "enabled" in miner_cfg
        assert "min_liquidity_usd" in miner_cfg
        assert "score_threshold" in miner_cfg
        assert "top_k" in miner_cfg
        assert "weights" in miner_cfg

    def test_alpha_miner_default_disabled(self):
        """Verify alpha_miner defaults to disabled."""
        config_path = Path("config/strategy_config.json")
        if not config_path.exists():
            pytest.skip("strategy_config.json not found")

        config = json.loads(config_path.read_text())
        miner_cfg = config.get("alpha_miner", {})
        assert miner_cfg.get("enabled") is False


# ---------------------------------------------------------------------------
# Scoring Consistency Tests
# ---------------------------------------------------------------------------


class TestScoringConsistency:
    """Tests for scoring consistency across runs."""

    def test_ema_smoothing_persistence(self, tmp_state_dir):
        """Verify EMA scores persist across miner runs."""
        from execution.alpha_miner import (
            AlphaMinerState,
            save_alpha_miner_state,
            load_alpha_miner_state,
            apply_ema_smoothing,
        )

        state_file = tmp_state_dir / "alpha_miner.json"

        # Simulate first run
        initial_scores = {"A": 0.5, "B": 0.6}
        initial_ema = apply_ema_smoothing(initial_scores, {}, 0.2)
        
        state1 = AlphaMinerState(
            updated_ts=time.time(),
            cycle_count=1,
            ema_scores=initial_ema,
        )
        save_alpha_miner_state(state1, state_file)

        # Load and simulate second run
        loaded = load_alpha_miner_state(state_file)
        new_scores = {"A": 0.7, "B": 0.4}  # Changed scores
        
        updated_ema = apply_ema_smoothing(new_scores, loaded.ema_scores, 0.2)

        # EMA should be smoothed (not equal to new raw scores)
        assert updated_ema["A"] != 0.7
        assert updated_ema["B"] != 0.4
        # But should move toward new scores
        assert updated_ema["A"] > initial_ema["A"]  # A went up
        assert updated_ema["B"] < initial_ema["B"]  # B went down


# ---------------------------------------------------------------------------
# Feature Extraction Integration Tests
# ---------------------------------------------------------------------------


class TestFeatureExtractionIntegration:
    """Tests for feature extraction with real data patterns."""

    def test_uptrending_market_features(self, mock_klines):
        """Verify uptrending market produces positive momentum."""
        from execution.alpha_miner import (
            extract_symbol_features,
            AlphaMinerConfig,
        )

        cfg = AlphaMinerConfig(lookback_bars=50)
        feat = extract_symbol_features(
            symbol="TESTUSDT",
            config=cfg,
            klines=mock_klines,
            volume_24h=500000.0,
            price=100.0,
        )

        assert feat is not None
        assert feat.long_momo > 0  # Uptrend detected
        assert feat.trend_consistency > 0.5  # More up days

    def test_high_liquidity_features(self, mock_klines):
        """Verify high liquidity produces good liquidity score."""
        from execution.alpha_miner import (
            extract_symbol_features,
            AlphaMinerConfig,
        )

        cfg = AlphaMinerConfig(
            min_liquidity_usd=100000.0,
            lookback_bars=50,
        )
        feat = extract_symbol_features(
            symbol="TESTUSDT",
            config=cfg,
            klines=mock_klines,
            volume_24h=10_000_000.0,  # Very high volume
            price=100.0,
        )

        assert feat is not None
        assert feat.liquidity_score > 0.7
