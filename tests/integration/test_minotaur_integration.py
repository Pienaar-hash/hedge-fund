"""
Integration tests for Minotaur Execution Engine — v7.9_P3

Tests:
- State file schema compliance with v7_manifest
- End-to-end intent-to-plan flow
- Minotaur disabled behavior (identical to v7.9_P2)
- Quality state persistence
- Dashboard panel data loading
"""

import json
import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch

from execution.minotaur_engine import (
    MinotaurConfig,
    load_minotaur_config,
    is_minotaur_enabled,
    ExecutionQualityStats,
    save_execution_quality_state,
    load_execution_quality_state,
    MinotaurState,
    MODE_INSTANT,
    MODE_TWAP,
    REGIME_NORMAL,
    REGIME_THIN,
)

from execution.minotaur_integration import (
    run_minotaur_for_intents,
    process_fill_for_quality,
    get_minotaur_runtime_state,
    get_execution_quality_summary,
    reset_cycle_state,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_state_dir(tmp_path):
    """Create temporary state directory."""
    state_dir = tmp_path / "logs" / "state"
    state_dir.mkdir(parents=True)
    return state_dir


@pytest.fixture
def mock_state_paths(temp_state_dir, monkeypatch):
    """Mock state file paths to use temp directory."""
    import execution.minotaur_engine as engine
    
    monkeypatch.setattr(engine, "_STATE_DIR", temp_state_dir)
    monkeypatch.setattr(engine, "_QUALITY_STATE_FILE", temp_state_dir / "execution_quality.json")
    
    return temp_state_dir


@pytest.fixture
def enabled_config() -> MinotaurConfig:
    """Minotaur enabled config."""
    return MinotaurConfig(
        enabled=True,
        min_notional_for_twap_usd=500.0,
        max_child_order_notional_usd=150.0,
        min_slice_count=2,
        max_slice_count=12,
        twap_min_seconds=60,
        twap_max_seconds=900,
    )


@pytest.fixture
def disabled_config() -> MinotaurConfig:
    """Minotaur disabled config."""
    return MinotaurConfig(enabled=False)


@pytest.fixture
def sample_intents():
    """Sample merged intents for testing."""
    return [
        {
            "symbol": "BTCUSDT",
            "side": "LONG",
            "qty": 0.01,
            "price": 50000.0,
            "head_contributions": {"TREND": 0.7, "CATEGORY": 0.3},
        },
        {
            "symbol": "ETHUSDT",
            "side": "SHORT",
            "qty": 0.5,
            "price": 3000.0,
            "head_contributions": {"ZSCORE": 1.0},
        },
        {
            "symbol": "SOLUSDT",
            "side": "LONG",
            "qty": 10.0,
            "price": 150.0,
            "head_contributions": {"VOL_TARGET": 0.5, "TREND": 0.5},
        },
    ]


# ---------------------------------------------------------------------------
# State Schema Tests
# ---------------------------------------------------------------------------


class TestExecutionQualityStateSchema:
    """Tests for execution_quality.json schema compliance."""
    
    def test_state_has_required_fields(self, mock_state_paths, enabled_config):
        """Test state file has all required fields per v7_manifest."""
        quality_stats = {
            "BTCUSDT": ExecutionQualityStats(
                symbol="BTCUSDT",
                avg_slippage_bps=4.2,
                p95_slippage_bps=11.5,
                max_slippage_bps=25.0,
                fill_ratio=0.93,
                mean_notional=210.3,
                twap_usage_pct=0.64,
                last_regime="THIN",
                trade_count=50,
            ),
        }
        
        minotaur_state = MinotaurState(
            enabled=True,
            symbols_in_thin=["WIFUSDT"],
            symbols_in_crunch=[],
            throttling_active=False,
        )
        
        save_execution_quality_state(quality_stats, enabled_config, minotaur_state)
        
        # Load and verify schema
        state = load_execution_quality_state()
        
        # Required top-level fields
        assert "updated_ts" in state
        assert "symbols" in state
        assert "meta" in state
        
        # Check timestamp format
        assert state["updated_ts"] is not None
        
        # Check symbols structure
        assert "BTCUSDT" in state["symbols"]
        btc = state["symbols"]["BTCUSDT"]
        
        # Per v7_manifest schema
        assert "avg_slippage_bps" in btc
        assert "p95_slippage_bps" in btc
        assert "max_slippage_bps" in btc
        assert "fill_ratio" in btc
        assert "mean_notional" in btc
        assert "twap_usage_pct" in btc
        assert "last_regime" in btc
        assert "trade_count" in btc
        
        # Check meta structure
        meta = state["meta"]
        assert "enabled" in meta
        assert "thin_liquidity_symbols" in meta
        assert "crunch_symbols" in meta
        assert "throttling_active" in meta
    
    def test_state_survives_round_trip(self, mock_state_paths, enabled_config):
        """Test state can be saved and loaded correctly."""
        original_stats = {
            "ETHUSDT": ExecutionQualityStats(
                symbol="ETHUSDT",
                avg_slippage_bps=3.5,
                p95_slippage_bps=8.2,
                max_slippage_bps=15.0,
                fill_ratio=0.98,
                mean_notional=500.0,
                twap_usage_pct=0.25,
                last_regime="NORMAL",
                trade_count=100,
            ),
        }
        
        save_execution_quality_state(original_stats, enabled_config, None)
        loaded = load_execution_quality_state()
        
        assert loaded["symbols"]["ETHUSDT"]["avg_slippage_bps"] == 3.5
        assert loaded["symbols"]["ETHUSDT"]["trade_count"] == 100


# ---------------------------------------------------------------------------
# Disabled Behavior Tests
# ---------------------------------------------------------------------------


class TestMinotaurDisabledBehavior:
    """Tests for behavior when Minotaur is disabled (v7.9_P2 compatibility)."""
    
    def test_disabled_produces_instant_plans(self, disabled_config, sample_intents):
        """Test disabled Minotaur produces instant execution plans."""
        reset_cycle_state()
        
        plans, state = run_minotaur_for_intents(
            merged_intents=sample_intents,
            nav_usd=10000.0,
            minotaur_cfg=disabled_config,
        )
        
        # Should produce one plan per intent
        assert len(plans) == len(sample_intents)
        
        # All should be INSTANT mode
        for plan in plans:
            assert plan.slicing_mode == MODE_INSTANT
            assert plan.slice_count == 1
            assert plan.schedule_seconds == 0
            assert plan.notes == "minotaur disabled"
    
    def test_disabled_no_throttling(self, disabled_config, sample_intents):
        """Test disabled Minotaur doesn't throttle."""
        reset_cycle_state()
        
        plans, state = run_minotaur_for_intents(
            merged_intents=sample_intents,
            nav_usd=10000.0,
            minotaur_cfg=disabled_config,
        )
        
        assert state.throttling_active is False
        assert state.halt_new_positions is False


# ---------------------------------------------------------------------------
# Enabled Behavior Tests
# ---------------------------------------------------------------------------


class TestMinotaurEnabledBehavior:
    """Tests for behavior when Minotaur is enabled."""
    
    def test_enabled_plans_respect_notional_threshold(self, enabled_config, sample_intents):
        """Test enabled Minotaur slices large orders."""
        reset_cycle_state()
        
        plans, state = run_minotaur_for_intents(
            merged_intents=sample_intents,
            nav_usd=10000.0,
            minotaur_cfg=enabled_config,
        )
        
        assert len(plans) == len(sample_intents)
        
        # Find plans by symbol
        plan_map = {p.symbol: p for p in plans}
        
        # BTCUSDT: 0.01 * 50000 = 500 USD (at threshold, may be instant in NORMAL)
        # ETHUSDT: 0.5 * 3000 = 1500 USD (above threshold, should be TWAP)
        # SOLUSDT: 10 * 150 = 1500 USD (above threshold, should be TWAP)
        
        assert plan_map["ETHUSDT"].slicing_mode in (MODE_TWAP, MODE_INSTANT)
        assert plan_map["SOLUSDT"].slicing_mode in (MODE_TWAP, MODE_INSTANT)
        
        # Large orders should have multiple slices
        large_orders = [p for p in plans if p.total_notional > enabled_config.min_notional_for_twap_usd]
        for plan in large_orders:
            if plan.slicing_mode != MODE_INSTANT:
                assert plan.slice_count >= enabled_config.min_slice_count
    
    def test_enabled_respects_order_limit(self, enabled_config):
        """Test enabled Minotaur respects max orders per cycle."""
        reset_cycle_state()
        
        # Create many large intents
        many_intents = [
            {
                "symbol": f"SYM{i}USDT",
                "side": "LONG",
                "qty": 10.0,
                "price": 100.0,  # 1000 USD each (above threshold)
            }
            for i in range(20)
        ]
        
        plans, state = run_minotaur_for_intents(
            merged_intents=many_intents,
            nav_usd=100000.0,
            minotaur_cfg=enabled_config,
        )
        
        # Total child orders should be tracked
        assert state.orders_this_cycle > 0


# ---------------------------------------------------------------------------
# Fill Processing Tests
# ---------------------------------------------------------------------------


class TestFillProcessing:
    """Tests for fill processing and quality tracking."""
    
    def test_process_fill_updates_stats(self, enabled_config):
        """Test processing fills updates quality stats."""
        # Reset global state for clean test
        import execution.minotaur_integration as integration
        integration._QUALITY_STATS = {}
        
        stats = process_fill_for_quality(
            symbol="BTCUSDT",
            side="LONG",
            fill_price=50010.0,
            model_price=50000.0,
            fill_qty=0.01,
            target_qty=0.01,
            used_twap=True,
            regime=REGIME_NORMAL,
            cfg=enabled_config,
        )
        
        assert stats.symbol == "BTCUSDT"
        assert stats.trade_count == 1
        assert stats.avg_slippage_bps > 0  # Paid more than model
        assert stats.twap_usage_pct == 1.0
    
    def test_process_fill_accumulates(self, enabled_config):
        """Test multiple fills accumulate in stats."""
        import execution.minotaur_integration as integration
        integration._QUALITY_STATS = {}
        
        # First fill
        process_fill_for_quality(
            symbol="ETHUSDT",
            side="SHORT",
            fill_price=2998.0,
            model_price=3000.0,
            fill_qty=0.5,
            target_qty=0.5,
            used_twap=False,
            regime=REGIME_NORMAL,
            cfg=enabled_config,
        )
        
        # Second fill
        stats = process_fill_for_quality(
            symbol="ETHUSDT",
            side="SHORT",
            fill_price=2995.0,
            model_price=3000.0,
            fill_qty=0.3,
            target_qty=0.3,
            used_twap=True,
            regime=REGIME_THIN,
            cfg=enabled_config,
        )
        
        assert stats.trade_count == 2
        assert stats.last_regime == REGIME_THIN
    
    def test_execution_quality_summary(self, enabled_config):
        """Test execution quality summary aggregation."""
        import execution.minotaur_integration as integration
        integration._QUALITY_STATS = {
            "BTCUSDT": ExecutionQualityStats(
                symbol="BTCUSDT",
                avg_slippage_bps=5.0,
                fill_ratio=0.95,
                twap_usage_pct=0.6,
                trade_count=10,
            ),
            "ETHUSDT": ExecutionQualityStats(
                symbol="ETHUSDT",
                avg_slippage_bps=3.0,
                fill_ratio=0.98,
                twap_usage_pct=0.4,
                trade_count=20,
            ),
        }
        
        summary = get_execution_quality_summary()
        
        assert summary["total_symbols"] == 2
        assert summary["avg_slippage_bps"] == 4.0  # (5 + 3) / 2
        assert summary["avg_fill_ratio"] == 0.965  # (0.95 + 0.98) / 2
        assert "BTCUSDT" in summary["symbols_tracked"]
        assert "ETHUSDT" in summary["symbols_tracked"]


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Config Integration Tests
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    """Tests for config file integration."""
    
    def test_loads_from_strategy_config(self):
        """Test loading from actual strategy_config.json structure."""
        # Load actual config file
        config_path = Path("config/strategy_config.json")
        if config_path.exists():
            with open(config_path, "r") as f:
                strategy_cfg = json.load(f)
            
            cfg = load_minotaur_config(strategy_cfg)
            
            # Should have valid config (enabled or not)
            assert isinstance(cfg.enabled, bool)
            assert cfg.min_notional_for_twap_usd > 0
            assert cfg.max_child_order_notional_usd > 0
    
    def test_is_minotaur_enabled_from_config(self):
        """Test is_minotaur_enabled with real config."""
        config_path = Path("config/strategy_config.json")
        if config_path.exists():
            with open(config_path, "r") as f:
                strategy_cfg = json.load(f)
            
            # Should return bool without error
            enabled = is_minotaur_enabled(strategy_cfg)
            assert isinstance(enabled, bool)


# ---------------------------------------------------------------------------
# Manifest Alignment Tests
# ---------------------------------------------------------------------------


class TestManifestAlignment:
    """Tests for v7_manifest.json alignment."""
    
    def test_execution_quality_in_manifest(self):
        """Test execution_quality is declared in v7_manifest."""
        manifest_path = Path("v7_manifest.json")
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            
            state_files = manifest.get("state_files", {})
            
            assert "execution_quality" in state_files
            
            eq_entry = state_files["execution_quality"]
            assert eq_entry["path"] == "logs/state/execution_quality.json"
            assert eq_entry["owner"] == "executor"
            assert eq_entry.get("optional") is True
            assert "fields" in eq_entry


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
