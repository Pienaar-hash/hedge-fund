"""
Integration tests for Execution Alpha Engine — v7.9_P4

Tests:
- State file schema compliance with v7_manifest
- End-to-end fill-to-alpha flow
- Minotaur integration hooks
- Dashboard panel data loading
"""

import json
import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any

from execution.execution_alpha import (
    ExecutionAlphaConfig,
    load_execution_alpha_config,
    AlphaSample,
    SymbolAlphaStats,
    HeadAlphaStats,
    AlphaAggregator,
    save_execution_alpha_state,
    load_execution_alpha_state,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_state_dir(tmp_path):
    """Create temporary state directory."""
    state_dir = tmp_path / "logs" / "state"
    state_dir.mkdir(parents=True)
    events_dir = tmp_path / "logs" / "execution"
    events_dir.mkdir(parents=True)
    return tmp_path


@pytest.fixture
def mock_state_paths(temp_state_dir, monkeypatch):
    """Mock state file paths to use temp directory."""
    import execution.execution_alpha as alpha_module
    
    state_dir = temp_state_dir / "logs" / "state"
    events_dir = temp_state_dir / "logs" / "execution"
    
    monkeypatch.setattr(alpha_module, "_STATE_DIR", state_dir)
    monkeypatch.setattr(alpha_module, "_ALPHA_STATE_FILE", state_dir / "execution_alpha.json")
    monkeypatch.setattr(alpha_module, "_EVENTS_DIR", events_dir)
    monkeypatch.setattr(alpha_module, "_EVENTS_FILE", events_dir / "execution_alpha_events.jsonl")
    
    return temp_state_dir


@pytest.fixture
def enabled_config() -> ExecutionAlphaConfig:
    """Execution alpha enabled config."""
    return ExecutionAlphaConfig(
        enabled=True,
        model_price_source="router_model",
        use_expected_fill_price_field=True,
        lookback_trades=100,
        min_samples_for_penalty=10,
        symbol_drag_bps_soft=8.0,
        symbol_drag_bps_hard=20.0,
        symbol_min_multiplier=0.6,
        alerts_enabled=True,
        tail_slippage_bps=30.0,
    )


@pytest.fixture
def sample_aggregator(enabled_config) -> AlphaAggregator:
    """Create aggregator with sample data."""
    agg = AlphaAggregator(lookback_trades=enabled_config.lookback_trades)
    
    # Add some samples
    for i in range(20):
        agg.add_sample(AlphaSample(
            ts=1234567890.0 + i,
            symbol="BTCUSDT",
            side="BUY",
            qty=0.01,
            fill_price=50100.0 + i * 10,
            model_price=50000.0,
            alpha_usd=-1.0 - i * 0.1,
            alpha_bps=-20.0 - i,
            drag_bps=20.0 + i,
            regime="NORMAL" if i % 3 == 0 else "THIN",
            head_contributions={"TREND": 0.6, "ZSCORE": 0.4},
        ))
    
    for i in range(15):
        agg.add_sample(AlphaSample(
            ts=1234567900.0 + i,
            symbol="ETHUSDT",
            side="SELL",
            qty=0.5,
            fill_price=2990.0 - i * 2,
            model_price=3000.0,
            alpha_usd=-5.0 - i * 0.5,
            alpha_bps=-33.0 - i * 2,
            drag_bps=33.0 + i * 2,
            regime="NORMAL",
            head_contributions={"EMERGENT_ALPHA": 1.0},
        ))
    
    return agg


# ---------------------------------------------------------------------------
# State Schema Tests
# ---------------------------------------------------------------------------


class TestExecutionAlphaStateSchema:
    """Tests for execution_alpha.json schema compliance."""
    
    def test_state_has_required_fields(self, mock_state_paths, sample_aggregator, enabled_config):
        """Test state file has all required fields per v7_manifest."""
        save_execution_alpha_state(sample_aggregator, enabled_config, [])
        
        state = load_execution_alpha_state()
        
        # Required top-level fields
        assert "updated_ts" in state
        assert "symbols" in state
        assert "heads" in state
        assert "meta" in state
        
        # Check symbols structure
        assert "BTCUSDT" in state["symbols"]
        btc = state["symbols"]["BTCUSDT"]
        
        # Per v7_manifest schema
        assert "samples" in btc
        assert "cum_alpha_usd" in btc
        assert "avg_alpha_bps" in btc
        assert "p95_alpha_bps" in btc
        assert "p99_alpha_bps" in btc
        assert "avg_drag_bps" in btc
        assert "regime_breakdown" in btc
        assert "suggested_multiplier" in btc
        
        # Check heads structure
        assert "TREND" in state["heads"]
        trend = state["heads"]["TREND"]
        assert "samples" in trend
        assert "cum_alpha_usd" in trend
        assert "suggested_multiplier" in trend
        
        # Check meta structure
        meta = state["meta"]
        assert "enabled" in meta
        assert "total_cum_alpha_usd" in meta
        assert "avg_alpha_bps" in meta
        assert "penalties_active" in meta
    
    def test_state_survives_round_trip(self, mock_state_paths, sample_aggregator, enabled_config):
        """Test state can be saved and loaded correctly."""
        save_execution_alpha_state(sample_aggregator, enabled_config, [{"event": "TEST"}])
        
        loaded = load_execution_alpha_state()
        
        assert loaded["symbols"]["BTCUSDT"]["samples"] == 20
        assert loaded["symbols"]["ETHUSDT"]["samples"] == 15
        assert loaded["heads"]["TREND"]["samples"] == 20
        assert loaded["meta"]["enabled"] is True


# ---------------------------------------------------------------------------
# Minotaur Integration Tests
# ---------------------------------------------------------------------------


class TestMinotaurIntegration:
    """Tests for Minotaur integration hooks."""
    
    def test_process_fill_for_alpha_disabled(self):
        """process_fill_for_alpha returns None when disabled."""
        from execution.minotaur_integration import process_fill_for_alpha
        
        cfg = ExecutionAlphaConfig(enabled=False)
        fill = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "qty": 0.01,
            "price": 50100.0,
            "expected_fill_price": 50000.0,
        }
        
        result = process_fill_for_alpha(fill, cfg=cfg)
        
        assert result is None
    
    def test_process_fill_for_alpha_enabled(self, mock_state_paths, enabled_config):
        """process_fill_for_alpha creates sample when enabled."""
        # Reset global state
        import execution.minotaur_integration as integration
        integration._ALPHA_AGGREGATOR = None
        integration._ALPHA_CONFIG = None
        integration._ALPHA_ALERTS = []
        
        from execution.minotaur_integration import process_fill_for_alpha
        
        fill = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "qty": 0.01,
            "price": 50100.0,
            "expected_fill_price": 50000.0,
        }
        
        result = process_fill_for_alpha(
            fill,
            regime="THIN",
            head_contributions={"TREND": 1.0},
            cfg=enabled_config,
        )
        
        assert result is not None
        assert result.symbol == "BTCUSDT"
        assert result.alpha_usd < 0  # Overpaid
        assert result.regime == "THIN"
    
    def test_get_alpha_aggregator(self):
        """get_alpha_aggregator returns or creates aggregator."""
        # Reset global state
        import execution.minotaur_integration as integration
        integration._ALPHA_AGGREGATOR = None
        integration._ALPHA_CONFIG = None
        
        from execution.minotaur_integration import get_alpha_aggregator
        
        agg = get_alpha_aggregator()
        
        # Should return an aggregator (or None if module not available)
        # Either way, calling it shouldn't raise
        assert agg is None or hasattr(agg, "add_sample")


# ---------------------------------------------------------------------------
# Dashboard Panel Tests
# ---------------------------------------------------------------------------


class TestDashboardPanel:
    """Tests for dashboard panel data loading."""
    
    def test_panel_handles_missing_state(self, mock_state_paths):
        """Test panel handles missing state file gracefully."""
        import dashboard.execution_alpha_panel as panel
        
        state_dir = mock_state_paths / "logs" / "state"
        events_dir = mock_state_paths / "logs" / "execution"
        
        panel._ALPHA_STATE_FILE = state_dir / "execution_alpha.json"
        panel._EVENTS_FILE = events_dir / "execution_alpha_events.jsonl"
        
        data = panel.get_panel_data()
        
        assert data["symbol_count"] == 0
        assert data["head_count"] == 0
        assert data["alerts"] == []
        assert data["meta"] == {}
    
    def test_panel_loads_state(self, mock_state_paths, sample_aggregator, enabled_config):
        """Test panel loads state file correctly."""
        # Save state
        save_execution_alpha_state(sample_aggregator, enabled_config, [])
        
        # Patch panel paths
        import dashboard.execution_alpha_panel as panel
        
        state_dir = mock_state_paths / "logs" / "state"
        events_dir = mock_state_paths / "logs" / "execution"
        
        panel._ALPHA_STATE_FILE = state_dir / "execution_alpha.json"
        panel._EVENTS_FILE = events_dir / "execution_alpha_events.jsonl"
        
        data = panel.get_panel_data()
        
        assert data["symbol_count"] == 2  # BTCUSDT, ETHUSDT
        assert data["head_count"] == 3  # TREND, ZSCORE, EMERGENT_ALPHA
        assert "BTCUSDT" in data["state"]["symbols"]


# ---------------------------------------------------------------------------
# Config Integration Tests
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    """Tests for config file integration."""
    
    def test_loads_from_strategy_config(self):
        """Test loading from actual strategy_config.json structure."""
        config_path = Path("config/strategy_config.json")
        if config_path.exists():
            with open(config_path, "r") as f:
                strategy_cfg = json.load(f)
            
            cfg = load_execution_alpha_config(strategy_cfg)
            
            # Should have valid config (enabled or not)
            assert isinstance(cfg.enabled, bool)
            assert cfg.lookback_trades > 0
            assert cfg.model_price_source in ("router_model", "mid", "bid_ask_side")


# ---------------------------------------------------------------------------
# Manifest Alignment Tests
# ---------------------------------------------------------------------------


class TestManifestAlignment:
    """Tests for v7_manifest.json alignment."""
    
    def test_execution_alpha_in_manifest(self):
        """Test execution_alpha is declared in v7_manifest."""
        manifest_path = Path("v7_manifest.json")
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            
            state_files = manifest.get("state_files", {})
            
            assert "execution_alpha" in state_files
            
            alpha_entry = state_files["execution_alpha"]
            assert alpha_entry["path"] == "logs/state/execution_alpha.json"
            assert alpha_entry["owner"] == "executor"
            assert alpha_entry.get("optional") is True
            assert "fields" in alpha_entry
            assert "symbols" in alpha_entry["fields"]
            assert "heads" in alpha_entry["fields"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
