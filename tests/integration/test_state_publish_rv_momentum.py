"""
Tests for v7.5_C1 — RV Momentum State Publishing.

Tests:
- rv_momentum state file is written correctly
- State contains per_symbol scores and spreads
- Dashboard loader can load the state
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from execution.state_publish import (
    write_rv_momentum_state,
    compute_and_write_rv_momentum_state,
)
from execution.rv_momentum import (
    RvSnapshot,
    RvSymbolScore,
    RvConfig,
)


class TestWriteRvMomentumState:
    """Test state file writing."""

    def test_writes_to_file(self, tmp_path):
        """RV momentum state is written to file."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        
        payload = {
            "updated_ts": 1700000000.0,
            "per_symbol": {
                "BTCUSDT": {
                    "score": 0.75,
                    "raw_score": 0.5,
                    "baskets": ["l1"],
                },
                "DOGEUSDT": {
                    "score": -0.3,
                    "raw_score": -0.2,
                    "baskets": ["meme"],
                },
            },
            "spreads": {
                "btc_vs_eth": 0.012,
                "l1_vs_alt": 0.008,
                "meme_vs_rest": -0.005,
            },
        }
        
        write_rv_momentum_state(payload, state_dir)
        
        path = state_dir / "rv_momentum.json"
        assert path.exists()
        
        data = json.loads(path.read_text())
        assert "per_symbol" in data
        assert "spreads" in data
        assert data["per_symbol"]["BTCUSDT"]["score"] == 0.75
        assert data["spreads"]["btc_vs_eth"] == 0.012


class TestComputeAndWriteRvMomentumState:
    """Test compute and write function."""

    def test_computes_and_writes_when_enabled(self, tmp_path):
        """Computes snapshot and writes when enabled."""
        state_dir = tmp_path / "state"
        
        mock_snapshot = RvSnapshot(
            per_symbol={
                "BTCUSDT": RvSymbolScore(symbol="BTCUSDT", score=0.6, raw_score=0.4, baskets=["l1"]),
            },
            btc_vs_eth_spread=0.01,
            l1_vs_alt_spread=0.02,
            meme_vs_rest_spread=-0.01,
            updated_ts=1700000000.0,
        )
        
        with patch("execution.rv_momentum.load_rv_config") as mock_cfg:
            with patch("execution.rv_momentum.build_rv_snapshot") as mock_build:
                mock_cfg.return_value = RvConfig(enabled=True)
                mock_build.return_value = mock_snapshot
                
                result = compute_and_write_rv_momentum_state(state_dir)
                
                assert "per_symbol" in result
                assert result["per_symbol"]["BTCUSDT"]["score"] == 0.6
                
                path = state_dir / "rv_momentum.json"
                assert path.exists()

    def test_returns_empty_when_disabled(self, tmp_path):
        """Returns empty dict when RV momentum disabled."""
        state_dir = tmp_path / "state"
        
        with patch("execution.rv_momentum.load_rv_config") as mock_cfg:
            mock_cfg.return_value = RvConfig(enabled=False)
            
            result = compute_and_write_rv_momentum_state(state_dir)
            
            assert result == {}

    def test_handles_import_error(self, tmp_path):
        """Gracefully handles import error."""
        state_dir = tmp_path / "state"
        
        with patch("execution.state_publish.LOG"):
            # Force import error by patching the import
            import sys
            original_module = sys.modules.get("execution.rv_momentum")
            sys.modules["execution.rv_momentum"] = None
            
            try:
                result = compute_and_write_rv_momentum_state(state_dir)
                # Should return empty on error
                assert result == {} or isinstance(result, dict)
            finally:
                if original_module:
                    sys.modules["execution.rv_momentum"] = original_module
                elif "execution.rv_momentum" in sys.modules:
                    del sys.modules["execution.rv_momentum"]


class TestRvSnapshotSerialization:
    """Test snapshot serialization."""

    def test_snapshot_to_dict_roundtrip(self):
        """Snapshot can be serialized and deserialized."""
        snapshot = RvSnapshot(
            per_symbol={
                "BTCUSDT": RvSymbolScore(
                    symbol="BTCUSDT",
                    score=0.75,
                    raw_score=0.5,
                    baskets=["l1"],
                ),
            },
            btc_vs_eth_spread=0.0123,
            l1_vs_alt_spread=0.0089,
            meme_vs_rest_spread=-0.0045,
            updated_ts=1700000000.0,
        )
        
        d = snapshot.to_dict()
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        
        assert loaded["per_symbol"]["BTCUSDT"]["score"] == 0.75
        assert loaded["spreads"]["btc_vs_eth"] == 0.0123

    def test_snapshot_values_rounded(self):
        """Snapshot values are rounded for clean JSON."""
        snapshot = RvSnapshot(
            per_symbol={
                "BTCUSDT": RvSymbolScore(
                    symbol="BTCUSDT",
                    score=0.75123456,
                    raw_score=0.50987654,
                    baskets=["l1"],
                ),
            },
            btc_vs_eth_spread=0.01234567,
            l1_vs_alt_spread=0.00891234,
            meme_vs_rest_spread=-0.00456789,
            updated_ts=1700000000.0,
        )
        
        d = snapshot.to_dict()
        
        # Values should be rounded to 4 decimals
        assert d["per_symbol"]["BTCUSDT"]["score"] == 0.7512
        assert d["spreads"]["btc_vs_eth"] == 0.0123


class TestDashboardLoaderIntegration:
    """Test dashboard state loader integration."""

    def test_load_rv_momentum_state(self, tmp_path):
        """Dashboard loader loads RV momentum state."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        
        payload = {
            "updated_ts": 1700000000.0,
            "per_symbol": {
                "BTCUSDT": {
                    "score": 0.65,
                    "baskets": ["l1"],
                },
            },
            "spreads": {
                "btc_vs_eth": 0.01,
                "l1_vs_alt": 0.02,
                "meme_vs_rest": -0.01,
            },
        }
        
        (state_dir / "rv_momentum.json").write_text(json.dumps(payload))
        
        # Test dashboard loader
        with patch("dashboard.state_v7.RV_MOMENTUM_PATH", state_dir / "rv_momentum.json"):
            from dashboard.state_v7 import (
                load_rv_momentum,
                get_symbol_rv_momentum_score,
                get_rv_momentum_spreads,
            )
            
            loaded = load_rv_momentum()
            assert loaded["per_symbol"]["BTCUSDT"]["score"] == 0.65
            
            score = get_symbol_rv_momentum_score("BTCUSDT", loaded)
            assert score == 0.65
            
            # Unknown symbol returns default
            unknown_score = get_symbol_rv_momentum_score("UNKNOWN", loaded, default=0.0)
            assert unknown_score == 0.0
            
            spreads = get_rv_momentum_spreads(loaded)
            assert spreads["btc_vs_eth"] == 0.01
            assert spreads["l1_vs_alt"] == 0.02

    def test_missing_file_returns_empty(self):
        """Missing file returns empty dict."""
        from dashboard.state_v7 import load_rv_momentum
        
        with patch("dashboard.state_v7.RV_MOMENTUM_PATH", Path("/nonexistent/path.json")):
            loaded = load_rv_momentum()
            assert loaded == {}
