"""
Tests for state publishing of vol regime data (v7.4 B2).
"""
from __future__ import annotations

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from execution.state_publish import publish_vol_regime_snapshot


class TestPublishVolRegimeSnapshot:
    """Tests for vol regime state publishing."""

    def test_publishes_to_file(self, tmp_path):
        """Vol regime snapshot is written to file."""
        state_dir = tmp_path / "state"
        
        symbols_data = [
            {
                "symbol": "BTCUSDT",
                "vol_regime": "normal",
                "vol": {"short": 0.015, "long": 0.012, "ratio": 1.25},
            },
            {
                "symbol": "ETHUSDT",
                "vol_regime": "high",
                "vol": {"short": 0.025, "long": 0.015, "ratio": 1.67},
            },
        ]
        
        with patch("execution.state_publish.STATE_DIR", state_dir):
            publish_vol_regime_snapshot(symbols_data)
        
        path = state_dir / "vol_regimes.json"
        assert path.exists()
        
        data = json.loads(path.read_text())
        assert "symbols" in data
        assert "vol_regime_summary" in data
        assert "updated_ts" in data

    def test_summary_counts(self, tmp_path):
        """Summary counts regimes correctly."""
        state_dir = tmp_path / "state"
        
        symbols_data = [
            {"symbol": "SYM1", "vol_regime": "low"},
            {"symbol": "SYM2", "vol_regime": "normal"},
            {"symbol": "SYM3", "vol_regime": "normal"},
            {"symbol": "SYM4", "vol_regime": "high"},
            {"symbol": "SYM5", "vol_regime": "crisis"},
        ]
        
        with patch("execution.state_publish.STATE_DIR", state_dir):
            publish_vol_regime_snapshot(symbols_data)
        
        path = state_dir / "vol_regimes.json"
        data = json.loads(path.read_text())
        
        summary = data["vol_regime_summary"]
        assert summary["low"] == 1
        assert summary["normal"] == 2
        assert summary["high"] == 1
        assert summary["crisis"] == 1

    def test_handles_empty_list(self, tmp_path):
        """Empty list produces empty snapshot."""
        state_dir = tmp_path / "state"
        
        with patch("execution.state_publish.STATE_DIR", state_dir):
            publish_vol_regime_snapshot([])
        
        path = state_dir / "vol_regimes.json"
        data = json.loads(path.read_text())
        
        assert data["symbols"] == []
        assert data["vol_regime_summary"] == {"low": 0, "normal": 0, "high": 0, "crisis": 0}

    def test_preserves_vol_data(self, tmp_path):
        """Vol data is preserved in snapshot."""
        state_dir = tmp_path / "state"
        
        symbols_data = [
            {
                "symbol": "BTCUSDT",
                "vol_regime": "high",
                "vol": {
                    "short": 0.0234,
                    "long": 0.0156,
                    "ratio": 1.5,
                },
            },
        ]
        
        with patch("execution.state_publish.STATE_DIR", state_dir):
            publish_vol_regime_snapshot(symbols_data)
        
        path = state_dir / "vol_regimes.json"
        data = json.loads(path.read_text())
        
        btc = data["symbols"][0]
        assert btc["symbol"] == "BTCUSDT"
        assert btc["vol_regime"] == "high"
        assert btc["vol"]["short"] == 0.0234
        assert btc["vol"]["long"] == 0.0156
        assert btc["vol"]["ratio"] == 1.5


class TestVolRegimeStateStructure:
    """Tests for expected state structure."""

    def test_expected_fields(self):
        """Validate expected field structure."""
        expected_symbol_entry = {
            "symbol": "BTCUSDT",
            "vol_regime": "normal",
            "vol": {
                "short": 0.015,
                "long": 0.012,
                "ratio": 1.25,
            },
        }
        
        # Verify required fields
        assert "symbol" in expected_symbol_entry
        assert "vol_regime" in expected_symbol_entry
        assert "vol" in expected_symbol_entry
        assert "short" in expected_symbol_entry["vol"]
        assert "long" in expected_symbol_entry["vol"]
        assert "ratio" in expected_symbol_entry["vol"]

    def test_expected_summary_fields(self):
        """Validate expected summary structure."""
        expected_summary = {
            "low": 0,
            "normal": 0,
            "high": 0,
            "crisis": 0,
        }
        
        assert set(expected_summary.keys()) == {"low", "normal", "high", "crisis"}


class TestDashboardStateLoading:
    """Tests for dashboard state loading."""

    def test_load_vol_regimes(self, tmp_path):
        """Dashboard can load vol regimes state."""
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True)
        
        data = {
            "symbols": [
                {"symbol": "BTCUSDT", "vol_regime": "high", "vol": {"short": 0.02, "long": 0.01, "ratio": 2.0}},
            ],
            "vol_regime_summary": {"low": 0, "normal": 0, "high": 1, "crisis": 0},
            "updated_ts": 1234567890.0,
        }
        
        path = state_dir / "vol_regimes.json"
        path.write_text(json.dumps(data))
        
        # Import here to patch correctly
        with patch("dashboard.state_v7.VOL_REGIMES_PATH", path):
            from dashboard.state_v7 import load_vol_regimes
            loaded = load_vol_regimes()
        
        assert loaded["vol_regime_summary"]["high"] == 1
        assert loaded["symbols"][0]["symbol"] == "BTCUSDT"

    def test_load_missing_file_returns_empty(self, tmp_path):
        """Missing file returns empty dict."""
        missing_path = tmp_path / "nonexistent" / "vol_regimes.json"
        
        with patch("dashboard.state_v7.VOL_REGIMES_PATH", missing_path):
            from dashboard.state_v7 import load_vol_regimes
            loaded = load_vol_regimes()
        
        assert loaded == {}
