"""
Tests for v7.5_B2 — Router Quality State Publishing.

Tests:
- router_quality block exists with correct keys
- Dashboard loader consumes the block without error
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from execution.state_publish import (
    write_router_quality_state,
    compute_and_write_router_quality_state,
)
from execution.router_metrics import (
    RouterQualitySnapshot,
    build_router_quality_state_snapshot,
    RouterQualityConfig,
)


class TestWriteRouterQualityState:
    """Test state writing."""

    def test_writes_to_file(self, tmp_path):
        """Router quality state is written to file."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        
        payload = {
            "updated_ts": 1700000000.0,
            "enabled": True,
            "summary": {
                "symbol_count": 2,
                "avg_score": 0.75,
                "min_score": 0.60,
                "max_score": 0.90,
                "low_quality_count": 0,
                "high_quality_count": 1,
            },
            "symbols": {
                "BTCUSDT": {
                    "score": 0.90,
                    "bucket": "A_HIGH",
                    "ewma_expected_bps": 2.0,
                    "ewma_realized_bps": 2.5,
                    "slippage_drift_bps": 0.5,
                    "twap_skip_ratio": 0.02,
                    "trade_count": 100,
                },
                "ETHUSDT": {
                    "score": 0.60,
                    "bucket": "B_MEDIUM",
                    "ewma_expected_bps": 3.5,
                    "ewma_realized_bps": 6.0,
                    "slippage_drift_bps": 2.5,
                    "twap_skip_ratio": 0.10,
                    "trade_count": 50,
                },
            },
        }
        
        write_router_quality_state(payload, state_dir)
        
        path = state_dir / "router_quality.json"
        assert path.exists()
        
        data = json.loads(path.read_text())
        assert data["enabled"] is True
        assert "summary" in data
        assert "symbols" in data
        assert data["summary"]["symbol_count"] == 2
        assert "BTCUSDT" in data["symbols"]
        assert data["symbols"]["BTCUSDT"]["score"] == 0.90


class TestBuildRouterQualityStateSnapshot:
    """Test snapshot building."""

    def test_builds_snapshot_with_summary(self):
        """Snapshot includes summary and per-symbol data."""
        # Mock slippage and liquidity data
        mock_slippage_stats = MagicMock()
        mock_slippage_stats.ewma_expected_bps = 3.0
        mock_slippage_stats.ewma_realized_bps = 4.0
        mock_slippage_stats.trade_count = 25
        
        mock_liquidity_model = MagicMock()
        mock_liquidity_model.get_bucket_name.return_value = "A_HIGH"
        
        with patch("execution.slippage_model.get_all_slippage_stats") as mock_get_stats:
            with patch("execution.liquidity_model.get_liquidity_model") as mock_get_liq:
                with patch("execution.router_metrics.get_recent_twap_events") as mock_twap:
                    mock_get_stats.return_value = {"BTCUSDT": mock_slippage_stats}
                    mock_get_liq.return_value = mock_liquidity_model
                    mock_twap.return_value = []  # No TWAP events
                    
                    cfg = RouterQualityConfig()
                    snapshot = build_router_quality_state_snapshot(cfg)
                    
                    assert "updated_ts" in snapshot
                    assert "enabled" in snapshot
                    assert "summary" in snapshot
                    assert "symbols" in snapshot
                    
                    # Check summary
                    summary = snapshot["summary"]
                    assert "symbol_count" in summary
                    assert "avg_score" in summary
                    assert "low_quality_count" in summary
                    
                    # Check symbols
                    assert "BTCUSDT" in snapshot["symbols"]
                    btc_data = snapshot["symbols"]["BTCUSDT"]
                    assert "score" in btc_data
                    assert "bucket" in btc_data
                    assert "ewma_expected_bps" in btc_data

    def test_empty_snapshot_when_no_data(self):
        """Returns valid snapshot even with no slippage data."""
        with patch("execution.slippage_model.get_all_slippage_stats") as mock_get_stats:
            mock_get_stats.return_value = {}
            
            cfg = RouterQualityConfig()
            snapshot = build_router_quality_state_snapshot(cfg)
            
            assert snapshot["symbols"] == {}
            assert snapshot["summary"]["symbol_count"] == 0
            # avg_score defaults to base_score when no data
            assert snapshot["summary"]["avg_score"] == cfg.base_score


class TestComputeAndWriteRouterQualityState:
    """Test combined compute and write."""

    def test_computes_and_writes(self, tmp_path):
        """Computes snapshot and writes to file."""
        state_dir = tmp_path / "state"
        
        with patch("execution.router_metrics.build_router_quality_state_snapshot") as mock_build:
            mock_build.return_value = {
                "updated_ts": 1700000000.0,
                "enabled": True,
                "summary": {"symbol_count": 0},
                "symbols": {},
            }
            
            result = compute_and_write_router_quality_state(state_dir)
            
            assert result["enabled"] is True
            mock_build.assert_called_once()
            
            path = state_dir / "router_quality.json"
            assert path.exists()


class TestDashboardLoaderIntegration:
    """Test dashboard state loader integration."""

    def test_load_router_quality(self, tmp_path):
        """Dashboard loader loads router quality state."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        
        payload = {
            "updated_ts": 1700000000.0,
            "enabled": True,
            "summary": {
                "symbol_count": 1,
                "avg_score": 0.82,
            },
            "symbols": {
                "BTCUSDT": {
                    "score": 0.82,
                    "bucket": "A_HIGH",
                },
            },
        }
        
        (state_dir / "router_quality.json").write_text(json.dumps(payload))
        
        # Test dashboard loader
        with patch("dashboard.state_v7.ROUTER_QUALITY_PATH", state_dir / "router_quality.json"):
            from dashboard.state_v7 import load_router_quality, get_symbol_router_quality_score
            
            loaded = load_router_quality()
            assert loaded["enabled"] is True
            assert loaded["summary"]["symbol_count"] == 1
            
            score = get_symbol_router_quality_score("BTCUSDT", loaded)
            assert score == 0.82
            
            # Unknown symbol returns default
            unknown_score = get_symbol_router_quality_score("UNKNOWN", loaded, default=0.75)
            assert unknown_score == 0.75

    def test_missing_file_returns_default(self):
        """Missing file returns empty dict."""
        from dashboard.state_v7 import load_router_quality
        
        with patch("dashboard.state_v7.ROUTER_QUALITY_PATH", Path("/nonexistent/path.json")):
            loaded = load_router_quality()
            assert loaded == {}


class TestRouterQualitySnapshotSerialization:
    """Test snapshot serialization."""

    def test_snapshot_to_dict_roundtrip(self):
        """Snapshot can be serialized and deserialized."""
        snapshot = RouterQualitySnapshot(
            symbol="BTCUSDT",
            score=0.85,
            bucket="A_HIGH",
            ewma_expected_bps=2.5,
            ewma_realized_bps=3.0,
            slippage_drift_bps=0.5,
            twap_skip_ratio=0.05,
            trade_count=100,
        )
        
        d = snapshot.to_dict()
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        
        assert loaded["score"] == 0.85
        assert loaded["bucket"] == "A_HIGH"
        assert loaded["ewma_expected_bps"] == 2.5
        assert loaded["slippage_drift_bps"] == 0.5

    def test_snapshot_values_rounded(self):
        """Snapshot values are rounded for clean JSON."""
        snapshot = RouterQualitySnapshot(
            symbol="BTCUSDT",
            score=0.85123456,
            bucket="A_HIGH",
            ewma_expected_bps=2.567891,
            ewma_realized_bps=3.123456,
            slippage_drift_bps=0.555565,
            twap_skip_ratio=0.054321,
            trade_count=100,
        )
        
        d = snapshot.to_dict()
        
        # Values should be rounded
        assert d["score"] == 0.8512
        assert d["ewma_expected_bps"] == 2.5679
        assert d["twap_skip_ratio"] == 0.0543
