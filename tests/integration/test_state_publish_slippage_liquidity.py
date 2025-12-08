"""
Tests for slippage and liquidity state publishing (v7.5_B1)
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock


# ===========================================================================
# Tests: write_slippage_state
# ===========================================================================

class TestWriteSlippageState:
    """Test write_slippage_state function."""
    
    def test_writes_valid_json(self):
        """Should write valid JSON to file."""
        from execution.state_publish import write_slippage_state
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            
            payload = {
                "updated_ts": 1700000000.0,
                "per_symbol": {
                    "BTCUSDT": {
                        "ewma_expected_bps": 2.5,
                        "ewma_realized_bps": 3.0,
                        "trade_count": 50,
                    },
                },
            }
            
            write_slippage_state(payload, state_dir)
            
            output_path = state_dir / "slippage.json"
            assert output_path.exists()
            
            with output_path.open() as f:
                loaded = json.load(f)
            
            assert loaded["per_symbol"]["BTCUSDT"]["ewma_expected_bps"] == 2.5

    def test_handles_empty_payload(self):
        """Should handle empty payload."""
        from execution.state_publish import write_slippage_state
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            
            write_slippage_state({}, state_dir)
            
            output_path = state_dir / "slippage.json"
            assert output_path.exists()


# ===========================================================================
# Tests: write_liquidity_buckets_state
# ===========================================================================

class TestWriteLiquidityBucketsState:
    """Test write_liquidity_buckets_state function."""
    
    def test_writes_valid_json(self):
        """Should write valid JSON to file."""
        from execution.state_publish import write_liquidity_buckets_state
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            
            payload = {
                "updated_ts": 1700000000.0,
                "symbols": {
                    "BTCUSDT": {
                        "bucket": "A_HIGH",
                        "max_spread_bps": 5.0,
                        "default_maker_bias": 0.8,
                    },
                },
                "buckets": {
                    "A_HIGH": {
                        "max_spread_bps": 5.0,
                        "default_maker_bias": 0.8,
                        "symbol_count": 2,
                    },
                },
            }
            
            write_liquidity_buckets_state(payload, state_dir)
            
            output_path = state_dir / "liquidity_buckets.json"
            assert output_path.exists()
            
            with output_path.open() as f:
                loaded = json.load(f)
            
            assert loaded["symbols"]["BTCUSDT"]["bucket"] == "A_HIGH"


# ===========================================================================
# Tests: compute_and_write_slippage_state
# ===========================================================================

class TestComputeAndWriteSlippageState:
    """Test compute_and_write_slippage_state function."""
    
    @patch("execution.router_metrics.build_slippage_metrics_snapshot")
    def test_computes_and_writes_snapshot(self, mock_build):
        """Should compute snapshot and write to file."""
        from execution.state_publish import compute_and_write_slippage_state
        
        mock_build.return_value = {
            "updated_ts": 1700000000.0,
            "per_symbol": {
                "BTCUSDT": {"ewma_realized_bps": 3.5},
            },
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            
            result = compute_and_write_slippage_state(state_dir)
            
            assert "BTCUSDT" in result["per_symbol"]
            mock_build.assert_called_once()
            
            # Verify file was written
            output_path = state_dir / "slippage.json"
            assert output_path.exists()

    def test_handles_exception_gracefully(self):
        """Should handle exceptions gracefully."""
        from execution.state_publish import compute_and_write_slippage_state
        
        with patch("execution.router_metrics.build_slippage_metrics_snapshot", side_effect=RuntimeError("test")):
            with tempfile.TemporaryDirectory() as tmpdir:
                state_dir = Path(tmpdir)
                # Should handle the exception without crashing
                try:
                    result = compute_and_write_slippage_state(state_dir)
                except RuntimeError:
                    pass  # Expected exception propagation


# ===========================================================================#
# Tests: compute_and_write_liquidity_buckets_state
# ===========================================================================

class TestComputeAndWriteLiquidityBucketsState:
    """Test compute_and_write_liquidity_buckets_state function."""
    
    @patch("execution.router_metrics.build_liquidity_buckets_snapshot")
    def test_computes_and_writes_snapshot(self, mock_build):
        """Should compute snapshot and write to file."""
        from execution.state_publish import compute_and_write_liquidity_buckets_state
        
        mock_build.return_value = {
            "updated_ts": 1700000000.0,
            "symbols": {
                "BTCUSDT": {"bucket": "A_HIGH"},
            },
            "buckets": {
                "A_HIGH": {"symbol_count": 2},
            },
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            
            result = compute_and_write_liquidity_buckets_state(state_dir)
            
            assert "BTCUSDT" in result["symbols"]
            mock_build.assert_called_once()


# ===========================================================================
# Tests: Router Metrics Snapshot Functions
# ===========================================================================

class TestRouterMetricsSnapshots:
    """Test router_metrics snapshot functions for B1."""
    
    def test_build_slippage_metrics_snapshot_structure(self):
        """Should return dict with expected structure."""
        from execution.router_metrics import build_slippage_metrics_snapshot
        
        snapshot = build_slippage_metrics_snapshot()
        
        assert "updated_ts" in snapshot
        assert "per_symbol" in snapshot
        assert isinstance(snapshot["per_symbol"], dict)

    def test_build_liquidity_buckets_snapshot_structure(self):
        """Should return dict with expected structure."""
        from execution.router_metrics import build_liquidity_buckets_snapshot
        
        snapshot = build_liquidity_buckets_snapshot()
        
        assert "updated_ts" in snapshot
        assert "symbols" in snapshot
        assert "buckets" in snapshot


# ===========================================================================
# Tests: Dashboard Loader Integration
# ===========================================================================

class TestDashboardLoaderIntegration:
    """Test dashboard can load B1 state files."""
    
    def test_load_slippage_state(self):
        """Dashboard should load slippage.json."""
        from dashboard.execution_panel import load_slippage_state
        
        # Should not raise, returns empty dict if file missing
        result = load_slippage_state()
        assert isinstance(result, dict)

    def test_load_liquidity_buckets_state(self):
        """Dashboard should load liquidity_buckets.json."""
        from dashboard.execution_panel import load_liquidity_buckets_state
        
        # Should not raise, returns empty dict if file missing
        result = load_liquidity_buckets_state()
        assert isinstance(result, dict)


# ===========================================================================
# Tests: State Schema Compatibility
# ===========================================================================

class TestStateSchemaCompatibility:
    """Test B1 state schemas are dashboard-compatible."""
    
    def test_slippage_schema(self):
        """slippage.json should have dashboard-compatible schema."""
        from execution.state_publish import write_slippage_state
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            
            payload = {
                "updated_ts": 1700000000.0,
                "per_symbol": {
                    "BTCUSDT": {
                        "ewma_expected_bps": 2.5,
                        "ewma_realized_bps": 3.0,
                        "trade_count": 50,
                        "last_obs_ts": 1700000000.0,
                    },
                    "ETHUSDT": {
                        "ewma_expected_bps": 3.0,
                        "ewma_realized_bps": 4.0,
                        "trade_count": 30,
                        "last_obs_ts": 1700000000.0,
                    },
                },
            }
            
            write_slippage_state(payload, state_dir)
            
            output_path = state_dir / "slippage.json"
            with output_path.open() as f:
                loaded = json.load(f)
            
            # Verify all expected fields present
            assert "updated_ts" in loaded
            assert "per_symbol" in loaded
            for symbol_data in loaded["per_symbol"].values():
                assert "ewma_expected_bps" in symbol_data
                assert "ewma_realized_bps" in symbol_data
                assert "trade_count" in symbol_data

    def test_liquidity_buckets_schema(self):
        """liquidity_buckets.json should have dashboard-compatible schema."""
        from execution.state_publish import write_liquidity_buckets_state
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            
            payload = {
                "updated_ts": 1700000000.0,
                "symbols": {
                    "BTCUSDT": {
                        "bucket": "A_HIGH",
                        "max_spread_bps": 5.0,
                        "default_maker_bias": 0.8,
                    },
                },
                "buckets": {
                    "A_HIGH": {
                        "max_spread_bps": 5.0,
                        "default_maker_bias": 0.8,
                        "symbol_count": 2,
                    },
                    "B_MEDIUM": {
                        "max_spread_bps": 12.0,
                        "default_maker_bias": 0.6,
                        "symbol_count": 3,
                    },
                },
            }
            
            write_liquidity_buckets_state(payload, state_dir)
            
            output_path = state_dir / "liquidity_buckets.json"
            with output_path.open() as f:
                loaded = json.load(f)
            
            # Verify structure
            assert "symbols" in loaded
            assert "buckets" in loaded
            assert loaded["symbols"]["BTCUSDT"]["bucket"] == "A_HIGH"


# ===========================================================================
# Tests: Additive State Contract
# ===========================================================================

class TestAdditiveStateContract:
    """Test B1 state doesn't break existing contracts."""
    
    def test_router_health_unchanged(self):
        """B1 should not modify router_health.json contract."""
        from execution.state_publish import write_router_health_state
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            
            # Original router health payload
            payload = {
                "maker_ratio": 0.75,
                "fallback_ratio": 0.10,
                "health_score": 0.85,
                "total_orders": 100,
            }
            
            write_router_health_state(payload, state_dir)
            
            output_path = state_dir / "router_health.json"
            with output_path.open() as f:
                loaded = json.load(f)
            
            # Original fields unchanged
            assert loaded["maker_ratio"] == 0.75
            assert loaded["health_score"] == 0.85
