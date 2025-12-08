"""
Tests for v7.5_C2 state publishing for factor diagnostics.

Verifies:
- State file structure
- Dashboard loaders consume state correctly
- State write/read round-trip
"""
import pytest
import json
import tempfile
from pathlib import Path


class TestFactorDiagnosticsStateStructure:
    """Test factor diagnostics state structure."""
    
    def test_snapshot_to_dict_structure(self):
        """FactorDiagnosticsSnapshot.to_dict has correct structure."""
        from execution.intel.symbol_score_v6 import build_factor_vector
        from execution.factor_diagnostics import (
            build_factor_diagnostics_snapshot,
            FactorDiagnosticsConfig,
        )
        
        vectors = [
            build_factor_vector("BTC", {"trend": 0.8, "carry": 0.3}, 0.65, "LONG"),
            build_factor_vector("ETH", {"trend": 0.6, "carry": 0.4}, 0.55, "LONG"),
        ]
        
        cfg = FactorDiagnosticsConfig(
            enabled=True,
            factors=["trend", "carry"],
            normalization_mode="zscore",
            max_abs_zscore=3.0,
        )
        
        snapshot = build_factor_diagnostics_snapshot(vectors, cfg)
        d = snapshot.to_dict()
        
        assert "updated_ts" in d
        assert "per_symbol" in d
        assert "covariance" in d
        assert "config" in d
    
    def test_per_symbol_has_factors_and_direction(self):
        """per_symbol entries have factors dict and direction."""
        from execution.intel.symbol_score_v6 import build_factor_vector
        from execution.factor_diagnostics import (
            build_factor_diagnostics_snapshot,
            FactorDiagnosticsConfig,
        )
        
        vectors = [
            build_factor_vector("BTC", {"trend": 0.8}, 0.65, "LONG"),
        ]
        
        cfg = FactorDiagnosticsConfig(
            enabled=True,
            factors=["trend"],
        )
        
        snapshot = build_factor_diagnostics_snapshot(vectors, cfg)
        d = snapshot.to_dict()
        
        # Key format is SYMBOL:DIRECTION
        entry = d["per_symbol"].get("BTC:LONG", {})
        assert "factors" in entry
        assert "direction" in entry
        assert entry["direction"] == "LONG"
    
    def test_covariance_has_matrix_and_vols(self):
        """Covariance block has matrices and volatilities."""
        from execution.intel.symbol_score_v6 import build_factor_vector
        from execution.factor_diagnostics import (
            build_factor_diagnostics_snapshot,
            FactorDiagnosticsConfig,
        )
        
        vectors = [
            build_factor_vector("A", {"x": 0.5, "y": 0.3}, 0.5),
            build_factor_vector("B", {"x": 0.7, "y": 0.8}, 0.6),
            build_factor_vector("C", {"x": 0.2, "y": 0.5}, 0.4),
        ]
        
        cfg = FactorDiagnosticsConfig(factors=["x", "y"])
        
        snapshot = build_factor_diagnostics_snapshot(vectors, cfg)
        d = snapshot.to_dict()
        
        cov = d["covariance"]
        assert "factors" in cov
        assert "covariance_matrix" in cov
        assert "correlation_matrix" in cov
        assert "factor_vols" in cov


class TestFactorPnlStateStructure:
    """Test factor PnL state structure."""
    
    def test_pnl_snapshot_to_dict_structure(self):
        """FactorPnlSnapshot.to_dict has correct structure."""
        from execution.factor_pnl_attribution import TradeRecord, compute_factor_pnl_snapshot
        
        trades = [
            TradeRecord("BTC", "LONG", 100.0, {"trend": 0.8, "carry": 0.2}),
        ]
        
        result = compute_factor_pnl_snapshot(trades, ["trend", "carry"])
        d = result.to_dict()
        
        assert "by_factor" in d
        assert "pct_by_factor" in d
        assert "total_pnl_usd" in d
        assert "window_days" in d
        assert "trade_count" in d
        assert "updated_ts" in d


class TestStateWriteRead:
    """Test state file write/read round-trip."""
    
    def test_factor_diagnostics_write_read(self):
        """Factor diagnostics state survives write/read cycle."""
        from execution.intel.symbol_score_v6 import build_factor_vector
        from execution.factor_diagnostics import (
            build_factor_diagnostics_snapshot,
            FactorDiagnosticsConfig,
            write_factor_diagnostics_state,
            load_factor_diagnostics_state,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "factor_diagnostics.json"
            
            vectors = [
                build_factor_vector("BTC", {"trend": 0.8}, 0.65, "LONG"),
            ]
            cfg = FactorDiagnosticsConfig(factors=["trend"])
            snapshot = build_factor_diagnostics_snapshot(vectors, cfg)
            
            # Write
            write_factor_diagnostics_state(snapshot, path)
            
            # Read
            loaded = load_factor_diagnostics_state(path)
            
            assert "per_symbol" in loaded
            assert "BTC:LONG" in loaded["per_symbol"]
    
    def test_factor_pnl_write_read(self):
        """Factor PnL state survives write/read cycle."""
        from execution.factor_pnl_attribution import (
            TradeRecord,
            compute_factor_pnl_snapshot,
            write_factor_pnl_state,
            load_factor_pnl_state,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "factor_pnl.json"
            
            trades = [TradeRecord("BTC", "LONG", 100.0, {"trend": 1.0})]
            snapshot = compute_factor_pnl_snapshot(trades, ["trend"])
            
            # Write
            write_factor_pnl_state(snapshot, path)
            
            # Read
            loaded = load_factor_pnl_state(path)
            
            assert loaded["total_pnl_usd"] == 100.0
            assert loaded["by_factor"]["trend"] == 100.0


class TestDashboardLoaders:
    """Test dashboard state_v7 loaders."""
    
    def test_load_factor_diagnostics_state_empty(self):
        """load_factor_diagnostics_state returns empty dict for missing file."""
        from dashboard.state_v7 import load_factor_diagnostics_state
        
        result = load_factor_diagnostics_state()
        assert isinstance(result, dict)
    
    def test_load_factor_pnl_state_empty(self):
        """load_factor_pnl_state returns empty dict for missing file."""
        from dashboard.state_v7 import load_factor_pnl_state
        
        result = load_factor_pnl_state()
        assert isinstance(result, dict)
    
    def test_get_factor_correlation_matrix_empty(self):
        """get_factor_correlation_matrix handles empty state."""
        from dashboard.state_v7 import get_factor_correlation_matrix
        
        factors, matrix = get_factor_correlation_matrix({})
        
        assert factors == []
        assert matrix == []
    
    def test_get_factor_volatilities_empty(self):
        """get_factor_volatilities handles empty state."""
        from dashboard.state_v7 import get_factor_volatilities
        
        result = get_factor_volatilities({})
        
        assert result == {}
    
    def test_get_factor_pnl_summary_empty(self):
        """get_factor_pnl_summary handles empty state."""
        from dashboard.state_v7 import get_factor_pnl_summary
        
        result = get_factor_pnl_summary({})
        
        assert result["total_pnl_usd"] == 0.0
        assert result["trade_count"] == 0
    
    def test_get_symbol_factor_fingerprint_empty(self):
        """get_symbol_factor_fingerprint handles missing symbol."""
        from dashboard.state_v7 import get_symbol_factor_fingerprint
        
        result = get_symbol_factor_fingerprint("BTCUSDT", "LONG", {})
        
        assert result == {}


class TestStatePublishFunctions:
    """Test state_publish.py factor functions."""
    
    def test_write_factor_diagnostics_state_exists(self):
        """write_factor_diagnostics_state function exists."""
        from execution.state_publish import write_factor_diagnostics_state
        
        assert callable(write_factor_diagnostics_state)
    
    def test_write_factor_pnl_state_exists(self):
        """write_factor_pnl_state function exists."""
        from execution.state_publish import write_factor_pnl_state
        
        assert callable(write_factor_pnl_state)
    
    def test_compute_and_write_factor_diagnostics_state_exists(self):
        """compute_and_write_factor_diagnostics_state function exists."""
        from execution.state_publish import compute_and_write_factor_diagnostics_state
        
        assert callable(compute_and_write_factor_diagnostics_state)
    
    def test_compute_and_write_factor_pnl_state_exists(self):
        """compute_and_write_factor_pnl_state function exists."""
        from execution.state_publish import compute_and_write_factor_pnl_state
        
        assert callable(compute_and_write_factor_pnl_state)


class TestJsonSerializable:
    """Test that all state objects are JSON serializable."""
    
    def test_factor_diagnostics_json_serializable(self):
        """Factor diagnostics snapshot is fully JSON serializable."""
        from execution.intel.symbol_score_v6 import build_factor_vector
        from execution.factor_diagnostics import (
            build_factor_diagnostics_snapshot,
            FactorDiagnosticsConfig,
        )
        
        vectors = [
            build_factor_vector("BTC", {"trend": 0.8, "carry": 0.3}, 0.65),
            build_factor_vector("ETH", {"trend": 0.6, "carry": 0.4}, 0.55),
        ]
        cfg = FactorDiagnosticsConfig(factors=["trend", "carry"])
        snapshot = build_factor_diagnostics_snapshot(vectors, cfg)
        
        # Should not raise
        serialized = json.dumps(snapshot.to_dict())
        
        # Round-trip
        parsed = json.loads(serialized)
        assert "per_symbol" in parsed
        assert "covariance" in parsed
    
    def test_factor_pnl_json_serializable(self):
        """Factor PnL snapshot is fully JSON serializable."""
        from execution.factor_pnl_attribution import TradeRecord, compute_factor_pnl_snapshot
        
        trades = [
            TradeRecord("BTC", "LONG", 100.0, {"trend": 0.8, "carry": 0.2}),
        ]
        result = compute_factor_pnl_snapshot(trades, ["trend", "carry"])
        
        # Should not raise
        serialized = json.dumps(result.to_dict())
        
        # Round-trip
        parsed = json.loads(serialized)
        assert "by_factor" in parsed
        assert "total_pnl_usd" in parsed
