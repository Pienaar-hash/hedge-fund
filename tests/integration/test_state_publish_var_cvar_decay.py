"""
Tests for state publishing of VaR, CVaR, and Alpha Decay (v7.5_A1)
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile


# ===========================================================================
# Tests: Risk Advanced State Publishing
# ===========================================================================

class TestWriteRiskAdvancedState:
    """Test writing risk_advanced.json state file."""
    
    def test_writes_valid_json(self):
        """Should write valid JSON to file."""
        from execution.state_publish import write_risk_advanced_state
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            
            payload = {
                "updated_ts": 1700000000.0,
                "var": {
                    "portfolio_var_usd": 10000.0,
                    "portfolio_var_nav_pct": 0.10,
                },
            }
            
            write_risk_advanced_state(payload, state_dir)
            
            output_path = state_dir / "risk_advanced.json"
            assert output_path.exists()
            
            with output_path.open() as f:
                loaded = json.load(f)
            
            assert loaded["var"]["portfolio_var_usd"] == 10000.0

    def test_handles_empty_payload(self):
        """Should handle empty payload gracefully."""
        from execution.state_publish import write_risk_advanced_state
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            
            write_risk_advanced_state({}, state_dir)
            
            output_path = state_dir / "risk_advanced.json"
            assert output_path.exists()


class TestWriteAlphaDecayState:
    """Test writing alpha_decay.json state file."""
    
    def test_writes_decay_snapshot(self):
        """Should write decay snapshot to file."""
        from execution.state_publish import write_alpha_decay_state
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            
            payload = {
                "updated_ts": 1700000000.0,
                "config": {
                    "enabled": True,
                    "half_life_minutes": 45,
                },
                "symbols": {
                    "BTCUSDT": {
                        "long": {"decay_multiplier": 0.8, "age_minutes": 20},
                        "short": {"decay_multiplier": 1.0, "age_minutes": 0},
                    },
                },
            }
            
            write_alpha_decay_state(payload, state_dir)
            
            output_path = state_dir / "alpha_decay.json"
            assert output_path.exists()
            
            with output_path.open() as f:
                loaded = json.load(f)
            
            assert loaded["symbols"]["BTCUSDT"]["long"]["decay_multiplier"] == 0.8


# ===========================================================================
# Tests: Compute and Write Functions
# ===========================================================================

class TestComputeAndWriteRiskAdvancedState:
    """Test compute_and_write_risk_advanced_state function."""
    
    @patch("execution.vol_risk.build_risk_advanced_snapshot")
    def test_computes_and_writes_snapshot(self, mock_build):
        """Should compute snapshot and write to file."""
        from execution.state_publish import compute_and_write_risk_advanced_state
        
        mock_build.return_value = {
            "updated_ts": 1700000000.0,
            "var": {"portfolio_var_nav_pct": 0.08},
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            
            positions = [{"symbol": "BTCUSDT", "notional": 10000.0}]
            result = compute_and_write_risk_advanced_state(positions, 100000.0, state_dir)
            
            assert result["var"]["portfolio_var_nav_pct"] == 0.08
            mock_build.assert_called_once()

    def test_handles_exception_gracefully(self):
        """Should return empty dict on exception."""
        from execution.state_publish import compute_and_write_risk_advanced_state
        
        with patch("execution.vol_risk.build_risk_advanced_snapshot", side_effect=Exception("test error")):
            with tempfile.TemporaryDirectory() as tmpdir:
                result = compute_and_write_risk_advanced_state([], 100000.0, Path(tmpdir))
                
                assert result == {}


class TestComputeAndWriteAlphaDecayState:
    """Test compute_and_write_alpha_decay_state function."""
    
    @patch("execution.intel.symbol_score_v6.build_alpha_decay_snapshot")
    def test_computes_and_writes_snapshot(self, mock_build):
        """Should compute snapshot and write to file."""
        from execution.state_publish import compute_and_write_alpha_decay_state
        
        mock_build.return_value = {
            "updated_ts": 1700000000.0,
            "symbols": {"BTCUSDT": {"long": {"decay_multiplier": 0.9}}},
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            
            result = compute_and_write_alpha_decay_state(["BTCUSDT"], state_dir)
            
            assert "BTCUSDT" in result["symbols"]
            mock_build.assert_called_once()


# ===========================================================================
# Tests: Risk Snapshot VaR/CVaR Enrichment
# ===========================================================================

class TestRiskSnapshotVarCvarEnrichment:
    """Test VaR/CVaR enrichment in write_risk_snapshot_state."""
    
    def test_risk_snapshot_includes_var_field_when_present(self):
        """Risk snapshot should include VaR field when added to payload."""
        from execution.state_publish import _write_state_file
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            
            # Simulate enriched payload with VaR
            payload = {
                "nav_total": 100000.0,
                "dd_frac": 0.05,
                "risk_mode": "OK",
                "var": {
                    "portfolio_var_usd": 8000.0,
                    "portfolio_var_nav_pct": 0.08,
                    "within_limit": True,
                },
            }
            
            _write_state_file("risk_snapshot.json", payload, state_dir)
            
            output_path = state_dir / "risk_snapshot.json"
            with output_path.open() as f:
                loaded = json.load(f)
            
            assert "var" in loaded
            assert loaded["var"]["portfolio_var_usd"] == 8000.0


# ===========================================================================
# Tests: Dashboard-Safe Schema
# ===========================================================================

class TestDashboardSafeSchema:
    """Test that state files have dashboard-safe schemas."""
    
    def test_risk_advanced_schema(self):
        """risk_advanced.json should have dashboard-compatible schema."""
        from execution.state_publish import write_risk_advanced_state
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            
            payload = {
                "updated_ts": 1700000000.0,
                "var": {
                    "portfolio_var_usd": 10000.0,
                    "portfolio_var_nav_pct": 0.10,
                    "max_portfolio_var_nav_pct": 0.12,
                    "within_limit": True,
                    "confidence": 0.99,
                },
                "cvar": {
                    "per_symbol": {
                        "BTCUSDT": {
                            "cvar_nav_pct": 0.02,
                            "limit": 0.04,
                            "within_limit": True,
                        },
                    },
                    "max_position_cvar_nav_pct": 0.04,
                    "confidence": 0.95,
                },
            }
            
            write_risk_advanced_state(payload, state_dir)
            
            output_path = state_dir / "risk_advanced.json"
            with output_path.open() as f:
                loaded = json.load(f)
            
            # Verify schema structure for dashboard consumption
            assert "var" in loaded
            assert "portfolio_var_nav_pct" in loaded["var"]
            assert "within_limit" in loaded["var"]
            
            assert "cvar" in loaded
            assert "per_symbol" in loaded["cvar"]

    def test_alpha_decay_schema(self):
        """alpha_decay.json should have dashboard-compatible schema."""
        from execution.state_publish import write_alpha_decay_state
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            
            payload = {
                "updated_ts": 1700000000.0,
                "config": {
                    "enabled": True,
                    "half_life_minutes": 45,
                    "min_decay_multiplier": 0.35,
                },
                "symbols": {
                    "BTCUSDT": {
                        "long": {
                            "decay_multiplier": 0.75,
                            "age_minutes": 25.5,
                            "at_minimum": False,
                        },
                        "short": {
                            "decay_multiplier": 1.0,
                            "age_minutes": 0.0,
                            "at_minimum": False,
                        },
                    },
                },
            }
            
            write_alpha_decay_state(payload, state_dir)
            
            output_path = state_dir / "alpha_decay.json"
            with output_path.open() as f:
                loaded = json.load(f)
            
            # Verify schema structure for dashboard consumption
            assert "config" in loaded
            assert "symbols" in loaded
            assert "BTCUSDT" in loaded["symbols"]
            assert "decay_multiplier" in loaded["symbols"]["BTCUSDT"]["long"]


# ===========================================================================
# Tests: Additive State Contract
# ===========================================================================

class TestAdditiveStateContract:
    """Test that new state files don't break existing contracts."""
    
    def test_risk_snapshot_preserves_existing_fields(self):
        """New VaR/CVaR fields should be additive to existing risk_snapshot."""
        from execution.state_publish import _write_state_file
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            
            # Existing fields that must be preserved
            payload = {
                "dd_frac": 0.05,
                "daily_loss_frac": 0.02,
                "risk_mode": "OK",
                "circuit_breaker": {"active": False},
                "correlation_groups": {},
                # New A1 fields
                "var": {"portfolio_var_nav_pct": 0.08},
                "cvar": {"per_symbol": {}},
            }
            
            _write_state_file("risk_snapshot.json", payload, state_dir)
            
            output_path = state_dir / "risk_snapshot.json"
            with output_path.open() as f:
                loaded = json.load(f)
            
            # All existing fields must be present
            assert "dd_frac" in loaded
            assert "daily_loss_frac" in loaded
            assert "risk_mode" in loaded
            assert "circuit_breaker" in loaded
            
            # New fields also present
            assert "var" in loaded
            assert "cvar" in loaded
