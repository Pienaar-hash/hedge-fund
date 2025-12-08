"""
Tests for risk snapshot state publishing with circuit breaker fields.

Tests that write_risk_snapshot_state includes portfolio_dd_pct and circuit_breaker fields.
"""
from __future__ import annotations

import json
import pathlib
import tempfile
from typing import Any, Dict
from unittest import mock

import pytest


class TestRiskSnapshotCircuitBreakerFields:
    """Test suite for circuit breaker fields in risk_snapshot.json."""

    def test_risk_snapshot_includes_circuit_breaker_fields(self, tmp_path):
        """Risk snapshot should include portfolio_dd_pct and circuit_breaker."""
        from execution import state_publish

        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a mock nav_log.json
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        nav_log_path = logs_dir / "nav_log.json"
        nav_log_path.write_text(json.dumps([
            {"nav": 9000.0},
            {"nav": 10000.0},
            {"nav": 9500.0},
        ]))
        
        # Mock at module level for nested imports
        risk_config_mock = {
            "circuit_breakers": {
                "max_portfolio_dd_nav_pct": 0.10,
            }
        }
        
        risk_mode_mock = mock.Mock(
            mode=mock.Mock(value="OK"),
            reason="",
            score=0.0,
        )
        
        # Mock the LOG_DIR and risk config
        with mock.patch.object(state_publish, 'LOG_DIR', logs_dir):
            with mock.patch('execution.risk_loader.load_risk_config', return_value=risk_config_mock):
                with mock.patch('execution.risk_engine_v6.compute_risk_mode_from_state', return_value=risk_mode_mock):
                    payload = {
                        "dd_state": {
                            "drawdown": {
                                "dd_pct": 5.0,
                                "daily_loss": {"pct": 1.0},
                            }
                        }
                    }
                    
                    state_publish.write_risk_snapshot_state(payload, state_dir)
        
        # Read the written file
        result_path = state_dir / "risk_snapshot.json"
        assert result_path.exists()
        
        with result_path.open("r", encoding="utf-8") as f:
            result = json.load(f)
        
        # Check portfolio_dd_pct is present
        assert "portfolio_dd_pct" in result
        # DD = (10000 - 9500) / 10000 = 0.05
        assert result["portfolio_dd_pct"] is not None
        assert abs(result["portfolio_dd_pct"] - 0.05) < 0.001
        
        # Check circuit_breaker block is present
        assert "circuit_breaker" in result
        cb = result["circuit_breaker"]
        assert "max_portfolio_dd_nav_pct" in cb
        assert cb["max_portfolio_dd_nav_pct"] == 0.10
        assert "active" in cb
        # DD 5% < threshold 10%, so not active
        assert cb["active"] is False

    def test_risk_snapshot_circuit_breaker_active_when_tripped(self, tmp_path):
        """Circuit breaker should be active when DD exceeds threshold."""
        from execution import state_publish

        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        
        # Create nav_log with 15% drawdown
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        nav_log_path = logs_dir / "nav_log.json"
        nav_log_path.write_text(json.dumps([
            {"nav": 9000.0},
            {"nav": 10000.0},  # peak
            {"nav": 8500.0},   # 15% drawdown
        ]))
        
        risk_config_mock = {
            "circuit_breakers": {
                "max_portfolio_dd_nav_pct": 0.10,  # 10% threshold
            }
        }
        
        risk_mode_mock = mock.Mock(
            mode=mock.Mock(value="DEFENSIVE"),
            reason="circuit_breaker",
            score=0.5,
        )
        
        with mock.patch.object(state_publish, 'LOG_DIR', logs_dir):
            with mock.patch('execution.risk_loader.load_risk_config', return_value=risk_config_mock):
                with mock.patch('execution.risk_engine_v6.compute_risk_mode_from_state', return_value=risk_mode_mock):
                    payload = {
                        "dd_state": {
                            "drawdown": {
                                "dd_pct": 15.0,
                                "daily_loss": {"pct": 2.0},
                            }
                        }
                    }
                    
                    state_publish.write_risk_snapshot_state(payload, state_dir)
        
        result_path = state_dir / "risk_snapshot.json"
        with result_path.open("r", encoding="utf-8") as f:
            result = json.load(f)
        
        # DD 15% > threshold 10%, so active
        assert result["circuit_breaker"]["active"] is True
        assert abs(result["portfolio_dd_pct"] - 0.15) < 0.001

    def test_risk_snapshot_no_circuit_config(self, tmp_path):
        """When circuit breaker not configured, fields should show None/False."""
        from execution import state_publish

        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        nav_log_path = logs_dir / "nav_log.json"
        nav_log_path.write_text(json.dumps([{"nav": 10000.0}]))
        
        # No circuit_breakers config
        risk_config_mock = {}
        
        risk_mode_mock = mock.Mock(
            mode=mock.Mock(value="OK"),
            reason="",
            score=0.0,
        )
        
        with mock.patch.object(state_publish, 'LOG_DIR', logs_dir):
            with mock.patch('execution.risk_loader.load_risk_config', return_value=risk_config_mock):
                with mock.patch('execution.risk_engine_v6.compute_risk_mode_from_state', return_value=risk_mode_mock):
                    payload = {"dd_state": {"drawdown": {}}}
                    state_publish.write_risk_snapshot_state(payload, state_dir)
        
        result_path = state_dir / "risk_snapshot.json"
        with result_path.open("r", encoding="utf-8") as f:
            result = json.load(f)
        
        assert "circuit_breaker" in result
        cb = result["circuit_breaker"]
        assert cb["max_portfolio_dd_nav_pct"] is None
        assert cb["active"] is False

    def test_risk_snapshot_empty_nav_history(self, tmp_path):
        """When nav history is empty, portfolio_dd_pct should be None."""
        from execution import state_publish

        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        # No nav_log.json file
        
        risk_config_mock = {
            "circuit_breakers": {
                "max_portfolio_dd_nav_pct": 0.10,
            }
        }
        
        risk_mode_mock = mock.Mock(
            mode=mock.Mock(value="OK"),
            reason="",
            score=0.0,
        )
        
        with mock.patch.object(state_publish, 'LOG_DIR', logs_dir):
            with mock.patch('execution.risk_loader.load_risk_config', return_value=risk_config_mock):
                with mock.patch('execution.risk_engine_v6.compute_risk_mode_from_state', return_value=risk_mode_mock):
                    payload = {"dd_state": {"drawdown": {}}}
                    state_publish.write_risk_snapshot_state(payload, state_dir)
        
        result_path = state_dir / "risk_snapshot.json"
        with result_path.open("r", encoding="utf-8") as f:
            result = json.load(f)
        
        assert result["portfolio_dd_pct"] is None
        assert result["circuit_breaker"]["active"] is False

    def test_risk_snapshot_preserves_existing_fields(self, tmp_path):
        """Circuit breaker fields should be additive, not overwrite existing fields."""
        from execution import state_publish

        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        nav_log_path = logs_dir / "nav_log.json"
        nav_log_path.write_text(json.dumps([{"nav": 10000.0}]))
        
        risk_config_mock = {
            "circuit_breakers": {
                "max_portfolio_dd_nav_pct": 0.10,
            }
        }
        
        risk_mode_mock = mock.Mock(
            mode=mock.Mock(value="OK"),
            reason="",
            score=0.0,
        )
        
        with mock.patch.object(state_publish, 'LOG_DIR', logs_dir):
            with mock.patch('execution.risk_loader.load_risk_config', return_value=risk_config_mock):
                with mock.patch('execution.risk_engine_v6.compute_risk_mode_from_state', return_value=risk_mode_mock):
                    # Payload with existing fields
                    payload = {
                        "dd_state": {
                            "drawdown": {
                                "dd_pct": 3.0,
                                "peak": 10000.0,
                                "daily_loss": {"pct": 1.0},
                            }
                        },
                        "nav_health": {"fresh": True},
                        "existing_field": "should_be_preserved",
                    }
                    
                    state_publish.write_risk_snapshot_state(payload, state_dir)
        
        result_path = state_dir / "risk_snapshot.json"
        with result_path.open("r", encoding="utf-8") as f:
            result = json.load(f)
        
        # Check existing fields are preserved
        assert result.get("existing_field") == "should_be_preserved"
        assert result.get("nav_health") == {"fresh": True}
        assert "dd_state" in result
        
        # Check new fields are added
        assert "portfolio_dd_pct" in result
        assert "circuit_breaker" in result
        assert "dd_frac" in result
        assert "daily_loss_frac" in result
