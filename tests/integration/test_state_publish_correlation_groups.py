"""
Test suite for correlation_groups in execution/state_publish.py.

Tests that correlation_groups block is correctly written to risk_snapshot.json.
"""

import json
import os
import pathlib
import tempfile
import pytest
from unittest.mock import patch

from execution.risk_loader import CorrelationGroupConfig, CorrelationGroupsConfig


@pytest.fixture
def temp_state_dir():
    """Create a temporary directory for state files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield pathlib.Path(tmpdir)


@pytest.fixture
def mock_correlation_config():
    """Create a mock correlation groups config."""
    return CorrelationGroupsConfig(
        groups={
            "L1_bluechips": CorrelationGroupConfig(
                symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
                max_group_nav_pct=0.35,
            ),
            "L2_alts": CorrelationGroupConfig(
                symbols=["DOGEUSDT", "LINKUSDT"],
                max_group_nav_pct=0.25,
            ),
        }
    )


def _base_payload(nav_usd: float = 50000.0) -> dict:
    """Create a base risk snapshot payload."""
    return {
        "nav_usd": nav_usd,
        "dd_state": {
            "drawdown": {"dd_pct": 0.05, "daily_loss": {"pct": 0.02}},
        },
        "risk_mode": "OK",
        "ts": 1700000000,
    }


class TestCorrelationGroupsInRiskSnapshot:
    """Tests for correlation_groups block in risk snapshot."""
    
    def test_correlation_groups_written_to_snapshot(
        self,
        temp_state_dir,
        mock_correlation_config,
    ) -> None:
        """correlation_groups block should appear in risk_snapshot.json."""
        # Mock exposure returns for each group
        def mock_compute_exposure(positions, nav_total_usd, corr_cfg):
            return {
                "L1_bluechips": 0.30,
                "L2_alts": 0.06,
            }
        
        with patch("execution.state_publish.STATE_DIR", temp_state_dir), \
             patch("execution.risk_loader.load_correlation_groups_config", return_value=mock_correlation_config), \
             patch("execution.correlation_groups.compute_group_exposure_nav_pct", side_effect=mock_compute_exposure):
            
            from execution.state_publish import write_risk_snapshot_state
            
            payload = _base_payload()
            write_risk_snapshot_state(payload)
        
        snapshot_path = temp_state_dir / "risk_snapshot.json"
        assert snapshot_path.exists()
        
        with open(snapshot_path) as f:
            snapshot = json.load(f)
        
        # Check correlation_groups exists
        assert "correlation_groups" in snapshot
    
    def test_empty_correlation_groups_when_not_configured(
        self,
        temp_state_dir,
    ) -> None:
        """correlation_groups should be empty dict when no groups configured."""
        empty_config = CorrelationGroupsConfig(groups={})
        
        with patch("execution.state_publish.STATE_DIR", temp_state_dir), \
             patch("execution.risk_loader.load_correlation_groups_config", return_value=empty_config):
            
            from execution.state_publish import write_risk_snapshot_state
            
            write_risk_snapshot_state(_base_payload())
        
        snapshot_path = temp_state_dir / "risk_snapshot.json"
        with open(snapshot_path) as f:
            snapshot = json.load(f)
        
        # Empty or missing is OK
        corr_groups = snapshot.get("correlation_groups")
        assert corr_groups is None or corr_groups == {}
    
    def test_correlation_groups_error_handling(
        self,
        temp_state_dir,
    ) -> None:
        """Snapshot should still write if correlation config fails to load."""
        def failing_load():
            raise FileNotFoundError("correlation_groups.json not found")
        
        with patch("execution.state_publish.STATE_DIR", temp_state_dir), \
             patch("execution.risk_loader.load_correlation_groups_config", side_effect=failing_load):
            
            from execution.state_publish import write_risk_snapshot_state
            
            # Should not raise; should gracefully degrade
            write_risk_snapshot_state(_base_payload())
        
        snapshot_path = temp_state_dir / "risk_snapshot.json"
        assert snapshot_path.exists()
        
        with open(snapshot_path) as f:
            snapshot = json.load(f)
        
        # correlation_groups should be empty or missing, not crash
        corr_groups = snapshot.get("correlation_groups")
        assert corr_groups is None or corr_groups == {}
