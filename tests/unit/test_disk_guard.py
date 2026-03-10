"""Tests for executor disk pressure guard (_maybe_check_disk_pressure)."""
from __future__ import annotations

import time
from collections import namedtuple
from pathlib import Path
from unittest.mock import patch

import pytest

# Minimal stubs so executor_live can import without Binance/network deps
import tests.conftest  # noqa: F401 — ensure shared fixtures load


DiskUsage = namedtuple("DiskUsage", ["total", "used", "free"])

# 38GB disk examples
DISK_OK = DiskUsage(total=38 * 1024**3, used=19 * 1024**3, free=19 * 1024**3)        # 50%
DISK_WARN = DiskUsage(total=38 * 1024**3, used=32 * 1024**3, free=6 * 1024**3)       # ~84%
DISK_CRITICAL = DiskUsage(total=38 * 1024**3, used=35 * 1024**3, free=3 * 1024**3)   # ~92%


@pytest.fixture(autouse=True)
def _reset_disk_guard_state():
    """Reset module-level state between tests."""
    import execution.executor_live as ex
    ex._STATE.last_disk_check_ts = 0.0
    ex._STATE.last_disk_alert_ts = 0.0
    yield
    ex._STATE.last_disk_check_ts = 0.0
    ex._STATE.last_disk_alert_ts = 0.0


class TestDiskGuard:
    """Tests for _maybe_check_disk_pressure."""

    def test_ok_disk_no_alert(self, tmp_path: Path):
        import execution.executor_live as ex
        ex._ENV_EVENTS_PATH = tmp_path / "env_events.jsonl"
        with patch("shutil.disk_usage", return_value=DISK_OK):
            ex._maybe_check_disk_pressure(ex._STATE, force=True)
        # No file written when disk is healthy
        assert not ex._ENV_EVENTS_PATH.exists()

    def test_warning_threshold_emits_event(self, tmp_path: Path):
        import execution.executor_live as ex
        ex._ENV_EVENTS_PATH = tmp_path / "env_events.jsonl"
        with patch("shutil.disk_usage", return_value=DISK_WARN):
            ex._maybe_check_disk_pressure(ex._STATE, force=True)
        assert ex._ENV_EVENTS_PATH.exists()
        content = ex._ENV_EVENTS_PATH.read_text()
        assert '"disk_pressure"' in content
        assert '"warning"' in content

    def test_critical_threshold_emits_event(self, tmp_path: Path):
        import execution.executor_live as ex
        ex._ENV_EVENTS_PATH = tmp_path / "env_events.jsonl"
        with patch("shutil.disk_usage", return_value=DISK_CRITICAL):
            ex._maybe_check_disk_pressure(ex._STATE, force=True)
        assert ex._ENV_EVENTS_PATH.exists()
        content = ex._ENV_EVENTS_PATH.read_text()
        assert '"disk_pressure"' in content
        assert '"critical"' in content

    def test_interval_gating_skips_check(self, tmp_path: Path):
        """Without force=True, check is skipped if interval hasn't elapsed."""
        import execution.executor_live as ex
        ex._ENV_EVENTS_PATH = tmp_path / "env_events.jsonl"
        ex._STATE.last_disk_check_ts = time.time()  # Just checked
        with patch("shutil.disk_usage", return_value=DISK_CRITICAL) as mock_du:
            ex._maybe_check_disk_pressure(ex._STATE, force=False)
        mock_du.assert_not_called()
        assert not ex._ENV_EVENTS_PATH.exists()

    def test_alert_cooldown_prevents_spam(self, tmp_path: Path):
        """Second alert within cooldown window is suppressed."""
        import execution.executor_live as ex
        ex._ENV_EVENTS_PATH = tmp_path / "env_events.jsonl"
        with patch("shutil.disk_usage", return_value=DISK_WARN):
            ex._maybe_check_disk_pressure(ex._STATE, force=True)
            # First alert written
            assert ex._ENV_EVENTS_PATH.exists()
            first_size = ex._ENV_EVENTS_PATH.stat().st_size
            # Reset check interval but NOT alert cooldown
            ex._STATE.last_disk_check_ts = 0.0
            ex._maybe_check_disk_pressure(ex._STATE, force=True)
            # File size unchanged — second alert suppressed
            assert ex._ENV_EVENTS_PATH.stat().st_size == first_size

    def test_force_bypasses_interval(self, tmp_path: Path):
        """force=True always runs the check regardless of interval."""
        import execution.executor_live as ex
        ex._ENV_EVENTS_PATH = tmp_path / "env_events.jsonl"
        ex._STATE.last_disk_check_ts = time.time()  # Just checked
        with patch("shutil.disk_usage", return_value=DISK_WARN):
            ex._maybe_check_disk_pressure(ex._STATE, force=True)
        assert ex._ENV_EVENTS_PATH.exists()

    def test_oserror_handled_gracefully(self, tmp_path: Path):
        """OSError from disk_usage doesn't crash the executor."""
        import execution.executor_live as ex
        ex._ENV_EVENTS_PATH = tmp_path / "env_events.jsonl"
        with patch("shutil.disk_usage", side_effect=OSError("device not found")):
            ex._maybe_check_disk_pressure(ex._STATE, force=True)  # Should not raise
        assert not ex._ENV_EVENTS_PATH.exists()
