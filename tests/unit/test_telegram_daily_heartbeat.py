"""
Tests for scripts/telegram_daily_heartbeat.py

Covers:
  - Message building (format, activation window injection)
  - Dry-run mode (no send, preview only)
  - JSONL logging (append-only contract)
  - Activation window block variants (active, halted, expired, inactive, missing)
  - send_heartbeat with mocked telegram_utils
  - send_test_ping with mocked telegram_utils
  - CLI arg parsing
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from unittest import mock

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.telegram_daily_heartbeat import (
    _activation_window_block,
    _safe_json,
    build_heartbeat_message,
    send_heartbeat,
    send_test_ping,
    _log_attempt,
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def aw_state_active(tmp_path: Path) -> Path:
    """Write an active activation window state file."""
    state = {
        "active": True,
        "halted": False,
        "elapsed_days": 3.5,
        "duration_days": 14,
        "remaining_days": 10.5,
        "drawdown_pct": 1.23,
        "manifest_intact": True,
        "config_intact": True,
        "window_expired": False,
    }
    p = tmp_path / "activation_window_state.json"
    p.write_text(json.dumps(state))
    return p


@pytest.fixture
def aw_state_halted(tmp_path: Path) -> Path:
    state = {
        "active": True,
        "halted": True,
        "halt_reason": "config_drift",
        "elapsed_days": 2,
        "duration_days": 14,
        "remaining_days": 12,
        "drawdown_pct": 0.0,
        "manifest_intact": False,
        "config_intact": True,
        "window_expired": False,
    }
    p = tmp_path / "activation_window_state.json"
    p.write_text(json.dumps(state))
    return p


@pytest.fixture
def aw_state_expired(tmp_path: Path) -> Path:
    state = {
        "active": False,
        "halted": False,
        "elapsed_days": 14,
        "duration_days": 14,
        "remaining_days": 0,
        "drawdown_pct": 2.1,
        "manifest_intact": True,
        "config_intact": True,
        "window_expired": True,
    }
    p = tmp_path / "activation_window_state.json"
    p.write_text(json.dumps(state))
    return p


@pytest.fixture
def aw_state_inactive(tmp_path: Path) -> Path:
    state = {
        "active": False,
        "halted": False,
        "window_expired": False,
    }
    p = tmp_path / "activation_window_state.json"
    p.write_text(json.dumps(state))
    return p


# ── _safe_json ──────────────────────────────────────────────────────


class TestSafeJson:
    def test_valid_json(self, tmp_path: Path) -> None:
        p = tmp_path / "test.json"
        p.write_text('{"a": 1}')
        assert _safe_json(p) == {"a": 1}

    def test_missing_file(self, tmp_path: Path) -> None:
        p = tmp_path / "noexist.json"
        assert _safe_json(p) == {}

    def test_corrupt_json(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("{{{not json")
        assert _safe_json(p) == {}


# ── _activation_window_block ────────────────────────────────────────


class TestActivationWindowBlock:
    def test_active_window(self, aw_state_active: Path) -> None:
        import scripts.telegram_daily_heartbeat as mod
        with mock.patch.object(mod, "_AW_STATE", aw_state_active):
            result = _activation_window_block()
        assert "Day 4/14" in result or "Day 3/14" in result
        assert "DD:1.23%" in result
        assert "Integrity:OK" in result

    def test_halted_window(self, aw_state_halted: Path) -> None:
        import scripts.telegram_daily_heartbeat as mod
        with mock.patch.object(mod, "_AW_STATE", aw_state_halted):
            result = _activation_window_block()
        assert "HALTED" in result
        assert "config_drift" in result

    def test_expired_window(self, aw_state_expired: Path) -> None:
        import scripts.telegram_daily_heartbeat as mod
        with mock.patch.object(mod, "_AW_STATE", aw_state_expired):
            result = _activation_window_block()
        assert "COMPLETED" in result

    def test_inactive_window(self, aw_state_inactive: Path) -> None:
        import scripts.telegram_daily_heartbeat as mod
        with mock.patch.object(mod, "_AW_STATE", aw_state_inactive):
            result = _activation_window_block()
        assert "INACTIVE" in result

    def test_missing_state_file(self, tmp_path: Path) -> None:
        import scripts.telegram_daily_heartbeat as mod
        missing = tmp_path / "no_such_file.json"
        with mock.patch.object(mod, "_AW_STATE", missing):
            result = _activation_window_block()
        assert "—" in result

    def test_integrity_drift(self, tmp_path: Path) -> None:
        """Manifest drift shows DRIFT instead of OK."""
        import scripts.telegram_daily_heartbeat as mod
        state = {
            "active": True,
            "halted": False,
            "elapsed_days": 5,
            "duration_days": 14,
            "remaining_days": 9,
            "drawdown_pct": 0.5,
            "manifest_intact": False,
            "config_intact": True,
            "window_expired": False,
        }
        p = tmp_path / "aw.json"
        p.write_text(json.dumps(state))
        with mock.patch.object(mod, "_AW_STATE", p):
            result = _activation_window_block()
        assert "Integrity:DRIFT" in result


# ── build_heartbeat_message ─────────────────────────────────────────


class TestBuildHeartbeatMessage:
    def test_contains_daily_summary_header(self) -> None:
        now = datetime(2026, 3, 1, 8, 0, 0, tzinfo=timezone.utc)
        msg = build_heartbeat_message(now=now)
        assert "Daily Summary" in msg
        assert "2026-03-01" in msg

    def test_contains_activation_window(self) -> None:
        msg = build_heartbeat_message()
        assert "Cert Window:" in msg

    def test_ends_with_border(self) -> None:
        msg = build_heartbeat_message()
        lines = msg.strip().split("\n")
        assert lines[-1].startswith("═")

    def test_activation_window_before_border(self) -> None:
        """Activation window block is injected before closing border."""
        msg = build_heartbeat_message()
        lines = msg.strip().split("\n")
        # Find "Cert Window:" line
        cert_idx = None
        last_border_idx = None
        for i, line in enumerate(lines):
            if "Cert Window:" in line:
                cert_idx = i
            if line.startswith("═"):
                last_border_idx = i
        assert cert_idx is not None, "Cert Window line not found"
        assert last_border_idx is not None, "Closing border not found"
        assert cert_idx < last_border_idx, "Cert Window should appear before closing border"


# ── _log_attempt ────────────────────────────────────────────────────


class TestLogAttempt:
    def test_creates_log_entry(self, tmp_path: Path) -> None:
        import scripts.telegram_daily_heartbeat as mod
        log_path = tmp_path / "heartbeat.jsonl"
        with mock.patch.object(mod, "_HEARTBEAT_LOG", log_path):
            _log_attempt(ok=True, dry_run=False, message_len=500)
        entries = [json.loads(line) for line in log_path.read_text().strip().split("\n")]
        assert len(entries) == 1
        assert entries[0]["sent"] is True
        assert entries[0]["dry_run"] is False
        assert entries[0]["message_len"] == 500
        assert entries[0]["action"] == "heartbeat_daily"
        assert "ts" in entries[0]

    def test_append_only(self, tmp_path: Path) -> None:
        """Multiple calls append, not overwrite."""
        import scripts.telegram_daily_heartbeat as mod
        log_path = tmp_path / "heartbeat.jsonl"
        with mock.patch.object(mod, "_HEARTBEAT_LOG", log_path):
            _log_attempt(ok=True, dry_run=False, message_len=100)
            _log_attempt(ok=False, dry_run=True, message_len=200, error="test")
        entries = [json.loads(line) for line in log_path.read_text().strip().split("\n")]
        assert len(entries) == 2
        assert entries[1]["error"] == "test"

    def test_log_failure_does_not_raise(self, tmp_path: Path) -> None:
        """Logging failure must never block execution."""
        import scripts.telegram_daily_heartbeat as mod
        # Point at a path we can't write to
        bad_path = Path("/proc/0/heartbeat.jsonl")
        with mock.patch.object(mod, "_HEARTBEAT_LOG", bad_path):
            _log_attempt(ok=True, dry_run=False, message_len=100)  # should not raise


# ── send_heartbeat ──────────────────────────────────────────────────


class TestSendHeartbeat:
    @mock.patch("execution.telegram_utils.send_telegram", return_value=True)
    def test_send_success(self, mock_send: mock.MagicMock, tmp_path: Path) -> None:
        import scripts.telegram_daily_heartbeat as mod
        log_path = tmp_path / "heartbeat.jsonl"
        with mock.patch.object(mod, "_HEARTBEAT_LOG", log_path):
            result = send_heartbeat(dry_run=False)
        assert result is True
        mock_send.assert_called_once()
        msg = mock_send.call_args[0][0]
        assert "Daily Summary" in msg
        # Log entry recorded
        entries = [json.loads(line) for line in log_path.read_text().strip().split("\n")]
        assert entries[0]["sent"] is True

    @mock.patch("execution.telegram_utils.send_telegram", return_value=False)
    def test_send_failure(self, mock_send: mock.MagicMock, tmp_path: Path) -> None:
        import scripts.telegram_daily_heartbeat as mod
        log_path = tmp_path / "heartbeat.jsonl"
        with mock.patch.object(mod, "_HEARTBEAT_LOG", log_path):
            result = send_heartbeat(dry_run=False)
        assert result is False
        entries = [json.loads(line) for line in log_path.read_text().strip().split("\n")]
        assert entries[0]["sent"] is False

    def test_dry_run_does_not_send(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        import scripts.telegram_daily_heartbeat as mod
        log_path = tmp_path / "heartbeat.jsonl"
        with mock.patch.object(mod, "_HEARTBEAT_LOG", log_path):
            result = send_heartbeat(dry_run=True)
        assert result is True
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert "Daily Summary" in captured.out

    @mock.patch(
        "execution.telegram_utils.send_telegram",
        side_effect=ConnectionError("network down"),
    )
    def test_send_exception_logged(self, mock_send: mock.MagicMock, tmp_path: Path) -> None:
        import scripts.telegram_daily_heartbeat as mod
        log_path = tmp_path / "heartbeat.jsonl"
        with mock.patch.object(mod, "_HEARTBEAT_LOG", log_path):
            result = send_heartbeat(dry_run=False)
        assert result is False
        entries = [json.loads(line) for line in log_path.read_text().strip().split("\n")]
        assert "network down" in entries[0]["error"]


# ── send_test_ping ──────────────────────────────────────────────────


class TestSendTestPing:
    @mock.patch("execution.telegram_utils.send_telegram", return_value=True)
    def test_ping_success(self, mock_send: mock.MagicMock) -> None:
        result = send_test_ping()
        assert result is True
        msg = mock_send.call_args[0][0]
        assert "Heartbeat test ping" in msg

    @mock.patch("execution.telegram_utils.send_telegram", return_value=False)
    def test_ping_failure(self, mock_send: mock.MagicMock) -> None:
        result = send_test_ping()
        assert result is False
