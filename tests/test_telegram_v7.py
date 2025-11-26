"""
Unit tests for v7 Telegram alerts: state-driven, low-noise.

Tests verify:
  - Non-JSON or JSON without required keys is suppressed (no POST) in 4h-only mode
  - Proper 4h JSON payload is POSTed and telegram_state.json is updated
  - Duplicate attempt with same last_4h_close_ts is suppressed
  - Rate limit EXEC_TELEGRAM_MAX_PER_MIN=0 blocks all sends
  - State persistence with atomic writes
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest import mock

import pytest

import execution.telegram_alerts_v7 as ta7
import execution.telegram_utils as tu


class TestTelegramUtilsStateHelpers:
    """Test state persistence helpers in telegram_utils."""

    def test_load_telegram_state_creates_default(self, tmp_path):
        """load_telegram_state returns default structure when file missing."""
        state_path = tmp_path / "nonexistent.json"
        state = tu.load_telegram_state(str(state_path))
        assert isinstance(state, dict)
        assert "atr_regime" in state
        assert "drawdown_state" in state
        assert "router_quality" in state
        assert "aum_total" in state
        assert "last_4h_close_ts" in state
        assert "last_sent" in state

    def test_save_and_load_telegram_state(self, tmp_path):
        """save_telegram_state persists atomically and load reads back."""
        state_path = tmp_path / "telegram_state.json"
        state = {
            "atr_regime": "high",
            "drawdown_state": "none",
            "router_quality": "good",
            "aum_total": 12345.67,
            "last_4h_close_ts": 1764177600,
            "last_sent": {"close_4h": {"bar_ts": 1764177600, "ts": 1764177601}},
        }
        assert tu.save_telegram_state(state, str(state_path))
        assert state_path.exists()

        loaded = tu.load_telegram_state(str(state_path))
        assert loaded["atr_regime"] == "high"
        assert loaded["last_4h_close_ts"] == 1764177600
        assert loaded["aum_total"] == 12345.67

    def test_atomic_write_creates_no_tmp(self, tmp_path):
        """After save, there should be no .tmp file left behind."""
        state_path = tmp_path / "telegram_state.json"
        tu.save_telegram_state({"atr_regime": "low", "last_4h_close_ts": 0, "last_sent": {}}, str(state_path))
        assert state_path.exists()
        assert not state_path.with_suffix(".tmp").exists()


class TestSendTelegram4hOnlyMode:
    """Test send_telegram filtering in EXEC_TELEGRAM_4H_ONLY mode."""

    def test_4h_only_mode_blocks_non_json(self, monkeypatch):
        """Non-JSON messages are blocked in 4h-only mode."""
        monkeypatch.setenv("TELEGRAM_ENABLED", "1")
        monkeypatch.setenv("BOT_TOKEN", "test_token")
        monkeypatch.setenv("CHAT_ID", "test_chat")
        monkeypatch.setenv("EXEC_TELEGRAM_4H_ONLY", "1")

        with mock.patch("requests.post") as mock_post:
            result = tu.send_telegram("Hello, this is a plain text message")
            assert result is False
            mock_post.assert_not_called()

    def test_4h_only_mode_blocks_json_without_required_keys(self, monkeypatch):
        """JSON without atr_regime or last_4h_close_ts is blocked."""
        monkeypatch.setenv("TELEGRAM_ENABLED", "1")
        monkeypatch.setenv("BOT_TOKEN", "test_token")
        monkeypatch.setenv("CHAT_ID", "test_chat")
        monkeypatch.setenv("EXEC_TELEGRAM_4H_ONLY", "1")

        with mock.patch("requests.post") as mock_post:
            # JSON without required keys
            result = tu.send_telegram('{"some_key": "value", "another": 123}')
            assert result is False
            mock_post.assert_not_called()

    def test_4h_only_mode_allows_proper_payload(self, monkeypatch):
        """JSON with atr_regime and last_4h_close_ts is allowed."""
        monkeypatch.setenv("TELEGRAM_ENABLED", "1")
        monkeypatch.setenv("BOT_TOKEN", "test_token")
        monkeypatch.setenv("CHAT_ID", "test_chat")
        monkeypatch.setenv("EXEC_TELEGRAM_4H_ONLY", "1")
        # Clear rate limit state
        tu._send_timestamps.clear()
        tu._recent_msgs.clear()

        mock_response = mock.Mock()
        mock_response.ok = True
        mock_response.status_code = 200

        with mock.patch("requests.post", return_value=mock_response) as mock_post:
            payload = {
                "atr_regime": "low",
                "drawdown_state": "none",
                "router_quality": "good",
                "aum_total": 11173.87,
                "last_4h_close_ts": 1764177600,
            }
            result = tu.send_telegram(json.dumps(payload))
            assert result is True
            mock_post.assert_called_once()

    def test_4h_only_mode_allows_with_only_atr_regime(self, monkeypatch):
        """JSON with only atr_regime key is allowed."""
        monkeypatch.setenv("TELEGRAM_ENABLED", "1")
        monkeypatch.setenv("BOT_TOKEN", "test_token")
        monkeypatch.setenv("CHAT_ID", "test_chat")
        monkeypatch.setenv("EXEC_TELEGRAM_4H_ONLY", "1")
        tu._send_timestamps.clear()
        tu._recent_msgs.clear()

        mock_response = mock.Mock()
        mock_response.ok = True

        with mock.patch("requests.post", return_value=mock_response) as mock_post:
            result = tu.send_telegram('{"atr_regime": "high"}')
            assert result is True
            mock_post.assert_called_once()


class TestRateLimiting:
    """Test rate limiting behavior."""

    def test_rate_cap_zero_blocks_all(self, monkeypatch):
        """EXEC_TELEGRAM_MAX_PER_MIN=0 blocks all sends immediately."""
        monkeypatch.setenv("TELEGRAM_ENABLED", "1")
        monkeypatch.setenv("BOT_TOKEN", "test_token")
        monkeypatch.setenv("CHAT_ID", "test_chat")
        monkeypatch.setenv("EXEC_TELEGRAM_MAX_PER_MIN", "0")

        with mock.patch("requests.post") as mock_post:
            result = tu.send_telegram("Any message")
            assert result is False
            mock_post.assert_not_called()

    def test_identical_message_suppressed(self, monkeypatch):
        """Identical messages within MIN_IDENTICAL_S are suppressed."""
        monkeypatch.setenv("TELEGRAM_ENABLED", "1")
        monkeypatch.setenv("BOT_TOKEN", "test_token")
        monkeypatch.setenv("CHAT_ID", "test_chat")
        monkeypatch.setenv("EXEC_TELEGRAM_4H_ONLY", "0")
        monkeypatch.setenv("EXEC_TELEGRAM_MAX_PER_MIN", "10")
        tu._send_timestamps.clear()
        tu._recent_msgs.clear()

        mock_response = mock.Mock()
        mock_response.ok = True

        with mock.patch("requests.post", return_value=mock_response) as mock_post:
            # First send succeeds
            result1 = tu.send_telegram("Test message")
            assert result1 is True
            assert mock_post.call_count == 1

            # Second identical send is suppressed
            result2 = tu.send_telegram("Test message")
            assert result2 is False
            assert mock_post.call_count == 1  # No additional call


class TestTelegramAlertsV7:
    """Test telegram_alerts_v7 module behavior."""

    def test_load_state_creates_default(self, tmp_path, monkeypatch):
        """load_state creates default state file if missing."""
        state_path = tmp_path / "telegram_state.json"
        monkeypatch.setattr(ta7, "STATE_PATH", state_path)

        state = ta7.load_state()
        assert isinstance(state, dict)
        assert state.get("last_4h_close_ts") == 0
        assert "last_sent" in state

    def test_run_alerts_only_4h_close(self, tmp_path, monkeypatch):
        """run_alerts only sends 4h-close alert (low-noise policy)."""
        sent = []

        def fake_send(message: str, silent: bool = False):
            sent.append({"msg": message, "silent": bool(silent), "ts": time.time()})
            return True

        monkeypatch.setattr(ta7.telegram_utils, "send_telegram", fake_send)
        monkeypatch.setattr(
            ta7,
            "_load_config",
            lambda: {
                "enabled": True,
                "bot_token_env": "TELEGRAM_BOT_TOKEN",
                "chat_id_env": "TELEGRAM_CHAT_ID",
                "min_interval_seconds": 1,
                "alerts": {
                    "atr_regime": {"enabled": True, "min_interval_seconds": 1},
                    "dd_state": {"enabled": True, "min_interval_seconds": 1},
                    "risk_mode": {"enabled": True, "min_interval_seconds": 1},
                    "close_4h": {"enabled": True, "min_interval_seconds": 1},
                },
            },
        )
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "x")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "y")
        monkeypatch.setattr(ta7, "STATE_PATH", tmp_path / "telegram_state.json")

        # Calculate a 4h bar timestamp
        now_ts = time.time()
        bar_ts = int(now_ts // (4 * 3600)) * (4 * 3600)

        # Run with nav and kpis snapshot
        ta7.run_alerts({
            "now_ts": now_ts,
            "kpis_snapshot": {"atr_regime": "low", "dd_state": "none"},
            "nav_snapshot": {"nav": 11173.87, "ts": now_ts},
        })

        # Should have sent a 4h JSON payload
        assert len(sent) == 1
        payload = json.loads(sent[0]["msg"])
        assert "atr_regime" in payload
        assert "last_4h_close_ts" in payload
        assert payload["atr_regime"] == "low"
        assert payload["last_4h_close_ts"] == bar_ts

    def test_duplicate_4h_bar_suppressed(self, tmp_path, monkeypatch):
        """Duplicate 4h bar with same last_4h_close_ts is suppressed."""
        sent = []

        def fake_send(message: str, silent: bool = False):
            sent.append({"msg": message, "silent": bool(silent)})
            return True

        monkeypatch.setattr(ta7.telegram_utils, "send_telegram", fake_send)
        monkeypatch.setattr(
            ta7,
            "_load_config",
            lambda: {
                "enabled": True,
                "bot_token_env": "TELEGRAM_BOT_TOKEN",
                "chat_id_env": "TELEGRAM_CHAT_ID",
                "min_interval_seconds": 1,
                "alerts": {"close_4h": {"enabled": True, "min_interval_seconds": 1}},
            },
        )
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "x")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "y")
        monkeypatch.setattr(ta7, "STATE_PATH", tmp_path / "telegram_state.json")

        now_ts = time.time()
        bar_ts = int(now_ts // (4 * 3600)) * (4 * 3600)

        # First run should send
        ta7.run_alerts({
            "now_ts": now_ts,
            "kpis_snapshot": {"atr_regime": "low"},
            "nav_snapshot": {"nav": 10000, "ts": now_ts},
        })
        assert len(sent) == 1

        # Second run within same 4h bar should be suppressed
        sent.clear()
        ta7.run_alerts({
            "now_ts": now_ts + 60,  # 1 minute later, same 4h bar
            "kpis_snapshot": {"atr_regime": "low"},
            "nav_snapshot": {"nav": 10000, "ts": now_ts + 60},
        })
        assert len(sent) == 0

    def test_state_file_updated_after_send(self, tmp_path, monkeypatch):
        """State file is updated with canonical keys after successful send."""
        sent = []

        def fake_send(message: str, silent: bool = False):
            sent.append(message)
            return True

        monkeypatch.setattr(ta7.telegram_utils, "send_telegram", fake_send)
        monkeypatch.setattr(
            ta7,
            "_load_config",
            lambda: {
                "enabled": True,
                "bot_token_env": "TELEGRAM_BOT_TOKEN",
                "chat_id_env": "TELEGRAM_CHAT_ID",
                "min_interval_seconds": 1,
                "alerts": {"close_4h": {"enabled": True, "min_interval_seconds": 1}},
            },
        )
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "x")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "y")

        state_path = tmp_path / "telegram_state.json"
        monkeypatch.setattr(ta7, "STATE_PATH", state_path)

        now_ts = time.time()
        bar_ts = int(now_ts // (4 * 3600)) * (4 * 3600)

        ta7.run_alerts({
            "now_ts": now_ts,
            "kpis_snapshot": {"atr_regime": "high", "dd_state": "mild", "router_quality": "good"},
            "nav_snapshot": {"nav": 12345.67, "ts": now_ts},
        })

        assert state_path.exists()
        state = json.loads(state_path.read_text())
        assert state["atr_regime"] == "high"
        assert state["drawdown_state"] == "mild"
        assert state["router_quality"] == "good"
        assert state["aum_total"] == 12345.67
        assert state["last_4h_close_ts"] == bar_ts

    def test_4h_payload_has_exact_keys(self, tmp_path, monkeypatch):
        """4h payload contains exactly the required keys."""
        sent = []

        def fake_send(message: str, silent: bool = False):
            sent.append(message)
            return True

        monkeypatch.setattr(ta7.telegram_utils, "send_telegram", fake_send)
        monkeypatch.setattr(
            ta7,
            "_load_config",
            lambda: {
                "enabled": True,
                "bot_token_env": "TELEGRAM_BOT_TOKEN",
                "chat_id_env": "TELEGRAM_CHAT_ID",
                "min_interval_seconds": 1,
                "alerts": {"close_4h": {"enabled": True}},
            },
        )
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "x")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "y")
        monkeypatch.setattr(ta7, "STATE_PATH", tmp_path / "telegram_state.json")

        ta7.run_alerts({
            "now_ts": time.time(),
            "kpis_snapshot": {"atr_regime": "low"},
            "nav_snapshot": {"nav": 11173.87},
        })

        assert len(sent) == 1
        payload = json.loads(sent[0])

        # Must have exactly these keys
        required_keys = {"atr_regime", "drawdown_state", "router_quality", "aum_total", "last_4h_close_ts"}
        assert set(payload.keys()) == required_keys

        # Type checks
        assert isinstance(payload["atr_regime"], str)
        assert payload["drawdown_state"] is None or isinstance(payload["drawdown_state"], str)
        assert payload["router_quality"] is None or isinstance(payload["router_quality"], str)
        assert payload["aum_total"] is None or isinstance(payload["aum_total"], (int, float))
        assert isinstance(payload["last_4h_close_ts"], int)


class TestAlertJsonlLogging:
    """Test JSONL audit logging."""

    def test_write_alert_jsonl(self, tmp_path, monkeypatch):
        """write_alert_jsonl appends to JSONL file."""
        jsonl_path = tmp_path / "alerts" / "alerts_v7.jsonl"

        # Write two alerts
        tu.write_alert_jsonl({"type": "4h", "message": "test1"}, str(jsonl_path))
        tu.write_alert_jsonl({"type": "4h", "message": "test2"}, str(jsonl_path))

        assert jsonl_path.exists()
        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) == 2

        alert1 = json.loads(lines[0])
        assert alert1["type"] == "4h"
        assert alert1["message"] == "test1"
        assert "ts" in alert1


class TestIntegrationScenario:
    """Integration-style tests for the full alert flow."""

    def test_full_4h_close_flow(self, tmp_path, monkeypatch):
        """Full flow: 4h bar closes, alert sent, state updated, duplicate suppressed."""
        posts = []

        def mock_post(*args, **kwargs):
            posts.append(kwargs.get("json", {}))
            resp = mock.Mock()
            resp.ok = True
            resp.status_code = 200
            return resp

        monkeypatch.setenv("TELEGRAM_ENABLED", "1")
        monkeypatch.setenv("BOT_TOKEN", "test_token")
        monkeypatch.setenv("CHAT_ID", "test_chat")
        monkeypatch.setenv("EXEC_TELEGRAM_4H_ONLY", "1")
        monkeypatch.setenv("EXEC_TELEGRAM_MAX_PER_MIN", "10")
        tu._send_timestamps.clear()
        tu._recent_msgs.clear()

        monkeypatch.setattr(
            ta7,
            "_load_config",
            lambda: {
                "enabled": True,
                "bot_token_env": "BOT_TOKEN",
                "chat_id_env": "CHAT_ID",
                "alerts": {"close_4h": {"enabled": True}},
            },
        )

        state_path = tmp_path / "telegram_state.json"
        monkeypatch.setattr(ta7, "STATE_PATH", state_path)

        with mock.patch("requests.post", side_effect=mock_post):
            now_ts = time.time()

            # First 4h bar
            ta7.run_alerts({
                "now_ts": now_ts,
                "kpis_snapshot": {"atr_regime": "low", "dd_state": "none"},
                "nav_snapshot": {"nav": 11173.87, "ts": now_ts},
            })

            assert len(posts) == 1
            msg_text = posts[0].get("text", "")
            # Extract JSON from message (format: "timestamp\n{json}")
            json_part = msg_text.split("\n", 1)[1] if "\n" in msg_text else msg_text
            payload = json.loads(json_part)
            assert payload["atr_regime"] == "low"

            # Verify state file
            state = json.loads(state_path.read_text())
            assert state["atr_regime"] == "low"

            # Same bar should be suppressed (even with different message to avoid identical suppression)
            posts.clear()
            tu._recent_msgs.clear()  # Clear to avoid identical suppression
            ta7.run_alerts({
                "now_ts": now_ts + 10,
                "kpis_snapshot": {"atr_regime": "high"},  # Changed atr_regime
                "nav_snapshot": {"nav": 11200.00, "ts": now_ts + 10},
            })
            assert len(posts) == 0  # Suppressed because same 4h bar
