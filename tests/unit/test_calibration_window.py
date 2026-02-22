"""
Tests for Calibration Window — Episode-Capped Trading Gate.

Covers:
  - Config loading (enabled/disabled/missing/dual-key)
  - Episode counting since start_ts
  - KILL_SWITCH activation on cap
  - Idempotent halt (no duplicate doctrine logs)
  - Drawdown kill
  - Sizing override
  - Inactive when section missing or ACK not set
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest

from execution.calibration_window import (
    CALIBRATION_ACK_ENV,
    _count_episodes_since,
    _load_calibration_config,
    check_calibration_window,
    get_calibration_sizing_override,
    log_calibration_boot_status,
)


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clean_env():
    """Ensure CALIBRATION_WINDOW_ACK and KILL_SWITCH are clean between tests."""
    os.environ.pop(CALIBRATION_ACK_ENV, None)
    os.environ.pop("KILL_SWITCH", None)
    yield
    os.environ.pop(CALIBRATION_ACK_ENV, None)
    os.environ.pop("KILL_SWITCH", None)


# ── Fixtures ────────────────────────────────────────────────────────────

def _make_runtime_yaml(tmp_path: Path, cw_section: Dict[str, Any]) -> Path:
    """Write a runtime.yaml with given calibration_window section."""
    import yaml
    cfg = {"calibration_window": cw_section}
    p = tmp_path / "runtime.yaml"
    p.write_text(yaml.dump(cfg))
    return p


def _make_episode_ledger(tmp_path: Path, episodes: list) -> Path:
    """Write a minimal episode_ledger.json."""
    p = tmp_path / "episode_ledger.json"
    p.write_text(json.dumps({"episodes": episodes}))
    return p


def _episode(exit_ts: str, symbol: str = "BTCUSDT") -> Dict[str, Any]:
    return {"symbol": symbol, "exit_ts": exit_ts, "realized_pnl_usdt": 1.0}


# ── Config Loading ──────────────────────────────────────────────────────

class TestConfigLoading:

    def test_missing_section_returns_none(self, tmp_path: Path) -> None:
        import yaml
        p = tmp_path / "runtime.yaml"
        p.write_text(yaml.dump({"runtime": {"env": "test"}}))
        with patch("execution.calibration_window.RUNTIME_YAML", p):
            assert _load_calibration_config() is None

    def test_disabled_returns_none(self, tmp_path: Path) -> None:
        p = _make_runtime_yaml(tmp_path, {"enabled": False, "episode_cap": 30})
        with patch("execution.calibration_window.RUNTIME_YAML", p):
            assert _load_calibration_config() is None

    def test_enabled_returns_config(self, tmp_path: Path) -> None:
        p = _make_runtime_yaml(tmp_path, {
            "enabled": True,
            "episode_cap": 30,
            "start_ts": "2026-02-22T00:00:00Z",
        })
        os.environ[CALIBRATION_ACK_ENV] = "1"
        with patch("execution.calibration_window.RUNTIME_YAML", p):
            cfg = _load_calibration_config()
            assert cfg is not None
            assert cfg["episode_cap"] == 30

    def test_enabled_without_ack_returns_none(self, tmp_path: Path) -> None:
        """Config enabled=true but no ACK env → inactive (dual-key)."""
        p = _make_runtime_yaml(tmp_path, {
            "enabled": True,
            "episode_cap": 30,
            "start_ts": "2026-02-22T00:00:00Z",
        })
        # ACK intentionally NOT set
        with patch("execution.calibration_window.RUNTIME_YAML", p):
            assert _load_calibration_config() is None

    def test_enabled_with_ack_false_returns_none(self, tmp_path: Path) -> None:
        """ACK=0 explicitly → inactive."""
        p = _make_runtime_yaml(tmp_path, {
            "enabled": True,
            "episode_cap": 30,
        })
        os.environ[CALIBRATION_ACK_ENV] = "0"
        with patch("execution.calibration_window.RUNTIME_YAML", p):
            assert _load_calibration_config() is None

    def test_missing_file_returns_none(self) -> None:
        with patch("execution.calibration_window.RUNTIME_YAML", Path("/nonexistent.yaml")):
            assert _load_calibration_config() is None


# ── Episode Counting ────────────────────────────────────────────────────

class TestEpisodeCounting:

    def test_counts_episodes_since(self, tmp_path: Path) -> None:
        p = _make_episode_ledger(tmp_path, [
            _episode("2026-02-20T10:00:00Z"),
            _episode("2026-02-21T10:00:00Z"),
            _episode("2026-02-22T10:00:00Z"),
            _episode("2026-02-23T10:00:00Z"),
        ])
        with patch("execution.calibration_window.EPISODE_LEDGER_PATH", p):
            assert _count_episodes_since("2026-02-21T00:00:00Z") == 3
            assert _count_episodes_since("2026-02-23T00:00:00Z") == 1
            assert _count_episodes_since("2026-02-24T00:00:00Z") == 0

    def test_mixed_format_z_vs_offset(self, tmp_path: Path) -> None:
        """Ledger uses +00:00 with fractional seconds; start_ts uses Z.
        Datetime comparison must handle both correctly."""
        p = _make_episode_ledger(tmp_path, [
            _episode("2026-02-20T10:00:00.123456+00:00"),
            _episode("2026-02-21T10:00:00.789012+00:00"),
            _episode("2026-02-22T10:00:00.345678+00:00"),
        ])
        with patch("execution.calibration_window.EPISODE_LEDGER_PATH", p):
            # start_ts in Z format, ledger in +00:00 format
            assert _count_episodes_since("2026-02-21T00:00:00Z") == 2
            # start_ts in +00:00 format
            assert _count_episodes_since("2026-02-22T00:00:00+00:00") == 1

    def test_start_ts_with_fractional_seconds(self, tmp_path: Path) -> None:
        """start_ts with fractional seconds should still match."""
        p = _make_episode_ledger(tmp_path, [
            _episode("2026-02-22T10:00:00.500000+00:00"),
            _episode("2026-02-22T10:00:00.100000+00:00"),
        ])
        with patch("execution.calibration_window.EPISODE_LEDGER_PATH", p):
            assert _count_episodes_since("2026-02-22T10:00:00.300000+00:00") == 1

    def test_empty_ledger(self, tmp_path: Path) -> None:
        p = _make_episode_ledger(tmp_path, [])
        with patch("execution.calibration_window.EPISODE_LEDGER_PATH", p):
            assert _count_episodes_since("2026-01-01T00:00:00Z") == 0

    def test_missing_ledger(self) -> None:
        with patch("execution.calibration_window.EPISODE_LEDGER_PATH", Path("/nonexistent.json")):
            assert _count_episodes_since("2026-01-01T00:00:00Z") == 0


# ── Calibration Window Check ───────────────────────────────────────────

class TestCalibrationWindowCheck:

    def test_inactive_when_no_config(self, tmp_path: Path) -> None:
        import yaml
        p = tmp_path / "runtime.yaml"
        p.write_text(yaml.dump({}))
        with patch("execution.calibration_window.RUNTIME_YAML", p):
            status = check_calibration_window()
            assert status["active"] is False

    def test_active_below_cap(self, tmp_path: Path) -> None:
        import execution.calibration_window as cw
        cw._kill_switch_fired = False
        os.environ[CALIBRATION_ACK_ENV] = "1"

        runtime = _make_runtime_yaml(tmp_path, {
            "enabled": True,
            "episode_cap": 30,
            "start_ts": "2026-02-22T00:00:00Z",
            "drawdown_kill_pct": 0.05,
        })
        ledger = _make_episode_ledger(tmp_path, [
            _episode("2026-02-22T01:00:00Z"),
            _episode("2026-02-22T02:00:00Z"),
        ])

        with patch("execution.calibration_window.RUNTIME_YAML", runtime), \
             patch("execution.calibration_window.EPISODE_LEDGER_PATH", ledger), \
             patch("execution.calibration_window._get_portfolio_dd_pct", return_value=0.01):
            # Clear any previous KILL_SWITCH
            os.environ.pop("KILL_SWITCH", None)
            status = check_calibration_window()
            assert status["active"] is True
            assert status["episodes_completed"] == 2
            assert status["episodes_remaining"] == 28
            assert status["halted"] is False
            assert status["dd_breached"] is False
            assert os.environ.get("KILL_SWITCH") is None

    def test_halt_at_cap(self, tmp_path: Path) -> None:
        import execution.calibration_window as cw
        cw._kill_switch_fired = False
        os.environ[CALIBRATION_ACK_ENV] = "1"

        episodes = [_episode(f"2026-02-22T{i:02d}:00:00Z") for i in range(5)]
        runtime = _make_runtime_yaml(tmp_path, {
            "enabled": True,
            "episode_cap": 5,
            "start_ts": "2026-02-22T00:00:00Z",
        })
        ledger = _make_episode_ledger(tmp_path, episodes)
        doctrine = tmp_path / "doctrine_events.jsonl"

        with patch("execution.calibration_window.RUNTIME_YAML", runtime), \
             patch("execution.calibration_window.EPISODE_LEDGER_PATH", ledger), \
             patch("execution.calibration_window.DOCTRINE_LOG", doctrine), \
             patch("execution.calibration_window._get_portfolio_dd_pct", return_value=0.0):
            os.environ.pop("KILL_SWITCH", None)
            status = check_calibration_window()
            assert status["halted"] is True
            assert status["episodes_completed"] == 5
            assert os.environ.get("KILL_SWITCH") == "1"
            # Doctrine event logged
            assert doctrine.exists()
            event = json.loads(doctrine.read_text().strip())
            assert event["event"] == "CALIBRATION_WINDOW_HALT"

        # Cleanup
        os.environ.pop("KILL_SWITCH", None)

    def test_idempotent_halt(self, tmp_path: Path) -> None:
        """Second call after cap does not duplicate doctrine log."""
        import execution.calibration_window as cw
        cw._kill_switch_fired = False
        os.environ[CALIBRATION_ACK_ENV] = "1"

        episodes = [_episode(f"2026-02-22T{i:02d}:00:00Z") for i in range(5)]
        runtime = _make_runtime_yaml(tmp_path, {
            "enabled": True,
            "episode_cap": 5,
            "start_ts": "2026-02-22T00:00:00Z",
        })
        ledger = _make_episode_ledger(tmp_path, episodes)
        doctrine = tmp_path / "doctrine_events.jsonl"

        with patch("execution.calibration_window.RUNTIME_YAML", runtime), \
             patch("execution.calibration_window.EPISODE_LEDGER_PATH", ledger), \
             patch("execution.calibration_window.DOCTRINE_LOG", doctrine), \
             patch("execution.calibration_window._get_portfolio_dd_pct", return_value=0.0):
            os.environ.pop("KILL_SWITCH", None)
            check_calibration_window()
            check_calibration_window()  # second call
            # Only one doctrine event
            lines = [l for l in doctrine.read_text().strip().split("\n") if l]
            assert len(lines) == 1

        os.environ.pop("KILL_SWITCH", None)

    def test_no_start_ts_inactive(self, tmp_path: Path) -> None:
        os.environ[CALIBRATION_ACK_ENV] = "1"
        runtime = _make_runtime_yaml(tmp_path, {
            "enabled": True,
            "episode_cap": 30,
            "start_ts": "",
        })
        with patch("execution.calibration_window.RUNTIME_YAML", runtime):
            status = check_calibration_window()
            assert status["active"] is False


# ── Drawdown Kill ───────────────────────────────────────────────────────

class TestDrawdownKill:

    def test_halt_on_drawdown_breach(self, tmp_path: Path) -> None:
        """KILL_SWITCH fires when DD >= drawdown_kill_pct, even below episode cap."""
        import execution.calibration_window as cw
        cw._kill_switch_fired = False
        os.environ[CALIBRATION_ACK_ENV] = "1"

        runtime = _make_runtime_yaml(tmp_path, {
            "enabled": True,
            "episode_cap": 30,
            "start_ts": "2026-02-22T00:00:00Z",
            "drawdown_kill_pct": 0.05,
        })
        ledger = _make_episode_ledger(tmp_path, [
            _episode("2026-02-22T01:00:00Z"),
        ])
        doctrine = tmp_path / "doctrine_events.jsonl"

        with patch("execution.calibration_window.RUNTIME_YAML", runtime), \
             patch("execution.calibration_window.EPISODE_LEDGER_PATH", ledger), \
             patch("execution.calibration_window.DOCTRINE_LOG", doctrine), \
             patch("execution.calibration_window._get_portfolio_dd_pct", return_value=0.06):
            os.environ.pop("KILL_SWITCH", None)
            status = check_calibration_window()
            assert status["halted"] is True
            assert status["dd_breached"] is True
            assert status["current_dd_pct"] == 0.06
            assert os.environ.get("KILL_SWITCH") == "1"
            # Doctrine event references drawdown
            event = json.loads(doctrine.read_text().strip())
            assert "drawdown" in event["action"]

        os.environ.pop("KILL_SWITCH", None)

    def test_no_halt_below_drawdown_threshold(self, tmp_path: Path) -> None:
        """DD below threshold does not trigger halt."""
        import execution.calibration_window as cw
        cw._kill_switch_fired = False
        os.environ[CALIBRATION_ACK_ENV] = "1"

        runtime = _make_runtime_yaml(tmp_path, {
            "enabled": True,
            "episode_cap": 30,
            "start_ts": "2026-02-22T00:00:00Z",
            "drawdown_kill_pct": 0.05,
        })
        ledger = _make_episode_ledger(tmp_path, [
            _episode("2026-02-22T01:00:00Z"),
        ])

        with patch("execution.calibration_window.RUNTIME_YAML", runtime), \
             patch("execution.calibration_window.EPISODE_LEDGER_PATH", ledger), \
             patch("execution.calibration_window._get_portfolio_dd_pct", return_value=0.03):
            os.environ.pop("KILL_SWITCH", None)
            status = check_calibration_window()
            assert status["halted"] is False
            assert status["dd_breached"] is False
            assert os.environ.get("KILL_SWITCH") is None

    def test_drawdown_kill_pct_zero_disabled(self, tmp_path: Path) -> None:
        """drawdown_kill_pct=0 disables the DD check (only episode cap applies)."""
        import execution.calibration_window as cw
        cw._kill_switch_fired = False
        os.environ[CALIBRATION_ACK_ENV] = "1"

        runtime = _make_runtime_yaml(tmp_path, {
            "enabled": True,
            "episode_cap": 30,
            "start_ts": "2026-02-22T00:00:00Z",
            "drawdown_kill_pct": 0.0,
        })
        ledger = _make_episode_ledger(tmp_path, [
            _episode("2026-02-22T01:00:00Z"),
        ])

        with patch("execution.calibration_window.RUNTIME_YAML", runtime), \
             patch("execution.calibration_window.EPISODE_LEDGER_PATH", ledger), \
             patch("execution.calibration_window._get_portfolio_dd_pct", return_value=0.10):
            os.environ.pop("KILL_SWITCH", None)
            status = check_calibration_window()
            assert status["halted"] is False
            assert status["dd_breached"] is False


# ── Sizing Override ─────────────────────────────────────────────────────

class TestSizingOverride:

    def test_returns_override_when_active(self, tmp_path: Path) -> None:
        os.environ[CALIBRATION_ACK_ENV] = "1"
        p = _make_runtime_yaml(tmp_path, {
            "enabled": True,
            "episode_cap": 30,
            "start_ts": "2026-02-22T00:00:00Z",
            "per_trade_nav_pct": 0.005,
        })
        with patch("execution.calibration_window.RUNTIME_YAML", p):
            assert get_calibration_sizing_override() == 0.005

    def test_returns_none_when_inactive(self, tmp_path: Path) -> None:
        import yaml
        p = tmp_path / "runtime.yaml"
        p.write_text(yaml.dump({}))
        with patch("execution.calibration_window.RUNTIME_YAML", p):
            assert get_calibration_sizing_override() is None

    def test_returns_none_when_no_override_key(self, tmp_path: Path) -> None:
        os.environ[CALIBRATION_ACK_ENV] = "1"
        p = _make_runtime_yaml(tmp_path, {
            "enabled": True,
            "episode_cap": 30,
            "start_ts": "2026-02-22T00:00:00Z",
        })
        with patch("execution.calibration_window.RUNTIME_YAML", p):
            assert get_calibration_sizing_override() is None


# ── Boot Status Logging ──────────────────────────────────────────────────

class TestBootStatus:

    def test_boot_inactive_no_config(self, tmp_path: Path, caplog) -> None:
        import yaml
        p = tmp_path / "runtime.yaml"
        p.write_text(yaml.dump({}))
        with patch("execution.calibration_window.RUNTIME_YAML", p):
            import logging
            with caplog.at_level(logging.INFO):
                log_calibration_boot_status()
            assert "INACTIVE" in caplog.text

    def test_boot_active(self, tmp_path: Path, caplog) -> None:
        os.environ[CALIBRATION_ACK_ENV] = "1"
        p = _make_runtime_yaml(tmp_path, {
            "enabled": True,
            "episode_cap": 30,
            "start_ts": "2026-02-22T00:00:00Z",
            "drawdown_kill_pct": 0.05,
            "per_trade_nav_pct": 0.005,
        })
        with patch("execution.calibration_window.RUNTIME_YAML", p):
            import logging
            with caplog.at_level(logging.INFO):
                log_calibration_boot_status()
            assert "ACTIVE" in caplog.text
            assert "cap=30" in caplog.text
            assert "sizing=0.0050" in caplog.text

    def test_boot_dual_key_missing(self, tmp_path: Path, caplog) -> None:
        """Config enabled but ACK missing → warns about dual-key."""
        p = _make_runtime_yaml(tmp_path, {
            "enabled": True,
            "episode_cap": 30,
            "start_ts": "2026-02-22T00:00:00Z",
        })
        with patch("execution.calibration_window.RUNTIME_YAML", p):
            import logging
            with caplog.at_level(logging.WARNING):
                log_calibration_boot_status()
            assert "dual-key" in caplog.text.lower()
