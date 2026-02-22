"""Tests for ops/daily_summary.py — generate_daily_summary()."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from ops.daily_summary import generate_daily_summary


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path, monkeypatch):
    """Create minimal state files in tmp_path so the summary doesn't read prod."""
    import ops.daily_summary as mod

    # NAV state
    nav_state = tmp_path / "nav_state.json"
    nav_state.write_text(json.dumps({
        "nav_usd": 10000.00,
        "nav": 10000.00,
        "nav_mode": "live_wallet",
    }))
    monkeypatch.setattr(mod, "_NAV_STATE", nav_state)

    # Positions
    pos_state = tmp_path / "positions_state.json"
    pos_state.write_text(json.dumps({
        "positions": [
            {"symbol": "BTCUSDT", "unrealizedProfit": 42.50},
            {"symbol": "ETHUSDT", "unrealizedProfit": -10.00},
        ],
    }))
    monkeypatch.setattr(mod, "_POSITIONS_STATE", pos_state)

    # Episode ledger
    ledger = tmp_path / "episode_ledger.json"
    ledger.write_text(json.dumps({
        "episode_count": 50,
        "stats": {
            "total_net_pnl": -123.45,
            "win_rate": 40.0,
            "winners": 20,
            "losers": 30,
        },
        "episodes": [],
    }))
    monkeypatch.setattr(mod, "_EPISODE_LEDGER", ledger)

    # NAV log (24h history)
    now_t = time.time()
    nav_log = tmp_path / "nav_log.json"
    entries = [
        {"nav": 10050.0, "t": now_t - 86400, "unrealized_pnl": 0},
        {"nav": 10030.0, "t": now_t - 43200, "unrealized_pnl": 0},
        {"nav": 10000.0, "t": now_t, "unrealized_pnl": 0},
    ]
    nav_log.write_text(json.dumps(entries))
    monkeypatch.setattr(mod, "_NAV_LOG", nav_log)

    # Risk snapshot
    risk = tmp_path / "risk_snapshot.json"
    risk.write_text(json.dumps({"max_drawdown_pct": 2.35}))
    monkeypatch.setattr(mod, "_RISK_SNAPSHOT", risk)

    # Sentinel X
    sentinel = tmp_path / "sentinel_x.json"
    sentinel.write_text(json.dumps({
        "regime": "TREND_UP",
        "confidence": 0.82,
    }))
    monkeypatch.setattr(mod, "_SENTINEL_X", sentinel)

    # Runtime YAML (calibration window active)
    runtime = tmp_path / "runtime.yaml"
    runtime.write_text(
        "calibration_window:\n"
        "  enabled: true\n"
        "  episode_cap: 30\n"
        "  start_ts: '2026-01-01T00:00:00Z'\n"
        "  drawdown_kill_pct: 0.05\n"
    )
    monkeypatch.setattr(mod, "_RUNTIME_YAML", runtime)

    # Binary lab state (disabled)
    binary = tmp_path / "binary_lab_state.json"
    binary.write_text(json.dumps({"enabled": False}))
    monkeypatch.setattr(mod, "_BINARY_LAB", binary)


class TestGenerateDailySummary:

    def test_returns_string(self):
        result = generate_daily_summary()
        assert isinstance(result, str)
        assert len(result) > 100

    def test_contains_nav(self):
        result = generate_daily_summary()
        assert "$10,000.00" in result
        assert "live_wallet" in result

    def test_contains_24h_pnl(self):
        result = generate_daily_summary()
        assert "24h PnL" in result
        # NAV went from ~10050 to 10000 = -50
        assert "$-50.00" in result

    def test_contains_positions(self):
        result = generate_daily_summary()
        assert "Open Positions:   2" in result
        assert "Unrealized PnL" in result

    def test_contains_episode_stats(self):
        result = generate_daily_summary()
        assert "Episodes:         50" in result
        assert "W:20 / L:30" in result
        assert "Win Rate:         40.0%" in result
        assert "$-123.45" in result

    def test_contains_drawdown(self):
        result = generate_daily_summary()
        assert "2.35%" in result

    def test_contains_regime(self):
        result = generate_daily_summary()
        assert "TREND_UP" in result
        assert "82%" in result

    def test_contains_calibration(self):
        result = generate_daily_summary()
        assert "ACTIVE" in result
        assert "0/30" in result
        assert "DD Kill" in result
        assert "5.0%" in result

    def test_binary_sleeve_disabled(self):
        result = generate_daily_summary()
        assert "Binary Sleeve:    DISABLED" in result

    def test_custom_timestamp(self):
        ts = datetime(2026, 3, 15, 12, 0, tzinfo=timezone.utc)
        result = generate_daily_summary(now=ts)
        assert "2026-03-15 12:00 UTC" in result

    def test_missing_state_files_graceful(self, tmp_path, monkeypatch):
        """Summary should not crash if state files are missing."""
        import ops.daily_summary as mod
        monkeypatch.setattr(mod, "_NAV_STATE", tmp_path / "nonexistent.json")
        monkeypatch.setattr(mod, "_POSITIONS_STATE", tmp_path / "nonexistent2.json")
        monkeypatch.setattr(mod, "_EPISODE_LEDGER", tmp_path / "nonexistent3.json")
        monkeypatch.setattr(mod, "_NAV_LOG", tmp_path / "nonexistent4.json")
        monkeypatch.setattr(mod, "_RISK_SNAPSHOT", tmp_path / "nonexistent5.json")
        monkeypatch.setattr(mod, "_SENTINEL_X", tmp_path / "nonexistent6.json")
        monkeypatch.setattr(mod, "_RUNTIME_YAML", tmp_path / "nonexistent7.yaml")
        monkeypatch.setattr(mod, "_BINARY_LAB", tmp_path / "nonexistent8.json")

        result = generate_daily_summary()
        assert isinstance(result, str)
        assert "Portfolio NAV:    $0.00" in result
        assert "INACTIVE" in result

    def test_no_positions_hides_unrealized(self, tmp_path, monkeypatch):
        """When 0 positions, Unrealized PnL line should not appear."""
        import ops.daily_summary as mod
        pos = tmp_path / "pos_empty.json"
        pos.write_text(json.dumps({"positions": []}))
        monkeypatch.setattr(mod, "_POSITIONS_STATE", pos)

        result = generate_daily_summary()
        assert "Open Positions:   0" in result
        assert "Unrealized PnL" not in result

    def test_binary_enabled(self, tmp_path, monkeypatch):
        import ops.daily_summary as mod
        binary = tmp_path / "binary_on.json"
        binary.write_text(json.dumps({"enabled": True, "pnl": 55.0}))
        monkeypatch.setattr(mod, "_BINARY_LAB", binary)

        result = generate_daily_summary()
        assert "Binary Sleeve:    ENABLED" in result
        assert "$55.00" in result

    def test_calibration_inactive(self, tmp_path, monkeypatch):
        import ops.daily_summary as mod
        runtime = tmp_path / "runtime_no_cal.yaml"
        runtime.write_text("calibration_window:\n  enabled: false\n")
        monkeypatch.setattr(mod, "_RUNTIME_YAML", runtime)

        result = generate_daily_summary()
        assert "Calibration:      INACTIVE" in result
