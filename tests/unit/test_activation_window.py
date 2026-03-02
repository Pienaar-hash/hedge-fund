"""
Tests for Activation Window v8.0 — Full-Stack System Certification Protocol.

Covers:
    - Config loading (dual-key activation, missing/disabled)
    - Timestamp parsing
    - Window timing (elapsed, remaining, expired)
    - Structural integrity (manifest hash, config hash drift)
    - Drawdown kill
    - Binary Lab freeze check
    - DLE mismatch counting
    - KILL_SWITCH firing (idempotent)
    - State file emission
    - Boot status logging
    - Sizing override
    - Stack health collection
    - activation_verify.py 7-gate verification
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict
from unittest import mock

import pytest
import yaml

import execution.activation_window as aw
from execution.activation_window import (
    _file_hash,
    _parse_iso_ts,
    _reset_globals,
    check_activation_window,
    collect_stack_health,
    get_activation_sizing_override,
    log_activation_boot_status,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_env():
    """Strip activation and kill switch env vars between tests."""
    for key in ("ACTIVATION_WINDOW_ACK", "KILL_SWITCH"):
        os.environ.pop(key, None)
    _reset_globals()
    yield
    for key in ("ACTIVATION_WINDOW_ACK", "KILL_SWITCH"):
        os.environ.pop(key, None)
    _reset_globals()


def _make_runtime_yaml(tmp_path: Path, aw_cfg: Dict[str, Any]) -> Path:
    """Write a runtime.yaml with activation_window section."""
    p = tmp_path / "runtime.yaml"
    with p.open("w") as f:
        yaml.dump({"activation_window": aw_cfg}, f)
    return p


def _make_manifest(tmp_path: Path, content: str = '{"version": "v8"}') -> Path:
    p = tmp_path / "v7_manifest.json"
    p.write_text(content)
    return p


def _make_episode_ledger(tmp_path: Path, episodes: list) -> Path:
    p = tmp_path / "episode_ledger.json"
    with p.open("w") as f:
        json.dump({"episodes": episodes}, f)
    return p


def _make_nav_state(tmp_path: Path, nav_usd: float = 9666.0) -> Path:
    p = tmp_path / "nav_state.json"
    with p.open("w") as f:
        json.dump({"nav_usd": nav_usd, "nav_age_s": 5}, f)
    return p


def _make_activation_state(tmp_path: Path, state: Dict[str, Any]) -> Path:
    p = tmp_path / "activation_window_state.json"
    with p.open("w") as f:
        json.dump(state, f)
    return p


def _aw_config(**overrides) -> Dict[str, Any]:
    defaults = {
        "enabled": True,
        "duration_days": 14,
        "start_ts": "2026-02-15T00:00:00Z",
        "drawdown_kill_pct": 0.05,
        "per_trade_nav_pct": 0.005,
    }
    defaults.update(overrides)
    return defaults


# ===========================================================================
# Tests: Timestamp parsing
# ===========================================================================

class TestTimestampParsing:
    def test_z_suffix(self):
        dt = _parse_iso_ts("2026-02-25T12:00:00Z")
        assert dt is not None
        assert dt.tzinfo is not None
        assert dt.year == 2026

    def test_utc_offset(self):
        dt = _parse_iso_ts("2026-02-25T12:00:00+00:00")
        assert dt is not None

    def test_fractional_seconds(self):
        dt = _parse_iso_ts("2026-02-25T12:00:00.123456+00:00")
        assert dt is not None

    def test_empty(self):
        assert _parse_iso_ts("") is None

    def test_invalid(self):
        assert _parse_iso_ts("not-a-date") is None


# ===========================================================================
# Tests: Config loading
# ===========================================================================

class TestConfigLoading:
    def test_missing_section(self, tmp_path):
        p = tmp_path / "runtime.yaml"
        with p.open("w") as f:
            yaml.dump({"other": True}, f)
        result = aw._load_activation_config(p)
        assert result is None

    def test_disabled(self, tmp_path):
        p = _make_runtime_yaml(tmp_path, {"enabled": False})
        os.environ["ACTIVATION_WINDOW_ACK"] = "1"
        result = aw._load_activation_config(p)
        assert result is None

    def test_enabled_with_ack(self, tmp_path):
        p = _make_runtime_yaml(tmp_path, _aw_config())
        os.environ["ACTIVATION_WINDOW_ACK"] = "1"
        result = aw._load_activation_config(p)
        assert result is not None
        assert result["enabled"] is True

    def test_enabled_without_ack(self, tmp_path):
        p = _make_runtime_yaml(tmp_path, _aw_config())
        # No ACK set
        result = aw._load_activation_config(p)
        assert result is None

    def test_ack_zero(self, tmp_path):
        p = _make_runtime_yaml(tmp_path, _aw_config())
        os.environ["ACTIVATION_WINDOW_ACK"] = "0"
        result = aw._load_activation_config(p)
        assert result is None


# ===========================================================================
# Tests: File hashing
# ===========================================================================

class TestFileHash:
    def test_deterministic(self, tmp_path):
        f = tmp_path / "test.json"
        f.write_text('{"hello": "world"}')
        h1 = _file_hash(f)
        h2 = _file_hash(f)
        assert h1 == h2
        assert len(h1) == 16  # truncated SHA-256

    def test_changes_on_edit(self, tmp_path):
        f = tmp_path / "test.json"
        f.write_text("v1")
        h1 = _file_hash(f)
        f.write_text("v2")
        h2 = _file_hash(f)
        assert h1 != h2

    def test_missing_file(self, tmp_path):
        assert _file_hash(tmp_path / "nope.json") is None


# ===========================================================================
# Tests: Window timing
# ===========================================================================

class TestWindowTiming:
    def test_window_active_mid_period(self, tmp_path):
        start = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        p = _make_runtime_yaml(tmp_path, _aw_config(start_ts=start))
        manifest = _make_manifest(tmp_path)
        state_path = tmp_path / "state.json"
        doctrine = tmp_path / "doctrine.jsonl"

        os.environ["ACTIVATION_WINDOW_ACK"] = "1"
        with mock.patch.object(aw, "NAV_STATE_PATH", tmp_path / "nav.json"):
            _make_nav_state(tmp_path)
            with mock.patch.object(aw, "BINARY_LAB_STATE_PATH", tmp_path / "bl.json"):
                with mock.patch.object(aw, "DLE_SHADOW_LOG", tmp_path / "dle.jsonl"):
                    with mock.patch.object(aw, "EPISODE_LEDGER_PATH", tmp_path / "ep.json"):
                        _make_episode_ledger(tmp_path, [])
                        with mock.patch("execution.activation_window._get_portfolio_dd_pct", return_value=0.01):
                            with mock.patch("execution.activation_window._count_risk_vetoes_since", return_value=5):
                                status = check_activation_window(
                                    runtime_yaml=p,
                                    manifest_path=manifest,
                                    state_path=state_path,
                                    doctrine_log=doctrine,
                                )

        assert status["active"] is True
        assert status["halted"] is False
        assert 6.5 < status["elapsed_days"] < 7.5
        assert 6.5 < status["remaining_days"] < 7.5
        assert status["window_expired"] is False
        assert status["manifest_intact"] is True

    def test_window_expired(self, tmp_path):
        start = (datetime.now(timezone.utc) - timedelta(days=15)).isoformat()
        p = _make_runtime_yaml(tmp_path, _aw_config(start_ts=start))
        manifest = _make_manifest(tmp_path)
        state_path = tmp_path / "state.json"
        doctrine = tmp_path / "doctrine.jsonl"

        os.environ["ACTIVATION_WINDOW_ACK"] = "1"
        with mock.patch.object(aw, "NAV_STATE_PATH", tmp_path / "nav.json"), \
             mock.patch.object(aw, "BINARY_LAB_STATE_PATH", tmp_path / "bl.json"), \
             mock.patch.object(aw, "DLE_SHADOW_LOG", tmp_path / "dle.jsonl"), \
             mock.patch.object(aw, "EPISODE_LEDGER_PATH", tmp_path / "ep.json"), \
             mock.patch("execution.activation_window._get_portfolio_dd_pct", return_value=0.01), \
             mock.patch("execution.activation_window._count_risk_vetoes_since", return_value=5):
            _make_episode_ledger(tmp_path, [])
            status = check_activation_window(
                runtime_yaml=p, manifest_path=manifest,
                state_path=state_path, doctrine_log=doctrine,
            )

        assert status["active"] is True
        assert status["halted"] is True
        assert status["window_expired"] is True
        assert "window_complete" in status["halt_reason"]
        assert os.environ.get("KILL_SWITCH") == "1"


# ===========================================================================
# Tests: Structural integrity — manifest drift
# ===========================================================================

class TestManifestIntegrity:
    def test_manifest_drift_triggers_halt(self, tmp_path):
        start = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
        p = _make_runtime_yaml(tmp_path, _aw_config(start_ts=start))
        manifest = _make_manifest(tmp_path, '{"version": "v8-original"}')
        state_path = tmp_path / "state.json"
        doctrine = tmp_path / "doctrine.jsonl"

        os.environ["ACTIVATION_WINDOW_ACK"] = "1"

        with mock.patch.object(aw, "NAV_STATE_PATH", tmp_path / "nav.json"), \
             mock.patch.object(aw, "BINARY_LAB_STATE_PATH", tmp_path / "bl.json"), \
             mock.patch.object(aw, "DLE_SHADOW_LOG", tmp_path / "dle.jsonl"), \
             mock.patch.object(aw, "EPISODE_LEDGER_PATH", tmp_path / "ep.json"), \
             mock.patch("execution.activation_window._get_portfolio_dd_pct", return_value=0.01), \
             mock.patch("execution.activation_window._count_risk_vetoes_since", return_value=0):
            _make_episode_ledger(tmp_path, [])
            # First check: captures boot hash
            status1 = check_activation_window(
                runtime_yaml=p, manifest_path=manifest,
                state_path=state_path, doctrine_log=doctrine,
            )
            assert status1["halted"] is False
            assert status1["manifest_intact"] is True

            # Mutate manifest
            _reset_globals()
            # Keep the boot hash but set a new one
            aw._boot_manifest_hash = status1["boot_manifest_hash"]
            manifest.write_text('{"version": "v8-MUTATED"}')

            status2 = check_activation_window(
                runtime_yaml=p, manifest_path=manifest,
                state_path=state_path, doctrine_log=doctrine,
            )

        assert status2["manifest_intact"] is False
        assert status2["halted"] is True
        assert "manifest_drift" in status2["halt_reason"]


# ===========================================================================
# Tests: Config drift
# ===========================================================================

class TestConfigDrift:
    def test_config_drift_triggers_halt(self, tmp_path):
        start = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
        cfg = _aw_config(start_ts=start)
        p = _make_runtime_yaml(tmp_path, cfg)
        manifest = _make_manifest(tmp_path)
        state_path = tmp_path / "state.json"
        doctrine = tmp_path / "doctrine.jsonl"

        os.environ["ACTIVATION_WINDOW_ACK"] = "1"

        with mock.patch.object(aw, "NAV_STATE_PATH", tmp_path / "nav.json"), \
             mock.patch.object(aw, "BINARY_LAB_STATE_PATH", tmp_path / "bl.json"), \
             mock.patch.object(aw, "DLE_SHADOW_LOG", tmp_path / "dle.jsonl"), \
             mock.patch.object(aw, "EPISODE_LEDGER_PATH", tmp_path / "ep.json"), \
             mock.patch("execution.activation_window._get_portfolio_dd_pct", return_value=0.01), \
             mock.patch("execution.activation_window._count_risk_vetoes_since", return_value=0):
            _make_episode_ledger(tmp_path, [])
            # First check: captures boot hash
            status1 = check_activation_window(
                runtime_yaml=p, manifest_path=manifest,
                state_path=state_path, doctrine_log=doctrine,
            )
            assert status1["config_intact"] is True

            # Mutate config
            _reset_globals()
            aw._boot_config_hash = status1["boot_config_hash"]
            cfg["per_trade_nav_pct"] = 0.01  # changed!
            _make_runtime_yaml(tmp_path, cfg)

            status2 = check_activation_window(
                runtime_yaml=p, manifest_path=manifest,
                state_path=state_path, doctrine_log=doctrine,
            )

        assert status2["config_intact"] is False
        assert status2["halted"] is True
        assert "config_drift" in status2["halt_reason"]


# ===========================================================================
# Tests: Drawdown kill
# ===========================================================================

class TestDrawdownKill:
    def test_dd_breach_triggers_halt(self, tmp_path):
        start = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
        p = _make_runtime_yaml(tmp_path, _aw_config(start_ts=start))
        manifest = _make_manifest(tmp_path)
        state_path = tmp_path / "state.json"
        doctrine = tmp_path / "doctrine.jsonl"

        os.environ["ACTIVATION_WINDOW_ACK"] = "1"

        with mock.patch.object(aw, "NAV_STATE_PATH", tmp_path / "nav.json"), \
             mock.patch.object(aw, "BINARY_LAB_STATE_PATH", tmp_path / "bl.json"), \
             mock.patch.object(aw, "DLE_SHADOW_LOG", tmp_path / "dle.jsonl"), \
             mock.patch.object(aw, "EPISODE_LEDGER_PATH", tmp_path / "ep.json"), \
             mock.patch("execution.activation_window._get_portfolio_dd_pct", return_value=0.06), \
             mock.patch("execution.activation_window._count_risk_vetoes_since", return_value=0):
            _make_episode_ledger(tmp_path, [])
            status = check_activation_window(
                runtime_yaml=p, manifest_path=manifest,
                state_path=state_path, doctrine_log=doctrine,
            )

        assert status["halted"] is True
        assert status["dd_breached"] is True
        assert "drawdown_kill" in status["halt_reason"]
        assert os.environ.get("KILL_SWITCH") == "1"

    def test_dd_below_threshold_no_halt(self, tmp_path):
        start = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
        p = _make_runtime_yaml(tmp_path, _aw_config(start_ts=start))
        manifest = _make_manifest(tmp_path)
        state_path = tmp_path / "state.json"
        doctrine = tmp_path / "doctrine.jsonl"

        os.environ["ACTIVATION_WINDOW_ACK"] = "1"

        with mock.patch.object(aw, "NAV_STATE_PATH", tmp_path / "nav.json"), \
             mock.patch.object(aw, "BINARY_LAB_STATE_PATH", tmp_path / "bl.json"), \
             mock.patch.object(aw, "DLE_SHADOW_LOG", tmp_path / "dle.jsonl"), \
             mock.patch.object(aw, "EPISODE_LEDGER_PATH", tmp_path / "ep.json"), \
             mock.patch("execution.activation_window._get_portfolio_dd_pct", return_value=0.02), \
             mock.patch("execution.activation_window._count_risk_vetoes_since", return_value=0):
            _make_episode_ledger(tmp_path, [])
            status = check_activation_window(
                runtime_yaml=p, manifest_path=manifest,
                state_path=state_path, doctrine_log=doctrine,
            )

        assert status["halted"] is False
        assert status["dd_breached"] is False


# ===========================================================================
# Tests: Binary Lab freeze
# ===========================================================================

class TestBinaryLabFreeze:
    def test_freeze_violation_triggers_halt(self, tmp_path):
        start = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
        p = _make_runtime_yaml(tmp_path, _aw_config(start_ts=start))
        manifest = _make_manifest(tmp_path)
        state_path = tmp_path / "state.json"
        doctrine = tmp_path / "doctrine.jsonl"
        bl_state = tmp_path / "bl.json"
        with bl_state.open("w") as f:
            json.dump({"freeze_intact": False, "status": "SHADOW"}, f)

        os.environ["ACTIVATION_WINDOW_ACK"] = "1"

        with mock.patch.object(aw, "NAV_STATE_PATH", tmp_path / "nav.json"), \
             mock.patch.object(aw, "BINARY_LAB_STATE_PATH", bl_state), \
             mock.patch.object(aw, "DLE_SHADOW_LOG", tmp_path / "dle.jsonl"), \
             mock.patch.object(aw, "EPISODE_LEDGER_PATH", tmp_path / "ep.json"), \
             mock.patch("execution.activation_window._get_portfolio_dd_pct", return_value=0.01), \
             mock.patch("execution.activation_window._count_risk_vetoes_since", return_value=0):
            _make_episode_ledger(tmp_path, [])
            status = check_activation_window(
                runtime_yaml=p, manifest_path=manifest,
                state_path=state_path, doctrine_log=doctrine,
            )

        assert status["halted"] is True
        assert "binary_lab_freeze_violation" in status["halt_reason"]


# ===========================================================================
# Tests: KILL_SWITCH idempotency
# ===========================================================================

class TestKillSwitchIdempotency:
    def test_only_one_doctrine_event(self, tmp_path):
        start = (datetime.now(timezone.utc) - timedelta(days=15)).isoformat()
        p = _make_runtime_yaml(tmp_path, _aw_config(start_ts=start))
        manifest = _make_manifest(tmp_path)
        state_path = tmp_path / "state.json"
        doctrine = tmp_path / "doctrine.jsonl"

        os.environ["ACTIVATION_WINDOW_ACK"] = "1"

        with mock.patch.object(aw, "NAV_STATE_PATH", tmp_path / "nav.json"), \
             mock.patch.object(aw, "BINARY_LAB_STATE_PATH", tmp_path / "bl.json"), \
             mock.patch.object(aw, "DLE_SHADOW_LOG", tmp_path / "dle.jsonl"), \
             mock.patch.object(aw, "EPISODE_LEDGER_PATH", tmp_path / "ep.json"), \
             mock.patch("execution.activation_window._get_portfolio_dd_pct", return_value=0.01), \
             mock.patch("execution.activation_window._count_risk_vetoes_since", return_value=0):
            _make_episode_ledger(tmp_path, [])
            # Call twice
            check_activation_window(
                runtime_yaml=p, manifest_path=manifest,
                state_path=state_path, doctrine_log=doctrine,
            )
            check_activation_window(
                runtime_yaml=p, manifest_path=manifest,
                state_path=state_path, doctrine_log=doctrine,
            )

        # Only one doctrine event
        if doctrine.exists():
            lines = doctrine.read_text().strip().split("\n")
            events = [json.loads(l) for l in lines if l.strip()]
            halt_events = [e for e in events if e.get("event") == "ACTIVATION_WINDOW_HALT"]
            assert len(halt_events) == 1


# ===========================================================================
# Tests: State file emission
# ===========================================================================

class TestStateFileEmission:
    def test_state_written_when_active(self, tmp_path):
        start = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        p = _make_runtime_yaml(tmp_path, _aw_config(start_ts=start))
        manifest = _make_manifest(tmp_path)
        state_path = tmp_path / "state.json"
        doctrine = tmp_path / "doctrine.jsonl"

        os.environ["ACTIVATION_WINDOW_ACK"] = "1"

        with mock.patch.object(aw, "NAV_STATE_PATH", tmp_path / "nav.json"), \
             mock.patch.object(aw, "BINARY_LAB_STATE_PATH", tmp_path / "bl.json"), \
             mock.patch.object(aw, "DLE_SHADOW_LOG", tmp_path / "dle.jsonl"), \
             mock.patch.object(aw, "EPISODE_LEDGER_PATH", tmp_path / "ep.json"), \
             mock.patch("execution.activation_window._get_portfolio_dd_pct", return_value=0.01), \
             mock.patch("execution.activation_window._count_risk_vetoes_since", return_value=3):
            _make_episode_ledger(tmp_path, [])
            check_activation_window(
                runtime_yaml=p, manifest_path=manifest,
                state_path=state_path, doctrine_log=doctrine,
            )

        assert state_path.exists()
        data = json.loads(state_path.read_text())
        assert data["active"] is True
        assert "elapsed_days" in data
        assert "manifest_intact" in data

    def test_state_written_when_inactive(self, tmp_path):
        p = _make_runtime_yaml(tmp_path, {"enabled": False})
        state_path = tmp_path / "state.json"
        manifest = _make_manifest(tmp_path)

        check_activation_window(
            runtime_yaml=p, manifest_path=manifest,
            state_path=state_path, doctrine_log=tmp_path / "doc.jsonl",
        )

        assert state_path.exists()
        data = json.loads(state_path.read_text())
        assert data["active"] is False


# ===========================================================================
# Tests: Inactive window
# ===========================================================================

class TestInactiveWindow:
    def test_inactive_when_disabled(self, tmp_path):
        p = _make_runtime_yaml(tmp_path, {"enabled": False})
        manifest = _make_manifest(tmp_path)
        status = check_activation_window(
            runtime_yaml=p, manifest_path=manifest,
            state_path=tmp_path / "s.json", doctrine_log=tmp_path / "d.jsonl",
        )
        assert status["active"] is False

    def test_inactive_when_no_start_ts(self, tmp_path):
        p = _make_runtime_yaml(tmp_path, _aw_config(start_ts=""))
        manifest = _make_manifest(tmp_path)
        os.environ["ACTIVATION_WINDOW_ACK"] = "1"
        status = check_activation_window(
            runtime_yaml=p, manifest_path=manifest,
            state_path=tmp_path / "s.json", doctrine_log=tmp_path / "d.jsonl",
        )
        assert status["active"] is False


# ===========================================================================
# Tests: Boot status
# ===========================================================================

class TestBootStatus:
    def test_boot_inactive(self, tmp_path, caplog):
        import logging
        p = _make_runtime_yaml(tmp_path, {"enabled": False})
        with caplog.at_level(logging.INFO):
            log_activation_boot_status(runtime_yaml=p, manifest_path=tmp_path / "m.json")
        assert "INACTIVE" in caplog.text

    def test_boot_active(self, tmp_path, caplog):
        import logging
        p = _make_runtime_yaml(tmp_path, _aw_config())
        manifest = _make_manifest(tmp_path)
        os.environ["ACTIVATION_WINDOW_ACK"] = "1"
        _make_nav_state(tmp_path)
        with mock.patch.object(aw, "NAV_STATE_PATH", tmp_path / "nav_state.json"):
            with caplog.at_level(logging.INFO):
                log_activation_boot_status(runtime_yaml=p, manifest_path=manifest)
        assert "ACTIVE" in caplog.text
        assert "14d window" in caplog.text

    def test_boot_dual_key_missing(self, tmp_path, caplog):
        import logging
        p = _make_runtime_yaml(tmp_path, _aw_config())
        # No ACK set
        with caplog.at_level(logging.WARNING):
            log_activation_boot_status(runtime_yaml=p, manifest_path=tmp_path / "m.json")
        assert "dual-key missing" in caplog.text


# ===========================================================================
# Tests: Sizing override
# ===========================================================================

class TestSizingOverride:
    def test_returns_override_when_active(self, tmp_path):
        p = _make_runtime_yaml(tmp_path, _aw_config())
        os.environ["ACTIVATION_WINDOW_ACK"] = "1"
        result = get_activation_sizing_override(runtime_yaml=p)
        assert result == 0.005

    def test_returns_none_when_inactive(self, tmp_path):
        p = _make_runtime_yaml(tmp_path, {"enabled": False})
        result = get_activation_sizing_override(runtime_yaml=p)
        assert result is None

    def test_returns_none_when_no_pct(self, tmp_path):
        cfg = _aw_config()
        del cfg["per_trade_nav_pct"]
        p = _make_runtime_yaml(tmp_path, cfg)
        os.environ["ACTIVATION_WINDOW_ACK"] = "1"
        result = get_activation_sizing_override(runtime_yaml=p)
        assert result is None


# ===========================================================================
# Tests: Stack health collection
# ===========================================================================

class TestStackHealth:
    def test_collects_health(self, tmp_path):
        p = _make_runtime_yaml(tmp_path, _aw_config())
        manifest = _make_manifest(tmp_path)
        os.environ["ACTIVATION_WINDOW_ACK"] = "1"

        with mock.patch.object(aw, "NAV_STATE_PATH", tmp_path / "nav.json"), \
             mock.patch.object(aw, "BINARY_LAB_STATE_PATH", tmp_path / "bl.json"), \
             mock.patch.object(aw, "DLE_SHADOW_LOG", tmp_path / "dle.jsonl"), \
             mock.patch.object(aw, "EPISODE_LEDGER_PATH", tmp_path / "ep.json"), \
             mock.patch("execution.activation_window._get_portfolio_dd_pct", return_value=0.02), \
             mock.patch("execution.activation_window._count_risk_vetoes_since", return_value=10):
            _make_episode_ledger(tmp_path, [])
            health = collect_stack_health(runtime_yaml=p, manifest_path=manifest)

        assert "nav_usd" in health
        assert health["dd_within_limits"] is True
        assert health["drawdown_pct"] == 0.02
        assert health["manifest_hash"] is not None


# ===========================================================================
# Tests: activation_verify.py 7-gate verification
# ===========================================================================

class TestActivationVerify:
    def test_all_gates_pass(self, tmp_path):
        from scripts.activation_verify import run_verification

        state = {
            "active": True,
            "start_ts": "2026-02-12T00:00:00Z",
            "nav_usd": 9666.0,
            "drawdown_pct": 0.02,
            "drawdown_kill_pct": 0.05,
            "manifest_intact": True,
            "config_intact": True,
            "binary_lab_freeze_ok": True,
            "dle_mismatches": 0,
            "elapsed_days": 14.0,
            "episodes_completed": 45,
            "boot_manifest_hash": "abc123",
            "current_manifest_hash": "abc123",
        }
        state_path = _make_activation_state(tmp_path, state)

        result = run_verification(
            activation_state_path=state_path,
            manifest_path=tmp_path / "m.json",
            binary_lab_trades_log=tmp_path / "bl.jsonl",
            risk_vetoes_log=tmp_path / "rv.jsonl",
            dle_shadow_log=tmp_path / "dle.jsonl",
        )

        assert result["verdict"] == "GO"
        assert result["passed"] == 7
        assert result["total_gates"] == 7
        assert result["action"] == "Promote to Production"

    def test_one_gate_fails_extend(self, tmp_path):
        from scripts.activation_verify import run_verification

        state = {
            "active": True,
            "start_ts": "2026-02-12T00:00:00Z",
            "nav_usd": 9666.0,
            "drawdown_pct": 0.02,
            "drawdown_kill_pct": 0.05,
            "manifest_intact": False,  # FAIL
            "config_intact": True,
            "binary_lab_freeze_ok": True,
            "dle_mismatches": 0,
            "elapsed_days": 14.0,
            "episodes_completed": 45,
        }
        state_path = _make_activation_state(tmp_path, state)

        result = run_verification(
            activation_state_path=state_path,
            manifest_path=tmp_path / "m.json",
            binary_lab_trades_log=tmp_path / "bl.jsonl",
            risk_vetoes_log=tmp_path / "rv.jsonl",
            dle_shadow_log=tmp_path / "dle.jsonl",
        )

        assert result["verdict"] == "EXTEND"
        # manifest_intact=False fails gate 5, but gate 6 (no_freeze_violations)
        # checks both freeze_ok AND config_intact — config_intact is True so
        # freeze_ok=True means gate 6 passes.  Only gate 5 fails → 6/7.
        assert result["passed"] == 6

    def test_multiple_gates_fail_nogo(self, tmp_path):
        from scripts.activation_verify import run_verification

        state = {
            "active": True,
            "start_ts": "2026-02-12T00:00:00Z",
            "nav_usd": 0,  # FAIL
            "drawdown_pct": 0.08,  # FAIL
            "drawdown_kill_pct": 0.05,
            "manifest_intact": False,  # FAIL
            "config_intact": False,  # FAIL
            "binary_lab_freeze_ok": False,  # FAIL
            "dle_mismatches": 100,
            "elapsed_days": 14.0,
        }
        state_path = _make_activation_state(tmp_path, state)

        result = run_verification(
            activation_state_path=state_path,
            manifest_path=tmp_path / "m.json",
            binary_lab_trades_log=tmp_path / "bl.jsonl",
            risk_vetoes_log=tmp_path / "rv.jsonl",
            dle_shadow_log=tmp_path / "dle.jsonl",
        )

        assert result["verdict"] == "NO-GO"
        assert result["passed"] <= 5
        assert result["action"] == "Investigate, do not scale"


# ===========================================================================
# Tests: Episode and veto counting
# ===========================================================================

class TestCounting:
    def test_episode_count(self, tmp_path):
        episodes = [
            {"exit_ts": "2026-02-20T10:00:00+00:00", "symbol": "BTCUSDT"},
            {"exit_ts": "2026-02-21T10:00:00+00:00", "symbol": "ETHUSDT"},
            {"exit_ts": "2026-02-10T10:00:00+00:00", "symbol": "OLD"},  # before start
        ]
        ledger_path = tmp_path / "episode_ledger.json"
        with ledger_path.open("w") as f:
            json.dump({"episodes": episodes}, f)
        with mock.patch.object(aw, "EPISODE_LEDGER_PATH", ledger_path):
            count = aw._count_episodes_since("2026-02-15T00:00:00Z")
        assert count == 2

    def test_risk_veto_count_excludes_min_notional(self, tmp_path):
        """min_notional vetoes are plumbing, not governance risk."""
        veto_log = tmp_path / "risk_vetoes.jsonl"
        with veto_log.open("w") as f:
            # Governance vetoes — counted
            f.write(json.dumps({"ts": "2026-02-20T10:00:00Z", "veto_reason": "max_concurrent"}) + "\n")
            f.write(json.dumps({"ts": "2026-02-21T10:00:00Z", "veto_reason": "drawdown"}) + "\n")
            # Plumbing vetoes — excluded
            f.write(json.dumps({"ts": "2026-02-20T11:00:00Z", "veto_reason": "min_notional"}) + "\n")
            f.write(json.dumps({"ts": "2026-02-20T12:00:00Z", "veto_reason": "below_min_notional"}) + "\n")
            # Before start — excluded by timestamp
            f.write(json.dumps({"ts": "2026-02-10T10:00:00Z", "veto_reason": "max_concurrent"}) + "\n")
            # No veto_reason field — counted (empty string not in exclusion set)
            f.write(json.dumps({"ts": "2026-02-22T10:00:00Z"}) + "\n")

        with mock.patch.object(aw, "RISK_VETOES_LOG_PATH", veto_log):
            count = aw._count_risk_vetoes_since("2026-02-15T00:00:00Z")
        # max_concurrent + drawdown + no-reason = 3
        assert count == 3

    def test_risk_veto_count_all_min_notional_is_zero(self, tmp_path):
        """A flood of min_notional vetoes should produce count=0."""
        veto_log = tmp_path / "risk_vetoes.jsonl"
        with veto_log.open("w") as f:
            for i in range(100):
                f.write(json.dumps({"ts": "2026-02-20T10:00:00Z", "veto_reason": "min_notional"}) + "\n")

        with mock.patch.object(aw, "RISK_VETOES_LOG_PATH", veto_log):
            count = aw._count_risk_vetoes_since("2026-02-15T00:00:00Z")
        assert count == 0

    def test_plumbing_veto_count(self, tmp_path):
        """Plumbing counter is the complement of governance counter."""
        veto_log = tmp_path / "risk_vetoes.jsonl"
        with veto_log.open("w") as f:
            f.write(json.dumps({"ts": "2026-02-20T10:00:00Z", "veto_reason": "max_concurrent"}) + "\n")
            f.write(json.dumps({"ts": "2026-02-20T11:00:00Z", "veto_reason": "min_notional"}) + "\n")
            f.write(json.dumps({"ts": "2026-02-20T12:00:00Z", "veto_reason": "below_min_notional"}) + "\n")
            f.write(json.dumps({"ts": "2026-02-21T10:00:00Z", "veto_reason": "drawdown"}) + "\n")
            f.write(json.dumps({"ts": "2026-02-10T10:00:00Z", "veto_reason": "min_notional"}) + "\n")

        with mock.patch.object(aw, "RISK_VETOES_LOG_PATH", veto_log):
            plumbing = aw._count_plumbing_vetoes_since("2026-02-15T00:00:00Z")
            governance = aw._count_risk_vetoes_since("2026-02-15T00:00:00Z")
        # 2 plumbing after start, 2 governance after start
        assert plumbing == 2
        assert governance == 2

    def test_dle_mismatch_count(self, tmp_path):
        dle_log = tmp_path / "dle.jsonl"
        with dle_log.open("w") as f:
            f.write(json.dumps({"ts": "2026-02-20T10:00:00Z", "mismatch": True}) + "\n")
            f.write(json.dumps({"ts": "2026-02-21T10:00:00Z", "mismatch": False}) + "\n")
            f.write(json.dumps({"ts": "2026-02-22T10:00:00Z", "event": "DLE_MISMATCH"}) + "\n")
            f.write(json.dumps({"ts": "2026-02-10T10:00:00Z", "mismatch": True}) + "\n")
        with mock.patch.object(aw, "DLE_SHADOW_LOG", dle_log):
            count = aw._check_dle_anomalies("2026-02-15T00:00:00Z")
        assert count == 2  # Two after start with mismatch flag

    def test_binary_lab_freeze_ok(self, tmp_path):
        bl = tmp_path / "bl.json"
        with bl.open("w") as f:
            json.dump({"freeze_intact": True}, f)
        with mock.patch.object(aw, "BINARY_LAB_STATE_PATH", bl):
            assert aw._check_binary_lab_freeze() is True

    def test_binary_lab_freeze_violated(self, tmp_path):
        bl = tmp_path / "bl.json"
        with bl.open("w") as f:
            json.dump({"freeze_intact": False, "status": "SHADOW"}, f)
        with mock.patch.object(aw, "BINARY_LAB_STATE_PATH", bl):
            assert aw._check_binary_lab_freeze() is False

    def test_binary_lab_no_state(self, tmp_path):
        with mock.patch.object(aw, "BINARY_LAB_STATE_PATH", tmp_path / "nope.json"):
            assert aw._check_binary_lab_freeze() is True  # No state = no violation
