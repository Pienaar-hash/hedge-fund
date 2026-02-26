"""
Tests for Phase C / DLE governance binding of Activation Window v8.0.

Covers:
- DLE lifecycle event emission (STARTED, HALTED, COMPLETED, VERIFIED)
- Production scale gate (require_go_for_scale)
- Verdict recording and persistence
- verify_production_scale_eligible()
- get_scale_gate_cap()
"""
from __future__ import annotations

import json
import os
import textwrap
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest
import yaml

from execution.activation_window import (
    _emit_dle_lifecycle_event,
    _reset_globals,
    check_activation_window,
    collect_stack_health,
    get_activation_sizing_override,
    get_scale_gate_cap,
    log_activation_boot_status,
    record_verification_verdict,
    verify_production_scale_eligible,
    STRUCTURAL_GUARD_EVENT_TYPE,
    GUARD_TYPE,
    PHASE_ID,
    VERIFICATION_VERDICT_PATH,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_env(tmp_path, monkeypatch):
    """Reset module globals and env for every test."""
    _reset_globals()
    monkeypatch.delenv("KILL_SWITCH", raising=False)
    monkeypatch.delenv("ACTIVATION_WINDOW_ACK", raising=False)
    yield
    _reset_globals()


def _make_runtime_yaml(tmp_path: Path, overrides: Dict[str, Any] | None = None) -> Path:
    """Create a temporary runtime.yaml with activation_window section."""
    aw = {
        "enabled": True,
        "duration_days": 14,
        "start_ts": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
        "drawdown_kill_pct": 0.05,
        "per_trade_nav_pct": 0.005,
        "require_go_for_scale": True,
    }
    if overrides:
        aw.update(overrides)
    p = tmp_path / "runtime.yaml"
    p.write_text(yaml.dump({"activation_window": aw}))
    return p


def _make_manifest(tmp_path: Path, content: str = '{"hello": "world"}') -> Path:
    p = tmp_path / "v7_manifest.json"
    p.write_text(content)
    return p


def _make_verdict(
    tmp_path: Path,
    *,
    verdict: str = "GO",
    passed: int = 7,
    total_gates: int = 7,
    manifest_hash: str | None = None,
) -> Path:
    """Create a verdict file with given parameters."""
    data = {
        "verdict": verdict,
        "passed": passed,
        "total_gates": total_gates,
        "action": "Promote to Production" if verdict == "GO" else "Investigate",
        "manifest_hash": manifest_hash,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    p = tmp_path / "activation_verification_verdict.json"
    p.write_text(json.dumps(data))
    return p


def _read_dle_events(log_path: Path) -> list:
    """Read all DLE shadow events from a JSONL file."""
    if not log_path.exists():
        return []
    events = []
    for line in log_path.read_text().splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


# ---------------------------------------------------------------------------
# DLE Lifecycle Event Tests
# ---------------------------------------------------------------------------

class TestDLELifecycleEvents:
    """Test STRUCTURAL_GUARD event emission to DLE shadow log."""

    def test_emit_dle_lifecycle_event_writes_to_log(self, tmp_path):
        """STRUCTURAL_GUARD events are written to the shadow log."""
        log_path = tmp_path / "execution" / "dle_shadow_events.jsonl"

        written_events = []

        class MockWriter:
            def __init__(self, *args, **kwargs):
                pass

            def write(self, event):
                written_events.append(event)

        with patch("execution.dle_shadow.DLEShadowWriter", MockWriter):
            _emit_dle_lifecycle_event("STARTED", {
                "window_start_ts": "2026-02-15T00:00:00+00:00",
                "duration_days": 14,
                "manifest_hash": "abc123",
            })

        assert len(written_events) == 1
        evt = written_events[0]
        assert evt.event_type == STRUCTURAL_GUARD_EVENT_TYPE
        assert evt.schema_version == "dle_shadow_v2"
        assert evt.payload["guard_type"] == GUARD_TYPE
        assert evt.payload["action"] == "STARTED"
        assert evt.payload["phase_id"] == PHASE_ID

    def test_emit_dle_lifecycle_event_fail_open(self):
        """Emission failures never raise — fail-open for executor safety."""
        # Patch the import to fail
        with patch(
            "execution.activation_window._emit_dle_lifecycle_event"
        ) as mock_emit:
            mock_emit.side_effect = None  # no-op
            # Should not raise
            _emit_dle_lifecycle_event("STARTED", {})

    def test_boot_emits_started_event(self, tmp_path, monkeypatch):
        """log_activation_boot_status() emits STARTED DLE event."""
        monkeypatch.setenv("ACTIVATION_WINDOW_ACK", "1")
        yaml_path = _make_runtime_yaml(tmp_path)
        manifest_path = _make_manifest(tmp_path)

        emitted = []
        original_emit = _emit_dle_lifecycle_event

        def capture(action, details):
            emitted.append((action, details))

        with patch("execution.activation_window._emit_dle_lifecycle_event", capture):
            # Mock _get_nav_usd to return > 0
            with patch("execution.activation_window._get_nav_usd", return_value=1000.0):
                log_activation_boot_status(
                    runtime_yaml=yaml_path,
                    manifest_path=manifest_path,
                )

        assert len(emitted) == 1
        action, details = emitted[0]
        assert action == "STARTED"
        assert details["duration_days"] == 14
        assert "manifest_hash" in details
        assert "config_hash" in details

    def test_boot_emits_started_once(self, tmp_path, monkeypatch):
        """STARTED is only emitted once per boot (idempotent)."""
        monkeypatch.setenv("ACTIVATION_WINDOW_ACK", "1")
        yaml_path = _make_runtime_yaml(tmp_path)
        manifest_path = _make_manifest(tmp_path)

        emitted = []

        def capture(action, details):
            emitted.append((action, details))

        with patch("execution.activation_window._emit_dle_lifecycle_event", capture):
            with patch("execution.activation_window._get_nav_usd", return_value=1000.0):
                log_activation_boot_status(
                    runtime_yaml=yaml_path,
                    manifest_path=manifest_path,
                )
                log_activation_boot_status(
                    runtime_yaml=yaml_path,
                    manifest_path=manifest_path,
                )

        # Only one STARTED event despite two calls
        assert len(emitted) == 1

    def test_halt_emits_halted_on_drawdown(self, tmp_path, monkeypatch):
        """check_activation_window() emits HALTED on drawdown kill."""
        monkeypatch.setenv("ACTIVATION_WINDOW_ACK", "1")
        yaml_path = _make_runtime_yaml(tmp_path, {"drawdown_kill_pct": 0.03})
        manifest_path = _make_manifest(tmp_path)
        state_path = tmp_path / "state" / "activation_window_state.json"
        doctrine_log = tmp_path / "doctrine_events.jsonl"

        emitted = []

        def capture(action, details):
            emitted.append((action, details))

        with patch("execution.activation_window._emit_dle_lifecycle_event", capture):
            with patch("execution.activation_window._get_portfolio_dd_pct", return_value=0.04):
                with patch("execution.activation_window._get_nav_usd", return_value=1000.0):
                    with patch("execution.activation_window._check_binary_lab_freeze", return_value=True):
                        with patch("execution.activation_window._check_dle_anomalies", return_value=0):
                            with patch("execution.activation_window._count_episodes_since", return_value=5):
                                with patch("execution.activation_window._count_risk_vetoes_since", return_value=10):
                                    result = check_activation_window(
                                        runtime_yaml=yaml_path,
                                        manifest_path=manifest_path,
                                        state_path=state_path,
                                        doctrine_log=doctrine_log,
                                    )

        assert result["halted"]
        assert len(emitted) == 1
        action, details = emitted[0]
        assert action == "HALTED"
        assert "drawdown_kill" in details["halt_reason"]

    def test_halt_emits_completed_on_window_expire(self, tmp_path, monkeypatch):
        """check_activation_window() emits COMPLETED (not HALTED) when window expires."""
        monkeypatch.setenv("ACTIVATION_WINDOW_ACK", "1")
        start_ts = (datetime.now(timezone.utc) - timedelta(days=15)).isoformat()
        yaml_path = _make_runtime_yaml(tmp_path, {"start_ts": start_ts})
        manifest_path = _make_manifest(tmp_path)
        state_path = tmp_path / "state" / "activation_window_state.json"
        doctrine_log = tmp_path / "doctrine_events.jsonl"

        emitted = []

        def capture(action, details):
            emitted.append((action, details))

        with patch("execution.activation_window._emit_dle_lifecycle_event", capture):
            with patch("execution.activation_window._get_portfolio_dd_pct", return_value=0.01):
                with patch("execution.activation_window._get_nav_usd", return_value=1000.0):
                    with patch("execution.activation_window._check_binary_lab_freeze", return_value=True):
                        with patch("execution.activation_window._check_dle_anomalies", return_value=0):
                            with patch("execution.activation_window._count_episodes_since", return_value=100):
                                with patch("execution.activation_window._count_risk_vetoes_since", return_value=5):
                                    result = check_activation_window(
                                        runtime_yaml=yaml_path,
                                        manifest_path=manifest_path,
                                        state_path=state_path,
                                        doctrine_log=doctrine_log,
                                    )

        assert result["halted"]
        assert result["window_expired"]
        assert len(emitted) == 1
        action, details = emitted[0]
        assert action == "COMPLETED"  # NOT "HALTED"
        assert "window_complete" in details["halt_reason"]

    def test_halt_emits_halted_on_manifest_drift(self, tmp_path, monkeypatch):
        """Manifest drift emits HALTED, not COMPLETED."""
        monkeypatch.setenv("ACTIVATION_WINDOW_ACK", "1")
        yaml_path = _make_runtime_yaml(tmp_path)
        manifest_path = _make_manifest(tmp_path)
        state_path = tmp_path / "state" / "activation_window_state.json"
        doctrine_log = tmp_path / "doctrine_events.jsonl"

        # First call captures boot hash
        with patch("execution.activation_window._get_portfolio_dd_pct", return_value=0.0):
            with patch("execution.activation_window._get_nav_usd", return_value=1000.0):
                with patch("execution.activation_window._check_binary_lab_freeze", return_value=True):
                    with patch("execution.activation_window._check_dle_anomalies", return_value=0):
                        with patch("execution.activation_window._count_episodes_since", return_value=0):
                            with patch("execution.activation_window._count_risk_vetoes_since", return_value=0):
                                check_activation_window(
                                    runtime_yaml=yaml_path,
                                    manifest_path=manifest_path,
                                    state_path=state_path,
                                    doctrine_log=doctrine_log,
                                )

        # Now modify manifest (simulates drift)
        manifest_path.write_text('{"modified": true}')

        emitted = []

        def capture(action, details):
            emitted.append((action, details))

        with patch("execution.activation_window._emit_dle_lifecycle_event", capture):
            with patch("execution.activation_window._get_portfolio_dd_pct", return_value=0.0):
                with patch("execution.activation_window._get_nav_usd", return_value=1000.0):
                    with patch("execution.activation_window._check_binary_lab_freeze", return_value=True):
                        with patch("execution.activation_window._check_dle_anomalies", return_value=0):
                            with patch("execution.activation_window._count_episodes_since", return_value=0):
                                with patch("execution.activation_window._count_risk_vetoes_since", return_value=0):
                                    result = check_activation_window(
                                        runtime_yaml=yaml_path,
                                        manifest_path=manifest_path,
                                        state_path=state_path,
                                        doctrine_log=doctrine_log,
                                    )

        assert result["halted"]
        assert not result["manifest_intact"]
        assert len(emitted) == 1
        assert emitted[0][0] == "HALTED"


# ---------------------------------------------------------------------------
# Production Scale Gate Tests
# ---------------------------------------------------------------------------

class TestScaleGate:
    """Test the production scale gate (require_go_for_scale)."""

    def test_no_config_returns_none(self, tmp_path):
        """No activation_window config → no gate."""
        yaml_path = tmp_path / "runtime.yaml"
        yaml_path.write_text(yaml.dump({"other": "stuff"}))
        assert get_scale_gate_cap(runtime_yaml=yaml_path) is None

    def test_require_go_disabled_returns_none(self, tmp_path):
        """require_go_for_scale: false → no gate."""
        yaml_path = _make_runtime_yaml(tmp_path, {"require_go_for_scale": False})
        assert get_scale_gate_cap(runtime_yaml=yaml_path) is None

    def test_no_verdict_returns_cap(self, tmp_path):
        """No verdict file → sizing cap active."""
        yaml_path = _make_runtime_yaml(tmp_path)
        verdict_path = tmp_path / "nonexistent_verdict.json"
        cap = get_scale_gate_cap(
            runtime_yaml=yaml_path,
            verdict_path=verdict_path,
        )
        assert cap == 0.005

    def test_no_go_verdict_returns_cap(self, tmp_path):
        """Verdict != GO → sizing cap stays."""
        yaml_path = _make_runtime_yaml(tmp_path)
        manifest_path = _make_manifest(tmp_path)
        verdict_path = _make_verdict(
            tmp_path, verdict="EXTEND", passed=6, manifest_hash="abc"
        )
        cap = get_scale_gate_cap(
            runtime_yaml=yaml_path,
            verdict_path=verdict_path,
            manifest_path=manifest_path,
        )
        assert cap == 0.005

    def test_go_verdict_manifest_match_clears_gate(self, tmp_path):
        """7/7 GO + matching manifest → gate cleared (returns None)."""
        yaml_path = _make_runtime_yaml(tmp_path)
        manifest_path = _make_manifest(tmp_path)

        # Compute the manifest hash the same way activation_window does
        import hashlib
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()[:16]

        verdict_path = _make_verdict(
            tmp_path, verdict="GO", passed=7, manifest_hash=manifest_hash
        )
        cap = get_scale_gate_cap(
            runtime_yaml=yaml_path,
            verdict_path=verdict_path,
            manifest_path=manifest_path,
        )
        assert cap is None  # Gate cleared!

    def test_go_verdict_manifest_drift_keeps_cap(self, tmp_path):
        """7/7 GO but manifest changed → cap stays."""
        yaml_path = _make_runtime_yaml(tmp_path)
        manifest_path = _make_manifest(tmp_path)

        # Use a different hash than current manifest
        verdict_path = _make_verdict(
            tmp_path, verdict="GO", passed=7, manifest_hash="stale_hash_000"
        )
        cap = get_scale_gate_cap(
            runtime_yaml=yaml_path,
            verdict_path=verdict_path,
            manifest_path=manifest_path,
        )
        assert cap == 0.005  # Manifest drift → cap stays

    def test_incomplete_gates_keeps_cap(self, tmp_path):
        """GO verdict but only 6/7 gates → cap stays."""
        yaml_path = _make_runtime_yaml(tmp_path)
        manifest_path = _make_manifest(tmp_path)

        import hashlib
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()[:16]

        verdict_path = _make_verdict(
            tmp_path, verdict="GO", passed=6, total_gates=7, manifest_hash=manifest_hash
        )
        cap = get_scale_gate_cap(
            runtime_yaml=yaml_path,
            verdict_path=verdict_path,
            manifest_path=manifest_path,
        )
        assert cap == 0.005

    def test_custom_cap_value(self, tmp_path):
        """Scale gate uses per_trade_nav_pct from config."""
        yaml_path = _make_runtime_yaml(tmp_path, {"per_trade_nav_pct": 0.01})
        verdict_path = tmp_path / "nonexistent_verdict.json"
        cap = get_scale_gate_cap(
            runtime_yaml=yaml_path,
            verdict_path=verdict_path,
        )
        assert cap == 0.01


# ---------------------------------------------------------------------------
# verify_production_scale_eligible Tests
# ---------------------------------------------------------------------------

class TestVerifyProductionScaleEligible:
    """Test the ops-facing production scale eligibility check."""

    def test_no_verdict_not_eligible(self, tmp_path):
        """No verdict file → not eligible."""
        result = verify_production_scale_eligible(
            manifest_path=_make_manifest(tmp_path),
            verdict_path=tmp_path / "missing.json",
        )
        assert not result["eligible"]
        assert result["reason"] == "no_verification_verdict"

    def test_extend_verdict_not_eligible(self, tmp_path):
        """EXTEND verdict → not eligible."""
        manifest_path = _make_manifest(tmp_path)
        verdict_path = _make_verdict(tmp_path, verdict="EXTEND", passed=6)
        result = verify_production_scale_eligible(
            manifest_path=manifest_path,
            verdict_path=verdict_path,
        )
        assert not result["eligible"]
        assert "EXTEND" in result["reason"]

    def test_go_with_manifest_match_eligible(self, tmp_path):
        """7/7 GO + manifest match → eligible."""
        manifest_path = _make_manifest(tmp_path)
        import hashlib
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()[:16]
        verdict_path = _make_verdict(
            tmp_path, verdict="GO", passed=7, manifest_hash=manifest_hash
        )
        result = verify_production_scale_eligible(
            manifest_path=manifest_path,
            verdict_path=verdict_path,
        )
        assert result["eligible"]
        assert result["reason"] == "production_scale_authorized"
        assert result["manifest_match"]

    def test_go_with_manifest_drift_not_eligible(self, tmp_path):
        """7/7 GO but manifest drift → not eligible."""
        manifest_path = _make_manifest(tmp_path)
        verdict_path = _make_verdict(
            tmp_path, verdict="GO", passed=7, manifest_hash="old_hash"
        )
        result = verify_production_scale_eligible(
            manifest_path=manifest_path,
            verdict_path=verdict_path,
        )
        assert not result["eligible"]
        assert "manifest_drift" in result["reason"]


# ---------------------------------------------------------------------------
# Verdict Recording Tests
# ---------------------------------------------------------------------------

class TestVerdictRecording:
    """Test record_verification_verdict persistence and DLE event."""

    def test_records_verdict_to_file(self, tmp_path):
        """Verdict is persisted with manifest hash."""
        verdict_path = tmp_path / "state" / "verdict.json"
        manifest_path = _make_manifest(tmp_path)

        test_result = {
            "verdict": "GO",
            "passed": 7,
            "total_gates": 7,
            "action": "Promote to Production",
        }

        with patch("execution.activation_window._emit_dle_lifecycle_event"):
            record_verification_verdict(
                test_result,
                verdict_path=verdict_path,
                manifest_path=manifest_path,
            )

        assert verdict_path.exists()
        data = json.loads(verdict_path.read_text())
        assert data["verdict"] == "GO"
        assert data["passed"] == 7
        assert data["manifest_hash"] is not None
        assert data["recorded_at"] is not None

    def test_emits_verified_dle_event(self, tmp_path):
        """record_verification_verdict emits VERIFIED DLE event."""
        verdict_path = tmp_path / "state" / "verdict.json"
        manifest_path = _make_manifest(tmp_path)

        emitted = []

        def capture(action, details):
            emitted.append((action, details))

        test_result = {
            "verdict": "GO",
            "passed": 7,
            "total_gates": 7,
            "action": "Promote to Production",
        }

        with patch("execution.activation_window._emit_dle_lifecycle_event", capture):
            record_verification_verdict(
                test_result,
                verdict_path=verdict_path,
                manifest_path=manifest_path,
            )

        assert len(emitted) == 1
        action, details = emitted[0]
        assert action == "VERIFIED"
        assert details["verdict"] == "GO"
        assert details["passed"] == 7
        assert details["total_gates"] == 7

    def test_verdict_roundtrip_with_scale_gate(self, tmp_path, monkeypatch):
        """Recorded verdict is usable by scale gate to clear cap."""
        yaml_path = _make_runtime_yaml(tmp_path)
        manifest_path = _make_manifest(tmp_path)
        verdict_path = tmp_path / "state" / "verdict.json"

        test_result = {
            "verdict": "GO",
            "passed": 7,
            "total_gates": 7,
            "action": "Promote to Production",
        }

        with patch("execution.activation_window._emit_dle_lifecycle_event"):
            record_verification_verdict(
                test_result,
                verdict_path=verdict_path,
                manifest_path=manifest_path,
            )

        # Now check scale gate — should be cleared
        cap = get_scale_gate_cap(
            runtime_yaml=yaml_path,
            verdict_path=verdict_path,
            manifest_path=manifest_path,
        )
        assert cap is None  # Gate cleared by GO verdict


# ---------------------------------------------------------------------------
# Integration: Scale Gate Survives Window Disable
# ---------------------------------------------------------------------------

class TestScaleGateSurvivesDisable:
    """The scale gate persists even when activation_window.enabled is false."""

    def test_gate_active_when_window_disabled(self, tmp_path):
        """Window disabled but no verdict → cap still active."""
        aw = {
            "enabled": False,  # Window is disabled
            "duration_days": 14,
            "per_trade_nav_pct": 0.005,
            "require_go_for_scale": True,
        }
        yaml_path = tmp_path / "runtime.yaml"
        yaml_path.write_text(yaml.dump({"activation_window": aw}))
        verdict_path = tmp_path / "nonexistent.json"

        cap = get_scale_gate_cap(
            runtime_yaml=yaml_path,
            verdict_path=verdict_path,
        )
        assert cap == 0.005  # Cap enforced even though window disabled

    def test_gate_cleared_after_go_even_when_disabled(self, tmp_path):
        """Window disabled + GO verdict + manifest match → gate cleared."""
        aw = {
            "enabled": False,
            "duration_days": 14,
            "per_trade_nav_pct": 0.005,
            "require_go_for_scale": True,
        }
        yaml_path = tmp_path / "runtime.yaml"
        yaml_path.write_text(yaml.dump({"activation_window": aw}))
        manifest_path = _make_manifest(tmp_path)

        import hashlib
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()[:16]
        verdict_path = _make_verdict(
            tmp_path, verdict="GO", passed=7, manifest_hash=manifest_hash
        )

        cap = get_scale_gate_cap(
            runtime_yaml=yaml_path,
            verdict_path=verdict_path,
            manifest_path=manifest_path,
        )
        assert cap is None  # Cleared!


# ---------------------------------------------------------------------------
# Constants Tests
# ---------------------------------------------------------------------------

class TestConstants:
    """Verify governance constants are correctly defined."""

    def test_structural_guard_event_type(self):
        assert STRUCTURAL_GUARD_EVENT_TYPE == "STRUCTURAL_GUARD"

    def test_guard_type(self):
        assert GUARD_TYPE == "ACTIVATION_WINDOW"

    def test_phase_id(self):
        assert PHASE_ID == "PHASE_C"

    def test_verdict_path(self):
        assert str(VERIFICATION_VERDICT_PATH) == "logs/state/activation_verification_verdict.json"
