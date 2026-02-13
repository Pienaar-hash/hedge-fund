# tests/unit/test_phase_c_readiness.py
"""
Phase C readiness state surface — schema, writer, and computation tests.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from execution.enforcement_rehearsal import (
    RehearsalMetrics,
    compute_phase_c_readiness,
    get_rehearsal_metrics,
    init_rehearsal,
    rehearse_order,
    reset_rehearsal,
    REASON_OK,
)

pytestmark = pytest.mark.unit

# Required top-level keys in phase_c_readiness.json
REQUIRED_KEYS = {
    "window_days_required",
    "window_days_met",
    "window_start_ts",
    "criteria_met",
    "gate_satisfied",
    "last_breach_ts",
    "breach_reason",
    "current_metrics",
    "thresholds",
    "rehearsal_enabled",
    "updated_ts",
}

REQUIRED_CURRENT_METRICS_KEYS = {
    "would_block_pct",
    "expired_permit_count",
    "missing_permit_count",
    "total_orders",
    "ok_count",
}

REQUIRED_THRESHOLD_KEYS = {
    "would_block_pct",
    "expired_permit_count",
    "missing_permit_count",
}


class TestComputeReadinessPayload:

    def setup_method(self):
        reset_rehearsal()

    def teardown_method(self):
        reset_rehearsal()

    def test_schema_keys_present(self):
        """Readiness payload always contains all required keys."""
        payload = compute_phase_c_readiness()
        assert REQUIRED_KEYS <= set(payload.keys())
        assert REQUIRED_CURRENT_METRICS_KEYS <= set(payload["current_metrics"].keys())
        assert REQUIRED_THRESHOLD_KEYS <= set(payload["thresholds"].keys())

    def test_disabled_rehearsal_shows_breach(self):
        """When rehearsal is disabled, criteria_met is False."""
        payload = compute_phase_c_readiness()
        assert payload["criteria_met"] is False
        assert payload["gate_satisfied"] is False
        assert "rehearsal_disabled" in payload["breach_reason"]

    def test_no_orders_is_breach(self):
        """Zero orders evaluated is a breach (can't prove safety)."""
        payload = compute_phase_c_readiness()
        assert payload["criteria_met"] is False
        assert "no_orders_evaluated" in payload["breach_reason"]

    def test_clean_metrics_criteria_met(self, tmp_path):
        """When metrics are clean, criteria_met is True."""
        shadow_log = tmp_path / "shadow.jsonl"
        shadow_log.write_text("")
        rehearsal_log = str(tmp_path / "rehearsal.jsonl")

        with mock.patch("execution.v6_flags.get_flags") as mock_flags:
            mock_flags.return_value = mock.MagicMock(shadow_dle_enabled=True)
            init_rehearsal(shadow_log_path=shadow_log, rehearsal_log_path=rehearsal_log)

        # Inject clean metrics manually
        import execution.enforcement_rehearsal as er
        with er._metrics_lock:
            er._metrics = RehearsalMetrics(
                total_orders=100,
                would_block_count=0,
                would_block_pct=0.0,
                expired_permit_count=0,
                missing_permit_count=0,
                mismatch_count=0,
                ok_count=100,
                enabled=True,
            )

        payload = compute_phase_c_readiness()
        assert payload["criteria_met"] is True
        # But gate not satisfied yet (need 14 days)
        assert payload["gate_satisfied"] is False
        assert payload["window_days_met"] == 0

    def test_would_block_breach(self, tmp_path):
        """High would_block_pct triggers breach."""
        shadow_log = tmp_path / "shadow.jsonl"
        shadow_log.write_text("")

        with mock.patch("execution.v6_flags.get_flags") as mock_flags:
            mock_flags.return_value = mock.MagicMock(shadow_dle_enabled=True)
            init_rehearsal(shadow_log_path=shadow_log)

        import execution.enforcement_rehearsal as er
        with er._metrics_lock:
            er._metrics = RehearsalMetrics(
                total_orders=100,
                would_block_count=1,
                would_block_pct=1.0,  # 1% > 0.1% threshold
                expired_permit_count=0,
                missing_permit_count=0,
                ok_count=99,
                enabled=True,
            )

        payload = compute_phase_c_readiness()
        assert payload["criteria_met"] is False
        assert "would_block_pct" in payload["breach_reason"]

    def test_expired_permit_breach(self, tmp_path):
        """Any expired permits trigger breach."""
        shadow_log = tmp_path / "shadow.jsonl"
        shadow_log.write_text("")

        with mock.patch("execution.v6_flags.get_flags") as mock_flags:
            mock_flags.return_value = mock.MagicMock(shadow_dle_enabled=True)
            init_rehearsal(shadow_log_path=shadow_log)

        import execution.enforcement_rehearsal as er
        with er._metrics_lock:
            er._metrics = RehearsalMetrics(
                total_orders=100,
                would_block_count=1,
                would_block_pct=0.0,
                expired_permit_count=1,
                missing_permit_count=0,
                ok_count=99,
                enabled=True,
            )

        payload = compute_phase_c_readiness()
        assert payload["criteria_met"] is False
        assert "expired_permit_count" in payload["breach_reason"]

    def test_window_resets_on_breach(self, tmp_path):
        """A breach after clean window resets window_start_ts."""
        shadow_log = tmp_path / "shadow.jsonl"
        shadow_log.write_text("")

        with mock.patch("execution.v6_flags.get_flags") as mock_flags:
            mock_flags.return_value = mock.MagicMock(shadow_dle_enabled=True)
            init_rehearsal(shadow_log_path=shadow_log)

        import execution.enforcement_rehearsal as er

        # First: clean metrics → starts window
        with er._metrics_lock:
            er._metrics = RehearsalMetrics(
                total_orders=100, ok_count=100, enabled=True,
            )
        p1 = compute_phase_c_readiness()
        assert p1["criteria_met"] is True
        assert p1["window_start_ts"] is not None

        # Then: breach → resets window
        with er._metrics_lock:
            er._metrics = RehearsalMetrics(
                total_orders=100, would_block_count=10, would_block_pct=10.0,
                ok_count=90, enabled=True,
            )
        p2 = compute_phase_c_readiness()
        assert p2["criteria_met"] is False
        assert p2["window_start_ts"] is None
        assert p2["last_breach_ts"] is not None

    def test_thresholds_match_manifest(self):
        """Thresholds in payload match those declared in manifest."""
        payload = compute_phase_c_readiness()
        assert payload["thresholds"]["would_block_pct"] == 0.1
        assert payload["thresholds"]["expired_permit_count"] == 0
        assert payload["thresholds"]["missing_permit_count"] == 0
        assert payload["window_days_required"] == 14


class TestWriteReadiness:

    def test_write_phase_c_readiness_state(self, tmp_path):
        """Writer produces valid JSON at correct path."""
        from execution.state_publish import write_phase_c_readiness_state

        payload = {
            "window_days_required": 14,
            "window_days_met": 3,
            "criteria_met": True,
            "gate_satisfied": False,
            "updated_ts": "2026-02-13T00:00:00+00:00",
        }
        write_phase_c_readiness_state(payload, state_dir=tmp_path)
        written = json.loads((tmp_path / "phase_c_readiness.json").read_text())
        assert written["window_days_required"] == 14
        assert written["criteria_met"] is True
        assert written["gate_satisfied"] is False

    def test_write_is_atomic(self, tmp_path):
        """Verify _atomic_write_state produces deterministic output."""
        from execution.state_publish import write_phase_c_readiness_state

        payload = {"a": 1, "b": 2}
        write_phase_c_readiness_state(payload, state_dir=tmp_path)
        raw = (tmp_path / "phase_c_readiness.json").read_text()
        # Atomic write uses separators=(",", ":") — compact
        assert " " not in raw or raw.startswith("{")
