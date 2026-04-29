"""Schema contract test for logs/state/determinism.json."""

from __future__ import annotations

import json
import pathlib
import tempfile

import pytest

from execution.state_publish import write_determinism_state

pytestmark = [pytest.mark.integration]

REQUIRED_KEYS = [
    "determinism_status",
    "degraded",
    "held_degraded",
    "violations",
    "proc_read_failures",
    "updated_ts",
    "engine_version",
]

OPTIONAL_METRIC_KEYS = [
    "executor_swap_kb",
    "executor_rss_kb",
    "system_swap_used_mb",
    "mem_psi_avg10",
    "avail_mem_pct",
]


def _write_and_read(snapshot_dict: dict, proc_failures: int = 0) -> dict:
    with tempfile.TemporaryDirectory() as td:
        state_dir = pathlib.Path(td)
        write_determinism_state(snapshot_dict, proc_failures, state_dir)
        path = state_dir / "determinism.json"
        assert path.exists(), "determinism.json was not created"
        return json.loads(path.read_text(encoding="utf-8"))


class TestDeterminismStateSchema:
    def test_ok_state_has_required_keys(self):
        snap = {
            "ts": 1700000000.0,
            "degraded": False,
            "violations": [],
            "executor_swap_kb": 0,
            "executor_rss_kb": 250000,
            "system_swap_used_mb": 50,
            "mem_psi_avg10": 0.0,
            "avail_mem_pct": 55,
            "held_degraded": False,
        }
        data = _write_and_read(snap)
        for key in REQUIRED_KEYS:
            assert key in data, f"Missing required key: {key}"
        assert data["determinism_status"] == "OK"
        assert data["degraded"] is False
        assert data["held_degraded"] is False
        assert data["violations"] == []

    def test_degraded_state(self):
        snap = {
            "ts": 1700000000.0,
            "degraded": True,
            "violations": ["EXECUTOR_SWAP: 50MB swapped (threshold: 10MB)"],
            "executor_swap_kb": 51200,
            "executor_rss_kb": 250000,
            "system_swap_used_mb": 800,
            "mem_psi_avg10": 0.5,
            "avail_mem_pct": 20,
            "held_degraded": False,
        }
        data = _write_and_read(snap)
        assert data["determinism_status"] == "DEGRADED"
        assert data["degraded"] is True
        assert len(data["violations"]) == 1

    def test_held_degraded_state(self):
        snap = {
            "ts": 1700000000.0,
            "degraded": True,
            "violations": ["HYSTERESIS_HOLD: clean for 20s (need 60s)"],
            "executor_swap_kb": 0,
            "executor_rss_kb": 250000,
            "system_swap_used_mb": 0,
            "mem_psi_avg10": 0.0,
            "avail_mem_pct": 55,
            "held_degraded": True,
        }
        data = _write_and_read(snap)
        assert data["determinism_status"] == "HELD_DEGRADED"
        assert data["degraded"] is True
        assert data["held_degraded"] is True

    def test_proc_read_failures_passed_through(self):
        snap = {
            "ts": 1700000000.0,
            "degraded": False,
            "violations": [],
            "held_degraded": False,
        }
        data = _write_and_read(snap, proc_failures=42)
        assert data["proc_read_failures"] == 42

    def test_metric_keys_present_when_provided(self):
        snap = {
            "ts": 1700000000.0,
            "degraded": False,
            "violations": [],
            "executor_swap_kb": 0,
            "executor_rss_kb": 250000,
            "system_swap_used_mb": 50,
            "mem_psi_avg10": 0.1,
            "avail_mem_pct": 55,
            "held_degraded": False,
        }
        data = _write_and_read(snap)
        for key in OPTIONAL_METRIC_KEYS:
            assert key in data, f"Missing metric key: {key}"

    def test_valid_json_round_trip(self):
        """Output must be valid JSON that round-trips cleanly."""
        snap = {
            "ts": 1700000000.0,
            "degraded": True,
            "violations": ["test_violation"],
            "held_degraded": False,
        }
        data = _write_and_read(snap)
        # Re-serialize and deserialize to verify
        reserialized = json.loads(json.dumps(data))
        assert reserialized == data
