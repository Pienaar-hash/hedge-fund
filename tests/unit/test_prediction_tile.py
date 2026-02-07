# tests/unit/test_prediction_tile.py
"""
Tests for dashboard/components/prediction_tile.py — prediction telemetry loader.

Validates:
    - JSONL parsing for alert_ranking.jsonl
    - JSONL parsing for firewall_denials.jsonl
    - Ranking rate calculation
    - Missing snapshot count
    - Last timestamp extraction
    - Graceful fallbacks on missing/corrupt files
    - Phase/enabled from env vars
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadPredictionTelemetry:
    """Tests for load_prediction_telemetry()."""

    def test_empty_dir(self, tmp_path: Path):
        from dashboard.components.prediction_tile import load_prediction_telemetry
        state = load_prediction_telemetry(log_dir=tmp_path)
        assert state["ranking_total"] == 0
        assert state["ranking_applied"] == 0
        assert state["ranking_rate"] == 0.0
        assert state["no_snapshot_count"] == 0
        assert state["last_ranking_ts"] is None
        assert state["firewall_denials"] == 0
        assert state["firewall_by_verdict"] == {}

    def test_ranking_counts(self, tmp_path: Path):
        from dashboard.components.prediction_tile import load_prediction_telemetry
        records = [
            {"rankings_applied": True, "reason": "Ranked 2 alerts", "ts": "2026-02-06T10:00:00Z"},
            {"rankings_applied": False, "reason": "No prediction snapshots", "ts": "2026-02-06T10:01:00Z"},
            {"rankings_applied": True, "reason": "Ranked 3 alerts", "ts": "2026-02-06T10:02:00Z"},
        ]
        _write_jsonl(tmp_path / "alert_ranking.jsonl", records)
        state = load_prediction_telemetry(log_dir=tmp_path)
        assert state["ranking_total"] == 3
        assert state["ranking_applied"] == 2
        assert abs(state["ranking_rate"] - 2 / 3) < 0.01
        assert state["no_snapshot_count"] == 1
        assert state["last_ranking_ts"] == "2026-02-06T10:02:00Z"

    def test_firewall_denials(self, tmp_path: Path):
        from dashboard.components.prediction_tile import load_prediction_telemetry
        records = [
            {"verdict": "DENIED_PHASE", "consumer": "x"},
            {"verdict": "DENIED_BLACKLISTED", "consumer": "y"},
            {"verdict": "DENIED_PHASE", "consumer": "z"},
        ]
        _write_jsonl(tmp_path / "firewall_denials.jsonl", records)
        state = load_prediction_telemetry(log_dir=tmp_path)
        assert state["firewall_denials"] == 3
        assert state["firewall_by_verdict"]["DENIED_PHASE"] == 2
        assert state["firewall_by_verdict"]["DENIED_BLACKLISTED"] == 1

    def test_corrupt_jsonl_lines_skipped(self, tmp_path: Path):
        from dashboard.components.prediction_tile import load_prediction_telemetry
        log = tmp_path / "alert_ranking.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as f:
            f.write('{"rankings_applied": true, "ts": "2026-02-06T10:00:00Z"}\n')
            f.write("NOT VALID JSON\n")
            f.write('{"rankings_applied": false, "ts": "2026-02-06T10:01:00Z"}\n')
        state = load_prediction_telemetry(log_dir=tmp_path)
        assert state["ranking_total"] == 2
        assert state["ranking_applied"] == 1

    def test_phase_from_env(self, tmp_path: Path, monkeypatch):
        from dashboard.components.prediction_tile import load_prediction_telemetry
        monkeypatch.setenv("PREDICTION_PHASE", "P1_ADVISORY")
        monkeypatch.setenv("PREDICTION_DLE_ENABLED", "1")
        state = load_prediction_telemetry(log_dir=tmp_path)
        assert state["phase"] == "P1_ADVISORY"
        assert state["enabled"] is True

    def test_phase_default(self, tmp_path: Path, monkeypatch):
        from dashboard.components.prediction_tile import load_prediction_telemetry
        monkeypatch.delenv("PREDICTION_PHASE", raising=False)
        monkeypatch.delenv("PREDICTION_DLE_ENABLED", raising=False)
        state = load_prediction_telemetry(log_dir=tmp_path)
        assert state["phase"] == "P0_OBSERVE"
        assert state["enabled"] is False

    def test_all_rankings_applied(self, tmp_path: Path):
        from dashboard.components.prediction_tile import load_prediction_telemetry
        records = [
            {"rankings_applied": True, "reason": "Ranked", "ts": "2026-02-06T10:00:00Z"},
            {"rankings_applied": True, "reason": "Ranked", "ts": "2026-02-06T10:01:00Z"},
        ]
        _write_jsonl(tmp_path / "alert_ranking.jsonl", records)
        state = load_prediction_telemetry(log_dir=tmp_path)
        assert state["ranking_rate"] == 1.0
        assert state["no_snapshot_count"] == 0

    def test_zero_rankings_applied(self, tmp_path: Path):
        from dashboard.components.prediction_tile import load_prediction_telemetry
        records = [
            {"rankings_applied": False, "reason": "No prediction snapshots", "ts": "2026-02-06T10:00:00Z"},
        ]
        _write_jsonl(tmp_path / "alert_ranking.jsonl", records)
        state = load_prediction_telemetry(log_dir=tmp_path)
        assert state["ranking_rate"] == 0.0
        assert state["no_snapshot_count"] == 1

    def test_missing_ranking_file_doesnt_crash(self, tmp_path: Path):
        """Firewall log exists but ranking log doesn't."""
        from dashboard.components.prediction_tile import load_prediction_telemetry
        _write_jsonl(tmp_path / "firewall_denials.jsonl", [{"verdict": "DENIED_PHASE"}])
        state = load_prediction_telemetry(log_dir=tmp_path)
        assert state["ranking_total"] == 0
        assert state["firewall_denials"] == 1


class TestFormatTs:
    """Tests for _format_ts() display helper."""

    def test_none(self):
        from dashboard.components.prediction_tile import _format_ts
        assert _format_ts(None) == "—"

    def test_invalid(self):
        from dashboard.components.prediction_tile import _format_ts
        result = _format_ts("not-a-date")
        assert isinstance(result, str)  # doesn't crash
