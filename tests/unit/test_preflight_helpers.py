from __future__ import annotations

from pathlib import Path

from execution.preflight import state_health_report, version_report


def test_state_health_report_detects_missing(tmp_path: Path) -> None:
    health = state_health_report(state_dir=tmp_path, allowable_lag_seconds=1)
    assert "nav_state" in health["missing_files"]
    assert "engine_metadata" in health["missing_files"]


def test_version_report_alignment() -> None:
    report = version_report("v7.6")
    assert report["engine_version"] == "v7.6"
    assert report["docs_version"] == "v7.6"
    assert report["aligned"]
