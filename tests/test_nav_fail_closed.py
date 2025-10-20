import importlib
import json
import time
from pathlib import Path

import pytest

from execution import risk_limits


class _DummyLogger:
    def __init__(self) -> None:
        self.records = []

    def write(self, record):
        self.records.append(record)


def _write_snapshot(path: Path, *, age_seconds: float, sources_ok: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": time.time() - age_seconds,
        "nav_usd": 1234.56,
        "sources_ok": sources_ok,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _reload_with_tmpdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NAV_FRESHNESS_SECONDS", raising=False)
    monkeypatch.delenv("FAIL_CLOSED_ON_NAV_STALE", raising=False)
    importlib.reload(risk_limits)
    monkeypatch.setattr(risk_limits, "LOG_VETOES", _DummyLogger())


def test_nav_fresh_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _reload_with_tmpdir(tmp_path, monkeypatch)
    _write_snapshot(tmp_path / "cache" / "nav_confirmed.json", age_seconds=5, sources_ok=True)
    fresh = risk_limits.is_nav_fresh({}, threshold_s=90)
    assert fresh is True


def test_nav_stale_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _reload_with_tmpdir(tmp_path, monkeypatch)
    _write_snapshot(tmp_path / "cache" / "nav_confirmed.json", age_seconds=300, sources_ok=True)
    ok = risk_limits.enforce_nav_freshness_or_veto(
        {}, {}, {"nav_freshness_seconds": 90, "fail_closed_on_nav_stale": True}
    )
    assert ok is False


def test_nav_sources_unhealthy_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _reload_with_tmpdir(tmp_path, monkeypatch)
    _write_snapshot(tmp_path / "cache" / "nav_confirmed.json", age_seconds=1, sources_ok=False)
    fresh = risk_limits.is_nav_fresh({}, threshold_s=90)
    assert fresh is False
