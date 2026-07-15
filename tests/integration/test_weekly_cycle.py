from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _price_history():
    payload = {}
    start_dt = datetime(2025, 1, 6, 12, tzinfo=timezone.utc)
    for symbol, start, step in (
        ("BTCUSDT", 100.0, 4.0),
        ("ETHUSDT", 120.0, 3.0),
        ("SOLUSDT", 20.0, 1.2),
        ("LINKUSDT", 10.0, 0.4),
        ("DOGEUSDT", 0.2, 0.01),
        ):
        rows = []
        value = start
        for week in range(65):
            rows.append({"ts": (start_dt + timedelta(weeks=week)).isoformat(), "close": round(value, 6)})
            value += step
        payload[symbol] = rows
    return payload


def test_weekly_cycle_validate_only(tmp_path: Path):
    price_path = tmp_path / "prices.json"
    price_path.write_text(json.dumps(_price_history()), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_weekly_cycle.py",
            "--validate-only",
            "--prices-path",
            str(price_path),
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["schema_version"] == "weekly_momentum_v1"
    assert payload["risk_mode"] == "ACTIVE"
    assert len(payload["target_positions"]) <= 3


def test_legacy_flag_rejection(tmp_path: Path):
    price_path = tmp_path / "prices.json"
    legacy_path = tmp_path / "legacy.json"
    price_path.write_text(json.dumps(_price_history()), encoding="utf-8")
    legacy_path.write_text(json.dumps({"hydra_execution": {"enabled": True}}), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_weekly_cycle.py",
            "--validate-only",
            "--prices-path",
            str(price_path),
            "--legacy-config-path",
            str(legacy_path),
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
    )
    assert proc.returncode != 0
    assert "legacy strategy flags present" in proc.stderr or "legacy strategy flags present" in proc.stdout
