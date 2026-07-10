#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from execution.preflight import read_engine_metadata, state_health_report


STATE_DIR_DEFAULT = Path("logs/state")
ALLOWABLE_LAG_SECONDS_DEFAULT = 900.0
EXECUTION_HEALTH_PATH = "execution_health.json"


def _load_json(path: Path) -> Mapping[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, Mapping) else None


def _to_epoch_seconds(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 1e12:
            ts /= 1000.0
        return ts
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            return datetime.fromisoformat(text).timestamp()
        except Exception:
            return None
    return None


def _check_execution_health(state_dir: Path, allowable_lag_seconds: float) -> list[str]:
    issues: list[str] = []
    payload = _load_json(state_dir / EXECUTION_HEALTH_PATH)
    if payload is None:
        return [f"{EXECUTION_HEALTH_PATH}:missing_or_invalid"]
    updated_ts = _to_epoch_seconds(payload.get("updated_ts") or payload.get("ts"))
    if updated_ts is None:
        return [f"{EXECUTION_HEALTH_PATH}:missing_updated_ts"]
    age_seconds = time.time() - updated_ts
    if age_seconds > allowable_lag_seconds:
        issues.append(f"{EXECUTION_HEALTH_PATH}:stale:{age_seconds:.0f}s")
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repository smoke test for state health and executor telemetry")
    parser.add_argument("--state-dir", default=str(STATE_DIR_DEFAULT), help="Path to the local state directory")
    parser.add_argument(
        "--allowable-lag-seconds",
        type=float,
        default=ALLOWABLE_LAG_SECONDS_DEFAULT,
        help="Maximum allowed state/telemetry age before smoke fails",
    )
    args = parser.parse_args(argv)

    state_dir = Path(args.state_dir)
    health = state_health_report(
        state_dir=state_dir,
        allowable_lag_seconds=float(args.allowable_lag_seconds),
    )
    telemetry_issues = _check_execution_health(state_dir, float(args.allowable_lag_seconds))
    meta = read_engine_metadata(state_dir)

    issues = sum((list(values) for values in health.values()), [])
    issues.extend(telemetry_issues)

    version = meta.get("engine_version") or meta.get("version") or "unknown"
    updated_ts = meta.get("updated_ts") or meta.get("ts") or "n/a"

    print(f"engine_version={version} engine_metadata_updated_ts={updated_ts}")
    print(
        "state_health "
        f"missing={len(health['missing_files'])} "
        f"stale={len(health['stale_files'])} "
        f"schema={len(health['schema_violations'])} "
        f"cross={len(health['cross_surface_violations'])}"
    )
    print(f"telemetry execution_health_issues={len(telemetry_issues)}")

    if issues:
        for issue in issues:
            print(f"SMOKE_FAIL {issue}")
        return 1

    print("SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
