from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from dashboard.state_v7 import validate_surface_health
from execution.versioning import version_alignment

STATE_DIR_DEFAULT = Path("logs/state")


def state_health_report(
    state_dir: Path | str = STATE_DIR_DEFAULT,
    *,
    allowable_lag_seconds: float = 900.0,
) -> Dict[str, List[str]]:
    """Return missing/stale/schema/cross-surface issues for a state directory."""
    base = Path(state_dir)
    return validate_surface_health(state_dir=base, allowable_lag_seconds=allowable_lag_seconds)


def version_report(expected: str = "v7.6") -> Dict[str, Any]:
    """Summarize engine/docs version alignment."""
    engine_version, docs_version, aligned = version_alignment(expected)
    return {
        "engine_version": engine_version,
        "docs_version": docs_version,
        "expected": expected,
        "aligned": aligned,
    }


def assert_version_alignment(expected: str = "v7.6") -> None:
    report = version_report(expected)
    if not report["aligned"]:
        raise RuntimeError(
            f"VERSION mismatch: engine={report['engine_version']} docs={report['docs_version']} expected={expected}"
        )


def read_engine_metadata(state_dir: Path | str = STATE_DIR_DEFAULT) -> Dict[str, Any]:
    """Load engine_metadata.json if present; return {} on failure."""
    path = Path(state_dir) / "engine_metadata.json"
    try:
        if path.exists() and path.stat().st_size > 0:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, Mapping) else {}
    except Exception:
        return {}
    return {}


def run_step(cmd: Sequence[str], *, env: Mapping[str, str] | None = None) -> subprocess.CompletedProcess[Any]:
    """Run a shell command with check=False for preflight tasks."""
    return subprocess.run(cmd, check=False, env=dict(env) if env else None)
