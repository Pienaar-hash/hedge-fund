# prediction/state_surface.py
"""
Minimal dashboard state surface for the prediction belief layer.

Prevents "shadow belief layer" by making the prediction layer's health
observable from ``logs/state/prediction_state.json``.

Exposes:
    - last aggregate timestamp per question
    - per-dataset admission state
    - constraint violation counts
    - rollback trigger counts
    - belief event counts
    - firewall denial counts
    - solver status distribution

This file is read-only from the dashboard's perspective, matching the
one-way dependency invariant (execution/ → logs/state/, dashboard reads).
"""

from __future__ import annotations

import json
import os
import pathlib
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_STATE_DIR = _REPO_ROOT / "logs" / "state"
_PREDICTION_STATE_FILE = "prediction_state.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path: pathlib.Path, payload: Any) -> None:
    """Write JSON atomically (temp + replace). Matches state_publish pattern."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"),
                      default=str)
        os.replace(tmp_path, str(path))
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _count_jsonl_lines(path: pathlib.Path) -> int:
    """Count lines in a JSONL file. Returns 0 if missing."""
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def _count_jsonl_by_field(path: pathlib.Path, field: str) -> Dict[str, int]:
    """Count occurrences of each value of *field* in a JSONL file."""
    counts: Dict[str, int] = {}
    if not path.exists():
        return counts
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    key = str(record.get(field, "UNKNOWN"))
                    counts[key] = counts.get(key, 0) + 1
                except (json.JSONDecodeError, KeyError):
                    continue
    except Exception:
        pass
    return counts


def _last_ts_from_jsonl(path: pathlib.Path) -> Optional[str]:
    """Read the ts from the last record in a JSONL file."""
    if not path.exists():
        return None
    try:
        last_line = ""
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                last_line = line
        if last_line:
            return json.loads(last_line).get("ts")
    except Exception:
        pass
    return None


def build_prediction_state(
    *,
    dataset_admission: Optional[Dict[str, Any]] = None,
    log_dir: Optional[pathlib.Path] = None,
) -> Dict[str, Any]:
    """
    Build the prediction layer state snapshot for dashboard consumption.

    Reads from:
        - config/dataset_admission.json (or passed in)
        - logs/prediction/*.jsonl (event counts, timestamps)

    Returns a dict suitable for writing to logs/state/prediction_state.json.
    """
    pred_log_dir = (log_dir or _REPO_ROOT / "logs") / "prediction"

    # Belief events
    belief_log = pred_log_dir / "belief_events.jsonl"
    belief_count = _count_jsonl_lines(belief_log)
    last_belief_ts = _last_ts_from_jsonl(belief_log)

    # Aggregates
    agg_log = pred_log_dir / "aggregate_state.jsonl"
    agg_count = _count_jsonl_lines(agg_log)
    last_agg_ts = _last_ts_from_jsonl(agg_log)

    # Solver status distribution
    solver_statuses = _count_jsonl_by_field(
        agg_log, "status"
    ) if agg_log.exists() else {}
    # Try to extract from solver sub-object
    if not solver_statuses and agg_log.exists():
        solver_statuses = {}
        try:
            with agg_log.open("r") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        s = rec.get("solver", {}).get("status", "UNKNOWN")
                        solver_statuses[s] = solver_statuses.get(s, 0) + 1
                    except Exception:
                        continue
        except Exception:
            pass

    # DLE prediction events (decisions/permits/vetoes)
    dle_log = pred_log_dir / "dle_prediction_events.jsonl"
    dle_event_types = _count_jsonl_by_field(dle_log, "event_type")

    # Rollback triggers
    rollback_log = pred_log_dir / "rollback_triggers.jsonl"
    rollback_count = _count_jsonl_lines(rollback_log)
    rollback_by_type = _count_jsonl_by_field(rollback_log, "trigger_type")

    # Firewall denials
    denial_log = pred_log_dir / "firewall_denials.jsonl"
    denial_count = _count_jsonl_lines(denial_log)
    denial_by_verdict = _count_jsonl_by_field(denial_log, "verdict")

    # Episodes
    episode_log = pred_log_dir / "prediction_episodes.jsonl"
    episode_count = _count_jsonl_lines(episode_log)
    episode_by_state = _count_jsonl_by_field(episode_log, "state")

    # Dataset states (prediction sources only)
    prediction_datasets: Dict[str, str] = {}
    if dataset_admission:
        for ds_id, ds_info in dataset_admission.get("datasets", {}).items():
            if ds_id.startswith("prediction_"):
                prediction_datasets[ds_id] = ds_info.get("state", "UNKNOWN")

    # Phase flag
    phase = "P0_OBSERVE"
    if os.environ.get("PREDICTION_PHASE"):
        phase = os.environ["PREDICTION_PHASE"]

    return {
        "updated_ts": _utc_now_iso(),
        "phase": phase,
        "enabled": os.environ.get("PREDICTION_DLE_ENABLED", "0") == "1",
        "belief_events": {
            "count": belief_count,
            "last_ts": last_belief_ts,
        },
        "aggregates": {
            "count": agg_count,
            "last_ts": last_agg_ts,
            "solver_statuses": solver_statuses,
        },
        "dle_events": dle_event_types,
        "episodes": {
            "count": episode_count,
            "by_state": episode_by_state,
        },
        "rollback_triggers": {
            "count": rollback_count,
            "by_type": rollback_by_type,
        },
        "firewall": {
            "denial_count": denial_count,
            "by_verdict": denial_by_verdict,
        },
        "dataset_states": prediction_datasets,
    }


def write_prediction_state(
    *,
    dataset_admission: Optional[Dict[str, Any]] = None,
    state_dir: Optional[pathlib.Path] = None,
    log_dir: Optional[pathlib.Path] = None,
) -> Dict[str, Any]:
    """
    Build and write prediction state to logs/state/prediction_state.json.

    Returns the payload written.
    """
    payload = build_prediction_state(
        dataset_admission=dataset_admission,
        log_dir=log_dir,
    )
    target_dir = state_dir or _STATE_DIR
    path = target_dir / _PREDICTION_STATE_FILE
    _atomic_write(path, payload)
    return payload
