# prediction/rollback_triggers.py
"""
Rollback trigger hooks for the prediction belief layer.

Implements the rollback clause from the Dataset Admission Gate:
    - temporal restatement detected  → revoke
    - replay divergence              → revoke
    - latency breach                 → downgrade
    - silent gap                     → downgrade
    - regime authority corruption    → revoke

In P0 these hooks only LOG events — they never modify dataset state
or block ingestion.  The log evidence lets us later prove that a dataset
*would have been* revoked, which is the Dataset Admission Gate's
"survivability before usefulness" principle.

All events are appended to ``logs/prediction/rollback_triggers.jsonl``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ROLLBACK_LOG = _REPO_ROOT / "logs" / "prediction" / "rollback_triggers.jsonl"

# Thresholds (P0 defaults — observe and log, don't act)
DEFAULT_LATENCY_BREACH_MS = 5000       # p99 > 5s triggers a log
DEFAULT_GAP_SILENCE_SECONDS = 3600     # 1h without updates triggers a log
DEFAULT_REPLAY_TOLERANCE = 1e-9        # bit-identical threshold


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class TriggerType(str, Enum):
    TEMPORAL_RESTATEMENT = "TEMPORAL_RESTATEMENT"
    REPLAY_DIVERGENCE = "REPLAY_DIVERGENCE"
    LATENCY_BREACH = "LATENCY_BREACH"
    SILENT_GAP = "SILENT_GAP"
    REGIME_CORRUPTION = "REGIME_CORRUPTION"


class SuggestedAction(str, Enum):
    REVOKE = "REVOKE"          # dataset should be REJECTED
    DOWNGRADE = "DOWNGRADE"    # dataset should move to OBSERVE_ONLY
    LOG_ONLY = "LOG_ONLY"      # just record, no state change


@dataclass(frozen=True)
class RollbackTriggerEvent:
    """A single rollback trigger detection event."""
    ts: str
    trigger_type: str           # TriggerType value
    dataset_id: str
    suggested_action: str       # SuggestedAction value
    severity: str               # HIGH | MEDIUM | LOW
    evidence: Dict[str, Any]
    phase: str                  # P0_OBSERVE | P1_RESEARCH | P2_PRODUCTION
    enforced: bool              # False in P0 (log-only)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_trigger(event: RollbackTriggerEvent, log_path: Optional[Path] = None) -> None:
    """Append a rollback trigger event to JSONL. Fail-open."""
    path = log_path or _ROLLBACK_LOG
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"[rollback_triggers] log write failed: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Trigger detectors
# ---------------------------------------------------------------------------

def check_temporal_restatement(
    *,
    dataset_id: str,
    original_ts: str,
    restated_ts: str,
    original_value: float,
    restated_value: float,
    phase: str = "P0_OBSERVE",
    log_path: Optional[Path] = None,
) -> Optional[RollbackTriggerEvent]:
    """
    Detect if a dataset has restated historical data.

    Rollback clause: "if the feed restates history, revoke."
    """
    if original_ts != restated_ts:
        return None  # Different timestamps are updates, not restatements

    if abs(original_value - restated_value) < DEFAULT_REPLAY_TOLERANCE:
        return None  # Within tolerance

    event = RollbackTriggerEvent(
        ts=_utc_now_iso(),
        trigger_type=TriggerType.TEMPORAL_RESTATEMENT.value,
        dataset_id=dataset_id,
        suggested_action=SuggestedAction.REVOKE.value,
        severity="HIGH",
        evidence={
            "original_ts": original_ts,
            "restated_ts": restated_ts,
            "original_value": original_value,
            "restated_value": restated_value,
            "delta": abs(original_value - restated_value),
        },
        phase=phase,
        enforced=False,  # P0: log only
    )
    _append_trigger(event, log_path)
    return event


def check_replay_divergence(
    *,
    dataset_id: str,
    expected_hash: str,
    actual_hash: str,
    context: Optional[Dict[str, Any]] = None,
    phase: str = "P0_OBSERVE",
    log_path: Optional[Path] = None,
) -> Optional[RollbackTriggerEvent]:
    """
    Detect replay divergence — same inputs produced different outputs.

    Rollback clause: determinism violation triggers revoke.
    """
    if expected_hash == actual_hash:
        return None

    event = RollbackTriggerEvent(
        ts=_utc_now_iso(),
        trigger_type=TriggerType.REPLAY_DIVERGENCE.value,
        dataset_id=dataset_id,
        suggested_action=SuggestedAction.REVOKE.value,
        severity="HIGH",
        evidence={
            "expected_hash": expected_hash,
            "actual_hash": actual_hash,
            **(context or {}),
        },
        phase=phase,
        enforced=False,
    )
    _append_trigger(event, log_path)
    return event


def check_latency_breach(
    *,
    dataset_id: str,
    latency_ms: float,
    threshold_ms: float = DEFAULT_LATENCY_BREACH_MS,
    phase: str = "P0_OBSERVE",
    log_path: Optional[Path] = None,
) -> Optional[RollbackTriggerEvent]:
    """
    Detect latency breach — source response time exceeds threshold.

    Rollback clause: "if latency explodes, downgrade."
    """
    if latency_ms <= threshold_ms:
        return None

    event = RollbackTriggerEvent(
        ts=_utc_now_iso(),
        trigger_type=TriggerType.LATENCY_BREACH.value,
        dataset_id=dataset_id,
        suggested_action=SuggestedAction.DOWNGRADE.value,
        severity="MEDIUM",
        evidence={
            "latency_ms": latency_ms,
            "threshold_ms": threshold_ms,
            "breach_ratio": round(latency_ms / threshold_ms, 2),
        },
        phase=phase,
        enforced=False,
    )
    _append_trigger(event, log_path)
    return event


def check_silent_gap(
    *,
    dataset_id: str,
    last_event_ts: float,
    now_ts: Optional[float] = None,
    gap_threshold_seconds: float = DEFAULT_GAP_SILENCE_SECONDS,
    phase: str = "P0_OBSERVE",
    log_path: Optional[Path] = None,
) -> Optional[RollbackTriggerEvent]:
    """
    Detect silent gap — no events received for longer than threshold.

    Rollback clause: silent data loss triggers downgrade.
    """
    current = now_ts or time.time()
    gap = current - last_event_ts

    if gap <= gap_threshold_seconds:
        return None

    event = RollbackTriggerEvent(
        ts=_utc_now_iso(),
        trigger_type=TriggerType.SILENT_GAP.value,
        dataset_id=dataset_id,
        suggested_action=SuggestedAction.DOWNGRADE.value,
        severity="MEDIUM",
        evidence={
            "last_event_ts": last_event_ts,
            "current_ts": current,
            "gap_seconds": round(gap, 1),
            "threshold_seconds": gap_threshold_seconds,
        },
        phase=phase,
        enforced=False,
    )
    _append_trigger(event, log_path)
    return event


def check_regime_corruption(
    *,
    dataset_id: str,
    evidence: Dict[str, Any],
    phase: str = "P0_OBSERVE",
    log_path: Optional[Path] = None,
) -> RollbackTriggerEvent:
    """
    Flag a dataset for corrupting regime authority.

    Rollback clause: "if it corrupts regime authority, revoke."
    This is always logged, always HIGH severity, always suggests REVOKE.
    """
    event = RollbackTriggerEvent(
        ts=_utc_now_iso(),
        trigger_type=TriggerType.REGIME_CORRUPTION.value,
        dataset_id=dataset_id,
        suggested_action=SuggestedAction.REVOKE.value,
        severity="HIGH",
        evidence=evidence,
        phase=phase,
        enforced=False,
    )
    _append_trigger(event, log_path)
    return event


# ---------------------------------------------------------------------------
# Batch check (convenience)
# ---------------------------------------------------------------------------

def run_health_checks(
    *,
    dataset_id: str,
    last_event_ts: Optional[float] = None,
    last_latency_ms: Optional[float] = None,
    phase: str = "P0_OBSERVE",
    log_path: Optional[Path] = None,
) -> List[RollbackTriggerEvent]:
    """
    Run all applicable rollback trigger checks for a dataset.
    Returns list of triggered events (empty = healthy).
    """
    triggered: List[RollbackTriggerEvent] = []

    if last_event_ts is not None:
        result = check_silent_gap(
            dataset_id=dataset_id,
            last_event_ts=last_event_ts,
            phase=phase,
            log_path=log_path,
        )
        if result:
            triggered.append(result)

    if last_latency_ms is not None:
        result = check_latency_breach(
            dataset_id=dataset_id,
            latency_ms=last_latency_ms,
            phase=phase,
            log_path=log_path,
        )
        if result:
            triggered.append(result)

    return triggered
