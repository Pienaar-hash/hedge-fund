"""
Dataset Rollback Logger — Passive logging for rollback trigger detection.

This module logs potential rollback events without taking action.
Used for verification before enabling automatic rollback enforcement.

Reference: docs/DATASET_ROLLBACK_CLAUSE.md
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from execution.utils.dataset_registry import (
    DatasetInfo,
    DatasetState,
    DatasetTier,
    get_dataset_info,
    get_dataset_tier,
)

__all__ = [
    "RollbackAction",
    "RollbackTriggerType",
    "RollbackTrigger",
    "RollbackEvent",
    "log_rollback_event",
    "log_rollback_trigger",
    "would_trigger_rollback",
]


class RollbackAction(str, Enum):
    """Rollback action types per DATASET_ROLLBACK_CLAUSE.md"""
    DOWNGRADE = "DOWNGRADE"
    REVOKE = "REVOKE"
    SUBSTITUTE = "SUBSTITUTE"


class RollbackTriggerType(str, Enum):
    """Rollback trigger classification"""
    MANDATORY = "mandatory"
    DISCRETIONARY = "discretionary"
    FORBIDDEN = "forbidden"


@dataclass
class RollbackTrigger:
    """Details of what triggered a rollback consideration."""
    trigger_type: RollbackTriggerType
    reason: str
    details: str
    
    # Mandatory trigger reasons
    TEMPORAL_VIOLATION = "temporal_violation"
    REPLAY_DIVERGENCE = "replay_divergence"
    REGIME_CORRUPTION = "regime_corruption"
    LATENCY_BREACH = "latency_breach"
    SILENT_GAP = "silent_gap"
    
    # Discretionary trigger reasons
    QUALITY_DEGRADATION = "quality_degradation"
    LATENCY_DRIFT = "latency_drift"
    PARTIAL_OUTAGE = "partial_outage"
    UPSTREAM_DEPRECATION = "upstream_deprecation"


@dataclass
class RollbackEvent:
    """Full rollback event record."""
    ts: str
    dataset_id: str
    action: RollbackAction
    from_state: DatasetState
    to_state: DatasetState
    trigger: RollbackTrigger
    automatic: bool
    operator: Optional[str]
    justification: Optional[str]
    fallback_activated: bool
    fallback_dataset: Optional[str]
    cycle_id: Optional[str]
    dry_run: bool  # True = logged only, not enforced
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "dataset_id": self.dataset_id,
            "action": self.action.value,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "trigger": {
                "type": self.trigger.trigger_type.value,
                "reason": self.trigger.reason,
                "details": self.trigger.details,
            },
            "authority": {
                "automatic": self.automatic,
                "operator": self.operator,
                "justification": self.justification,
            },
            "fallback": {
                "activated": self.fallback_activated,
                "fallback_dataset": self.fallback_dataset,
            },
            "cycle_id": self.cycle_id,
            "dry_run": self.dry_run,
        }


def _rollback_log_path() -> str:
    """Path to rollback event log."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "logs",
        "execution",
        "dataset_rollback.jsonl",
    )


def _get_current_cycle_id() -> Optional[str]:
    """Get current cycle ID from state if available."""
    try:
        cycle_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "logs",
            "state",
            "engine_metadata.json",
        )
        if os.path.exists(cycle_path):
            with open(cycle_path) as f:
                data = json.load(f)
                return data.get("cycle_id")
    except Exception:
        pass
    return None


def log_rollback_event(event: RollbackEvent) -> None:
    """
    Append rollback event to JSONL log.
    
    Events are logged regardless of dry_run status.
    """
    log_path = _rollback_log_path()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    with open(log_path, "a") as f:
        f.write(json.dumps(event.to_dict()) + "\n")


def log_rollback_trigger(
    dataset_id: str,
    trigger: RollbackTrigger,
    proposed_action: RollbackAction = RollbackAction.DOWNGRADE,
    dry_run: bool = True,
) -> RollbackEvent:
    """
    Log a rollback trigger detection.
    
    This is the primary entry point for passive rollback monitoring.
    By default, dry_run=True means no enforcement, just logging.
    
    Returns the event that was logged.
    """
    info = get_dataset_info(dataset_id)
    
    # Determine target state based on action
    if proposed_action == RollbackAction.REVOKE:
        to_state = DatasetState.REJECTED
    elif proposed_action == RollbackAction.DOWNGRADE:
        # Step down one level
        if info.state == DatasetState.PRODUCTION_ELIGIBLE:
            to_state = DatasetState.RESEARCH_ONLY
        elif info.state == DatasetState.RESEARCH_ONLY:
            to_state = DatasetState.OBSERVE_ONLY
        else:
            to_state = DatasetState.REJECTED
    else:  # SUBSTITUTE
        to_state = info.state  # State unchanged, just fallback activated
    
    event = RollbackEvent(
        ts=datetime.now(timezone.utc).isoformat(),
        dataset_id=dataset_id,
        action=proposed_action,
        from_state=info.state,
        to_state=to_state,
        trigger=trigger,
        automatic=trigger.trigger_type == RollbackTriggerType.MANDATORY,
        operator=None,
        justification=None,
        fallback_activated=False,
        fallback_dataset=None,
        cycle_id=_get_current_cycle_id(),
        dry_run=dry_run,
    )
    
    log_rollback_event(event)
    return event


def would_trigger_rollback(
    dataset_id: str,
    trigger_reason: str,
    trigger_details: str = "",
) -> tuple[bool, Optional[RollbackTrigger], Optional[str]]:
    """
    Check if a condition would trigger rollback for a dataset.
    
    Returns:
        (would_trigger, trigger_if_yes, blocking_reason_if_forbidden)
    
    This does NOT log or take action - use for pre-flight checks.
    """
    tier = get_dataset_tier(dataset_id)
    
    # Mandatory triggers
    mandatory_reasons = {
        RollbackTrigger.TEMPORAL_VIOLATION,
        RollbackTrigger.REPLAY_DIVERGENCE,
        RollbackTrigger.REGIME_CORRUPTION,
        RollbackTrigger.LATENCY_BREACH,
        RollbackTrigger.SILENT_GAP,
    }
    
    # Discretionary triggers
    discretionary_reasons = {
        RollbackTrigger.QUALITY_DEGRADATION,
        RollbackTrigger.LATENCY_DRIFT,
        RollbackTrigger.PARTIAL_OUTAGE,
        RollbackTrigger.UPSTREAM_DEPRECATION,
    }
    
    # Determine trigger type
    if trigger_reason in mandatory_reasons:
        trigger_type = RollbackTriggerType.MANDATORY
    elif trigger_reason in discretionary_reasons:
        trigger_type = RollbackTriggerType.DISCRETIONARY
    else:
        # Unknown trigger reason
        return False, None, f"Unknown trigger reason: {trigger_reason}"
    
    trigger = RollbackTrigger(
        trigger_type=trigger_type,
        reason=trigger_reason,
        details=trigger_details,
    )
    
    # Check if rollback is forbidden mid-cycle for this tier
    if tier in (DatasetTier.EXISTENTIAL, DatasetTier.AUTHORITATIVE):
        if trigger_type == RollbackTriggerType.DISCRETIONARY:
            return False, trigger, f"Discretionary rollback forbidden for {tier.value} tier"
        # Mandatory triggers still apply, but require cycle boundary
        return True, trigger, f"Rollback allowed but requires cycle boundary for {tier.value} tier"
    
    # ADVISORY and OBSERVATIONAL can be rolled back immediately
    return True, trigger, None


# ============================================================
# Trigger Detection Helpers
# ============================================================

def detect_latency_breach(
    dataset_id: str,
    observed_p99_ms: float,
    characterized_p99_ms: float,
    threshold_multiplier: float = 3.0,
) -> Optional[RollbackTrigger]:
    """
    Detect if latency has breached the characterized bound.
    
    Returns trigger if breach detected, None otherwise.
    """
    if observed_p99_ms > characterized_p99_ms * threshold_multiplier:
        return RollbackTrigger(
            trigger_type=RollbackTriggerType.MANDATORY,
            reason=RollbackTrigger.LATENCY_BREACH,
            details=f"p99={observed_p99_ms}ms exceeds {threshold_multiplier}x characterized ({characterized_p99_ms}ms)",
        )
    return None


def detect_silent_gap(
    dataset_id: str,
    last_update_ts: datetime,
    max_gap_seconds: float,
) -> Optional[RollbackTrigger]:
    """
    Detect if dataset has gone silent beyond acceptable gap.
    
    Returns trigger if gap detected, None otherwise.
    """
    now = datetime.now(timezone.utc)
    gap_seconds = (now - last_update_ts).total_seconds()
    
    if gap_seconds > max_gap_seconds:
        return RollbackTrigger(
            trigger_type=RollbackTriggerType.MANDATORY,
            reason=RollbackTrigger.SILENT_GAP,
            details=f"No update for {gap_seconds:.1f}s (max={max_gap_seconds}s)",
        )
    return None


def detect_replay_divergence(
    dataset_id: str,
    expected_hash: str,
    actual_hash: str,
) -> Optional[RollbackTrigger]:
    """
    Detect if replay produces different output.
    
    Returns trigger if divergence detected, None otherwise.
    """
    if expected_hash != actual_hash:
        return RollbackTrigger(
            trigger_type=RollbackTriggerType.MANDATORY,
            reason=RollbackTrigger.REPLAY_DIVERGENCE,
            details=f"Expected hash {expected_hash[:16]}... got {actual_hash[:16]}...",
        )
    return None
