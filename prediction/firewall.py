# prediction/firewall.py
"""
Prediction Influence Firewall — the single enforcement point that prevents
prediction outputs from leaking authority into execution.

Invariant (hard-coded, not configurable):
    Prediction outputs are OBSERVATIONAL by default and cannot influence
    Doctrine, Sentinel-X, sizing, exits, or router decisions unless:
        1. The prediction dataset state is PRODUCTION_ELIGIBLE
        2. The prediction phase is P2_PRODUCTION
        3. The consumer explicitly declares its intent via this function

This is Dataset Admission Gate language applied to *consumption*, not just
ingestion.  Every downstream module that wants to read prediction aggregates
MUST call ``request_advisory()`` — there is no other path.

If the firewall denies access, it logs the denial to
``logs/prediction/firewall_denials.jsonl`` so we can prove authority
containment in post-hoc audit.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DENIALS_LOG = _REPO_ROOT / "logs" / "prediction" / "firewall_denials.jsonl"

# Phase must be at least P2 for any production influence
_PRODUCTION_PHASES = frozenset({"P2_PRODUCTION"})

# Phases where advisory-only consumers are allowed to read outputs.
# P0_OBSERVE blocks all consumers.  P1_ADVISORY allows advisory-only reads.
_ADVISORY_PHASES = frozenset({"P1_ADVISORY", "P2_PRODUCTION"})

# Consumers that are NEVER allowed to use prediction outputs, regardless of
# dataset state or phase.  This is a hard blacklist — Doctrine supremacy.
_BLACKLISTED_CONSUMERS = frozenset({
    "doctrine_kernel",
    "sentinel_x",
})

# Consumers that may read prediction outputs ONLY as advisory (no authority)
_ADVISORY_ONLY_CONSUMERS = frozenset({
    "router_health",
    "symbol_prioritization",
    "alert_ranking",
    "research",
    "dashboard",
})


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class FirewallVerdict(str, Enum):
    ALLOWED = "ALLOWED"
    DENIED_PHASE = "DENIED_PHASE"
    DENIED_DATASET = "DENIED_DATASET"
    DENIED_BLACKLISTED = "DENIED_BLACKLISTED"
    DENIED_DISABLED = "DENIED_DISABLED"


@dataclass(frozen=True)
class AdvisoryPayload:
    """Typed payload returned to consumers when firewall allows access."""
    question_id: str
    probs: Dict[str, float]
    aggregate_hash: str
    aggregate_ts: str
    dataset_states: Dict[str, str]
    phase: str
    advisory_only: bool       # True = consumer must treat as non-authoritative


@dataclass(frozen=True)
class FirewallResult:
    """Result of a firewall check — either an advisory payload or a denial."""
    verdict: FirewallVerdict
    payload: Optional[AdvisoryPayload]
    reason: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_denial(
    consumer: str,
    question_id: str,
    verdict: FirewallVerdict,
    reason: str,
    log_path: Optional[Path] = None,
) -> None:
    """Append a denial record to the firewall denials log. Fail-open."""
    path = log_path or _DENIALS_LOG
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": _utc_now_iso(),
            "consumer": consumer,
            "question_id": question_id,
            "verdict": verdict.value,
            "reason": reason,
        }
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"[prediction.firewall] denial log failed: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def request_advisory(
    *,
    consumer: str,
    question_id: str,
    probs: Dict[str, float],
    aggregate_hash: str = "",
    aggregate_ts: str = "",
    dataset_states: Optional[Dict[str, str]] = None,
    phase: str = "P0_OBSERVE",
    enabled: Optional[bool] = None,
    denial_log_path: Optional[Path] = None,
) -> FirewallResult:
    """
    The ONLY entry point for consuming prediction outputs.

    Returns FirewallResult with:
        - verdict: ALLOWED or DENIED_*
        - payload: AdvisoryPayload if allowed, None if denied
        - reason: human-readable explanation

    Consumers MUST check ``result.verdict == FirewallVerdict.ALLOWED`` before
    using the payload.  Any other pattern is a bug.

    Authority rules:
        1. If prediction layer is disabled → DENIED_DISABLED
        2. If consumer is blacklisted → DENIED_BLACKLISTED (always, no override)
        3. If phase < P2_PRODUCTION → allowed as advisory-only (dashboard/research)
        4. If phase == P2_PRODUCTION but any source dataset is not PRODUCTION_ELIGIBLE
           → DENIED_DATASET
        5. Otherwise → ALLOWED with advisory_only flag set appropriately
    """
    is_enabled = enabled if enabled is not None else (
        os.environ.get("PREDICTION_DLE_ENABLED", "0") == "1"
    )

    # Gate 1: master switch
    if not is_enabled:
        _append_denial(consumer, question_id, FirewallVerdict.DENIED_DISABLED,
                       "Prediction layer disabled", denial_log_path)
        return FirewallResult(
            verdict=FirewallVerdict.DENIED_DISABLED,
            payload=None,
            reason="Prediction layer disabled (PREDICTION_DLE_ENABLED=0)",
        )

    # Gate 2: hard blacklist (Doctrine supremacy)
    if consumer in _BLACKLISTED_CONSUMERS:
        _append_denial(consumer, question_id, FirewallVerdict.DENIED_BLACKLISTED,
                       f"Consumer '{consumer}' is blacklisted — prediction outputs "
                       f"can NEVER influence {consumer}", denial_log_path)
        return FirewallResult(
            verdict=FirewallVerdict.DENIED_BLACKLISTED,
            payload=None,
            reason=f"Consumer '{consumer}' is permanently blacklisted from prediction outputs",
        )

    ds_states = dataset_states or {}

    # Gate 3: phase check — P0 blocks all consumers, P1 allows advisory-only
    is_production = phase in _PRODUCTION_PHASES
    is_advisory = phase in _ADVISORY_PHASES

    if not is_advisory:
        # P0_OBSERVE: no consumers allowed — observe-only means no consumption
        _append_denial(consumer, question_id, FirewallVerdict.DENIED_PHASE,
                       f"Phase '{phase}' does not allow consumers (P0 = observe-only)",
                       denial_log_path)
        return FirewallResult(
            verdict=FirewallVerdict.DENIED_PHASE,
            payload=None,
            reason=f"Phase '{phase}' is observe-only — no consumers allowed",
        )

    advisory_only = not is_production or consumer in _ADVISORY_ONLY_CONSUMERS

    # Gate 4: in production phase, ALL contributing datasets must be PRODUCTION_ELIGIBLE
    if is_production:
        non_eligible = {
            ds_id: state for ds_id, state in ds_states.items()
            if state != "PRODUCTION_ELIGIBLE"
        }
        if non_eligible:
            reason = (f"Production phase but datasets not PRODUCTION_ELIGIBLE: "
                      f"{non_eligible}")
            _append_denial(consumer, question_id, FirewallVerdict.DENIED_DATASET,
                           reason, denial_log_path)
            return FirewallResult(
                verdict=FirewallVerdict.DENIED_DATASET,
                payload=None,
                reason=reason,
            )

    # All gates passed
    payload = AdvisoryPayload(
        question_id=question_id,
        probs=probs,
        aggregate_hash=aggregate_hash,
        aggregate_ts=aggregate_ts,
        dataset_states=ds_states,
        phase=phase,
        advisory_only=advisory_only,
    )

    return FirewallResult(
        verdict=FirewallVerdict.ALLOWED,
        payload=payload,
        reason="Allowed" + (" (advisory-only)" if advisory_only else ""),
    )


def is_consumer_allowed(consumer: str) -> bool:
    """Quick check — is this consumer even eligible to request prediction data?"""
    return consumer not in _BLACKLISTED_CONSUMERS
