# prediction/dle_prediction_gate.py
"""
DLE Gate for predictions — issues Decision and Permit objects for belief
writes, mirroring the execution DLE pattern in execution/dle_shadow.py.

Authority rules (parallel action set for predictions):
    WRITE_BELIEF          — standard belief update
    UPDATE_CONSTRAINTS    — rare, admin-only constraint mutations
    PROMOTE_DATASET_STATE — cycle-boundary dataset state transitions

A Decision specifies:
    - which dataset_ids or actors may write
    - which question_ids
    - max delta per update (prevents "pump")
    - TTL and max uses

A Permit binds a SINGLE BeliefEvent — single-use, short TTL, frozen snapshots.

Phase control:
    P0 (OBSERVE_ONLY)  — shadow permits, log only, no downstream
    P1 (RESEARCH_ONLY) — permits issued, aggregates computed, dashboard-only
    P2 (PRODUCTION)    — advisory hints, never overrides Sentinel-X

Invariants:
    - Deterministic IDs (SHA-256 derived, not random)
    - Fail-open: gate failures never crash the ingestion pipeline
    - All decisions + permits logged to JSONL
    - Shadow mode by default (matches execution DLE Phase A)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "dle_prediction_v1"
DEFAULT_LOG_PATH = "logs/prediction/dle_prediction_events.jsonl"

# Feature flags (env vars, mirrors execution/v6_flags.py pattern)
PREDICTION_DLE_ENABLED = os.environ.get("PREDICTION_DLE_ENABLED", "0") == "1"
PREDICTION_DLE_WRITE_LOGS = os.environ.get("PREDICTION_DLE_WRITE_LOGS", "1") == "1"

# Default policy
DEFAULT_POLICY_VERSION = "prediction_v1.0"
DEFAULT_PHASE_ID = "P0_OBSERVE"  # shadow/observe-only by default
DEFAULT_MAX_DELTA = 0.50         # max probability change per update
DEFAULT_TTL_SECONDS = 300        # 5-minute permit TTL
DEFAULT_MAX_USES = 1             # single-use permits


# ---------------------------------------------------------------------------
# Helpers (deterministic, mirrors dle_shadow.py)
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _hash_snapshot(name: str, obj: Any) -> str:
    if obj is None:
        return f"{name}_MISSING_" + _sha256_hex("MISSING")[:16]
    return f"{name}_" + _sha256_hex(_stable_json(obj))[:16]


# ---------------------------------------------------------------------------
# ID derivation (deterministic)
# ---------------------------------------------------------------------------

def derive_prediction_decision_id(
    *,
    phase_id: str,
    action_class: str,
    constraints: Dict[str, Any],
    policy_version: str,
) -> str:
    """Deterministic Decision ID for prediction authority."""
    payload = {
        "phase_id": phase_id,
        "action_class": action_class,
        "constraints": constraints,
        "policy_version": policy_version,
    }
    return "DEC_" + _sha256_hex(_stable_json(payload))[:12]


def derive_prediction_permit_id(
    *,
    decision_id: str,
    dataset_id: str,
    question_id: str,
    outcome_id: str,
    issued_at_iso: str,
) -> str:
    """Deterministic Permit ID for a single belief write."""
    payload = {
        "decision_id": decision_id,
        "dataset_id": dataset_id,
        "question_id": question_id,
        "outcome_id": outcome_id,
        "issued_at": issued_at_iso,
    }
    return "PRM_" + _sha256_hex(_stable_json(payload))[:12]


# ---------------------------------------------------------------------------
# Data objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PredictionDecision:
    """Authorizes a class of belief updates."""
    decision_id: str
    ts: str
    phase_id: str                    # P0_OBSERVE | P1_RESEARCH | P2_PRODUCTION
    action_class: str                # WRITE_BELIEF | UPDATE_CONSTRAINTS | PROMOTE_DATASET_STATE
    policy_version: str
    allowed_datasets: Tuple[str, ...]
    allowed_questions: Tuple[str, ...]
    max_delta: float                 # max probability change per update
    ttl_seconds: int
    max_uses: int
    constraints: Dict[str, Any]      # frozen policy constraints


@dataclass(frozen=True)
class PredictionPermit:
    """Single-use authorization for one belief event write."""
    permit_id: str
    ts: str
    decision_id: str
    dataset_id: str
    question_id: str
    outcome_id: str
    single_use: bool
    ttl_seconds: int
    snapshots: Dict[str, str]        # frozen state hashes


@dataclass(frozen=True)
class PredictionDLEEvent:
    """Envelope for all prediction DLE log entries."""
    schema_version: str
    event_type: str                  # DECISION | PERMIT | VETO
    ts: str
    payload: Dict[str, Any]


# ---------------------------------------------------------------------------
# Writer (fail-open, mirrors DLEShadowWriter)
# ---------------------------------------------------------------------------

class PredictionDLEWriter:
    """Append-only JSONL writer for prediction DLE events. Fail-open."""

    def __init__(self, log_path: str = DEFAULT_LOG_PATH) -> None:
        self.log_path = log_path
        self._write_failures = 0
        self._repo_root = Path(__file__).resolve().parent.parent

    def _resolve_path(self) -> Path:
        p = Path(self.log_path)
        if not p.is_absolute():
            p = self._repo_root / p
        return p

    def write(self, event: PredictionDLEEvent) -> None:
        try:
            path = self._resolve_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            line = _stable_json(asdict(event))
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            self._write_failures += 1
            print(
                f"[DLE_PREDICTION] Warning: log write failed ({self._write_failures} total): {e}",
                file=sys.stderr,
            )

    @property
    def write_failure_count(self) -> int:
        return self._write_failures


# ---------------------------------------------------------------------------
# Veto reasons
# ---------------------------------------------------------------------------

class VetoReason:
    DISABLED = "PREDICTION_DLE_DISABLED"
    DATASET_REJECTED = "DATASET_REJECTED"
    QUESTION_NOT_FOUND = "QUESTION_NOT_FOUND"
    OUTCOME_NOT_FOUND = "OUTCOME_NOT_FOUND"
    QUESTION_INACTIVE = "QUESTION_INACTIVE"
    DELTA_EXCEEDED = "MAX_DELTA_EXCEEDED"
    DATASET_NOT_ALLOWED = "DATASET_NOT_ALLOWED"
    QUESTION_NOT_ALLOWED = "QUESTION_NOT_ALLOWED"


# ---------------------------------------------------------------------------
# Gate logic
# ---------------------------------------------------------------------------

def build_decision(
    *,
    phase_id: str = DEFAULT_PHASE_ID,
    action_class: str = "WRITE_BELIEF",
    policy_version: str = DEFAULT_POLICY_VERSION,
    allowed_datasets: Optional[List[str]] = None,
    allowed_questions: Optional[List[str]] = None,
    max_delta: float = DEFAULT_MAX_DELTA,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    max_uses: int = DEFAULT_MAX_USES,
) -> PredictionDecision:
    """Build a prediction Decision with deterministic ID."""
    constraints = {
        "max_delta": max_delta,
        "ttl_seconds": ttl_seconds,
        "max_uses": max_uses,
    }
    ts = _utc_now_iso()
    decision_id = derive_prediction_decision_id(
        phase_id=phase_id,
        action_class=action_class,
        constraints=constraints,
        policy_version=policy_version,
    )
    return PredictionDecision(
        decision_id=decision_id,
        ts=ts,
        phase_id=phase_id,
        action_class=action_class,
        policy_version=policy_version,
        allowed_datasets=tuple(allowed_datasets or []),
        allowed_questions=tuple(allowed_questions or []),
        max_delta=max_delta,
        ttl_seconds=ttl_seconds,
        max_uses=max_uses,
        constraints=constraints,
    )


def issue_permit(
    *,
    decision: PredictionDecision,
    dataset_id: str,
    question_id: str,
    outcome_id: str,
    constraint_hash: str = "",
    prior_aggregate_hash: str = "",
    config_hash: str = "",
) -> PredictionPermit:
    """Issue a single-use Permit bound to a specific belief write."""
    ts = _utc_now_iso()
    permit_id = derive_prediction_permit_id(
        decision_id=decision.decision_id,
        dataset_id=dataset_id,
        question_id=question_id,
        outcome_id=outcome_id,
        issued_at_iso=ts,
    )
    return PredictionPermit(
        permit_id=permit_id,
        ts=ts,
        decision_id=decision.decision_id,
        dataset_id=dataset_id,
        question_id=question_id,
        outcome_id=outcome_id,
        single_use=True,
        ttl_seconds=decision.ttl_seconds,
        snapshots={
            "constraints": constraint_hash or _hash_snapshot("constraints", None),
            "prior_aggregate": prior_aggregate_hash or _hash_snapshot("prior_aggregate", None),
            "config": config_hash or _hash_snapshot("config", None),
        },
    )


def gate_belief_write(
    *,
    dataset_id: str,
    question_id: str,
    outcome_id: str,
    p: float,
    prior_p: Optional[float] = None,
    dataset_state: str = "REJECTED",
    question_graph: Optional[Dict[str, Any]] = None,
    decision: Optional[PredictionDecision] = None,
    writer: Optional[PredictionDLEWriter] = None,
    enabled: Optional[bool] = None,
    write_logs: Optional[bool] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Full DLE gate for a belief write request.

    Returns (decision_id, permit_id, veto_reason).
    If veto_reason is not None, the write was rejected.

    Mirrors the shadow_build_chain pattern from dle_shadow.py.
    """
    is_enabled = enabled if enabled is not None else PREDICTION_DLE_ENABLED
    should_write = write_logs if write_logs is not None else PREDICTION_DLE_WRITE_LOGS
    w = writer or PredictionDLEWriter()
    ts = _utc_now_iso()

    if not is_enabled:
        return (None, None, VetoReason.DISABLED)

    # Build or use provided Decision
    dec = decision or build_decision()

    # --- Veto checks ---

    # 1. Dataset admission
    if dataset_state == "REJECTED":
        _log_veto(w, should_write, ts, dec.decision_id, VetoReason.DATASET_REJECTED,
                  {"dataset_id": dataset_id})
        return (dec.decision_id, None, VetoReason.DATASET_REJECTED)

    # 2. Dataset allowed by Decision scope
    if dec.allowed_datasets and dataset_id not in dec.allowed_datasets:
        _log_veto(w, should_write, ts, dec.decision_id, VetoReason.DATASET_NOT_ALLOWED,
                  {"dataset_id": dataset_id, "allowed": list(dec.allowed_datasets)})
        return (dec.decision_id, None, VetoReason.DATASET_NOT_ALLOWED)

    # 3. Question exists and is active
    qg = question_graph or {}
    q_def = qg.get("questions", {}).get(question_id)
    if not q_def:
        _log_veto(w, should_write, ts, dec.decision_id, VetoReason.QUESTION_NOT_FOUND,
                  {"question_id": question_id})
        return (dec.decision_id, None, VetoReason.QUESTION_NOT_FOUND)

    if q_def.get("state", "ACTIVE") != "ACTIVE":
        _log_veto(w, should_write, ts, dec.decision_id, VetoReason.QUESTION_INACTIVE,
                  {"question_id": question_id, "state": q_def.get("state")})
        return (dec.decision_id, None, VetoReason.QUESTION_INACTIVE)

    # 4. Question allowed by Decision scope
    if dec.allowed_questions and question_id not in dec.allowed_questions:
        _log_veto(w, should_write, ts, dec.decision_id, VetoReason.QUESTION_NOT_ALLOWED,
                  {"question_id": question_id, "allowed": list(dec.allowed_questions)})
        return (dec.decision_id, None, VetoReason.QUESTION_NOT_ALLOWED)

    # 5. Outcome exists
    if outcome_id not in q_def.get("outcomes", []):
        _log_veto(w, should_write, ts, dec.decision_id, VetoReason.OUTCOME_NOT_FOUND,
                  {"question_id": question_id, "outcome_id": outcome_id,
                   "valid_outcomes": q_def.get("outcomes", [])})
        return (dec.decision_id, None, VetoReason.OUTCOME_NOT_FOUND)

    # 6. Max delta check (prevents "pump")
    if prior_p is not None and abs(p - prior_p) > dec.max_delta:
        _log_veto(w, should_write, ts, dec.decision_id, VetoReason.DELTA_EXCEEDED,
                  {"delta": abs(p - prior_p), "max_delta": dec.max_delta,
                   "p": p, "prior_p": prior_p})
        return (dec.decision_id, None, VetoReason.DELTA_EXCEEDED)

    # --- All checks passed: issue Permit ---
    permit = issue_permit(
        decision=dec,
        dataset_id=dataset_id,
        question_id=question_id,
        outcome_id=outcome_id,
    )

    # Log decision + permit
    if should_write:
        try:
            w.write(PredictionDLEEvent(
                schema_version=SCHEMA_VERSION,
                event_type="DECISION",
                ts=ts,
                payload=asdict(dec),
            ))
            w.write(PredictionDLEEvent(
                schema_version=SCHEMA_VERSION,
                event_type="PERMIT",
                ts=ts,
                payload=asdict(permit),
            ))
        except Exception:
            pass  # fail-open

    return (dec.decision_id, permit.permit_id, None)


def _log_veto(
    writer: PredictionDLEWriter,
    write_logs: bool,
    ts: str,
    decision_id: str,
    reason: str,
    context: Dict[str, Any],
) -> None:
    """Log a veto event. Fail-open."""
    if not write_logs:
        return
    try:
        writer.write(PredictionDLEEvent(
            schema_version=SCHEMA_VERSION,
            event_type="VETO",
            ts=ts,
            payload={
                "decision_id": decision_id,
                "veto_reason": reason,
                "context": context,
            },
        ))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Convenience: full gate + ingest in one call
# ---------------------------------------------------------------------------

def gate_and_ingest(
    *,
    dataset_id: str,
    question_id: str,
    outcome_id: str,
    p: float,
    confidence: float,
    evidence_hash: Optional[str] = None,
    prior_p: Optional[float] = None,
    question_graph: Optional[Dict[str, Any]] = None,
    admission: Optional[Dict[str, Any]] = None,
    decision: Optional[PredictionDecision] = None,
    writer: Optional[PredictionDLEWriter] = None,
    enabled: Optional[bool] = None,
    write_logs: Optional[bool] = None,
    belief_log_path: Optional[Path] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Combined DLE gate check + belief ingestion.

    Returns (decision_id, permit_id, veto_reason).
    If veto_reason is None, the belief was admitted and logged.
    """
    from prediction.belief_ingest import ingest_belief

    adm = admission or {}
    qg = question_graph or {}

    # Determine dataset state from admission config
    ds_entry = adm.get("datasets", {}).get(dataset_id, {})
    ds_state = ds_entry.get("state", "REJECTED")

    decision_id, permit_id, veto = gate_belief_write(
        dataset_id=dataset_id,
        question_id=question_id,
        outcome_id=outcome_id,
        p=p,
        prior_p=prior_p,
        dataset_state=ds_state,
        question_graph=qg,
        decision=decision,
        writer=writer,
        enabled=enabled,
        write_logs=write_logs,
    )

    if veto is not None:
        return (decision_id, permit_id, veto)

    # Ingest the belief (already gated)
    event = ingest_belief(
        dataset_id=dataset_id,
        question_id=question_id,
        outcome_id=outcome_id,
        p=p,
        confidence=confidence,
        evidence_hash=evidence_hash,
        decision_id=decision_id or "",
        permit_id=permit_id or "",
        question_graph=qg,
        admission=adm,
        belief_log_path=belief_log_path,
    )

    if event is None:
        return (decision_id, permit_id, "INGEST_FAILED")

    return (decision_id, permit_id, None)


# ---------------------------------------------------------------------------
# Global writer (lazy init, mirrors dle_shadow.py pattern)
# ---------------------------------------------------------------------------

_prediction_writer: Optional[PredictionDLEWriter] = None


def get_prediction_writer() -> PredictionDLEWriter:
    global _prediction_writer
    if _prediction_writer is None:
        _prediction_writer = PredictionDLEWriter()
    return _prediction_writer


def reset_prediction_writer() -> None:
    """Reset global instance (for testing only)."""
    global _prediction_writer
    _prediction_writer = None
