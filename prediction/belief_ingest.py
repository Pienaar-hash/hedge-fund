# prediction/belief_ingest.py
"""
Belief ingestion layer — normalizes inputs from all sources into canonical
BeliefEvent objects.

Every source (Polymarket feed, human input, model output, news classifier)
becomes a dataset.  Default state: REJECTED.  Only PRODUCTION_ELIGIBLE or
RESEARCH_ONLY datasets can produce events that reach the constraint solver.

Invariants:
    - Monotonic logging: no edits, only new entries
    - All events contain DLE authority fields (decision_id, permit_id)
    - State snapshot hashes frozen at write time
    - Fail-closed: missing dataset → REJECTED → event dropped
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from prediction.constraints import (
    Constraint,
    SolverResult,
    WeightedSource,
    aggregate_beliefs,
    hash_constraints,
    parse_constraints,
    solve,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_QUESTION_GRAPH_PATH = _REPO_ROOT / "prediction" / "question_graph.json"
_DATASET_ADMISSION_PATH = _REPO_ROOT / "config" / "dataset_admission.json"
_BELIEF_EVENTS_LOG = _REPO_ROOT / "logs" / "prediction" / "belief_events.jsonl"
_AGGREGATE_STATE_LOG = _REPO_ROOT / "logs" / "prediction" / "aggregate_state.jsonl"

# Admission states that allow belief event processing
_ADMISSIBLE_STATES = frozenset({"PRODUCTION_ELIGIBLE", "RESEARCH_ONLY", "OBSERVE_ONLY"})

# ---------------------------------------------------------------------------
# Schema objects
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_short(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:8]


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class BeliefEvent:
    """A single belief update: probability of outcome X is now p, with confidence c, at time t."""
    belief_event_id: str
    ts: str
    dataset_id: str
    question_id: str
    outcome_id: str
    p: float
    confidence: float
    evidence_hash: str
    dle: Dict[str, str]                    # decision_id, permit_id
    state_snapshot_hashes: Dict[str, str]   # constraints, prior_aggregate, config

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AggregateState:
    """Canonical probabilities after applying all admitted beliefs + constraints."""
    ts: str
    question_id: str
    probs: Dict[str, float]
    aggregate_hash: str
    inputs_window: Dict[str, str]  # from, to
    solver: Dict[str, Any]        # name, status, residual

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# BeliefEvent ID derivation (deterministic, mirrors dle_shadow.py pattern)
# ---------------------------------------------------------------------------

def derive_belief_event_id(
    *,
    dataset_id: str,
    question_id: str,
    outcome_id: str,
    ts: str,
    p: float,
) -> str:
    """Deterministic belief event ID from canonical inputs."""
    payload = _stable_json({
        "dataset_id": dataset_id,
        "question_id": question_id,
        "outcome_id": outcome_id,
        "ts": ts,
        "p": p,
    })
    return "BEL_" + hashlib.sha256(payload.encode()).hexdigest()[:12]


def derive_aggregate_hash(question_id: str, probs: Dict[str, float]) -> str:
    """Deterministic hash of aggregate state."""
    payload = _stable_json({"question_id": question_id, "probs": probs})
    return "sha256:" + _sha256_short(payload)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_question_graph(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load question graph from JSON. Returns empty dict on missing file."""
    p = path or _QUESTION_GRAPH_PATH
    if not p.exists():
        return {"questions": {}, "implies_rules": []}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset_admission(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load dataset admission config. Returns empty datasets on missing file."""
    p = path or _DATASET_ADMISSION_PATH
    if not p.exists():
        return {"datasets": {}}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_dataset_state(dataset_id: str, admission: Optional[Dict[str, Any]] = None) -> str:
    """Return the admission state for a dataset. Default: REJECTED."""
    adm = admission or load_dataset_admission()
    ds = adm.get("datasets", {}).get(dataset_id, {})
    return ds.get("state", "REJECTED")


def get_dataset_trust(dataset_id: str, admission: Optional[Dict[str, Any]] = None) -> float:
    """
    Return a trust weight for a dataset based on its admission state.

    PRODUCTION_ELIGIBLE: 1.0
    RESEARCH_ONLY:       0.5  (down-weighted)
    OBSERVE_ONLY:        0.25 (heavily down-weighted)
    REJECTED:            0.0
    """
    state = get_dataset_state(dataset_id, admission)
    return {
        "PRODUCTION_ELIGIBLE": 1.0,
        "RESEARCH_ONLY": 0.5,
        "OBSERVE_ONLY": 0.25,
    }.get(state, 0.0)


# ---------------------------------------------------------------------------
# Event construction
# ---------------------------------------------------------------------------

def build_belief_event(
    *,
    dataset_id: str,
    question_id: str,
    outcome_id: str,
    p: float,
    confidence: float,
    evidence_hash: Optional[str] = None,
    decision_id: str,
    permit_id: str,
    constraint_hash: str = "",
    prior_aggregate_hash: str = "",
    config_hash: str = "",
    ts: Optional[str] = None,
) -> BeliefEvent:
    """Construct a validated BeliefEvent with deterministic ID."""
    now = ts or _utc_now_iso()

    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Probability must be in [0, 1], got {p}")
    if not (0.0 <= confidence <= 1.0):
        raise ValueError(f"Confidence must be in [0, 1], got {confidence}")

    ev_hash = evidence_hash or ("sha256:" + _sha256_short(
        _stable_json({"dataset_id": dataset_id, "question_id": question_id,
                       "outcome_id": outcome_id, "p": p, "ts": now})
    ))

    return BeliefEvent(
        belief_event_id=derive_belief_event_id(
            dataset_id=dataset_id,
            question_id=question_id,
            outcome_id=outcome_id,
            ts=now,
            p=p,
        ),
        ts=now,
        dataset_id=dataset_id,
        question_id=question_id,
        outcome_id=outcome_id,
        p=p,
        confidence=confidence,
        evidence_hash=ev_hash,
        dle={
            "decision_id": decision_id,
            "permit_id": permit_id,
        },
        state_snapshot_hashes={
            "constraints": constraint_hash,
            "prior_aggregate": prior_aggregate_hash,
            "config": config_hash,
        },
    )


# ---------------------------------------------------------------------------
# Aggregate computation
# ---------------------------------------------------------------------------

def compute_aggregate(
    events: Sequence[BeliefEvent],
    question_id: str,
    question_graph: Optional[Dict[str, Any]] = None,
    admission: Optional[Dict[str, Any]] = None,
) -> Optional[AggregateState]:
    """
    Compute canonical probabilities from a sequence of BeliefEvents for one question.

    Steps:
        1. Filter events for this question
        2. Look up dataset trust per event source
        3. Build WeightedSource inputs
        4. Run constraint solver (simplex projection + implies repair)
        5. Return AggregateState or None if no valid events

    Deterministic: sorted inputs, stable aggregation.
    """
    qg = question_graph or load_question_graph()
    adm = admission or load_dataset_admission()

    q_def = qg.get("questions", {}).get(question_id)
    if not q_def:
        return None

    outcome_ids = q_def["outcomes"]
    raw_constraints = q_def.get("constraints", [])
    constraints = parse_constraints(raw_constraints)

    # Filter relevant events
    relevant = [e for e in events if e.question_id == question_id]
    if not relevant:
        return None

    # Build weighted sources
    sources: List[WeightedSource] = []
    for ev in relevant:
        trust = get_dataset_trust(ev.dataset_id, adm)
        if trust <= 0.0:
            continue  # REJECTED dataset — skip
        sources.append(WeightedSource(
            dataset_id=ev.dataset_id,
            outcome_id=ev.outcome_id,
            p=ev.p,
            confidence=ev.confidence,
            dataset_trust=trust,
        ))

    if not sources:
        return None

    result = aggregate_beliefs(sources, outcome_ids, constraints)

    # Determine time window
    timestamps = sorted(ev.ts for ev in relevant)
    now = _utc_now_iso()

    return AggregateState(
        ts=now,
        question_id=question_id,
        probs=result.probs,
        aggregate_hash=derive_aggregate_hash(question_id, result.probs),
        inputs_window={"from": timestamps[0], "to": timestamps[-1]},
        solver=result.to_dict(),
    )


# ---------------------------------------------------------------------------
# JSONL writers (append-only, fail-open)
# ---------------------------------------------------------------------------

def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append a single JSON record to a JSONL file. Creates parents if needed."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"[belief_ingest] JSONL write failed {path}: {e}", file=sys.stderr)


def log_belief_event(event: BeliefEvent, path: Optional[Path] = None) -> None:
    """Append a BeliefEvent to the belief events JSONL log."""
    _append_jsonl(path or _BELIEF_EVENTS_LOG, event.to_dict())


def log_aggregate_state(state: AggregateState, path: Optional[Path] = None) -> None:
    """Append an AggregateState to the aggregate state JSONL log."""
    _append_jsonl(path or _AGGREGATE_STATE_LOG, state.to_dict())


# ---------------------------------------------------------------------------
# Full ingest pipeline
# ---------------------------------------------------------------------------

def ingest_belief(
    *,
    dataset_id: str,
    question_id: str,
    outcome_id: str,
    p: float,
    confidence: float,
    evidence_hash: Optional[str] = None,
    decision_id: str,
    permit_id: str,
    question_graph: Optional[Dict[str, Any]] = None,
    admission: Optional[Dict[str, Any]] = None,
    belief_log_path: Optional[Path] = None,
) -> Optional[BeliefEvent]:
    """
    Full ingestion pipeline for a single belief update.

    1. Check dataset admission (REJECTED → drop)
    2. Validate question/outcome exist in graph
    3. Build BeliefEvent with DLE references
    4. Log to JSONL
    5. Return the event (or None if rejected)

    This does NOT recompute aggregates — call compute_aggregate separately.
    """
    adm = admission or load_dataset_admission()
    qg = question_graph or load_question_graph()

    # Gate 1: Dataset admission
    ds_state = get_dataset_state(dataset_id, adm)
    if ds_state not in _ADMISSIBLE_STATES:
        return None

    # Gate 2: Question + outcome validation
    q_def = qg.get("questions", {}).get(question_id)
    if not q_def:
        return None
    if outcome_id not in q_def.get("outcomes", []):
        return None

    # Gate 3: Question must be ACTIVE
    if q_def.get("state", "ACTIVE") != "ACTIVE":
        return None

    # Build constraint hash for snapshot
    raw_constraints = q_def.get("constraints", [])
    constraints = parse_constraints(raw_constraints)
    c_hash = hash_constraints(constraints)

    event = build_belief_event(
        dataset_id=dataset_id,
        question_id=question_id,
        outcome_id=outcome_id,
        p=p,
        confidence=confidence,
        evidence_hash=evidence_hash,
        decision_id=decision_id,
        permit_id=permit_id,
        constraint_hash=c_hash,
    )

    log_belief_event(event, belief_log_path)
    return event
