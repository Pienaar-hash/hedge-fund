# prediction/prediction_episodes.py
"""
Prediction Episode tracker — binds the lifecycle of prediction events:

    Decision → Permit → BeliefEvent(s) → Aggregate changes → Outcome resolution

After an event resolves, sources are scored via a proper scoring rule
(Brier score by default) and dataset trust weights can be updated as ADVISORY.

This matches the Episode schema concept from execution/episode_ledger.py:
end-to-end, not just "a price".

Invariants:
    - Episodes are append-only (new entries, no edits)
    - Scoring is deterministic (same outcomes → same scores)
    - Trust weight updates are ADVISORY tier only (never auto-promoted)
    - Fail-open: scoring failures never crash the pipeline
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EPISODES_LOG = _REPO_ROOT / "logs" / "prediction" / "prediction_episodes.jsonl"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Scoring rules (deterministic, pure)
# ---------------------------------------------------------------------------

def brier_score(predicted: float, outcome: float) -> float:
    """
    Brier score for a single binary outcome.

    Lower is better. Range: [0, 2].
    predicted: forecasted probability  ∈ [0, 1]
    outcome:   1.0 if event occurred, 0.0 otherwise
    """
    return (predicted - outcome) ** 2


def log_score(predicted: float, outcome: float, *, epsilon: float = 1e-15) -> float:
    """
    Logarithmic scoring rule.

    More sensitive to confident wrong predictions.
    Returns negative log-likelihood (lower is better, -∞ is worst).
    """
    p = max(epsilon, min(1.0 - epsilon, predicted))
    if outcome >= 0.5:
        return -math.log(p)
    return -math.log(1.0 - p)


def score_forecast(
    probs: Dict[str, float],
    resolved_outcome: str,
    scoring_rule: str = "brier",
) -> Dict[str, Any]:
    """
    Score a probability forecast against a resolved outcome.

    Returns per-outcome scores and an aggregate score.
    """
    scores: Dict[str, float] = {}
    for outcome_id, p in sorted(probs.items()):
        actual = 1.0 if outcome_id == resolved_outcome else 0.0
        if scoring_rule == "log":
            scores[outcome_id] = log_score(p, actual)
        else:
            scores[outcome_id] = brier_score(p, actual)

    aggregate = sum(scores.values()) / max(len(scores), 1)

    return {
        "scoring_rule": scoring_rule,
        "per_outcome": scores,
        "aggregate_score": aggregate,
        "resolved_outcome": resolved_outcome,
    }


# ---------------------------------------------------------------------------
# Episode data objects
# ---------------------------------------------------------------------------

@dataclass
class PredictionEpisode:
    """
    End-to-end record linking belief updates → aggregate → outcome.

    Lifecycle:
        OPEN      → beliefs being collected
        RESOLVED  → outcome known, scored
        EXPIRED   → question TTL passed without resolution
    """
    episode_id: str
    question_id: str
    state: str                                  # OPEN | RESOLVED | EXPIRED
    created_ts: str
    resolved_ts: Optional[str] = None

    # DLE chain
    decision_ids: List[str] = field(default_factory=list)
    permit_ids: List[str] = field(default_factory=list)
    belief_event_ids: List[str] = field(default_factory=list)

    # Aggregation
    initial_probs: Optional[Dict[str, float]] = None
    final_probs: Optional[Dict[str, float]] = None
    belief_count: int = 0

    # Resolution
    resolved_outcome: Optional[str] = None
    scoring: Optional[Dict[str, Any]] = None

    # Source attribution
    source_scores: Optional[Dict[str, Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def derive_episode_id(question_id: str, created_ts: str) -> str:
    """Deterministic episode ID."""
    payload = _stable_json({"question_id": question_id, "created_ts": created_ts})
    return "PEP_" + hashlib.sha256(payload.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Episode lifecycle
# ---------------------------------------------------------------------------

def open_episode(
    question_id: str,
    initial_probs: Optional[Dict[str, float]] = None,
) -> PredictionEpisode:
    """Create a new open episode for a question."""
    ts = _utc_now_iso()
    return PredictionEpisode(
        episode_id=derive_episode_id(question_id, ts),
        question_id=question_id,
        state="OPEN",
        created_ts=ts,
        initial_probs=initial_probs,
    )


def record_belief(
    episode: PredictionEpisode,
    *,
    belief_event_id: str,
    decision_id: str,
    permit_id: str,
) -> PredictionEpisode:
    """Record a belief event into an episode. Returns updated episode."""
    episode.belief_event_ids.append(belief_event_id)
    if decision_id not in episode.decision_ids:
        episode.decision_ids.append(decision_id)
    if permit_id not in episode.permit_ids:
        episode.permit_ids.append(permit_id)
    episode.belief_count += 1
    return episode


def resolve_episode(
    episode: PredictionEpisode,
    *,
    resolved_outcome: str,
    final_probs: Dict[str, float],
    source_beliefs: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    scoring_rule: str = "brier",
) -> PredictionEpisode:
    """
    Resolve an episode: set outcome, compute scores, attribute sources.

    source_beliefs: {dataset_id: [{outcome_id, p, confidence}, ...]}
    """
    episode.state = "RESOLVED"
    episode.resolved_ts = _utc_now_iso()
    episode.resolved_outcome = resolved_outcome
    episode.final_probs = final_probs
    episode.scoring = score_forecast(final_probs, resolved_outcome, scoring_rule)

    # Per-source scoring
    if source_beliefs:
        episode.source_scores = {}
        for dataset_id, beliefs in sorted(source_beliefs.items()):
            # Build source-specific probability vector
            source_probs: Dict[str, float] = {}
            for b in beliefs:
                source_probs[b["outcome_id"]] = b["p"]
            if source_probs:
                episode.source_scores[dataset_id] = score_forecast(
                    source_probs, resolved_outcome, scoring_rule
                )

    return episode


def expire_episode(episode: PredictionEpisode) -> PredictionEpisode:
    """Mark an episode as expired (no resolution received)."""
    episode.state = "EXPIRED"
    episode.resolved_ts = _utc_now_iso()
    return episode


# ---------------------------------------------------------------------------
# JSONL writer (append-only)
# ---------------------------------------------------------------------------

def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"[prediction_episodes] JSONL write failed {path}: {e}", file=sys.stderr)


def log_episode(episode: PredictionEpisode, path: Optional[Path] = None) -> None:
    """Append an episode record to the prediction episodes log."""
    _append_jsonl(path or _EPISODES_LOG, episode.to_dict())


# ---------------------------------------------------------------------------
# Trust weight advisory (NEVER authoritative)
# ---------------------------------------------------------------------------

def compute_trust_advisory(
    source_scores: Dict[str, Dict[str, Any]],
    *,
    baseline_trust: float = 1.0,
    sensitivity: float = 0.1,
) -> Dict[str, float]:
    """
    Compute ADVISORY trust weight adjustments based on source scoring.

    Returns {dataset_id: suggested_trust_delta}.
    Positive = source performed well, negative = poorly.

    This is ADVISORY ONLY — never auto-applied to dataset admission.
    Promotion requires explicit cycle-boundary review.
    """
    advisories: Dict[str, float] = {}
    for dataset_id, scores in sorted(source_scores.items()):
        agg = scores.get("aggregate_score", 0.5)
        # Brier: 0 = perfect, 0.25 = random, 0.5+ = bad
        # Map to trust delta: good forecasters get positive, bad get negative
        if scores.get("scoring_rule") == "brier":
            # Centered at 0.25 (random baseline for binary)
            delta = (0.25 - agg) * sensitivity
        else:
            # Log score: lower is better, centered at -log(0.5) ≈ 0.693
            delta = (0.693 - agg) * sensitivity
        advisories[dataset_id] = round(delta, 6)
    return advisories
