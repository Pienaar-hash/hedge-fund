# prediction/alert_ranker.py
"""
P1 Advisory Consumer — Alert Ranking via Prediction Uncertainty.

This is the **first and only** advisory consumer in P1.  It reorders an
existing list of execution alerts based on prediction-layer signals,
without generating, dropping, or modifying any alerts.

Hard invariants (tested):
    1. **Set equality** — same alerts in, same alerts out (only order changes)
    2. **No executor imports** — no coupling to execution decision paths
    3. **Firewall gated** — every prediction read goes through ``request_advisory()``
    4. **No network calls** — pure function of inputs
    5. **No writes outside prediction scope** — only ``logs/prediction/``

Ranking heuristic:
    - Higher uncertainty (entropy near 1.0 for binary) → rank higher (needs attention)
    - Larger recent shift (|Δprob|) → rank higher (change signal)
    - Missing prediction data → no reordering (fail-open)

Usage:
    ranked = rank_alerts(
        alerts=[...],                 # existing alert dicts
        prediction_snapshots={...},   # question_id → probs/shift
        phase="P1_ADVISORY",
        enabled=True,
    )
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from prediction.firewall import FirewallVerdict, request_advisory

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RANK_LOG = _REPO_ROOT / "logs" / "prediction" / "alert_ranking.jsonl"

# Consumer identity — must match firewall's advisory-only list
_CONSUMER_ID = "alert_ranking"


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PredictionSnapshot:
    """Prediction state for one question, used as ranking input."""
    question_id: str
    probs: Dict[str, float]
    aggregate_hash: str = ""
    aggregate_ts: str = ""
    dataset_states: Optional[Dict[str, str]] = None
    prior_probs: Optional[Dict[str, float]] = None   # previous snapshot for shift


@dataclass(frozen=True)
class RankingResult:
    """Output of the alert ranker."""
    alerts: List[Dict[str, Any]]        # same alerts, possibly reordered
    rankings_applied: bool              # True if prediction modified order
    reason: str                         # human-readable explanation
    firewall_verdicts: Dict[str, str]   # question_id → verdict


# ---------------------------------------------------------------------------
# Scoring helpers (pure functions, no I/O)
# ---------------------------------------------------------------------------

def _entropy(probs: Dict[str, float]) -> float:
    """Shannon entropy of a probability distribution. Higher = more uncertain."""
    h = 0.0
    for p in probs.values():
        if p > 0:
            h -= p * math.log2(p)
    return h


def _max_entropy(n_outcomes: int) -> float:
    """Maximum entropy for n outcomes (uniform distribution)."""
    if n_outcomes <= 1:
        return 0.0
    return math.log2(n_outcomes)


def _normalized_entropy(probs: Dict[str, float]) -> float:
    """Entropy normalized to [0, 1]. 1.0 = maximum uncertainty."""
    n = len(probs)
    if n <= 1:
        return 0.0
    max_h = _max_entropy(n)
    if max_h == 0:
        return 0.0
    return _entropy(probs) / max_h


def _max_shift(current: Dict[str, float],
               prior: Optional[Dict[str, float]]) -> float:
    """Maximum absolute probability shift across outcomes. 0 if no prior."""
    if not prior:
        return 0.0
    max_delta = 0.0
    for outcome, prob in current.items():
        prev = prior.get(outcome, prob)
        max_delta = max(max_delta, abs(prob - prev))
    return max_delta


def _relevance_score(probs: Dict[str, float],
                     prior_probs: Optional[Dict[str, float]]) -> float:
    """
    Compute composite relevance score for alert ranking.

    Higher = more relevant (deserves higher rank / more attention).

    Components:
        - Normalized entropy (0–1): uncertainty signal
        - Max shift (0–1): change signal
        - Combined with equal weight (tunable later)
    """
    uncertainty = _normalized_entropy(probs)
    shift = _max_shift(probs, prior_probs)
    # Equal weight — can be tuned in P2 with empirical data
    return 0.5 * uncertainty + 0.5 * shift


# ---------------------------------------------------------------------------
# Logging (append-only, fail-open)
# ---------------------------------------------------------------------------

def _log_ranking(
    alerts_count: int,
    rankings_applied: bool,
    firewall_verdicts: Dict[str, str],
    reason: str,
    log_path: Optional[Path] = None,
) -> None:
    """Append a ranking event to the alert ranking log. Fail-open."""
    path = log_path or _RANK_LOG
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "consumer": _CONSUMER_ID,
            "alerts_count": alerts_count,
            "rankings_applied": rankings_applied,
            "firewall_verdicts": firewall_verdicts,
            "reason": reason,
        }
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"[prediction.alert_ranker] log failed: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rank_alerts(
    *,
    alerts: Sequence[Dict[str, Any]],
    prediction_snapshots: Optional[Dict[str, PredictionSnapshot]] = None,
    phase: str = "P0_OBSERVE",
    enabled: bool = False,
    log_path: Optional[Path] = None,
    denial_log_path: Optional[Path] = None,
) -> RankingResult:
    """
    Reorder alerts based on prediction-layer uncertainty and shift signals.

    This is a **pure reorder** — no alerts are added, removed, or modified.

    Parameters
    ----------
    alerts
        Existing alert dicts (must have at least 'type' or 'severity').
    prediction_snapshots
        Map of question_id → ``PredictionSnapshot``.  If None or empty,
        alerts are returned in original order.
    phase
        Current prediction phase (P0_OBSERVE, P1_ADVISORY, P2_PRODUCTION).
    enabled
        Whether the prediction layer is enabled.
    log_path
        Override for the ranking log file.
    denial_log_path
        Override for the firewall denial log file.

    Returns
    -------
    RankingResult
        Contains the (possibly reordered) alerts and metadata.
    """
    alert_list = list(alerts)  # defensive copy — never mutate input
    verdicts: Dict[str, str] = {}

    # Early exit: nothing to rank
    if not alert_list:
        return RankingResult(
            alerts=[],
            rankings_applied=False,
            reason="No alerts to rank",
            firewall_verdicts={},
        )

    # Early exit: no prediction data
    if not prediction_snapshots:
        _log_ranking(len(alert_list), False, {}, "No prediction snapshots",
                     log_path)
        return RankingResult(
            alerts=alert_list,
            rankings_applied=False,
            reason="No prediction snapshots provided",
            firewall_verdicts={},
        )

    # Gate each prediction snapshot through the firewall
    permitted_scores: Dict[str, float] = {}

    for q_id, snapshot in prediction_snapshots.items():
        fw_result = request_advisory(
            consumer=_CONSUMER_ID,
            question_id=q_id,
            probs=snapshot.probs,
            aggregate_hash=snapshot.aggregate_hash,
            aggregate_ts=snapshot.aggregate_ts,
            dataset_states=snapshot.dataset_states or {},
            phase=phase,
            enabled=enabled,
            denial_log_path=denial_log_path,
        )
        verdicts[q_id] = fw_result.verdict.value

        if fw_result.verdict == FirewallVerdict.ALLOWED and fw_result.payload:
            score = _relevance_score(
                fw_result.payload.probs,
                snapshot.prior_probs,
            )
            permitted_scores[q_id] = score

    # If no predictions were permitted, return original order
    if not permitted_scores:
        reason = "All prediction snapshots denied by firewall"
        _log_ranking(len(alert_list), False, verdicts, reason, log_path)
        return RankingResult(
            alerts=alert_list,
            rankings_applied=False,
            reason=reason,
            firewall_verdicts=verdicts,
        )

    # Compute a composite boost for each alert
    # An alert gets a boost if it mentions a symbol/type that maps to a
    # permitted question.  For now: use the max relevance score across
    # all permitted questions as a global boost applied via sort stability.
    max_relevance = max(permitted_scores.values()) if permitted_scores else 0.0

    # Build per-alert relevance by matching alert's question_id field (if present)
    # or falling back to the global max relevance
    def _alert_relevance(alert: Dict[str, Any]) -> float:
        q_id = alert.get("prediction_question_id")
        if q_id and q_id in permitted_scores:
            return permitted_scores[q_id]
        # If alert tags a question_id, use it; otherwise use max
        return max_relevance if permitted_scores else 0.0

    # Keep existing severity as primary sort, then prediction relevance as tiebreaker
    severity_order = {"critical": 3, "warning": 2, "info": 1}

    def _sort_key(alert: Dict[str, Any]) -> tuple:
        sev = str(alert.get("severity", "info")).lower()
        sev_val = severity_order.get(sev, 0)
        rel = _alert_relevance(alert)
        return (-sev_val, -rel)

    ranked = sorted(alert_list, key=_sort_key)

    # Check if order actually changed
    order_changed = ranked != alert_list

    reason = (f"Ranked {len(alert_list)} alerts using {len(permitted_scores)} "
              f"prediction snapshots (max_relevance={max_relevance:.3f})")

    _log_ranking(len(alert_list), order_changed, verdicts, reason, log_path)

    return RankingResult(
        alerts=ranked,
        rankings_applied=order_changed,
        reason=reason,
        firewall_verdicts=verdicts,
    )
