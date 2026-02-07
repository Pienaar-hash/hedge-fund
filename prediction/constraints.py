# prediction/constraints.py
"""
Deterministic constraint engine for the prediction belief ledger.

Core MHD move — enforces logical consistency centrally so we don't
rely on arbitrage or participant sophistication.

Invariants:
    - Deterministic: same inputs → bit-identical output
    - Fail closed: if constraints can't be satisfied → INVALID aggregate
    - No side effects: pure functions, no I/O

Constraint types:
    SUM_TO_ONE  — outcomes in group sum to 1.0
    BOUNDS      — each outcome probability ∈ [min, max]
    IMPLIES     — p(A) ≤ p(B) when A implies B
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TOLERANCE = 1e-9
MAX_PROJECTION_ITERATIONS = 100  # Dykstra's for combined constraints


class SolverStatus(str, Enum):
    OK = "OK"
    CLIPPED = "CLIPPED"            # feasible after clip, residual > 0
    INVALID = "INVALID"            # constraints unsatisfiable
    NO_CONSTRAINTS = "NO_CONSTRAINTS"


@dataclass(frozen=True)
class SolverResult:
    """Immutable result of constraint projection."""
    status: SolverStatus
    probs: Dict[str, float]
    residual: float
    iterations: int
    solver_name: str = "proj_simplex_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.solver_name,
            "status": self.status.value,
            "residual": self.residual,
            "iterations": self.iterations,
        }


@dataclass(frozen=True)
class Constraint:
    """Single logical constraint parsed from question_graph.json."""
    ctype: str                          # SUM_TO_ONE | BOUNDS | IMPLIES
    outcomes: Tuple[str, ...]           # outcome IDs affected
    min_val: float = 0.0
    max_val: float = 1.0
    # IMPLIES only:
    if_question: Optional[str] = None
    if_outcome: Optional[str] = None
    then_question: Optional[str] = None
    then_outcome: Optional[str] = None


# ---------------------------------------------------------------------------
# Simplex projection (Euclidean, O(n log n))
# ---------------------------------------------------------------------------

def project_onto_simplex(v: List[float], target_sum: float = 1.0) -> List[float]:
    """
    Project vector *v* onto the probability simplex Σ = target_sum, each ≥ 0.

    Algorithm: sort-based, O(n log n), deterministic.
    Reference: Duchi et al. 2008 — "Efficient Projections onto the ℓ1-Ball".

    Guarantees:
        - Output sums to *target_sum* within floating-point epsilon
        - Every element ∈ [0, target_sum]
        - Bit-identical for identical inputs on same platform
    """
    n = len(v)
    if n == 0:
        return []
    if n == 1:
        return [target_sum]

    # Sort descending
    u = sorted(v, reverse=True)

    cssv = 0.0
    rho = 0
    for j in range(n):
        cssv += u[j]
        test = u[j] - (cssv - target_sum) / (j + 1)
        if test > 0:
            rho = j + 1

    theta = (sum(u[:rho]) - target_sum) / rho
    result = [max(x - theta, 0.0) for x in v]
    return result


def clip_bounds(probs: Dict[str, float], outcomes: Sequence[str],
                lo: float = 0.0, hi: float = 1.0) -> Dict[str, float]:
    """Clip outcome probabilities to [lo, hi]. Pure, deterministic."""
    out = dict(probs)
    for o in outcomes:
        if o in out:
            out[o] = max(lo, min(hi, out[o]))
    return out


# ---------------------------------------------------------------------------
# IMPLIES repair
# ---------------------------------------------------------------------------

def repair_implies(
    probs: Dict[str, float],
    if_outcome: str,
    then_outcome: str,
    affected_sum_group: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, float], float]:
    """
    Enforce p(A) ≤ p(B) by minimal deterministic adjustment.

    If violated: raise p(B) to p(A), then renormalize the SUM_TO_ONE
    group *affected_sum_group* if provided.

    Returns (adjusted probs, residual).
    """
    out = dict(probs)
    pa = out.get(if_outcome, 0.0)
    pb = out.get(then_outcome, 0.0)

    if pa <= pb + DEFAULT_TOLERANCE:
        return out, 0.0

    delta = pa - pb
    out[then_outcome] = pa

    # Renormalize the group that contains then_outcome if provided
    if affected_sum_group:
        group_vals = [out.get(o, 0.0) for o in affected_sum_group]
        total = sum(group_vals)
        if total > DEFAULT_TOLERANCE:
            projected = project_onto_simplex(group_vals, target_sum=1.0)
            for o, pv in zip(affected_sum_group, projected):
                out[o] = pv

    return out, abs(delta)


# ---------------------------------------------------------------------------
# Full constraint solver
# ---------------------------------------------------------------------------

def parse_constraints(raw: List[Dict[str, Any]]) -> List[Constraint]:
    """Parse constraint dicts from question_graph.json into Constraint objects."""
    parsed: List[Constraint] = []
    for c in raw:
        ctype = c["type"]
        if ctype == "SUM_TO_ONE":
            parsed.append(Constraint(
                ctype="SUM_TO_ONE",
                outcomes=tuple(c["outcomes"]),
            ))
        elif ctype == "BOUNDS":
            parsed.append(Constraint(
                ctype="BOUNDS",
                outcomes=tuple(c["outcomes"]),
                min_val=float(c.get("min", 0.0)),
                max_val=float(c.get("max", 1.0)),
            ))
        elif ctype == "IMPLIES":
            parsed.append(Constraint(
                ctype="IMPLIES",
                outcomes=(),
                if_question=c.get("if", {}).get("Q"),
                if_outcome=c.get("if", {}).get("O"),
                then_question=c.get("then", {}).get("Q"),
                then_outcome=c.get("then", {}).get("O"),
            ))
        else:
            raise ValueError(f"Unknown constraint type: {ctype}")
    return parsed


def solve(
    unconstrained: Dict[str, float],
    constraints: List[Constraint],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    max_iter: int = MAX_PROJECTION_ITERATIONS,
) -> SolverResult:
    """
    Apply constraints to an unconstrained probability vector.

    Algorithm:
        1. Apply BOUNDS constraints (clip)
        2. Apply SUM_TO_ONE constraints (simplex projection)
        3. Apply IMPLIES constraints (minimal repair + re-project)
        4. Repeat until convergence or max_iter

    Returns SolverResult with status, final probs, residual, iterations.
    """
    if not constraints:
        return SolverResult(
            status=SolverStatus.NO_CONSTRAINTS,
            probs=dict(unconstrained),
            residual=0.0,
            iterations=0,
        )

    probs = dict(unconstrained)
    total_residual = 0.0
    iterations = 0

    # Separate constraint types
    bounds_cs = [c for c in constraints if c.ctype == "BOUNDS"]
    sum_cs = [c for c in constraints if c.ctype == "SUM_TO_ONE"]
    implies_cs = [c for c in constraints if c.ctype == "IMPLIES"]

    # Build lookup: outcome → SUM_TO_ONE group
    outcome_to_sum_group: Dict[str, Tuple[str, ...]] = {}
    for sc in sum_cs:
        for o in sc.outcomes:
            outcome_to_sum_group[o] = sc.outcomes

    for iteration in range(max_iter):
        prev = dict(probs)
        iterations = iteration + 1

        # 1. BOUNDS
        for bc in bounds_cs:
            probs = clip_bounds(probs, bc.outcomes, lo=bc.min_val, hi=bc.max_val)

        # 2. SUM_TO_ONE (simplex projection per group)
        for sc in sum_cs:
            vals = [probs.get(o, 0.0) for o in sc.outcomes]
            projected = project_onto_simplex(vals, target_sum=1.0)
            for o, pv in zip(sc.outcomes, projected):
                probs[o] = pv

        # 3. IMPLIES
        for ic in implies_cs:
            if ic.if_outcome and ic.then_outcome:
                group = outcome_to_sum_group.get(ic.then_outcome)
                probs, _ = repair_implies(
                    probs,
                    if_outcome=ic.if_outcome,
                    then_outcome=ic.then_outcome,
                    affected_sum_group=list(group) if group else None,
                )

        # Convergence check
        max_delta = max(
            abs(probs.get(k, 0.0) - prev.get(k, 0.0))
            for k in set(probs) | set(prev)
        )
        if max_delta < tolerance:
            break

    # Compute final residual (sum violation for each group)
    total_residual = 0.0
    for sc in sum_cs:
        group_sum = sum(probs.get(o, 0.0) for o in sc.outcomes)
        total_residual += abs(group_sum - 1.0)
    for bc in bounds_cs:
        for o in bc.outcomes:
            p = probs.get(o, 0.0)
            if p < bc.min_val - tolerance or p > bc.max_val + tolerance:
                total_residual += abs(p - max(bc.min_val, min(bc.max_val, p)))

    if total_residual > tolerance * 10:
        return SolverResult(
            status=SolverStatus.INVALID,
            probs=probs,
            residual=total_residual,
            iterations=iterations,
        )

    status = SolverStatus.OK if total_residual < tolerance else SolverStatus.CLIPPED
    return SolverResult(
        status=status,
        probs=probs,
        residual=total_residual,
        iterations=iterations,
    )


# ---------------------------------------------------------------------------
# Aggregation: weighted belief merge + constraint enforcement
# ---------------------------------------------------------------------------

@dataclass
class WeightedSource:
    """A single source's probability update with trust/confidence weights."""
    dataset_id: str
    outcome_id: str
    p: float
    confidence: float = 1.0
    dataset_trust: float = 1.0


def aggregate_beliefs(
    sources: List[WeightedSource],
    outcome_ids: Sequence[str],
    constraints: List[Constraint],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> SolverResult:
    """
    Merge multiple belief sources into constrained canonical probabilities.

    1. Compute weighted average per outcome (weight = confidence × dataset_trust)
    2. Apply constraint solver

    Deterministic: sorted source iteration, stable numerics.
    """
    # Group by outcome
    weighted_sums: Dict[str, float] = {o: 0.0 for o in outcome_ids}
    weight_totals: Dict[str, float] = {o: 0.0 for o in outcome_ids}

    # Sort sources for determinism
    for src in sorted(sources, key=lambda s: (s.outcome_id, s.dataset_id)):
        w = src.confidence * src.dataset_trust
        if w <= 0.0:
            continue
        if src.outcome_id in weighted_sums:
            weighted_sums[src.outcome_id] += src.p * w
            weight_totals[src.outcome_id] += w

    unconstrained: Dict[str, float] = {}
    for o in outcome_ids:
        total_w = weight_totals[o]
        if total_w > 0:
            unconstrained[o] = weighted_sums[o] / total_w
        else:
            # Uniform prior when no data
            unconstrained[o] = 1.0 / len(outcome_ids) if outcome_ids else 0.0

    return solve(unconstrained, constraints, tolerance=tolerance)


# ---------------------------------------------------------------------------
# Hashing (deterministic, for audit trail)
# ---------------------------------------------------------------------------

def hash_constraints(constraints: List[Constraint]) -> str:
    """Stable hash of constraint set for snapshot embedding."""
    import hashlib
    payload = json.dumps(
        [{"ctype": c.ctype, "outcomes": list(c.outcomes),
          "min": c.min_val, "max": c.max_val,
          "if_q": c.if_question, "if_o": c.if_outcome,
          "then_q": c.then_question, "then_o": c.then_outcome}
         for c in sorted(constraints, key=lambda c: (c.ctype, c.outcomes))],
        sort_keys=True, separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()[:8]
