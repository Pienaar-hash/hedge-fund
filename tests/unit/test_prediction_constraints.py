# tests/unit/test_prediction_constraints.py
"""
Tests for prediction/constraints.py — deterministic constraint solver.

Covers:
    - Simplex projection (SUM_TO_ONE)
    - Bounds clipping
    - IMPLIES repair
    - Full constraint solve pipeline
    - Weighted belief aggregation
    - Deterministic hashing
    - Edge cases (empty, single outcome, invalid)
"""

from __future__ import annotations

import pytest
from prediction.constraints import (
    Constraint,
    SolverResult,
    SolverStatus,
    WeightedSource,
    aggregate_beliefs,
    clip_bounds,
    hash_constraints,
    parse_constraints,
    project_onto_simplex,
    repair_implies,
    solve,
)


# ---------------------------------------------------------------------------
# Simplex projection
# ---------------------------------------------------------------------------

class TestSimplexProjection:
    def test_already_on_simplex(self):
        result = project_onto_simplex([0.3, 0.5, 0.2])
        assert abs(sum(result) - 1.0) < 1e-10
        assert all(r >= 0 for r in result)

    def test_negative_values_clipped(self):
        result = project_onto_simplex([-0.5, 0.8, 0.3])
        assert abs(sum(result) - 1.0) < 1e-10
        assert all(r >= -1e-15 for r in result)

    def test_all_zeros(self):
        result = project_onto_simplex([0.0, 0.0, 0.0])
        assert abs(sum(result) - 1.0) < 1e-10

    def test_single_outcome(self):
        result = project_onto_simplex([0.7])
        assert len(result) == 1
        assert abs(result[0] - 1.0) < 1e-10

    def test_empty(self):
        result = project_onto_simplex([])
        assert result == []

    def test_two_outcomes_overshooting(self):
        result = project_onto_simplex([0.8, 0.9])
        assert abs(sum(result) - 1.0) < 1e-10
        assert all(r >= 0 for r in result)

    def test_deterministic(self):
        """Same inputs → bit-identical output."""
        v = [0.4, 0.1, 0.7, 0.2]
        r1 = project_onto_simplex(v)
        r2 = project_onto_simplex(v)
        assert r1 == r2

    def test_large_vector(self):
        v = [1.0 / i for i in range(1, 21)]
        result = project_onto_simplex(v)
        assert abs(sum(result) - 1.0) < 1e-10
        assert all(r >= -1e-15 for r in result)


# ---------------------------------------------------------------------------
# Bounds clipping
# ---------------------------------------------------------------------------

class TestClipBounds:
    def test_within_bounds(self):
        probs = {"A": 0.3, "B": 0.7}
        clipped = clip_bounds(probs, ["A", "B"])
        assert clipped == probs

    def test_clip_low(self):
        probs = {"A": -0.1, "B": 0.7}
        clipped = clip_bounds(probs, ["A", "B"])
        assert clipped["A"] == 0.0

    def test_clip_high(self):
        probs = {"A": 0.3, "B": 1.5}
        clipped = clip_bounds(probs, ["A", "B"])
        assert clipped["B"] == 1.0

    def test_custom_bounds(self):
        probs = {"A": 0.05}
        clipped = clip_bounds(probs, ["A"], lo=0.1, hi=0.9)
        assert clipped["A"] == 0.1


# ---------------------------------------------------------------------------
# IMPLIES repair
# ---------------------------------------------------------------------------

class TestImpliesRepair:
    def test_no_violation(self):
        probs = {"A": 0.3, "B": 0.5}
        result, residual = repair_implies(probs, "A", "B")
        assert result == probs
        assert residual == 0.0

    def test_violation_raised(self):
        probs = {"A": 0.7, "B": 0.3}
        result, residual = repair_implies(probs, "A", "B")
        assert result["B"] >= result["A"]
        assert residual > 0.0

    def test_with_sum_group_renormalization(self):
        probs = {"A": 0.7, "B": 0.3, "C": 0.3}
        result, _ = repair_implies(probs, "A", "B", affected_sum_group=["B", "C"])
        # B was raised, group should be renormalized
        group_sum = result["B"] + result["C"]
        assert abs(group_sum - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Constraint parsing
# ---------------------------------------------------------------------------

class TestParseConstraints:
    def test_sum_to_one(self):
        raw = [{"type": "SUM_TO_ONE", "outcomes": ["O_a", "O_b"]}]
        parsed = parse_constraints(raw)
        assert len(parsed) == 1
        assert parsed[0].ctype == "SUM_TO_ONE"
        assert parsed[0].outcomes == ("O_a", "O_b")

    def test_bounds(self):
        raw = [{"type": "BOUNDS", "outcomes": ["O_a"], "min": 0.1, "max": 0.9}]
        parsed = parse_constraints(raw)
        assert parsed[0].min_val == 0.1
        assert parsed[0].max_val == 0.9

    def test_implies(self):
        raw = [{"type": "IMPLIES",
                "if": {"Q": "Q_x", "O": "O_yes"},
                "then": {"Q": "Q_y", "O": "O_yes"}}]
        parsed = parse_constraints(raw)
        assert parsed[0].ctype == "IMPLIES"
        assert parsed[0].if_outcome == "O_yes"
        assert parsed[0].then_outcome == "O_yes"

    def test_unknown_type_raises(self):
        raw = [{"type": "UNKNOWN", "outcomes": []}]
        with pytest.raises(ValueError, match="Unknown constraint type"):
            parse_constraints(raw)


# ---------------------------------------------------------------------------
# Full solver
# ---------------------------------------------------------------------------

class TestSolve:
    def test_binary_sum_to_one(self):
        """Basic binary question: probabilities sum to 1."""
        constraints = [
            Constraint(ctype="SUM_TO_ONE", outcomes=("O_yes", "O_no")),
            Constraint(ctype="BOUNDS", outcomes=("O_yes", "O_no")),
        ]
        unconstrained = {"O_yes": 0.8, "O_no": 0.7}
        result = solve(unconstrained, constraints)
        assert result.status in (SolverStatus.OK, SolverStatus.CLIPPED)
        assert abs(sum(result.probs.values()) - 1.0) < 1e-8
        assert all(0.0 <= v <= 1.0 + 1e-10 for v in result.probs.values())

    def test_three_way(self):
        constraints = [
            Constraint(ctype="SUM_TO_ONE", outcomes=("A", "B", "C")),
            Constraint(ctype="BOUNDS", outcomes=("A", "B", "C")),
        ]
        unconstrained = {"A": 0.5, "B": 0.3, "C": 0.4}
        result = solve(unconstrained, constraints)
        assert abs(sum(result.probs.values()) - 1.0) < 1e-8

    def test_no_constraints(self):
        result = solve({"A": 0.5, "B": 0.3}, [])
        assert result.status == SolverStatus.NO_CONSTRAINTS
        assert result.probs == {"A": 0.5, "B": 0.3}

    def test_deterministic(self):
        """Same inputs produce identical output."""
        constraints = [
            Constraint(ctype="SUM_TO_ONE", outcomes=("A", "B", "C")),
            Constraint(ctype="BOUNDS", outcomes=("A", "B", "C")),
        ]
        u = {"A": 0.4, "B": 0.1, "C": 0.7}
        r1 = solve(u, constraints)
        r2 = solve(u, constraints)
        assert r1.probs == r2.probs
        assert r1.residual == r2.residual


# ---------------------------------------------------------------------------
# Weighted aggregation
# ---------------------------------------------------------------------------

class TestAggregateBeliefs:
    def test_single_source(self):
        sources = [
            WeightedSource(dataset_id="d1", outcome_id="A", p=0.6, confidence=1.0, dataset_trust=1.0),
            WeightedSource(dataset_id="d1", outcome_id="B", p=0.4, confidence=1.0, dataset_trust=1.0),
        ]
        constraints = [Constraint(ctype="SUM_TO_ONE", outcomes=("A", "B"))]
        result = aggregate_beliefs(sources, ["A", "B"], constraints)
        assert abs(result.probs["A"] - 0.6) < 1e-8
        assert abs(result.probs["B"] - 0.4) < 1e-8

    def test_two_sources_weighted(self):
        sources = [
            WeightedSource(dataset_id="d1", outcome_id="A", p=0.8, confidence=1.0, dataset_trust=1.0),
            WeightedSource(dataset_id="d2", outcome_id="A", p=0.4, confidence=1.0, dataset_trust=1.0),
            WeightedSource(dataset_id="d1", outcome_id="B", p=0.2, confidence=1.0, dataset_trust=1.0),
            WeightedSource(dataset_id="d2", outcome_id="B", p=0.6, confidence=1.0, dataset_trust=1.0),
        ]
        constraints = [Constraint(ctype="SUM_TO_ONE", outcomes=("A", "B"))]
        result = aggregate_beliefs(sources, ["A", "B"], constraints)
        # Equal weight: A = (0.8 + 0.4)/2 = 0.6, B = (0.2 + 0.6)/2 = 0.4
        assert abs(result.probs["A"] - 0.6) < 1e-8
        assert abs(result.probs["B"] - 0.4) < 1e-8

    def test_zero_trust_excluded(self):
        sources = [
            WeightedSource(dataset_id="d1", outcome_id="A", p=0.9, confidence=1.0, dataset_trust=0.0),
            WeightedSource(dataset_id="d2", outcome_id="A", p=0.5, confidence=1.0, dataset_trust=1.0),
            WeightedSource(dataset_id="d2", outcome_id="B", p=0.5, confidence=1.0, dataset_trust=1.0),
        ]
        constraints = [Constraint(ctype="SUM_TO_ONE", outcomes=("A", "B"))]
        result = aggregate_beliefs(sources, ["A", "B"], constraints)
        # d1 excluded (trust=0), only d2 contributes
        assert abs(result.probs["A"] - 0.5) < 1e-8

    def test_confidence_weighting(self):
        sources = [
            WeightedSource(dataset_id="d1", outcome_id="A", p=0.9, confidence=0.1, dataset_trust=1.0),
            WeightedSource(dataset_id="d2", outcome_id="A", p=0.3, confidence=0.9, dataset_trust=1.0),
        ]
        constraints = []
        result = aggregate_beliefs(sources, ["A"], constraints)
        # Weight-weighted: (0.9*0.1 + 0.3*0.9) / (0.1 + 0.9) = (0.09 + 0.27) / 1.0 = 0.36
        assert abs(result.probs["A"] - 0.36) < 1e-8


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

class TestHashConstraints:
    def test_deterministic(self):
        cs = [Constraint(ctype="SUM_TO_ONE", outcomes=("A", "B"))]
        h1 = hash_constraints(cs)
        h2 = hash_constraints(cs)
        assert h1 == h2
        assert h1.startswith("sha256:")

    def test_different_constraints_different_hash(self):
        cs1 = [Constraint(ctype="SUM_TO_ONE", outcomes=("A", "B"))]
        cs2 = [Constraint(ctype="BOUNDS", outcomes=("A",), min_val=0.0, max_val=0.5)]
        assert hash_constraints(cs1) != hash_constraints(cs2)
