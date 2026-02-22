#!/usr/bin/env python3
"""
Binary Lab Activation Gate Evaluator
=====================================

Pure decision logic — reads the structured JSON emitted by
``hybrid_variance_audit.py --json`` and applies the 9 deterministic gates
from the Binary Lab Activation Decision Sheet.

Usage
-----
    python scripts/hybrid_variance_audit.py --synthetic --json | \
        python scripts/evaluate_binary_lab_gate.py

    # Or from a saved file
    python scripts/evaluate_binary_lab_gate.py < audit_output.json

Output
------
A JSON object on stdout:

    {
      "binary_lab_activation": "GO" | "NO_GO",
      "gates_evaluated": 9,
      "gates_passed": N,
      "failed_conditions": ["gate_1_non_degeneracy", ...],
      "summary": { ... per-gate verdicts ... }
    }

Exit codes:
  0 — GO (all 9 gates pass)
  1 — NO_GO (at least one gate failed)
  2 — Bad input / parse error
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Tuple


# ── Gate definitions ────────────────────────────────────────────────────
# Each gate is a (name, path-or-callable, description) triple.
# The evaluator walks the JSON to resolve boolean verdicts.

GATES: List[Tuple[str, str]] = [
    ("gate_0_reconstruction",   "Reconstruction error below tolerance"),
    ("gate_1_non_degeneracy",   "No zero-variance degeneracy in components"),
    ("condition_a_router",      "Router contribution < 50% of Var(H)"),
    ("condition_b_signal",      "Signal share (T+C+E) ≥ 60% of Var(H)"),
    ("condition_c_weights",     "Weight consistency ≤ 20% avg deviation"),
    ("dispersion_floor",        "σ(H) ≥ 0.0070"),
    ("within_symbol",           "σ_within ≥ 0.0030 for ≥ 2 symbols"),
    ("conviction_distribution", "Non-unscored ≥ 30%, medium+ ≥ 8%, spread ≥ 3 bands"),
    ("persistence",             "Consecutive sub-windows meeting criteria ≥ 3"),
]


def _resolve(data: Dict[str, Any], gate_name: str) -> bool:
    """Resolve a single gate to True/False from the decision-sheet JSON."""
    # Gates 0-4 live under .conditions
    if gate_name in data.get("conditions", {}):
        return bool(data["conditions"][gate_name])

    # Gate 5: dispersion floor
    if gate_name == "dispersion_floor":
        return bool(data.get("dispersion_floor", {}).get("pass", False))

    # Gate 6: within-symbol
    if gate_name == "within_symbol":
        return bool(data.get("within_symbol", {}).get("pass", False))

    # Gate 7: conviction distribution
    if gate_name == "conviction_distribution":
        return bool(data.get("conviction", {}).get("pass", False))

    # Gate 8: persistence
    if gate_name == "persistence":
        return bool(data.get("persistence", {}).get("pass", False))

    return False


def evaluate(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate all 9 gates against a decision-sheet JSON payload.

    Returns a structured verdict dict.
    """
    failed: List[str] = []
    summary: Dict[str, Dict[str, Any]] = {}

    for gate_name, description in GATES:
        passed = _resolve(data, gate_name)
        summary[gate_name] = {
            "pass": passed,
            "description": description,
        }
        if not passed:
            failed.append(gate_name)

    activation = "GO" if not failed else "NO_GO"

    # Data sufficiency annotations — attribute failures to insufficient data
    # rather than structural imbalance when applicable.
    ds = data.get("data_sufficiency", {})
    insufficient_data: List[str] = []
    if not ds.get("weights_sufficient", True) and "condition_c_weights" in failed:
        insufficient_data.append("condition_c_weights")
    if not ds.get("persistence_sufficient", True) and "persistence" in failed:
        insufficient_data.append("persistence")

    return {
        "binary_lab_activation": activation,
        "gates_evaluated": len(GATES),
        "gates_passed": len(GATES) - len(failed),
        "failed_conditions": failed,
        "insufficient_data": insufficient_data,
        "summary": summary,
    }


def main() -> None:
    """Read JSON from stdin, evaluate, print result, set exit code."""
    raw = sys.stdin.read().strip()
    if not raw:
        print(json.dumps({"error": "No input received on stdin"}), file=sys.stderr)
        sys.exit(2)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"Invalid JSON: {exc}"}), file=sys.stderr)
        sys.exit(2)

    result = evaluate(data)

    # Human-friendly header on stderr
    n_pass = result["gates_passed"]
    n_total = result["gates_evaluated"]
    verdict = result["binary_lab_activation"]
    marker = "✓" if verdict == "GO" else "✗"
    print(f"\n  Binary Lab Gate: {marker} {verdict}  ({n_pass}/{n_total} gates passed)", file=sys.stderr)
    if result["failed_conditions"]:
        print(f"  Failed: {', '.join(result['failed_conditions'])}", file=sys.stderr)
    if result.get("insufficient_data"):
        print(f"  Insufficient data: {', '.join(result['insufficient_data'])}", file=sys.stderr)
    print(file=sys.stderr)

    # Machine-readable JSON on stdout
    print(json.dumps(result, indent=2))

    sys.exit(0 if verdict == "GO" else 1)


if __name__ == "__main__":
    main()
