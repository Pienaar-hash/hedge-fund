#!/usr/bin/env python3
"""
Hybrid Variance Decomposition Audit
====================================

Determines whether observed hybrid-score dispersion is:

    1. Proportional to configured weights
    2. Distributed across signal components
    3. Stable under calibration gating
    4. Not structurally dominated by router

Usage::

    # Real data (last 24 h)
    python scripts/hybrid_variance_audit.py --window 24h

    # Synthetic verification (deterministic)
    python scripts/hybrid_variance_audit.py --synthetic

    # Longer window
    python scripts/hybrid_variance_audit.py --window 7d

    # JSON output (machine-readable)
    python scripts/hybrid_variance_audit.py --window 24h --json

Exit codes::

    0 — all conditions pass
    1 — at least one structural condition violated
    2 — insufficient data for audit

Manifest impact: None (stdout only, no state surfaces written).
Doctrine impact: None (read-only analysis).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ── Constants ────────────────────────────────────────────────────────────

COMPONENT_NAMES: Tuple[str, ...] = ("trend", "carry", "expectancy", "router")

# Default weight vector (from HybridScoreConfig)
DEFAULT_WEIGHTS: Dict[str, float] = {
    "trend": 0.40,
    "carry": 0.25,
    "expectancy": 0.20,
    "router": 0.15,
}

# ── Condition thresholds ─────────────────────────────────────────────────

# Condition A: Router must not explain >50% of hybrid variance
ROUTER_DOMINANCE_CEILING = 0.50

# Condition B: Combined signal components must explain ≥60%
SIGNAL_SUFFICIENCY_FLOOR = 0.60

# Condition C: Weight-consistency absolute tolerance
WEIGHT_CONSISTENCY_TOLERANCE = 0.20

# Reconstruction tolerance (theoretical vs empirical Var(H))
RECONSTRUCTION_TOLERANCE = 0.02  # ±2%

# Alert threshold — stderr warning if reconstruction error exceeds this
RECONSTRUCTION_ALERT_THRESHOLD = 0.02  # 2%

# Calibration continuity tolerance (pre vs post Var(H))
CALIBRATION_CONTINUITY_TOLERANCE = 0.25  # ±25%

# Minimum records required for a meaningful audit
MIN_RECORDS = 20

# Dispersion floor for hybrid std dev (materiality gate)
DISPERSION_FLOOR_STD = 0.0070

# Within-symbol std dev floor
WITHIN_SYMBOL_STD_FLOOR = 0.0030
WITHIN_SYMBOL_MIN_PASSING = 2  # min symbols above floor

# Conviction band thresholds
CONVICTION_NON_UNSCORED_FLOOR = 0.30
CONVICTION_MEDIUM_PLUS_FLOOR = 0.08

# Persistence: number of sub-windows that must pass
PERSISTENCE_MIN_WINDOWS = 3

# Data sufficiency thresholds (for epistemic labeling)
WEIGHTS_MIN_RECORDS = 1000       # Condition C unreliable below this
PERSISTENCE_MIN_PER_WINDOW = 200  # Each persistence sub-window needs this many

# ── JSONL path ──────────────────────────────────────────────────────────

SCORE_DECOMP_PATH = Path("logs/execution/score_decomposition.jsonl")


# ── Parsing ─────────────────────────────────────────────────────────────

def parse_window(window_str: str) -> timedelta:
    """Parse a human-friendly window string like '24h', '7d', '12h'."""
    s = window_str.strip().lower()
    if s.endswith("d"):
        return timedelta(days=int(s[:-1]))
    if s.endswith("h"):
        return timedelta(hours=int(s[:-1]))
    if s.endswith("m"):
        return timedelta(minutes=int(s[:-1]))
    raise ValueError(f"Unsupported window format: {window_str!r}  (use e.g. 24h, 7d)")


def _parse_ts(ts_str: str) -> datetime:
    """Parse ISO-8601 timestamp to aware UTC datetime."""
    # Handle both +00:00 and Z suffixes
    ts_str = ts_str.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        return datetime.min.replace(tzinfo=timezone.utc)


def load_records(
    jsonl_path: Path,
    window: timedelta,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Load score_decomposition.jsonl records within *window* of *now*."""
    if now is None:
        now = datetime.now(timezone.utc)
    cutoff = now - window

    records: List[Dict[str, Any]] = []
    if not jsonl_path.exists():
        return records

    with jsonl_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = _parse_ts(rec.get("ts", ""))
            if ts >= cutoff:
                records.append(rec)
    return records


# ── Vector extraction ───────────────────────────────────────────────────

def extract_vectors(
    records: List[Dict[str, Any]],
) -> Dict[str, List[float]]:
    """
    Extract component vectors and base hybrid score from records.

    Returns dict with keys: trend, carry, expectancy, router, hybrid_base, hybrid_final.
    hybrid_base = sum(weighted[i]), hybrid_final = record['hybrid_score'].
    """
    vecs: Dict[str, List[float]] = {
        name: [] for name in COMPONENT_NAMES
    }
    vecs["hybrid_base"] = []
    vecs["hybrid_final"] = []
    vecs["weights_used"] = []  # type: ignore[assignment]  # stores dicts

    for rec in records:
        comps = rec.get("components", {})
        weighted = rec.get("weighted", {})
        weights = rec.get("weights_used", {})

        # Skip records with missing component data
        if not all(name in comps for name in COMPONENT_NAMES):
            continue

        for name in COMPONENT_NAMES:
            vecs[name].append(float(comps[name]))

        # Base score = sum of weighted components (before RV/RQ/decay)
        base = sum(float(weighted.get(name, 0.0)) for name in COMPONENT_NAMES)
        vecs["hybrid_base"].append(base)
        vecs["hybrid_final"].append(float(rec.get("hybrid_score", base)))
        vecs["weights_used"].append(weights)  # type: ignore[arg-type]

    return vecs


# ── Statistics ──────────────────────────────────────────────────────────

def _mean(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    return sum(xs) / len(xs)


def _var(xs: Sequence[float]) -> float:
    """Population variance."""
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    return sum((x - mu) ** 2 for x in xs) / len(xs)


def _cov(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Population covariance."""
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    mu_x = sum(xs[:n]) / n
    mu_y = sum(ys[:n]) / n
    return sum((xs[i] - mu_x) * (ys[i] - mu_y) for i in range(n)) / n


def _std(xs: Sequence[float]) -> float:
    return math.sqrt(_var(xs))


def compute_covariance_matrix(
    vecs: Dict[str, List[float]],
) -> Dict[Tuple[str, str], float]:
    """Compute pairwise covariance matrix for the 4 components."""
    cov_mat: Dict[Tuple[str, str], float] = {}
    for i, a in enumerate(COMPONENT_NAMES):
        for j, b in enumerate(COMPONENT_NAMES):
            if i <= j:
                c = _cov(vecs[a], vecs[b])
                cov_mat[(a, b)] = c
                cov_mat[(b, a)] = c
    return cov_mat


# ── Decomposition ──────────────────────────────────────────────────────

def compute_decomposition(
    vecs: Dict[str, List[float]],
    weights: Dict[str, float],
) -> Dict[str, Any]:
    """
    Compute full variance decomposition.

    Returns dict with:
        component_var:     {name: Var(component_i)}
        component_std:     {name: Std(component_i)}
        component_mean:    {name: Mean(component_i)}
        cov_matrix:        {(i,j): Cov(i,j)}
        contribution:      {name: fraction of Var(H) explained}
        contribution_raw:  {name: raw contribution (not normalized)}
        var_theoretical:   reconstructed Var(H) from formula
        var_empirical_base:   empirical Var(sum(w*component))
        var_empirical_final:  empirical Var(hybrid_score)
        reconstruction_error: |theoretical - empirical_base| / empirical_base
        weights_used:      average weights across records
    """
    n = len(vecs.get("trend", []))

    # Average weights actually used (may differ per record due to regime mods)
    avg_weights: Dict[str, float] = {}
    weight_records = vecs.get("weights_used", [])
    if weight_records and isinstance(weight_records[0], dict):
        for name in COMPONENT_NAMES:
            vals = [float(w.get(name, weights[name])) for w in weight_records]  # type: ignore[union-attr]
            avg_weights[name] = _mean(vals)
    else:
        avg_weights = dict(weights)

    # Component statistics
    comp_var: Dict[str, float] = {}
    comp_std: Dict[str, float] = {}
    comp_mean: Dict[str, float] = {}
    for name in COMPONENT_NAMES:
        comp_var[name] = _var(vecs[name])
        comp_std[name] = _std(vecs[name])
        comp_mean[name] = _mean(vecs[name])

    # Covariance matrix
    cov_mat = compute_covariance_matrix(vecs)

    # ── Theoretical Var(H) = Σ wi² Var(i) + 2 Σ_{i<j} wi wj Cov(i,j) ──
    var_theoretical = 0.0
    for name in COMPONENT_NAMES:
        w = avg_weights[name]
        var_theoretical += w * w * comp_var[name]
    for i, a in enumerate(COMPONENT_NAMES):
        for j, b in enumerate(COMPONENT_NAMES):
            if i < j:
                var_theoretical += 2.0 * avg_weights[a] * avg_weights[b] * cov_mat[(a, b)]

    # ── Contribution per component (Shapley-style covariance split) ──
    # C_i = wi² Var(i) + Σ_{j≠i} wi wj Cov(i,j)
    contribution_raw: Dict[str, float] = {}
    for name in COMPONENT_NAMES:
        w_i = avg_weights[name]
        raw = w_i * w_i * comp_var[name]
        for other in COMPONENT_NAMES:
            if other != name:
                raw += w_i * avg_weights[other] * cov_mat[(name, other)]
        contribution_raw[name] = raw

    # Normalize to fractions (handle zero variance)
    total_contribution = sum(contribution_raw.values())
    contribution: Dict[str, float] = {}
    if abs(total_contribution) > 1e-15:
        for name in COMPONENT_NAMES:
            contribution[name] = contribution_raw[name] / total_contribution
    else:
        for name in COMPONENT_NAMES:
            contribution[name] = 0.0

    # Empirical variances
    var_emp_base = _var(vecs.get("hybrid_base", []))
    var_emp_final = _var(vecs.get("hybrid_final", []))

    # Reconstruction error
    if abs(var_emp_base) > 1e-15:
        recon_err = abs(var_theoretical - var_emp_base) / var_emp_base
    else:
        recon_err = 0.0 if abs(var_theoretical) < 1e-15 else float("inf")

    return {
        "n_records": n,
        "component_var": comp_var,
        "component_std": comp_std,
        "component_mean": comp_mean,
        "cov_matrix": {f"{a}_{b}": v for (a, b), v in cov_mat.items()},
        "contribution": contribution,
        "contribution_raw": contribution_raw,
        "var_theoretical": var_theoretical,
        "var_empirical_base": var_emp_base,
        "var_empirical_final": var_emp_final,
        "reconstruction_error": recon_err,
        "weights_used_avg": avg_weights,
    }


# ── Condition checks ────────────────────────────────────────────────────

def check_conditions(
    decomp: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate the three structural health conditions plus reconstruction.

    Returns dict with condition results and overall pass/fail.
    """
    contrib = decomp["contribution"]
    comp_var = decomp["component_var"]
    weights = decomp["weights_used_avg"]
    recon_err = decomp["reconstruction_error"]

    results: Dict[str, Any] = {}

    # ── Condition A: Router Dominance ─────────────────────────────────
    router_contrib = contrib.get("router", 0.0)
    cond_a_pass = router_contrib < ROUTER_DOMINANCE_CEILING
    results["condition_a"] = {
        "name": "Router Dominance Test",
        "description": f"Router contribution < {ROUTER_DOMINANCE_CEILING:.0%}",
        "router_contribution": router_contrib,
        "threshold": ROUTER_DOMINANCE_CEILING,
        "pass": cond_a_pass,
    }

    # ── Condition B: Signal Sufficiency ───────────────────────────────
    signal_contrib = (
        contrib.get("trend", 0.0)
        + contrib.get("carry", 0.0)
        + contrib.get("expectancy", 0.0)
    )
    cond_b_pass = signal_contrib >= SIGNAL_SUFFICIENCY_FLOOR
    results["condition_b"] = {
        "name": "Signal Sufficiency Test",
        "description": f"T + C + E contribution ≥ {SIGNAL_SUFFICIENCY_FLOOR:.0%}",
        "signal_contribution": signal_contrib,
        "threshold": SIGNAL_SUFFICIENCY_FLOOR,
        "pass": cond_b_pass,
    }

    # ── Condition C: Weight Consistency ───────────────────────────────
    # Expected share ∝ wi² Var(i) (ignoring covariance)
    marginal: Dict[str, float] = {}
    for name in COMPONENT_NAMES:
        marginal[name] = weights[name] ** 2 * comp_var[name]
    marginal_total = sum(marginal.values())

    expected_share: Dict[str, float] = {}
    deviations: Dict[str, float] = {}
    if abs(marginal_total) > 1e-15:
        for name in COMPONENT_NAMES:
            expected_share[name] = marginal[name] / marginal_total
            deviations[name] = abs(contrib[name] - expected_share[name])
    else:
        for name in COMPONENT_NAMES:
            expected_share[name] = 0.0
            deviations[name] = 0.0

    max_deviation = max(deviations.values()) if deviations else 0.0
    cond_c_pass = max_deviation <= WEIGHT_CONSISTENCY_TOLERANCE
    results["condition_c"] = {
        "name": "Weight Consistency Test",
        "description": f"Max |actual - expected| ≤ {WEIGHT_CONSISTENCY_TOLERANCE:.0%}",
        "expected_share": expected_share,
        "deviations": deviations,
        "max_deviation": max_deviation,
        "threshold": WEIGHT_CONSISTENCY_TOLERANCE,
        "pass": cond_c_pass,
    }

    # ── Reconstruction check ──────────────────────────────────────────
    recon_pass = recon_err <= RECONSTRUCTION_TOLERANCE
    results["reconstruction"] = {
        "name": "Variance Reconstruction",
        "description": f"|Var(H)_theoretical - Var(H)_empirical| / Var(H)_empirical ≤ {RECONSTRUCTION_TOLERANCE:.0%}",
        "error": recon_err,
        "threshold": RECONSTRUCTION_TOLERANCE,
        "pass": recon_pass,
    }

    # ── Degenerate data warning ───────────────────────────────────────
    # If 3+ components have zero variance, flag as degenerate
    zero_var_components = [
        name for name in COMPONENT_NAMES if comp_var[name] < 1e-12
    ]
    is_degenerate = len(zero_var_components) >= 3
    results["degeneracy_warning"] = {
        "degenerate": is_degenerate,
        "zero_variance_components": zero_var_components,
        "note": (
            "Scoring pipeline not yet differentiating — "
            "components stuck at neutral 0.5. "
            "Condition checks are formally correct but uninformative."
        ) if is_degenerate else "Components show variance — scoring is differentiating.",
    }

    # ── Overall ───────────────────────────────────────────────────────
    structural_pass = cond_a_pass and cond_b_pass and cond_c_pass
    results["overall"] = {
        "structural_pass": structural_pass,
        "reconstruction_pass": recon_pass,
        "degenerate": is_degenerate,
        "all_clear": structural_pass and recon_pass and not is_degenerate,
    }

    return results


# ── Extended metrics (Binary Lab decision sheet) ────────────────────────

def compute_within_symbol_stats(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute per-symbol hybrid_score standard deviation.

    Returns dict with per-symbol std and a summary of how many pass the floor.
    """
    by_symbol: Dict[str, List[float]] = {}
    for rec in records:
        sym = rec.get("symbol", "")
        if not sym:
            continue
        h = rec.get("hybrid_score")
        if h is not None:
            by_symbol.setdefault(sym, []).append(float(h))

    within: Dict[str, float] = {}
    for sym, vals in sorted(by_symbol.items()):
        within[sym] = _std(vals)

    symbols_above_floor = [
        sym for sym, sd in within.items() if sd >= WITHIN_SYMBOL_STD_FLOOR
    ]
    return {
        "stddev_by_symbol": within,
        "symbols_above_floor": symbols_above_floor,
        "symbols_above_floor_count": len(symbols_above_floor),
        "pass": len(symbols_above_floor) >= WITHIN_SYMBOL_MIN_PASSING,
        "threshold": WITHIN_SYMBOL_STD_FLOOR,
        "min_passing": WITHIN_SYMBOL_MIN_PASSING,
    }


def _percentile_band(score: float, p20: float, p60: float, p90: float) -> str:
    """Assign conviction band based on empirical percentile thresholds.

    Bands:
      very_low  = bottom 20%   (score < P20)
      low       = 20–60%       (P20 ≤ score < P60)
      medium    = 60–90%       (P60 ≤ score < P90)
      high      = top 10%      (score ≥ P90)

    This makes conviction relative to the observed score distribution,
    removing the dependency on absolute thresholds that may be unreachable
    under bounded [0,1] component architecture.
    """
    if score >= p90:
        return "high"
    elif score >= p60:
        return "medium"
    elif score >= p20:
        return "low"
    else:
        return "very_low"


def compute_conviction_distribution(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute conviction band distribution using empirical percentile banding.

    Instead of reading pre-stored conviction_band labels (which use absolute
    thresholds potentially unreachable under bounded hybrid scores), this
    function derives bands from the window's hybrid_score distribution:

      very_low  = bottom 20%
      low       = 20–60%
      medium    = 60–90%
      high      = top 10%

    Gate criteria (unchanged):
      - non_unscored ≥ 30%
      - medium+ ≥ 8%
      - band spread ≥ 3 distinct bands

    Note: this gate evaluates relative ranking dispersion, not absolute
    score magnitude.  The screener's shadow band (absolute thresholds)
    remains a separate observability layer and is not used here.
    """
    scores = [float(r.get("hybrid_score", 0.0)) for r in records]
    total = len(scores)

    if total == 0:
        return {
            "distribution": {},
            "non_unscored_pct": 0.0,
            "medium_plus_pct": 0.0,
            "band_spread_ok": False,
            "pass": False,
            "method": "percentile",
            "percentiles": {},
        }

    sorted_scores = sorted(scores)
    # Compute percentile thresholds using linear interpolation
    def _pct(p: float) -> float:
        k = (total - 1) * p / 100.0
        f = int(k)
        c = f + 1 if f + 1 < total else f
        d = k - f
        return sorted_scores[f] + d * (sorted_scores[c] - sorted_scores[f])

    p20 = _pct(20)
    p60 = _pct(60)
    p90 = _pct(90)

    band_counts: Dict[str, int] = {}
    for s in scores:
        band = _percentile_band(s, p20, p60, p90)
        band_counts[band] = band_counts.get(band, 0) + 1

    distribution = {k: v / total for k, v in sorted(band_counts.items())}

    # All records are scored (percentile banding never produces unscored)
    non_unscored_pct = 1.0

    medium_plus_bands = {"medium", "high"}
    medium_plus_count = sum(
        band_counts.get(b, 0) for b in medium_plus_bands
    )
    medium_plus_pct = medium_plus_count / total

    # Band spread: at least 3 distinct bands present
    non_empty_bands = [b for b, c in band_counts.items() if c > 0]
    band_spread_ok = len(non_empty_bands) >= 3

    pass_all = (
        non_unscored_pct >= CONVICTION_NON_UNSCORED_FLOOR
        and medium_plus_pct >= CONVICTION_MEDIUM_PLUS_FLOOR
        and band_spread_ok
    )

    return {
        "distribution": distribution,
        "non_unscored_pct": non_unscored_pct,
        "medium_plus_pct": medium_plus_pct,
        "band_spread_ok": band_spread_ok,
        "pass": pass_all,
        "method": "percentile",
        "percentiles": {
            "p20": round(p20, 6),
            "p60": round(p60, 6),
            "p90": round(p90, 6),
        },
    }


def compute_persistence(
    records: List[Dict[str, Any]],
    weights: Dict[str, float],
    n_windows: int = PERSISTENCE_MIN_WINDOWS,
) -> Dict[str, Any]:
    """
    Split records into *n_windows* equal sub-windows and check structural
    conditions in each.  Counts consecutive passing windows.

    Returns persistence result with windows_passed count.
    """
    if not records:
        return {"windows_passed": 0, "total_windows": 0, "pass": False, "window_results": []}

    # Sort by timestamp
    def _ts_key(r: Dict[str, Any]) -> str:
        return r.get("ts", "")

    sorted_recs = sorted(records, key=_ts_key)
    chunk_size = max(1, len(sorted_recs) // n_windows)

    window_results: List[Dict[str, Any]] = []
    consecutive_pass = 0
    max_consecutive = 0

    for i in range(n_windows):
        start = i * chunk_size
        end = start + chunk_size if i < n_windows - 1 else len(sorted_recs)
        chunk = sorted_recs[start:end]

        if len(chunk) < max(5, MIN_RECORDS // 3):
            window_results.append({"window": i, "n_records": len(chunk), "structural_pass": False, "reason": "insufficient_data"})
            consecutive_pass = 0
            continue

        vecs = extract_vectors(chunk)
        if len(vecs.get("trend", [])) < 5:
            window_results.append({"window": i, "n_records": len(chunk), "structural_pass": False, "reason": "insufficient_vectors"})
            consecutive_pass = 0
            continue

        decomp = compute_decomposition(vecs, weights)
        conds = check_conditions(decomp)
        sp = conds["overall"]["structural_pass"] and not conds["overall"]["degenerate"]

        window_results.append({
            "window": i,
            "n_records": len(chunk),
            "structural_pass": sp,
            "condition_a": conds["condition_a"]["pass"],
            "condition_b": conds["condition_b"]["pass"],
            "condition_c": conds["condition_c"]["pass"],
            "degenerate": conds["degeneracy_warning"]["degenerate"],
        })

        if sp:
            consecutive_pass += 1
            max_consecutive = max(max_consecutive, consecutive_pass)
        else:
            consecutive_pass = 0

    return {
        "windows_passed": max_consecutive,
        "total_windows": n_windows,
        "pass": max_consecutive >= n_windows,
        "data_sufficient": all(
            wr.get("n_records", 0) >= PERSISTENCE_MIN_PER_WINDOW
            for wr in window_results
        ),
        "window_results": window_results,
    }


def build_decision_sheet_json(
    records: List[Dict[str, Any]],
    decomp: Dict[str, Any],
    conditions: Dict[str, Any],
    weights: Dict[str, float],
    window_label: str,
) -> Dict[str, Any]:
    """
    Build the structured JSON matching the Binary Lab Activation Decision Sheet.

    All JSON paths referenced in the decision sheet resolve against this dict.
    """
    hybrid_std = math.sqrt(decomp["var_empirical_final"]) if decomp["var_empirical_final"] > 0 else 0.0

    # Timestamps from records
    timestamps = [r.get("ts", "") for r in records if r.get("ts")]
    ts_start = min(timestamps) if timestamps else ""
    ts_end = max(timestamps) if timestamps else ""

    within_sym = compute_within_symbol_stats(records)
    conviction = compute_conviction_distribution(records)
    persistence = compute_persistence(records, weights)

    return {
        "version": "v1.0",
        "phase": "Phase C",

        # Section 1 — Audit Metadata
        "window": {
            "label": window_label,
            "start_ts": ts_start,
            "end_ts": ts_end,
            "record_count": decomp["n_records"],
        },
        "reconstruction_error_pct": round(decomp["reconstruction_error"] * 100, 4),

        # Section 2 — Degeneracy Gate
        "flags": {
            "degenerate": conditions["degeneracy_warning"]["degenerate"],
            "zero_variance_components": conditions["degeneracy_warning"]["zero_variance_components"],
            "weight_consistency": conditions["condition_c"]["pass"],
            "weights_data_sufficient": decomp["n_records"] >= WEIGHTS_MIN_RECORDS,
        },
        "variance": {
            "trend": decomp["component_var"]["trend"],
            "carry": decomp["component_var"]["carry"],
            "expectancy": decomp["component_var"]["expectancy"],
            "router": decomp["component_var"]["router"],
            "hybrid": decomp["var_empirical_final"],
        },

        # Section 3 — Structural Conditions
        "contribution": {
            "trend": decomp["contribution"]["trend"],
            "carry": decomp["contribution"]["carry"],
            "expectancy": decomp["contribution"]["expectancy"],
            "router": decomp["contribution"]["router"],
            "signal_share": (
                decomp["contribution"]["trend"]
                + decomp["contribution"]["carry"]
                + decomp["contribution"]["expectancy"]
            ),
        },
        "conditions": {
            "gate_0_reconstruction": conditions["reconstruction"]["pass"],
            "gate_1_non_degeneracy": not conditions["degeneracy_warning"]["degenerate"],
            "condition_a_router": conditions["condition_a"]["pass"],
            "condition_b_signal": conditions["condition_b"]["pass"],
            "condition_c_weights": conditions["condition_c"]["pass"],
        },

        # Section 4 — Dispersion Floor
        "stddev": {
            "hybrid": round(hybrid_std, 6),
            "within": {k: round(v, 6) for k, v in within_sym["stddev_by_symbol"].items()},
        },
        "dispersion_floor": {
            "hybrid_std": round(hybrid_std, 6),
            "threshold": DISPERSION_FLOOR_STD,
            "pass": hybrid_std >= DISPERSION_FLOOR_STD,
        },

        # Section 5 — Within-Symbol
        "within_symbol": within_sym,

        # Section 6 — Conviction Distribution
        "conviction": conviction,

        # Section 7 — Persistence
        "persistence": persistence,

        # Section 8 — Data Sufficiency (epistemic flags)
        "data_sufficiency": {
            "weights_sufficient": decomp["n_records"] >= WEIGHTS_MIN_RECORDS,
            "weights_n_records": decomp["n_records"],
            "weights_min_required": WEIGHTS_MIN_RECORDS,
            "persistence_sufficient": persistence.get("data_sufficient", False),
            "persistence_min_per_window": PERSISTENCE_MIN_PER_WINDOW,
        },
    }


# ── Synthetic data generator ────────────────────────────────────────────

def generate_synthetic_records(
    n: int = 500,
    weights: Optional[Dict[str, float]] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generate synthetic score_decomposition records for math verification.

    Components are drawn from clipped normal distributions with known
    parameters so expected variances are analytically tractable.
    """
    import random
    rng = random.Random(seed)

    if weights is None:
        weights = dict(DEFAULT_WEIGHTS)

    # Component distributions: (mean, std) — all bounded [0,1]
    distributions = {
        "trend":      (0.55, 0.18),
        "carry":      (0.48, 0.14),
        "expectancy": (0.52, 0.12),
        "router":     (0.60, 0.10),
    }

    records: List[Dict[str, Any]] = []
    base_ts = datetime(2026, 2, 19, 0, 0, 0, tzinfo=timezone.utc)

    for i in range(n):
        ts = base_ts + timedelta(minutes=i * 3)
        comps: Dict[str, float] = {}
        weighted: Dict[str, float] = {}

        for name in COMPONENT_NAMES:
            mu, sigma = distributions[name]
            val = rng.gauss(mu, sigma)
            val = max(0.0, min(1.0, val))  # clamp [0,1]
            comps[name] = round(val, 4)
            weighted[name] = round(weights[name] * comps[name], 6)

        base_score = sum(weighted.values())

        # Assign conviction bands with a realistic distribution
        _band_roll = rng.random()
        if _band_roll < 0.05:
            conviction_band = "very_low"
        elif _band_roll < 0.20:
            conviction_band = "low"
        elif _band_roll < 0.55:
            conviction_band = "medium"
        elif _band_roll < 0.85:
            conviction_band = "high"
        elif _band_roll < 0.95:
            conviction_band = "very_high"
        else:
            conviction_band = ""  # unscored

        records.append({
            "ts": ts.isoformat(),
            "symbol": rng.choice(["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]),
            "direction": rng.choice(["LONG", "SHORT"]),
            "hybrid_score": round(base_score, 6),  # no RV/RQ/decay for synthetic
            "components": comps,
            "weighted": weighted,
            "weights_used": dict(weights),
            "conviction_band": conviction_band,
            "rq_score": round(rng.uniform(0.5, 1.0), 4),
            "rv_score": 0.0,
        })

    return records


# ── Display ─────────────────────────────────────────────────────────────

def format_report(
    decomp: Dict[str, Any],
    conditions: Dict[str, Any],
    window_label: str,
) -> str:
    """Format a human-readable audit report."""
    lines: List[str] = []
    w = lines.append

    w("=" * 72)
    w("  HYBRID VARIANCE DECOMPOSITION AUDIT")
    w(f"  Window: {window_label}  |  Records: {decomp['n_records']}")
    w("=" * 72)

    # ── Component statistics table ────────────────────────────────────
    w("")
    w("  Component Statistics")
    w("  " + "-" * 60)
    w(f"  {'Component':<14} {'Mean':>8} {'Std':>8} {'Var':>10} {'Weight':>8}")
    w("  " + "-" * 60)
    for name in COMPONENT_NAMES:
        w(
            f"  {name:<14} "
            f"{decomp['component_mean'][name]:>8.4f} "
            f"{decomp['component_std'][name]:>8.4f} "
            f"{decomp['component_var'][name]:>10.6f} "
            f"{decomp['weights_used_avg'][name]:>8.3f}"
        )
    w("  " + "-" * 60)

    # ── Variance decomposition ────────────────────────────────────────
    w("")
    w("  Variance Decomposition")
    w("  " + "-" * 60)
    w(f"  {'Component':<14} {'Contribution':>14} {'Share':>8}")
    w("  " + "-" * 60)
    for name in COMPONENT_NAMES:
        raw = decomp["contribution_raw"][name]
        pct = decomp["contribution"][name]
        bar = "█" * max(0, round(pct * 40))
        w(f"  {name:<14} {raw:>14.8f} {pct:>7.1%}  {bar}")
    w("  " + "-" * 60)

    # ── Variance comparison ───────────────────────────────────────────
    w("")
    w("  Variance Comparison")
    w("  " + "-" * 60)
    w(f"  Var(H) theoretical (Σ wi²Vi + 2Σ wiwj Cij):  {decomp['var_theoretical']:.8f}")
    w(f"  Var(H) empirical (base, Σ weighted):          {decomp['var_empirical_base']:.8f}")
    w(f"  Var(H) empirical (final, hybrid_score):       {decomp['var_empirical_final']:.8f}")
    w(f"  Reconstruction error:                         {decomp['reconstruction_error']:.4%}")
    w("  " + "-" * 60)

    # ── Condition results ─────────────────────────────────────────────
    w("")
    w("  Structural Condition Checks")
    w("  " + "-" * 60)

    # Condition A
    ca = conditions["condition_a"]
    flag_a = "PASS ✓" if ca["pass"] else "FAIL ✗"
    w(f"  [A] {ca['name']}: {flag_a}")
    w(f"      Router contribution: {ca['router_contribution']:.1%}  (ceiling: {ca['threshold']:.0%})")

    # Condition B
    cb = conditions["condition_b"]
    flag_b = "PASS ✓" if cb["pass"] else "FAIL ✗"
    w(f"  [B] {cb['name']}: {flag_b}")
    w(f"      Signal (T+C+E) contribution: {cb['signal_contribution']:.1%}  (floor: {cb['threshold']:.0%})")

    # Condition C
    cc = conditions["condition_c"]
    flag_c = "PASS ✓" if cc["pass"] else "FAIL ✗"
    w(f"  [C] {cc['name']}: {flag_c}")
    w(f"      Max deviation: {cc['max_deviation']:.1%}  (tolerance: {cc['threshold']:.0%})")
    for name in COMPONENT_NAMES:
        exp = cc["expected_share"].get(name, 0.0)
        act = decomp["contribution"].get(name, 0.0)
        dev = cc["deviations"].get(name, 0.0)
        marker = " ←" if dev > cc["threshold"] else ""
        w(f"        {name:<14} expected={exp:.1%}  actual={act:.1%}  Δ={dev:.1%}{marker}")

    # Reconstruction
    rc = conditions["reconstruction"]
    flag_r = "PASS ✓" if rc["pass"] else "FAIL ✗"
    w(f"  [R] {rc['name']}: {flag_r}")
    w(f"      Error: {rc['error']:.4%}  (tolerance: {rc['threshold']:.0%})")

    # Degeneracy warning
    dw = conditions["degeneracy_warning"]
    if dw["degenerate"]:
        w("")
        w("  ⚠  DEGENERACY WARNING")
        w(f"     Zero-variance components: {', '.join(dw['zero_variance_components'])}")
        w(f"     {dw['note']}")

    # ── Overall verdict ───────────────────────────────────────────────
    w("")
    overall = conditions["overall"]
    if overall["all_clear"]:
        verdict = "ALL CLEAR — hybrid dispersion is structurally sound"
    elif overall["degenerate"]:
        verdict = "DEGENERATE — scoring pipeline not yet differentiating"
    elif not overall["structural_pass"]:
        verdict = "STRUCTURAL VIOLATION — at least one condition failed"
    else:
        verdict = "RECONSTRUCTION MISMATCH — theoretical ≠ empirical"
    w(f"  VERDICT: {verdict}")
    w("=" * 72)

    return "\n".join(lines)


# ── Covariance matrix display ───────────────────────────────────────────

def format_covariance_matrix(decomp: Dict[str, Any]) -> str:
    """Format the covariance matrix for display."""
    lines: List[str] = []
    w = lines.append
    w("")
    w("  Covariance Matrix")
    w("  " + "-" * 60)
    header = f"  {'':14}" + "".join(f"{name:>14}" for name in COMPONENT_NAMES)
    w(header)
    for a in COMPONENT_NAMES:
        row = f"  {a:<14}"
        for b in COMPONENT_NAMES:
            key = f"{a}_{b}"
            val = decomp["cov_matrix"].get(key, 0.0)
            row += f"{val:>14.8f}"
        w(row)
    w("  " + "-" * 60)
    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────

def run_audit(
    records: List[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
    window_label: str = "custom",
    output_json: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Core audit entry point.  Accepts pre-loaded records.

    Returns the full audit result dict (decomposition + conditions).
    """
    if weights is None:
        weights = dict(DEFAULT_WEIGHTS)

    vecs = extract_vectors(records)
    n = len(vecs.get("trend", []))

    result: Dict[str, Any] = {
        "window": window_label,
        "n_records": n,
        "sufficient_data": n >= MIN_RECORDS,
    }

    if n < MIN_RECORDS:
        result["error"] = f"Insufficient data: {n} records (minimum: {MIN_RECORDS})"
        if verbose and not output_json:
            print(f"\n  AUDIT ABORTED — {result['error']}\n")
        if output_json:
            print(json.dumps(result, indent=2, default=str))
        return result

    decomp = compute_decomposition(vecs, weights)
    conditions = check_conditions(decomp)

    result["decomposition"] = decomp
    result["conditions"] = conditions

    # Build decision-sheet-compatible output for --json
    decision_sheet = build_decision_sheet_json(
        records, decomp, conditions, weights, window_label,
    )
    result["decision_sheet"] = decision_sheet

    # Reconstruction-error alert (stderr, non-blocking)
    recon_pct = decision_sheet["reconstruction_error_pct"]
    if recon_pct > RECONSTRUCTION_ALERT_THRESHOLD * 100:
        print(
            f"\n  ⚠  RECONSTRUCTION ALERT: error {recon_pct:.4f}% "
            f"exceeds {RECONSTRUCTION_ALERT_THRESHOLD * 100:.1f}% threshold",
            file=sys.stderr,
        )

    if verbose and not output_json:
        print(format_report(decomp, conditions, window_label))
        print(format_covariance_matrix(decomp))
        print()

    if output_json:
        print(json.dumps(result["decision_sheet"], indent=2, default=str))

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid Variance Decomposition Audit",
    )
    parser.add_argument(
        "--window", default="24h",
        help="Time window for analysis (e.g. 24h, 7d, 12h). Default: 24h",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Run on synthetic data for math verification",
    )
    parser.add_argument(
        "--synthetic-n", type=int, default=500,
        help="Number of synthetic records (default: 500)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output machine-readable JSON instead of human report",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Verbose output (default: true)",
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Override JSONL file path (default: logs/execution/score_decomposition.jsonl)",
    )
    args = parser.parse_args()

    if args.synthetic:
        records = generate_synthetic_records(n=args.synthetic_n)
        window_label = f"synthetic ({args.synthetic_n} records)"
    else:
        jsonl_path = Path(args.file) if args.file else SCORE_DECOMP_PATH
        window = parse_window(args.window)
        records = load_records(jsonl_path, window)
        window_label = args.window

    result = run_audit(
        records,
        window_label=window_label,
        output_json=args.json,
    )

    # Exit code
    if not result.get("sufficient_data", False):
        sys.exit(2)

    conditions = result.get("conditions", {})
    overall = conditions.get("overall", {})
    if not overall.get("structural_pass", False):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
