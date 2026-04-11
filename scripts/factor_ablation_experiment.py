#!/usr/bin/env python3
"""Factor Ablation Grid — BTC Score Surface Decomposition.

Removes individual factors and factor-pairs from BTC hybrid scores,
then measures how each ablation affects score–PnL alignment.  Goal:
identify whether the harmful score displacement is driven by a single
factor or by a factor interaction (e.g. carry × trend).

Seven additive variants on scored BTC episodes:
  V0 — Baseline (original hybrid_score unchanged)
  V1 — −carry   (carry zeroed, remaining factors renormalized)
  V2 — −trend   (trend zeroed, remaining factors renormalized)
  V3 — −expectancy (expectancy zeroed, remaining factors renormalized)
  V4 — −carry−trend (both zeroed, remaining renormalized)
  V5 — −carry−expectancy (both zeroed, remaining renormalized)
  V6 — −trend−expectancy (both zeroed, remaining renormalized)

Factor estimation:
  - carry  : per-episode from funding/basis snapshots (directional)
  - trend  : BTC mean from score_decomposition.jsonl
  - expectancy : BTC mean from score_decomposition.jsonl
  - router : BTC mean from score_decomposition.jsonl

Ablation algebra:
  ablated = (hybrid − Σ(w_removed · f_removed)) / (1 − Σ(w_removed))

This is **observation-only** — no production scoring mutation.

Usage:
    PYTHONPATH=. python scripts/factor_ablation_experiment.py
    PYTHONPATH=. python scripts/factor_ablation_experiment.py --json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Reuse from carry_fix_experiment ─────────────────────────────────────────
from scripts.carry_fix_experiment import (
    MASK_MIDPOINT,
    NEAR_MISS_BAND,
    REFERENCE_MASK,
    _get_btc_carry_inputs,
    _kendall_tau,
    _pnl_stats,
    _quintile_spread,
    _spearman_rho,
    classify_region,
    compute_carry_score,
    load_basis_snapshot,
    load_episodes,
    load_funding_snapshot,
)

# ── Paths ───────────────────────────────────────────────────────────────────
SCORE_DECOMPOSITION = Path("logs/execution/score_decomposition.jsonl")

# ── Default weights (from HybridScoreConfig) ───────────────────────────────
DEFAULT_WEIGHTS: Dict[str, float] = {
    "trend": 0.40,
    "carry": 0.25,
    "expectancy": 0.20,
    "router": 0.15,
}

# ── Ablation variant specs ──────────────────────────────────────────────────
ABLATION_VARIANTS: List[Dict[str, Any]] = [
    {"label": "v0", "name": "baseline", "remove": []},
    {"label": "v1", "name": "−carry", "remove": ["carry"]},
    {"label": "v2", "name": "−trend", "remove": ["trend"]},
    {"label": "v3", "name": "−expectancy", "remove": ["expectancy"]},
    {"label": "v4", "name": "−carry−trend", "remove": ["carry", "trend"]},
    {"label": "v5", "name": "−carry−expectancy", "remove": ["carry", "expectancy"]},
    {"label": "v6", "name": "−trend−expectancy", "remove": ["trend", "expectancy"]},
]

VARIANT_LABELS = [v["label"] for v in ABLATION_VARIANTS]

# ── Reconstruction confidence tiers ────────────────────────────────────────
RMSE_STRONG = 0.03
RMSE_USABLE = 0.05

# ── Carry-fix crosscheck reference (from prior live run) ───────────────────
CARRY_FIX_REFERENCE = {
    "v0_tau": -0.34,
    "v1_tau_worsened": True,       # V1 τ did not improve in carry-fix
    "v1_midpoint_worsened": True,  # midpoint distance grew 0.030→0.044
    "v1_spillover_worsened": True, # spillover increased
}

# ── Interaction detection threshold ─────────────────────────────────────────
INTERACTION_THRESHOLD = 0.02


# ── Factor estimation from decomposition log ───────────────────────────────

def load_decomposition_means(
    path: Optional[Path] = None,
    symbol: str = "BTCUSDT",
) -> Dict[str, float]:
    """Load BTC factor means from score_decomposition.jsonl.

    Returns dict with keys: trend, carry, expectancy, router.
    If file missing or no data, returns neutral defaults (0.5 each).
    """
    p = path or SCORE_DECOMPOSITION
    neutral = {"trend": 0.5, "carry": 0.5, "expectancy": 0.5, "router": 0.5}
    if not p.exists():
        return neutral

    sums: Dict[str, float] = {"trend": 0.0, "carry": 0.0, "expectancy": 0.0, "router": 0.0}
    count = 0

    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if rec.get("symbol") != symbol:
                continue
            comps = rec.get("components", {})
            if not comps:
                continue
            for k in sums:
                sums[k] += float(comps.get(k, 0.5))
            count += 1

    if count == 0:
        return neutral

    return {k: sums[k] / count for k in sums}


def estimate_factor_values(
    direction: str,
    funding_rate: float,
    basis_pct: float,
    decomp_means: Dict[str, float],
) -> Dict[str, float]:
    """Estimate factor values for one BTC episode.

    carry is computed per-episode from funding/basis + direction.
    trend, expectancy, router use decomposition means.
    """
    carry = compute_carry_score(funding_rate, basis_pct, direction)
    return {
        "trend": decomp_means.get("trend", 0.5),
        "carry": carry,
        "expectancy": decomp_means.get("expectancy", 0.5),
        "router": decomp_means.get("router", 0.5),
    }


# ── Reconstruction quality ──────────────────────────────────────────────────

def reconstruct_score(
    factors: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Recompute hybrid score from factor values and weights."""
    w = weights or DEFAULT_WEIGHTS
    return sum(w.get(k, 0.0) * factors.get(k, 0.0) for k in w)


def reconstruction_quality(
    episodes: List[Dict[str, Any]],
    funding_rate: float,
    basis_pct: float,
    decomp_means: Dict[str, float],
) -> Dict[str, Any]:
    """Compare estimated Σ(w·f) vs actual hybrid_score per episode.

    Returns RMSE, max deviation, per-episode deltas, confidence tier.
    """
    deltas: List[Dict[str, Any]] = []
    sq_errors: List[float] = []

    for ep in episodes:
        h_score = ep.get("hybrid_score", 0)
        if not h_score or h_score <= 0:
            continue
        if ep.get("symbol") != "BTCUSDT":
            continue

        direction = ep.get("side", "LONG")
        factors = estimate_factor_values(direction, funding_rate, basis_pct, decomp_means)
        estimated = reconstruct_score(factors)
        delta = estimated - float(h_score)

        deltas.append({
            "episode_id": ep.get("episode_id", ""),
            "actual": round(float(h_score), 6),
            "estimated": round(estimated, 6),
            "delta": round(delta, 6),
        })
        sq_errors.append(delta ** 2)

    n = len(sq_errors)
    if n == 0:
        return {
            "n": 0,
            "rmse": 0.0,
            "max_deviation": 0.0,
            "mean_deviation": 0.0,
            "confidence_tier": "EXPLORATORY",
            "deltas": [],
        }

    rmse = math.sqrt(sum(sq_errors) / n)
    abs_deltas = [abs(d["delta"]) for d in deltas]
    max_dev = max(abs_deltas)
    mean_dev = sum(abs_deltas) / n

    if rmse < RMSE_STRONG:
        tier = "STRONG"
    elif rmse < RMSE_USABLE:
        tier = "USABLE"
    else:
        tier = "EXPLORATORY"

    return {
        "n": n,
        "rmse": round(rmse, 6),
        "max_deviation": round(max_dev, 6),
        "mean_deviation": round(mean_dev, 6),
        "confidence_tier": tier,
        "deltas": deltas,
    }


# ── Ablation engine ─────────────────────────────────────────────────────────

def ablate_score(
    hybrid_score: float,
    factors: Dict[str, float],
    weights: Dict[str, float],
    remove: List[str],
) -> float:
    """Remove specified factors and renormalize.

    ablated = (hybrid − Σ(w_removed · f_removed)) / (1 − Σ(w_removed))
    """
    if not remove:
        return hybrid_score

    w_removed_sum = sum(weights.get(k, 0.0) for k in remove)

    if w_removed_sum >= 1.0:
        return 0.0

    removed_contribution = sum(
        weights.get(k, 0.0) * factors.get(k, 0.0) for k in remove
    )

    return (hybrid_score - removed_contribution) / (1.0 - w_removed_sum)


# ── Grid computation ────────────────────────────────────────────────────────

def compute_ablation_grid(
    episodes: List[Dict[str, Any]],
    funding_rate: float,
    basis_pct: float,
    decomp_means: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Compute 7 ablation variants for scored BTC episodes.

    Returns list of enriched episode dicts with per-variant scores/regions.
    """
    w = weights or DEFAULT_WEIGHTS
    results: List[Dict[str, Any]] = []

    for ep in episodes:
        if ep.get("symbol") != "BTCUSDT":
            continue
        h_score = ep.get("hybrid_score", 0)
        if not h_score or h_score <= 0:
            continue
        pnl = ep.get("net_pnl")
        if pnl is None:
            continue

        direction = ep.get("side", "LONG")
        factors = estimate_factor_values(direction, funding_rate, basis_pct, decomp_means)

        row: Dict[str, Any] = {
            "episode_id": ep.get("episode_id", ""),
            "symbol": "BTCUSDT",
            "side": direction,
            "net_pnl": float(pnl),
            "entry_ts": ep.get("entry_ts", ""),
            "factors": {k: round(v, 6) for k, v in factors.items()},
        }

        for variant in ABLATION_VARIANTS:
            label = variant["label"]
            remove = variant["remove"]
            score = ablate_score(float(h_score), factors, w, remove)
            score = max(-1.0, min(1.0, score))
            row[f"{label}_score"] = round(score, 6)
            row[f"{label}_region"] = classify_region(score)

        results.append(row)

    return results


# ── Section 1: Reconstruction Quality ──────────────────────────────────────

def section_reconstruction(
    episodes: List[Dict[str, Any]],
    funding_rate: float,
    basis_pct: float,
    decomp_means: Dict[str, float],
) -> Dict[str, Any]:
    """Section 1: Reconstruction quality check."""
    quality = reconstruction_quality(episodes, funding_rate, basis_pct, decomp_means)
    return {"title": "Reconstruction Quality", **quality}


# ── Section 2: Carry-Only Crosscheck ───────────────────────────────────────

def section_carry_crosscheck(
    grid: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Section 2: Verify −carry ablation agrees directionally with carry-fix experiment.

    Checks sign/direction of τ change, midpoint distance, spillover pressure
    against the known carry-fix V1 results.
    """
    pnls = [r["net_pnl"] for r in grid]

    # V0 baseline
    v0_scores = [r["v0_score"] for r in grid]
    v0_tau = _kendall_tau(v0_scores, pnls)
    v0_regions = [r["v0_region"] for r in grid]
    v0_mean = sum(v0_scores) / len(v0_scores) if v0_scores else 0.0
    v0_midpoint_dist = abs(v0_mean - MASK_MIDPOINT)
    v0_in_mask = sum(1 for r in v0_regions if r == "mask_interior")
    v0_near_miss = sum(1 for r in v0_regions if r == "near_miss")
    v0_spillover = v0_near_miss / v0_in_mask if v0_in_mask > 0 else 0.0

    # V1 (−carry)
    v1_scores = [r["v1_score"] for r in grid]
    v1_tau = _kendall_tau(v1_scores, pnls)
    v1_regions = [r["v1_region"] for r in grid]
    v1_mean = sum(v1_scores) / len(v1_scores) if v1_scores else 0.0
    v1_midpoint_dist = abs(v1_mean - MASK_MIDPOINT)
    v1_in_mask = sum(1 for r in v1_regions if r == "mask_interior")
    v1_near_miss = sum(1 for r in v1_regions if r == "near_miss")
    v1_spillover = v1_near_miss / v1_in_mask if v1_in_mask > 0 else 0.0

    tau_worsened = v1_tau <= v0_tau
    midpoint_worsened = v1_midpoint_dist >= v0_midpoint_dist
    spillover_worsened = v1_spillover >= v0_spillover

    # Compare with carry-fix reference
    ref = CARRY_FIX_REFERENCE
    tau_agrees = tau_worsened == ref["v1_tau_worsened"]
    midpoint_agrees = midpoint_worsened == ref["v1_midpoint_worsened"]
    spillover_agrees = spillover_worsened == ref["v1_spillover_worsened"]

    all_agree = tau_agrees and midpoint_agrees and spillover_agrees
    status = "CROSSCHECK_PASS" if all_agree else "CROSSCHECK_FAIL"

    return {
        "title": "Carry-Only Crosscheck",
        "status": status,
        "v0_tau": round(v0_tau, 4),
        "v1_tau": round(v1_tau, 4),
        "tau_worsened": tau_worsened,
        "tau_agrees": tau_agrees,
        "v0_midpoint_dist": round(v0_midpoint_dist, 4),
        "v1_midpoint_dist": round(v1_midpoint_dist, 4),
        "midpoint_worsened": midpoint_worsened,
        "midpoint_agrees": midpoint_agrees,
        "v0_spillover": round(v0_spillover, 4),
        "v1_spillover": round(v1_spillover, 4),
        "spillover_worsened": spillover_worsened,
        "spillover_agrees": spillover_agrees,
    }


# ── Section 3: Correlation Grid ────────────────────────────────────────────

def section_correlation_grid(
    grid: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Section 3: Kendall τ, Spearman ρ, Q5−Q1 per variant."""
    pnls = [r["net_pnl"] for r in grid]
    result: Dict[str, Any] = {"n": len(grid)}

    for variant in ABLATION_VARIANTS:
        label = variant["label"]
        scores = [r[f"{label}_score"] for r in grid]
        result[label] = {
            "name": variant["name"],
            "kendall_tau": round(_kendall_tau(scores, pnls), 4),
            "spearman_rho": round(_spearman_rho(scores, pnls), 4),
            "q5_minus_q1": round(_quintile_spread(scores, pnls), 4),
        }

    return {"title": "Correlation Grid", **result}


# ── Section 4: Displacement Analysis ───────────────────────────────────────

def section_displacement(
    grid: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Section 4: Mean score, midpoint distance, spillover, %in mask per variant."""
    n = len(grid)
    result: Dict[str, Any] = {"n": n}

    for variant in ABLATION_VARIANTS:
        label = variant["label"]
        scores = [r[f"{label}_score"] for r in grid]
        regions = [r[f"{label}_region"] for r in grid]
        in_mask = sum(1 for r in regions if r == "mask_interior")
        near_miss = sum(1 for r in regions if r == "near_miss")
        spillover = near_miss / in_mask if in_mask > 0 else 0.0

        mean_score = sum(scores) / n if n else 0.0
        midpoint_dist = mean_score - MASK_MIDPOINT

        result[label] = {
            "name": variant["name"],
            "in_mask_pct": round(in_mask / n, 4) if n else 0.0,
            "in_mask_count": in_mask,
            "near_miss_count": near_miss,
            "spillover_pressure": round(spillover, 4),
            "mean_score": round(mean_score, 4),
            "midpoint_distance": round(midpoint_dist, 4),
        }

    return {"title": "Displacement Analysis", **result}


# ── Section 5: Economic Impact (with LONG/SHORT split) ─────────────────────

def section_economic(
    grid: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Section 5: EV by region per variant, with L/S direction split."""
    result: Dict[str, Any] = {}

    for variant in ABLATION_VARIANTS:
        label = variant["label"]
        region_key = f"{label}_region"

        # Overall EV by region
        regions: Dict[str, List[float]] = defaultdict(list)
        for r in grid:
            regions[r[region_key]].append(r["net_pnl"])

        region_stats = {}
        for region in ["mask_interior", "near_miss", "outside"]:
            region_stats[region] = _pnl_stats(regions.get(region, []))

        nm_pnls = regions.get("near_miss", [])
        abstention_savings = -sum(nm_pnls) if nm_pnls else 0.0

        # Direction split — LONG / SHORT inside mask
        long_mask_pnls = [
            r["net_pnl"] for r in grid
            if r[region_key] == "mask_interior" and r["side"] == "LONG"
        ]
        short_mask_pnls = [
            r["net_pnl"] for r in grid
            if r[region_key] == "mask_interior" and r["side"] == "SHORT"
        ]

        result[label] = {
            "name": variant["name"],
            "regions": region_stats,
            "abstention_savings": round(abstention_savings, 4),
            "long_mask": _pnl_stats(long_mask_pnls),
            "short_mask": _pnl_stats(short_mask_pnls),
        }

    return {"title": "Economic Impact", **result}


# ── Section 6: Interaction Detection ───────────────────────────────────────

def _delta_metric(
    grid: List[Dict[str, Any]],
    label: str,
    metric_fn: str,
) -> float:
    """Compute metric change vs V0 baseline for a given variant label."""
    pnls = [r["net_pnl"] for r in grid]
    v0_scores = [r["v0_score"] for r in grid]
    vx_scores = [r[f"{label}_score"] for r in grid]

    if metric_fn == "tau":
        return _kendall_tau(vx_scores, pnls) - _kendall_tau(v0_scores, pnls)
    elif metric_fn == "midpoint_distance":
        n = len(grid)
        v0_mean = sum(v0_scores) / n if n else 0.0
        vx_mean = sum(vx_scores) / n if n else 0.0
        return abs(vx_mean - MASK_MIDPOINT) - abs(v0_mean - MASK_MIDPOINT)
    elif metric_fn == "spillover":
        v0_regions = [r["v0_region"] for r in grid]
        vx_regions = [r[f"{label}_region"] for r in grid]

        def _spill(regions: List[str]) -> float:
            im = sum(1 for r in regions if r == "mask_interior")
            nm = sum(1 for r in regions if r == "near_miss")
            return nm / im if im > 0 else 0.0

        return _spill(vx_regions) - _spill(v0_regions)
    return 0.0


# Pair → single mapping
_PAIR_TO_SINGLES = {
    "v4": ("v1", "v2"),   # carry+trend → carry, trend
    "v5": ("v1", "v3"),   # carry+expectancy → carry, expectancy
    "v6": ("v2", "v3"),   # trend+expectancy → trend, expectancy
}


def section_interaction(
    grid: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Section 6: Detect interaction effects between ablated factor pairs.

    interaction_score = Δ_pair − (Δ_single_A + Δ_single_B)
    Positive = synergistic (pair effect exceeds sum of singles)
    Negative = antagonistic (factors compensate each other)
    """
    metrics = ["tau", "midpoint_distance", "spillover"]
    interactions: Dict[str, Any] = {}

    for pair_label, (single_a, single_b) in _PAIR_TO_SINGLES.items():
        pair_name = next(
            v["name"] for v in ABLATION_VARIANTS if v["label"] == pair_label
        )
        per_metric: Dict[str, Any] = {}

        for m in metrics:
            d_pair = _delta_metric(grid, pair_label, m)
            d_a = _delta_metric(grid, single_a, m)
            d_b = _delta_metric(grid, single_b, m)
            interaction = d_pair - (d_a + d_b)
            per_metric[m] = {
                "delta_pair": round(d_pair, 4),
                "delta_single_a": round(d_a, 4),
                "delta_single_b": round(d_b, 4),
                "interaction_score": round(interaction, 4),
                "classification": (
                    "synergistic" if interaction > INTERACTION_THRESHOLD
                    else "antagonistic" if interaction < -INTERACTION_THRESHOLD
                    else "independent"
                ),
            }

        interactions[pair_label] = {
            "name": pair_name,
            "metrics": per_metric,
        }

    # Find strongest interaction
    strongest_score = 0.0
    strongest_pair = ""
    strongest_metric = ""
    for pair_label, data in interactions.items():
        for m, mdata in data["metrics"].items():
            if abs(mdata["interaction_score"]) > abs(strongest_score):
                strongest_score = mdata["interaction_score"]
                strongest_pair = pair_label
                strongest_metric = m

    any_significant = abs(strongest_score) > INTERACTION_THRESHOLD

    return {
        "title": "Interaction Detection",
        "interactions": interactions,
        "strongest": {
            "pair": strongest_pair,
            "metric": strongest_metric,
            "score": round(strongest_score, 4),
            "significant": any_significant,
        },
    }


# ── Section 7: Verdict ─────────────────────────────────────────────────────

def section_verdict(
    recon: Dict[str, Any],
    crosscheck: Dict[str, Any],
    corr: Dict[str, Any],
    displacement: Dict[str, Any],
    economic: Dict[str, Any],
    interaction: Dict[str, Any],
) -> Dict[str, Any]:
    """Section 7: Final verdict, gated by reconstruction confidence."""
    confidence_tier = recon.get("confidence_tier", "EXPLORATORY")

    # Best single ablation (largest τ improvement among v1-v3)
    v0_tau = corr["v0"]["kendall_tau"]
    best_single_label = ""
    best_single_tau_delta = -999.0
    for label in ["v1", "v2", "v3"]:
        delta = corr[label]["kendall_tau"] - v0_tau
        if delta > best_single_tau_delta:
            best_single_tau_delta = delta
            best_single_label = label

    # Best pair ablation (largest τ improvement among v4-v6)
    best_pair_label = ""
    best_pair_tau_delta = -999.0
    for label in ["v4", "v5", "v6"]:
        delta = corr[label]["kendall_tau"] - v0_tau
        if delta > best_pair_tau_delta:
            best_pair_tau_delta = delta
            best_pair_label = label

    # Interaction detected?
    interaction_detected = interaction["strongest"]["significant"]

    # LONG/SHORT preservation — does best variant destroy SHORT profitability?
    best_overall = (
        best_pair_label if best_pair_tau_delta > best_single_tau_delta
        else best_single_label
    )
    short_mask_stats = economic.get(best_overall, {}).get("short_mask", {})
    short_preserved = (
        short_mask_stats.get("count", 0) > 0
        and short_mask_stats.get("mean", 0.0) > 0.0
    )

    # Crosscheck status
    crosscheck_ok = crosscheck.get("status") == "CROSSCHECK_PASS"

    # Recommendation
    if confidence_tier == "EXPLORATORY":
        recommendation = "INCONCLUSIVE"
        reason = "Reconstruction RMSE too high for reliable conclusions"
    elif not crosscheck_ok:
        recommendation = "INCONCLUSIVE"
        reason = "Carry-only crosscheck failed — factor estimation may be untrustworthy"
    elif interaction_detected and best_pair_tau_delta > best_single_tau_delta + 0.02:
        recommendation = "INTERACTION_DOMINANT"
        reason = (
            f"Pair ablation {best_pair_label} ({_variant_name(best_pair_label)}) "
            f"materially exceeds best single ablation "
            f"(Δτ_pair={best_pair_tau_delta:+.4f} vs Δτ_single={best_single_tau_delta:+.4f})"
        )
    elif best_single_tau_delta > 0.02:
        recommendation = "SINGLE_FACTOR_DOMINANT"
        reason = (
            f"Single ablation {best_single_label} ({_variant_name(best_single_label)}) "
            f"materially improves τ (Δτ={best_single_tau_delta:+.4f})"
        )
    else:
        recommendation = "INCONCLUSIVE"
        reason = "No ablation produces material τ improvement"

    return {
        "title": "Verdict",
        "confidence_tier": confidence_tier,
        "crosscheck_ok": crosscheck_ok,
        "best_single": {
            "label": best_single_label,
            "name": _variant_name(best_single_label),
            "tau_delta": round(best_single_tau_delta, 4),
        },
        "best_pair": {
            "label": best_pair_label,
            "name": _variant_name(best_pair_label),
            "tau_delta": round(best_pair_tau_delta, 4),
        },
        "interaction_detected": interaction_detected,
        "short_preserved": short_preserved,
        "recommendation": recommendation,
        "reason": reason,
    }


def _variant_name(label: str) -> str:
    for v in ABLATION_VARIANTS:
        if v["label"] == label:
            return v["name"]
    return label


# ── Report generation ──────────────────────────────────────────────────────

def generate_report(
    *,
    json_output: bool = False,
    episode_path: Optional[Path] = None,
    funding_path: Optional[Path] = None,
    basis_path: Optional[Path] = None,
    decomp_path: Optional[Path] = None,
    funding_rate_override: Optional[float] = None,
    basis_pct_override: Optional[float] = None,
) -> Dict[str, Any]:
    """Generate the full factor ablation experiment report."""
    episodes = load_episodes(episode_path)
    funding_snap = load_funding_snapshot(funding_path)
    basis_snap = load_basis_snapshot(basis_path)
    decomp_means = load_decomposition_means(decomp_path)

    funding_rate, basis_pct = _get_btc_carry_inputs(funding_snap, basis_snap)
    if funding_rate_override is not None:
        funding_rate = funding_rate_override
    if basis_pct_override is not None:
        basis_pct = basis_pct_override

    s_recon = section_reconstruction(episodes, funding_rate, basis_pct, decomp_means)
    grid = compute_ablation_grid(episodes, funding_rate, basis_pct, decomp_means)
    s_cross = section_carry_crosscheck(grid)
    s_corr = section_correlation_grid(grid)
    s_disp = section_displacement(grid)
    s_econ = section_economic(grid)
    s_inter = section_interaction(grid)
    s_verd = section_verdict(s_recon, s_cross, s_corr, s_disp, s_econ, s_inter)

    sections = [s_recon, s_cross, s_corr, s_disp, s_econ, s_inter, s_verd]

    report = {
        "generated_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "experiment": "Factor Ablation Grid",
        "symbol": "BTCUSDT",
        "factor_inputs": {
            "funding_rate": funding_rate,
            "basis_pct": basis_pct,
            "decomp_means": decomp_means,
            "weights": DEFAULT_WEIGHTS,
        },
        "total_episodes": len(grid),
        "sections": {s["title"]: s for s in sections},
    }

    if json_output:
        json.dump(report, sys.stdout, indent=2)
        print()
    else:
        _print_report(report, sections, grid)

    return report


# ── Text output ─────────────────────────────────────────────────────────────

def _print_report(
    report: Dict[str, Any],
    sections: List[Dict[str, Any]],
    grid: List[Dict[str, Any]],
) -> None:
    bar = "=" * 78
    now_iso = report["generated_at"]
    print(f"\n{bar}")
    print(f"  FACTOR ABLATION GRID — BTC Score Surface Decomposition")
    print(f"  Generated: {now_iso}")
    print(bar)

    fi = report["factor_inputs"]
    dm = fi["decomp_means"]
    print(f"\n  Factor Inputs:")
    print(f"    Funding rate (8h):   {fi['funding_rate']:.6f}")
    print(f"    Basis pct:           {fi['basis_pct']:.6f}")
    print(f"    Decomp means:        trend={dm['trend']:.4f}  carry={dm['carry']:.4f}  "
          f"expectancy={dm['expectancy']:.4f}  router={dm['router']:.4f}")
    print(f"    Weights:             trend={fi['weights']['trend']:.2f}  "
          f"carry={fi['weights']['carry']:.2f}  "
          f"expectancy={fi['weights']['expectancy']:.2f}  "
          f"router={fi['weights']['router']:.2f}")
    print(f"    BTC episodes:        {report['total_episodes']}")

    for s in sections:
        title = s["title"]
        print(f"\n{'─' * 78}")
        print(f"  {title}")
        print(f"{'─' * 78}")

        if title == "Reconstruction Quality":
            _print_reconstruction(s)
        elif title == "Carry-Only Crosscheck":
            _print_crosscheck(s)
        elif title == "Correlation Grid":
            _print_correlation_grid(s)
        elif title == "Displacement Analysis":
            _print_displacement(s)
        elif title == "Economic Impact":
            _print_economic(s)
        elif title == "Interaction Detection":
            _print_interaction(s)
        elif title == "Verdict":
            _print_verdict(s)

    # Episode table
    print(f"\n{'─' * 78}")
    print(f"  Episode-Level Score Comparison (7 variants)")
    print(f"{'─' * 78}")

    header = f"    {'EP':<10s} {'Side':<6s}"
    for v in ABLATION_VARIANTS:
        header += f" {v['label']:>7s}"
    header += f" {'PnL':>8s}"
    print(f"\n{header}")
    print(f"    {'─' * (len(header) - 4)}")
    for r in grid:
        line = f"    {r['episode_id']:<10s} {r['side']:<6s}"
        for v in ABLATION_VARIANTS:
            key = f"{v['label']}_score"
            line += f" {r[key]:7.4f}"
        line += f" {r['net_pnl']:8.4f}"
        print(line)

    print(f"\n{bar}")
    print(f"  END OF EXPERIMENT REPORT")
    print(bar)


def _print_reconstruction(s: Dict[str, Any]) -> None:
    print(f"\n    Episodes:          {s['n']}")
    print(f"    RMSE:              {s['rmse']:.6f}")
    print(f"    Max deviation:     {s['max_deviation']:.6f}")
    print(f"    Mean deviation:    {s['mean_deviation']:.6f}")
    print(f"    Confidence tier:   {s['confidence_tier']}")
    tier = s["confidence_tier"]
    if tier == "STRONG":
        print(f"    → RMSE < {RMSE_STRONG:.2f}: directional conclusions trustworthy")
    elif tier == "USABLE":
        print(f"    → RMSE < {RMSE_USABLE:.2f}: usable for hypothesis screening")
    else:
        print(f"    → RMSE ≥ {RMSE_USABLE:.2f}: interaction results advisory only")


def _print_crosscheck(s: Dict[str, Any]) -> None:
    print(f"\n    Status: {s['status']}")
    marker = lambda ok: "✓" if ok else "✗"  # noqa: E731
    print(f"    [{marker(s['tau_agrees'])}] τ direction:       "
          f"V0={s['v0_tau']:+.4f} → V1={s['v1_tau']:+.4f}  "
          f"worsened={s['tau_worsened']}  agrees={s['tau_agrees']}")
    print(f"    [{marker(s['midpoint_agrees'])}] midpoint dist:    "
          f"V0={s['v0_midpoint_dist']:.4f} → V1={s['v1_midpoint_dist']:.4f}  "
          f"worsened={s['midpoint_worsened']}  agrees={s['midpoint_agrees']}")
    print(f"    [{marker(s['spillover_agrees'])}] spillover:        "
          f"V0={s['v0_spillover']:.4f} → V1={s['v1_spillover']:.4f}  "
          f"worsened={s['spillover_worsened']}  agrees={s['spillover_agrees']}")


def _print_correlation_grid(s: Dict[str, Any]) -> None:
    print(f"\n    n = {s['n']}\n")
    print(f"    {'Label':<4s} {'Variant':<20s}  {'τ':>8s}  {'ρ':>8s}  {'Q5−Q1':>8s}")
    print(f"    {'─' * 54}")
    for v in ABLATION_VARIANTS:
        label = v["label"]
        d = s[label]
        print(f"    {label:<4s} {d['name']:<20s}  "
              f"{d['kendall_tau']:+8.4f}  {d['spearman_rho']:+8.4f}  "
              f"{d['q5_minus_q1']:+8.4f}")


def _print_displacement(s: Dict[str, Any]) -> None:
    mask_lo = REFERENCE_MASK["BTCUSDT"]["lo"]
    mask_hi = REFERENCE_MASK["BTCUSDT"]["hi"]
    print(f"\n    n = {s['n']}  mask = [{mask_lo:.4f}, {mask_hi:.4f}]  "
          f"midpoint = {MASK_MIDPOINT:.4f}\n")
    print(f"    {'Label':<4s} {'Variant':<20s}  {'%mask':>6s}  "
          f"{'spill':>8s}  {'mid_dist':>8s}  {'mean':>8s}")
    print(f"    {'─' * 62}")
    for v in ABLATION_VARIANTS:
        label = v["label"]
        d = s[label]
        print(f"    {label:<4s} {d['name']:<20s}  "
              f"{d['in_mask_pct']:6.2%}  {d['spillover_pressure']:8.4f}  "
              f"{d['midpoint_distance']:+8.4f}  {d['mean_score']:8.4f}")


def _print_economic(s: Dict[str, Any]) -> None:
    for v in ABLATION_VARIANTS:
        label = v["label"]
        d = s[label]
        print(f"\n    {label.upper()} ({d['name']}):")
        for region in ["mask_interior", "near_miss", "outside"]:
            rs = d["regions"].get(region, {})
            if rs.get("count", 0) == 0:
                print(f"      {region:<14s}  —")
            else:
                print(f"      {region:<14s}  n={rs['count']:3d}  "
                      f"EV={rs['mean']:+.4f}  win={rs['win_rate']:.1%}")
        print(f"      Abstention savings: {d['abstention_savings']:+.4f}")
        # Direction split
        lm = d["long_mask"]
        sm = d["short_mask"]
        print(f"      LONG  mask: n={lm['count']:3d}  "
              f"EV={lm['mean']:+.4f}  win={lm['win_rate']:.1%}"
              if lm["count"] > 0 else f"      LONG  mask: —")
        print(f"      SHORT mask: n={sm['count']:3d}  "
              f"EV={sm['mean']:+.4f}  win={sm['win_rate']:.1%}"
              if sm["count"] > 0 else f"      SHORT mask: —")


def _print_interaction(s: Dict[str, Any]) -> None:
    for pair_label, data in s["interactions"].items():
        print(f"\n    {pair_label.upper()} ({data['name']}):")
        for m, mdata in data["metrics"].items():
            print(f"      {m:<20s}  Δpair={mdata['delta_pair']:+.4f}  "
                  f"Δa={mdata['delta_single_a']:+.4f}  "
                  f"Δb={mdata['delta_single_b']:+.4f}  "
                  f"interaction={mdata['interaction_score']:+.4f}  "
                  f"[{mdata['classification']}]")

    strongest = s["strongest"]
    sig = "YES" if strongest["significant"] else "no"
    print(f"\n    Strongest: {strongest['pair']} / {strongest['metric']}  "
          f"score={strongest['score']:+.4f}  significant={sig}")


def _print_verdict(s: Dict[str, Any]) -> None:
    print(f"\n    Confidence tier:     {s['confidence_tier']}")
    print(f"    Crosscheck OK:       {'yes' if s['crosscheck_ok'] else 'NO'}")
    print(f"    Interaction found:   {'YES' if s['interaction_detected'] else 'no'}")
    print(f"    SHORT preserved:     {'yes' if s['short_preserved'] else 'NO'}")
    print(f"\n    Best single:  {s['best_single']['label']} "
          f"({s['best_single']['name']})  Δτ={s['best_single']['tau_delta']:+.4f}")
    print(f"    Best pair:    {s['best_pair']['label']} "
          f"({s['best_pair']['name']})  Δτ={s['best_pair']['tau_delta']:+.4f}")
    print(f"\n    ▸ RECOMMENDATION: {s['recommendation']}")
    print(f"      [{s['confidence_tier']}] {s['reason']}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Factor Ablation Grid — BTC Score Surface Decomposition",
    )
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument(
        "--funding-rate", type=float, default=None,
        help="Override BTC funding rate (8h raw)",
    )
    parser.add_argument(
        "--basis-pct", type=float, default=None,
        help="Override BTC basis percentage",
    )
    args = parser.parse_args()
    generate_report(
        json_output=args.json,
        funding_rate_override=args.funding_rate,
        basis_pct_override=args.basis_pct,
    )


if __name__ == "__main__":
    main()
