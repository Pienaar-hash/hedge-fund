#!/usr/bin/env python3
"""Layer 0 Carry-Fix Experiment — BTC Score Surface Recomputation.

Tests whether neutralizing the BTC carry-direction bias recenters the score
distribution toward the profitable mask without changing live execution.

Three additive variants on scored BTC episodes:
  V0 — Baseline (current hybrid_score unchanged)
  V1 — Carry neutralized (carry_score replaced with 0.5)
  V2 — Direction-neutral carry (carry magnitude kept, sign inversion removed)

Algebraic recomputation:
  hybrid_score ≈ Σ(w_i · f_i) after normalization
  We replace the carry factor and recompute while preserving other factors.

  Specifically:
    non_carry_score = (hybrid_score - w_carry * carry_factor) / (1 - w_carry)
    adjusted_score = (1 - w_carry) * non_carry_score + w_carry * new_carry

This is **observation-only** — no production scoring mutation.

Usage:
    PYTHONPATH=. python scripts/carry_fix_experiment.py
    PYTHONPATH=. python scripts/carry_fix_experiment.py --json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

# ── Paths ───────────────────────────────────────────────────────────────────
EPISODE_LEDGER = Path("logs/state/episode_ledger.json")
FUNDING_SNAPSHOT = Path("logs/state/funding_snapshot.json")
BASIS_SNAPSHOT = Path("logs/state/basis_snapshot.json")

# ── Profit mask (same as shadow_pnl_join.py) ────────────────────────────────
REFERENCE_MASK = {
    "BTCUSDT": {"lo": 0.4197, "hi": 0.4953},
}
MASK_MIDPOINT = (REFERENCE_MASK["BTCUSDT"]["lo"] + REFERENCE_MASK["BTCUSDT"]["hi"]) / 2
NEAR_MISS_BAND = 0.03

# ── Scoring weights (from HybridScoreConfig defaults) ──────────────────────
DEFAULT_CARRY_WEIGHT = 0.25
NEUTRAL_CARRY = 0.5


# ── Carry formula (mirrors symbol_score_v6.py) ─────────────────────────────

def scale_funding_rate(funding_rate: float, direction: str) -> float:
    """Funding rate → [0,1] carry component. Mirrors _scale_funding_rate."""
    annual_rate = funding_rate * 1095
    if direction.upper() == "LONG":
        raw = -annual_rate
    else:
        raw = annual_rate
    return 0.5 + 0.5 * math.tanh(raw / 0.20)


def scale_basis(basis_pct: float, direction: str) -> float:
    """Basis → [0,1] carry component. Mirrors _scale_basis."""
    if direction.upper() == "LONG":
        raw = -basis_pct
    else:
        raw = basis_pct
    return 0.5 + 0.5 * math.tanh(raw / 0.02)


def compute_carry_score(
    funding_rate: float,
    basis_pct: float,
    direction: str,
    funding_weight: float = 0.6,
    basis_weight: float = 0.4,
) -> float:
    """Compute carry score from funding + basis. Mirrors carry_score()."""
    f_score = scale_funding_rate(funding_rate, direction)
    b_score = scale_basis(basis_pct, direction)
    total_w = funding_weight + basis_weight
    if total_w <= 0:
        return NEUTRAL_CARRY
    return (funding_weight * f_score + basis_weight * b_score) / total_w


def compute_carry_score_direction_neutral(
    funding_rate: float,
    basis_pct: float,
    funding_weight: float = 0.6,
    basis_weight: float = 0.4,
) -> float:
    """V2: Carry magnitude without directional sign inversion.

    Uses absolute values of funding/basis to capture magnitude,
    then maps to score space without penalizing longs or shorts.
    """
    # Take magnitude of funding annual rate
    annual_rate = abs(funding_rate * 1095)
    f_score = 0.5 + 0.5 * math.tanh(annual_rate / 0.20)

    # Take magnitude of basis
    b_score = 0.5 + 0.5 * math.tanh(abs(basis_pct) / 0.02)

    total_w = funding_weight + basis_weight
    if total_w <= 0:
        return NEUTRAL_CARRY
    return (funding_weight * f_score + basis_weight * b_score) / total_w


# ── Score recomputation ─────────────────────────────────────────────────────

def recompute_hybrid_score(
    original_score: float,
    original_carry: float,
    new_carry: float,
    carry_weight: float = DEFAULT_CARRY_WEIGHT,
) -> float:
    """Algebraically replace carry factor in hybrid score.

    hybrid = w_carry * carry + (1 - w_carry) * non_carry_mean
    So: non_carry_mean = (hybrid - w_carry * carry) / (1 - w_carry)
    New: hybrid_new = w_carry * new_carry + (1 - w_carry) * non_carry_mean
    """
    if carry_weight >= 1.0:
        return new_carry
    non_carry_contribution = original_score - carry_weight * original_carry
    return non_carry_contribution + carry_weight * new_carry


# ── Region classification ───────────────────────────────────────────────────

def classify_region(score: float) -> str:
    ref = REFERENCE_MASK["BTCUSDT"]
    lo, hi = ref["lo"], ref["hi"]
    if lo <= score <= hi:
        return "mask_interior"
    if hi < score <= hi + NEAR_MISS_BAND:
        return "near_miss"
    return "outside"


# ── Data loading ────────────────────────────────────────────────────────────

def load_episodes(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    p = path or EPISODE_LEDGER
    if not p.exists():
        return []
    with open(p) as f:
        data = json.load(f)
    return data.get("episodes", data.get("entries", []))


def load_funding_snapshot(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or FUNDING_SNAPSHOT
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


def load_basis_snapshot(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or BASIS_SNAPSHOT
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


def _get_btc_carry_inputs(
    funding_snap: Dict[str, Any],
    basis_snap: Dict[str, Any],
) -> Tuple[float, float]:
    """Extract BTC funding_rate and basis_pct from snapshots."""
    syms_f = funding_snap.get("symbols", {})
    btc_f = syms_f.get("BTCUSDT", syms_f.get("btcusdt", {}))
    funding_rate = float(
        btc_f.get("rate", btc_f.get("funding_rate", 0.0))
        if isinstance(btc_f, dict) else btc_f
    )

    syms_b = basis_snap.get("symbols", {})
    btc_b = syms_b.get("BTCUSDT", syms_b.get("btcusdt", {}))
    basis_pct = float(btc_b.get("basis_pct", 0.0)) if isinstance(btc_b, dict) else 0.0

    return funding_rate, basis_pct


# ── Correlation metrics ─────────────────────────────────────────────────────

def _kendall_tau(xs: List[float], ys: List[float]) -> float:
    """Kendall's tau-b rank correlation."""
    n = len(xs)
    if n < 2:
        return 0.0
    concordant = 0
    discordant = 0
    tied_x = 0
    tied_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx == 0 and dy == 0:
                tied_x += 1
                tied_y += 1
            elif dx == 0:
                tied_x += 1
            elif dy == 0:
                tied_y += 1
            elif (dx > 0 and dy > 0) or (dx < 0 and dy < 0):
                concordant += 1
            else:
                discordant += 1
    n_pairs = n * (n - 1) / 2
    denom = math.sqrt((n_pairs - tied_x) * (n_pairs - tied_y))
    if denom == 0:
        return 0.0
    return (concordant - discordant) / denom


def _spearman_rho(xs: List[float], ys: List[float]) -> float:
    """Spearman rank correlation."""
    n = len(xs)
    if n < 2:
        return 0.0

    def _rank(vals: List[float]) -> List[float]:
        indexed = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and vals[indexed[j + 1]] == vals[indexed[j]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[indexed[k]] = avg_rank
            i = j + 1
        return ranks

    rx = _rank(xs)
    ry = _rank(ys)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = math.sqrt(sum((rx[i] - mean_rx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ry[i] - mean_ry) ** 2 for i in range(n)))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def _quintile_spread(scores: List[float], pnls: List[float]) -> float:
    """Q5 mean PnL minus Q1 mean PnL (sorted by score)."""
    if len(scores) < 5:
        return 0.0
    paired = sorted(zip(scores, pnls), key=lambda x: x[0])
    n = len(paired)
    q_size = max(1, n // 5)
    q1_pnls = [p for _, p in paired[:q_size]]
    q5_pnls = [p for _, p in paired[-q_size:]]
    q1_mean = sum(q1_pnls) / len(q1_pnls)
    q5_mean = sum(q5_pnls) / len(q5_pnls)
    return q5_mean - q1_mean


# ── PnL stats ──────────────────────────────────────────────────────────────

def _pnl_stats(pnls: List[float]) -> Dict[str, Any]:
    if not pnls:
        return {"count": 0, "total": 0.0, "mean": 0.0, "win_rate": 0.0}
    total = sum(pnls)
    mean = total / len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    return {
        "count": len(pnls),
        "total": round(total, 4),
        "mean": round(mean, 4),
        "win_rate": round(wins / len(pnls), 4),
    }


# ── Variant computation ────────────────────────────────────────────────────

def compute_variants(
    episodes: List[Dict[str, Any]],
    funding_rate: float,
    basis_pct: float,
    carry_weight: float = DEFAULT_CARRY_WEIGHT,
) -> List[Dict[str, Any]]:
    """Compute V0/V1/V2 scores for scored BTC episodes.

    Returns list of enriched episode dicts with v0/v1/v2 scores and regions.
    """
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

        # Estimate original carry score using current funding/basis
        original_carry = compute_carry_score(funding_rate, basis_pct, direction)

        # V0: baseline
        v0_score = float(h_score)

        # V1: carry neutralized (carry_score = 0.5)
        v1_score = recompute_hybrid_score(v0_score, original_carry, NEUTRAL_CARRY, carry_weight)

        # V2: direction-neutral carry (magnitude only, no sign inversion)
        neutral_carry = compute_carry_score_direction_neutral(funding_rate, basis_pct)
        v2_score = recompute_hybrid_score(v0_score, original_carry, neutral_carry, carry_weight)

        # Clamp to valid range
        v1_score = max(-1.0, min(1.0, v1_score))
        v2_score = max(-1.0, min(1.0, v2_score))

        results.append({
            "episode_id": ep.get("episode_id", ""),
            "symbol": "BTCUSDT",
            "side": direction,
            "net_pnl": float(pnl),
            "entry_ts": ep.get("entry_ts", ""),
            "v0_score": round(v0_score, 6),
            "v1_score": round(v1_score, 6),
            "v2_score": round(v2_score, 6),
            "v0_region": classify_region(v0_score),
            "v1_region": classify_region(v1_score),
            "v2_region": classify_region(v2_score),
            "estimated_carry": round(original_carry, 6),
            "neutral_carry": round(neutral_carry, 6),
        })

    return results


# ── Report sections ─────────────────────────────────────────────────────────

def section_correlation(
    variants: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Section 1: Score–PnL correlation per variant."""
    pnls = [v["net_pnl"] for v in variants]
    result: Dict[str, Any] = {"n": len(variants)}

    for label in ["v0", "v1", "v2"]:
        scores = [v[f"{label}_score"] for v in variants]
        result[label] = {
            "kendall_tau": round(_kendall_tau(scores, pnls), 4),
            "spearman_rho": round(_spearman_rho(scores, pnls), 4),
            "q5_minus_q1": round(_quintile_spread(scores, pnls), 4),
        }

    return {"title": "Score–PnL Correlation", **result}


def section_mask_alignment(
    variants: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Section 2: Mask alignment per variant."""
    result: Dict[str, Any] = {"n": len(variants)}

    for label in ["v0", "v1", "v2"]:
        scores = [v[f"{label}_score"] for v in variants]
        regions = [v[f"{label}_region"] for v in variants]
        n = len(scores)
        in_mask = sum(1 for r in regions if r == "mask_interior")
        near_miss = sum(1 for r in regions if r == "near_miss")
        spillover = near_miss / in_mask if in_mask > 0 else 0.0

        mean_score = sum(scores) / n if n else 0.0
        midpoint_dist = mean_score - MASK_MIDPOINT

        result[label] = {
            "in_mask_pct": round(in_mask / n, 4) if n else 0.0,
            "in_mask_count": in_mask,
            "near_miss_count": near_miss,
            "spillover_pressure": round(spillover, 4),
            "mean_score": round(mean_score, 4),
            "midpoint_distance": round(midpoint_dist, 4),
        }

    return {"title": "Mask Alignment", **result}


def section_economic_validation(
    variants: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Section 3: EV by region per variant."""
    result: Dict[str, Any] = {}

    for label in ["v0", "v1", "v2"]:
        regions: Dict[str, List[float]] = defaultdict(list)
        for v in variants:
            regions[v[f"{label}_region"]].append(v["net_pnl"])

        region_stats = {}
        for region in ["mask_interior", "near_miss", "outside"]:
            region_stats[region] = _pnl_stats(regions.get(region, []))

        # Abstention savings (near-miss trades CandD would skip)
        nm_pnls = regions.get("near_miss", [])
        abstention_savings = -sum(nm_pnls) if nm_pnls else 0.0

        result[label] = {
            "regions": region_stats,
            "abstention_savings": round(abstention_savings, 4),
        }

    return {"title": "Economic Validation", **result}


def section_selector_relevance(
    variants: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Section 4: Candidate D selector metrics per variant."""
    result: Dict[str, Any] = {}

    for label in ["v0", "v1", "v2"]:
        regions = [v[f"{label}_region"] for v in variants]
        scores = [v[f"{label}_score"] for v in variants]
        d_profit = sum(1 for r in regions if r == "mask_interior")
        zero_score = sum(1 for s in scores if s <= 0)
        result[label] = {
            "d_profit_region": d_profit,
            "d_abstain": len(variants) - d_profit,
            "zero_score_count": zero_score,
        }

    return {"title": "Selector Relevance", **result}


def section_pass_fail(
    correlation: Dict[str, Any],
    alignment: Dict[str, Any],
    economic: Dict[str, Any],
) -> Dict[str, Any]:
    """Section 5: Pass/fail assessment for each variant."""
    verdicts: Dict[str, Any] = {}

    for label in ["v1", "v2"]:
        checks: List[Dict[str, Any]] = []

        # 1. Score–PnL: correlation should improve (or at least not worsen)
        v0_tau = correlation["v0"]["kendall_tau"]
        vx_tau = correlation[label]["kendall_tau"]
        tau_delta = vx_tau - v0_tau
        checks.append({
            "name": "correlation_improves",
            "pass": tau_delta >= -0.05,
            "detail": f"Δτ = {tau_delta:+.4f} (V0={v0_tau:.4f}, {label.upper()}={vx_tau:.4f})",
        })

        # 2. Midpoint distance shrinks
        v0_dist = abs(alignment["v0"]["midpoint_distance"])
        vx_dist = abs(alignment[label]["midpoint_distance"])
        checks.append({
            "name": "midpoint_recenters",
            "pass": vx_dist < v0_dist,
            "detail": f"|dist| V0={v0_dist:.4f} → {label.upper()}={vx_dist:.4f}",
        })

        # 3. Spillover pressure declines
        v0_sp = alignment["v0"]["spillover_pressure"]
        vx_sp = alignment[label]["spillover_pressure"]
        checks.append({
            "name": "spillover_declines",
            "pass": vx_sp <= v0_sp,
            "detail": f"spillover V0={v0_sp:.4f} → {label.upper()}={vx_sp:.4f}",
        })

        # 4. Mask interior remains best EV region
        ev_mask = economic[label]["regions"]["mask_interior"]["mean"]
        ev_nm = economic[label]["regions"]["near_miss"]["mean"]
        checks.append({
            "name": "mask_best_ev",
            "pass": ev_mask > ev_nm or economic[label]["regions"]["near_miss"]["count"] == 0,
            "detail": f"EV(mask)={ev_mask:+.4f} vs EV(nm)={ev_nm:+.4f}",
        })

        # 5. No ZERO_SCORE pathology increase
        # (checked separately if data available)

        all_pass = all(c["pass"] for c in checks)
        verdicts[label] = {
            "checks": checks,
            "all_pass": all_pass,
            "recommendation": (
                "INTERESTING — proceed to Phase B"
                if all_pass
                else "REJECT — does not meet pass criteria"
            ),
        }

    return {"title": "Pass/Fail Assessment", "verdicts": verdicts}


# ── Report generation ──────────────────────────────────────────────────────

def generate_report(
    *,
    json_output: bool = False,
    episode_path: Optional[Path] = None,
    funding_path: Optional[Path] = None,
    basis_path: Optional[Path] = None,
    funding_rate_override: Optional[float] = None,
    basis_pct_override: Optional[float] = None,
) -> Dict[str, Any]:
    """Generate the full carry-fix experiment report."""
    episodes = load_episodes(episode_path)
    funding_snap = load_funding_snapshot(funding_path)
    basis_snap = load_basis_snapshot(basis_path)

    funding_rate, basis_pct = _get_btc_carry_inputs(funding_snap, basis_snap)
    if funding_rate_override is not None:
        funding_rate = funding_rate_override
    if basis_pct_override is not None:
        basis_pct = basis_pct_override

    variants = compute_variants(episodes, funding_rate, basis_pct)

    s_corr = section_correlation(variants)
    s_align = section_mask_alignment(variants)
    s_econ = section_economic_validation(variants)
    s_sel = section_selector_relevance(variants)
    s_pf = section_pass_fail(s_corr, s_align, s_econ)

    sections = [s_corr, s_align, s_econ, s_sel, s_pf]

    report = {
        "generated_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "experiment": "Layer 0 Carry-Fix",
        "symbol": "BTCUSDT",
        "carry_inputs": {
            "funding_rate": funding_rate,
            "basis_pct": basis_pct,
            "estimated_carry_long": round(
                compute_carry_score(funding_rate, basis_pct, "LONG"), 6
            ),
            "neutral_carry": NEUTRAL_CARRY,
            "direction_neutral_carry": round(
                compute_carry_score_direction_neutral(funding_rate, basis_pct), 6
            ),
        },
        "total_episodes": len(variants),
        "sections": {s["title"]: s for s in sections},
    }

    if json_output:
        json.dump(report, sys.stdout, indent=2)
        print()
    else:
        _print_report(report, sections, variants)

    return report


# ── Text output ─────────────────────────────────────────────────────────────

def _print_report(
    report: Dict[str, Any],
    sections: List[Dict[str, Any]],
    variants: List[Dict[str, Any]],
) -> None:
    bar = "=" * 72
    now_iso = report["generated_at"]
    print(f"\n{bar}")
    print(f"  LAYER 0 CARRY-FIX EXPERIMENT — BTC Score Surface")
    print(f"  Generated: {now_iso}")
    print(bar)

    ci = report["carry_inputs"]
    print(f"\n  Carry Inputs:")
    print(f"    Funding rate (8h):        {ci['funding_rate']:.6f}")
    print(f"    Basis pct:                {ci['basis_pct']:.6f}")
    print(f"    Estimated carry (LONG):   {ci['estimated_carry_long']:.4f}")
    print(f"    Neutral carry:            {ci['neutral_carry']:.4f}")
    print(f"    Direction-neutral carry:  {ci['direction_neutral_carry']:.4f}")
    print(f"    BTC episodes:             {report['total_episodes']}")

    for s in sections:
        title = s["title"]
        print(f"\n{'─' * 72}")
        print(f"  {title}")
        print(f"{'─' * 72}")

        if title == "Score–PnL Correlation":
            _print_correlation(s)
        elif title == "Mask Alignment":
            _print_alignment(s)
        elif title == "Economic Validation":
            _print_economic(s)
        elif title == "Selector Relevance":
            _print_selector(s)
        elif title == "Pass/Fail Assessment":
            _print_pass_fail(s)

    # Episode-level comparison table
    print(f"\n{'─' * 72}")
    print(f"  Episode-Level Score Comparison")
    print(f"{'─' * 72}")
    print(f"\n    {'EP':<10s} {'Side':<6s} {'V0':>7s} {'V1':>7s} {'V2':>7s} "
          f"{'PnL':>8s}  {'V0 Region':<14s} {'V1 Region':<14s} {'V2 Region'}")
    print(f"    {'─' * 90}")
    for v in variants:
        print(f"    {v['episode_id']:<10s} {v['side']:<6s} "
              f"{v['v0_score']:7.4f} {v['v1_score']:7.4f} {v['v2_score']:7.4f} "
              f"{v['net_pnl']:8.4f}  {v['v0_region']:<14s} {v['v1_region']:<14s} "
              f"{v['v2_region']}")

    print(f"\n{bar}")
    print(f"  END OF EXPERIMENT REPORT")
    print(bar)


def _print_correlation(s: Dict[str, Any]) -> None:
    print(f"\n    n = {s['n']}\n")
    print(f"    {'Metric':<16s}  {'V0':>8s}  {'V1':>8s}  {'V2':>8s}")
    print(f"    {'─' * 46}")
    for metric in ["kendall_tau", "spearman_rho", "q5_minus_q1"]:
        v0 = s["v0"][metric]
        v1 = s["v1"][metric]
        v2 = s["v2"][metric]
        print(f"    {metric:<16s}  {v0:+8.4f}  {v1:+8.4f}  {v2:+8.4f}")


def _print_alignment(s: Dict[str, Any]) -> None:
    print(f"\n    n = {s['n']}  mask = [{REFERENCE_MASK['BTCUSDT']['lo']:.4f}, "
          f"{REFERENCE_MASK['BTCUSDT']['hi']:.4f}]  midpoint = {MASK_MIDPOINT:.4f}\n")
    print(f"    {'Metric':<20s}  {'V0':>8s}  {'V1':>8s}  {'V2':>8s}")
    print(f"    {'─' * 50}")
    for metric in ["in_mask_pct", "spillover_pressure", "midpoint_distance", "mean_score"]:
        v0 = s["v0"][metric]
        v1 = s["v1"][metric]
        v2 = s["v2"][metric]
        fmt = "+.4f" if "distance" in metric else ".4f"
        print(f"    {metric:<20s}  {v0:{fmt}}  {v1:{fmt}}  {v2:{fmt}}")


def _print_economic(s: Dict[str, Any]) -> None:
    for label in ["v0", "v1", "v2"]:
        print(f"\n    {label.upper()}:")
        for region in ["mask_interior", "near_miss", "outside"]:
            rs = s[label]["regions"].get(region, {})
            if rs.get("count", 0) == 0:
                print(f"      {region:<14s}  —")
            else:
                print(f"      {region:<14s}  n={rs['count']:3d}  "
                      f"EV={rs['mean']:+.4f}  win={rs['win_rate']:.1%}")
        print(f"      Abstention savings: {s[label]['abstention_savings']:+.4f}")


def _print_selector(s: Dict[str, Any]) -> None:
    print(f"\n    {'Metric':<20s}  {'V0':>6s}  {'V1':>6s}  {'V2':>6s}")
    print(f"    {'─' * 42}")
    for metric in ["d_profit_region", "d_abstain", "zero_score_count"]:
        v0 = s["v0"][metric]
        v1 = s["v1"][metric]
        v2 = s["v2"][metric]
        print(f"    {metric:<20s}  {v0:6d}  {v1:6d}  {v2:6d}")


def _print_pass_fail(s: Dict[str, Any]) -> None:
    for label in ["v1", "v2"]:
        data = s["verdicts"][label]
        status = "PASS ✓" if data["all_pass"] else "FAIL ✗"
        print(f"\n    {label.upper()} — {status}")
        for c in data["checks"]:
            marker = "✓" if c["pass"] else "✗"
            print(f"      [{marker}] {c['name']}: {c['detail']}")
        print(f"      → {data['recommendation']}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Layer 0 Carry-Fix Experiment — BTC Score Surface",
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
