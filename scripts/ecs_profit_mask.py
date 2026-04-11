#!/usr/bin/env python3
"""ECS Profit Mask Derivation — identify PnL-positive score regions per symbol.

Derives an explicit PnL-positive mask as f(symbol, hydra_score) by
analysing the episode ledger with sliding-window PnL along the score axis.

The output is the `PNL_POSITIVE_REGIONS` constant (ready for copy-paste into
execution/shadow_selector_v2.py) together with a stability diagnostic
that shows how robust the boundaries are under 90% subsampling.

Usage:
    PYTHONPATH=. python scripts/ecs_profit_mask.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

EPISODE_LEDGER = Path("logs/state/episode_ledger.json")
CORE_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

# ── Stabilisation parameters ─────────────────────────────────────────────────
MIN_REGION_WIDTH = 0.01   # reject regions narrower than this (score units)
MERGE_GAP_THRESHOLD = 0.015  # merge positive regions separated by < this
MERGE_GAP_MIN_TRADES = 3  # gap must have ≥ this many trades to survive merge
MERGE_GAP_MIN_LOSS = -5.0  # gap must be clearly negative (total PnL < this) to survive merge
MIN_TRADES_IN_REGION = 3  # minimum trades for a region to be valid
SUBSAMPLE_RATE = 0.9      # fraction of data kept in each subsample
SUBSAMPLE_RUNS = 10       # number of stability subsamples
SUBSAMPLE_SEED = 42       # reproducible subsampling


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_episodes() -> List[Dict[str, Any]]:
    if not EPISODE_LEDGER.exists():
        return []
    with open(EPISODE_LEDGER) as f:
        data = json.load(f)
    return data.get("episodes", data.get("entries", []))


def _scored_by_symbol(episodes: List[Dict[str, Any]]) -> Dict[str, List[Tuple[float, float]]]:
    """Return {symbol: [(hydra_score, net_pnl), ...]} sorted by score."""
    result: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for ep in episodes:
        sym = ep.get("symbol", "")
        if sym not in CORE_SYMBOLS:
            continue
        score = ep.get("hybrid_score", 0)
        if not score or score <= 0:
            continue
        pnl = ep.get("net_pnl")
        if pnl is None:
            continue
        result[sym].append((float(score), float(pnl)))
    for sym in result:
        result[sym].sort()
    return result


# ── Core algorithm: sliding-window PnL → positive regions ────────────────────

def _find_positive_regions(
    scored: List[Tuple[float, float]],
    window: Optional[int] = None,
) -> List[Tuple[float, float]]:
    """Find contiguous score ranges where sliding-window PnL > 0.

    Args:
        scored: Sorted list of (hydra_score, net_pnl).
        window: Sliding window size; defaults to ~1/3 of data, clamped [3, 25].

    Returns:
        List of (lo, hi) tuples representing PnL-positive score regions.
    """
    if len(scored) < 3:
        return []

    if window is None:
        window = max(3, min(25, len(scored) // 3))

    # Compute sliding-window mean PnL at each position
    points: List[Tuple[float, float, float]] = []  # (center_score, lo_score, hi_score, mean_pnl)
    for i in range(len(scored) - window + 1):
        w = scored[i: i + window]
        scores = [s for s, _ in w]
        pnls = [p for _, p in w]
        center = (scores[0] + scores[-1]) / 2
        mean_pnl = sum(pnls) / len(pnls)
        points.append((scores[0], scores[-1], mean_pnl))

    if not points:
        return []

    # Identify contiguous positive segments using window edges
    raw_regions: List[Tuple[float, float]] = []
    in_positive = False
    region_lo = 0.0

    for lo_s, hi_s, mean_pnl in points:
        if mean_pnl > 0 and not in_positive:
            region_lo = lo_s
            in_positive = True
        elif mean_pnl <= 0 and in_positive:
            raw_regions.append((region_lo, hi_s))
            in_positive = False

    if in_positive:
        raw_regions.append((region_lo, points[-1][1]))

    # Apply stabilisation: min width filter
    filtered = [(lo, hi) for lo, hi in raw_regions if hi - lo >= MIN_REGION_WIDTH]

    # Apply stabilisation: merge nearby regions
    merged = _merge_nearby_regions(filtered, scored)

    # Filter by minimum trade count
    final = []
    for lo, hi in merged:
        trades_in_region = sum(1 for s, _ in scored if lo <= s <= hi)
        if trades_in_region >= MIN_TRADES_IN_REGION:
            final.append((round(lo, 4), round(hi, 4)))

    return final


def _merge_nearby_regions(
    regions: List[Tuple[float, float]],
    scored: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Merge regions separated by small gaps unless the gap is clearly negative."""
    if len(regions) <= 1:
        return regions

    merged: List[Tuple[float, float]] = [regions[0]]
    for lo, hi in regions[1:]:
        prev_lo, prev_hi = merged[-1]
        gap = lo - prev_hi

        if gap <= MERGE_GAP_THRESHOLD:
            # Check if gap has enough trades and clearly negative PnL
            gap_trades = [(s, p) for s, p in scored if prev_hi < s < lo]
            gap_n = len(gap_trades)
            gap_pnl = sum(p for _, p in gap_trades) if gap_trades else 0

            if gap_n < MERGE_GAP_MIN_TRADES or gap_pnl > MERGE_GAP_MIN_LOSS:
                # Gap is too small / not clearly negative → merge
                merged[-1] = (prev_lo, hi)
                continue

        merged.append((lo, hi))

    return merged


# ── Stability diagnostic ─────────────────────────────────────────────────────

def _overlap_score(base: List[Tuple[float, float]], sample: List[Tuple[float, float]]) -> float:
    """Compute overlap between two sets of regions as fraction of base covered.

    Returns 0.0 (no overlap) to 1.0 (perfect overlap).
    """
    if not base or not sample:
        return 0.0 if base or sample else 1.0

    # Compute total base length
    base_len = sum(hi - lo for lo, hi in base)
    if base_len < 1e-9:
        return 0.0

    # Compute intersection length
    overlap = 0.0
    for blo, bhi in base:
        for slo, shi in sample:
            inter_lo = max(blo, slo)
            inter_hi = min(bhi, shi)
            if inter_hi > inter_lo:
                overlap += inter_hi - inter_lo

    return min(1.0, overlap / base_len)


def _run_stability_diagnostic(
    scored: List[Tuple[float, float]],
    base_regions: List[Tuple[float, float]],
    symbol: str,
) -> List[Tuple[List[Tuple[float, float]], float]]:
    """Run 10 x 90% subsampling and report overlap with base regions."""
    rng = random.Random(SUBSAMPLE_SEED)
    results: List[Tuple[List[Tuple[float, float]], float]] = []

    for i in range(SUBSAMPLE_RUNS):
        n = len(scored)
        k = max(3, int(n * SUBSAMPLE_RATE))
        sample = sorted(rng.sample(scored, k))
        sample_regions = _find_positive_regions(sample)
        overlap = _overlap_score(base_regions, sample_regions)
        results.append((sample_regions, overlap))

    return results


# ── Output formatting ────────────────────────────────────────────────────────

def _format_regions(regions: List[Tuple[float, float]]) -> str:
    if not regions:
        return "[]"
    parts = [f"({lo:.4f}, {hi:.4f})" for lo, hi in regions]
    return "[" + ", ".join(parts) + "]"


def _pnl_in_region(scored: List[Tuple[float, float]], regions: List[Tuple[float, float]]) -> Tuple[int, float]:
    """Count trades and total PnL inside the given regions."""
    n = 0
    total = 0.0
    for s, p in scored:
        for lo, hi in regions:
            if lo <= s <= hi:
                n += 1
                total += p
                break
    return n, total


def _pnl_outside_region(scored: List[Tuple[float, float]], regions: List[Tuple[float, float]]) -> Tuple[int, float]:
    """Count trades and total PnL outside the given regions."""
    n = 0
    total = 0.0
    for s, p in scored:
        inside = any(lo <= s <= hi for lo, hi in regions)
        if not inside:
            n += 1
            total += p
    return n, total


# ── Main ─────────────────────────────────────────────────────────────────────

def run() -> None:
    episodes = _load_episodes()
    if not episodes:
        print("No episodes found.")
        return

    by_symbol = _scored_by_symbol(episodes)

    print("=" * 72)
    print("  ECS PROFIT MASK DERIVATION")
    print("=" * 72)
    print()
    print(f"  Episodes loaded: {len(episodes)}")
    for sym in CORE_SYMBOLS:
        print(f"    {sym}: {len(by_symbol.get(sym, []))} scored")
    print()

    all_regions: Dict[str, List[Tuple[float, float]]] = {}

    for sym in CORE_SYMBOLS:
        scored = by_symbol.get(sym, [])
        print("-" * 72)
        print(f"  {sym}  (n={len(scored)})")
        print("-" * 72)

        if len(scored) < 6:
            print(f"  Insufficient data (need >= 6)")
            all_regions[sym] = []
            print()
            continue

        # Derive base regions
        regions = _find_positive_regions(scored)

        # Post-filter: only keep regions where actual cumulative PnL > 0.
        # This prevents "least-bad" regions from appearing as profitable.
        regions = [
            (lo, hi) for lo, hi in regions
            if sum(p for s, p in scored if lo <= s <= hi) > 0
        ]

        all_regions[sym] = regions

        # Stats: inside vs outside
        in_n, in_pnl = _pnl_in_region(scored, regions)
        out_n, out_pnl = _pnl_outside_region(scored, regions)

        print(f"\n  PnL-positive regions: {_format_regions(regions)}")
        print(f"  Score range: [{scored[0][0]:.4f}, {scored[-1][0]:.4f}]")
        print()
        print(f"  Inside profit mask:   n={in_n:>4}  PnL={in_pnl:>+10.2f}"
              f"  mean={in_pnl/in_n:>+8.4f}" if in_n else "  Inside profit mask:   n=   0  (empty)")
        print(f"  Outside profit mask:  n={out_n:>4}  PnL={out_pnl:>+10.2f}"
              f"  mean={out_pnl/out_n:>+8.4f}" if out_n else "  Outside profit mask:  n=   0")
        print()

        # Stability diagnostic
        print(f"  STABILITY DIAGNOSTIC (10 × 90% subsamples):")
        print()
        stability = _run_stability_diagnostic(scored, regions, sym)

        overlaps = [ov for _, ov in stability]
        print(f"    {'Run':>5}  {'Regions':>40}  {'Overlap':>8}")
        print(f"    {'-'*5}  {'-'*40}  {'-'*8}")
        print(f"    {'BASE':>5}  {_format_regions(regions):>40}  {'1.000':>8}")
        for i, (sample_regions, overlap) in enumerate(stability):
            print(f"    {i+1:>5}  {_format_regions(sample_regions):>40}  {overlap:>8.3f}")

        mean_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
        min_overlap = min(overlaps) if overlaps else 0
        print()
        print(f"    Mean overlap: {mean_overlap:.3f}")
        print(f"    Min overlap:  {min_overlap:.3f}")
        if mean_overlap >= 0.7:
            print(f"    → STABLE (mean overlap ≥ 0.7)")
        elif mean_overlap >= 0.4:
            print(f"    → MODERATE (mean overlap 0.4–0.7, boundaries may shift)")
        else:
            print(f"    → UNSTABLE (mean overlap < 0.4, insufficient data)")
        print()

    # ── Final output: copy-paste constant ────────────────────────────
    print("=" * 72)
    print("  PNL_POSITIVE_REGIONS (copy-paste into shadow_selector_v2.py)")
    print("=" * 72)
    print()
    print("PNL_POSITIVE_REGIONS: Dict[str, List[tuple]] = {")
    for sym in CORE_SYMBOLS:
        regions = all_regions.get(sym, [])
        if not regions:
            print(f'    "{sym}": [],')
        else:
            parts = [f"({lo}, {hi})" for lo, hi in regions]
            print(f'    "{sym}": [{", ".join(parts)}],')
    print("}")
    print()

    # ── Summary ──────────────────────────────────────────────────────
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print()
    total_in_n = 0
    total_in_pnl = 0.0
    total_out_n = 0
    total_out_pnl = 0.0
    for sym in CORE_SYMBOLS:
        scored = by_symbol.get(sym, [])
        regions = all_regions.get(sym, [])
        in_n, in_pnl = _pnl_in_region(scored, regions)
        out_n, out_pnl = _pnl_outside_region(scored, regions)
        total_in_n += in_n
        total_in_pnl += in_pnl
        total_out_n += out_n
        total_out_pnl += out_pnl

    print(f"  Total inside profit mask:   n={total_in_n:>4}  PnL={total_in_pnl:>+10.2f}"
          + (f"  mean={total_in_pnl/total_in_n:>+8.4f}" if total_in_n else ""))
    print(f"  Total outside profit mask:  n={total_out_n:>4}  PnL={total_out_pnl:>+10.2f}"
          + (f"  mean={total_out_pnl/total_out_n:>+8.4f}" if total_out_n else ""))
    if total_in_n + total_out_n > 0:
        selectivity = total_in_n / (total_in_n + total_out_n) * 100
        print(f"  Selectivity: {selectivity:.1f}% of trades pass profit mask")
    print()


# ── Monotonicity salvageability test (DEC_ECS_PROFIT_MASK_V2) ────────────────

def test_mask_salvageability(
    symbol: str,
    episodes: Optional[List[Dict[str, Any]]] = None,
    n_buckets: int = 5,
) -> Dict[str, Any]:
    """Monotonicity test to determine if a symbol's profit mask is salvageable.

    Tests four criteria (ALL must pass):
      1. Spearman ρ > 0.15  (score is informative)
      2. Q5-Q1 spread > 0   (top quintile outperforms bottom)
      3. p-value < 0.10     (statistically significant)
      4. Monotonicity ratio ≥ 0.75  (≤1 bucket violation)

    Returns dict with test results and salvageable verdict.
    """
    # Lazy import to avoid circular dependency
    from execution.hydra_monotonicity import compute_monotonicity, compute_quintile_spread

    if episodes is None:
        episodes = _load_episodes()

    sym_eps = [ep for ep in episodes if ep.get("symbol") == symbol]

    mono = compute_monotonicity(sym_eps, n_buckets=n_buckets)
    q_spread = compute_quintile_spread(sym_eps)

    rho = mono.get("spearman")
    p_val = mono.get("p_value")
    q5_q1 = q_spread.get("q5_q1_spread")

    # Bucket-level monotonicity: count inversions
    buckets = mono.get("buckets", [])
    violations = 0
    for i in range(len(buckets) - 1):
        if buckets[i + 1]["mean_return"] < buckets[i]["mean_return"]:
            violations += 1
    n_gaps = max(1, len(buckets) - 1)
    mono_ratio = 1.0 - violations / n_gaps if buckets else 0.0

    salvageable = (
        rho is not None and rho > 0.15
        and q5_q1 is not None and q5_q1 > 0
        and p_val is not None and p_val < 0.10
        and mono_ratio >= 0.75
    )

    return {
        "symbol": symbol,
        "spearman": round(rho, 4) if rho is not None else None,
        "p_value": round(p_val, 6) if p_val is not None else None,
        "q5_q1_spread": round(q5_q1, 6) if q5_q1 is not None else None,
        "monotonicity_ratio": round(mono_ratio, 4),
        "violations": violations,
        "n_buckets": len(buckets),
        "salvageable": salvageable,
        "n_episodes": mono.get("n", 0),
        "slope": mono.get("slope", "unknown"),
        "buckets": buckets,
    }


def run_salvageability() -> None:
    """Run monotonicity salvageability test for ETH and SOL."""
    episodes = _load_episodes()
    if not episodes:
        print("No episodes found.")
        return

    print("=" * 72)
    print("  MONOTONICITY SALVAGEABILITY TEST")
    print("=" * 72)

    for sym in ["ETHUSDT", "SOLUSDT"]:
        result = test_mask_salvageability(sym, episodes)
        print()
        print(f"  {sym}")
        print(f"  {'─' * 50}")
        print(f"    Episodes:            {result['n_episodes']}")
        print(f"    Spearman ρ:          {result['spearman']}"
              f"  {'✓' if result['spearman'] is not None and result['spearman'] > 0.15 else '✗'}")
        print(f"    p-value:             {result['p_value']}"
              f"  {'✓' if result['p_value'] is not None and result['p_value'] < 0.10 else '✗'}")
        print(f"    Q5-Q1 spread:        {result['q5_q1_spread']}"
              f"  {'✓' if result['q5_q1_spread'] is not None and result['q5_q1_spread'] > 0 else '✗'}")
        print(f"    Monotonicity ratio:  {result['monotonicity_ratio']}"
              f"  {'✓' if result['monotonicity_ratio'] >= 0.75 else '✗'}")
        print(f"    Violations:          {result['violations']}/{result['n_buckets'] - 1 if result['n_buckets'] > 1 else 0}")
        print(f"    Slope:               {result['slope']}")
        print()
        if result["salvageable"]:
            print(f"    → SALVAGEABLE (all 4 criteria pass)")
        else:
            print(f"    → NOT SALVAGEABLE (mask discarded, revert to prior routing)")


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "--salvageability":
        run_salvageability()
    else:
        run()
