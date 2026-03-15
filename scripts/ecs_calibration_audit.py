#!/usr/bin/env python3
"""ECS Score Calibration Audit — conflict-case analysis.

Joins score_decomposition.jsonl (Hydra scores) with ecs_soak_events.jsonl
(winner labels) by timestamp proximity to assess score comparability.

For events that carry paired merge scores (post-enrichment), uses those
directly. For historical events, falls back to timestamp-join.

Usage:
    PYTHONPATH=. python scripts/ecs_calibration_audit.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SOAK_PATH = Path("logs/execution/ecs_soak_events.jsonl")
SCORE_DECOMP_PATH = Path("logs/execution/score_decomposition.jsonl")

# Maximum timestamp gap (seconds) for joining score decomposition → soak event
JOIN_TOLERANCE_S = 45.0

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    events = []
    if not path.exists():
        return events
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return events


def _parse_ts(val: Any) -> Optional[float]:
    """Parse timestamp from either float or ISO string."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        from datetime import datetime, timezone
        try:
            dt = datetime.fromisoformat(val)
            return dt.timestamp()
        except ValueError:
            pass
    return None


def _build_score_index(events: List[Dict[str, Any]]) -> Dict[str, List[Tuple[float, float]]]:
    """Build {symbol: [(ts, hybrid_score), ...]} from score decomposition."""
    index: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for e in events:
        ts = _parse_ts(e.get("ts"))
        sym = e.get("symbol")
        score = e.get("hybrid_score")
        if ts is not None and sym and score is not None:
            index[sym].append((ts, float(score)))
    for sym in index:
        index[sym].sort()
    return index


def _nearest_score(score_list: List[Tuple[float, float]], target_ts: float) -> Optional[float]:
    """Find nearest score within JOIN_TOLERANCE_S."""
    import bisect
    if not score_list:
        return None
    idx = bisect.bisect_left(score_list, (target_ts,))
    best = None
    best_gap = JOIN_TOLERANCE_S + 1
    for i in range(max(0, idx - 1), min(len(score_list), idx + 2)):
        gap = abs(score_list[i][0] - target_ts)
        if gap < best_gap:
            best_gap = gap
            best = score_list[i][1]
    return best if best_gap <= JOIN_TOLERANCE_S else None


def run_audit() -> None:
    soak_events = _load_jsonl(SOAK_PATH)
    score_events = _load_jsonl(SCORE_DECOMP_PATH)

    if not soak_events:
        print("No soak events found.")
        return

    score_index = _build_score_index(score_events)

    # ── Collect paired data ──────────────────────────────────────────
    # For each soak event, try to get both scores:
    #   1. From enriched soak event fields (merge_hydra_score, merge_legacy_score)
    #   2. Fallback: Hydra from score_decomposition join, legacy unavailable

    paired: Dict[str, List[Dict[str, Any]]] = defaultdict(list)  # symbol → records
    enriched_count = 0
    joined_count = 0
    unmatched = 0

    for e in soak_events:
        sym = e.get("symbol")
        if not sym:
            continue

        ts = _parse_ts(e.get("ts"))
        winner = e.get("ecs_winner")
        conflict = e.get("merge_conflict", False)
        h_score = e.get("merge_hydra_score")
        l_score = e.get("merge_legacy_score")

        if h_score is not None and l_score is not None:
            enriched_count += 1
            paired[sym].append({
                "ts": ts,
                "hydra_score": float(h_score),
                "legacy_score": float(l_score),
                "winner": winner,
                "conflict": True,
                "source": "enriched",
            })
        elif ts is not None and sym in score_index:
            hydra = _nearest_score(score_index[sym], ts)
            if hydra is not None:
                joined_count += 1
                paired[sym].append({
                    "ts": ts,
                    "hydra_score": hydra,
                    "legacy_score": None,
                    "winner": winner,
                    "conflict": conflict,
                    "source": "joined",
                })
            else:
                unmatched += 1
        else:
            unmatched += 1

    total_paired = sum(len(v) for v in paired.values())
    total_conflict = sum(1 for recs in paired.values() for r in recs if r["conflict"])
    total_enriched_conflict = sum(1 for recs in paired.values() for r in recs if r["source"] == "enriched")

    print("=" * 70)
    print("  ECS Score Calibration Audit")
    print("=" * 70)
    print()
    print(f"  Soak events:           {len(soak_events)}")
    print(f"  Score decomposition:   {len(score_events)}")
    print(f"  Paired (enriched):     {enriched_count}")
    print(f"  Paired (ts-joined):    {joined_count}")
    print(f"  Unmatched:             {unmatched}")
    print(f"  Total paired:          {total_paired}")
    print(f"  Conflict cases:        {total_conflict}")
    print()

    # ── Audit 1: Distribution overlap per symbol ─────────────────────
    print("-" * 70)
    print("  AUDIT 1: Score Distribution by Symbol")
    print("-" * 70)
    print()

    for sym in sorted(paired.keys()):
        recs = paired[sym]
        hydra_scores = [r["hydra_score"] for r in recs if r["hydra_score"] is not None]
        legacy_scores = [r["legacy_score"] for r in recs if r["legacy_score"] is not None]

        hydra_by_winner = defaultdict(list)
        for r in recs:
            if r["hydra_score"] is not None:
                hydra_by_winner[r["winner"]].append(r["hydra_score"])

        def _stats(vals: List[float]) -> str:
            if not vals:
                return "n=0"
            vals_s = sorted(vals)
            n = len(vals_s)
            mean = sum(vals_s) / n
            median = vals_s[n // 2]
            p25 = vals_s[int(n * 0.25)]
            p75 = vals_s[int(n * 0.75)]
            return f"n={n:>4}  mean={mean:.4f}  med={median:.4f}  p25={p25:.4f}  p75={p75:.4f}  range=[{vals_s[0]:.4f}, {vals_s[-1]:.4f}]"

        print(f"  {sym}")
        print(f"    Hydra scores (all):     {_stats(hydra_scores)}")
        if legacy_scores:
            print(f"    Legacy scores (all):    {_stats(legacy_scores)}")
        for w in sorted(hydra_by_winner.keys()):
            print(f"    Hydra when winner={w:>8}: {_stats(hydra_by_winner[w])}")
        print()

    # ── Audit 2: Enriched conflict analysis (paired scores) ─────────
    enriched_conflicts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sym, recs in paired.items():
        for r in recs:
            if r["source"] == "enriched" and r["legacy_score"] is not None:
                enriched_conflicts[sym].append(r)

    if enriched_conflicts:
        print("-" * 70)
        print("  AUDIT 2: Conflict Cases — Paired Score Analysis")
        print("-" * 70)
        print()

        for sym in sorted(enriched_conflicts.keys()):
            recs = enriched_conflicts[sym]
            if not recs:
                continue
            print(f"  {sym}  (n={len(recs)} conflict cases)")

            # Score distributions by winner
            hydra_wins = [r for r in recs if r["winner"] == "hydra"]
            legacy_wins = [r for r in recs if r["winner"] == "legacy"]

            def _pair_stats(rs: List[Dict[str, Any]], label: str) -> None:
                if not rs:
                    print(f"    {label}: n=0")
                    return
                h_scores = [r["hydra_score"] for r in rs]
                l_scores = [r["legacy_score"] for r in rs]
                deltas = [r["hydra_score"] - r["legacy_score"] for r in rs]
                h_mean = sum(h_scores) / len(h_scores)
                l_mean = sum(l_scores) / len(l_scores)
                d_mean = sum(deltas) / len(deltas)
                print(f"    {label}: n={len(rs)}")
                print(f"      Hydra mean={h_mean:.4f}  Legacy mean={l_mean:.4f}  delta mean={d_mean:+.4f}")

            _pair_stats(hydra_wins, "hydra wins")
            _pair_stats(legacy_wins, "legacy wins")

            # Winner curve: delta buckets
            all_deltas = [(r["hydra_score"] - r["legacy_score"], r["winner"]) for r in recs]
            all_deltas.sort()

            # Check if winner flips smoothly around delta=0
            pos_hydra = sum(1 for d, w in all_deltas if d > 0 and w == "hydra")
            pos_legacy = sum(1 for d, w in all_deltas if d > 0 and w == "legacy")
            neg_hydra = sum(1 for d, w in all_deltas if d < 0 and w == "hydra")
            neg_legacy = sum(1 for d, w in all_deltas if d < 0 and w == "legacy")
            zero_cases = sum(1 for d, _ in all_deltas if d == 0)

            print(f"    Winner curve (delta = hydra - legacy):")
            print(f"      delta > 0:  hydra wins {pos_hydra}, legacy wins {pos_legacy}")
            print(f"      delta < 0:  hydra wins {neg_hydra}, legacy wins {neg_legacy}")
            print(f"      delta = 0:  {zero_cases}")
            if pos_hydra + pos_legacy > 0:
                consistency_pos = pos_hydra / (pos_hydra + pos_legacy) * 100
                print(f"      Consistency (delta>0 → hydra): {consistency_pos:.1f}%")
            if neg_hydra + neg_legacy > 0:
                consistency_neg = neg_legacy / (neg_hydra + neg_legacy) * 100
                print(f"      Consistency (delta<0 → legacy): {consistency_neg:.1f}%")

            # Delta bucket distribution
            buckets: Dict[str, Dict[str, int]] = {}
            for d, w in all_deltas:
                if d < -0.10:
                    b = "<-0.10"
                elif d < -0.05:
                    b = "-0.10..-0.05"
                elif d < 0:
                    b = "-0.05..0.00"
                elif d == 0:
                    b = "=0.00"
                elif d < 0.05:
                    b = "0.00..+0.05"
                elif d < 0.10:
                    b = "+0.05..+0.10"
                else:
                    b = ">+0.10"
                if b not in buckets:
                    buckets[b] = {"hydra": 0, "legacy": 0}
                buckets[b][w] = buckets[b].get(w, 0) + 1

            print(f"    Delta buckets:")
            for b in ["<-0.10", "-0.10..-0.05", "-0.05..0.00", "=0.00",
                       "0.00..+0.05", "+0.05..+0.10", ">+0.10"]:
                if b in buckets:
                    h = buckets[b].get("hydra", 0)
                    l = buckets[b].get("legacy", 0)
                    total = h + l
                    print(f"      {b:>15}:  hydra={h:>3}  legacy={l:>3}  (n={total})")
            print()
    else:
        print("-" * 70)
        print("  AUDIT 2: No enriched conflict cases yet.")
        print("  Paired score logging has been added — data will accumulate.")
        print("-" * 70)
        print()

    # ── Audit 3: Hydra score when it wins vs loses (ts-joined proxy) ─
    print("-" * 70)
    print("  AUDIT 3: Hydra Score When It Wins vs Loses (all events)")
    print("-" * 70)
    print()

    for sym in sorted(paired.keys()):
        recs = paired[sym]
        hydra_when_wins = [r["hydra_score"] for r in recs if r["winner"] == "hydra" and r["hydra_score"] is not None]
        hydra_when_loses = [r["hydra_score"] for r in recs if r["winner"] == "legacy" and r["hydra_score"] is not None]

        def _mean(vs: List[float]) -> str:
            return f"{sum(vs)/len(vs):.4f}" if vs else "N/A"

        def _median(vs: List[float]) -> str:
            return f"{sorted(vs)[len(vs)//2]:.4f}" if vs else "N/A"

        if hydra_when_wins or hydra_when_loses:
            print(f"  {sym}")
            print(f"    Hydra wins  (n={len(hydra_when_wins):>4}): mean={_mean(hydra_when_wins)}  median={_median(hydra_when_wins)}")
            print(f"    Hydra loses (n={len(hydra_when_loses):>4}): mean={_mean(hydra_when_loses)}  median={_median(hydra_when_loses)}")
            if hydra_when_wins and hydra_when_loses:
                sep = float(_mean(hydra_when_wins)) - float(_mean(hydra_when_loses))
                print(f"    Separation:  {sep:+.4f}  {'(Hydra scores higher when it wins — expected)' if sep > 0 else '(INVERTED — Hydra scores lower when it wins!)'}")
            print()

    # ── Summary ──────────────────────────────────────────────────────
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print()
    if total_enriched_conflict > 0:
        print(f"  Enriched conflict data:  {total_enriched_conflict} events — full paired analysis available")
    else:
        print("  Enriched conflict data:  0 events — score logging just enabled,")
        print("  re-run after ~100 cycles for full paired calibration analysis.")
    print(f"  Joined Hydra data:       {joined_count} events — proxy analysis available")
    print()
    print("  Next steps:")
    if total_enriched_conflict == 0:
        print("  1. Wait for enriched soak events to accumulate (~100 cycles)")
        print("  2. Re-run: PYTHONPATH=. python scripts/ecs_calibration_audit.py")
        print("  3. Audit 2 will show full paired conflict analysis")
    else:
        print("  Review Audit 2 delta buckets for winner-curve consistency.")
        print("  If delta>0 → hydra and delta<0 → legacy both >95%, scores are comparable.")
        print("  If not, a calibration layer may be needed.")


if __name__ == "__main__":
    run_audit()
