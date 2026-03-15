#!/usr/bin/env python3
"""ECS Arbitration Phase Map — conflict-case score analysis.

Produces a text-based phase map showing Hydra score vs Legacy score
for every conflict case. Uses enriched soak events (paired scores)
when available, with timestamp-joined proxy data as fallback.

Sections:
  1. Enriched paired analysis (when available)
  2. Hydra distribution by winner (proxy)
  3. Decision boundary detection (sliding window 50% crossings)
  4. Regime segmentation (contiguous winner-dominant regions)

Usage:
    PYTHONPATH=. python scripts/ecs_phase_map.py
"""
from __future__ import annotations

import bisect
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SOAK_PATH = Path("logs/execution/ecs_soak_events.jsonl")
SCORE_DECOMP_PATH = Path("logs/execution/score_decomposition.jsonl")

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
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val).timestamp()
        except ValueError:
            pass
    return None


def _build_score_index(events: List[Dict[str, Any]]) -> Dict[str, List[Tuple[float, float]]]:
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


def _find_boundaries(records: List[Dict[str, Any]], window: int = 30) -> List[Dict[str, Any]]:
    """Slide a window over sorted hydra scores and find 50% crossings.

    Returns list of boundary dicts with threshold, confidence, and
    surrounding context.
    """
    if len(records) < window:
        return []

    sorted_recs = sorted(records, key=lambda r: r["hydra"])
    boundaries: List[Dict[str, Any]] = []

    prev_hydra_frac: Optional[float] = None
    for i in range(len(sorted_recs) - window + 1):
        w = sorted_recs[i : i + window]
        h_wins = sum(1 for r in w if r["winner"] == "hydra")
        h_frac = h_wins / window

        if prev_hydra_frac is not None:
            crossed_up = prev_hydra_frac < 0.5 and h_frac >= 0.5
            crossed_down = prev_hydra_frac >= 0.5 and h_frac < 0.5
            if crossed_up or crossed_down:
                mid_score = (w[0]["hydra"] + w[-1]["hydra"]) / 2
                direction = "hydra_enters" if crossed_up else "legacy_enters"
                # Sharpness: how quickly does it transition?
                lo_score = sorted_recs[max(0, i - 5)]["hydra"]
                hi_score = sorted_recs[min(len(sorted_recs) - 1, i + window + 4)]["hydra"]
                transition_width = hi_score - lo_score
                boundaries.append({
                    "threshold": mid_score,
                    "direction": direction,
                    "window_lo": w[0]["hydra"],
                    "window_hi": w[-1]["hydra"],
                    "transition_width": transition_width,
                    "h_frac_before": prev_hydra_frac,
                    "h_frac_after": h_frac,
                })
        prev_hydra_frac = h_frac

    # Deduplicate boundaries within 0.02 of each other
    deduped: List[Dict[str, Any]] = []
    for b in boundaries:
        if not deduped or abs(b["threshold"] - deduped[-1]["threshold"]) > 0.02:
            deduped.append(b)
    return deduped


def _segment_regimes(records: List[Dict[str, Any]], boundaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Partition score space into regimes using detected boundaries."""
    if not records:
        return []

    sorted_recs = sorted(records, key=lambda r: r["hydra"])
    lo = sorted_recs[0]["hydra"]
    hi = sorted_recs[-1]["hydra"]

    # Build edges from boundaries
    edges = [lo] + [b["threshold"] for b in boundaries] + [hi + 0.0001]
    segments: List[Dict[str, Any]] = []

    for i in range(len(edges) - 1):
        seg_lo, seg_hi = edges[i], edges[i + 1]
        in_seg = [r for r in sorted_recs if seg_lo <= r["hydra"] < seg_hi]
        if not in_seg:
            continue
        h_wins = sum(1 for r in in_seg if r["winner"] == "hydra")
        n = len(in_seg)
        h_pct = h_wins / n * 100
        dominant = "HYDRA" if h_pct >= 50 else "LEGACY"
        segments.append({
            "lo": seg_lo,
            "hi": seg_hi,
            "n": n,
            "hydra_wins": h_wins,
            "legacy_wins": n - h_wins,
            "hydra_pct": h_pct,
            "dominant": dominant,
        })

    return segments


def _print_decision_boundaries(paired: Dict[str, List[Dict[str, Any]]]) -> None:
    print("-" * 70)
    print("  DECISION BOUNDARIES (50% winner-probability crossings)")
    print("-" * 70)
    print()

    all_boundaries: Dict[str, List[Dict[str, Any]]] = {}

    for sym in sorted(paired.keys()):
        recs = [r for r in paired[sym] if r.get("hydra") is not None]
        if len(recs) < 30:
            print(f"  {sym}: insufficient data (n={len(recs)}, need >=30)")
            print()
            continue

        # Adaptive window: ~10% of data, clamped [20, 50]
        window = max(20, min(50, len(recs) // 10))
        boundaries = _find_boundaries(recs, window=window)
        all_boundaries[sym] = boundaries

        hydra_wins = [r for r in recs if r["winner"] == "hydra"]
        legacy_wins = [r for r in recs if r["winner"] == "legacy"]

        print(f"  {sym}  (n={len(recs)}, window={window})")
        print(f"    Overall: hydra={len(hydra_wins)}  legacy={len(legacy_wins)}")

        if not boundaries:
            if len(hydra_wins) == 0:
                print(f"    No boundaries found — Legacy dominant across entire range")
            elif len(legacy_wins) == 0:
                print(f"    No boundaries found — Hydra dominant across entire range")
            else:
                print(f"    No clean 50% crossings detected")
        else:
            print(f"    Detected {len(boundaries)} boundary(ies):")
            print()
            for j, b in enumerate(boundaries, 1):
                arrow = "→ HYDRA enters" if b["direction"] == "hydra_enters" else "→ LEGACY enters"
                sharpness = "SHARP" if b["transition_width"] < 0.03 else "GRADUAL" if b["transition_width"] < 0.06 else "WIDE"
                print(f"      B{j}: hydra_score ≈ {b['threshold']:.4f}  {arrow}")
                print(f"          range: [{b['window_lo']:.4f}, {b['window_hi']:.4f}]")
                print(f"          p(hydra) {b['h_frac_before']:.0%} → {b['h_frac_after']:.0%}")
                print(f"          transition width: {b['transition_width']:.4f}  ({sharpness})")
            print()

            # Compute piecewise regime description from actual data
            segments = _segment_regimes(recs, boundaries)
            if segments:
                print(f"    Regime structure:")
                for seg in segments:
                    print(f"      [{seg['lo']:.4f}, {seg['hi']:.4f})  → {seg['dominant']}  ({seg['hydra_pct']:.0f}% hydra, n={seg['n']})")

        print()


def _print_regime_segmentation(paired: Dict[str, List[Dict[str, Any]]]) -> None:
    print("-" * 70)
    print("  REGIME SEGMENTATION")
    print("-" * 70)
    print()

    for sym in sorted(paired.keys()):
        recs = [r for r in paired[sym] if r.get("hydra") is not None]
        if len(recs) < 30:
            continue

        window = max(20, min(50, len(recs) // 10))
        boundaries = _find_boundaries(recs, window=window)
        segments = _segment_regimes(recs, boundaries)

        print(f"  {sym}")
        if not segments:
            print(f"    No segments (no data)")
            print()
            continue

        print(f"    {'hydra score range':>25}  {'dominant':>8}  {'hydra':>5}  {'legacy':>6}  {'n':>4}  {'hydra%':>7}  {'confidence'}")
        print(f"    {'-'*25}  {'-'*8}  {'-'*5}  {'-'*6}  {'-'*4}  {'-'*7}  {'-'*15}")

        for seg in segments:
            n = seg["n"]
            h_pct = seg["hydra_pct"]
            # Confidence: how far from 50%
            conf_pct = abs(h_pct - 50.0)
            if conf_pct > 40:
                conf = "VERY HIGH"
            elif conf_pct > 25:
                conf = "HIGH"
            elif conf_pct > 10:
                conf = "MODERATE"
            else:
                conf = "LOW (ambiguous)"

            print(f"    [{seg['lo']:.4f}, {seg['hi']:.4f})  {seg['dominant']:>8}"
                  f"  {seg['hydra_wins']:>5}  {seg['legacy_wins']:>6}  {n:>4}"
                  f"  {h_pct:>6.1f}%  {conf}")

        # Summary line
        hydra_regimes = [s for s in segments if s["dominant"] == "HYDRA"]
        legacy_regimes = [s for s in segments if s["dominant"] == "LEGACY"]
        hydra_n = sum(s["n"] for s in hydra_regimes)
        legacy_n = sum(s["n"] for s in legacy_regimes)
        total = hydra_n + legacy_n
        print()
        print(f"    Regime coverage: HYDRA={hydra_n}/{total} ({hydra_n/total*100:.1f}%)"
              f"  LEGACY={legacy_n}/{total} ({legacy_n/total*100:.1f}%)"
              if total > 0 else "    No data")
        if boundaries:
            thresholds_str = ", ".join(f"{b['threshold']:.4f}" for b in boundaries)
            print(f"    Boundary thresholds: [{thresholds_str}]")
        print()


def run() -> None:
    soak_events = _load_jsonl(SOAK_PATH)
    score_events = _load_jsonl(SCORE_DECOMP_PATH)
    score_index = _build_score_index(score_events)

    # Collect paired data: prefer enriched, fallback to ts-join for hydra
    paired: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    enriched_count = 0
    joined_count = 0

    for e in soak_events:
        sym = e.get("symbol")
        if not sym:
            continue

        ts = _parse_ts(e.get("ts"))
        winner = e.get("ecs_winner")
        h_score = e.get("merge_hydra_score")
        l_score = e.get("merge_legacy_score")
        conflict = e.get("merge_conflict", False)
        cand_count = e.get("candidates_count", 0)

        if h_score is not None and l_score is not None:
            enriched_count += 1
            paired[sym].append({
                "hydra": float(h_score),
                "legacy": float(l_score),
                "winner": winner,
                "source": "enriched",
            })
        elif ts is not None and sym in score_index and cand_count >= 2:
            hydra = _nearest_score(score_index[sym], ts)
            if hydra is not None:
                joined_count += 1
                paired[sym].append({
                    "hydra": hydra,
                    "legacy": None,
                    "winner": winner,
                    "source": "joined",
                })

    total = enriched_count + joined_count

    print("=" * 70)
    print("  ECS Arbitration Phase Map")
    print("=" * 70)
    print()
    print(f"  Enriched paired events:  {enriched_count}")
    print(f"  Timestamp-joined events: {joined_count}")
    print(f"  Total analyzable:        {total}")
    print()

    # ── Section 1: Paired conflict analysis (enriched only) ──────────
    enriched_data = {sym: [r for r in recs if r["source"] == "enriched"]
                     for sym, recs in paired.items()}
    has_enriched = any(len(v) > 0 for v in enriched_data.values())

    if has_enriched:
        print("-" * 70)
        print("  PHASE MAP: Hydra Score vs Legacy Score (conflict cases)")
        print("-" * 70)
        print()

        for sym in sorted(enriched_data.keys()):
            recs = enriched_data[sym]
            if not recs:
                continue

            print(f"  {sym}  (n={len(recs)} conflict cases)")
            print()

            # Delta distribution and winner curve
            deltas = [(r["hydra"] - r["legacy"], r["winner"]) for r in recs]
            deltas.sort()

            # Bucket analysis
            buckets = [
                ("<-0.10", lambda d: d < -0.10),
                ("-0.10..-0.05", lambda d: -0.10 <= d < -0.05),
                ("-0.05..0.00", lambda d: -0.05 <= d < 0.00),
                ("0.00..+0.05", lambda d: 0.00 <= d < 0.05),
                ("+0.05..+0.10", lambda d: 0.05 <= d < 0.10),
                (">+0.10", lambda d: d >= 0.10),
            ]

            print(f"    {'delta bucket':>15}  {'hydra':>5}  {'legacy':>6}  {'n':>4}  {'hydra%':>7}  {'bar'}")
            print(f"    {'-'*15}  {'-'*5}  {'-'*6}  {'-'*4}  {'-'*7}  {'-'*20}")

            for label, pred in buckets:
                in_bucket = [(d, w) for d, w in deltas if pred(d)]
                h_wins = sum(1 for _, w in in_bucket if w == "hydra")
                l_wins = sum(1 for _, w in in_bucket if w == "legacy")
                n = len(in_bucket)
                if n == 0:
                    continue
                h_pct = h_wins / n * 100
                bar_len = int(h_pct / 5)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                print(f"    {label:>15}  {h_wins:>5}  {l_wins:>6}  {n:>4}  {h_pct:>6.1f}%  {bar}")

            print()

            # Score scatter summary (text-based)
            h_scores = [r["hydra"] for r in recs]
            l_scores = [r["legacy"] for r in recs]
            h_mean = sum(h_scores) / len(h_scores)
            l_mean = sum(l_scores) / len(l_scores)
            print(f"    Hydra mean={h_mean:.4f}  Legacy mean={l_mean:.4f}  delta mean={h_mean - l_mean:+.4f}")

            # Score range overlap
            h_min, h_max = min(h_scores), max(h_scores)
            l_min, l_max = min(l_scores), max(l_scores)
            overlap_lo = max(h_min, l_min)
            overlap_hi = min(h_max, l_max)
            if overlap_lo < overlap_hi:
                in_overlap = sum(1 for r in recs if overlap_lo <= r["hydra"] <= overlap_hi
                                 and overlap_lo <= r["legacy"] <= overlap_hi)
                print(f"    Overlap region: [{overlap_lo:.4f}, {overlap_hi:.4f}]  events_in_overlap={in_overlap}")
            else:
                print(f"    No score range overlap (scales are disjoint)")

            print()

    # ── Section 2: Hydra distribution by winner (joined proxy) ───────
    print("-" * 70)
    print("  HYDRA SCORE BY WINNER (proxy — conflict cases only)")
    print("-" * 70)
    print()

    for sym in sorted(paired.keys()):
        all_recs = [r for r in paired[sym] if r.get("hydra") is not None]
        if not all_recs:
            continue

        hydra_wins = [r["hydra"] for r in all_recs if r["winner"] == "hydra"]
        legacy_wins = [r["hydra"] for r in all_recs if r["winner"] == "legacy"]

        def _stats(vals: List[float]) -> str:
            if not vals:
                return "n=0"
            vals_s = sorted(vals)
            n = len(vals_s)
            mean = sum(vals_s) / n
            p25 = vals_s[int(n * 0.25)]
            med = vals_s[n // 2]
            p75 = vals_s[int(n * 0.75)]
            return f"n={n:>4}  mean={mean:.4f}  med={med:.4f}  p25={p25:.4f}  p75={p75:.4f}"

        print(f"  {sym}")
        print(f"    When hydra wins:  {_stats(hydra_wins)}")
        print(f"    When legacy wins: {_stats(legacy_wins)}")

        if hydra_wins and legacy_wins:
            h_mean = sum(hydra_wins) / len(hydra_wins)
            l_mean = sum(legacy_wins) / len(legacy_wins)
            sep = h_mean - l_mean
            direction = "EXPECTED (higher when wins)" if sep > 0 else "INVERTED (higher when loses)"
            print(f"    Separation: {sep:+.4f}  {direction}")

        # Histogram buckets (text-mode)
        all_scores = hydra_wins + legacy_wins
        if all_scores:
            lo = min(all_scores)
            hi = max(all_scores)
            step = max((hi - lo) / 8, 0.01)
            print(f"    Score range: [{lo:.4f}, {hi:.4f}]")
            print()
            print(f"    {'score range':>20}  {'hydra wins':>10}  {'legacy wins':>11}  {'hydra%':>7}")
            print(f"    {'-'*20}  {'-'*10}  {'-'*11}  {'-'*7}")
            edge = lo
            while edge < hi:
                upper = min(edge + step, hi + 0.0001)
                hw = sum(1 for v in hydra_wins if edge <= v < upper)
                lw = sum(1 for v in legacy_wins if edge <= v < upper)
                n = hw + lw
                if n > 0:
                    pct = hw / n * 100
                    print(f"    [{edge:.4f}, {upper:.4f})  {hw:>10}  {lw:>11}  {pct:>6.1f}%")
                edge = upper
        print()

    # ── Section 3: Decision Boundaries (sliding window) ──────────────
    _print_decision_boundaries(paired)

    # ── Section 4: Regime Segmentation ───────────────────────────────
    _print_regime_segmentation(paired)

    # ── Summary ──────────────────────────────────────────────────────
    print("=" * 70)
    print("  REGIME SWITCHING SUMMARY")
    print("=" * 70)
    print()

    if has_enriched:
        print("  Full paired analysis available (Audit sections above).")
        print("  Key metrics to check:")
        print("  - Delta buckets: does hydra% increase monotonically with delta?")
        print("  - If delta>0 but hydra% < 80%: calibration mismatch confirmed")
        print("  - If delta<0 but hydra% > 20%: calibration mismatch confirmed")
    else:
        print("  No enriched paired events yet.")
        print("  Proxy analysis (Hydra distribution by winner) available above.")
        print("  Re-run after enriched events accumulate for full conflict analysis.")

    print()
    print("  Per-symbol overview:")
    for sym in sorted(paired.keys()):
        all_recs = [r for r in paired[sym] if r.get("hydra") is not None]
        if not all_recs:
            continue
        hydra_wins = [r for r in all_recs if r["winner"] == "hydra"]
        legacy_wins = [r for r in all_recs if r["winner"] == "legacy"]
        n = len(all_recs)
        h_pct = len(hydra_wins) / n * 100 if n else 0

        window = max(20, min(50, n // 10))
        boundaries = _find_boundaries(all_recs, window=window) if n >= 30 else []
        n_boundaries = len(boundaries)
        regime_type = (
            "single-regime (legacy)" if len(hydra_wins) == 0
            else "single-regime (hydra)" if len(legacy_wins) == 0
            else f"{n_boundaries + 1}-regime switch" if n_boundaries > 0
            else "mixed (no clean boundary)"
        )

        thresh_str = ""
        if boundaries:
            thresh_str = "  boundaries=[" + ", ".join(f"{b['threshold']:.4f}" for b in boundaries) + "]"

        print(f"  {sym}: n={n:>4}  hydra={len(hydra_wins):>4} ({h_pct:>5.1f}%)  "
              f"legacy={len(legacy_wins):>4}  type={regime_type}{thresh_str}")


if __name__ == "__main__":
    run()
