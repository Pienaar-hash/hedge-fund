#!/usr/bin/env python3
"""Shadow PnL Join — Candidate D profit-surface counterfactual analysis.

Joins episode_ledger (realized PnL) with the Candidate D profit mask
to answer the core questions:

  1. Is EV(mask interior) > EV(near-miss band)?  → mask correctness
  2. What is the lost opportunity from near-miss abstention?
  3. Would widening the mask improve risk-adjusted returns?

Join strategy:
  - Episode hybrid_score classifies each trade into mask / near-miss / outside
  - Shadow v3 events provide additional lineage (ts-proximity enrichment)
  - No intent_id join (shadow ≠ order lifecycle)

This is **observation-only** — no behavioral change, no parameter writes.

Usage:
    PYTHONPATH=. python scripts/shadow_pnl_join.py
    PYTHONPATH=. python scripts/shadow_pnl_join.py --json
    PYTHONPATH=. python scripts/shadow_pnl_join.py --symbol BTCUSDT
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Paths ───────────────────────────────────────────────────────────────────
EPISODE_LEDGER = Path("logs/state/episode_ledger.json")
V2_SHADOW_LOG = Path("logs/execution/selector_v2_shadow.jsonl")

# ── Profit mask (same as shadow_validation_report.py) ───────────────────────
REFERENCE_MASK: Dict[str, Dict[str, float]] = {
    "BTCUSDT": {"lo": 0.4197, "hi": 0.4953},
}
NEAR_MISS_BAND = 0.03  # above mask hi
MIN_EPISODES_SUFFICIENT = 10

# ── Counterfactual widening steps ───────────────────────────────────────────
WIDEN_STEPS = [0.01, 0.02, 0.03, 0.05]


# ── Region classification ───────────────────────────────────────────────────

def classify_region(
    symbol: str,
    score: float,
) -> str:
    """Classify a score into mask_interior / near_miss / outside."""
    ref = REFERENCE_MASK.get(symbol)
    if ref is None:
        return "no_mask"
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


def load_shadow_events(
    path: Optional[Path] = None,
    v3_only: bool = False,
) -> List[Dict[str, Any]]:
    p = path or V2_SHADOW_LOG
    if not p.exists():
        return []
    events: List[Dict[str, Any]] = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if v3_only and ev.get("schema") != "selector_v2_shadow_v3":
                continue
            events.append(ev)
    return events


# ── Episode → Region scoring ───────────────────────────────────────────────

def _parse_ts(ts_str: str) -> float:
    """Parse ISO timestamp to unix seconds."""
    if not ts_str:
        return 0.0
    try:
        dt = datetime.fromisoformat(ts_str)
        return dt.timestamp()
    except (ValueError, TypeError):
        return 0.0


def build_scored_episodes(
    episodes: List[Dict[str, Any]],
    symbols: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Filter and classify episodes by mask region.

    Returns list of dicts with: episode_id, symbol, hybrid_score, net_pnl,
    region, entry_ts, exit_ts.
    """
    target_symbols = symbols or set(REFERENCE_MASK.keys())
    scored: List[Dict[str, Any]] = []

    for ep in episodes:
        sym = ep.get("symbol", "")
        if sym not in target_symbols:
            continue

        h_score = ep.get("hybrid_score", 0)
        if not h_score or h_score <= 0:
            continue

        pnl = ep.get("net_pnl")
        if pnl is None:
            continue

        region = classify_region(sym, float(h_score))

        scored.append({
            "episode_id": ep.get("episode_id", ""),
            "symbol": sym,
            "hybrid_score": float(h_score),
            "net_pnl": float(pnl),
            "region": region,
            "entry_ts": _parse_ts(ep.get("entry_ts", "")),
            "exit_ts": _parse_ts(ep.get("exit_ts", "")),
            "side": ep.get("side", ""),
            "intent_id": ep.get("intent_id", ""),
        })

    scored.sort(key=lambda r: r["entry_ts"])
    return scored


# ── Shadow event linkage (ts-proximity enrichment) ─────────────────────────

def enrich_with_shadow(
    scored: List[Dict[str, Any]],
    shadow_events: List[Dict[str, Any]],
    window_s: float = 120.0,
) -> Tuple[int, int]:
    """Enrich scored episodes with nearest v3 shadow event metadata.

    Returns (matched_count, unmatched_count).
    """
    # Build index: symbol → sorted events by ts
    idx: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ev in shadow_events:
        sym = ev.get("symbol", "")
        if sym:
            idx[sym].append(ev)
    for sym in idx:
        idx[sym].sort(key=lambda e: e.get("ts", 0))

    matched = 0
    unmatched = 0

    for ep in scored:
        sym = ep["symbol"]
        entry_ts = ep["entry_ts"]
        candidates = idx.get(sym, [])

        best: Optional[Dict[str, Any]] = None
        best_dist = float("inf")
        for ev in candidates:
            dist = abs(ev.get("ts", 0) - entry_ts)
            if dist < best_dist:
                best_dist = dist
                best = ev

        if best and best_dist <= window_s:
            ep["shadow_linked"] = True
            ep["shadow_ts"] = best.get("ts", 0)
            ep["shadow_d_rule"] = best.get("d_rule", "")
            ep["shadow_d_choice"] = best.get("d_choice", "")
            ep["link_distance_s"] = round(best_dist, 1)
            matched += 1
        else:
            ep["shadow_linked"] = False
            unmatched += 1

    return matched, unmatched


# ── PnL statistics ──────────────────────────────────────────────────────────

def _pnl_stats(pnls: List[float]) -> Dict[str, Any]:
    if not pnls:
        return {
            "count": 0, "total": 0.0, "mean": 0.0,
            "win_rate": 0.0, "sharpe": 0.0,
        }
    total = sum(pnls)
    mean = total / len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    win_rate = wins / len(pnls)

    # Sharpe
    sharpe = 0.0
    if len(pnls) >= 2:
        var = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
        std = var ** 0.5
        if std > 1e-9:
            sharpe = mean / std

    return {
        "count": len(pnls),
        "total": round(total, 4),
        "mean": round(mean, 4),
        "win_rate": round(win_rate, 4),
        "sharpe": round(sharpe, 4),
    }


# ── Report sections ────────────────────────────────────────────────────────

def section_region_pnl(
    scored: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Section 1: PnL by region (mask interior / near-miss / outside)."""
    by_symbol: Dict[str, Dict[str, Any]] = {}

    for sym in sorted(set(ep["symbol"] for ep in scored)):
        sym_eps = [ep for ep in scored if ep["symbol"] == sym]
        regions: Dict[str, List[float]] = defaultdict(list)
        for ep in sym_eps:
            regions[ep["region"]].append(ep["net_pnl"])

        region_stats = {}
        for region in ["mask_interior", "near_miss", "outside"]:
            region_stats[region] = _pnl_stats(regions.get(region, []))

        sufficient = sum(r["count"] for r in region_stats.values()) >= MIN_EPISODES_SUFFICIENT
        by_symbol[sym] = {
            "regions": region_stats,
            "total_episodes": len(sym_eps),
            "data_sufficiency": "OK" if sufficient else "INSUFFICIENT_DATA",
        }

    return {"title": "Region PnL Table", "by_symbol": by_symbol}


def section_near_miss_comparison(
    scored: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Section 2: Near-miss vs mask interior comparison."""
    results: Dict[str, Any] = {}

    for sym in sorted(set(ep["symbol"] for ep in scored)):
        sym_eps = [ep for ep in scored if ep["symbol"] == sym]
        mask_pnls = [ep["net_pnl"] for ep in sym_eps if ep["region"] == "mask_interior"]
        nm_pnls = [ep["net_pnl"] for ep in sym_eps if ep["region"] == "near_miss"]

        mask_stats = _pnl_stats(mask_pnls)
        nm_stats = _pnl_stats(nm_pnls)

        # Verdict
        if not mask_pnls or not nm_pnls:
            verdict = "insufficient_data"
            verdict_detail = "Need episodes in both regions"
        elif mask_stats["mean"] > nm_stats["mean"] and mask_stats["mean"] > 0:
            verdict = "mask_correct"
            verdict_detail = "EV(mask) > EV(near_miss) — mask boundary likely correct"
        elif nm_stats["mean"] > mask_stats["mean"] and nm_stats["mean"] > 0:
            verdict = "mask_too_tight"
            verdict_detail = "EV(near_miss) > EV(mask) — mask upper bound likely too low"
        elif mask_stats["mean"] <= 0 and nm_stats["mean"] <= 0:
            verdict = "thesis_degrading"
            verdict_detail = "Both regions negative EV — Candidate D thesis weakening"
        else:
            verdict = "inconclusive"
            verdict_detail = "Mixed signals — continue observation"

        ev_delta = round(nm_stats["mean"] - mask_stats["mean"], 4) if mask_pnls and nm_pnls else None

        results[sym] = {
            "mask_interior": mask_stats,
            "near_miss": nm_stats,
            "ev_delta": ev_delta,
            "verdict": verdict,
            "verdict_detail": verdict_detail,
        }

    return {"title": "Near-Miss vs Mask Comparison", "by_symbol": results}


def section_lost_ev(
    scored: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Section 3: Lost EV estimate from near-miss abstention.

    If Candidate D abstains on near-miss trades that would have been
    profitable, this estimates the cost of that abstention.
    """
    results: Dict[str, Any] = {}

    for sym in sorted(set(ep["symbol"] for ep in scored)):
        sym_eps = [ep for ep in scored if ep["symbol"] == sym]
        nm_eps = [ep for ep in sym_eps if ep["region"] == "near_miss"]

        if not nm_eps:
            results[sym] = {"status": "no_near_miss_episodes"}
            continue

        # These are trades the live system took, but Candidate D would abstain
        nm_pnls = [ep["net_pnl"] for ep in nm_eps]
        profitable = [p for p in nm_pnls if p > 0]
        losing = [p for p in nm_pnls if p <= 0]

        lost_gains = sum(profitable)
        avoided_losses = abs(sum(losing))
        net_lost_ev = sum(nm_pnls)

        results[sym] = {
            "near_miss_episodes": len(nm_eps),
            "profitable_abstained": len(profitable),
            "losing_abstained": len(losing),
            "lost_gains": round(lost_gains, 4),
            "avoided_losses": round(avoided_losses, 4),
            "net_lost_ev": round(net_lost_ev, 4),
            "abstention_verdict": (
                "beneficial" if net_lost_ev < 0
                else ("costly" if net_lost_ev > 0
                      else "neutral")
            ),
        }

    return {"title": "Lost EV Estimate", "by_symbol": results}


def section_counterfactual_widening(
    scored: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Section 4: Counterfactual mask widening analysis.

    For each widening step, reclassify episodes and compute new EV.
    Observation-only — no parameter changes.
    """
    results: Dict[str, Any] = {}

    for sym in sorted(set(ep["symbol"] for ep in scored)):
        ref = REFERENCE_MASK.get(sym)
        if ref is None:
            continue

        sym_eps = [ep for ep in scored if ep["symbol"] == sym]
        lo, hi = ref["lo"], ref["hi"]
        current_mask_pnls = [ep["net_pnl"] for ep in sym_eps if lo <= ep["hybrid_score"] <= hi]
        current_stats = _pnl_stats(current_mask_pnls)

        scenarios: List[Dict[str, Any]] = []
        for step in WIDEN_STEPS:
            new_hi = hi + step
            widened_pnls = [
                ep["net_pnl"] for ep in sym_eps
                if lo <= ep["hybrid_score"] <= new_hi
            ]
            widened_stats = _pnl_stats(widened_pnls)
            added = widened_stats["count"] - current_stats["count"]
            ev_change = round(widened_stats["mean"] - current_stats["mean"], 4) if widened_pnls else None

            scenarios.append({
                "widen_by": step,
                "new_hi": round(new_hi, 4),
                "episodes_added": added,
                "widened_stats": widened_stats,
                "ev_change_vs_current": ev_change,
            })

        results[sym] = {
            "current_mask": f"[{lo:.4f}, {hi:.4f}]",
            "current_stats": current_stats,
            "scenarios": scenarios,
        }

    return {"title": "Counterfactual Widening", "by_symbol": results}


def section_join_quality(
    scored: List[Dict[str, Any]],
    matched: int,
    unmatched: int,
) -> Dict[str, Any]:
    """Section 5: Join quality and evidence-admission status."""
    total = matched + unmatched
    match_rate = matched / total if total > 0 else 0.0

    # Linkage quality tier
    if match_rate >= 0.80:
        tier = "HIGH"
    elif match_rate >= 0.50:
        tier = "MODERATE"
    elif match_rate > 0:
        tier = "LOW"
    else:
        tier = "NONE"

    # Per-symbol episode counts
    sym_counts: Dict[str, int] = defaultdict(int)
    sym_scored: Dict[str, int] = defaultdict(int)
    for ep in scored:
        sym_counts[ep["symbol"]] += 1
        if ep.get("shadow_linked"):
            sym_scored[ep["symbol"]] += 1

    by_symbol = {}
    for sym in sorted(sym_counts.keys()):
        n = sym_counts[sym]
        by_symbol[sym] = {
            "episodes": n,
            "shadow_linked": sym_scored.get(sym, 0),
            "data_sufficiency": "OK" if n >= MIN_EPISODES_SUFFICIENT else "INSUFFICIENT_DATA",
        }

    return {
        "title": "Join Quality",
        "total_scored_episodes": len(scored),
        "shadow_matched": matched,
        "shadow_unmatched": unmatched,
        "match_rate": round(match_rate, 4),
        "linkage_tier": tier,
        "by_symbol": by_symbol,
    }


# ── Report generation ──────────────────────────────────────────────────────

def generate_report(
    *,
    symbol: Optional[str] = None,
    json_output: bool = False,
    episode_path: Optional[Path] = None,
    shadow_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generate the full PnL join report."""
    episodes = load_episodes(episode_path)
    shadow_events = load_shadow_events(shadow_path)

    target_symbols = {symbol} if symbol else None
    scored = build_scored_episodes(episodes, symbols=target_symbols)

    matched, unmatched = enrich_with_shadow(scored, shadow_events)

    sections = [
        section_region_pnl(scored),
        section_near_miss_comparison(scored),
        section_lost_ev(scored),
        section_counterfactual_widening(scored),
        section_join_quality(scored, matched, unmatched),
    ]

    report = {
        "generated_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_scored_episodes": len(scored),
        "sections": {s["title"]: s for s in sections},
    }

    if json_output:
        json.dump(report, sys.stdout, indent=2)
        print()
    else:
        _print_report(sections)

    return report


# ── Text output ─────────────────────────────────────────────────────────────

def _print_report(sections: List[Dict[str, Any]]) -> None:
    bar = "=" * 72
    now_iso = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"\n{bar}")
    print(f"  SHADOW PnL JOIN REPORT — Candidate D")
    print(f"  Generated: {now_iso}")
    print(bar)

    for s in sections:
        title = s.get("title", "")
        print(f"\n{'─' * 72}")
        print(f"  {title}")
        print(f"{'─' * 72}")

        if title == "Region PnL Table":
            _print_region_pnl(s)
        elif title == "Near-Miss vs Mask Comparison":
            _print_near_miss_comparison(s)
        elif title == "Lost EV Estimate":
            _print_lost_ev(s)
        elif title == "Counterfactual Widening":
            _print_widening(s)
        elif title == "Join Quality":
            _print_join_quality(s)

    print(f"\n{bar}")
    print(f"  END OF REPORT")
    print(bar)


def _print_region_pnl(s: Dict[str, Any]) -> None:
    for sym, data in sorted(s.get("by_symbol", {}).items()):
        suf = data.get("data_sufficiency", "OK")
        suf_tag = f"  [{suf}]" if suf != "OK" else ""
        print(f"\n  {sym}  (n={data['total_episodes']}){suf_tag}\n")
        print(f"    {'Region':<16s}  {'Count':>5s}  {'Total':>10s}  "
              f"{'Mean':>8s}  {'Win%':>6s}  {'Sharpe':>7s}")
        print(f"    {'─' * 60}")
        for region in ["mask_interior", "near_miss", "outside"]:
            rs = data["regions"].get(region, {})
            if rs.get("count", 0) == 0:
                print(f"    {region:<16s}  {'—':>5s}")
                continue
            print(f"    {region:<16s}  {rs['count']:5d}  "
                  f"{rs['total']:10.4f}  {rs['mean']:8.4f}  "
                  f"{rs['win_rate']:5.1%}  {rs['sharpe']:7.4f}")


def _print_near_miss_comparison(s: Dict[str, Any]) -> None:
    for sym, data in sorted(s.get("by_symbol", {}).items()):
        print(f"\n  {sym}")
        mask = data["mask_interior"]
        nm = data["near_miss"]
        print(f"    EV(mask):      {mask['mean']:+.4f}  (n={mask['count']})")
        print(f"    EV(near_miss): {nm['mean']:+.4f}  (n={nm['count']})")
        if data["ev_delta"] is not None:
            print(f"    EV delta:      {data['ev_delta']:+.4f}  "
                  f"(near_miss - mask)")
        print(f"\n    Verdict: {data['verdict'].upper()}")
        print(f"    {data['verdict_detail']}")


def _print_lost_ev(s: Dict[str, Any]) -> None:
    for sym, data in sorted(s.get("by_symbol", {}).items()):
        if data.get("status") == "no_near_miss_episodes":
            print(f"\n  {sym}: no near-miss episodes")
            continue
        print(f"\n  {sym}  ({data['near_miss_episodes']} near-miss episodes)")
        print(f"    Profitable trades abstained:  {data['profitable_abstained']}")
        print(f"    Losing trades abstained:      {data['losing_abstained']}")
        print(f"    Lost gains (foregone):        {data['lost_gains']:+.4f}")
        print(f"    Avoided losses:               {data['avoided_losses']:+.4f}")
        print(f"    Net lost EV:                  {data['net_lost_ev']:+.4f}")
        verdict = data["abstention_verdict"]
        marker = {"beneficial": "✓ abstention saves money",
                  "costly": "✗ abstention costs money",
                  "neutral": "─ neutral"}
        print(f"    Abstention impact: {marker.get(verdict, verdict)}")


def _print_widening(s: Dict[str, Any]) -> None:
    for sym, data in sorted(s.get("by_symbol", {}).items()):
        print(f"\n  {sym}  current mask: {data['current_mask']}")
        cs = data["current_stats"]
        print(f"    Current:  EV={cs['mean']:+.4f}  n={cs['count']}  "
              f"Sharpe={cs['sharpe']:.4f}\n")
        print(f"    {'Widen':>7s}  {'New Hi':>8s}  {'Added':>5s}  "
              f"{'EV':>8s}  {'ΔEV':>8s}  {'Sharpe':>7s}")
        print(f"    {'─' * 52}")
        for sc in data["scenarios"]:
            ev_str = f"{sc['widened_stats']['mean']:+.4f}"
            delta_str = f"{sc['ev_change_vs_current']:+.4f}" if sc["ev_change_vs_current"] is not None else "  n/a"
            print(f"    +{sc['widen_by']:.2f}   {sc['new_hi']:.4f}  "
                  f"{sc['episodes_added']:5d}  {ev_str:>8s}  "
                  f"{delta_str:>8s}  {sc['widened_stats']['sharpe']:7.4f}")


def _print_join_quality(s: Dict[str, Any]) -> None:
    print(f"\n  Total scored episodes:   {s['total_scored_episodes']}")
    print(f"  Shadow v3 matched:       {s['shadow_matched']}")
    print(f"  Shadow unmatched:        {s['shadow_unmatched']}")
    print(f"  Match rate:              {s['match_rate']:.1%}")
    print(f"  Linkage tier:            {s['linkage_tier']}")

    by_sym = s.get("by_symbol", {})
    if by_sym:
        print(f"\n    {'Symbol':<12s}  {'Episodes':>8s}  {'Linked':>6s}  {'Sufficiency'}")
        print(f"    {'─' * 45}")
        for sym, info in sorted(by_sym.items()):
            print(f"    {sym:<12s}  {info['episodes']:8d}  "
                  f"{info['shadow_linked']:6d}  {info['data_sufficiency']}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shadow PnL Join Report — Candidate D",
    )
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--symbol", type=str, default=None, help="Filter to symbol")
    args = parser.parse_args()
    generate_report(symbol=args.symbol, json_output=args.json)


if __name__ == "__main__":
    main()
