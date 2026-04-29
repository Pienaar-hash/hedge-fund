#!/usr/bin/env python3
"""Shadow Validation Report Generator — Candidate D Operational Diagnostics.

Reads ``selector_v2_shadow.jsonl`` (v1/v2/v3 mixed) and outputs:
  1. Daily Candidate D throughput summary
  2. ZERO_SCORE vs NO_SCORE diagnostics
  3. Mask hit rate & distribution
  4. Score-vs-mask boundary analysis
  5. Drift alerts (mask occupancy vs baseline)
  6. Promotion readiness snapshot (soak days, episode count, kill conditions)

This is a **live log consumer** — it reads the shadow JSONL directly,
not the episode ledger.  For retrospective PnL analysis, use
``ecs_shadow_v2_eval.py`` instead.

Usage:
    PYTHONPATH=. python scripts/shadow_validation_report.py
    PYTHONPATH=. python scripts/shadow_validation_report.py --json   # machine-readable
    PYTHONPATH=. python scripts/shadow_validation_report.py --days 7 # last N days
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

V2_SHADOW_LOG = Path("logs/execution/selector_v2_shadow.jsonl")

# ── Reference baselines (from ECS_STRUCTURAL_DIAGNOSTIC 2026-04-07) ─────────
REFERENCE_MASK = {
    "BTCUSDT": {"lo": 0.4197, "hi": 0.4953, "overlap": 0.889},
}
REFERENCE_DATE = "2026-04-07"
SOAK_START_DATE = "2026-04-07"

# ── Promotion thresholds ────────────────────────────────────────────────────
PROMOTE_SOAK_DAYS = 30
PROMOTE_EPISODE_COUNT = 100
PROMOTE_MIN_DAILY_D_PROFIT_REGION = 5

# ── Kill condition thresholds ───────────────────────────────────────────────
KILL_OVERLAP_MIN = 0.70
KILL_REGION_DRIFT_MAX = 0.03
KILL_REGION_COUNT_14D_MIN = 30
MIN_SCORES_SUFFICIENT = 30


# ── Event loading ───────────────────────────────────────────────────────────

def _load_events(
    *,
    since_ts: Optional[float] = None,
    v3_only: bool = False,
) -> List[Dict[str, Any]]:
    if not V2_SHADOW_LOG.exists():
        return []
    events: List[Dict[str, Any]] = []
    with open(V2_SHADOW_LOG) as f:
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
            if since_ts and ev.get("ts", 0) < since_ts:
                continue
            events.append(ev)
    return events


def _ts_to_date(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _ts_to_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sharpe(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    std = var ** 0.5
    return round(mean / std, 3) if std > 1e-9 else 0.0


# ── Section builders ────────────────────────────────────────────────────────

def _section_daily_throughput(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Section 1: Daily Candidate D throughput."""
    # Group by day
    daily: Dict[str, Dict[str, int]] = defaultdict(lambda: Counter())
    daily_symbols: Dict[str, Dict[str, int]] = defaultdict(lambda: Counter())

    for ev in events:
        day = _ts_to_date(ev["ts"])
        d_rule = ev.get("d_rule", "unknown")
        sym = ev.get("symbol", "?")
        daily[day][d_rule] += 1
        daily_symbols[day][sym] += 1

    rows = []
    for day in sorted(daily.keys()):
        rules = daily[day]
        total = sum(rules.values())
        profit_region = rules.get("D_profit_region", 0)
        abstain = rules.get("D_abstain", 0)
        zero = rules.get("D_zero_score", 0)
        no_score = rules.get("D_no_score", 0)
        rows.append({
            "date": day,
            "total": total,
            "D_profit_region": profit_region,
            "D_abstain": abstain,
            "D_zero_score": zero,
            "D_no_score": no_score,
            "symbols": dict(daily_symbols[day]),
        })

    return {
        "title": "Daily Candidate D Throughput",
        "rows": rows,
        "total_events": len(events),
        "total_days": len(daily),
    }


def _section_zero_score(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Section 2: ZERO_SCORE vs NO_SCORE diagnostics."""
    d_events = [e for e in events if e.get("d_rule") in ("D_zero_score", "D_no_score")]
    zero_count = sum(1 for e in d_events if e.get("d_rule") == "D_zero_score")
    no_count = sum(1 for e in d_events if e.get("d_rule") == "D_no_score")

    total = len(events)
    zero_rate = zero_count / total if total else 0.0
    no_rate = no_count / total if total else 0.0

    # Per-symbol breakdown
    sym_zero: Dict[str, int] = Counter()
    sym_no: Dict[str, int] = Counter()
    for e in d_events:
        sym = e.get("symbol", "?")
        if e.get("d_rule") == "D_zero_score":
            sym_zero[sym] += 1
        else:
            sym_no[sym] += 1

    return {
        "title": "ZERO_SCORE vs NO_SCORE Diagnostics",
        "total_events": total,
        "D_zero_score": zero_count,
        "D_zero_score_rate": round(zero_rate, 4),
        "D_no_score": no_count,
        "D_no_score_rate": round(no_rate, 4),
        "by_symbol_zero": dict(sym_zero),
        "by_symbol_no": dict(sym_no),
        "health": {
            "pipeline_ok": no_rate < 0.05,
            "model_informative": zero_rate < 0.10,
        },
    }


def _section_mask_hits(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Section 3: Mask hit rate and distribution."""
    # Only events with D-fields (v2+ or v3)
    d_events = [e for e in events if "d_rule" in e]
    if not d_events:
        return {"title": "Mask Hit Rate & Distribution", "status": "no_d_events"}

    total = len(d_events)
    profit_hits = [e for e in d_events if e.get("d_rule") == "D_profit_region"]
    hit_count = len(profit_hits)
    hit_rate = hit_count / total if total else 0.0

    # Per-symbol
    sym_hits: Dict[str, int] = Counter()
    sym_total: Dict[str, int] = Counter()
    for e in d_events:
        sym = e.get("symbol", "?")
        sym_total[sym] += 1
        if e.get("d_rule") == "D_profit_region":
            sym_hits[sym] += 1

    per_symbol = {}
    for sym in sorted(sym_total.keys()):
        h = sym_hits.get(sym, 0)
        t = sym_total[sym]
        per_symbol[sym] = {
            "hits": h,
            "total": t,
            "rate": round(h / t, 4) if t else 0.0,
        }

    return {
        "title": "Mask Hit Rate & Distribution",
        "total_d_events": total,
        "profit_region_hits": hit_count,
        "overall_hit_rate": round(hit_rate, 4),
        "by_symbol": per_symbol,
    }


def _section_score_boundary(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Section 4: Score distribution vs mask boundaries."""
    # Collect scores for symbols with masks
    score_data: Dict[str, List[float]] = defaultdict(list)
    for e in events:
        sym = e.get("symbol", "?")
        score = e.get("hydra_score")
        if score is not None and sym in REFERENCE_MASK:
            score_data[sym].append(float(score))

    analysis = {}
    for sym, ref in REFERENCE_MASK.items():
        scores = score_data.get(sym, [])
        if not scores:
            analysis[sym] = {"status": "no_data"}
            continue

        lo, hi = ref["lo"], ref["hi"]
        inside = sum(1 for s in scores if lo <= s <= hi)
        below = sum(1 for s in scores if s < lo)
        above = sum(1 for s in scores if s > hi)
        near_lo = sum(1 for s in scores if abs(s - lo) < 0.02)
        near_hi = sum(1 for s in scores if abs(s - hi) < 0.02)

        mean_score = sum(scores) / len(scores) if scores else 0.0
        mask_midpoint = (lo + hi) / 2

        analysis[sym] = {
            "total_scores": len(scores),
            "inside_mask": inside,
            "inside_rate": round(inside / len(scores), 4) if scores else 0.0,
            "below_mask": below,
            "above_mask": above,
            "near_lower_bound": near_lo,
            "near_upper_bound": near_hi,
            "mean_score": round(mean_score, 4),
            "mask_midpoint": round(mask_midpoint, 4),
            "score_vs_mask": "centered" if abs(mean_score - mask_midpoint) < 0.02
            else ("low_bias" if mean_score < mask_midpoint else "high_bias"),
        }

    return {
        "title": "Score vs Mask Boundary Analysis",
        "by_symbol": analysis,
    }


# ── Near-miss band: scores just above mask upper bound ──────────────────
NEAR_MISS_BAND = 0.03  # how far above mask_hi counts as "near miss"

# ── Score density histogram bins ────────────────────────────────────────
SCORE_BINS = [
    ("<0.42",       0.0,   0.42),
    ("0.42–0.45",   0.42,  0.45),
    ("0.45–0.4953", 0.45,  0.4953),  # mask interior
    ("0.4953–0.53", 0.4953, 0.53),   # near-miss zone
    (">0.53",       0.53,  9.99),
]


def _section_score_drift_metrics(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Section 8: Score drift metrics — distance, near-miss, density, velocity."""
    analysis: Dict[str, Any] = {}

    for sym, ref in REFERENCE_MASK.items():
        scored_events = [
            (float(e["hydra_score"]), e.get("ts", 0))
            for e in events
            if e.get("symbol") == sym and e.get("hydra_score") is not None
        ]
        if not scored_events:
            analysis[sym] = {"status": "no_data"}
            continue

        scores = [s for s, _ in scored_events]
        lo, hi = ref["lo"], ref["hi"]
        midpoint = (lo + hi) / 2

        # 1) Distance to mask midpoint
        distances = [s - midpoint for s in scores]
        mean_dist = sum(distances) / len(distances)
        var_dist = sum((d - mean_dist) ** 2 for d in distances) / len(distances)
        std_dist = var_dist ** 0.5

        # 2) Near-miss rate: scores in (mask_hi, mask_hi + NEAR_MISS_BAND]
        near_miss_hi = hi + NEAR_MISS_BAND
        near_misses = sum(1 for s in scores if hi < s <= near_miss_hi)
        near_miss_rate = near_misses / len(scores) if scores else 0.0

        # 3) Spillover pressure: near-miss / mask interior
        mask_count = sum(1 for s in scores if lo <= s <= hi)
        spillover = near_misses / mask_count if mask_count > 0 else 0.0

        # 4) Daily spillover series → velocity & momentum
        daily_scores: Dict[str, List[float]] = defaultdict(list)
        for s, ts in scored_events:
            day = _ts_to_date(ts)
            daily_scores[day].append(s)

        daily_spillover: List[tuple] = []  # (date, spillover)
        for day in sorted(daily_scores.keys()):
            day_s = daily_scores[day]
            day_mask = sum(1 for s in day_s if lo <= s <= hi)
            day_nm = sum(1 for s in day_s if hi < s <= near_miss_hi)
            day_sp = day_nm / day_mask if day_mask > 0 else 0.0
            daily_spillover.append((day, round(day_sp, 3)))

        # Velocity: latest day vs 7d trailing average
        velocity = None
        momentum = None
        if len(daily_spillover) >= 2:
            latest_sp = daily_spillover[-1][1]
            trailing_7d = [sp for _, sp in daily_spillover[-8:-1]]  # up to 7 prior days
            trailing_3d = [sp for _, sp in daily_spillover[-4:-1]]  # up to 3 prior days
            if trailing_7d:
                avg_7d = sum(trailing_7d) / len(trailing_7d)
                velocity = round(latest_sp - avg_7d, 4)
            if trailing_3d:
                avg_3d = sum(trailing_3d) / len(trailing_3d)
                momentum = round(latest_sp - avg_3d, 4)

        # 5) Score density histogram
        histogram: Dict[str, int] = {}
        for label, bin_lo, bin_hi in SCORE_BINS:
            histogram[label] = sum(1 for s in scores if bin_lo <= s < bin_hi)
        # Last bin is inclusive on upper end
        histogram[SCORE_BINS[-1][0]] = sum(
            1 for s in scores if s >= SCORE_BINS[-1][1]
        )

        sufficient = len(scores) >= MIN_SCORES_SUFFICIENT

        analysis[sym] = {
            "total_scores": len(scores),
            "data_sufficiency": "OK" if sufficient else "INSUFFICIENT_DATA",
            "mask_midpoint": round(midpoint, 4),
            "distance_mean": round(mean_dist, 4),
            "distance_std": round(std_dist, 4),
            "bias_direction": "high" if mean_dist > 0.005 else (
                "low" if mean_dist < -0.005 else "centered"
            ),
            "near_miss_count": near_misses,
            "near_miss_rate": round(near_miss_rate, 4),
            "near_miss_band": f"({hi:.4f}, {near_miss_hi:.4f}]",
            "mask_interior_count": mask_count,
            "spillover_pressure": round(spillover, 3),
            "spillover_severity": (
                "normal" if spillover < 0.5
                else ("warning" if spillover < 0.7 else "critical")
            ),
            "spillover_velocity": velocity,
            "spillover_momentum": momentum,
            "daily_spillover": daily_spillover[-7:],  # last 7 days for display
            "histogram": histogram,
        }

    return {
        "title": "Score Drift Metrics",
        "by_symbol": analysis,
    }


def _section_drift_alerts(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Section 5: Drift alerts vs reference baseline."""
    alerts: List[Dict[str, Any]] = []

    # Check mask hit rate drift (reference: ~61% for BTC in-regime)
    d_events = [e for e in events if "d_rule" in e]
    btc_d = [e for e in d_events if e.get("symbol") == "BTCUSDT"]
    btc_hits = sum(1 for e in btc_d if e.get("d_rule") == "D_profit_region")
    btc_rate = btc_hits / len(btc_d) if btc_d else 0.0

    # Mask boundary drift: check if logged boundaries differ from reference
    boundary_drift = False
    for e in reversed(events):
        if e.get("symbol") == "BTCUSDT" and e.get("d_mask_boundaries"):
            logged_bounds = e["d_mask_boundaries"]
            if logged_bounds:
                ref = REFERENCE_MASK["BTCUSDT"]
                if len(logged_bounds) == 1:
                    lo_drift = abs(logged_bounds[0][0] - ref["lo"])
                    hi_drift = abs(logged_bounds[0][1] - ref["hi"])
                    if lo_drift > KILL_REGION_DRIFT_MAX or hi_drift > KILL_REGION_DRIFT_MAX:
                        boundary_drift = True
                        alerts.append({
                            "type": "REGION_DRIFT",
                            "symbol": "BTCUSDT",
                            "severity": "KILL",
                            "detail": f"lo_drift={lo_drift:.4f} hi_drift={hi_drift:.4f} "
                                      f"threshold={KILL_REGION_DRIFT_MAX}",
                        })
            break

    # Check 14-day region count
    fourteen_days_ago = time.time() - 14 * 86400
    recent_btc_hits = sum(
        1 for e in btc_d
        if e.get("d_rule") == "D_profit_region" and e.get("ts", 0) >= fourteen_days_ago
    )
    if btc_d and recent_btc_hits < KILL_REGION_COUNT_14D_MIN:
        alerts.append({
            "type": "LOW_REGION_COUNT",
            "symbol": "BTCUSDT",
            "severity": "KILL",
            "detail": f"14d_hits={recent_btc_hits} threshold={KILL_REGION_COUNT_14D_MIN}",
        })

    # Score distribution shift
    recent = [e for e in events if e.get("ts", 0) >= fourteen_days_ago]
    old = [e for e in events if e.get("ts", 0) < fourteen_days_ago]
    if recent and old:
        recent_btc_scores = [e["hydra_score"] for e in recent
                             if e.get("symbol") == "BTCUSDT" and e.get("hydra_score")]
        old_btc_scores = [e["hydra_score"] for e in old
                          if e.get("symbol") == "BTCUSDT" and e.get("hydra_score")]
        if recent_btc_scores and old_btc_scores:
            recent_mean = sum(recent_btc_scores) / len(recent_btc_scores)
            old_mean = sum(old_btc_scores) / len(old_btc_scores)
            shift = abs(recent_mean - old_mean)
            if shift > 0.03:
                alerts.append({
                    "type": "SCORE_DISTRIBUTION_SHIFT",
                    "symbol": "BTCUSDT",
                    "severity": "WARNING",
                    "detail": f"mean_shift={shift:.4f} old_mean={old_mean:.4f} "
                              f"recent_mean={recent_mean:.4f}",
                })

    return {
        "title": "Drift Alerts",
        "alerts": alerts,
        "btc_mask_hit_rate": round(btc_rate, 4),
        "btc_total_d_events": len(btc_d),
        "btc_14d_region_hits": recent_btc_hits,
        "boundary_drift_detected": boundary_drift,
        "kill_conditions_active": any(a["severity"] == "KILL" for a in alerts),
    }


def _section_promotion_readiness(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Section 6: Promotion readiness snapshot."""
    now = time.time()

    # Soak start: first v3 event or configured date
    v3_events = [e for e in events if e.get("schema") == "selector_v2_shadow_v3"]
    if v3_events:
        soak_start_ts = min(e["ts"] for e in v3_events)
    else:
        soak_start_ts = datetime(2026, 4, 7, tzinfo=timezone.utc).timestamp()

    soak_days = (now - soak_start_ts) / 86400

    # D_profit_region count (post-V2 only — v3 events)
    v3_profit = sum(1 for e in v3_events if e.get("d_rule") == "D_profit_region")

    # Daily throughput check
    d_events_with_rule = [e for e in events if "d_rule" in e]
    days_with_events = len(set(_ts_to_date(e["ts"]) for e in d_events_with_rule))
    profit_days = len(set(
        _ts_to_date(e["ts"]) for e in d_events_with_rule
        if e.get("d_rule") == "D_profit_region"
    ))

    checks = {
        "soak_days": {
            "value": round(soak_days, 1),
            "threshold": PROMOTE_SOAK_DAYS,
            "pass": soak_days >= PROMOTE_SOAK_DAYS,
        },
        "v3_episode_count": {
            "value": len(v3_events),
            "threshold": PROMOTE_EPISODE_COUNT,
            "pass": len(v3_events) >= PROMOTE_EPISODE_COUNT,
        },
        "v3_profit_region_count": {
            "value": v3_profit,
            "note": "Target: meaningful D_profit_region events",
        },
        "schema_consistency": {
            "v1_count": sum(1 for e in events if e.get("schema") == "selector_v2_shadow_v1"),
            "v2_count": sum(1 for e in events if e.get("schema") == "selector_v2_shadow_v2"),
            "v3_count": len(v3_events),
            "note": "Only v3 events count toward promotion",
        },
    }

    all_pass = all(
        c.get("pass", False) for c in checks.values() if "pass" in c
    )

    return {
        "title": "Promotion Readiness",
        "promote_ready": all_pass,
        "soak_start": _ts_to_iso(soak_start_ts),
        "checks": checks,
        "days_with_d_events": days_with_events,
        "days_with_profit_hits": profit_days,
    }


def _section_decision_distribution(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Section 7: Shadow decision distribution (D verdicts)."""
    d_events = [e for e in events if "d_rule" in e]
    if not d_events:
        return {"title": "Decision Distribution", "status": "no_d_events"}

    rules = Counter(e.get("d_rule", "unknown") for e in d_events)
    total = len(d_events)
    distribution = {
        rule: {"count": cnt, "pct": round(cnt / total * 100, 1)}
        for rule, cnt in sorted(rules.items(), key=lambda x: -x[1])
    }

    # Per-symbol per-rule
    sym_rules: Dict[str, Dict[str, int]] = defaultdict(Counter)
    for e in d_events:
        sym_rules[e.get("symbol", "?")][e.get("d_rule", "?")] += 1

    return {
        "title": "Decision Distribution",
        "total": total,
        "distribution": distribution,
        "by_symbol": {sym: dict(rules) for sym, rules in sorted(sym_rules.items())},
    }


# ── Output formatting ──────────────────────────────────────────────────────

def _print_report(sections: List[Dict[str, Any]]) -> None:
    """Human-readable text report to stdout."""
    bar = "=" * 72
    now_iso = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"\n{bar}")
    print(f"  SHADOW VALIDATION REPORT — Candidate D")
    print(f"  Generated: {now_iso}")
    print(bar)

    for s in sections:
        print(f"\n{'─' * 72}")
        title = s.get("title", "Unknown Section")
        print(f"  {title}")
        print(f"{'─' * 72}")

        if title == "Daily Candidate D Throughput":
            _print_throughput(s)
        elif title == "ZERO_SCORE vs NO_SCORE Diagnostics":
            _print_zero_score(s)
        elif title == "Mask Hit Rate & Distribution":
            _print_mask_hits(s)
        elif title == "Score vs Mask Boundary Analysis":
            _print_score_boundary(s)
        elif title == "Score Drift Metrics":
            _print_score_drift(s)
        elif title == "Drift Alerts":
            _print_drift(s)
        elif title == "Promotion Readiness":
            _print_promotion(s)
        elif title == "Decision Distribution":
            _print_decisions(s)

    print(f"\n{bar}")
    print(f"  END OF REPORT")
    print(bar)


def _print_throughput(s: Dict[str, Any]) -> None:
    rows = s.get("rows", [])
    if not rows:
        print("\n  No events found.")
        return
    print(f"\n  Total events: {s['total_events']} over {s['total_days']} days\n")
    print(f"  {'Date':12s}  {'Total':>6s}  {'Profit':>7s}  {'Abstain':>7s}  "
          f"{'Zero':>5s}  {'NoScr':>5s}  Symbols")
    print(f"  {'─' * 65}")
    for r in rows[-14:]:  # last 14 days
        syms = " ".join(f"{k}:{v}" for k, v in sorted(r["symbols"].items()))
        print(f"  {r['date']:12s}  {r['total']:6d}  {r['D_profit_region']:7d}  "
              f"{r['D_abstain']:7d}  {r['D_zero_score']:5d}  "
              f"{r['D_no_score']:5d}  {syms}")


def _print_zero_score(s: Dict[str, Any]) -> None:
    print(f"\n  Total events:    {s['total_events']}")
    print(f"  D_zero_score:    {s['D_zero_score']} ({s['D_zero_score_rate']:.1%})")
    print(f"  D_no_score:      {s['D_no_score']} ({s['D_no_score_rate']:.1%})")

    health = s.get("health", {})
    pipe = "OK" if health.get("pipeline_ok") else "DEGRADED"
    model = "OK" if health.get("model_informative") else "DEGRADED"
    print(f"\n  Pipeline health:  {pipe}  (no_score < 5%)")
    print(f"  Model signal:     {model}  (zero_score < 10%)")

    if s.get("by_symbol_zero"):
        print(f"\n  Zero-score by symbol: {s['by_symbol_zero']}")
    if s.get("by_symbol_no"):
        print(f"  No-score by symbol:   {s['by_symbol_no']}")


def _print_mask_hits(s: Dict[str, Any]) -> None:
    if s.get("status") == "no_d_events":
        print("\n  No Candidate D events available.")
        return
    print(f"\n  Total D events:      {s['total_d_events']}")
    print(f"  Profit region hits:  {s['profit_region_hits']}")
    print(f"  Overall hit rate:    {s['overall_hit_rate']:.1%}")

    by_sym = s.get("by_symbol", {})
    if by_sym:
        print(f"\n  {'Symbol':12s}  {'Hits':>5s}  {'Total':>6s}  {'Rate':>6s}")
        print(f"  {'─' * 35}")
        for sym, data in sorted(by_sym.items()):
            print(f"  {sym:12s}  {data['hits']:5d}  {data['total']:6d}  "
                  f"{data['rate']:.1%}")


def _print_score_boundary(s: Dict[str, Any]) -> None:
    by_sym = s.get("by_symbol", {})
    for sym, data in sorted(by_sym.items()):
        if data.get("status") == "no_data":
            print(f"\n  {sym}: no score data")
            continue
        ref = REFERENCE_MASK[sym]
        print(f"\n  {sym}  (mask: {ref['lo']:.4f} – {ref['hi']:.4f})")
        print(f"    Total scores:    {data['total_scores']}")
        print(f"    Inside mask:     {data['inside_mask']} ({data['inside_rate']:.1%})")
        print(f"    Below mask:      {data['below_mask']}")
        print(f"    Above mask:      {data['above_mask']}")
        print(f"    Near lower:      {data['near_lower_bound']}  (±0.02)")
        print(f"    Near upper:      {data['near_upper_bound']}  (±0.02)")
        print(f"    Mean score:      {data['mean_score']:.4f}  "
              f"(mask mid: {data['mask_midpoint']:.4f})  → {data['score_vs_mask']}")


def _print_score_drift(s: Dict[str, Any]) -> None:
    by_sym = s.get("by_symbol", {})
    for sym, data in sorted(by_sym.items()):
        if data.get("status") == "no_data":
            print(f"\n  {sym}: no score data")
            continue
        suf = data.get("data_sufficiency", "OK")
        suf_tag = f"  [{suf}]" if suf != "OK" else ""
        print(f"\n  {sym}  (midpoint: {data['mask_midpoint']:.4f}, n={data['total_scores']}){suf_tag}")

        # Distance to midpoint
        direction = data["bias_direction"]
        arrow = "→ HIGH" if direction == "high" else ("→ LOW" if direction == "low" else "→ centered")
        print(f"    Distance to midpoint:  mean={data['distance_mean']:+.4f}  "
              f"std={data['distance_std']:.4f}  {arrow}")

        # Near-miss
        print(f"    Near-miss rate:        {data['near_miss_count']}/{data['total_scores']}  "
              f"({data['near_miss_rate']:.1%})  band={data['near_miss_band']}")
        if data["near_miss_rate"] > 0.15:
            print(f"    !! High near-miss rate suggests mask upper bound too low")

        # Spillover pressure
        severity = data.get('spillover_severity', 'unknown')
        severity_marker = {'normal': '', 'warning': '  !! WARNING', 'critical': '  !!! CRITICAL'}  
        sp_suffix = f"  ({suf})" if suf != "OK" else ""
        print(f"    Spillover pressure:    {data['spillover_pressure']:.3f}  "
              f"({data['near_miss_count']}/{data['mask_interior_count']})"
              f"{severity_marker.get(severity, '')}{sp_suffix}")
        if severity != 'normal':
            print(f"    Severity: {severity.upper()} — "
                  f"{'mask losing containment' if severity == 'critical' else 'drift accelerating'}")

        # Velocity and momentum
        vel = data.get("spillover_velocity")
        mom = data.get("spillover_momentum")
        if vel is not None or mom is not None:
            vel_str = f"{vel:+.4f}" if vel is not None else "n/a"
            mom_str = f"{mom:+.4f}" if mom is not None else "n/a"
            vel_arrow = ""
            if vel is not None:
                if vel > 0.02:
                    vel_arrow = " ▲ accelerating"
                elif vel < -0.02:
                    vel_arrow = " ▼ decelerating"
                else:
                    vel_arrow = " ─ stable"
            print(f"    Spillover velocity:    {vel_str} (vs 7d avg){vel_arrow}")
            print(f"    3-day momentum:        {mom_str} (vs 3d avg)")

        # Daily spillover series
        daily_sp = data.get("daily_spillover", [])
        if daily_sp:
            print(f"    Daily spillover (last {len(daily_sp)}d):")
            for day, sp in daily_sp:
                sp_bar = int(sp * 20)
                print(f"      {day}  {sp:.3f}  {'▓' * sp_bar}")

        # Histogram
        hist = data.get("histogram", {})
        total = data["total_scores"]
        print(f"    Score density:")
        for label, count in hist.items():
            pct = count / total * 100 if total else 0
            bar_len = max(0, int(pct / 2))
            mask_marker = " ◄ mask" if "0.45" in label and "0.49" in label else ""
            print(f"      {label:14s}  {count:4d}  ({pct:5.1f}%)  "
                  f"{'█' * bar_len}{mask_marker}")


def _print_drift(s: Dict[str, Any]) -> None:
    alerts = s.get("alerts", [])
    print(f"\n  BTC mask hit rate:     {s['btc_mask_hit_rate']:.1%}")
    print(f"  BTC total D events:    {s['btc_total_d_events']}")
    print(f"  BTC 14d region hits:   {s['btc_14d_region_hits']}")
    print(f"  Boundary drift:        {'YES' if s['boundary_drift_detected'] else 'no'}")
    print(f"  Kill conditions:       {'ACTIVE' if s['kill_conditions_active'] else 'none'}")

    if alerts:
        print(f"\n  ALERTS ({len(alerts)}):")
        for a in alerts:
            marker = "!!!" if a["severity"] == "KILL" else "  !"
            print(f"  {marker} [{a['type']}] {a['symbol']}: {a['detail']}")
    else:
        print(f"\n  No drift alerts.")


def _print_promotion(s: Dict[str, Any]) -> None:
    ready = s.get("promote_ready", False)
    status = "READY" if ready else "NOT READY"
    print(f"\n  Status: {status}")
    print(f"  Soak start: {s['soak_start']}")

    checks = s.get("checks", {})
    for name, chk in checks.items():
        if "pass" in chk:
            mark = "[PASS]" if chk["pass"] else "[FAIL]"
            print(f"    {mark} {name}: {chk['value']} / {chk['threshold']}")
        elif "note" in chk:
            val = chk.get("value", "")
            if val != "":
                print(f"    [INFO] {name}: {val}  ({chk['note']})")
            else:
                detail = "  ".join(f"{k}={v}" for k, v in chk.items() if k != "note")
                print(f"    [INFO] {name}: {detail}  ({chk['note']})")


def _print_decisions(s: Dict[str, Any]) -> None:
    if s.get("status") == "no_d_events":
        print("\n  No Candidate D events.")
        return
    print(f"\n  Total D decisions: {s['total']}\n")
    dist = s.get("distribution", {})
    for rule, data in dist.items():
        bar_len = max(1, int(data["pct"] / 2))
        print(f"  {rule:20s}  {data['count']:5d}  ({data['pct']:5.1f}%)  "
              f"{'█' * bar_len}")

    by_sym = s.get("by_symbol", {})
    if by_sym:
        print(f"\n  Per-symbol:")
        for sym, rules in by_sym.items():
            parts = "  ".join(f"{r}={c}" for r, c in sorted(rules.items()))
            print(f"    {sym:12s}  {parts}")


# ── Main ────────────────────────────────────────────────────────────────────

def generate_report(
    *,
    days: Optional[int] = None,
    json_output: bool = False,
) -> Dict[str, Any]:
    """Generate the full shadow validation report.

    Returns the report as a dict. Optionally prints text or JSON.
    """
    since_ts = None
    if days:
        since_ts = time.time() - days * 86400

    events = _load_events(since_ts=since_ts)

    sections = [
        _section_daily_throughput(events),
        _section_zero_score(events),
        _section_mask_hits(events),
        _section_score_boundary(events),
        _section_score_drift_metrics(events),
        _section_drift_alerts(events),
        _section_promotion_readiness(events),
        _section_decision_distribution(events),
    ]

    report = {
        "generated_at": _ts_to_iso(time.time()),
        "event_count": len(events),
        "filter_days": days,
        "sections": {s["title"]: s for s in sections},
    }

    if json_output:
        json.dump(report, sys.stdout, indent=2, default=str)
        print()
    else:
        _print_report(sections)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Shadow Validation Report — Candidate D")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    parser.add_argument("--days", type=int, default=None, help="Only last N days")
    args = parser.parse_args()
    generate_report(days=args.days, json_output=args.json)


if __name__ == "__main__":
    main()
