#!/usr/bin/env python3
"""FPS RTB sanity check — verify RTB fires only during regime transitions.

Usage:
    PYTHONPATH=. python scripts/fps_rtb_sanity_check.py [--hours 24]

Reads:
    logs/execution/futures_permit_surface_shadow.jsonl
    logs/state/fps_rtb_reset_marker.json

Reports:
    - RTB detections per regime transition
    - RTB fires in steady-state (should be 0)
    - Per-class summary (DR/TCP/ERE/VEB continuity)
    - Fee-pass persistence for DR
    - Symbol concentration for DR
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SHADOW_LOG = Path("logs/execution/futures_permit_surface_shadow.jsonl")
RTB_MARKER = Path("logs/state/fps_rtb_reset_marker.json")


def load_rows(hours: float | None = None) -> list[dict]:
    if not SHADOW_LOG.exists():
        print("ERROR: shadow log not found"); sys.exit(1)
    marker_ts = 0.0
    if RTB_MARKER.exists():
        marker = json.loads(RTB_MARKER.read_text())
        marker_ts = marker.get("rtb_reset_ts", 0.0)

    import time
    cutoff = time.time() - (hours * 3600) if hours else 0.0
    effective_cutoff = max(cutoff, marker_ts)  # for RTB, always use marker

    rows = []
    for line in SHADOW_LOG.open():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append(r)
    return rows, marker_ts, effective_cutoff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=None,
                        help="Limit analysis to last N hours (default: all post-fix)")
    args = parser.parse_args()

    rows, marker_ts, effective_cutoff = load_rows(args.hours)
    if not rows:
        print("No shadow data found."); return

    # Partition: post-fix vs pre-fix
    post_fix = [r for r in rows if r["ts"] >= marker_ts] if marker_ts else rows
    pre_fix_count = len(rows) - len(post_fix)

    print(f"=== FPS RTB Sanity Check ===")
    print(f"Total shadow rows:  {len(rows)}")
    print(f"Pre-fix (invalid):  {pre_fix_count}")
    print(f"Post-fix (valid):   {len(post_fix)}")
    if marker_ts:
        dt = datetime.fromtimestamp(marker_ts, tz=timezone.utc)
        print(f"RTB reset at:       {dt.isoformat()}")
    print()

    if not post_fix:
        print("No post-fix data yet. Wait for shadow collection.")
        return

    # --- RTB Analysis ---
    rtb_rows = [r for r in post_fix if r.get("setup_class") == "REGIME_TRANSITION_BREAK"]
    print(f"--- RTB Post-Fix ---")
    print(f"RTB detections: {len(rtb_rows)}")

    if rtb_rows:
        # Check: does RTB fire only when regime_previous != regime_current?
        steady_state_fires = []
        transition_fires = []
        for r in rtb_rows:
            prev = r.get("regime_previous", "")
            cur = r.get("regime_current", "")
            age = r.get("regime_age_bars", "?")
            if prev == cur or prev == "":
                steady_state_fires.append(r)
            else:
                transition_fires.append(r)

        print(f"  During transitions: {len(transition_fires)}")
        print(f"  In steady-state:    {len(steady_state_fires)}  {'*** BUG STILL PRESENT ***' if steady_state_fires else '(CLEAN)'}")

        if transition_fires:
            # Group by transition
            transitions = Counter()
            for r in transition_fires:
                key = f"{r.get('regime_previous','?')} -> {r.get('regime_current','?')}"
                transitions[key] += 1
            print(f"  Transitions seen:")
            for k, v in transitions.most_common():
                print(f"    {k}: {v} detections")

        if steady_state_fires:
            print(f"\n  *** STEADY-STATE FIRES (should be 0) ***")
            for r in steady_state_fires[:5]:
                dt = datetime.fromtimestamp(r["ts"], tz=timezone.utc)
                print(f"    {dt.strftime('%m-%d %H:%M')} {r.get('symbol','')} "
                      f"prev={r.get('regime_previous','?')} cur={r.get('regime_current','?')} "
                      f"age={r.get('regime_age_bars','?')}")
    else:
        print("  (no RTB detections yet — expected if regime is stable)")
    print()

    # --- Other Classes (continuous, not reset) ---
    print(f"--- Other Classes (continuous from all valid data) ---")
    # For non-RTB, use all rows (their data is valid even pre-fix)
    non_rtb = [r for r in rows if r.get("setup_class") and
               r["setup_class"] != "REGIME_TRANSITION_BREAK"]

    class_stats = defaultdict(lambda: {"total": 0, "permit": 0, "deny": 0, "abstain": 0})
    for r in non_rtb:
        cls = r["setup_class"]
        class_stats[cls]["total"] += 1
        v = r.get("verdict", "")
        if "PERMIT" in v:
            class_stats[cls]["permit"] += 1
        elif "DENY" in v:
            class_stats[cls]["deny"] += 1
        else:
            class_stats[cls]["abstain"] += 1

    for cls in sorted(class_stats):
        s = class_stats[cls]
        rate = f"{s['permit']/s['total']*100:.1f}%" if s['total'] else "n/a"
        print(f"  {cls}: {s['total']} total | {s['permit']} PERMIT | {s['deny']} DENY | {s['abstain']} ABSTAIN | rate={rate}")
    print()

    # --- DR Fee-Pass Persistence ---
    dr_rows = [r for r in rows if r.get("setup_class") == "DISLOCATION_REVERSION"]
    if dr_rows:
        print(f"--- DR Fee-Pass Persistence ---")
        # Group by 12h bucket
        dr_by_bucket = defaultdict(lambda: {"permit": 0, "total": 0})
        for r in dr_rows:
            dt = datetime.fromtimestamp(r["ts"], tz=timezone.utc)
            bucket = dt.strftime('%m-%d') + f" {'AM' if dt.hour < 12 else 'PM'}"
            dr_by_bucket[bucket]["total"] += 1
            if "PERMIT" in r.get("verdict", ""):
                dr_by_bucket[bucket]["permit"] += 1

        for bucket in sorted(dr_by_bucket):
            s = dr_by_bucket[bucket]
            rate = f"{s['permit']/s['total']*100:.0f}%" if s['total'] else "n/a"
            print(f"  {bucket}: {s['permit']}/{s['total']} permits ({rate})")

        # Regime distribution of DR permits
        dr_permits = [r for r in dr_rows if "PERMIT" in r.get("verdict", "")]
        if dr_permits:
            regime_dist = Counter(r.get("regime_current", "?") for r in dr_permits)
            print(f"  Regime distribution of DR permits: {dict(regime_dist.most_common())}")
        print()

    # --- Symbol Concentration (DR) ---
    if dr_rows:
        print(f"--- DR Symbol Concentration ---")
        dr_permits = [r for r in dr_rows if "PERMIT" in r.get("verdict", "")]
        if dr_permits:
            sym_dist = Counter(r.get("symbol", "?") for r in dr_permits)
            total = len(dr_permits)
            for sym, cnt in sym_dist.most_common():
                pct = cnt / total * 100
                flag = " *** CONCENTRATED" if pct > 80 else ""
                print(f"  {sym}: {cnt} ({pct:.0f}%){flag}")
        else:
            print("  No DR permits yet")
        print()

    # --- Summary Verdict ---
    print("=" * 50)
    issues = []
    if rtb_rows and steady_state_fires:
        issues.append(f"RTB steady-state fires: {len(steady_state_fires)}")
    dr_permits_list = [r for r in dr_rows if "PERMIT" in r.get("verdict", "")] if dr_rows else []
    if dr_permits_list:
        top_sym = Counter(r.get("symbol") for r in dr_permits_list).most_common(1)
        if top_sym and top_sym[0][1] / len(dr_permits_list) > 0.8:
            issues.append(f"DR concentrated in {top_sym[0][0]} ({top_sym[0][1]}/{len(dr_permits_list)})")

    if issues:
        print("ISSUES:")
        for i in issues:
            print(f"  - {i}")
    else:
        print("ALL CLEAR — no anomalies detected")


if __name__ == "__main__":
    main()
