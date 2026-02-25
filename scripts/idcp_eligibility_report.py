#!/usr/bin/env python3
"""
IDCP Eligibility 5-Metric Report Generator.

Polls binary_rounds.jsonl for rounds with full eligibility instrumentation
(eligible_strict_count present in intraround_stats). Once target_rounds
are available, produces the formal 5-metric report.

Usage:
    python3 scripts/idcp_eligibility_report.py [--target 5] [--poll-interval 300] [--timeout 4500]

Output:
    logs/prediction/idcp_eligibility_report.json  (machine-readable)
    stdout: human-readable summary
"""

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROUNDS_FILE = "logs/prediction/binary_rounds.jsonl"
REPORT_FILE = "logs/prediction/idcp_eligibility_report.json"


def load_eligibility_rounds(path: str) -> list[dict]:
    """Load all rounds that have full eligibility instrumentation."""
    rounds = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                stats = r.get("intraround_stats", {})
                if "eligible_strict_count" in stats:
                    rounds.append(r)
            except Exception:
                continue
    return rounds


def load_skew_only_rounds(path: str) -> list[dict]:
    """Load rounds that have skew data but may lack full eligibility."""
    rounds = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                stats = r.get("intraround_stats", {})
                if "skew_ms_mean" in stats:
                    rounds.append(r)
            except Exception:
                continue
    return rounds


def compute_report(elig_rounds: list[dict], skew_rounds: list[dict]) -> dict:
    """Compute the formal 5-metric report."""

    # ---- Metric 1: eligible_dislocation_samples (strict) ----
    total_strict = sum(
        r["intraround_stats"].get("eligible_strict_count", 0) for r in elig_rounds
    )

    # ---- Metric 2: eligible_dislocation_rounds (strict) ----
    rounds_with_strict = sum(
        1 for r in elig_rounds
        if r["intraround_stats"].get("eligible_strict_count", 0) > 0
    )

    # ---- Metric 3: max eligible window duration (s) ----
    max_eligible_window_s = 0.0
    for r in elig_rounds:
        stats = r["intraround_stats"]
        w = stats.get("eligible_strict_max_window_s", 0.0)
        if w and w > max_eligible_window_s:
            max_eligible_window_s = w

    # ---- Metric 4: skew_ms distribution during eligible dislocations ----
    # Collect skew_ms from all samples that were fee-adjusted dislocations
    eligible_skews = []
    eligible_depths_up = []
    eligible_depths_down = []

    for r in elig_rounds:
        for s in r.get("intraround_samples", []):
            fab = s.get("fee_adjusted_bundle")
            skew = s.get("skew_ms")
            stale_up = s.get("staleness_up_ms")
            stale_down = s.get("staleness_down_ms")

            if fab is not None and fab < 1.0 and skew is not None:
                # Check strict eligibility
                if (skew <= 1000
                    and stale_up is not None and stale_up <= 2000
                    and stale_down is not None and stale_down <= 2000):
                    eligible_skews.append(skew)
                    if s.get("depth_up") is not None:
                        eligible_depths_up.append(s["depth_up"])
                    if s.get("depth_down") is not None:
                        eligible_depths_down.append(s["depth_down"])

    skew_stats = {
        "count": len(eligible_skews),
        "mean_ms": round(statistics.mean(eligible_skews), 1) if eligible_skews else None,
        "p95_ms": round(sorted(eligible_skews)[int(len(eligible_skews) * 0.95)] if eligible_skews else 0, 1) or None,
        "max_ms": max(eligible_skews) if eligible_skews else None,
    }

    # ---- Metric 5: median depth during eligible dislocations ----
    depth_stats = {
        "median_depth_up": round(statistics.median(eligible_depths_up), 2) if eligible_depths_up else None,
        "median_depth_down": round(statistics.median(eligible_depths_down), 2) if eligible_depths_down else None,
    }

    # ---- Extended context from skew-only rounds (supplement) ----
    # All fee-adjusted dislocation samples with skew data (broader dataset)
    all_fab_dislocations_with_skew = []
    for r in skew_rounds:
        for s in r.get("intraround_samples", []):
            fab = s.get("fee_adjusted_bundle")
            skew = s.get("skew_ms")
            if fab is not None and fab < 1.0 and skew is not None:
                all_fab_dislocations_with_skew.append({
                    "skew_ms": skew,
                    "fab": fab,
                    "staleness_up_ms": s.get("staleness_up_ms"),
                    "staleness_down_ms": s.get("staleness_down_ms"),
                })

    skew_all = [d["skew_ms"] for d in all_fab_dislocations_with_skew]

    supplementary = {
        "total_rounds_with_skew": len(skew_rounds),
        "total_rounds_with_eligibility": len(elig_rounds),
        "total_fab_dislocation_samples_with_skew": len(all_fab_dislocations_with_skew),
        "fab_disloc_skew_le_1s": sum(1 for s in skew_all if s <= 1000),
        "fab_disloc_skew_le_2s": sum(1 for s in skew_all if s <= 2000),
        "fab_disloc_skew_le_5s": sum(1 for s in skew_all if s <= 5000),
        "fab_disloc_skew_mean_ms": round(statistics.mean(skew_all), 1) if skew_all else None,
        "fab_disloc_skew_median_ms": round(statistics.median(skew_all), 1) if skew_all else None,
        "fab_disloc_skew_max_ms": max(skew_all) if skew_all else None,
    }

    # Also: lenient eligible counts
    total_lenient = sum(
        r["intraround_stats"].get("eligible_lenient_count", 0) for r in elig_rounds
    )
    rounds_with_lenient = sum(
        1 for r in elig_rounds
        if r["intraround_stats"].get("eligible_lenient_count", 0) > 0
    )

    report = {
        "report_generated_at": datetime.now(timezone.utc).isoformat(),
        "instrumented_rounds": len(elig_rounds),
        "metrics": {
            "1_eligible_dislocation_samples_strict": total_strict,
            "2_eligible_dislocation_rounds_strict": rounds_with_strict,
            "3_max_eligible_window_duration_s": max_eligible_window_s,
            "4_skew_during_eligible_dislocations": skew_stats,
            "5_depth_during_eligible_dislocations": depth_stats,
        },
        "lenient_supplement": {
            "eligible_lenient_samples": total_lenient,
            "eligible_lenient_rounds": rounds_with_lenient,
        },
        "supplementary_skew_context": supplementary,
        "verdict": None,  # filled below
    }

    # Auto-verdict
    if total_strict == 0 and total_lenient == 0 and len(elig_rounds) >= 5:
        report["verdict"] = (
            "THESIS DEAD — Zero eligible dislocations across all instrumented rounds. "
            "All observed sub-1.0 bundles were artifacts of asynchronous leg reads "
            f"(median skew {supplementary.get('fab_disloc_skew_median_ms', '?')}ms). "
            "Binary deterministic sleeve has no exploitable edge."
        )
    elif total_strict == 0 and len(elig_rounds) >= 5:
        report["verdict"] = (
            f"THESIS DEAD (strict) — Zero strict-eligible dislocations. "
            f"{total_lenient} lenient-eligible found but below actionable threshold."
        )
    elif total_strict > 0:
        report["verdict"] = (
            f"INCONCLUSIVE — {total_strict} strict-eligible samples found. "
            "Requires deeper analysis of depth, frequency, and feasibility."
        )
    else:
        report["verdict"] = (
            f"INSUFFICIENT DATA — Only {len(elig_rounds)} instrumented rounds. "
            "Need at least 5 for a verdict."
        )

    return report


def print_report(report: dict):
    """Print human-readable report."""
    m = report["metrics"]
    s = report["supplementary_skew_context"]
    l = report["lenient_supplement"]

    print("=" * 70)
    print("  IDCP ELIGIBILITY 5-METRIC REPORT")
    print(f"  Generated: {report['report_generated_at']}")
    print(f"  Instrumented rounds: {report['instrumented_rounds']}")
    print("=" * 70)
    print()
    print("  FORMAL METRICS (strict: skew ≤ 1s, staleness ≤ 2s)")
    print("  " + "-" * 50)
    print(f"  1. Eligible dislocation samples:  {m['1_eligible_dislocation_samples_strict']}")
    print(f"  2. Eligible dislocation rounds:   {m['2_eligible_dislocation_rounds_strict']}")
    print(f"  3. Max eligible window (s):       {m['3_max_eligible_window_duration_s']}")
    print(f"  4. Skew during eligible:")
    sk = m["4_skew_during_eligible_dislocations"]
    if sk["count"] == 0:
        print(f"     (no eligible samples — skew stats N/A)")
    else:
        print(f"     count={sk['count']}  mean={sk['mean_ms']}ms  p95={sk['p95_ms']}ms  max={sk['max_ms']}ms")
    print(f"  5. Depth during eligible:")
    dp = m["5_depth_during_eligible_dislocations"]
    if dp["median_depth_up"] is None:
        print(f"     (no eligible samples — depth stats N/A)")
    else:
        print(f"     median_up=${dp['median_depth_up']}  median_down=${dp['median_depth_down']}")
    print()
    print("  LENIENT THRESHOLD (skew ≤ 2s, staleness ≤ 5s)")
    print("  " + "-" * 50)
    print(f"  Lenient eligible samples:  {l['eligible_lenient_samples']}")
    print(f"  Lenient eligible rounds:   {l['eligible_lenient_rounds']}")
    print()
    print("  SUPPLEMENTARY SKEW CONTEXT (all rounds with skew data)")
    print("  " + "-" * 50)
    print(f"  Rounds with skew data:              {s['total_rounds_with_skew']}")
    print(f"  Fee-adj dislocation samples w/skew:  {s['total_fab_dislocation_samples_with_skew']}")
    print(f"  Of those, skew ≤ 1s:                 {s['fab_disloc_skew_le_1s']}")
    print(f"  Of those, skew ≤ 2s:                 {s['fab_disloc_skew_le_2s']}")
    print(f"  Of those, skew ≤ 5s:                 {s['fab_disloc_skew_le_5s']}")
    print(f"  Skew mean:   {s['fab_disloc_skew_mean_ms']}ms")
    print(f"  Skew median: {s['fab_disloc_skew_median_ms']}ms")
    print(f"  Skew max:    {s['fab_disloc_skew_max_ms']}ms")
    print()
    print("  VERDICT")
    print("  " + "-" * 50)
    print(f"  {report['verdict']}")
    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=5, help="Target eligibility rounds")
    parser.add_argument("--poll-interval", type=int, default=120, help="Poll interval seconds")
    parser.add_argument("--timeout", type=int, default=4500, help="Max wait seconds")
    parser.add_argument("--no-wait", action="store_true", help="Generate report immediately")
    args = parser.parse_args()

    os.chdir(Path(__file__).resolve().parent.parent)

    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] IDCP Eligibility Report Generator")
    print(f"  Target rounds: {args.target}")
    print(f"  Poll interval: {args.poll_interval}s")
    print(f"  Timeout: {args.timeout}s")
    print()

    start = time.time()
    while True:
        elig_rounds = load_eligibility_rounds(ROUNDS_FILE)
        skew_rounds = load_skew_only_rounds(ROUNDS_FILE)
        elapsed = time.time() - start

        print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] "
              f"Eligibility rounds: {len(elig_rounds)}/{args.target}  "
              f"Skew rounds: {len(skew_rounds)}  "
              f"Elapsed: {int(elapsed)}s")

        if len(elig_rounds) >= args.target or args.no_wait:
            break

        if elapsed > args.timeout:
            print(f"  Timeout reached ({args.timeout}s). Generating report with {len(elig_rounds)} rounds.")
            break

        time.sleep(args.poll_interval)

    # Generate report
    report = compute_report(elig_rounds, skew_rounds)
    print()
    print_report(report)

    # Save
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
