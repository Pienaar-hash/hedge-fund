#!/usr/bin/env python3
"""
Binary Lab S2 — post-hoc evaluation script.

Reads ``logs/execution/binary_lab_s2_trades.jsonl`` and prints:

1. **Edge bucket table** — pure numeric labels (Amendment 8)
2. **Model-lift table** — model Brier vs baseline Brier (Amendment 1)
3. **Probability calibration table** — predicted vs actual buckets
4. **Rolling Brier** — 20-round moving average
5. **Friction-kill stats** — SKIP_FRICTION_ERASED_EDGE count (Amendment 6)
6. **Summary**

⚠  Economics are reconstructed from mid + spread/2.  Until direct bid/ask
   snapshots are available, all edge / PnL figures are provisional.
   See ``quote_reconstruction_mode`` field on each record.

Usage::

    PYTHONPATH=. python scripts/binary_lab_s2_eval.py
    PYTHONPATH=. python scripts/binary_lab_s2_eval.py --path logs/execution/binary_lab_s2_trades.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_PATH = Path("logs/execution/binary_lab_s2_trades.jsonl")


def _load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        print(f"[error] trade log not found: {path}", file=sys.stderr)
        sys.exit(1)
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _pct(num: float, den: float) -> str:
    if den == 0:
        return "  n/a"
    return f"{100.0 * num / den:6.1f}%"


def _fmt(val: float, decimals: int = 4) -> str:
    return f"{val:.{decimals}f}"


# -----------------------------------------------------------------------
# Edge bucket table (Amendment 8)
# -----------------------------------------------------------------------

def _edge_bucket_table(resolved: List[Dict[str, Any]]) -> None:
    buckets: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"count": 0, "wins": 0, "pnl": 0.0, "brier_sum": 0.0},
    )
    for r in resolved:
        b = r.get("edge_bucket", "unknown")
        buckets[b]["count"] += 1
        if r.get("resolved_outcome") == "WIN":
            buckets[b]["wins"] += 1
        buckets[b]["pnl"] += r.get("pnl_usd", 0.0)
        buckets[b]["brier_sum"] += r.get("brier_component", 0.0)

    print("\n=== Edge Bucket Table ===")
    print(f"{'Bucket':<16} {'Count':>6} {'Win%':>7} {'PnL':>10} {'AvgBrier':>10}")
    print("-" * 53)
    for b in sorted(buckets.keys()):
        s = buckets[b]
        n = s["count"]
        avg_brier = s["brier_sum"] / n if n else 0.0
        print(
            f"{b:<16} {n:>6} {_pct(s['wins'], n):>7} "
            f"{s['pnl']:>10.2f} {avg_brier:>10.4f}"
        )


# -----------------------------------------------------------------------
# Model-lift table (Amendment 1)
# -----------------------------------------------------------------------

def _model_lift_table(resolved: List[Dict[str, Any]]) -> None:
    model_brier_sum = 0.0
    baseline_brier_sum = 0.0
    count = 0
    for r in resolved:
        mb = r.get("brier_component")
        bb = r.get("baseline_brier_component")
        if mb is not None and bb is not None:
            model_brier_sum += mb
            baseline_brier_sum += bb
            count += 1

    print("\n=== Model vs Baseline (Brier Score — lower is better) ===")
    if count == 0:
        print("  No resolved rounds with both Brier components.")
        return
    avg_model = model_brier_sum / count
    avg_baseline = baseline_brier_sum / count
    lift = avg_baseline - avg_model
    print(f"  Rounds:          {count}")
    print(f"  Model Brier:     {avg_model:.6f}")
    print(f"  Baseline Brier:  {avg_baseline:.6f}")
    print(f"  Lift (B-M):      {lift:+.6f}  {'(model better)' if lift > 0 else '(baseline better)' if lift < 0 else '(tie)'}")


# -----------------------------------------------------------------------
# Probability calibration table
# -----------------------------------------------------------------------

def _calibration_table(resolved: List[Dict[str, Any]]) -> None:
    bins: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"count": 0, "outcomes": 0.0, "pred_sum": 0.0},
    )
    for r in resolved:
        p = r.get("p_model_yes")
        outcome_yes = r.get("outcome_yes")
        if p is None or outcome_yes is None:
            continue
        bucket = f"{int(p * 10) * 10:02d}-{int(p * 10) * 10 + 10:02d}%"
        bins[bucket]["count"] += 1
        bins[bucket]["outcomes"] += 1.0 if outcome_yes else 0.0
        bins[bucket]["pred_sum"] += p

    print("\n=== Probability Calibration Table ===")
    print(f"{'Bucket':<10} {'Count':>6} {'Predicted':>10} {'Actual':>10} {'Gap':>8}")
    print("-" * 48)
    for b in sorted(bins.keys()):
        s = bins[b]
        n = s["count"]
        if n == 0:
            continue
        pred = s["pred_sum"] / n
        actual = s["outcomes"] / n
        gap = pred - actual
        print(f"{b:<10} {n:>6} {pred:>10.3f} {actual:>10.3f} {gap:>+8.3f}")


# -----------------------------------------------------------------------
# Rolling Brier (20-round window)
# -----------------------------------------------------------------------

def _rolling_brier(resolved: List[Dict[str, Any]], window: int = 20) -> None:
    briers = [r.get("brier_component", 0.0) for r in resolved]
    if len(briers) < window:
        print(f"\n=== Rolling Brier ({window}-round) ===")
        print(f"  Not enough data ({len(briers)} < {window}).")
        return

    print(f"\n=== Rolling Brier ({window}-round) ===")
    print(f"{'Window':<12} {'Avg Brier':>10}")
    print("-" * 24)
    for i in range(window, len(briers) + 1, max(1, len(briers) // 10)):
        chunk = briers[i - window:i]
        avg = sum(chunk) / len(chunk)
        print(f"  {i - window + 1:>4}-{i:<4}  {avg:>10.4f}")


# -----------------------------------------------------------------------
# Friction-kill stats (Amendment 6)
# -----------------------------------------------------------------------

def _friction_kill_stats(all_records: List[Dict[str, Any]]) -> None:
    friction_kills = [
        r for r in all_records
        if r.get("deny_reason") == "SKIP_FRICTION_ERASED_EDGE"
        or r.get("skip_reason") == "SKIP_FRICTION_ERASED_EDGE"
    ]
    invalid_reconstruction = [
        r for r in all_records
        if r.get("deny_reason") == "SKIP_INVALID_QUOTE_RECONSTRUCTION"
        or r.get("skip_reason") == "SKIP_INVALID_QUOTE_RECONSTRUCTION"
    ]
    no_trades = [r for r in all_records if r.get("event_type") == "NO_TRADE"]

    print("\n=== Friction / Skip Stats ===")
    print(f"  Total NO_TRADE events:                 {len(no_trades)}")
    print(f"  SKIP_FRICTION_ERASED_EDGE:             {len(friction_kills)}")
    print(f"  SKIP_INVALID_QUOTE_RECONSTRUCTION:     {len(invalid_reconstruction)}")


# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------

def _summary(all_records: List[Dict[str, Any]], resolved: List[Dict[str, Any]]) -> None:
    entries = [r for r in all_records if r.get("event_type") == "ENTRY"]
    no_trades = [r for r in all_records if r.get("event_type") == "NO_TRADE"]
    total_pnl = sum(r.get("pnl_usd", 0.0) for r in resolved)
    total_fees = sum(r.get("fee_usd", 0.0) for r in resolved)
    wins = sum(1 for r in resolved if r.get("resolved_outcome") == "WIN")

    print("\n=== Summary ===")
    print(f"  Entries:     {len(entries)}")
    print(f"  Resolved:    {len(resolved)}")
    print(f"  No-trade:    {len(no_trades)}")
    print(f"  Win rate:    {_pct(wins, len(resolved))}")
    print(f"  Total PnL:   {total_pnl:+.4f} USDT")
    print(f"  Total fees:  {total_fees:.4f} USDT")
    print()
    print("  ⚠  Economics are reconstructed (mid + spread/2).")
    print("     All edge / PnL figures are provisional until direct")
    print("     bid/ask snapshots replace the reconstruction layer.")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Binary Lab S2 evaluation")
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_PATH,
        help="Path to binary_lab_s2_trades.jsonl",
    )
    args = parser.parse_args()

    all_records = _load_records(args.path)
    resolved = [r for r in all_records if r.get("event_type") == "ROUND_CLOSED"]

    print(f"Loaded {len(all_records)} records ({len(resolved)} resolved) from {args.path}")

    _summary(all_records, resolved)
    _edge_bucket_table(resolved)
    _model_lift_table(resolved)
    _calibration_table(resolved)
    _rolling_brier(resolved)
    _friction_kill_stats(all_records)


if __name__ == "__main__":
    main()
