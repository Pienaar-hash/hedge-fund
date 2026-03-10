#!/usr/bin/env python3
"""
Edge Calibration Plot — Predicted Edge vs Realized Return.

Reads episode_ledger.json and produces:
  1. Scatter plot: predicted edge (x) vs realized return (y)
  2. Bucket calibration: mean realized return per predicted-edge bucket
  3. Ideal calibration line (y = x)

Usage:
    PYTHONPATH=. python research/edge_calibration_plot.py
    PYTHONPATH=. python research/edge_calibration_plot.py --save calibration.png
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, List

_LEDGER_PATH = Path("logs/state/episode_ledger.json")


def _safe_float(val: Any) -> float:
    if val is None:
        return 0.0
    try:
        v = float(val)
        return v if math.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _load_episodes() -> List[dict]:
    data = json.loads(_LEDGER_PATH.read_text())
    return data.get("episodes", [])


def _extract_calibration_data(episodes: List[dict]):
    """Extract (predicted_edge, realized_return) pairs."""
    pred, real = [], []
    for ep in episodes:
        edge = _safe_float(ep.get("expected_edge"))
        if not (edge > 0 and math.isfinite(edge)):
            conv = _safe_float(ep.get("conviction_score"))
            edge = max(0.0, conv - 0.5)
        if edge <= 0:
            continue

        entry_px = _safe_float(ep.get("avg_entry_price"))
        exit_px = _safe_float(ep.get("avg_exit_price"))
        if entry_px <= 0 or exit_px <= 0:
            continue

        side = str(ep.get("side", "")).upper()
        if side == "LONG":
            realized = (exit_px - entry_px) / entry_px
        elif side == "SHORT":
            realized = (entry_px - exit_px) / entry_px
        else:
            continue

        pred.append(edge)
        real.append(realized)
    return pred, real


def _bucket_calibration(pred, real, n_buckets=6):
    """Group by predicted-edge percentile buckets, return bucket midpoints and mean realized."""
    if not pred:
        return [], [], [], []

    pairs = sorted(zip(pred, real))
    bucket_size = max(1, len(pairs) // n_buckets)

    midpoints, means, labels, counts = [], [], [], []
    for i in range(0, len(pairs), bucket_size):
        chunk = pairs[i : i + bucket_size]
        p_vals = [c[0] for c in chunk]
        r_vals = [c[1] for c in chunk]
        mid = sum(p_vals) / len(p_vals)
        mean_r = sum(r_vals) / len(r_vals)
        lo = min(p_vals)
        hi = max(p_vals)
        midpoints.append(mid)
        means.append(mean_r)
        labels.append(f"{lo:.1%}-{hi:.1%}")
        counts.append(len(chunk))
    return midpoints, means, labels, counts


def main():
    parser = argparse.ArgumentParser(description="Edge Calibration Chart")
    parser.add_argument("--save", type=str, help="Save to file instead of showing")
    args = parser.parse_args()

    try:
        import matplotlib
        if args.save:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    episodes = _load_episodes()
    pred, real = _extract_calibration_data(episodes)

    if not pred:
        print("No episodes with usable edge data found.")
        sys.exit(0)

    err = sum(real) / sum(pred) if sum(pred) > 0 else None
    print(f"Episodes with edge data: {len(pred)}")
    print(f"ERR = {err:.4f}" if err is not None else "ERR = N/A")

    midpoints, means, labels, counts = _bucket_calibration(pred, real)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Scatter plot ────────────────────────────────────────────────
    ax1.scatter(pred, real, alpha=0.4, s=20, color="#4a90d9", edgecolors="none")
    max_val = max(max(pred), max(abs(r) for r in real)) * 1.1
    ax1.plot([0, max_val], [0, max_val], "k--", alpha=0.4, label="ideal (y=x)")
    ax1.axhline(0, color="#888", linewidth=0.5)
    ax1.set_xlabel("Predicted Edge")
    ax1.set_ylabel("Realized Return")
    ax1.set_title(f"Edge Calibration (n={len(pred)}, ERR={err:.2f})")
    ax1.legend()

    # ── Bucket calibration ──────────────────────────────────────────
    colors = ["#21ba45" if m > 0 else "#db2828" for m in means]
    bars = ax2.bar(range(len(midpoints)), means, color=colors, alpha=0.7, width=0.7)
    ax2.plot(range(len(midpoints)), midpoints, "ko--", markersize=4, alpha=0.5, label="predicted (ideal)")
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax2.set_xlabel("Predicted Edge Bucket")
    ax2.set_ylabel("Mean Realized Return")
    ax2.set_title("Bucket Calibration")
    ax2.axhline(0, color="#888", linewidth=0.5)
    ax2.legend()
    for i, (bar, c) in enumerate(zip(bars, counts)):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"n={c}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
