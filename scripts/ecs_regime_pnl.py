#!/usr/bin/env python3
"""ECS Regime PnL Attribution — expected PnL contribution per engine regime.

Uses detected regime boundaries from the phase map to partition the
episode ledger by hybrid_score, then computes PnL statistics per regime.

This answers: which regimes produce alpha and which destroy it?

Usage:
    PYTHONPATH=. python scripts/ecs_regime_pnl.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

EPISODE_LEDGER = Path("logs/state/episode_ledger.json")
SOAK_PATH = Path("logs/execution/ecs_soak_events.jsonl")

# Regime boundaries from phase map analysis (866 events, 2026-03-15)
# These are the hydra_score thresholds where engine dominance flips.
REGIME_BOUNDARIES: Dict[str, List[float]] = {
    "BTCUSDT": [0.5236],
    "ETHUSDT": [0.4291, 0.4883],
    "SOLUSDT": [],  # single-regime: Legacy dominant
}

# Engine labels for each regime segment (below first boundary, between, above last)
REGIME_LABELS: Dict[str, List[str]] = {
    "BTCUSDT": ["HYDRA_REGIME", "LEGACY_REGIME"],
    "ETHUSDT": ["LEGACY_LOW", "HYDRA_REGIME", "LEGACY_HIGH"],
    "SOLUSDT": ["LEGACY_ONLY"],
}


def _load_episodes() -> List[Dict[str, Any]]:
    if not EPISODE_LEDGER.exists():
        return []
    with open(EPISODE_LEDGER) as f:
        data = json.load(f)
    return data.get("episodes", data.get("entries", []))


def _load_soak_engine_map() -> Dict[Tuple[str, str], str]:
    """Build (symbol, intent_id) → ecs_winner from soak events."""
    mapping: Dict[Tuple[str, str], str] = {}
    if not SOAK_PATH.exists():
        return mapping
    with open(SOAK_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
                sym = e.get("symbol")
                iid = e.get("intent_id")
                winner = e.get("ecs_winner")
                if sym and winner:
                    if iid:
                        mapping[(sym, iid)] = winner
            except json.JSONDecodeError:
                pass
    return mapping


def _classify_regime(symbol: str, score: float) -> str:
    """Classify a hybrid_score into a regime label."""
    bounds = REGIME_BOUNDARIES.get(symbol, [])
    labels = REGIME_LABELS.get(symbol, ["UNKNOWN"])

    if not bounds:
        return labels[0] if labels else "UNKNOWN"

    for i, threshold in enumerate(bounds):
        if score < threshold:
            return labels[i] if i < len(labels) else "UNKNOWN"

    return labels[-1] if labels else "UNKNOWN"


def _pnl_stats(pnls: List[float]) -> Dict[str, Any]:
    """Compute PnL statistics."""
    if not pnls:
        return {"n": 0}
    n = len(pnls)
    total = sum(pnls)
    mean = total / n
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]
    win_rate = len(winners) / n * 100
    sorted_pnls = sorted(pnls)
    median = sorted_pnls[n // 2]

    # Expectancy = avg_win * win_rate - avg_loss * loss_rate
    avg_win = sum(winners) / len(winners) if winners else 0
    avg_loss = abs(sum(losers) / len(losers)) if losers else 0
    wr = len(winners) / n
    expectancy = avg_win * wr - avg_loss * (1 - wr)

    # Profit factor
    gross_win = sum(winners) if winners else 0
    gross_loss = abs(sum(losers)) if losers else 0.001
    profit_factor = gross_win / gross_loss

    return {
        "n": n,
        "total_pnl": total,
        "mean_pnl": mean,
        "median_pnl": median,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "best": max(pnls),
        "worst": min(pnls),
    }


def run() -> None:
    episodes = _load_episodes()
    soak_map = _load_soak_engine_map()

    if not episodes:
        print("No episodes found.")
        return

    # Filter to core symbols with regime analysis
    core_symbols = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
    core_episodes = [ep for ep in episodes if ep.get("symbol") in core_symbols]

    print("=" * 72)
    print("  ECS Regime PnL Attribution")
    print("=" * 72)
    print()
    print(f"  Total episodes: {len(episodes)}")
    print(f"  Core symbol episodes: {len(core_episodes)}")
    print(f"  Soak engine mappings: {len(soak_map)}")
    print()

    # ── Section 1: Per-symbol regime PnL ─────────────────────────────
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        sym_eps = [ep for ep in core_episodes if ep.get("symbol") == sym]
        if not sym_eps:
            continue

        # Filter episodes with valid scores (exclude score=0 which means missing)
        scored = [ep for ep in sym_eps if ep.get("hybrid_score", 0) > 0]
        zero_score = [ep for ep in sym_eps if ep.get("hybrid_score", 0) == 0]

        bounds = REGIME_BOUNDARIES.get(sym, [])
        labels = REGIME_LABELS.get(sym, [])
        bounds_str = ", ".join(f"{b:.4f}" for b in bounds) if bounds else "none"

        print("-" * 72)
        print(f"  {sym}  (n={len(sym_eps)}, scored={len(scored)}, zero_score={len(zero_score)})")
        print(f"  Boundaries: [{bounds_str}]")
        print(f"  Regimes: {' → '.join(labels)}")
        print()

        # Classify each scored episode
        regime_pnls: Dict[str, List[float]] = defaultdict(list)
        regime_scores: Dict[str, List[float]] = defaultdict(list)

        for ep in scored:
            score = ep.get("hybrid_score", 0)
            pnl = ep.get("net_pnl", 0)
            regime = _classify_regime(sym, score)
            regime_pnls[regime].append(pnl)
            regime_scores[regime].append(score)

        # Also track zero-score episodes
        if zero_score:
            for ep in zero_score:
                regime_pnls["ZERO_SCORE"].append(ep.get("net_pnl", 0))

        # Print per-regime stats
        all_labels = list(labels) + (["ZERO_SCORE"] if zero_score else [])
        print(f"    {'regime':<16} {'n':>5}  {'total_pnl':>10}  {'mean_pnl':>9}  {'win_rate':>8}  {'expect':>8}  {'PF':>6}  {'score_range'}")
        print(f"    {'-'*16} {'-'*5}  {'-'*10}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*20}")

        sym_total_pnl = sum(ep.get("net_pnl", 0) for ep in sym_eps)

        for label in all_labels:
            pnls = regime_pnls.get(label, [])
            scores = regime_scores.get(label, [])
            stats = _pnl_stats(pnls)
            n = stats["n"]
            if n == 0:
                continue

            score_str = ""
            if scores:
                score_str = f"[{min(scores):.4f}, {max(scores):.4f}]"

            print(f"    {label:<16} {n:>5}  {stats['total_pnl']:>+10.2f}  {stats['mean_pnl']:>+9.4f}"
                  f"  {stats['win_rate']:>7.1f}%  {stats['expectancy']:>+8.4f}"
                  f"  {stats['profit_factor']:>6.2f}  {score_str}")

        print()

        # Contribution breakdown
        print(f"    PnL contribution to {sym} total ({sym_total_pnl:+.2f}):")
        for label in all_labels:
            pnls = regime_pnls.get(label, [])
            if not pnls:
                continue
            total = sum(pnls)
            pct = (total / sym_total_pnl * 100) if sym_total_pnl != 0 else 0
            bar_len = min(40, max(1, int(abs(pct) / 2.5)))
            bar = ("▓" if total > 0 else "░") * bar_len
            print(f"      {label:<16} {total:>+10.2f}  ({pct:>+6.1f}%)  {bar}")

        print()

    # ── Section 2: Score-bucketed PnL curve ──────────────────────────
    print("-" * 72)
    print("  SCORE-BUCKETED PnL CURVE (all core symbols combined)")
    print("-" * 72)
    print()

    all_scored = [(ep.get("hybrid_score", 0), ep.get("net_pnl", 0), ep.get("symbol"))
                  for ep in core_episodes if ep.get("hybrid_score", 0) > 0]
    all_scored.sort()

    if all_scored:
        lo = all_scored[0][0]
        hi = all_scored[-1][0]
        step = max((hi - lo) / 10, 0.01)

        print(f"    {'score bucket':>20}  {'n':>4}  {'total_pnl':>10}  {'mean_pnl':>9}  {'win%':>6}  {'BTC':>4}  {'ETH':>4}  {'SOL':>4}")
        print(f"    {'-'*20}  {'-'*4}  {'-'*10}  {'-'*9}  {'-'*6}  {'-'*4}  {'-'*4}  {'-'*4}")

        edge = lo
        while edge < hi:
            upper = min(edge + step, hi + 0.0001)
            bucket = [(s, p, sym) for s, p, sym in all_scored if edge <= s < upper]
            if bucket:
                pnls = [p for _, p, _ in bucket]
                n = len(pnls)
                total = sum(pnls)
                mean = total / n
                wr = sum(1 for p in pnls if p > 0) / n * 100
                btc = sum(1 for _, _, s in bucket if s == "BTCUSDT")
                eth = sum(1 for _, _, s in bucket if s == "ETHUSDT")
                sol = sum(1 for _, _, s in bucket if s == "SOLUSDT")
                print(f"    [{edge:.4f}, {upper:.4f})  {n:>4}  {total:>+10.2f}  {mean:>+9.4f}  {wr:>5.1f}%  {btc:>4}  {eth:>4}  {sol:>4}")
            edge = upper
        print()

    # ── Section 3: Per-symbol score-vs-PnL monotonicity ──────────────
    print("-" * 72)
    print("  SCORE vs PnL MONOTONICITY (does higher score = better PnL?)")
    print("-" * 72)
    print()

    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        scored = [(ep.get("hybrid_score", 0), ep.get("net_pnl", 0))
                  for ep in core_episodes
                  if ep.get("symbol") == sym and ep.get("hybrid_score", 0) > 0]
        scored.sort()

        if len(scored) < 10:
            print(f"  {sym}: insufficient scored episodes (n={len(scored)})")
            print()
            continue

        # Split into terciles
        n = len(scored)
        t1 = scored[:n // 3]
        t2 = scored[n // 3: 2 * n // 3]
        t3 = scored[2 * n // 3:]

        def _tercile_stats(recs: List[Tuple[float, float]], label: str) -> str:
            scores = [s for s, _ in recs]
            pnls = [p for _, p in recs]
            n = len(pnls)
            total = sum(pnls)
            mean = total / n
            wr = sum(1 for p in pnls if p > 0) / n * 100
            return (f"    {label:<10} score=[{min(scores):.4f}, {max(scores):.4f}]"
                    f"  n={n:>3}  pnl={total:>+8.2f}  mean={mean:>+7.4f}  win={wr:>5.1f}%")

        print(f"  {sym}:")
        print(_tercile_stats(t1, "LOW"))
        print(_tercile_stats(t2, "MID"))
        print(_tercile_stats(t3, "HIGH"))

        # Monotonicity check
        means = [sum(p for _, p in t) / len(t) for t in [t1, t2, t3]]
        if means[0] < means[1] < means[2]:
            verdict = "MONOTONIC (higher score = better PnL)"
        elif means[0] > means[1] > means[2]:
            verdict = "INVERTED (higher score = worse PnL)"
        else:
            verdict = "NON-MONOTONIC (mixed relationship)"
        print(f"    → {verdict}")

        # Check if regime boundaries align with PnL regime transitions
        bounds = REGIME_BOUNDARIES.get(sym, [])
        if bounds:
            below_first = [p for s, p in scored if s < bounds[0]]
            above_first = [p for s, p in scored if s >= bounds[0]]
            if below_first and above_first:
                below_mean = sum(below_first) / len(below_first)
                above_mean = sum(above_first) / len(above_first)
                print(f"    Boundary {bounds[0]:.4f}: below mean={below_mean:+.4f} (n={len(below_first)})  above mean={above_mean:+.4f} (n={len(above_first)})")
                if below_mean > above_mean:
                    print(f"    → PnL CONFIRMS regime boundary (Hydra-regime outperforms)")
                else:
                    print(f"    → PnL does NOT confirm regime boundary at this threshold")
        print()

    # ── Section 4: Regime allocation summary ─────────────────────────
    print("=" * 72)
    print("  REGIME PnL SUMMARY")
    print("=" * 72)
    print()

    total_pnl = sum(ep.get("net_pnl", 0) for ep in core_episodes)
    hydra_regime_pnl = 0.0
    hydra_regime_n = 0
    legacy_regime_pnl = 0.0
    legacy_regime_n = 0

    for ep in core_episodes:
        sym = ep.get("symbol")
        score = ep.get("hybrid_score", 0)
        pnl = ep.get("net_pnl", 0)
        if score <= 0:
            continue
        regime = _classify_regime(sym, score)
        if "HYDRA" in regime:
            hydra_regime_pnl += pnl
            hydra_regime_n += 1
        else:
            legacy_regime_pnl += pnl
            legacy_regime_n += 1

    total_n = hydra_regime_n + legacy_regime_n
    print(f"  Total core PnL:    {total_pnl:>+10.2f}  (n={len(core_episodes)})")
    print(f"  Hydra-regime PnL:  {hydra_regime_pnl:>+10.2f}  (n={hydra_regime_n})")
    print(f"  Legacy-regime PnL: {legacy_regime_pnl:>+10.2f}  (n={legacy_regime_n})")
    print()
    if hydra_regime_n and legacy_regime_n:
        h_mean = hydra_regime_pnl / hydra_regime_n
        l_mean = legacy_regime_pnl / legacy_regime_n
        print(f"  Mean PnL per trade:")
        print(f"    Hydra regime:  {h_mean:>+.4f}  ({hydra_regime_n} trades)")
        print(f"    Legacy regime: {l_mean:>+.4f}  ({legacy_regime_n} trades)")
        edge = h_mean - l_mean
        print(f"    Edge (Hydra - Legacy): {edge:>+.4f}")
        if edge > 0:
            print(f"    → Hydra-regime trades have BETTER mean PnL")
        else:
            print(f"    → Legacy-regime trades have BETTER mean PnL")


if __name__ == "__main__":
    run()
