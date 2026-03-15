#!/usr/bin/env python3
"""ECS Selector Regret Curve — measures PnL lost to sub-optimal routing.

For each scored episode, computes:
  regret = PnL lost by taking a trade that the regime-optimal
           strategy would have skipped (or vice versa).

The "optimal" baseline is Hydra-only: take trades only when the
hybrid_score falls in a Hydra-dominant regime, skip otherwise.

Sections:
  1. Per-trade regret (chronological)
  2. Cumulative regret walk
  3. Score-axis regret heatmap (where does regret concentrate?)
  4. Regret by symbol
  5. Three tracking metrics: Hydra EV, selector regret, Sharpe stability

Usage:
    PYTHONPATH=. python scripts/ecs_regret_curve.py
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

EPISODE_LEDGER = Path("logs/state/episode_ledger.json")

# Regime boundaries from phase map (866 soak events, 2026-03-15)
REGIME_BOUNDARIES: Dict[str, List[float]] = {
    "BTCUSDT": [0.5236],
    "ETHUSDT": [0.4291, 0.4883],
    "SOLUSDT": [],
}

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


def _classify_regime(symbol: str, score: float) -> str:
    bounds = REGIME_BOUNDARIES.get(symbol, [])
    labels = REGIME_LABELS.get(symbol, ["UNKNOWN"])
    if not bounds:
        return labels[0] if labels else "UNKNOWN"
    for i, threshold in enumerate(bounds):
        if score < threshold:
            return labels[i] if i < len(labels) else "UNKNOWN"
    return labels[-1] if labels else "UNKNOWN"


def _sharpe(pnls: List[float]) -> Optional[float]:
    if len(pnls) < 3:
        return None
    mean = sum(pnls) / len(pnls)
    var = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
    std = math.sqrt(var) if var > 0 else 0
    return mean / std if std > 0 else None


def run() -> None:
    episodes = _load_episodes()
    core_symbols = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
    scored = [ep for ep in episodes
              if ep.get("symbol") in core_symbols
              and ep.get("hybrid_score", 0) > 0]

    if not scored:
        print("No scored episodes found.")
        return

    scored_chrono = sorted(scored, key=lambda ep: ep.get("entry_ts", ""))

    print("=" * 72)
    print("  ECS Selector Regret Curve")
    print("=" * 72)
    print()
    print(f"  Scored episodes: {len(scored_chrono)}")
    print()

    # Classify each trade and compute per-trade regret
    trades: List[Dict[str, Any]] = []
    for ep in scored_chrono:
        sym = ep.get("symbol", "")
        score = ep.get("hybrid_score", 0)
        pnl = ep.get("net_pnl", 0)
        regime = _classify_regime(sym, score)
        is_hydra = "HYDRA" in regime

        # Regret model:
        # - Optimal strategy = take only Hydra-regime trades
        # - If Hydra regime: both strategies take it → regret = 0
        # - If Legacy regime: optimal skips, actual takes → regret = -pnl
        #   (positive regret = money lost by not skipping)
        if is_hydra:
            regret = 0.0
            action = "TAKE"
        else:
            regret = -pnl  # positive when pnl is negative (we lost money)
            action = "SKIP_OPTIMAL"

        trades.append({
            "sym": sym,
            "score": score,
            "pnl": pnl,
            "regime": regime,
            "is_hydra": is_hydra,
            "regret": regret,
            "action": action,
            "ts": ep.get("entry_ts", ""),
        })

    # ── Section 1: Cumulative regret walk ────────────────────────────
    print("-" * 72)
    print("  CUMULATIVE REGRET WALK (chronological)")
    print("  Regret = PnL lost by taking Legacy-regime trades")
    print("-" * 72)
    print()

    cum_regret = 0.0
    cum_actual = 0.0
    cum_optimal = 0.0  # Hydra-only cumulative

    print(f"    {'#':>4}  {'date':>10}  {'sym':>8}  {'score':>6}  {'pnl':>8}  {'regime':<16}"
          f"  {'regret':>8}  {'cum_regret':>10}  {'cum_actual':>10}  {'cum_optimal':>11}")
    print(f"    {'-'*4}  {'-'*10}  {'-'*8}  {'-'*6}  {'-'*8}  {'-'*16}"
          f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*11}")

    checkpoints: List[Dict[str, Any]] = []
    step = max(1, len(trades) // 20)  # ~20 rows

    for i, t in enumerate(trades):
        cum_regret += t["regret"]
        cum_actual += t["pnl"]
        if t["is_hydra"]:
            cum_optimal += t["pnl"]

        if (i + 1) % step == 0 or i == len(trades) - 1:
            checkpoints.append({
                "i": i + 1,
                "ts": t["ts"][:10] if t["ts"] else "",
                "cum_regret": cum_regret,
                "cum_actual": cum_actual,
                "cum_optimal": cum_optimal,
            })
            print(f"    {i+1:>4}  {t['ts'][:10] if t['ts'] else '':>10}  {t['sym']:>8}"
                  f"  {t['score']:>6.4f}  {t['pnl']:>+8.2f}  {t['regime']:<16}"
                  f"  {t['regret']:>+8.2f}  {cum_regret:>+10.2f}  {cum_actual:>+10.2f}"
                  f"  {cum_optimal:>+11.2f}")

    print()
    print(f"  Final: regret={cum_regret:+.2f}  actual={cum_actual:+.2f}  optimal={cum_optimal:+.2f}")
    print(f"  Regret as % of actual loss: {cum_regret / abs(cum_actual) * 100:.1f}%"
          if cum_actual != 0 else "")
    print()

    # ── Section 2: Score-axis regret heatmap ─────────────────────────
    print("-" * 72)
    print("  SCORE-AXIS REGRET HEATMAP")
    print("  (Where along the score axis does regret concentrate?)")
    print("-" * 72)
    print()

    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        sym_trades = [t for t in trades if t["sym"] == sym]
        if not sym_trades:
            continue

        scores = [t["score"] for t in sym_trades]
        lo, hi = min(scores), max(scores)
        n_buckets = 8
        step_size = max((hi - lo) / n_buckets, 0.005)

        bounds = REGIME_BOUNDARIES.get(sym, [])
        bounds_str = ", ".join(f"{b:.4f}" for b in bounds) if bounds else "none"

        print(f"  {sym}  (n={len(sym_trades)}, boundaries=[{bounds_str}])")
        print()
        print(f"    {'score range':>22}  {'n':>3}  {'total_pnl':>10}  {'regret':>8}  {'regime':<16}  {'regret bar'}")
        print(f"    {'-'*22}  {'-'*3}  {'-'*10}  {'-'*8}  {'-'*16}  {'-'*25}")

        edge = lo
        max_abs_regret = max(abs(t["regret"]) for t in sym_trades) or 1
        sym_total_regret = sum(t["regret"] for t in sym_trades)

        while edge < hi:
            upper = min(edge + step_size, hi + 0.0001)
            bucket = [t for t in sym_trades if edge <= t["score"] < upper]
            if bucket:
                n = len(bucket)
                total_pnl = sum(t["pnl"] for t in bucket)
                total_regret = sum(t["regret"] for t in bucket)
                regime = bucket[0]["regime"]

                # Visual bar: regret magnitude
                bar_len = int(abs(total_regret) / max(abs(sym_total_regret), 0.01) * 20)
                bar_len = min(25, max(0, bar_len))
                if total_regret > 0:
                    bar = "█" * bar_len  # positive regret = money lost
                else:
                    bar = "·" * min(3, n)  # zero/negative regret = good

                print(f"    [{edge:.4f}, {upper:.4f})  {n:>3}  {total_pnl:>+10.2f}"
                      f"  {total_regret:>+8.2f}  {regime:<16}  {bar}")
            edge = upper

        print()

    # ── Section 3: Regret by symbol ──────────────────────────────────
    print("-" * 72)
    print("  REGRET BY SYMBOL")
    print("-" * 72)
    print()

    print(f"    {'symbol':>8}  {'n':>4}  {'hydra_n':>7}  {'legacy_n':>8}  {'actual_pnl':>10}"
          f"  {'optimal_pnl':>11}  {'regret':>8}  {'regret%':>8}")
    print(f"    {'-'*8}  {'-'*4}  {'-'*7}  {'-'*8}  {'-'*10}"
          f"  {'-'*11}  {'-'*8}  {'-'*8}")

    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        sym_t = [t for t in trades if t["sym"] == sym]
        if not sym_t:
            continue
        n = len(sym_t)
        hydra_n = sum(1 for t in sym_t if t["is_hydra"])
        legacy_n = n - hydra_n
        actual = sum(t["pnl"] for t in sym_t)
        optimal = sum(t["pnl"] for t in sym_t if t["is_hydra"])
        regret = sum(t["regret"] for t in sym_t)
        regret_pct = regret / abs(actual) * 100 if actual != 0 else 0

        print(f"    {sym:>8}  {n:>4}  {hydra_n:>7}  {legacy_n:>8}  {actual:>+10.2f}"
              f"  {optimal:>+11.2f}  {regret:>+8.2f}  {regret_pct:>+7.1f}%")

    print()

    # ── Section 4: Largest individual regret trades ──────────────────
    print("-" * 72)
    print("  TOP 10 HIGHEST-REGRET TRADES")
    print("  (Legacy-regime trades with largest losses)")
    print("-" * 72)
    print()

    by_regret = sorted(trades, key=lambda t: t["regret"], reverse=True)[:10]

    print(f"    {'#':>3}  {'date':>10}  {'sym':>8}  {'score':>6}  {'pnl':>8}  {'regret':>8}  {'regime'}")
    print(f"    {'-'*3}  {'-'*10}  {'-'*8}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*16}")

    for j, t in enumerate(by_regret, 1):
        if t["regret"] <= 0:
            break
        print(f"    {j:>3}  {t['ts'][:10] if t['ts'] else '':>10}  {t['sym']:>8}"
              f"  {t['score']:>6.4f}  {t['pnl']:>+8.2f}  {t['regret']:>+8.2f}  {t['regime']}")

    print()

    # ── Section 5: Three tracking metrics ────────────────────────────
    print("-" * 72)
    print("  TRACKING METRICS (for ongoing monitoring)")
    print("-" * 72)
    print()

    # Metric 1: Hydra EV
    hydra_pnls = [t["pnl"] for t in trades if t["is_hydra"]]
    if hydra_pnls:
        h_mean = sum(hydra_pnls) / len(hydra_pnls)
        h_sharpe = _sharpe(hydra_pnls)
        h_wr = sum(1 for p in hydra_pnls if p > 0) / len(hydra_pnls) * 100
        print(f"  Metric 1 — Hydra EV:")
        print(f"    mean_pnl     = {h_mean:+.4f}")
        print(f"    sharpe       = {h_sharpe:+.3f}" if h_sharpe else "    sharpe       = N/A (too few)")
        print(f"    win_rate     = {h_wr:.1f}%")
        print(f"    n_trades     = {len(hydra_pnls)}")
        print(f"    status       = {'POSITIVE ✓' if h_mean > 0 else 'NEGATIVE ✗'}")
    else:
        print(f"  Metric 1 — Hydra EV: no Hydra-regime trades")
    print()

    # Metric 2: Selector regret
    total_regret = sum(t["regret"] for t in trades)
    avg_regret = total_regret / len(trades) if trades else 0
    regret_trades = sum(1 for t in trades if t["regret"] > 0)
    print(f"  Metric 2 — Selector Regret:")
    print(f"    total_regret = {total_regret:+.2f}")
    print(f"    avg_regret   = {avg_regret:+.4f} per trade")
    print(f"    regret_trades= {regret_trades}/{len(trades)} ({regret_trades/len(trades)*100:.1f}%)")
    print(f"    status       = {'HIGH — selector losing money' if avg_regret > 1 else 'MODERATE' if avg_regret > 0 else 'LOW'}")
    print()

    # Metric 3: Regime Sharpe stability
    # Compare first half vs second half of Hydra-regime trades
    print(f"  Metric 3 — Regime Sharpe Stability:")
    if len(hydra_pnls) >= 6:
        mid = len(hydra_pnls) // 2
        first_half = hydra_pnls[:mid]
        second_half = hydra_pnls[mid:]
        s1 = _sharpe(first_half)
        s2 = _sharpe(second_half)
        m1 = sum(first_half) / len(first_half)
        m2 = sum(second_half) / len(second_half)
        print(f"    first_half:  mean={m1:+.4f}  sharpe={s1:+.3f}" if s1 else f"    first_half:  mean={m1:+.4f}  sharpe=N/A")
        print(f"    second_half: mean={m2:+.4f}  sharpe={s2:+.3f}" if s2 else f"    second_half: mean={m2:+.4f}  sharpe=N/A")
        if s1 is not None and s2 is not None:
            drift = abs(s2 - s1)
            print(f"    sharpe_drift= {drift:.3f}  ({'STABLE' if drift < 0.3 else 'DRIFTING' if drift < 0.8 else 'UNSTABLE'})")
        else:
            print(f"    sharpe_drift= N/A (insufficient data)")
    else:
        print(f"    insufficient Hydra trades for stability check (n={len(hydra_pnls)}, need >=6)")
    print()

    # ── Summary ──────────────────────────────────────────────────────
    print("=" * 72)
    print("  REGRET SUMMARY")
    print("=" * 72)
    print()
    print(f"  Actual PnL (all scored):   {cum_actual:>+10.2f}  ({len(trades)} trades)")
    print(f"  Optimal PnL (Hydra-only):  {cum_optimal:>+10.2f}  ({sum(1 for t in trades if t['is_hydra'])} trades)")
    print(f"  Total regret:              {total_regret:>+10.2f}")
    print(f"  Regret per trade:          {avg_regret:>+10.4f}")
    print()

    if cum_actual < 0 and cum_optimal > 0:
        print(f"  The selector is routing to a net-loss outcome while")
        print(f"  the regime-optimal strategy produces positive PnL.")
        print(f"  Gap: {cum_optimal - cum_actual:+.2f} ({abs((cum_optimal - cum_actual) / abs(cum_actual) * 100):.0f}% of actual loss)")
    elif cum_actual < 0 and cum_optimal <= 0:
        print(f"  Both strategies negative, but optimal loses less.")
        saved = cum_actual - cum_optimal
        print(f"  Avoided loss: {saved:+.2f}")
    elif cum_actual >= 0:
        print(f"  Actual PnL already positive — selector is working.")

    print()
    print("  Action: continue monitoring. Do not modify selector until")
    print(f"  Hydra-regime trades exceed 100 (currently {sum(1 for t in trades if t['is_hydra'])}).")


if __name__ == "__main__":
    run()
