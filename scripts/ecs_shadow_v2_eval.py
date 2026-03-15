#!/usr/bin/env python3
"""ECS Shadow Selector v2 Evaluation — Phase 5 research analysis.

Joins v2 shadow decisions with realized PnL from episode_ledger to
evaluate three experimental routing rules against live ECS.

For each scored episode, produces:
  1. ecs_choice   — what live ECS chose
  2. v2_choice    — what each candidate selector would choose
  3. v2_abstain   — whether v2 would decline the trade
  4. regret_delta — PnL difference vs live ECS

Sections:
  1. Summary statistics (per candidate, per symbol)
  2. Routing disagreement map (where v2 ≠ ECS)
  3. Regret heatmap by score bucket × symbol
  4. Abstention benefit curve (ECS vs Hydra-only vs v2 vs v2+abstain)
  5. Cumulative PnL walk (four strategies)
  6. Data sufficiency check

Usage:
    PYTHONPATH=. python scripts/ecs_shadow_v2_eval.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

EPISODE_LEDGER = Path("logs/state/episode_ledger.json")
V2_SHADOW_LOG = Path("logs/execution/selector_v2_shadow.jsonl")
SOAK_LOG = Path("logs/execution/ecs_soak_events.jsonl")

# Regime boundaries (same as shadow_selector_v2.py)
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


def _classify_regime(symbol: str, score: float) -> str:
    bounds = REGIME_BOUNDARIES.get(symbol, [])
    labels = REGIME_LABELS.get(symbol, ["UNKNOWN"])
    if not bounds:
        return labels[0] if labels else "UNKNOWN"
    for i, threshold in enumerate(bounds):
        if score < threshold:
            return labels[i] if i < len(labels) else "UNKNOWN"
    return labels[-1] if labels else "UNKNOWN"


def _load_episodes() -> List[Dict[str, Any]]:
    if not EPISODE_LEDGER.exists():
        return []
    with open(EPISODE_LEDGER) as f:
        data = json.load(f)
    return data.get("episodes", data.get("entries", []))


def _load_v2_events() -> List[Dict[str, Any]]:
    if not V2_SHADOW_LOG.exists():
        return []
    events = []
    with open(V2_SHADOW_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def _load_soak_events() -> List[Dict[str, Any]]:
    if not SOAK_LOG.exists():
        return []
    events = []
    with open(SOAK_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def _pnl_stats(pnls: List[float]) -> Dict[str, Any]:
    if not pnls:
        return {"count": 0, "total": 0, "mean": 0, "win_rate": 0}
    total = sum(pnls)
    mean = total / len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    return {
        "count": len(pnls),
        "total": round(total, 2),
        "mean": round(mean, 2),
        "win_rate": round(wins / len(pnls) * 100, 1),
    }


def _sharpe(pnls: List[float]) -> float:
    if len(pnls) < 2:
        return 0.0
    mean = sum(pnls) / len(pnls)
    var = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
    std = var ** 0.5
    return round(mean / std, 3) if std > 1e-9 else 0.0


# ── Join v2 shadow events with PnL ──────────────────────────────────────────

def _build_scored_dataset() -> List[Dict[str, Any]]:
    """Join soak events (scores) with episode PnL using timestamp proximity.

    Returns list of dicts with: symbol, hydra_score, legacy_score,
    score_delta, ecs_choice, regime, pnl, and v2 verdicts.
    """
    episodes = _load_episodes()
    soak_events = _load_soak_events()

    # Build episode index: (symbol, close_ts_bucket) → pnl
    ep_index: Dict[Tuple[str, int], float] = {}
    for ep in episodes:
        sym = ep.get("symbol", "")
        pnl = ep.get("pnl") or ep.get("realized_pnl") or ep.get("pnl_usdt")
        close_ts = ep.get("close_ts") or ep.get("closed_at") or ep.get("ts")
        if sym and pnl is not None and close_ts is not None:
            try:
                ts_bucket = int(float(close_ts))
                ep_index[(sym, ts_bucket)] = float(pnl)
            except (ValueError, TypeError):
                continue

    # Build soak index: (symbol, ts_bucket) → soak_event
    soak_index: Dict[str, List[Dict]] = defaultdict(list)
    for ev in soak_events:
        sym = ev.get("symbol", "")
        if sym and ev.get("merge_hydra_score") is not None:
            soak_index[sym].append(ev)

    # Sort soak events by timestamp
    for sym in soak_index:
        soak_index[sym].sort(key=lambda e: e.get("ts", 0))

    # Join: for each soak event with scores, find closest episode PnL
    scored: List[Dict[str, Any]] = []
    for sym, events in soak_index.items():
        sym_episodes = [(ts, pnl) for (s, ts), pnl in ep_index.items() if s == sym]
        sym_episodes.sort()

        for ev in events:
            h_score = ev.get("merge_hydra_score")
            l_score = ev.get("merge_legacy_score")
            if h_score is None:
                continue

            soak_ts = ev.get("ts", 0)
            ecs_winner = ev.get("ecs_winner", "unknown")

            # Find closest episode within 300s window
            best_pnl = None
            best_dist = float("inf")
            for ep_ts, ep_pnl in sym_episodes:
                dist = abs(ep_ts - soak_ts)
                if dist < best_dist and dist < 300:
                    best_dist = dist
                    best_pnl = ep_pnl

            if best_pnl is None:
                continue

            regime = _classify_regime(sym, float(h_score))
            score_delta = ev.get("score_delta")
            if score_delta is None and l_score is not None:
                score_delta = round(float(h_score) - float(l_score), 6)

            scored.append({
                "symbol": sym,
                "ts": soak_ts,
                "hydra_score": float(h_score),
                "legacy_score": float(l_score) if l_score is not None else 0.0,
                "score_delta": score_delta,
                "regime": regime,
                "ecs_choice": ecs_winner,
                "pnl": float(best_pnl),
            })

    scored.sort(key=lambda r: r["ts"])
    return scored


def _apply_v2_selectors(
    scored: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Apply all three v2 candidate selectors to scored dataset."""
    from execution.shadow_selector_v2 import (
        _selector_a,
        _selector_b,
        _selector_c,
    )

    for row in scored:
        sym = row["symbol"]
        h = row["hydra_score"]
        l = row["legacy_score"]
        ecs = row["ecs_choice"]

        a = _selector_a(sym, h, ecs)
        b = _selector_b(sym, h)
        c = _selector_c(sym, h, l)

        row["a_choice"] = a["v2_choice"]
        row["a_abstain"] = a["v2_abstain"]
        row["a_rule"] = a["rule"]
        row["b_choice"] = b["v2_choice"]
        row["b_abstain"] = b["v2_abstain"]
        row["b_rule"] = b["rule"]
        row["c_choice"] = c["v2_choice"]
        row["c_abstain"] = c["v2_abstain"]
        row["c_rule"] = c["rule"]

        # Regret deltas: PnL that v2 would save vs ECS
        # If v2 abstains → regret_delta = -pnl (avoids the loss or misses the win)
        # If v2 agrees with ECS → regret_delta = 0
        # If v2 disagrees → cannot know counterfactual PnL without replay
        #   conservative: assume same PnL → regret_delta = 0
        for prefix in ("a", "b"):
            choice = row[f"{prefix}_choice"]
            abstain = row[f"{prefix}_abstain"]
            if abstain:
                # Abstaining means PnL would be 0 instead of actual
                row[f"{prefix}_regret_delta"] = round(-row["pnl"], 2)
            elif choice == ecs:
                row[f"{prefix}_regret_delta"] = 0.0
            else:
                # Disagreement: would have chosen different engine
                # PnL unknown without replay — mark as unknown
                row[f"{prefix}_regret_delta"] = None

    return scored


# ── Output sections ──────────────────────────────────────────────────────────

def _print_header(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def _print_summary(scored: List[Dict[str, Any]]) -> None:
    """Section 1: Summary statistics per candidate, per symbol."""
    _print_header("1. SUMMARY STATISTICS")

    print(f"\nTotal scored episodes: {len(scored)}")

    # ECS actual
    all_pnl = [r["pnl"] for r in scored]
    stats = _pnl_stats(all_pnl)
    print(f"\nECS (live):  N={stats['count']}  PnL={stats['total']:+.2f}"
          f"  mean={stats['mean']:+.2f}  win={stats['win_rate']:.1f}%"
          f"  Sharpe={_sharpe(all_pnl)}")

    # Hydra-only (regime filter)
    hydra_pnl = [r["pnl"] for r in scored if "HYDRA" in r["regime"]]
    h_stats = _pnl_stats(hydra_pnl)
    print(f"Hydra-only:  N={h_stats['count']}  PnL={h_stats['total']:+.2f}"
          f"  mean={h_stats['mean']:+.2f}  win={h_stats['win_rate']:.1f}%"
          f"  Sharpe={_sharpe(hydra_pnl)}")

    # Per-candidate summary
    for label, prefix in [("Candidate A (band)", "a"), ("Candidate B (region)", "b")]:
        # Trades v2 would take
        taken = [r for r in scored if not r.get(f"{prefix}_abstain")]
        taken_pnl = [r["pnl"] for r in taken]
        t_stats = _pnl_stats(taken_pnl)

        # Trades v2 would abstain
        abstained = [r for r in scored if r.get(f"{prefix}_abstain")]
        abs_pnl = [r["pnl"] for r in abstained]
        abs_stats = _pnl_stats(abs_pnl)

        print(f"\n{label}:")
        print(f"  Taken:     N={t_stats['count']}  PnL={t_stats['total']:+.2f}"
              f"  mean={t_stats['mean']:+.2f}  Sharpe={_sharpe(taken_pnl)}")
        print(f"  Abstained: N={abs_stats['count']}  avoided_PnL={abs_stats['total']:+.2f}"
              f"  (losses avoided={sum(-p for p in abs_pnl if p < 0):.2f})")

    # Per-symbol breakdown
    symbols = sorted(set(r["symbol"] for r in scored))
    print(f"\nPer-symbol ECS:")
    for sym in symbols:
        sym_pnl = [r["pnl"] for r in scored if r["symbol"] == sym]
        s = _pnl_stats(sym_pnl)
        print(f"  {sym:12s} N={s['count']:3d}  PnL={s['total']:+8.2f}"
              f"  mean={s['mean']:+.2f}")


def _print_routing_disagreement(scored: List[Dict[str, Any]]) -> None:
    """Section 2: Where v2 disagrees with ECS."""
    _print_header("2. ROUTING DISAGREEMENT MAP")

    for label, prefix in [("Candidate A", "a"), ("Candidate B", "b")]:
        disagree = [r for r in scored
                    if r.get(f"{prefix}_choice") != r["ecs_choice"]]
        agree = [r for r in scored
                 if r.get(f"{prefix}_choice") == r["ecs_choice"]]

        print(f"\n{label}:")
        print(f"  Agreement: {len(agree)}/{len(scored)}"
              f" ({len(agree)/len(scored)*100:.1f}%)" if scored else "  No data")
        print(f"  Disagreements: {len(disagree)}")

        if disagree:
            # Show score distribution of disagreements
            abstain_d = [r for r in disagree if r.get(f"{prefix}_abstain")]
            reroute_d = [r for r in disagree if not r.get(f"{prefix}_abstain")]

            if abstain_d:
                abs_pnl = sum(r["pnl"] for r in abstain_d)
                print(f"    Abstentions: {len(abstain_d)}"
                      f"  (avoided PnL: {abs_pnl:+.2f})")
            if reroute_d:
                re_pnl = sum(r["pnl"] for r in reroute_d)
                print(f"    Reroutes:    {len(reroute_d)}"
                      f"  (affected PnL: {re_pnl:+.2f})")

            # Show top 5 disagreements by absolute PnL impact
            disagree_sorted = sorted(disagree, key=lambda r: abs(r["pnl"]),
                                     reverse=True)[:5]
            print(f"    Top disagreements by |PnL|:")
            for r in disagree_sorted:
                print(f"      {r['symbol']:10s} score={r['hydra_score']:.4f}"
                      f"  ecs={r['ecs_choice']:7s} v2={r.get(f'{prefix}_choice'):7s}"
                      f"  pnl={r['pnl']:+.2f}"
                      f"  {'ABSTAIN' if r.get(f'{prefix}_abstain') else ''}")


def _print_regret_heatmap(scored: List[Dict[str, Any]]) -> None:
    """Section 3: Regret by score bucket × symbol."""
    _print_header("3. REGRET HEATMAP (Candidate B — positive-region routing)")

    # Bucket by hydra_score decile
    buckets = defaultdict(lambda: defaultdict(list))
    for r in scored:
        bucket = round(r["hydra_score"], 1)  # 0.1 buckets
        sym = r["symbol"]
        regret = r.get("b_regret_delta")
        if regret is not None:
            buckets[bucket][sym].append(regret)

    if not buckets:
        print("\n  No regret data available.")
        return

    symbols = sorted(set(r["symbol"] for r in scored))
    header = f"  {'Bucket':>8s}"
    for sym in symbols:
        header += f"  {sym:>12s}"
    header += f"  {'TOTAL':>10s}"
    print(f"\n{header}")
    print("  " + "-" * (len(header) - 2))

    for bucket in sorted(buckets.keys()):
        row = f"  {bucket:8.1f}"
        total_regret = 0.0
        for sym in symbols:
            regrets = buckets[bucket].get(sym, [])
            if regrets:
                s = sum(regrets)
                total_regret += s
                row += f"  {s:+10.2f}({len(regrets)})"
            else:
                row += f"  {'---':>12s}"
        row += f"  {total_regret:+10.2f}"
        print(row)


def _print_abstention_curve(scored: List[Dict[str, Any]]) -> None:
    """Section 4: Cumulative PnL walk — ECS vs Hydra-only vs v2 vs v2+abstain."""
    _print_header("4. ABSTENTION BENEFIT CURVE")

    cum_ecs = 0.0
    cum_hydra = 0.0
    cum_b = 0.0
    cum_b_abstain = 0.0

    walk: List[Dict[str, float]] = []

    for r in scored:
        pnl = r["pnl"]
        is_hydra_regime = "HYDRA" in r["regime"]

        cum_ecs += pnl
        if is_hydra_regime:
            cum_hydra += pnl

        # Candidate B: take if Hydra regime, else abstain
        if not r.get("b_abstain"):
            cum_b += pnl
            cum_b_abstain += pnl
        # If abstained, cum_b_abstain stays same (PnL = 0)

        walk.append({
            "ecs": round(cum_ecs, 2),
            "hydra_only": round(cum_hydra, 2),
            "b_taken": round(cum_b, 2),
            "b_with_abstain": round(cum_b_abstain, 2),
        })

    if not walk:
        print("\n  No data.")
        return

    # Print every 10th point and the final
    print(f"\n  {'#':>5s}  {'ECS':>10s}  {'Hydra-only':>12s}"
          f"  {'B_taken':>10s}  {'B+abstain':>12s}")
    print("  " + "-" * 55)
    indices = list(range(0, len(walk), max(1, len(walk) // 10)))
    if (len(walk) - 1) not in indices:
        indices.append(len(walk) - 1)
    for idx in indices:
        w = walk[idx]
        print(f"  {idx + 1:5d}  {w['ecs']:+10.2f}  {w['hydra_only']:+12.2f}"
              f"  {w['b_taken']:+10.2f}  {w['b_with_abstain']:+12.2f}")

    # Final comparison
    final = walk[-1]
    print(f"\n  FINAL:")
    print(f"    ECS (live):    {final['ecs']:+.2f}")
    print(f"    Hydra-only:    {final['hydra_only']:+.2f}")
    print(f"    B (taken):     {final['b_taken']:+.2f}")
    print(f"    B + abstain:   {final['b_with_abstain']:+.2f}")

    # Improvement vs ECS
    if abs(final["ecs"]) > 0.01:
        b_improvement = final["b_with_abstain"] - final["ecs"]
        print(f"\n    B+abstain improvement vs ECS: {b_improvement:+.2f}")
        print(f"    Losses avoided by abstention: "
              f"{sum(-r['pnl'] for r in scored if r.get('b_abstain') and r['pnl'] < 0):.2f}")


def _print_evaluation_metrics(scored: List[Dict[str, Any]]) -> None:
    """Section 5: Key evaluation metrics."""
    _print_header("5. EVALUATION METRICS")

    hydra_trades = [r for r in scored if "HYDRA" in r["regime"]]
    legacy_trades = [r for r in scored if "HYDRA" not in r["regime"]]

    hydra_pnl = [r["pnl"] for r in hydra_trades]
    legacy_pnl = [r["pnl"] for r in legacy_trades]
    all_pnl = [r["pnl"] for r in scored]

    print(f"\n  Hydra-regime trades: {len(hydra_trades)}")
    print(f"  Legacy-regime trades: {len(legacy_trades)}")

    h_ev = sum(hydra_pnl) / len(hydra_pnl) if hydra_pnl else 0
    print(f"\n  Hydra EV:        {h_ev:+.2f}/trade")
    print(f"  Hydra Sharpe:    {_sharpe(hydra_pnl)}")

    if legacy_trades:
        regret = sum(-r["pnl"] for r in legacy_trades)
        print(f"  Selector regret: {regret:+.2f} total"
              f" ({regret / len(scored):.2f}/trade)")

    # H-L spread
    h_cum = sum(hydra_pnl)
    l_cum = sum(legacy_pnl)
    print(f"\n  H-L spread:      {h_cum - l_cum:+.2f}"
          f" (H={h_cum:+.2f}, L={l_cum:+.2f})")

    # Candidate B abstention efficiency
    b_abstained = [r for r in scored if r.get("b_abstain")]
    b_taken_pnl = [r["pnl"] for r in scored if not r.get("b_abstain")]
    if b_abstained:
        losses_avoided = sum(-r["pnl"] for r in b_abstained if r["pnl"] < 0)
        wins_missed = sum(r["pnl"] for r in b_abstained if r["pnl"] > 0)
        print(f"\n  Candidate B abstention:")
        print(f"    Trades abstained:  {len(b_abstained)}")
        print(f"    Losses avoided:    {losses_avoided:+.2f}")
        print(f"    Wins missed:       {wins_missed:+.2f}")
        print(f"    Net benefit:       {losses_avoided - wins_missed:+.2f}")

    # Capacity ratio: is v2 capturing most of Hydra's edge?
    pnl_hydra_only = h_cum
    pnl_b = sum(b_taken_pnl) if b_taken_pnl else 0.0
    if abs(pnl_hydra_only) > 0.01:
        cap_ratio = pnl_b / pnl_hydra_only
        print(f"\n  Capacity ratio (B / Hydra-only): {cap_ratio:.3f}")
        if cap_ratio > 0.8:
            print(f"    → Near-optimal routing")
        elif cap_ratio > 0.5:
            print(f"    → Moderate edge capture")
        else:
            print(f"    → Routing losing too much edge")


def _print_data_sufficiency(scored: List[Dict[str, Any]]) -> None:
    """Section 6: Data sufficiency check."""
    _print_header("6. DATA SUFFICIENCY")

    hydra_n = sum(1 for r in scored if "HYDRA" in r["regime"])
    target = 100

    print(f"\n  Hydra-regime trades: {hydra_n} / {target} target")
    print(f"  Progress: {'█' * min(50, int(50 * hydra_n / target))}"
          f"{'░' * max(0, 50 - int(50 * hydra_n / target))}"
          f" {hydra_n / target * 100:.0f}%")

    if hydra_n < target:
        print(f"\n  ⚠ INSUFFICIENT DATA for selector modification.")
        print(f"    Need {target - hydra_n} more Hydra-regime trades.")
        print(f"    Keep selector_v2 in shadow-only mode.")
    else:
        print(f"\n  ✓ SUFFICIENT DATA for selector evaluation.")
        print(f"    Review metrics above before any live changes.")

    # V2 shadow events count
    v2_events = _load_v2_events()
    print(f"\n  v2 shadow events logged: {len(v2_events)}")

    # Per-symbol Hydra counts
    symbols = sorted(set(r["symbol"] for r in scored))
    for sym in symbols:
        n = sum(1 for r in scored if r["symbol"] == sym and "HYDRA" in r["regime"])
        print(f"    {sym}: {n} Hydra-regime trades")


def main() -> None:
    print("ECS Shadow Selector v2 Evaluation")
    print("=" * 70)

    scored = _build_scored_dataset()
    if not scored:
        print("\nNo scored episodes found.")
        print("Ensure episode_ledger.json and ecs_soak_events.jsonl exist")
        print("with merge_hydra_score fields populated.")
        return

    scored = _apply_v2_selectors(scored)

    _print_summary(scored)
    _print_routing_disagreement(scored)
    _print_regret_heatmap(scored)
    _print_abstention_curve(scored)
    _print_evaluation_metrics(scored)
    _print_data_sufficiency(scored)


if __name__ == "__main__":
    main()
