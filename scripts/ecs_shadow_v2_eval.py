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
from typing import Any, Dict, List, Optional

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

CORE_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}


def _build_scored_dataset() -> List[Dict[str, Any]]:
    """Build scored dataset from episode ledger (hybrid_score + net_pnl).

    Episodes already carry hybrid_score (= hydra_score used for regime
    classification) and net_pnl.  For legacy_score, we optionally enrich
    from soak events matched by (symbol, entry_ts) proximity.

    Returns list of dicts with: symbol, hydra_score, legacy_score,
    score_delta, ecs_choice, regime, pnl.
    """
    episodes = _load_episodes()
    soak_events = _load_soak_events()

    # Build soak index for legacy_score enrichment: symbol → sorted list
    soak_index: Dict[str, List[Dict]] = defaultdict(list)
    for ev in soak_events:
        sym = ev.get("symbol", "")
        if sym and ev.get("merge_hydra_score") is not None:
            soak_index[sym].append(ev)
    for sym in soak_index:
        soak_index[sym].sort(key=lambda e: e.get("ts", 0))

    def _find_legacy_score(sym: str, entry_ts_unix: float) -> Optional[float]:
        """Find closest soak event's legacy score within 120s of entry."""
        candidates = soak_index.get(sym, [])
        best_score: Optional[float] = None
        best_dist = float("inf")
        for ev in candidates:
            dist = abs(ev.get("ts", 0) - entry_ts_unix)
            if dist < best_dist and dist < 120:
                best_dist = dist
                best_score = ev.get("merge_legacy_score")
        return float(best_score) if best_score is not None else None

    scored: List[Dict[str, Any]] = []
    for ep in episodes:
        sym = ep.get("symbol", "")
        if sym not in CORE_SYMBOLS:
            continue

        h_score = ep.get("hybrid_score", 0)
        if not h_score or h_score <= 0:
            continue

        pnl = ep.get("net_pnl")
        if pnl is None:
            continue

        # Parse entry_ts for soak enrichment
        entry_ts_str = ep.get("entry_ts", "")
        entry_ts_unix = 0.0
        if entry_ts_str:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(entry_ts_str)
                entry_ts_unix = dt.timestamp()
            except (ValueError, TypeError):
                pass

        # Parse exit_ts for chronological ordering
        exit_ts_str = ep.get("exit_ts", "")
        exit_ts_unix = 0.0
        if exit_ts_str:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(exit_ts_str)
                exit_ts_unix = dt.timestamp()
            except (ValueError, TypeError):
                pass

        regime = _classify_regime(sym, float(h_score))
        ecs_choice = ep.get("engine_source", "unknown") or "unknown"

        # Enrich with legacy_score from soak events
        l_score = _find_legacy_score(sym, entry_ts_unix) if entry_ts_unix > 0 else None
        score_delta = round(float(h_score) - l_score, 6) if l_score is not None else None

        scored.append({
            "symbol": sym,
            "ts": exit_ts_unix or entry_ts_unix,
            "hydra_score": float(h_score),
            "legacy_score": l_score if l_score is not None else 0.0,
            "score_delta": score_delta,
            "regime": regime,
            "ecs_choice": ecs_choice,
            "pnl": float(pnl),
        })

    scored.sort(key=lambda r: r["ts"])
    return scored


def _apply_v2_selectors(
    scored: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Apply all four v2 candidate selectors to scored dataset."""
    from execution.shadow_selector_v2 import (
        _selector_a,
        _selector_b,
        _selector_c,
        _selector_d,
        in_profit_region,
    )

    for row in scored:
        sym = row["symbol"]
        h = row["hydra_score"]
        l = row["legacy_score"]
        ecs = row["ecs_choice"]

        a = _selector_a(sym, h, ecs)
        b = _selector_b(sym, h)
        c = _selector_c(sym, h, l)
        d = _selector_d(sym, h)

        row["a_choice"] = a["v2_choice"]
        row["a_abstain"] = a["v2_abstain"]
        row["a_rule"] = a["rule"]
        row["b_choice"] = b["v2_choice"]
        row["b_abstain"] = b["v2_abstain"]
        row["b_rule"] = b["rule"]
        row["c_choice"] = c["v2_choice"]
        row["c_abstain"] = c["v2_abstain"]
        row["c_rule"] = c["rule"]
        row["d_choice"] = d["v2_choice"]
        row["d_abstain"] = d["v2_abstain"]
        row["d_rule"] = d["rule"]
        row["profit_region"] = in_profit_region(sym, h)

        # Regret deltas: PnL that v2 would save vs ECS
        # If v2 abstains → regret_delta = -pnl (avoids the loss or misses the win)
        # If v2 agrees with ECS → regret_delta = 0
        # If v2 disagrees → cannot know counterfactual PnL without replay
        #   conservative: assume same PnL → regret_delta = 0
        for prefix in ("a", "b", "d"):
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
    for label, prefix in [(
        "Candidate A (band)", "a"),
        ("Candidate B (region)", "b"),
        ("Candidate D (profit)", "d"),
    ]:
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

    for label, prefix in [("Candidate A", "a"), ("Candidate B", "b"), ("Candidate D", "d")]:
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
    _print_header("3. REGRET HEATMAP")

    for cand_label, prefix in [("Candidate B (arbitration)", "b"), ("Candidate D (profit)", "d")]:
        print(f"\n  {cand_label}:")

        # Bucket by hydra_score decile
        buckets: Dict[float, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        for r in scored:
            bucket = round(r["hydra_score"], 1)  # 0.1 buckets
            sym = r["symbol"]
            regret = r.get(f"{prefix}_regret_delta")
            if regret is not None:
                buckets[bucket][sym].append(regret)

        if not buckets:
            print("    No regret data available.")
            continue

        symbols = sorted(set(r["symbol"] for r in scored))
        header = f"    {'Bucket':>8s}"
        for sym in symbols:
            header += f"  {sym:>12s}"
        header += f"  {'TOTAL':>10s}"
        print(f"\n{header}")
        print("    " + "-" * (len(header) - 4))

        for bucket in sorted(buckets.keys()):
            row = f"    {bucket:8.1f}"
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
        print()


def _print_abstention_curve(scored: List[Dict[str, Any]]) -> None:
    """Section 4: Cumulative PnL walk — ECS vs Hydra-only vs B vs D."""
    _print_header("4. ABSTENTION BENEFIT CURVE")

    cum_ecs = 0.0
    cum_hydra = 0.0
    cum_b = 0.0
    cum_b_abstain = 0.0
    cum_d = 0.0
    cum_d_abstain = 0.0

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

        # Candidate D: take if profit region, else abstain
        if not r.get("d_abstain"):
            cum_d += pnl
            cum_d_abstain += pnl

        walk.append({
            "ecs": round(cum_ecs, 2),
            "hydra_only": round(cum_hydra, 2),
            "b_taken": round(cum_b, 2),
            "b_with_abstain": round(cum_b_abstain, 2),
            "d_taken": round(cum_d, 2),
            "d_with_abstain": round(cum_d_abstain, 2),
        })

    if not walk:
        print("\n  No data.")
        return

    # Print every 10th point and the final
    print(f"\n  {'#':>5s}  {'ECS':>10s}  {'Hydra-only':>12s}"
          f"  {'B_taken':>10s}  {'B+abstain':>12s}"
          f"  {'D_taken':>10s}  {'D+abstain':>12s}")
    print("  " + "-" * 79)
    indices = list(range(0, len(walk), max(1, len(walk) // 10)))
    if (len(walk) - 1) not in indices:
        indices.append(len(walk) - 1)
    for idx in indices:
        w = walk[idx]
        print(f"  {idx + 1:5d}  {w['ecs']:+10.2f}  {w['hydra_only']:+12.2f}"
              f"  {w['b_taken']:+10.2f}  {w['b_with_abstain']:+12.2f}"
              f"  {w['d_taken']:+10.2f}  {w['d_with_abstain']:+12.2f}")

    # Final comparison
    final = walk[-1]
    print(f"\n  FINAL:")
    print(f"    ECS (live):    {final['ecs']:+.2f}")
    print(f"    Hydra-only:    {final['hydra_only']:+.2f}")
    print(f"    B (taken):     {final['b_taken']:+.2f}")
    print(f"    B + abstain:   {final['b_with_abstain']:+.2f}")
    print(f"    D (taken):     {final['d_taken']:+.2f}")
    print(f"    D + abstain:   {final['d_with_abstain']:+.2f}")

    # Improvement vs ECS
    if abs(final["ecs"]) > 0.01:
        b_improvement = final["b_with_abstain"] - final["ecs"]
        d_improvement = final["d_with_abstain"] - final["ecs"]
        print(f"\n    B+abstain improvement vs ECS: {b_improvement:+.2f}")
        print(f"    D+abstain improvement vs ECS: {d_improvement:+.2f}")
        print(f"    Losses avoided by B abstention: "
              f"{sum(-r['pnl'] for r in scored if r.get('b_abstain') and r['pnl'] < 0):.2f}")
        print(f"    Losses avoided by D abstention: "
              f"{sum(-r['pnl'] for r in scored if r.get('d_abstain') and r['pnl'] < 0):.2f}")


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

    # Candidate D abstention efficiency
    d_abstained = [r for r in scored if r.get("d_abstain")]
    d_taken_pnl = [r["pnl"] for r in scored if not r.get("d_abstain")]
    if d_abstained:
        d_losses_avoided = sum(-r["pnl"] for r in d_abstained if r["pnl"] < 0)
        d_wins_missed = sum(r["pnl"] for r in d_abstained if r["pnl"] > 0)
        print(f"\n  Candidate D abstention:")
        print(f"    Trades abstained:  {len(d_abstained)}")
        print(f"    Losses avoided:    {d_losses_avoided:+.2f}")
        print(f"    Wins missed:       {d_wins_missed:+.2f}")
        print(f"    Net benefit:       {d_losses_avoided - d_wins_missed:+.2f}")

    # Capacity ratio: is v2 capturing most of Hydra's edge?
    pnl_hydra_only = h_cum
    pnl_b = sum(b_taken_pnl) if b_taken_pnl else 0.0
    pnl_d = sum(d_taken_pnl) if d_taken_pnl else 0.0
    if abs(pnl_hydra_only) > 0.01:
        cap_ratio_b = pnl_b / pnl_hydra_only
        cap_ratio_d = pnl_d / pnl_hydra_only
        print(f"\n  Capacity ratio (B / Hydra-only): {cap_ratio_b:.3f}")
        print(f"  Capacity ratio (D / Hydra-only): {cap_ratio_d:.3f}")
        for label, cr in [("B", cap_ratio_b), ("D", cap_ratio_d)]:
            if cr > 0.8:
                print(f"    {label}: Near-optimal routing")
            elif cr > 0.5:
                print(f"    {label}: Moderate edge capture")
            else:
                print(f"    {label}: Routing losing too much edge")


def _print_b_vs_d_comparison(scored: List[Dict[str, Any]]) -> None:
    """Section 6: B vs D head-to-head — arbitration vs profit surface."""
    _print_header("6. B vs D HEAD-TO-HEAD (arbitration vs profit surface)")

    # Categorise each trade by B/D agreement
    both_take: List[Dict] = []
    b_take_d_abstain: List[Dict] = []
    d_take_b_abstain: List[Dict] = []
    both_abstain: List[Dict] = []

    for r in scored:
        b_abs = r.get("b_abstain", False)
        d_abs = r.get("d_abstain", False)
        if not b_abs and not d_abs:
            both_take.append(r)
        elif not b_abs and d_abs:
            b_take_d_abstain.append(r)
        elif b_abs and not d_abs:
            d_take_b_abstain.append(r)
        else:
            both_abstain.append(r)

    def _group_stats(rows: List[Dict]) -> str:
        if not rows:
            return "n=0"
        pnls = [r["pnl"] for r in rows]
        total = sum(pnls)
        mean = total / len(pnls)
        wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        return f"n={len(rows):>3}  PnL={total:>+8.2f}  mean={mean:>+6.2f}  win={wr:.0f}%"

    print(f"\n  Both take:          {_group_stats(both_take)}")
    print(f"  B takes, D abstains: {_group_stats(b_take_d_abstain)}")
    print(f"  D takes, B abstains: {_group_stats(d_take_b_abstain)}")
    print(f"  Both abstain:       {_group_stats(both_abstain)}")

    # Where B takes but D abstains — these are the trades D filters out
    if b_take_d_abstain:
        print(f"\n  Trades B takes but D filters out (arbitration-only zone):")
        sym_counts: Dict[str, List[float]] = defaultdict(list)
        for r in b_take_d_abstain:
            sym_counts[r["symbol"]].append(r["pnl"])
        for sym in sorted(sym_counts.keys()):
            pnls = sym_counts[sym]
            total = sum(pnls)
            print(f"    {sym:12s} n={len(pnls):>3}  PnL={total:>+8.2f}")
        total_filtered = sum(r["pnl"] for r in b_take_d_abstain)
        if total_filtered < 0:
            print(f"    → D correctly filters {abs(total_filtered):.2f} of losses")
        else:
            print(f"    → D filters {total_filtered:+.2f} (filtering profitable trades!)")

    # Where D takes but B abstains — edge D captures that B misses
    if d_take_b_abstain:
        print(f"\n  Trades D takes but B abstains (profit surface outside arbitration):")
        sym_counts2: Dict[str, List[float]] = defaultdict(list)
        for r in d_take_b_abstain:
            sym_counts2[r["symbol"]].append(r["pnl"])
        for sym in sorted(sym_counts2.keys()):
            pnls = sym_counts2[sym]
            total = sum(pnls)
            print(f"    {sym:12s} n={len(pnls):>3}  PnL={total:>+8.2f}")


def _print_profit_region_pnl(scored: List[Dict[str, Any]]) -> None:
    """Section 7: PnL split by profit_region bucket."""
    _print_header("7. PnL BY PROFIT REGION")

    inside = [r for r in scored if r.get("profit_region", False)]
    outside = [r for r in scored if not r.get("profit_region", False)]

    def _bucket_line(label: str, rows: List[Dict]) -> str:
        if not rows:
            return f"  {label:20s}  n=  0  PnL=    0.00  mean=  0.00  win=  0%"
        pnls = [r["pnl"] for r in rows]
        total = sum(pnls)
        mean = total / len(pnls)
        wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        return (f"  {label:20s}  n={len(pnls):>3}  PnL={total:>+8.2f}"
                f"  mean={mean:>+6.2f}  win={wr:.0f}%")

    print(f"\n{_bucket_line('Inside profit region', inside)}")
    print(f"{_bucket_line('Outside profit region', outside)}")

    # Per-symbol breakdown
    symbols = sorted(set(r["symbol"] for r in scored))
    if symbols:
        print(f"\n  Per-symbol breakdown:")
        for sym in symbols:
            sym_in = [r for r in inside if r["symbol"] == sym]
            sym_out = [r for r in outside if r["symbol"] == sym]
            in_pnl = sum(r["pnl"] for r in sym_in) if sym_in else 0.0
            out_pnl = sum(r["pnl"] for r in sym_out) if sym_out else 0.0
            print(f"    {sym:12s}  inside: n={len(sym_in):>3} PnL={in_pnl:>+8.2f}"
                  f"   outside: n={len(sym_out):>3} PnL={out_pnl:>+8.2f}")

    # Edge concentration
    if inside:
        in_total = sum(r["pnl"] for r in inside)
        all_total = sum(r["pnl"] for r in scored)
        if all_total != 0:
            pct = in_total / abs(all_total) * 100
            print(f"\n  Edge concentration: {pct:.0f}% of absolute PnL is inside profit regions")


def _print_data_sufficiency(scored: List[Dict[str, Any]]) -> None:
    """Section 8: Data sufficiency check."""
    _print_header("8. DATA SUFFICIENCY")

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
    _print_b_vs_d_comparison(scored)
    _print_profit_region_pnl(scored)
    _print_data_sufficiency(scored)


if __name__ == "__main__":
    main()
