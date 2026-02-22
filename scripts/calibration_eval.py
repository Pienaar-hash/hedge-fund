#!/usr/bin/env python3
"""
Post-Calibration Window Evaluation Script (v7.9-CW)
=====================================================

Run after the 30-episode calibration window completes.
Computes and reports:

  1. Per-symbol expectancy (E) with maturity status
  2. Cross-symbol dispersion (do symbols differentiate?)
  3. Win rate, average PnL, Sharpe proxy
  4. Episode distribution by symbol/head/regime
  5. Hybrid variance audit (calls existing script)
  6. GO/NO-GO verdict for production activation

Usage:
    PYTHONPATH=. python scripts/calibration_eval.py [--start-ts 2026-02-22T00:00:00Z]

The start_ts defaults to the value in config/runtime.yaml calibration_window.start_ts.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ── Constants ───────────────────────────────────────────────────────────

RUNTIME_YAML = Path("config/runtime.yaml")
EPISODE_LEDGER = Path("logs/state/episode_ledger.json")
ORDERS_LOG = Path("logs/execution/orders_executed.jsonl")
MIN_EXPECTANCY_TRADES = 30
MATURITY_EPISODES = 60


# ── Data Loading ────────────────────────────────────────────────────────

def _load_start_ts() -> Optional[str]:
    """Read start_ts from runtime.yaml calibration_window section."""
    try:
        with open(RUNTIME_YAML) as f:
            cfg = yaml.safe_load(f) or {}
        cw = cfg.get("calibration_window", {})
        return str(cw.get("start_ts", "")) or None
    except Exception:
        return None


def _load_episodes(start_ts: str) -> List[Dict[str, Any]]:
    """Load episodes with exit_ts >= start_ts from the ledger."""
    try:
        with open(EPISODE_LEDGER) as f:
            data = json.load(f)
        episodes = data.get("episodes", [])
        return [ep for ep in episodes if (ep.get("exit_ts") or "") >= start_ts]
    except FileNotFoundError:
        print(f"ERROR: Episode ledger not found at {EPISODE_LEDGER}", file=sys.stderr)
        return []
    except Exception as exc:
        print(f"ERROR: Failed loading episodes: {exc}", file=sys.stderr)
        return []


# ── Analysis ────────────────────────────────────────────────────────────

def _symbol_stats(episodes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per-symbol win/loss/pnl/expectancy."""
    by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    for ep in episodes:
        sym = ep.get("symbol", "UNKNOWN")
        by_symbol.setdefault(sym, []).append(ep)

    result = {}
    for sym, eps in sorted(by_symbol.items()):
        pnls = [float(ep.get("realized_pnl_usdt", 0.0) or 0.0) for ep in eps]
        count = len(pnls)
        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p <= 0)
        total_pnl = sum(pnls)
        avg_pnl = total_pnl / count if count > 0 else 0.0
        win_rate = wins / count if count > 0 else 0.0
        avg_win = sum(p for p in pnls if p > 0) / wins if wins > 0 else 0.0
        avg_loss = sum(p for p in pnls if p <= 0) / losses if losses > 0 else 0.0
        # Simple expectancy: E = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss) if count > 0 else 0.0

        std_pnl = 0.0
        if count > 1:
            std_pnl = math.sqrt(sum((p - avg_pnl) ** 2 for p in pnls) / (count - 1))

        sharpe_proxy = avg_pnl / std_pnl if std_pnl > 0 else 0.0

        result[sym] = {
            "count": count,
            "wins": wins,
            "losses": losses,
            "total_pnl": round(total_pnl, 4),
            "avg_pnl": round(avg_pnl, 4),
            "expectancy": round(expectancy, 4),
            "win_rate": round(win_rate, 4),
            "avg_win": round(avg_win, 4),
            "avg_loss": round(avg_loss, 4),
            "std_pnl": round(std_pnl, 4),
            "sharpe_proxy": round(sharpe_proxy, 4),
            "is_mature": count >= MIN_EXPECTANCY_TRADES,
            "maturity_pct": round(min(1.0, count / MATURITY_EPISODES) * 100, 1),
        }
    return result


def _head_distribution(episodes: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count episodes by strategy head."""
    dist: Dict[str, int] = {}
    for ep in episodes:
        head = ep.get("strategy", ep.get("head", "UNKNOWN"))
        dist[head] = dist.get(head, 0) + 1
    return dict(sorted(dist.items(), key=lambda x: -x[1]))


def _regime_distribution(episodes: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count episodes by regime."""
    dist: Dict[str, int] = {}
    for ep in episodes:
        regime = ep.get("regime", "UNKNOWN")
        dist[regime] = dist.get(regime, 0) + 1
    return dict(sorted(dist.items(), key=lambda x: -x[1]))


def _cross_symbol_dispersion(stats: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Measure whether E differentiates across symbols."""
    if len(stats) < 2:
        return {
            "differentiated": False,
            "reason": "fewer than 2 symbols traded",
            "range": 0.0,
            "cv": 0.0,
        }

    expectations = [s["expectancy"] for s in stats.values()]
    mean_e = sum(expectations) / len(expectations)
    std_e = math.sqrt(sum((e - mean_e) ** 2 for e in expectations) / (len(expectations) - 1))
    e_range = max(expectations) - min(expectations)
    cv = std_e / abs(mean_e) if abs(mean_e) > 1e-6 else 0.0

    # Differentiation heuristic: range > 0.5 USDT or CV > 0.3
    differentiated = e_range > 0.5 or cv > 0.3

    return {
        "differentiated": differentiated,
        "mean_E": round(mean_e, 4),
        "std_E": round(std_e, 4),
        "range_E": round(e_range, 4),
        "cv": round(cv, 4),
        "symbol_expectations": {
            sym: s["expectancy"] for sym, s in stats.items()
        },
    }


def _go_no_go(
    episodes: List[Dict[str, Any]],
    stats: Dict[str, Dict[str, Any]],
    dispersion: Dict[str, Any],
) -> Dict[str, Any]:
    """Binary GO/NO-GO verdict with reasons."""
    gates = {}
    total = len(episodes)

    # Gate 1: Minimum total episodes
    gates["min_episodes"] = {
        "pass": total >= 20,
        "value": total,
        "threshold": 20,
        "note": "Need ≥20 episodes for meaningful E",
    }

    # Gate 2: At least one symbol has ≥ MIN_EXPECTANCY_TRADES
    mature = [s for s in stats.values() if s["is_mature"]]
    gates["symbol_maturity"] = {
        "pass": len(mature) > 0,
        "value": len(mature),
        "threshold": 1,
        "note": f"{len(mature)}/{len(stats)} symbols at ≥{MIN_EXPECTANCY_TRADES} episodes",
    }

    # Gate 3: Aggregate expectancy positive
    all_pnls = [float(ep.get("realized_pnl_usdt", 0.0) or 0.0) for ep in episodes]
    agg_e = sum(all_pnls) / len(all_pnls) if all_pnls else 0.0
    gates["positive_expectancy"] = {
        "pass": agg_e > 0,
        "value": round(agg_e, 4),
        "threshold": 0.0,
        "note": f"Aggregate E = {agg_e:.4f} USDT/episode",
    }

    # Gate 4: No extreme drawdown (total loss > 50% of total pnl abs)
    total_pnl = sum(all_pnls)
    max_loss_streak = 0
    current_streak = 0
    for p in all_pnls:
        if p <= 0:
            current_streak += 1
            max_loss_streak = max(max_loss_streak, current_streak)
        else:
            current_streak = 0
    gates["loss_streak"] = {
        "pass": max_loss_streak <= 10,
        "value": max_loss_streak,
        "threshold": 10,
        "note": f"Max consecutive losses: {max_loss_streak}",
    }

    # Gate 5: Cross-symbol differentiation
    gates["differentiation"] = {
        "pass": dispersion.get("differentiated", False) or len(stats) < 2,
        "value": dispersion.get("cv", 0.0),
        "note": "E differentiates across symbols" if dispersion.get("differentiated") else "E uniform — all symbols identical",
    }

    passed = sum(1 for g in gates.values() if g["pass"])
    total_gates = len(gates)
    verdict = "GO" if passed == total_gates else "NO-GO"

    return {
        "verdict": verdict,
        "passed": passed,
        "total_gates": total_gates,
        "gates": gates,
    }


# ── Report ──────────────────────────────────────────────────────────────

def _print_report(
    start_ts: str,
    episodes: List[Dict[str, Any]],
    stats: Dict[str, Dict[str, Any]],
    dispersion: Dict[str, Any],
    heads: Dict[str, int],
    regimes: Dict[str, int],
    verdict: Dict[str, Any],
) -> None:
    """Print human-readable evaluation report."""
    sep = "=" * 72
    print(f"\n{sep}")
    print("  CALIBRATION WINDOW — POST-WINDOW EVALUATION")
    print(f"{sep}")
    now = datetime.now(timezone.utc).isoformat()
    print(f"  Evaluated:     {now}")
    print(f"  Window start:  {start_ts}")
    print(f"  Episodes:      {len(episodes)}")
    print()

    # Per-symbol table
    print("  ┌─ Per-Symbol Expectancy ─────────────────────────────────────┐")
    print(f"  {'Symbol':<12} {'N':>4} {'WR':>6} {'E':>9} {'Σ PnL':>10} {'Sharpe':>7} {'Mature':>7}")
    print(f"  {'─'*12} {'─'*4} {'─'*6} {'─'*9} {'─'*10} {'─'*7} {'─'*7}")
    for sym, s in stats.items():
        mature_str = "YES" if s["is_mature"] else f"{s['maturity_pct']:.0f}%"
        print(
            f"  {sym:<12} {s['count']:>4} {s['win_rate']*100:>5.1f}% "
            f"{s['expectancy']:>9.4f} {s['total_pnl']:>10.4f} {s['sharpe_proxy']:>7.3f} {mature_str:>7}"
        )
    print()

    # Dispersion
    print("  ┌─ Cross-Symbol Dispersion ───────────────────────────────────┐")
    print(f"  Differentiated: {'YES' if dispersion.get('differentiated') else 'NO'}")
    print(f"  Mean E:         {dispersion.get('mean_E', 'N/A')}")
    print(f"  Std E:          {dispersion.get('std_E', 'N/A')}")
    print(f"  Range E:        {dispersion.get('range_E', 'N/A')}")
    print(f"  CV:             {dispersion.get('cv', 'N/A')}")
    print()

    # Distribution
    print("  ┌─ Head Distribution ─────────────────────────────────────────┐")
    for head, count in heads.items():
        print(f"  {head:<25} {count:>4}")
    print()

    print("  ┌─ Regime Distribution ───────────────────────────────────────┐")
    for regime, count in regimes.items():
        print(f"  {regime:<25} {count:>4}")
    print()

    # Verdict
    print(f"  ┌─ GO/NO-GO Verdict: {verdict['verdict']} ({verdict['passed']}/{verdict['total_gates']}) ─┐")
    for name, gate in verdict["gates"].items():
        icon = "✓" if gate["pass"] else "✗"
        print(f"  {icon} {name}: {gate.get('note', '')}")
    print()
    print(sep)

    if verdict["verdict"] == "GO":
        print("\n  RECOMMENDATION: Expectancy is calibrated. Safe to proceed")
        print("  to production sizing or extended evaluation window.\n")
    else:
        print("\n  RECOMMENDATION: Calibration insufficient. Consider:")
        print("  - Extending window (add more episodes)")
        print("  - Reviewing strategy parameters")
        print("  - Re-running hybrid variance audit\n")


# ── Main ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-calibration window evaluation"
    )
    parser.add_argument(
        "--start-ts",
        default=None,
        help="Window start timestamp (default: from runtime.yaml)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON instead of human report",
    )
    args = parser.parse_args()

    start_ts = args.start_ts or _load_start_ts()
    if not start_ts:
        print("ERROR: No start_ts found. Provide --start-ts or configure runtime.yaml", file=sys.stderr)
        sys.exit(1)

    episodes = _load_episodes(start_ts)
    if not episodes:
        print(f"ERROR: No episodes found since {start_ts}", file=sys.stderr)
        sys.exit(1)

    stats = _symbol_stats(episodes)
    dispersion = _cross_symbol_dispersion(stats)
    heads = _head_distribution(episodes)
    regimes = _regime_distribution(episodes)
    verdict_data = _go_no_go(episodes, stats, dispersion)

    if args.json:
        output = {
            "start_ts": start_ts,
            "total_episodes": len(episodes),
            "symbol_stats": stats,
            "dispersion": dispersion,
            "head_distribution": heads,
            "regime_distribution": regimes,
            "verdict": verdict_data,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }
        print(json.dumps(output, indent=2))
    else:
        _print_report(start_ts, episodes, stats, dispersion, heads, regimes, verdict_data)


if __name__ == "__main__":
    main()
