#!/usr/bin/env python3
"""
14-Day Performance Brief — Auto-Generator (v7.9-S1)

Reads state files and execution logs to produce the data tables
needed for the 14-day performance brief.  Does NOT modify any
state — purely read-only observability.

Usage:
    PYTHONPATH=. python scripts/generate_14d_brief.py [--since YYYY-MM-DD] [--until YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from statistics import mean, median, stdev

# ── Paths ────────────────────────────────────────────────────────────────────
NAV_STATE = Path("logs/state/nav_state.json")
RISK_SNAPSHOT = Path("logs/state/risk_snapshot.json")
EPISODE_LEDGER = Path("logs/state/episode_ledger.json")
SENTINEL_X = Path("logs/state/sentinel_x.json")
DIAGNOSTICS = Path("logs/state/diagnostics.json")

RISK_VETOES = Path("logs/execution/risk_vetoes.jsonl")
DOCTRINE_EVENTS = Path("logs/doctrine_events.jsonl")
SCORE_DECOMP = Path("logs/execution/score_decomposition.jsonl")
ORDERS_ATTEMPTED = Path("logs/execution/orders_attempted.jsonl")
SIZING_SNAPSHOTS = Path("logs/execution/sizing_snapshots.jsonl")

CONFIG_FILES = [
    Path("config/strategy_config.json"),
    Path("config/risk_limits.json"),
    Path("config/pairs_universe.json"),
]


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _load_jsonl(path: Path, since: datetime | None = None) -> list[dict]:
    """Load JSONL, optionally filtering by ts >= since."""
    if not path.exists():
        return []
    lines = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if since:
            ts_str = obj.get("ts") or ""
            if ts_str and ts_str < since.isoformat():
                continue
        lines.append(obj)
    return lines


def _load_jsonl_multi(base_path: Path, since: datetime | None = None) -> list[dict]:
    """Load JSONL from base path + any rotated variants (.1, .2, etc)."""
    results = _load_jsonl(base_path, since)
    # Check for rotated files
    for rotated in sorted(base_path.parent.glob(f"{base_path.name}.*")):
        if rotated.suffix in (".jsonl",):
            continue
        results.extend(_load_jsonl(rotated, since))
    return results


def section_capital(nav_state: dict, risk_snap: dict) -> None:
    print("\n## 1. Capital Summary\n")
    # NAV can be top-level or inside a series array
    nav = nav_state.get("nav") or nav_state.get("nav_usd") or 0
    if not nav:
        series = nav_state.get("series") or []
        if series:
            latest = series[-1] if isinstance(series, list) else {}
            nav = latest.get("nav") or latest.get("equity") or 0
    nav = float(nav or 0)
    print(f"  Current NAV:        ${nav:,.2f}")
    dd_pct = risk_snap.get("portfolio_dd_pct", 0) or 0
    dd_frac = risk_snap.get("dd_frac", 0) or 0
    daily_loss = risk_snap.get("daily_loss_frac", 0) or 0
    risk_mode = risk_snap.get("risk_mode", "unknown")
    print(f"  Risk Mode:          {risk_mode}")
    print(f"  Portfolio DD:       {dd_pct:.4%}")
    print(f"  Daily Loss Frac:    {daily_loss:.4%}")
    print(f"  DD Fraction:        {dd_frac:.4%}")


def section_trade_activity(
    episodes: list[dict],
    attempts: list[dict],
    vetoes: list[dict],
) -> None:
    print("\n## 2. Trade Activity\n")
    entry_attempts = [a for a in attempts if not a.get("reduce_only")]
    exit_attempts = [a for a in attempts if a.get("reduce_only")]
    entry_vetoes = [v for v in vetoes]

    print(f"  Episodes completed:     {len(episodes)}")
    print(f"  Entry attempts:         {len(entry_attempts)}")
    print(f"  Entry vetoes:           {len(entry_vetoes)}")
    total = len(entry_attempts) + len(entry_vetoes)
    if total > 0:
        accept_pct = len(entry_attempts) / total * 100
        print(f"  Acceptance rate:        {accept_pct:.1f}%")
    else:
        print("  Acceptance rate:        N/A (no attempts)")


def section_conviction_distribution(sizing: list[dict]) -> None:
    print("\n## 3. Conviction Distribution\n")
    band_counts: Counter = Counter()
    for s in sizing:
        band = s.get("conviction_band_shadow") or s.get("conviction_band") or "unscored"
        band_counts[band] += 1

    if not band_counts:
        print("  No sizing snapshots available yet.")
        return

    ordered = ["very_high", "high", "medium", "low", "very_low", "unscored"]
    print(f"  {'Band':<15} {'Count':>6} {'%':>8}")
    print(f"  {'-'*15} {'-'*6} {'-'*8}")
    total = sum(band_counts.values())
    for band in ordered:
        count = band_counts.get(band, 0)
        pct = count / total * 100 if total else 0
        print(f"  {band:<15} {count:>6} {pct:>7.1f}%")


def section_pnl_by_band(episodes: list[dict]) -> None:
    print("\n## 4. PnL by Conviction Band\n")
    band_data: dict[str, list[dict]] = defaultdict(list)
    for ep in episodes:
        band = ep.get("conviction_band") or "unscored"
        band_data[band].append(ep)

    if not any(band_data.values()):
        print("  No episodes with conviction band data yet.")
        return

    ordered = ["very_high", "high", "medium", "low", "very_low", "unscored"]
    print(f"  {'Band':<15} {'Eps':>5} {'WinRate':>8} {'Net PnL':>10} {'AvgDur(h)':>10}")
    print(f"  {'-'*15} {'-'*5} {'-'*8} {'-'*10} {'-'*10}")
    for band in ordered:
        eps = band_data.get(band, [])
        if not eps:
            print(f"  {band:<15} {'0':>5} {'—':>8} {'—':>10} {'—':>10}")
            continue
        winners = sum(1 for e in eps if (e.get("net_pnl") or 0) > 0)
        wr = winners / len(eps) * 100 if eps else 0
        net = sum(e.get("net_pnl", 0) or 0 for e in eps)
        avg_dur = mean([e.get("duration_hours", 0) or 0 for e in eps]) if eps else 0
        print(f"  {band:<15} {len(eps):>5} {wr:>7.1f}% ${net:>9.2f} {avg_dur:>9.1f}")


def section_hybrid_dispersion(decomp: list[dict]) -> None:
    print("\n## 5. Hybrid Score Dispersion\n")
    scores = [d.get("hybrid_score", 0) for d in decomp if "hybrid_score" in d]
    if not scores:
        print("  No score decomposition data yet.")
        return

    print(f"  Count:   {len(scores)}")
    print(f"  Mean:    {mean(scores):.4f}")
    print(f"  Median:  {median(scores):.4f}")
    if len(scores) >= 2:
        print(f"  Std Dev: {stdev(scores):.4f}")
    print(f"  Min:     {min(scores):.4f}")
    print(f"  Max:     {max(scores):.4f}")
    sorted_s = sorted(scores)
    q25 = sorted_s[len(sorted_s) // 4] if len(sorted_s) >= 4 else sorted_s[0]
    q75 = sorted_s[3 * len(sorted_s) // 4] if len(sorted_s) >= 4 else sorted_s[-1]
    print(f"  IQR:     {q25:.4f} – {q75:.4f}")


def section_strategy_attribution(episodes: list[dict]) -> None:
    print("\n## 6. Strategy Attribution\n")
    strat_data: dict[str, list[dict]] = defaultdict(list)
    for ep in episodes:
        strat = ep.get("strategy") or "unattributed"
        strat_data[strat].append(ep)

    print(f"  {'Strategy':<20} {'Eps':>5} {'WinRate':>8} {'Net PnL':>10} {'AvgConf':>8}")
    print(f"  {'-'*20} {'-'*5} {'-'*8} {'-'*10} {'-'*8}")
    for strat in sorted(strat_data.keys()):
        eps = strat_data[strat]
        winners = sum(1 for e in eps if (e.get("net_pnl") or 0) > 0)
        wr = winners / len(eps) * 100 if eps else 0
        net = sum(e.get("net_pnl", 0) or 0 for e in eps)
        confs = [e.get("confidence", 0) or 0 for e in eps]
        avg_conf = mean(confs) if confs else 0
        print(f"  {strat:<20} {len(eps):>5} {wr:>7.1f}% ${net:>9.2f} {avg_conf:>7.2f}")


def section_veto_analysis(vetoes: list[dict], doctrine: list[dict]) -> None:
    print("\n## 8. Veto Analysis\n")
    # Risk vetoes
    reason_counts: Counter = Counter()
    for v in vetoes:
        reason = v.get("veto_reason") or v.get("original_reason") or "unknown"
        reason_counts[reason] += 1

    # Doctrine vetoes
    doctrine_vetoes = [d for d in doctrine if d.get("verdict") == "VETO" or not d.get("allowed", True)]
    for d in doctrine_vetoes:
        reason_counts[f"doctrine:{d.get('reason', 'unknown')}"] += 1

    if not reason_counts:
        print("  No vetoes in period.")
        return

    total = sum(reason_counts.values())
    print(f"  {'Reason':<40} {'Count':>6} {'%':>7}")
    print(f"  {'-'*40} {'-'*6} {'-'*7}")
    for reason, count in reason_counts.most_common(15):
        pct = count / total * 100
        print(f"  {reason:<40} {count:>6} {pct:>6.1f}%")


def section_regime_context(sentinel: dict) -> None:
    print("\n## 7. Regime Context (Current)\n")
    print(f"  Primary:    {sentinel.get('primary_regime', 'unknown')}")
    print(f"  Secondary:  {sentinel.get('secondary_regime', 'unknown')}")
    print(f"  Crisis:     {sentinel.get('crisis_flag', False)}")
    probs = sentinel.get("smoothed_probs") or sentinel.get("regime_probs") or {}
    if probs:
        print(f"\n  Regime Probabilities:")
        for regime, prob in sorted(probs.items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 40)
            print(f"    {regime:<15} {prob:>6.1%} {bar}")


def section_config_hash() -> None:
    print("\n## 11. Config Hash (Discipline Verification)\n")
    import hashlib
    for path in CONFIG_FILES:
        if path.exists():
            h = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            print(f"  {str(path):<45} sha256:{h}...")
        else:
            print(f"  {str(path):<45} MISSING")


def section_operational_health(diagnostics: dict) -> None:
    print("\n## 9. Operational Health\n")
    rt = diagnostics.get("runtime_diagnostics", {})
    veto_counters = rt.get("veto_counters", {})
    liveness = rt.get("liveness", {})
    print(f"  Engine version:     {diagnostics.get('engine_version', 'unknown')}")
    print(f"  Total signals:      {veto_counters.get('total_signals', 0)}")
    print(f"  Total orders:       {veto_counters.get('total_orders', 0)}")
    print(f"  Total vetoes:       {veto_counters.get('total_vetoes', 0)}")
    print(f"  Last signal:        {veto_counters.get('last_signal_ts', 'never')}")
    print(f"  Last order:         {veto_counters.get('last_order_ts', 'never')}")
    idle = liveness.get("details", {})
    missing = liveness.get("missing", [])
    if missing:
        print(f"  Liveness missing:   {', '.join(missing)}")
    print(f"  Score decomp log:   {'✓ exists' if SCORE_DECOMP.exists() else '✗ missing'}")
    print(f"  Sizing snapshots:   {'✓ exists' if SIZING_SNAPSHOTS.exists() else '✗ missing'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="14-Day Performance Brief Generator")
    parser.add_argument("--since", help="Start date (YYYY-MM-DD)", default=None)
    parser.add_argument("--until", help="End date (YYYY-MM-DD)", default=None)
    args = parser.parse_args()

    since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc) if args.since else None
    until = datetime.fromisoformat(args.until).replace(tzinfo=timezone.utc) if args.until else None

    # Load state files
    nav_state = _load_json(NAV_STATE)
    risk_snap = _load_json(RISK_SNAPSHOT)
    sentinel = _load_json(SENTINEL_X)
    diagnostics = _load_json(DIAGNOSTICS)

    # Load episode ledger
    ledger = _load_json(EPISODE_LEDGER)
    episodes = ledger.get("episodes_v2") or ledger.get("episodes", [])
    if since:
        episodes = [e for e in episodes if (e.get("entry_ts") or "") >= since.isoformat()]
    if until:
        episodes = [e for e in episodes if (e.get("entry_ts") or "") <= until.isoformat()]

    # Load execution logs
    attempts = _load_jsonl(ORDERS_ATTEMPTED, since)
    vetoes = _load_jsonl(RISK_VETOES, since)
    decomp = _load_jsonl(SCORE_DECOMP, since)
    doctrine = _load_jsonl(DOCTRINE_EVENTS, since)
    sizing = _load_jsonl(SIZING_SNAPSHOTS, since)

    # Header
    since_str = args.since or "inception"
    until_str = args.until or "now"
    print("=" * 65)
    print(f"  14-DAY PERFORMANCE BRIEF")
    print(f"  Period: {since_str} → {until_str}")
    print(f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  System: v7.9-S1 (post-contraction)")
    print("=" * 65)

    # Sections
    section_capital(nav_state, risk_snap)
    section_trade_activity(episodes, attempts, vetoes)
    section_conviction_distribution(sizing)
    section_pnl_by_band(episodes)
    section_hybrid_dispersion(decomp)
    section_strategy_attribution(episodes)
    section_regime_context(sentinel)
    section_veto_analysis(vetoes, doctrine)
    section_operational_health(diagnostics)
    section_config_hash()

    print("\n" + "=" * 65)
    print("  END OF BRIEF")
    print("=" * 65)


if __name__ == "__main__":
    main()
