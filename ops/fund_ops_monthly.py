"""
Fund-Ops Monthly Template Generator — Section Content Blocks

Produces labeled [SECTION_N] content blocks for the monthly fund-ops
package. Pure read-only — never writes state.

Rules:
  - No headings. No section titles. Only content blocks.
  - Each block labeled exactly [SECTION_1] .. [SECTION_7].
  - If data unavailable for a field, write DATA_UNAVAILABLE.
  - No commentary. No conclusions. No persuasion.

Data sources:
  - logs/state/nav_state.json              → Current NAV
  - logs/nav_log.json                      → NAV history (start/end)
  - logs/state/episode_ledger.json         → Trade activity, conviction
  - logs/state/risk_snapshot.json          → Drawdown, risk caps
  - logs/state/sentinel_x.json            → Regime distribution
  - logs/state/binary_lab_state.json       → Binary lab status
  - logs/state/activation_window_state.json → Cert window
  - logs/execution/risk_vetoes.jsonl       → Veto attribution
  - config/runtime.yaml                    → Calibration config

Usage:
  PYTHONPATH=. python ops/fund_ops_monthly.py                # stdout
  PYTHONPATH=. python ops/fund_ops_monthly.py --days 30      # custom window
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Paths ───────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
_NAV_STATE = _ROOT / "logs" / "state" / "nav_state.json"
_NAV_LOG = _ROOT / "logs" / "nav_log.json"
_EPISODE_LEDGER = _ROOT / "logs" / "state" / "episode_ledger.json"
_RISK_SNAPSHOT = _ROOT / "logs" / "state" / "risk_snapshot.json"
_SENTINEL_X = _ROOT / "logs" / "state" / "sentinel_x.json"
_BINARY_LAB = _ROOT / "logs" / "state" / "binary_lab_state.json"
_AW_STATE = _ROOT / "logs" / "state" / "activation_window_state.json"
_RISK_VETOES = _ROOT / "logs" / "execution" / "risk_vetoes.jsonl"
_RUNTIME_YAML = _ROOT / "config" / "runtime.yaml"


def _safe_json(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        return json.loads(path.read_text())
    except Exception:
        return {}


def _safe_yaml(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        import yaml
        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


def _safe_jsonl(path: Path) -> List[Dict[str, Any]]:
    try:
        if not path.exists():
            return []
        entries = []
        for line in path.read_text().strip().split("\n"):
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
        return entries
    except Exception:
        return []


def _nav_at_offset(entries: List[Dict], offset_seconds: float) -> Optional[float]:
    """Find NAV closest to (now - offset_seconds)."""
    if not entries:
        return None
    target = time.time() - offset_seconds
    best = None
    best_dist = float("inf")
    for e in entries:
        t = e.get("t", 0)
        dist = abs(t - target)
        if dist < best_dist:
            best = e
            best_dist = dist
    return float(best["nav"]) if best and "nav" in best else None


# ── Section builders ────────────────────────────────────────────────


def section_1_capital_state(days: int = 30) -> str:
    """[SECTION_1] Capital State."""
    nav_state = _safe_json(_NAV_STATE)
    nav_now = float(
        nav_state.get("total_equity")
        or nav_state.get("nav_usd")
        or nav_state.get("nav")
        or 0
    )

    nav_entries = []
    try:
        if _NAV_LOG.exists():
            nav_entries = json.loads(_NAV_LOG.read_text())
    except Exception:
        pass

    nav_start = _nav_at_offset(nav_entries, days * 86400)

    risk = _safe_json(_RISK_SNAPSHOT)
    dd_frac = float(risk.get("portfolio_dd_pct") or risk.get("dd_frac") or 0)
    max_dd = dd_frac * 100 if dd_frac < 1 else dd_frac

    # Risk cap breach counts from vetoes
    vetoes = _safe_jsonl(_RISK_VETOES)
    cutoff = time.time() - days * 86400
    period_vetoes = []
    for v in vetoes:
        ts = v.get("ts", 0)
        try:
            if float(ts) > cutoff:
                period_vetoes.append(v)
        except (ValueError, TypeError):
            pass
    breach_reasons = Counter(v.get("veto_reason", "unknown") for v in period_vetoes)
    daily_breaches = breach_reasons.get("daily_loss", 0)
    weekly_breaches = breach_reasons.get("portfolio_dd_circuit", 0)

    lines = ["[SECTION_1]"]
    lines.append(f"Starting NAV:       ${nav_start:,.2f}" if nav_start else "Starting NAV:       DATA_UNAVAILABLE")
    lines.append(f"Ending NAV:         ${nav_now:,.2f}" if nav_now else "Ending NAV:         DATA_UNAVAILABLE")

    if nav_start and nav_now and nav_start > 0:
        net_return = (nav_now - nav_start) / nav_start * 100
        lines.append(f"Net Return:         {net_return:+.2f}%")
    else:
        lines.append("Net Return:         DATA_UNAVAILABLE")

    lines.append(f"Max Drawdown:       {max_dd:.2f}%")
    lines.append(f"Risk Cap Breaches:  daily_loss={daily_breaches}, portfolio_dd_circuit={weekly_breaches}")

    return "\n".join(lines)


def section_2_trade_activity(days: int = 30) -> str:
    """[SECTION_2] Trade Activity."""
    ledger = _safe_json(_EPISODE_LEDGER)
    stats = ledger.get("stats", {})
    episodes = ledger.get("episodes", [])
    episode_count = ledger.get("episode_count", len(episodes))

    winners = int(stats.get("winners", 0) or 0)
    losers = int(stats.get("losers", 0) or 0)
    win_rate = float(stats.get("win_rate", 0) or 0)
    total_net_pnl = float(stats.get("total_net_pnl", 0) or 0)

    # Daily trade frequency
    by_day: Dict[str, int] = defaultdict(int)
    for ep in episodes:
        ts = ep.get("entry_ts", "")
        if ts:
            by_day[ts[:10]] += 1
    n_days = len(by_day)
    avg_trades_day = episode_count / n_days if n_days > 0 else 0

    # Acceptance rate from risk vetoes vs total signals
    vetoes = _safe_jsonl(_RISK_VETOES)
    total_vetoes = len(vetoes)
    total_signals = episode_count + total_vetoes
    acceptance_rate = (episode_count / total_signals * 100) if total_signals > 0 else 0

    # Conviction distribution
    conviction_counter: Counter[str] = Counter()
    for ep in episodes:
        c = ep.get("conviction") or ep.get("conviction_band") or "untagged"
        conviction_counter[str(c)] += 1

    # Hybrid score dispersion
    hybrid_scores = []
    for ep in episodes:
        hs = ep.get("hybrid_score") or ep.get("score")
        if hs is not None:
            try:
                hybrid_scores.append(float(hs))
            except (ValueError, TypeError):
                pass

    lines = ["[SECTION_2]"]
    lines.append(f"Episodes Closed:    {episode_count}")
    lines.append(f"Win Rate:           {win_rate:.1f}% (W:{winners} / L:{losers})")
    lines.append(f"Realised PnL:       ${total_net_pnl:,.2f}")
    lines.append(f"Avg Trades/Day:     {avg_trades_day:.1f}")
    lines.append(f"Acceptance Rate:    {acceptance_rate:.1f}% ({episode_count} accepted / {total_vetoes} vetoed)")

    lines.append("Conviction Distribution:")
    for band, count in conviction_counter.most_common():
        pct = count / episode_count * 100 if episode_count > 0 else 0
        label = band if band else "untagged"
        lines.append(f"  {label}: {count} ({pct:.1f}%)")

    if hybrid_scores:
        import statistics
        lines.append("Hybrid Score Dispersion:")
        lines.append(f"  mean={statistics.mean(hybrid_scores):.3f}, "
                      f"median={statistics.median(hybrid_scores):.3f}, "
                      f"stdev={statistics.stdev(hybrid_scores):.3f}" if len(hybrid_scores) > 1
                      else f"  mean={statistics.mean(hybrid_scores):.3f}, n={len(hybrid_scores)}")
    else:
        lines.append("Hybrid Score Dispersion: DATA_UNAVAILABLE")

    return "\n".join(lines)


def section_3_regime_context() -> str:
    """[SECTION_3] Regime Context."""
    sentinel = _safe_json(_SENTINEL_X)
    primary = sentinel.get("primary_regime", "DATA_UNAVAILABLE")
    secondary = sentinel.get("secondary_regime", "")
    probs = sentinel.get("smoothed_probs") or sentinel.get("regime_probs") or {}
    history = sentinel.get("history_meta", {})
    consecutive = history.get("consecutive_count", 0)
    last_n = history.get("last_n_labels", [])

    lines = ["[SECTION_3]"]
    lines.append("Regime Distribution:")
    for regime, prob in sorted(probs.items(), key=lambda x: -x[1]):
        lines.append(f"  {regime}: {prob:.1%}")

    if primary != "DATA_UNAVAILABLE":
        obs = f"Primary regime {primary} for {consecutive} consecutive cycles."
        if secondary and secondary != primary:
            obs += f" Secondary: {secondary}."
        lines.append(obs)
    else:
        lines.append("DATA_UNAVAILABLE")

    return "\n".join(lines)


def section_4_risk_discipline() -> str:
    """[SECTION_4] Risk & Discipline."""
    aw = _safe_json(_AW_STATE)
    binary = _safe_json(_BINARY_LAB)
    risk = _safe_json(_RISK_SNAPSHOT)

    manifest_ok = aw.get("manifest_intact", "DATA_UNAVAILABLE")
    config_ok = aw.get("config_intact", "DATA_UNAVAILABLE")
    boot_manifest = aw.get("boot_manifest_hash", "DATA_UNAVAILABLE")
    boot_config = aw.get("boot_config_hash", "DATA_UNAVAILABLE")
    binary_freeze = binary.get("freeze_intact", "DATA_UNAVAILABLE")
    binary_hash = binary.get("config_hash", "DATA_UNAVAILABLE")
    circuit = risk.get("circuit_breaker", {})
    circuit_active = circuit.get("active", False)

    lines = ["[SECTION_4]"]
    lines.append(f"Manifest Integrity: {'INTACT' if manifest_ok is True else 'DRIFT' if manifest_ok is False else manifest_ok}")
    lines.append(f"Config Integrity:   {'INTACT' if config_ok is True else 'DRIFT' if config_ok is False else config_ok}")
    lines.append(f"Boot Manifest Hash: {boot_manifest}")
    lines.append(f"Boot Config Hash:   {boot_config}")
    lines.append(f"Binary Freeze:      {'INTACT' if binary_freeze is True else 'BROKEN' if binary_freeze is False else binary_freeze}")
    lines.append(f"Binary Config Hash: {binary_hash}")
    lines.append(f"Circuit Breaker:    {'TRIGGERED' if circuit_active else 'OK'}")

    return "\n".join(lines)


def section_5_binary_lab() -> str:
    """[SECTION_5] Binary Lab Status."""
    binary = _safe_json(_BINARY_LAB)
    if not binary:
        return "[SECTION_5]\nNot deployed."

    status = binary.get("status", "UNKNOWN")
    mode = binary.get("mode", "")
    day = binary.get("day", 0)
    day_total = binary.get("day_total", 30)
    metrics = binary.get("metrics", {})
    capital = binary.get("capital", {})
    pnl = float(capital.get("pnl_usd", 0) or 0)
    nav_usd = float(capital.get("current_nav_usd", 0) or 0)
    total_trades = metrics.get("total_trades", 0)
    wins = metrics.get("wins", 0)
    losses = metrics.get("losses", 0)
    win_rate = metrics.get("win_rate")
    kill_line = binary.get("kill_line", {})
    kill_breached = kill_line.get("breached", False)
    kill_distance = float(kill_line.get("distance_usd", 0) or 0)
    freeze = binary.get("freeze_intact", "DATA_UNAVAILABLE")
    violations = binary.get("rule_violations", 0)
    term_reason = binary.get("termination_reason")

    lines = ["[SECTION_5]"]
    lines.append(f"Status:             {status} ({mode})" if mode else f"Status:             {status}")
    lines.append(f"Day:                {day}/{day_total}")
    lines.append(f"Capital Allocated:  ${nav_usd:,.2f}")
    lines.append(f"PnL:                ${pnl:,.2f}")
    lines.append(f"Total Trades:       {total_trades} (W:{wins} / L:{losses})")
    lines.append(f"Win Rate:           {win_rate:.1f}%" if win_rate is not None else "Win Rate:           N/A (no trades)")
    lines.append(f"Kill Line:          {'BREACHED' if kill_breached else 'OK'} (distance: ${kill_distance:,.2f})")
    lines.append(f"Freeze Intact:      {'YES' if freeze is True else 'NO' if freeze is False else freeze}")
    lines.append(f"Rule Violations:    {violations}")
    if term_reason:
        lines.append(f"Termination:        {term_reason}")

    return "\n".join(lines)


def section_6_structural(days: int = 30) -> str:
    """[SECTION_6] Structural Developments — from git log."""
    try:
        result = subprocess.run(
            ["git", "--no-pager", "log", "--oneline",
             f"--since={days} days ago", "--until=now"],
            capture_output=True,
            text=True,
            cwd=str(_ROOT),
            timeout=10,
        )
        commits = result.stdout.strip().split("\n") if result.stdout.strip() else []
    except Exception:
        commits = []

    if not commits:
        return "[SECTION_6]\nDATA_UNAVAILABLE"

    # Filter for structural commits (feat:, fix:, guard:, overhaul:, chore:)
    structural = []
    for c in commits:
        # Skip merge commits and minor ops
        parts = c.split(" ", 1)
        if len(parts) < 2:
            continue
        msg = parts[1]
        if any(msg.startswith(p) for p in ("feat:", "fix:", "guard:", "overhaul:", "chore:", "Phase")):
            structural.append(f"- {msg}")

    if not structural:
        return "[SECTION_6]\nNo structural changes in period."

    lines = ["[SECTION_6]"]
    lines.extend(structural)
    return "\n".join(lines)


def section_7_measurement_focus() -> str:
    """[SECTION_7] Measurement Focus (Next 30 Days)."""
    aw = _safe_json(_AW_STATE)
    binary = _safe_json(_BINARY_LAB)

    lines = ["[SECTION_7]"]

    # Derive from current system state
    if aw.get("active"):
        remaining = aw.get("remaining_days", 0)
        lines.append(f"- Complete 14-day certification window ({remaining:.0f}d remaining)")
        lines.append("- Monitor manifest + config integrity daily")
        lines.append("- Track drawdown vs 5% kill line")

    if binary.get("status") == "ACTIVE" and binary.get("mode") == "SHADOW":
        day = binary.get("day", 0)
        day_total = binary.get("day_total", 30)
        lines.append(f"- Binary Lab shadow observation continues (day {day}/{day_total})")
        lines.append("- Evaluate shadow PnL accuracy vs live execution")

    lines.append("- Measure doctrine acceptance rate trend")
    lines.append("- Track regime stability and conviction distribution shift")
    lines.append("")
    lines.append("No parameter changes scheduled. Observation only.")

    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────


def generate_monthly_report(days: int = 30) -> str:
    """Generate all 7 section blocks for the fund-ops monthly template."""
    sections = [
        section_1_capital_state(days=days),
        section_2_trade_activity(days=days),
        section_3_regime_context(),
        section_4_risk_discipline(),
        section_5_binary_lab(),
        section_6_structural(days=days),
        section_7_measurement_focus(),
    ]
    return "\n\n".join(sections)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fund-Ops Monthly Template Generator")
    parser.add_argument("--days", type=int, default=30, help="Lookback window in days (default: 30)")
    args = parser.parse_args()
    print(generate_monthly_report(days=args.days))
