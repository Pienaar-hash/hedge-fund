"""
Daily Summary Generator — Plain Text Portfolio Snapshot

Produces a single-string summary of portfolio state for messaging,
logging, or display. No Telegram, no scheduling, no webhooks.

Data sources:
  - logs/state/nav_state.json          → NAV
  - logs/state/positions_state.json    → Open positions count
  - logs/state/episode_ledger.json     → Episode stats, calibration count
  - logs/nav_log.json                  → NAV-delta PnL (24h)
  - config/runtime.yaml                → Calibration window config
  - logs/state/risk_snapshot.json      → Max drawdown
  - logs/state/sentinel_x.json        → Current regime
  - logs/state/binary_lab_state.json   → Binary sleeve status
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


# ── Paths ───────────────────────────────────────────────────────────
_NAV_STATE = Path("logs/state/nav_state.json")
_POSITIONS_STATE = Path("logs/state/positions_state.json")
_EPISODE_LEDGER = Path("logs/state/episode_ledger.json")
_NAV_LOG = Path("logs/nav_log.json")
_RUNTIME_YAML = Path("config/runtime.yaml")
_RISK_SNAPSHOT = Path("logs/state/risk_snapshot.json")
_SENTINEL_X = Path("logs/state/sentinel_x.json")
_BINARY_LAB = Path("logs/state/binary_lab_state.json")


def _safe_json(path: Path) -> Dict[str, Any]:
    """Load JSON file, return empty dict on any failure."""
    try:
        if not path.exists():
            return {}
        return json.loads(path.read_text())
    except Exception:
        return {}


def _safe_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML file, return empty dict on any failure."""
    try:
        if not path.exists():
            return {}
        import yaml
        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


def _nav_pnl_24h() -> float:
    """Compute 24h PnL from nav_log.json (NAV delta method)."""
    try:
        if not _NAV_LOG.exists():
            return 0.0
        entries = json.loads(_NAV_LOG.read_text())
        if not entries or len(entries) < 2:
            return 0.0
        latest = entries[-1]
        nav_now = float(latest.get("nav", 0))
        t_now = float(latest.get("t", time.time()))
        target = t_now - 86400
        best = None
        best_dist = float("inf")
        for e in entries:
            t = e.get("t")
            if t is None:
                continue
            dist = abs(t - target)
            if dist < best_dist:
                best = e
                best_dist = dist
        if best is None:
            return 0.0
        return round(nav_now - float(best.get("nav", nav_now)), 2)
    except Exception:
        return 0.0


def generate_daily_summary(now: Optional[datetime] = None) -> str:
    """Generate a plain-text daily portfolio summary.

    Returns a multi-line string suitable for logging, messaging,
    or display. Pure read-only — never writes state.

    Parameters
    ----------
    now : datetime, optional
        Override for current timestamp (useful for testing).
        Defaults to UTC now.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    lines: list[str] = []
    lines.append(f"═══ Daily Summary — {now.strftime('%Y-%m-%d %H:%M UTC')} ═══")
    lines.append("")

    # ── NAV ─────────────────────────────────────────────────────
    nav_state = _safe_json(_NAV_STATE)
    nav_usd = float(
        nav_state.get("nav_usd")
        or nav_state.get("nav")
        or 0
    )
    nav_mode = nav_state.get("nav_mode", "unknown")
    lines.append(f"Portfolio NAV:    ${nav_usd:,.2f}  ({nav_mode})")

    # ── 24h PnL (NAV delta) ────────────────────────────────────
    pnl_24h = _nav_pnl_24h()
    pnl_sign = "+" if pnl_24h >= 0 else ""
    pnl_pct = (pnl_24h / nav_usd * 100) if nav_usd > 0 else 0.0
    lines.append(f"24h PnL:          {pnl_sign}${pnl_24h:,.2f}  ({pnl_sign}{pnl_pct:.2f}%)")
    lines.append("")

    # ── Futures Positions ──────────────────────────────────────
    pos_state = _safe_json(_POSITIONS_STATE)
    positions = pos_state.get("positions", [])
    open_count = len(positions)
    unrealized = sum(float(p.get("unrealizedProfit", 0) or 0) for p in positions)
    lines.append(f"Open Positions:   {open_count}")
    if open_count > 0:
        lines.append(f"Unrealized PnL:   ${unrealized:,.2f}")
    lines.append("")

    # ── Episode Ledger ─────────────────────────────────────────
    ledger = _safe_json(_EPISODE_LEDGER)
    stats = ledger.get("stats", {})
    episode_count = ledger.get("episode_count", 0)
    total_net_pnl = float(stats.get("total_net_pnl", 0) or 0)
    win_rate = float(stats.get("win_rate", 0) or 0)
    winners = int(stats.get("winners", 0) or 0)
    losers = int(stats.get("losers", 0) or 0)
    lines.append(f"Episodes:         {episode_count}  (W:{winners} / L:{losers})")
    lines.append(f"Win Rate:         {win_rate:.1f}%")
    lines.append(f"Realised PnL:     ${total_net_pnl:,.2f}")
    lines.append("")

    # ── Risk / Drawdown ────────────────────────────────────────
    risk = _safe_json(_RISK_SNAPSHOT)
    max_dd = float(risk.get("max_drawdown_pct", 0) or risk.get("drawdown_pct", 0) or 0)
    lines.append(f"Max Drawdown:     {max_dd:.2f}%")

    # ── Regime ─────────────────────────────────────────────────
    sentinel = _safe_json(_SENTINEL_X)
    regime = sentinel.get("regime") or sentinel.get("current_regime") or "UNKNOWN"
    confidence = float(sentinel.get("confidence", 0) or 0)
    lines.append(f"Regime:           {regime}  (conf: {confidence:.0%})")
    lines.append("")

    # ── Calibration Window ─────────────────────────────────────
    runtime = _safe_yaml(_RUNTIME_YAML)
    cw = runtime.get("calibration_window", {})
    if cw.get("enabled"):
        cap = cw.get("episode_cap", "?")
        dd_kill = cw.get("drawdown_kill_pct", 0)
        start_ts = cw.get("start_ts", "?")
        # Count calibration episodes from ledger
        cal_episodes = _count_calibration_episodes(ledger, str(start_ts))
        lines.append(f"Calibration:      ACTIVE — {cal_episodes}/{cap} episodes")
        lines.append(f"  DD Kill:        {float(dd_kill) * 100:.1f}%")
        lines.append(f"  Started:        {start_ts}")
    else:
        lines.append("Calibration:      INACTIVE")
    lines.append("")

    # ── Binary Sleeve ──────────────────────────────────────────
    binary = _safe_json(_BINARY_LAB)
    binary_enabled = binary.get("enabled", False)
    binary_pnl = float(binary.get("pnl", 0) or 0)
    if binary_enabled:
        lines.append(f"Binary Sleeve:    ENABLED — PnL ${binary_pnl:,.2f}")
    else:
        lines.append("Binary Sleeve:    DISABLED")

    lines.append("")
    lines.append("═══════════════════════════════════════════════")

    return "\n".join(lines)


def _count_calibration_episodes(
    ledger: Dict[str, Any],
    start_ts: str,
) -> int:
    """Count episodes closed since calibration start_ts."""
    try:
        from datetime import datetime as dt
        cleaned = start_ts.replace("Z", "+00:00")
        start_dt = dt.fromisoformat(cleaned)
    except (ValueError, TypeError):
        return 0

    episodes = ledger.get("episodes", [])
    count = 0
    for ep in episodes:
        exit_ts = ep.get("exit_ts", "")
        if not exit_ts:
            continue
        try:
            cleaned = exit_ts.replace("Z", "+00:00")
            exit_dt = datetime.fromisoformat(cleaned)
            if exit_dt >= start_dt:
                count += 1
        except (ValueError, TypeError):
            continue
    return count


if __name__ == "__main__":
    print(generate_daily_summary())
