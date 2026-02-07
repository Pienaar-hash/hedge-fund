"""
Episode Ledger — Completed Trade Cycle Tracking (v7.8)

This module derives completed trading episodes from execution logs.
It is READ-ONLY observability — it does NOT influence execution decisions.

Purpose:
  - Track completed trade cycles (entry → exit)
  - Prevent "snapshot blindness" (positions_state == [] ≠ no trading)
  - Provide historical participation visibility
  - Support postmortem analysis

An episode is:
  - A position that was opened AND closed
  - Identified by symbol + position side + entry window
  - Contains: entry/exit times, regimes, PnL, fees, exit reason, duration
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
from collections import defaultdict

from dateutil import parser as dateparser

logger = logging.getLogger(__name__)

# State file path
EPISODE_LEDGER_PATH = Path("logs/state/episode_ledger.json")
EXECUTION_LOG_PATH = Path("logs/execution/orders_executed.jsonl")
DOCTRINE_LOG_PATH = Path("logs/doctrine_events.jsonl")


@dataclass
class Episode:
    """A completed trade cycle: entry → exit."""
    
    episode_id: str
    symbol: str
    side: str  # LONG or SHORT
    
    # Timing
    entry_ts: str
    exit_ts: str
    duration_hours: float
    
    # Execution
    entry_fills: int
    exit_fills: int
    entry_notional: float
    exit_notional: float
    total_qty: float
    avg_entry_price: float
    avg_exit_price: float
    
    # PnL
    gross_pnl: float
    fees: float
    net_pnl: float
    
    # Context
    regime_at_entry: str
    regime_at_exit: str
    exit_reason: str  # tp, sl, thesis, regime_flip, unknown
    
    # Metadata
    strategy: str = "unknown"
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EpisodeLedger:
    """Container for all completed episodes."""
    
    episodes: list[Episode] = field(default_factory=list)
    last_rebuild_ts: str = ""
    stats: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "episodes": [e.to_dict() for e in self.episodes],
            "last_rebuild_ts": self.last_rebuild_ts,
            "stats": self.stats,
            "episode_count": len(self.episodes),
        }


def _parse_ts(ts_val) -> Optional[datetime]:
    """Parse timestamp from various formats."""
    if ts_val is None:
        return None
    try:
        if isinstance(ts_val, (int, float)):
            return datetime.fromtimestamp(ts_val, tz=timezone.utc)
        return dateparser.parse(str(ts_val))
    except Exception:
        return None


def _load_execution_log() -> list[dict]:
    """Load all order fill events from execution log."""
    if not EXECUTION_LOG_PATH.exists():
        return []
    
    fills = []
    with open(EXECUTION_LOG_PATH, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                if event.get("event_type") == "order_fill":
                    fills.append(event)
            except json.JSONDecodeError:
                continue
    return fills


def _load_doctrine_events() -> dict[str, str]:
    """Load regime at time of doctrine events (for regime context)."""
    # This is a simplified lookup - in production would need more sophisticated
    # regime tracking. For now, return empty dict and use 'unknown'.
    return {}


def _extract_exit_reason(fill: dict) -> str:
    """Extract exit reason from fill metadata.
    
    Priority order:
    1. Explicit metadata.exit.reason (from exit_scanner)
    2. Strategy name hints (tp_exit, sl_exit, etc.)
    3. Intent source field (auto_reduce = position_flip)
    4. Attempt ID suffix (_reduce = flip)
    5. Unknown fallback
    """
    meta = fill.get("metadata", {})
    exit_info = meta.get("exit", {})
    reason = exit_info.get("reason", "")
    
    # Normalize to lowercase for matching
    reason_lower = reason.lower() if reason else ""
    
    # 1. Explicit exit reason from exit_scanner
    if reason_lower in ("tp", "sl", "thesis", "regime_flip"):
        return reason_lower
    
    # 2. Check for strategy hints
    strategy = str(meta.get("strategy", "") or "").lower()
    if "exit" in strategy:
        if "tp" in strategy:
            return "tp"
        if "sl" in strategy:
            return "sl"
    
    # 3. Check intent source field (screener auto_reduce = position flip)
    source = str(fill.get("source", "") or "").lower()
    if source == "auto_reduce":
        return "position_flip"
    
    # 4. Check attempt_id suffix (flip reduce operations)
    attempt_id = str(fill.get("attempt_id", "") or "")
    if attempt_id.endswith("_reduce"):
        return "position_flip"
    
    # 5. Check if this is a reduceOnly close triggered by new signal (signal flip)
    if fill.get("reduceOnly") or fill.get("reduce_only"):
        # reduceOnly without explicit reason = signal-driven close
        return "signal_close"
    
    return "unknown"


def _extract_strategy(fill: dict) -> str:
    """Extract strategy name from fill metadata."""
    meta = fill.get("metadata", {})
    return meta.get("strategy", "unknown")


def _compute_metadata_pnl(
    fills: list[dict],
    since_date: Optional[str],
    until_date: Optional[str],
) -> dict:
    """
    Compute realized PnL from exit fill metadata.
    
    This is an independent estimator that uses entry_price stored in exit metadata.
    More robust than episode matching for partial fills and scaling.
    
    Returns dict with: exits, gross_pnl, fees, net_pnl
    """
    realized_pnl = 0.0
    total_fees = 0.0
    exit_count = 0
    exits_with_entry_price = 0
    
    for f in fills:
        # Only process exits (reduceOnly fills)
        if not f.get("reduceOnly"):
            continue
        
        # Check if exit is in window
        ts = _parse_ts(f.get("ts"))
        if ts:
            date_str = ts.strftime("%Y-%m-%d")
            if since_date and date_str < since_date:
                continue
            if until_date and date_str > until_date:
                continue
        
        exit_count += 1
        total_fees += float(f.get("fee_total", 0) or 0)
        
        # Extract entry price from metadata
        meta = f.get("metadata", {})
        exit_info = meta.get("exit", {}) or meta.get("tp_sl", {}) or {}
        entry_price = exit_info.get("entry_price") or meta.get("entry_price")
        
        if entry_price:
            exits_with_entry_price += 1
            exit_price = float(f.get("avgPrice", 0) or 0)
            qty = float(f.get("executedQty", 0) or 0)
            pos_side = f.get("positionSide", "")
            
            if pos_side == "LONG":
                pnl = (exit_price - float(entry_price)) * qty
            else:  # SHORT
                pnl = (float(entry_price) - exit_price) * qty
            realized_pnl += pnl
    
    return {
        "exits": exit_count,
        "exits_with_entry_price": exits_with_entry_price,
        "gross_pnl": round(realized_pnl, 2),
        "fees": round(total_fees, 2),
        "net_pnl": round(realized_pnl - total_fees, 2),
    }


def build_episode_ledger(
    since_date: Optional[str] = None,
    until_date: Optional[str] = None,
    lookback_days: int = 7,
) -> EpisodeLedger:
    """
    Rebuild episode ledger from execution logs.
    
    Algorithm:
      1. Load fills (with lookback buffer to capture entries before window)
      2. Group by (symbol, positionSide)
      3. Match entries to exits by cumulative quantity
      4. Create episode for each completed cycle
      5. Filter episodes by EXIT timestamp in [since_date, until_date]
    
    Window semantics: "episodes ending in window" — an episode is included
    if its exit_ts falls within the window, regardless of when entry occurred.
    """
    fills = _load_execution_log()
    
    if not fills:
        return EpisodeLedger(
            last_rebuild_ts=datetime.now(timezone.utc).isoformat(),
            stats={"total_fills": 0, "episodes_found": 0},
        )
    
    # Compute lookback date for fill filtering (entry could be before window)
    fill_since_date: Optional[str] = None
    if since_date:
        try:
            since_dt = datetime.strptime(since_date, "%Y-%m-%d")
            lookback_dt = since_dt - timedelta(days=lookback_days)
            fill_since_date = lookback_dt.strftime("%Y-%m-%d")
        except ValueError:
            fill_since_date = since_date  # fallback to exact date
    
    # Pre-filter fills with lookback buffer (for performance, not semantics)
    if fill_since_date or until_date:
        filtered = []
        for f in fills:
            ts = _parse_ts(f.get("ts"))
            if ts is None:
                continue
            date_str = ts.strftime("%Y-%m-%d")
            if fill_since_date and date_str < fill_since_date:
                continue
            if until_date and date_str > until_date:
                continue
            filtered.append(f)
        fills = filtered
    
    # Group fills by (symbol, positionSide)
    # positionSide: LONG = bought to open, sold to close
    #               SHORT = sold to open, bought to close
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    
    for f in fills:
        symbol = f.get("symbol", "")
        pos_side = f.get("positionSide", "")
        if not symbol or not pos_side:
            continue
        groups[(symbol, pos_side)].append(f)
    
    episodes = []
    episode_counter = 0
    
    for (symbol, pos_side), group_fills in groups.items():
        # Sort by timestamp
        group_fills.sort(key=lambda x: x.get("ts", ""))
        
        # Track position state
        open_qty = 0.0
        entry_fills: list[dict] = []
        
        for f in group_fills:
            qty = float(f.get("executedQty", 0))
            is_reduce = f.get("reduceOnly", False)
            side = f.get("side", "")
            
            # Determine if this is entry or exit
            # LONG position: BUY = entry, SELL = exit
            # SHORT position: SELL = entry, BUY = exit
            if pos_side == "LONG":
                is_entry = (side == "BUY" and not is_reduce)
                is_exit = (side == "SELL" or is_reduce)
            else:  # SHORT
                is_entry = (side == "SELL" and not is_reduce)
                is_exit = (side == "BUY" or is_reduce)
            
            if is_entry:
                open_qty += qty
                entry_fills.append(f)
            
            elif is_exit and open_qty > 0:
                # This is an exit - check if it closes the position
                exit_qty = min(qty, open_qty)
                open_qty -= exit_qty
                
                # If position is now closed (or very close), create episode
                if open_qty < 0.0001 and entry_fills:
                    episode_counter += 1
                    
                    # Calculate entry stats
                    entry_notional = sum(
                        float(e.get("avgPrice", 0)) * float(e.get("executedQty", 0))
                        for e in entry_fills
                    )
                    entry_qty = sum(float(e.get("executedQty", 0)) for e in entry_fills)
                    entry_fees = sum(float(e.get("fee_total", 0)) for e in entry_fills)
                    avg_entry = entry_notional / entry_qty if entry_qty > 0 else 0
                    
                    # Exit stats (just this fill for now - simplified)
                    exit_price = float(f.get("avgPrice", 0))
                    exit_notional = exit_price * exit_qty
                    exit_fee = float(f.get("fee_total", 0))
                    
                    # PnL calculation
                    if pos_side == "LONG":
                        gross_pnl = (exit_price - avg_entry) * exit_qty
                    else:  # SHORT
                        gross_pnl = (avg_entry - exit_price) * exit_qty
                    
                    total_fees = entry_fees + exit_fee
                    net_pnl = gross_pnl - total_fees
                    
                    # Timing
                    entry_ts = entry_fills[0].get("ts", "")
                    exit_ts = f.get("ts", "")
                    
                    entry_dt = _parse_ts(entry_ts)
                    exit_dt = _parse_ts(exit_ts)
                    duration_hours = 0.0
                    if entry_dt and exit_dt:
                        duration_hours = (exit_dt - entry_dt).total_seconds() / 3600
                    
                    episode = Episode(
                        episode_id=f"EP_{episode_counter:04d}",
                        symbol=symbol,
                        side=pos_side,
                        entry_ts=entry_ts,
                        exit_ts=exit_ts,
                        duration_hours=round(duration_hours, 2),
                        entry_fills=len(entry_fills),
                        exit_fills=1,  # Simplified
                        entry_notional=round(entry_notional, 2),
                        exit_notional=round(exit_notional, 2),
                        total_qty=round(entry_qty, 6),
                        avg_entry_price=round(avg_entry, 4),
                        avg_exit_price=round(exit_price, 4),
                        gross_pnl=round(gross_pnl, 4),
                        fees=round(total_fees, 4),
                        net_pnl=round(net_pnl, 4),
                        regime_at_entry="unknown",  # Would need doctrine correlation
                        regime_at_exit="unknown",
                        exit_reason=_extract_exit_reason(f),
                        strategy=_extract_strategy(entry_fills[0]) if entry_fills else "unknown",
                    )
                    episodes.append(episode)
                    
                    # Reset for next episode
                    entry_fills = []
                    open_qty = 0.0
    
    # Filter episodes by EXIT timestamp in window (the canonical semantics)
    # "Episodes ending in window" — entry can be before, but exit must be in range
    if since_date or until_date:
        window_episodes = []
        for ep in episodes:
            exit_dt = _parse_ts(ep.exit_ts)
            if exit_dt is None:
                continue
            exit_date_str = exit_dt.strftime("%Y-%m-%d")
            if since_date and exit_date_str < since_date:
                continue
            if until_date and exit_date_str > until_date:
                continue
            window_episodes.append(ep)
        episodes = window_episodes
    
    # Calculate aggregate stats
    total_gross = sum(e.gross_pnl for e in episodes)
    total_fees = sum(e.fees for e in episodes)
    total_net = sum(e.net_pnl for e in episodes)
    
    winners = [e for e in episodes if e.net_pnl > 0]
    losers = [e for e in episodes if e.net_pnl < 0]
    
    # Compute max drawdown from cumulative PnL
    # Note: This is PnL-based drawdown (from trading peak to trough)
    # Not NAV-based drawdown (which would require starting capital)
    max_dd_pct = 0.0
    max_dd_abs = 0.0
    if episodes:
        # Sort by exit_ts for sequential equity calculation
        sorted_eps = sorted(episodes, key=lambda e: e.exit_ts)
        cumulative_pnl = 0.0
        peak_pnl = 0.0
        for ep in sorted_eps:
            cumulative_pnl += ep.net_pnl
            if cumulative_pnl > peak_pnl:
                peak_pnl = cumulative_pnl
            dd = peak_pnl - cumulative_pnl
            if dd > max_dd_abs:
                max_dd_abs = dd
        # Express as percentage of NAV-based equity peak
        # Fallback: if we have no NAV context, use 10_000 as reasonable base
        try:
            import json as _json
            from pathlib import Path as _Path
            _nav_path = _Path("logs/state/nav_state.json")
            _nav_base = float(_json.loads(_nav_path.read_text()).get("total_equity", 10000)) if _nav_path.exists() else 10000
        except Exception:
            _nav_base = 10000
        _equity_peak = _nav_base + peak_pnl
        if _equity_peak > 0 and max_dd_abs > 0:
            max_dd_pct = round((max_dd_abs / _equity_peak) * 100, 2)
        else:
            max_dd_pct = 0.0
    
    # Metadata-based PnL estimator (independent cross-check)
    # Uses entry_price from exit fill metadata — robust for partial fills
    meta_pnl = _compute_metadata_pnl(fills, since_date, until_date)
    
    stats = {
        "total_fills": len(fills),
        "episodes_found": len(episodes),
        "total_gross_pnl": round(total_gross, 2),
        "total_fees": round(total_fees, 2),
        "total_net_pnl": round(total_net, 2),
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": round(len(winners) / len(episodes) * 100, 1) if episodes else 0,
        "avg_duration_hours": round(sum(e.duration_hours for e in episodes) / len(episodes), 1) if episodes else 0,
        "max_drawdown_pct": max_dd_pct,
        "max_drawdown_abs": round(max_dd_abs, 2),
        "exit_reasons": {
            "tp": len([e for e in episodes if e.exit_reason == "tp"]),
            "sl": len([e for e in episodes if e.exit_reason == "sl"]),
            "thesis": len([e for e in episodes if e.exit_reason == "thesis"]),
            "regime_flip": len([e for e in episodes if e.exit_reason == "regime_flip"]),
            "unknown": len([e for e in episodes if e.exit_reason == "unknown"]),
        },
        # Cross-check from exit metadata (more robust for partial fills)
        "metadata_pnl": meta_pnl,
    }
    
    return EpisodeLedger(
        episodes=episodes,
        last_rebuild_ts=datetime.now(timezone.utc).isoformat(),
        stats=stats,
    )


def save_episode_ledger(ledger: EpisodeLedger) -> None:
    """Write episode ledger to state file."""
    EPISODE_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EPISODE_LEDGER_PATH, "w") as f:
        json.dump(ledger.to_dict(), f, indent=2)
    logger.info(f"Episode ledger saved: {len(ledger.episodes)} episodes")


def load_episode_ledger() -> Optional[EpisodeLedger]:
    """Load episode ledger from state file."""
    if not EPISODE_LEDGER_PATH.exists():
        return None
    
    try:
        with open(EPISODE_LEDGER_PATH, "r") as f:
            data = json.load(f)
        
        episodes = [Episode(**e) for e in data.get("episodes", [])]
        return EpisodeLedger(
            episodes=episodes,
            last_rebuild_ts=data.get("last_rebuild_ts", ""),
            stats=data.get("stats", {}),
        )
    except Exception as e:
        logger.error(f"Failed to load episode ledger: {e}")
        return None


def rebuild_and_save(
    since_date: Optional[str] = None,
    until_date: Optional[str] = None,
) -> EpisodeLedger:
    """Convenience function: rebuild and save ledger."""
    ledger = build_episode_ledger(since_date, until_date)
    save_episode_ledger(ledger)
    return ledger


def print_ledger_summary(ledger: EpisodeLedger) -> None:
    """Print human-readable summary of episode ledger."""
    stats = ledger.stats
    print("=" * 70)
    print("EPISODE LEDGER SUMMARY")
    print("=" * 70)
    print(f"Last Rebuild:     {ledger.last_rebuild_ts}")
    print(f"Total Episodes:   {stats.get('episodes_found', 0)}")
    print(f"Total Fills:      {stats.get('total_fills', 0)}")
    print()
    print("PnL BREAKDOWN")
    print("-" * 40)
    print(f"  Gross PnL:      ${stats.get('total_gross_pnl', 0):+,.2f}")
    print(f"  Total Fees:     ${stats.get('total_fees', 0):,.2f}")
    print(f"  Net PnL:        ${stats.get('total_net_pnl', 0):+,.2f}")
    print()
    print("PERFORMANCE")
    print("-" * 40)
    print(f"  Winners:        {stats.get('winners', 0)}")
    print(f"  Losers:         {stats.get('losers', 0)}")
    print(f"  Win Rate:       {stats.get('win_rate', 0):.1f}%")
    print(f"  Avg Duration:   {stats.get('avg_duration_hours', 0):.1f}h")
    print()
    print("EXIT REASONS")
    print("-" * 40)
    reasons = stats.get("exit_reasons", {})
    for reason, count in reasons.items():
        print(f"  {reason:12}  {count}")
    print()
    
    # Metadata-based cross-check (more robust for partial fills)
    meta_pnl = stats.get("metadata_pnl", {})
    if meta_pnl:
        print("METADATA PnL (exit-based cross-check)")
        print("-" * 40)
        print(f"  Exit fills:     {meta_pnl.get('exits', 0)} ({meta_pnl.get('exits_with_entry_price', 0)} with entry_price)")
        print(f"  Gross PnL:      ${meta_pnl.get('gross_pnl', 0):+,.2f}")
        print(f"  Total Fees:     ${meta_pnl.get('fees', 0):,.2f}")
        print(f"  Net PnL:        ${meta_pnl.get('net_pnl', 0):+,.2f}")
        print()


# CLI entry point
if __name__ == "__main__":
    import sys
    
    since = sys.argv[1] if len(sys.argv) > 1 else None
    until = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"Building episode ledger (since={since}, until={until})...")
    ledger = rebuild_and_save(since, until)
    print_ledger_summary(ledger)
