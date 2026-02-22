"""
Episode Ledger Summary Widget — v7.9

Compact card showing completed trade episodes.
Read-only observability — no charts, no timelines.

Data source: logs/state/episode_ledger.json
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import streamlit as st


def _parse_rebuild_ts(state: Dict[str, Any]) -> Tuple[Optional[datetime], float]:
    """Parse last_rebuild_ts and compute age in seconds."""
    ts_str = state.get("last_rebuild_ts", "")
    if not ts_str:
        return None, float('inf')
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        age_s = (datetime.now(timezone.utc) - dt).total_seconds()
        return dt, age_s
    except (ValueError, TypeError):
        return None, float('inf')


# ---------------------------------------------------------------------------
# State Loader
# ---------------------------------------------------------------------------

def load_episode_ledger_state(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load episode ledger state from file."""
    p = path or Path("logs/state/episode_ledger.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


# ---------------------------------------------------------------------------
# Widget Renderer
# ---------------------------------------------------------------------------

def render_episode_ledger_summary(state: Optional[Dict[str, Any]] = None) -> None:
    """
    Render compact episode ledger summary.
    
    Shows:
    - Episodes closed
    - Net PnL
    - Exit reason breakdown
    - Avg duration
    - Staleness indicator (>30min warning)
    
    One compact card. No charts. No timelines.
    """
    if state is None:
        state = load_episode_ledger_state()
    
    if not state:
        return
    
    stats = state.get("stats", {})
    episode_count = state.get("episode_count", 0)
    
    # Parse rebuild timestamp and compute age
    rebuild_dt, age_s = _parse_rebuild_ts(state)
    STALE_THRESHOLD_S = 1800  # 30 minutes (cron runs every 15)
    is_stale = age_s > STALE_THRESHOLD_S
    
    # Format age display
    if rebuild_dt:
        age_mins = int(age_s / 60)
        if age_mins < 60:
            age_display = f"{age_mins}m ago"
        else:
            age_hours = age_mins // 60
            age_display = f"{age_hours}h ago"
        if is_stale:
            age_badge = f'<span style="color: #f59e0b; font-size: 0.65em;">⚠️ {age_display}</span>'
        else:
            age_badge = f'<span style="color: #22c55e; font-size: 0.65em;">✓ {age_display}</span>'
    else:
        age_badge = '<span style="color: #ef4444; font-size: 0.65em;">⚠️ unknown</span>'
    
    # Core metrics
    total_net_pnl = stats.get("total_net_pnl", 0)
    winners = stats.get("winners", 0)
    losers = stats.get("losers", 0)
    win_rate = stats.get("win_rate", 0)
    avg_duration = stats.get("avg_duration_hours", 0)
    exit_reasons = stats.get("exit_reasons", {})

    # Last close timestamp
    episodes_list = state.get("episodes", [])
    last_close_display = ""
    if episodes_list:
        last_exit_ts = max(
            (e.get("exit_ts", "") for e in episodes_list),
            default="",
        )
        if last_exit_ts:
            try:
                last_dt = datetime.fromisoformat(last_exit_ts.replace("Z", "+00:00"))
                days_ago = (datetime.now(timezone.utc) - last_dt).days
                if days_ago == 0:
                    last_close_display = "Last close: today"
                elif days_ago == 1:
                    last_close_display = "Last close: yesterday"
                else:
                    last_close_display = f"Last close: {days_ago}d ago"
            except (ValueError, TypeError):
                pass
    
    # PnL color
    pnl_color = "#22c55e" if total_net_pnl >= 0 else "#ef4444"
    pnl_sign = "+" if total_net_pnl >= 0 else ""
    
    # Exit reason breakdown
    exit_html = ""
    exit_order = ["tp", "sl", "thesis", "regime_flip", "position_flip", "signal_close", "unknown"]
    exit_labels = {
        "tp": "TP",
        "sl": "SL", 
        "thesis": "Thesis",
        "regime_flip": "Regime",
        "position_flip": "Flip",
        "signal_close": "Signal",
        "unknown": "Other"
    }
    exit_colors = {
        "tp": "#22c55e",
        "sl": "#ef4444",
        "thesis": "#9370db",
        "regime_flip": "#f59e0b",
        "position_flip": "#5dade2",
        "signal_close": "#888",
        "unknown": "#555"
    }
    
    total_exits = sum(exit_reasons.values()) if exit_reasons else 0
    if total_exits > 0:
        for reason in exit_order:
            count = exit_reasons.get(reason, 0)
            if count > 0:
                pct = count / total_exits * 100
                color = exit_colors.get(reason, "#888")
                label = exit_labels.get(reason, reason)
                exit_html += f'''
                <span style="
                    display: inline-block;
                    background: {color}22;
                    color: {color};
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-size: 0.65em;
                    margin-right: 4px;
                ">{label}: {count}</span>
                '''
    
    # Build widget HTML
    html = f'''
    <div style="
        background: linear-gradient(135deg, #1a1d24 0%, #12141a 100%);
        border: 1px solid #2d3139;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    ">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
            <span style="font-size: 0.75em; color: #888; text-transform: uppercase; letter-spacing: 0.5px;">
                📋 Episode Ledger
            </span>
            <div style="display: flex; align-items: center; gap: 8px;">
                {age_badge}
                <span style="color: #666; font-size: 0.7em;">
                    {episode_count} closed
                </span>
            </div>
        </div>
        
        <div style="display: flex; gap: 20px; align-items: flex-end;">
            <!-- Net PnL -->
            <div style="min-width: 100px;">
                <div style="font-size: 1.4em; font-weight: 700; color: {pnl_color};">
                    {pnl_sign}${abs(total_net_pnl):.2f}
                </div>
                <div style="font-size: 0.7em; color: #666;">Closed PnL (Episodes)</div>
                <div style="font-size: 0.6em; color: #555;">{last_close_display}</div>
            </div>
            
            <!-- Win/Loss -->
            <div style="text-align: center;">
                <div style="font-size: 1.1em;">
                    <span style="color: #22c55e; font-weight: 600;">{winners}W</span>
                    <span style="color: #666;">/</span>
                    <span style="color: #ef4444; font-weight: 600;">{losers}L</span>
                </div>
                <div style="font-size: 0.7em; color: #666;">Record</div>
            </div>
            
            <!-- Win Rate -->
            <div style="text-align: center;">
                <div style="font-size: 1.1em; font-weight: 600; color: {'#22c55e' if win_rate >= 50 else '#f59e0b' if win_rate >= 40 else '#ef4444'};">
                    {win_rate:.1f}%
                </div>
                <div style="font-size: 0.7em; color: #666;">Win Rate</div>
            </div>
            
            <!-- Avg Duration -->
            <div style="text-align: center;">
                <div style="font-size: 1.1em; font-weight: 600; color: #888;">
                    {avg_duration:.1f}h
                </div>
                <div style="font-size: 0.7em; color: #666;">Avg Duration</div>
            </div>
        </div>
        
        <!-- Exit Reasons -->
        <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid #2d3139;">
            <span style="font-size: 0.65em; color: #666; margin-right: 8px;">Exit Reasons:</span>
            {exit_html if exit_html else '<span style="font-size: 0.65em; color: #666;">—</span>'}
        </div>
    </div>
    '''
    
    st.html(html)


# ---------------------------------------------------------------------------
# Standalone Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_state = {
        "episode_count": 1,
        "stats": {
            "total_net_pnl": -0.36,
            "winners": 0,
            "losers": 1,
            "win_rate": 0,
            "avg_duration_hours": 2.5,
            "exit_reasons": {
                "tp": 0,
                "sl": 0,
                "thesis": 0,
                "regime_flip": 1,
                "unknown": 0
            }
        }
    }
    render_episode_ledger_summary(test_state)
