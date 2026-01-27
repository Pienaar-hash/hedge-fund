"""
Episode Ledger Summary Widget — v7.9

Compact card showing completed trade episodes.
Read-only observability — no charts, no timelines.

Data source: logs/state/episode_ledger.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st


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
    
    One compact card. No charts. No timelines.
    """
    if state is None:
        state = load_episode_ledger_state()
    
    if not state:
        return
    
    stats = state.get("stats", {})
    episode_count = state.get("episode_count", 0)
    
    # Core metrics
    total_net_pnl = stats.get("total_net_pnl", 0)
    winners = stats.get("winners", 0)
    losers = stats.get("losers", 0)
    win_rate = stats.get("win_rate", 0)
    avg_duration = stats.get("avg_duration_hours", 0)
    exit_reasons = stats.get("exit_reasons", {})
    
    # PnL color
    pnl_color = "#21c354" if total_net_pnl >= 0 else "#d94a4a"
    pnl_sign = "+" if total_net_pnl >= 0 else ""
    
    # Exit reason breakdown
    exit_html = ""
    exit_order = ["tp", "sl", "thesis", "regime_flip", "unknown"]
    exit_labels = {
        "tp": "TP",
        "sl": "SL", 
        "thesis": "Thesis",
        "regime_flip": "Regime",
        "unknown": "Other"
    }
    exit_colors = {
        "tp": "#21c354",
        "sl": "#d94a4a",
        "thesis": "#9370db",
        "regime_flip": "#f2c037",
        "unknown": "#888"
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
            <span style="color: #666; font-size: 0.7em;">
                {episode_count} closed
            </span>
        </div>
        
        <div style="display: flex; gap: 20px; align-items: flex-end;">
            <!-- Net PnL -->
            <div style="min-width: 100px;">
                <div style="font-size: 1.4em; font-weight: 700; color: {pnl_color};">
                    {pnl_sign}${abs(total_net_pnl):.2f}
                </div>
                <div style="font-size: 0.7em; color: #666;">Net PnL</div>
            </div>
            
            <!-- Win/Loss -->
            <div style="text-align: center;">
                <div style="font-size: 1.1em;">
                    <span style="color: #21c354; font-weight: 600;">{winners}W</span>
                    <span style="color: #666;">/</span>
                    <span style="color: #d94a4a; font-weight: 600;">{losers}L</span>
                </div>
                <div style="font-size: 0.7em; color: #666;">Record</div>
            </div>
            
            <!-- Win Rate -->
            <div style="text-align: center;">
                <div style="font-size: 1.1em; font-weight: 600; color: {'#21c354' if win_rate >= 0.5 else '#f2c037' if win_rate >= 0.4 else '#d94a4a'};">
                    {win_rate:.0%}
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
