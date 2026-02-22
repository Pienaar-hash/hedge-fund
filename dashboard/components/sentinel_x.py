"""
Sentinel-X Compact Widget — v7.9

Compact regime classifier display for main dashboard.
Shows smoothed probabilities, primary regime, stability counter.

Data source: logs/state/sentinel_x.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st


# ---------------------------------------------------------------------------
# State Loader
# ---------------------------------------------------------------------------

def load_sentinel_x_state(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load Sentinel-X state from file."""
    p = path or Path("logs/state/sentinel_x.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


# ---------------------------------------------------------------------------
# Compact Widget Renderer
# ---------------------------------------------------------------------------

REGIME_COLORS = {
    "TREND_UP": "#22c55e",
    "TREND_DOWN": "#f44",
    "MEAN_REVERT": "#9370db",
    "BREAKOUT": "#f59e0b",
    "CHOPPY": "#888",
    "CRISIS": "#ff1744",
}

REGIME_ICONS = {
    "TREND_UP": "📈",
    "TREND_DOWN": "📉",
    "MEAN_REVERT": "🔄",
    "BREAKOUT": "💥",
    "CHOPPY": "〰️",
    "CRISIS": "🚨",
}


def render_sentinel_x_compact(state: Optional[Dict[str, Any]] = None) -> None:
    """
    Render compact Sentinel-X widget for main dashboard.
    
    Shows:
    - Smoothed regime probabilities (bar chart)
    - Primary regime with icon
    - Stability counter (consecutive same-regime cycles)
    """
    if state is None:
        state = load_sentinel_x_state()
    
    if not state:
        return
    
    primary_regime = state.get("primary_regime", "CHOPPY")
    secondary_regime = state.get("secondary_regime")
    smoothed_probs = state.get("smoothed_probs", {})
    history_meta = state.get("history_meta", {})
    crisis_flag = state.get("crisis_flag", False)
    cycle_count = state.get("cycle_count", 0)
    
    # Stability metrics
    consecutive_count = history_meta.get("consecutive_count", 0)
    pending_regime = history_meta.get("pending_regime")
    
    primary_color = REGIME_COLORS.get(primary_regime, "#888")
    primary_icon = REGIME_ICONS.get(primary_regime, "❓")
    
    # Build probability bars HTML
    prob_bars_html = ""
    for regime in ["TREND_UP", "TREND_DOWN", "MEAN_REVERT", "BREAKOUT", "CHOPPY", "CRISIS"]:
        prob = smoothed_probs.get(regime, 0)
        color = REGIME_COLORS.get(regime, "#888")
        bar_width = max(2, int(prob * 100))  # minimum 2% width for visibility
        
        # Highlight primary regime
        opacity = "1" if regime == primary_regime else "0.6"
        
        prob_bars_html += f'''
        <div style="display: flex; align-items: center; gap: 4px; margin: 2px 0;">
            <span style="width: 75px; font-size: 0.65em; color: #888; opacity: {opacity};">
                {regime.replace('_', ' ')}
            </span>
            <div style="flex: 1; background: #2d3139; border-radius: 2px; height: 8px; overflow: hidden;">
                <div style="
                    background: {color};
                    height: 100%;
                    width: {bar_width}%;
                    opacity: {opacity};
                "></div>
            </div>
            <span style="width: 35px; text-align: right; font-size: 0.65em; color: #888; opacity: {opacity};">
                {prob:.0%}
            </span>
        </div>
        '''
    
    # Crisis indicator
    crisis_html = ""
    if crisis_flag:
        crisis_html = '''
        <div style="
            background: #ff174422;
            border: 1px solid #ff1744;
            border-radius: 4px;
            padding: 4px 8px;
            margin-top: 8px;
            font-size: 0.75em;
            color: #ff1744;
        ">🚨 CRISIS OVERRIDE ACTIVE</div>
        '''
    
    # Pending regime indicator
    pending_html = ""
    if pending_regime and pending_regime != primary_regime:
        pending_color = REGIME_COLORS.get(pending_regime, "#888")
        pending_html = f'''
        <div style="font-size: 0.7em; color: {pending_color}; margin-top: 4px;">
            Pending: {pending_regime}
        </div>
        '''
    
    # Build full widget HTML
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
                🧠 Sentinel-X Regime
            </span>
            <span style="color: #666; font-size: 0.7em;">
                Cycle #{cycle_count:,}
            </span>
        </div>
        
        <div style="display: flex; gap: 20px;">
            <!-- Primary Regime Display -->
            <div style="text-align: center; min-width: 100px;">
                <div style="font-size: 2em;">{primary_icon}</div>
                <div style="font-size: 1.1em; font-weight: 700; color: {primary_color};">
                    {primary_regime}
                </div>
                <div style="font-size: 0.7em; color: #666; margin-top: 2px;">
                    {consecutive_count} cycles stable
                </div>
                {pending_html}
            </div>
            
            <!-- Probability Bars -->
            <div style="flex: 1;">
                {prob_bars_html}
            </div>
        </div>
        
        {crisis_html}
    </div>
    '''
    
    st.html(html)


# ---------------------------------------------------------------------------
# Standalone Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_state = {
        "primary_regime": "CHOPPY",
        "secondary_regime": "CHOPPY",
        "smoothed_probs": {
            "TREND_UP": 0.0001,
            "TREND_DOWN": 0.0001,
            "MEAN_REVERT": 0.4989,
            "BREAKOUT": 0.0001,
            "CHOPPY": 0.5008,
            "CRISIS": 0.0001
        },
        "history_meta": {
            "consecutive_count": 37,
            "pending_regime": "MEAN_REVERT"
        },
        "crisis_flag": False,
        "cycle_count": 12229
    }
    render_sentinel_x_compact(test_state)
