"""
Hydra Status Strip — v7.9

Compact display showing multi-strategy head status.
Proves capital is intentionally idle.

Data source: logs/state/hydra_state.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st


# ---------------------------------------------------------------------------
# State Loader
# ---------------------------------------------------------------------------

def load_hydra_state(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load Hydra state from file."""
    p = path or Path("logs/state/hydra_state.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


# ---------------------------------------------------------------------------
# Widget Renderer
# ---------------------------------------------------------------------------

HEAD_COLORS = {
    "TREND": "#4CAF50",
    "MEAN_REVERT": "#2196F3",
    "RELATIVE_VALUE": "#FF9800",
    "CATEGORY": "#9C27B0",
    "VOL_HARVEST": "#F44336",
    "EMERGENT_ALPHA": "#00BCD4",
}

HEAD_ICONS = {
    "TREND": "📈",
    "MEAN_REVERT": "🔄",
    "RELATIVE_VALUE": "⚖️",
    "CATEGORY": "📊",
    "VOL_HARVEST": "🎯",
    "EMERGENT_ALPHA": "✨",
}


def render_hydra_status_strip(state: Optional[Dict[str, Any]] = None) -> None:
    """
    Render compact Hydra status strip.
    
    Shows:
    - Heads enabled
    - Budget utilization
    - Positions per head
    
    Purpose: Prove capital is intentionally idle.
    """
    if state is None:
        state = load_hydra_state()
    
    if not state:
        return
    
    head_budgets = state.get("head_budgets", {})
    head_usage = state.get("head_usage", {})
    head_positions = state.get("head_positions", {})
    meta = state.get("meta", {})
    cycle_count = state.get("cycle_count", 0)
    
    enabled_heads = meta.get("enabled_heads", [])
    total_intents = meta.get("total_raw_intents", 0)
    
    # Calculate totals
    total_budget = sum(head_budgets.values())
    total_usage = sum(head_usage.values()) if head_usage else 0
    total_positions = sum(head_positions.values()) if head_positions else 0
    utilization = (total_usage / total_budget * 100) if total_budget > 0 else 0
    
    # Build head cards HTML
    head_cards_html = ""
    for head in ["TREND", "MEAN_REVERT", "RELATIVE_VALUE", "CATEGORY", "VOL_HARVEST", "EMERGENT_ALPHA"]:
        budget = head_budgets.get(head, 0)
        usage = head_usage.get(head, 0) if head_usage else 0
        positions = head_positions.get(head, 0) if head_positions else 0
        color = HEAD_COLORS.get(head, "#888")
        icon = HEAD_ICONS.get(head, "•")
        
        # Determine status
        is_enabled = head in enabled_heads or budget > 0
        head_util = (usage / budget * 100) if budget > 0 else 0
        
        # Status indicator
        if not is_enabled:
            status_color = "#444"
            status_text = "OFF"
        elif positions > 0:
            status_color = "#22c55e"
            status_text = f"{positions} pos"
        else:
            status_color = "#666"
            status_text = "idle"
        
        head_cards_html += f'''
        <div style="
            flex: 1;
            min-width: 90px;
            text-align: center;
            padding: 8px 4px;
            border-radius: 4px;
            background: {'#1a1d24' if is_enabled else '#12141a'};
            opacity: {'1' if is_enabled else '0.5'};
        ">
            <div style="font-size: 1.2em;">{icon}</div>
            <div style="font-size: 0.65em; color: {color}; font-weight: 600; margin: 2px 0;">
                {head.replace('_', ' ')}
            </div>
            <div style="font-size: 0.6em; color: {status_color};">
                {status_text}
            </div>
            <div style="font-size: 0.55em; color: #555; margin-top: 2px;">
                {budget:.0%} budget
            </div>
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
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 0.75em; color: #888; text-transform: uppercase; letter-spacing: 0.5px;">
                    🐉 Hydra Engine
                </span>
                <span style="
                    background: {'#22c55e22' if total_positions > 0 else '#88888822'};
                    color: {'#22c55e' if total_positions > 0 else '#888'};
                    padding: 2px 8px;
                    border-radius: 4px;
                    font-size: 0.7em;
                    font-weight: 600;
                ">{total_positions} positions</span>
            </div>
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="color: #666; font-size: 0.7em;">
                    Utilization: {utilization:.0f}%
                </span>
                <span style="color: #555; font-size: 0.65em;">
                    Intents: {total_intents}
                </span>
            </div>
        </div>
        
        <div style="display: flex; gap: 6px;">
            {head_cards_html}
        </div>
    </div>
    '''
    
    st.html(html)


# ---------------------------------------------------------------------------
# Standalone Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_state = {
        "head_budgets": {
            "TREND": 0.5,
            "MEAN_REVERT": 0.25,
            "RELATIVE_VALUE": 0.3,
            "CATEGORY": 0.2,
            "VOL_HARVEST": 0.2,
            "EMERGENT_ALPHA": 0.15
        },
        "head_usage": {},
        "head_positions": {
            "TREND": 0,
            "MEAN_REVERT": 0,
            "RELATIVE_VALUE": 0,
            "CATEGORY": 0,
            "VOL_HARVEST": 0,
            "EMERGENT_ALPHA": 0
        },
        "cycle_count": 0,
        "meta": {
            "total_raw_intents": 0,
            "enabled_heads": []
        }
    }
    render_hydra_status_strip(test_state)
