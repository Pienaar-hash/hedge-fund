"""
Regime Pressure Widget — v7.9

Displays regime stability metrics to explain WHY the system is flat.
Read-only observability — no controls, no inference.

Data source: logs/state/regime_pressure.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st


# ---------------------------------------------------------------------------
# State Loader
# ---------------------------------------------------------------------------

def load_regime_pressure_state(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load regime pressure state from file."""
    p = path or Path("logs/state/regime_pressure.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


# ---------------------------------------------------------------------------
# Widget Renderer
# ---------------------------------------------------------------------------

def render_regime_pressure_widget(state: Optional[Dict[str, Any]] = None) -> None:
    """
    Render compact regime pressure widget.
    
    Shows:
    - Current regime + confidence
    - Near-flip counts (24h / 7d)
    - Dwell time
    - Hostility badge
    
    Placement: Top-row, beside KPI strip.
    """
    if state is None:
        state = load_regime_pressure_state()
    
    if not state:
        # No state available — silent return
        return
    
    current = state.get("current", {})
    pressure = state.get("pressure", {})
    churn = state.get("churn", {})
    context = state.get("context", {})
    
    # Extract values
    regime = current.get("regime", "UNKNOWN")
    confidence = current.get("confidence", 0)
    dwell_hours = current.get("dwell_time_hours", 0)
    stability_distance = current.get("stability_distance", 0)
    
    near_flip_24h = pressure.get("near_flip_count_24h", 0)
    near_flip_7d = pressure.get("near_flip_count_7d", 0)
    conf_velocity = pressure.get("confidence_velocity", 0)
    
    regime_changes_24h = churn.get("regime_changes_24h", 0)
    regime_changes_7d = churn.get("regime_changes_7d", 0)
    avg_dwell = churn.get("avg_dwell_time_hours_7d", 0)
    
    hostility = context.get("market_hostility", "UNKNOWN")
    
    # Regime colors
    regime_colors = {
        "TREND_UP": "#22c55e",
        "TREND_DOWN": "#f44",
        "MEAN_REVERT": "#9370db",
        "BREAKOUT": "#f59e0b",
        "CHOPPY": "#888",
        "CRISIS": "#ff1744",
    }
    
    # Hostility badge colors
    hostility_colors = {
        "CALM": "#22c55e",       # Green
        "MODERATE": "#f59e0b",   # Amber
        "HOSTILE": "#ef4444",    # Red
        "EXTREME": "#dc2626",    # Deep red
        "UNCERTAIN": "#f59e0b",
        "FAVORABLE": "#22c55e",
    }
    
    regime_color = regime_colors.get(regime, "#888")
    hostility_color = hostility_colors.get(hostility, "#888")
    
    # Confidence bar color based on stability
    if confidence >= 0.65:
        conf_color = "#22c55e"  # Green — stable
    elif confidence >= 0.55:
        conf_color = "#f59e0b"  # Gold — marginal
    else:
        conf_color = "#ef4444"  # Red — unstable / near-flip zone
    
    # Build HTML widget
    html = f'''
    <div style="
        background: linear-gradient(135deg, #1a1d24 0%, #12141a 100%);
        border: 1px solid #2d3139;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    ">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 0.75em; color: #888; text-transform: uppercase; letter-spacing: 0.5px;">
                    Regime Pressure
                </span>
                <span style="
                    background: {hostility_color}22;
                    color: {hostility_color};
                    padding: 2px 8px;
                    border-radius: 4px;
                    font-size: 0.7em;
                    font-weight: 600;
                ">{hostility}</span>
            </div>
            <span style="color: #666; font-size: 0.7em;">
                Dwell: {dwell_hours:.1f}h
            </span>
        </div>
        
        <div style="display: flex; gap: 16px; align-items: flex-end;">
            <!-- Regime + Confidence -->
            <div style="flex: 1;">
                <div style="font-size: 1.4em; font-weight: 700; color: {regime_color};">
                    {regime}
                </div>
                <div style="margin-top: 4px;">
                    <div style="
                        background: #2d3139;
                        border-radius: 4px;
                        height: 6px;
                        width: 100%;
                        overflow: hidden;
                    ">
                        <div style="
                            background: {conf_color};
                            height: 100%;
                            width: {confidence * 100:.0f}%;
                            transition: width 0.3s;
                        "></div>
                    </div>
                    <div style="font-size: 0.75em; color: #888; margin-top: 2px;">
                        Confidence: {confidence:.0%}
                    </div>
                </div>
            </div>
            
            <!-- Flip Metrics -->
            <div style="text-align: center; min-width: 70px;">
                <div style="font-size: 1.2em; font-weight: 600; color: {'#ef4444' if near_flip_24h > 50 else '#f59e0b' if near_flip_24h > 20 else '#888'};">
                    {near_flip_24h}
                </div>
                <div style="font-size: 0.65em; color: #666;">Flips 24h</div>
            </div>
            
            <div style="text-align: center; min-width: 70px;">
                <div style="font-size: 1.2em; font-weight: 600; color: {'#ef4444' if near_flip_7d > 200 else '#f59e0b' if near_flip_7d > 100 else '#888'};">
                    {near_flip_7d}
                </div>
                <div style="font-size: 0.65em; color: #666;">Flips 7d</div>
            </div>
            
            <!-- Regime Changes -->
            <div style="text-align: center; min-width: 70px;">
                <div style="font-size: 1.2em; font-weight: 600; color: {'#ef4444' if regime_changes_24h > 4 else '#f59e0b' if regime_changes_24h > 2 else '#22c55e'};">
                    {regime_changes_24h}
                </div>
                <div style="font-size: 0.65em; color: #666;">Changes 24h</div>
            </div>
            
            <div style="text-align: center; min-width: 70px;">
                <div style="font-size: 1.2em; font-weight: 600; color: #888;">
                    {avg_dwell:.1f}h
                </div>
                <div style="font-size: 0.65em; color: #666;">Avg Dwell</div>
            </div>
        </div>
    </div>
    '''
    
    st.html(html)


# ---------------------------------------------------------------------------
# Standalone Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test with sample data
    test_state = {
        "current": {
            "regime": "CHOPPY",
            "confidence": 0.50,
            "dwell_time_hours": 1.69,
            "stability_distance": 0
        },
        "pressure": {
            "confidence_velocity": -0.0092,
            "near_flip_count_24h": 84,
            "near_flip_count_7d": 424
        },
        "churn": {
            "regime_changes_24h": 5,
            "regime_changes_7d": 27,
            "avg_dwell_time_hours_7d": 6.26
        },
        "context": {
            "market_hostility": "HOSTILE"
        }
    }
    render_regime_pressure_widget(test_state)
