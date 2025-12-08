"""
Regime Panel (v7) - ATR and Drawdown regime visualization.

Displays a 4x4 heatmap showing the intersection of:
- ATR regimes (columns): Low, Normal, Elevated, Extreme
- Drawdown regimes (rows): Low, Moderate, High, Critical

Color codes:
- Low DD + Low ATR → green (safe)
- Higher quadrants → gold → orange → red (increasing risk)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from dashboard.nav_helpers import safe_float


STATE_DIR = Path(os.getenv("STATE_DIR") or "logs/state")
REGIMES_PATH = STATE_DIR / "regimes.json"

# ATR regime labels (columns)
ATR_REGIME_LABELS = ["Low", "Normal", "Elevated", "Extreme"]

# Drawdown regime labels (rows)
DD_REGIME_LABELS = ["Low", "Moderate", "High", "Critical"]

# Heatmap color matrix [dd_regime][atr_regime]
# Risk increases from top-left to bottom-right
REGIME_COLORS = [
    # Low DD row
    ["#21c354", "#7ed957", "#f2c037", "#f2c037"],  # Green → Gold
    # Moderate DD row
    ["#7ed957", "#f2c037", "#f2c037", "#ff8c42"],  # Light green → Gold → Orange
    # High DD row
    ["#f2c037", "#f2c037", "#ff8c42", "#d94a4a"],  # Gold → Orange → Red
    # Critical DD row
    ["#ff8c42", "#ff8c42", "#d94a4a", "#ff0033"],  # Orange → Red → Bright Red
]

# Text colors for contrast
REGIME_TEXT_COLORS = [
    ["#000", "#000", "#000", "#000"],
    ["#000", "#000", "#000", "#fff"],
    ["#000", "#000", "#fff", "#fff"],
    ["#fff", "#fff", "#fff", "#fff"],
]


def _load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file safely."""
    try:
        if not path.exists() or path.stat().st_size <= 0:
            return {}
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def load_regimes_snapshot(state_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load the regimes.json state file."""
    base_dir = state_dir or STATE_DIR
    path = base_dir / "regimes.json"
    return _load_json(path)


def get_regime_color(dd_regime: int, atr_regime: int) -> str:
    """Get the background color for a regime cell."""
    dd_idx = max(0, min(3, dd_regime))
    atr_idx = max(0, min(3, atr_regime))
    return REGIME_COLORS[dd_idx][atr_idx]


def get_regime_text_color(dd_regime: int, atr_regime: int) -> str:
    """Get the text color for a regime cell."""
    dd_idx = max(0, min(3, dd_regime))
    atr_idx = max(0, min(3, atr_regime))
    return REGIME_TEXT_COLORS[dd_idx][atr_idx]


def render_regime_heatmap(
    snapshot: Dict[str, Any],
    show_values: bool = True,
) -> None:
    """
    Render the 4x4 regime heatmap.

    Args:
        snapshot: Regime snapshot from regimes.json
        show_values: Whether to show ATR/DD values in current cell
    """
    st.markdown("""
    <style>
    .regime-heatmap {
        display: grid;
        grid-template-columns: 80px repeat(4, 1fr);
        grid-template-rows: 30px repeat(4, 60px);
        gap: 2px;
        margin: 10px 0;
        font-size: 12px;
    }
    .regime-header {
        background: #1a1a2e;
        color: #888;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 10px;
    }
    .regime-row-label {
        background: #1a1a2e;
        color: #888;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 10px;
        writing-mode: horizontal-tb;
    }
    .regime-cell {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        border-radius: 4px;
        transition: transform 0.2s;
    }
    .regime-cell:hover {
        transform: scale(1.05);
    }
    .regime-cell.active {
        box-shadow: 0 0 0 3px #fff, 0 0 10px rgba(255,255,255,0.5);
    }
    .regime-value {
        font-size: 10px;
        opacity: 0.9;
    }
    .regime-corner {
        background: transparent;
    }
    </style>
    """, unsafe_allow_html=True)

    atr_regime = int(snapshot.get("atr_regime", 0))
    dd_regime = int(snapshot.get("dd_regime", 0))
    atr_value = safe_float(snapshot.get("atr_value"))
    dd_frac = safe_float(snapshot.get("dd_frac"))
    matrix = snapshot.get("regime_matrix", [[0]*4 for _ in range(4)])

    # Build grid HTML
    cells = []
    
    # Corner cell
    cells.append('<div class="regime-corner"></div>')
    
    # ATR header row
    for label in ATR_REGIME_LABELS:
        cells.append(f'<div class="regime-header">{label}</div>')
    
    # Data rows
    for dd_idx, dd_label in enumerate(DD_REGIME_LABELS):
        # Row label
        cells.append(f'<div class="regime-row-label">{dd_label}</div>')
        
        # Cells
        for atr_idx in range(4):
            is_active = (dd_idx == dd_regime and atr_idx == atr_regime)
            bg_color = get_regime_color(dd_idx, atr_idx)
            text_color = get_regime_text_color(dd_idx, atr_idx)
            active_class = "active" if is_active else ""
            
            value_html = ""
            if is_active and show_values:
                atr_str = f"{atr_value:.2f}%" if atr_value is not None else "—"
                dd_str = f"{dd_frac*100:.1f}%" if dd_frac is not None else "—"
                value_html = f'<div class="regime-value">ATR: {atr_str}<br>DD: {dd_str}</div>'
            
            cells.append(
                f'<div class="regime-cell {active_class}" '
                f'style="background:{bg_color};color:{text_color}">'
                f'{value_html}</div>'
            )
    
    grid_html = '<div class="regime-heatmap">' + ''.join(cells) + '</div>'
    st.markdown(grid_html, unsafe_allow_html=True)


def render_regime_summary(snapshot: Dict[str, Any]) -> None:
    """
    Render a compact text summary of current regimes.

    Args:
        snapshot: Regime snapshot from regimes.json
    """
    atr_regime = int(snapshot.get("atr_regime", 0))
    dd_regime = int(snapshot.get("dd_regime", 0))
    atr_name = snapshot.get("atr_regime_name", ATR_REGIME_LABELS[atr_regime].lower())
    dd_name = snapshot.get("dd_regime_name", DD_REGIME_LABELS[dd_regime].lower())
    
    atr_value = safe_float(snapshot.get("atr_value"))
    dd_frac = safe_float(snapshot.get("dd_frac"))
    
    atr_str = f"{atr_value:.2f}%" if atr_value is not None else "—"
    dd_str = f"{dd_frac*100:.1f}%" if dd_frac is not None else "—"
    
    # Emoji based on overall risk level
    risk_level = atr_regime + dd_regime
    if risk_level <= 1:
        emoji = "🟢"
    elif risk_level <= 3:
        emoji = "🟡"
    elif risk_level <= 5:
        emoji = "🟠"
    else:
        emoji = "🔴"
    
    st.caption(f"{emoji} ATR: {atr_name} ({atr_str}) | DD: {dd_name} ({dd_str})")


def render_regime_card(snapshot: Dict[str, Any]) -> None:
    """
    Render a full regime card with heatmap and metrics.

    Args:
        snapshot: Regime snapshot from regimes.json
    """
    st.markdown("""
    <style>
    .regime-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #333;
        margin-bottom: 15px;
    }
    .regime-card-header {
        font-size: 14px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 15px;
        border-bottom: 1px solid #333;
        padding-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="regime-card">
        <div class="regime-card-header">📊 REGIME MATRIX</div>
    </div>
    """, unsafe_allow_html=True)
    
    render_regime_heatmap(snapshot)
    
    # Legend
    col1, col2 = st.columns(2)
    with col1:
        st.caption("**Columns:** ATR Volatility (Low → Extreme)")
    with col2:
        st.caption("**Rows:** Drawdown Level (Low → Critical)")
