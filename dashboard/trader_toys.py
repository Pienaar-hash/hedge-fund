# mypy: ignore-errors
"""Trader Toys — CSS-only visual gauges and meters for the dashboard."""
from __future__ import annotations

import streamlit as st
from typing import Optional


def inject_trader_toys_css() -> None:
    """Inject CSS styles for all trader toy components."""
    st.markdown("""
    <style>
    /* ===== LIQUID GAUGE ===== */
    .liquid-gauge {
        position: relative;
        width: 100px;
        height: 100px;
        border-radius: 50%;
        background: linear-gradient(135deg, #1a1a2e 0%, #0f0f1a 100%);
        box-shadow: 
            inset 0 0 20px rgba(0,0,0,0.5),
            0 0 10px rgba(0,200,255,0.2);
        overflow: hidden;
        margin: 10px auto;
    }
    
    .liquid-gauge::before {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        background: linear-gradient(180deg, 
            rgba(0,200,255,0.8) 0%, 
            rgba(0,150,255,0.6) 100%);
        border-radius: 0 0 50% 50%;
        animation: wave 3s ease-in-out infinite;
    }
    
    .liquid-gauge.green::before {
        background: linear-gradient(180deg, 
            rgba(0,255,100,0.8) 0%, 
            rgba(0,200,80,0.6) 100%);
    }
    
    .liquid-gauge.yellow::before {
        background: linear-gradient(180deg, 
            rgba(255,200,0,0.8) 0%, 
            rgba(255,150,0,0.6) 100%);
    }
    
    .liquid-gauge.red::before {
        background: linear-gradient(180deg, 
            rgba(255,80,80,0.8) 0%, 
            rgba(200,50,50,0.6) 100%);
    }
    
    .liquid-gauge .gauge-value {
        position: absolute;
        top: 40%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 18px;
        font-weight: 700;
        color: #fff;
        text-shadow: 0 0 10px rgba(0,0,0,0.8);
        z-index: 10;
    }
    
    .liquid-gauge .gauge-label {
        position: absolute;
        bottom: 8px;
        left: 50%;
        transform: translateX(-50%);
        font-size: 10px;
        color: #aaa;
        white-space: nowrap;
        text-shadow: 0 0 5px rgba(0,0,0,0.9);
        z-index: 11;
    }
    
    @keyframes wave {
        0%, 100% { transform: translateY(2px); }
        50% { transform: translateY(-2px); }
    }
    
    /* ===== RADIAL PROGRESS ===== */
    .radial-progress {
        position: relative;
        width: 90px;
        height: 90px;
        margin: 10px auto;
    }
    
    .radial-progress svg {
        transform: rotate(-90deg);
    }
    
    .radial-progress .progress-bg {
        fill: none;
        stroke: #1a1a2e;
        stroke-width: 8;
    }
    
    .radial-progress .progress-bar {
        fill: none;
        stroke-width: 8;
        stroke-linecap: round;
        transition: stroke-dashoffset 0.5s ease;
    }
    
    .radial-progress .progress-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 16px;
        font-weight: 700;
        color: #fff;
    }
    
    .radial-progress .progress-label {
        position: absolute;
        bottom: -20px;
        left: 50%;
        transform: translateX(-50%);
        font-size: 10px;
        color: #888;
        white-space: nowrap;
    }
    
    /* ===== PULSE DOT ===== */
    .pulse-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        animation: pulse 2s ease-in-out infinite;
    }
    
    .pulse-dot.green { background: #00cc00; box-shadow: 0 0 10px #00cc00; }
    .pulse-dot.yellow { background: #ffaa00; box-shadow: 0 0 10px #ffaa00; }
    .pulse-dot.red { background: #ff4444; box-shadow: 0 0 10px #ff4444; }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.6; transform: scale(1.2); }
    }
    
    /* ===== MINI BAR ===== */
    .mini-bar-container {
        width: 100%;
        height: 8px;
        background: #1a1a2e;
        border-radius: 4px;
        overflow: hidden;
        margin: 5px 0;
    }
    
    .mini-bar {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    
    .mini-bar.green { background: linear-gradient(90deg, #00aa00, #00ff00); }
    .mini-bar.yellow { background: linear-gradient(90deg, #cc8800, #ffcc00); }
    .mini-bar.red { background: linear-gradient(90deg, #aa0000, #ff4444); }
    .mini-bar.blue { background: linear-gradient(90deg, #0066aa, #00aaff); }
    
    /* ===== STAT CARD ===== */
    .stat-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        border: 1px solid #333;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(0,200,255,0.15);
    }
    
    .stat-card .stat-value {
        font-size: 24px;
        font-weight: 700;
        margin: 5px 0;
    }
    
    .stat-card .stat-label {
        font-size: 11px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* ===== GAUGE ROW ===== */
    .gauge-row {
        display: flex;
        justify-content: space-around;
        align-items: flex-start;
        gap: 20px;
        padding: 15px 0;
    }
    
    .gauge-item {
        text-align: center;
        flex: 1;
    }
    </style>
    """, unsafe_allow_html=True)


def render_liquid_gauge(
    value: float,
    max_value: float = 100,
    label: str = "",
    thresholds: tuple = (50, 80),  # (warning, danger)
    inverse: bool = False,  # True = lower is better (like drawdown)
    size: int = 100,
) -> None:
    """Render a CSS-only liquid-fill circular gauge.
    
    Args:
        value: Current value
        max_value: Maximum value (100%)
        label: Label text below gauge
        thresholds: (warning_threshold, danger_threshold)
        inverse: If True, colors are inverted (green for low values)
        size: Gauge diameter in pixels
    """
    pct = min(100, max(0, (value / max_value) * 100)) if max_value > 0 else 0
    
    # Determine color based on thresholds
    if inverse:
        if pct <= thresholds[0]:
            color = "green"
        elif pct <= thresholds[1]:
            color = "yellow"
        else:
            color = "red"
    else:
        if pct >= thresholds[1]:
            color = "green"
        elif pct >= thresholds[0]:
            color = "yellow"
        else:
            color = "red"
    
    # Display text
    if value >= 1000:
        display_val = f"{value/1000:.1f}k"
    elif value >= 100:
        display_val = f"{value:.0f}"
    else:
        display_val = f"{value:.1f}"
    
    st.markdown(f"""
    <div class="liquid-gauge {color}" style="width:{size}px;height:{size}px;">
        <style>
            .liquid-gauge.{color}::before {{ height: {pct}%; }}
        </style>
        <span class="gauge-value">{display_val}%</span>
        <span class="gauge-label">{label}</span>
    </div>
    """, unsafe_allow_html=True)


def render_radial_progress(
    value: float,
    max_value: float = 100,
    label: str = "",
    color: str = "#00aaff",
    size: int = 90,
) -> None:
    """Render a CSS radial progress ring.
    
    Args:
        value: Current value
        max_value: Maximum value
        label: Label text
        color: Stroke color
        size: Diameter in pixels
    """
    pct = min(100, max(0, (value / max_value) * 100)) if max_value > 0 else 0
    circumference = 2 * 3.14159 * 35  # radius = 35
    offset = circumference - (pct / 100) * circumference
    
    # Format display value
    if value >= 1000:
        display_val = f"${value/1000:.1f}k"
    elif value >= 1:
        display_val = f"${value:.0f}"
    else:
        display_val = f"{pct:.0f}%"
    
    st.markdown(f"""
    <div class="radial-progress" style="width:{size}px;height:{size}px;">
        <svg width="{size}" height="{size}" viewBox="0 0 90 90">
            <circle class="progress-bg" cx="45" cy="45" r="35"/>
            <circle class="progress-bar" cx="45" cy="45" r="35" 
                stroke="{color}"
                stroke-dasharray="{circumference}"
                stroke-dashoffset="{offset}"/>
        </svg>
        <span class="progress-text">{display_val}</span>
        <span class="progress-label">{label}</span>
    </div>
    """, unsafe_allow_html=True)


def render_mini_bar(
    value: float,
    max_value: float = 100,
    color: str = "blue",
    label: Optional[str] = None,
) -> None:
    """Render a mini horizontal progress bar.
    
    Args:
        value: Current value
        max_value: Maximum value
        color: Bar color (green, yellow, red, blue)
        label: Optional label
    """
    pct = min(100, max(0, (value / max_value) * 100)) if max_value > 0 else 0
    
    html = f"""
    <div class="mini-bar-container">
        <div class="mini-bar {color}" style="width:{pct}%;"></div>
    </div>
    """
    if label:
        html = f"<div style='font-size:11px;color:#888;margin-bottom:2px;'>{label}</div>" + html
    
    st.markdown(html, unsafe_allow_html=True)


def render_pulse_indicator(status: str, label: str = "") -> None:
    """Render a pulsing status indicator.
    
    Args:
        status: Status color (green, yellow, red)
        label: Text label
    """
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;">
        <span class="pulse-dot {status}"></span>
        <span style="color:#ccc;font-size:14px;">{label}</span>
    </div>
    """, unsafe_allow_html=True)


def render_gauge_panel(
    exposure_pct: float,
    margin_used_pct: float,
    drawdown_pct: float,
    risk_capacity_pct: float,
) -> None:
    """Render a row of 4 liquid gauges for key risk metrics.
    
    Args:
        exposure_pct: Gross exposure as % of NAV
        margin_used_pct: Margin utilization %
        drawdown_pct: Current drawdown %
        risk_capacity_pct: Available risk capacity %
    """
    st.markdown("""
    <div class="gauge-row">
        <div class="gauge-item" id="gauge-exposure"></div>
        <div class="gauge-item" id="gauge-margin"></div>
        <div class="gauge-item" id="gauge-drawdown"></div>
        <div class="gauge-item" id="gauge-capacity"></div>
    </div>
    """, unsafe_allow_html=True)
    
    cols = st.columns(4)
    
    with cols[0]:
        render_liquid_gauge(
            value=exposure_pct,
            max_value=150,  # 150% is max gross exposure
            label="Exposure",
            thresholds=(80, 120),
            inverse=True,
            size=85
        )
    
    with cols[1]:
        render_liquid_gauge(
            value=margin_used_pct,
            max_value=100,
            label="Margin Used",
            thresholds=(50, 80),
            inverse=True,
            size=85
        )
    
    with cols[2]:
        render_liquid_gauge(
            value=drawdown_pct,
            max_value=30,  # 30% max drawdown
            label="Drawdown",
            thresholds=(10, 20),
            inverse=True,
            size=85
        )
    
    with cols[3]:
        render_liquid_gauge(
            value=risk_capacity_pct,
            max_value=100,
            label="Risk Capacity",
            thresholds=(30, 60),
            inverse=False,
            size=85
        )


def render_stat_card(
    value: str,
    label: str,
    color: str = "#00aaff",
    delta: Optional[str] = None,
    delta_color: str = "#00cc00",
) -> None:
    """Render a styled stat card.
    
    Args:
        value: Main value to display
        label: Label text
        color: Value color
        delta: Optional delta/change text
        delta_color: Delta text color
    """
    delta_html = f'<div style="font-size:12px;color:{delta_color};margin-top:3px;">{delta}</div>' if delta else ""
    
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">{label}</div>
        <div class="stat-value" style="color:{color};">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)
