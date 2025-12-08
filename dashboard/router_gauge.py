"""
Router Health Gauge (v7)

Visual gauge showing router performance with:
- Circular ring gauge (AUM donut style)
- Health score in [0,1]
- Color-coded ring (green → gold → red)
"""
from __future__ import annotations

import streamlit as st
from typing import Any


# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------
COLOR_OK = "#21c354"
COLOR_WARN = "#f2c037"
COLOR_DEGRADED = "#d94a4a"
COLOR_NEUTRAL = "#888888"


def _get_health_score_color(score: float) -> str:
    """Return color based on health score threshold."""
    if score >= 0.80:
        return COLOR_OK
    elif score >= 0.50:
        return COLOR_WARN
    else:
        return COLOR_DEGRADED


def _get_fill_rate_color(fill_rate: float) -> str:
    """Return color based on fill rate threshold."""
    if fill_rate >= 0.90:
        return COLOR_OK
    elif fill_rate >= 0.70:
        return COLOR_WARN
    else:
        return COLOR_DEGRADED


def _get_health_status(router_state: dict[str, Any]) -> tuple[str, str]:
    """
    Determine router health status and description.
    
    Returns:
        Tuple of (status_label, status_color)
    """
    if not router_state:
        return "UNKNOWN", COLOR_NEUTRAL
    
    # Prefer health score if available
    health_score = router_state.get("router_health_score")
    if health_score is not None:
        if health_score >= 0.80:
            return "HEALTHY", COLOR_OK
        elif health_score >= 0.50:
            return "MARGINAL", COLOR_WARN
        else:
            return "POOR", COLOR_DEGRADED
    
    # Fallback to fill rate
    fill_rate = router_state.get("fill_rate", 0.0)
    is_degraded = router_state.get("degraded", False)
    
    if is_degraded:
        return "DEGRADED", COLOR_DEGRADED
    elif fill_rate >= 0.90:
        return "HEALTHY", COLOR_OK
    elif fill_rate >= 0.70:
        return "MARGINAL", COLOR_WARN
    else:
        return "POOR", COLOR_DEGRADED


# ---------------------------------------------------------------------------
# Circular Ring Gauge (AUM Donut Style)
# ---------------------------------------------------------------------------
def render_router_circle_gauge(router_state: dict[str, Any]) -> None:
    """
    Render a circular ring gauge for router health score.
    
    Matches the AUM donut style with:
    - Outer ring showing health score percentage
    - Color gradient based on score
    - Central score display
    """
    if not router_state:
        st.warning("Router state unavailable")
        return
    
    # Get health score (default to computing from maker_ratio)
    health_score = router_state.get("router_health_score")
    if health_score is None:
        # Fallback: use maker_ratio as proxy
        maker_ratio = router_state.get("maker_ratio") or router_state.get("maker_fill_rate") or 0.0
        health_score = float(maker_ratio) if maker_ratio else 0.0
    
    # Get color based on score
    ring_color = _get_health_score_color(health_score)
    status_label, _ = _get_health_status(router_state)
    
    # Calculate ring percentage (0-100)
    ring_pct = health_score * 100
    
    # SVG circular gauge
    st.markdown(
        f"""
        <div style="display: flex; flex-direction: column; align-items: center; padding: 10px;">
            <svg width="140" height="140" viewBox="0 0 140 140">
                <!-- Background circle -->
                <circle
                    cx="70" cy="70" r="55"
                    fill="none"
                    stroke="#333"
                    stroke-width="12"
                />
                <!-- Health score ring -->
                <circle
                    cx="70" cy="70" r="55"
                    fill="none"
                    stroke="{ring_color}"
                    stroke-width="12"
                    stroke-linecap="round"
                    stroke-dasharray="{ring_pct * 3.456} 345.6"
                    transform="rotate(-90 70 70)"
                    style="transition: stroke-dasharray 0.5s ease;"
                />
                <!-- Center text -->
                <text x="70" y="65" text-anchor="middle" fill="{ring_color}" font-size="28" font-weight="bold">
                    {health_score:.0%}
                </text>
                <text x="70" y="85" text-anchor="middle" fill="#888" font-size="11">
                    {status_label}
                </text>
            </svg>
            <div style="font-size: 12px; color: #888; margin-top: 5px;">Router Health</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main Gauge Renderer
# ---------------------------------------------------------------------------
def render_router_gauge(router_state: dict[str, Any]) -> None:
    """
    Render router health gauge.
    
    Displays:
    - Circular health score gauge
    - Maker ratio and metrics
    - Health status indicator
    """
    st.markdown("### 🔄 Router Health")
    
    if not router_state:
        st.warning("Router state unavailable")
        return
    
    # Two-column layout: circle gauge + metrics
    col_gauge, col_metrics = st.columns([1, 2])
    
    with col_gauge:
        render_router_circle_gauge(router_state)
    
    with col_metrics:
        # Extract metrics
        health_score = router_state.get("router_health_score", 0.0)
        maker_ratio = router_state.get("maker_ratio") or router_state.get("maker_fill_rate") or 0.0
        fallback_ratio = router_state.get("fallback_ratio") or 0.0
        reject_ratio = router_state.get("reject_ratio") or router_state.get("reject_rate") or 0.0
        avg_slippage = router_state.get("avg_slippage_bps") or router_state.get("slip_q50_bps") or 0.0
        
        # Get status
        status_label, status_color = _get_health_status(router_state)
        
        # Render status badge
        st.markdown(
            f"""
            <div style="
                display: inline-block;
                padding: 4px 12px;
                background-color: {status_color}20;
                border: 2px solid {status_color};
                border-radius: 8px;
                margin-bottom: 10px;
            ">
                <span style="color: {status_color}; font-weight: bold; font-size: 14px;">
                    ● {status_label}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Metrics grid
        m1, m2 = st.columns(2)
        
        with m1:
            maker_color = COLOR_OK if maker_ratio >= 0.7 else (COLOR_WARN if maker_ratio >= 0.5 else COLOR_DEGRADED)
            st.markdown(
                f"""
                <div style="padding: 8px; background: #1a1a2e; border-radius: 8px; margin-bottom: 8px;">
                    <div style="font-size: 11px; color: #888;">Maker Ratio</div>
                    <div style="font-size: 20px; font-weight: bold; color: {maker_color};">{maker_ratio:.1%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            fb_color = COLOR_OK if fallback_ratio <= 0.1 else (COLOR_WARN if fallback_ratio <= 0.3 else COLOR_DEGRADED)
            st.markdown(
                f"""
                <div style="padding: 8px; background: #1a1a2e; border-radius: 8px;">
                    <div style="font-size: 11px; color: #888;">Fallback Ratio</div>
                    <div style="font-size: 20px; font-weight: bold; color: {fb_color};">{fallback_ratio:.1%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        with m2:
            slip_color = COLOR_OK if avg_slippage <= 5 else (COLOR_WARN if avg_slippage <= 15 else COLOR_DEGRADED)
            st.markdown(
                f"""
                <div style="padding: 8px; background: #1a1a2e; border-radius: 8px; margin-bottom: 8px;">
                    <div style="font-size: 11px; color: #888;">Avg Slippage</div>
                    <div style="font-size: 20px; font-weight: bold; color: {slip_color};">{avg_slippage:.1f} bps</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            rej_color = COLOR_OK if reject_ratio <= 0.05 else (COLOR_WARN if reject_ratio <= 0.15 else COLOR_DEGRADED)
            st.markdown(
                f"""
                <div style="padding: 8px; background: #1a1a2e; border-radius: 8px;">
                    <div style="font-size: 11px; color: #888;">Reject Rate</div>
                    <div style="font-size: 20px; font-weight: bold; color: {rej_color};">{reject_ratio:.1%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Compact Gauge (for embedding)
# ---------------------------------------------------------------------------
def render_router_gauge_compact(router_state: dict[str, Any]) -> None:
    """
    Render a compact router gauge suitable for embedding.
    
    Shows health score circle and status in minimal space.
    """
    if not router_state:
        st.markdown("🔄 Router: **UNKNOWN**")
        return
    
    health_score = router_state.get("router_health_score", 0.0)
    status_label, status_color = _get_health_status(router_state)
    
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 15px;">
            <svg width="50" height="50" viewBox="0 0 50 50">
                <circle cx="25" cy="25" r="20" fill="none" stroke="#333" stroke-width="5"/>
                <circle cx="25" cy="25" r="20" fill="none" stroke="{status_color}" stroke-width="5"
                        stroke-dasharray="{health_score * 125.6} 125.6" transform="rotate(-90 25 25)"/>
                <text x="25" y="29" text-anchor="middle" fill="{status_color}" font-size="12" font-weight="bold">
                    {health_score:.0%}
                </text>
            </svg>
            <div>
                <div style="font-size: 14px; font-weight: bold; color: {status_color};">{status_label}</div>
                <div style="font-size: 11px; color: #888;">Router Health</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )