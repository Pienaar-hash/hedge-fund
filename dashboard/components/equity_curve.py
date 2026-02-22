"""
Equity Curve Component — Portfolio NAV over time.

Renders an SVG line chart from logs/nav_log.json.
Single line: Portfolio NAV ($). Fill gradient, grid, axis labels.

Data source: logs/nav_log.json — array of {nav, t, unrealized_pnl}
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import streamlit as st
import streamlit.components.v1 as st_components

_NAV_LOG_PATH = Path("logs/nav_log.json")

# Chart dimensions
_CHART_W = 960
_CHART_H = 240
_PAD_L = 70   # left padding for y-axis labels
_PAD_R = 20
_PAD_T = 24
_PAD_B = 36   # bottom for x-axis labels


def _load_nav_series(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load NAV log with basic validation."""
    p = path or _NAV_LOG_PATH
    try:
        if not p.exists():
            return []
        data = json.loads(p.read_text())
        if not isinstance(data, list):
            return []
        # Filter entries with both nav and t
        return [e for e in data if e.get("nav") is not None and e.get("t") is not None]
    except Exception:
        return []


def _downsample(entries: List[Dict[str, Any]], max_points: int = 300) -> List[Dict[str, Any]]:
    """Downsample to max_points using simple stride."""
    if len(entries) <= max_points:
        return entries
    step = len(entries) / max_points
    result = []
    idx = 0.0
    while idx < len(entries):
        result.append(entries[int(idx)])
        idx += step
    # Always include last point
    if result[-1] is not entries[-1]:
        result.append(entries[-1])
    return result


def _format_time(ts: float) -> str:
    """Format unix timestamp to HH:MM or MMM DD depending on span."""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%H:%M")


def _format_time_date(ts: float) -> str:
    """Format as MMM DD."""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%b %d")


def _build_equity_svg(
    entries: List[Dict[str, Any]],
) -> Union[str, Tuple[str, str, float, float, float, str, str]]:
    """Build an SVG equity curve chart.

    Returns empty string if insufficient data, otherwise a tuple of
    (svg_html, span_text, end_nav, delta, delta_pct, delta_color, delta_sign).
    """
    if len(entries) < 2:
        return ""

    navs = [float(e["nav"]) for e in entries]
    times = [float(e["t"]) for e in entries]

    nav_min = min(navs)
    nav_max = max(navs)
    t_min = times[0]
    t_max = times[-1]

    # Pad y-range by 10% for visual breathing room on flat curves
    nav_range = nav_max - nav_min if nav_max != nav_min else 1.0
    y_pad = nav_range * 0.10
    nav_min_padded = nav_min - y_pad
    nav_max_padded = nav_max + y_pad
    nav_range_padded = nav_max_padded - nav_min_padded

    t_range = t_max - t_min if t_max != t_min else 1.0

    plot_w = _CHART_W - _PAD_L - _PAD_R
    plot_h = _CHART_H - _PAD_T - _PAD_B

    # Scale helpers (use padded range for visual scaling)
    def sx(t: float) -> float:
        return _PAD_L + ((t - t_min) / t_range) * plot_w

    def sy(nav: float) -> float:
        return _PAD_T + plot_h - ((nav - nav_min_padded) / nav_range_padded) * plot_h

    # Build polyline points
    pts = [(sx(t), sy(n)) for t, n in zip(times, navs)]
    points_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)

    # Fill path: line + close along bottom
    bottom_y = _PAD_T + plot_h
    fill_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    fill_path = (
        f"M {pts[0][0]:.1f},{bottom_y:.1f} "
        f"L {fill_points} "
        f"L {pts[-1][0]:.1f},{bottom_y:.1f} Z"
    )

    # Determine line color: green if up, red if down
    up = navs[-1] >= navs[0]
    color = "#10b981" if up else "#ef4444"
    fill_color = "rgba(16,185,129,0.12)" if up else "rgba(239,68,68,0.12)"
    grad_id = "navGradUp" if up else "navGradDn"
    grad_top = "rgba(16,185,129,0.25)" if up else "rgba(239,68,68,0.25)"

    # Y-axis labels (5 ticks) — use real data range, not padded
    y_labels = ""
    for i in range(5):
        val = nav_min + (nav_range * i / 4)
        y = sy(val)
        y_labels += (
            f'<text x="{_PAD_L - 8}" y="{y + 4}" '
            f'text-anchor="end" fill="#666" font-size="11" '
            f'font-family="monospace">${val:,.0f}</text>\n'
        )
        # Grid line
        y_labels += (
            f'<line x1="{_PAD_L}" y1="{y}" x2="{_CHART_W - _PAD_R}" y2="{y}" '
            f'stroke="#222630" stroke-width="1" stroke-dasharray="4,4" />\n'
        )

    # X-axis labels (5 ticks)
    span_hours = t_range / 3600
    x_labels = ""
    for i in range(5):
        t = t_min + (t_range * i / 4)
        x = sx(t)
        if span_hours > 48:
            label = _format_time_date(t)
        else:
            label = _format_time(t)
        x_labels += (
            f'<text x="{x}" y="{_CHART_H - 6}" '
            f'text-anchor="middle" fill="#666" font-size="11" '
            f'font-family="monospace">{label}</text>\n'
        )

    # NAV change stats
    start_nav = navs[0]
    end_nav = navs[-1]
    delta = end_nav - start_nav
    delta_pct = (delta / start_nav * 100) if start_nav > 0 else 0
    delta_sign = "+" if delta >= 0 else "-"
    delta_color = "#10b981" if delta >= 0 else "#ef4444"

    # Span display
    if span_hours < 24:
        span_text = f"{span_hours:.1f}h"
    elif span_hours < 48:
        span_text = f"{span_hours / 24:.1f}d"
    else:
        span_days = span_hours / 24
        start_dt = datetime.fromtimestamp(t_min, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(t_max, tz=timezone.utc)
        span_text = f"{start_dt.strftime('%b %d')} — {end_dt.strftime('%b %d')} ({span_days:.0f}d)"

    svg = f'''
    <svg width="100%" height="{_CHART_H}" viewBox="0 0 {_CHART_W} {_CHART_H}"
         xmlns="http://www.w3.org/2000/svg" style="display:block;"
         preserveAspectRatio="xMidYMid meet">
        <defs>
            <linearGradient id="{grad_id}" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="{grad_top}" />
                <stop offset="100%" stop-color="rgba(0,0,0,0)" />
            </linearGradient>
        </defs>
        <!-- Plot background -->
        <rect x="{_PAD_L}" y="{_PAD_T}" width="{plot_w}" height="{plot_h}"
              fill="#0f1218" rx="2" />
        <!-- Grid -->
        {y_labels}
        {x_labels}
        <!-- Fill area under curve -->
        <path d="{fill_path}" fill="url(#{grad_id})" />
        <!-- NAV line -->
        <polyline fill="none" stroke="{color}" stroke-width="2"
                  stroke-linejoin="round" stroke-linecap="round"
                  points="{points_str}" />
        <!-- Current value dot -->
        <circle cx="{pts[-1][0]:.1f}" cy="{pts[-1][1]:.1f}" r="3" fill="{color}" />
    </svg>
    '''

    return svg, span_text, end_nav, delta, delta_pct, delta_color, delta_sign


def render_equity_curve(nav_log_path: Optional[Path] = None) -> None:
    """Render the equity curve widget in the dashboard.

    Reads logs/nav_log.json, downsamples, and renders an SVG chart.
    If data is insufficient (<2 points), renders a placeholder.
    """
    entries = _load_nav_series(nav_log_path)

    if len(entries) < 2:
        st.caption("Equity Curve — awaiting NAV history")
        return

    sampled = _downsample(entries, max_points=300)
    result = _build_equity_svg(sampled)

    if not result:
        return

    svg, span_text, current_nav, delta, delta_pct, delta_color, delta_sign = result

    html = f'''
    <div style="
        background: linear-gradient(135deg, #1a1d24 0%, #12141a 100%);
        border: 1px solid #2d3139;
        border-radius: 8px;
        padding: 20px;
        margin: 8px 0;
        width: 100%;
        box-sizing: border-box;
    ">
        <!-- Header row -->
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px;">
            <div style="display: flex; align-items: baseline; gap: 12px;">
                <span style="font-size: 0.8em; color: #888; text-transform: uppercase; letter-spacing: 0.5px;">
                    Equity Curve
                </span>
                <span style="font-size: 0.7em; color: #555;">
                    {span_text}
                </span>
            </div>
            <div style="display: flex; align-items: baseline; gap: 16px;">
                <span style="font-size: 1.2em; font-weight: 700; color: #ccc;">
                    ${current_nav:,.2f}
                </span>
                <span style="font-size: 0.9em; font-weight: 600; color: {delta_color};">
                    {delta_sign}${abs(delta):,.2f} ({delta_sign}{abs(delta_pct):.2f}%)
                </span>
            </div>
        </div>
        <!-- Chart -->
        <div style="width: 100%; overflow: hidden;">
            {svg}
        </div>
    </div>
    '''

    iframe_h = _CHART_H + 110  # header + chart + padding
    st_components.html(html, height=iframe_h, scrolling=False)
