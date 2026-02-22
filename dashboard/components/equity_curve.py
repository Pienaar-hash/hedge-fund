"""
Equity Curve Component — Portfolio NAV over time.

Renders an SVG line chart from logs/nav_log.json.
Single line: Portfolio NAV ($). Clean, no gradients, no overlays.

Data source: logs/nav_log.json — array of {nav, t, unrealized_pnl}
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

_NAV_LOG_PATH = Path("logs/nav_log.json")

# Chart dimensions
_CHART_W = 720
_CHART_H = 200
_PAD_L = 70   # left padding for y-axis labels
_PAD_R = 20
_PAD_T = 20
_PAD_B = 40   # bottom for x-axis labels


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


def _build_equity_svg(entries: List[Dict[str, Any]]) -> str:
    """Build an SVG equity curve chart.

    Returns empty string if insufficient data.
    """
    if len(entries) < 2:
        return ""

    navs = [float(e["nav"]) for e in entries]
    times = [float(e["t"]) for e in entries]

    nav_min = min(navs)
    nav_max = max(navs)
    t_min = times[0]
    t_max = times[-1]

    # Avoid division by zero
    nav_range = nav_max - nav_min if nav_max != nav_min else 1.0
    t_range = t_max - t_min if t_max != t_min else 1.0

    plot_w = _CHART_W - _PAD_L - _PAD_R
    plot_h = _CHART_H - _PAD_T - _PAD_B

    # Scale helpers
    def sx(t: float) -> float:
        return _PAD_L + ((t - t_min) / t_range) * plot_w

    def sy(nav: float) -> float:
        return _PAD_T + plot_h - ((nav - nav_min) / nav_range) * plot_h

    # Build polyline points
    points = " ".join(f"{sx(t):.1f},{sy(n):.1f}" for t, n in zip(times, navs))

    # Determine line color: green if up, red if down
    color = "#10b981" if navs[-1] >= navs[0] else "#ef4444"

    # Y-axis labels (5 ticks)
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
            f'stroke="#1a1d24" stroke-width="1" />\n'
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
            f'<text x="{x}" y="{_CHART_H - 5}" '
            f'text-anchor="middle" fill="#666" font-size="11" '
            f'font-family="monospace">{label}</text>\n'
        )

    # NAV change stats
    start_nav = navs[0]
    end_nav = navs[-1]
    delta = end_nav - start_nav
    delta_pct = (delta / start_nav * 100) if start_nav > 0 else 0
    delta_sign = "+" if delta >= 0 else ""
    delta_color = "#10b981" if delta >= 0 else "#ef4444"

    # Span display
    if span_hours < 24:
        span_text = f"{span_hours:.1f}h"
    else:
        span_text = f"{span_hours / 24:.1f}d"

    svg = f'''
    <svg width="100%" viewBox="0 0 {_CHART_W} {_CHART_H}" xmlns="http://www.w3.org/2000/svg"
         style="background: transparent;">
        <!-- Grid -->
        {y_labels}
        {x_labels}
        <!-- Plot border -->
        <rect x="{_PAD_L}" y="{_PAD_T}" width="{plot_w}" height="{plot_h}"
              fill="none" stroke="#1a1d24" stroke-width="1" />
        <!-- NAV line -->
        <polyline fill="none" stroke="{color}" stroke-width="1.5"
                  stroke-linejoin="round" stroke-linecap="round"
                  points="{points}" />
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
        st.html('''
        <div class="quant-card" style="padding: 16px; text-align: center;">
            <div class="section-header">
                <h2>Equity Curve</h2>
            </div>
            <div style="color: #555; padding: 32px 0;">
                Insufficient NAV history — awaiting data collection.
            </div>
        </div>
        ''')
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
        padding: 16px;
        margin: 8px 0;
    ">
        <!-- Header row -->
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
            <div style="display: flex; align-items: baseline; gap: 12px;">
                <span style="font-size: 0.75em; color: #888; text-transform: uppercase; letter-spacing: 0.5px;">
                    Equity Curve
                </span>
                <span style="font-size: 0.7em; color: #555;">
                    {span_text} span
                </span>
            </div>
            <div style="display: flex; align-items: baseline; gap: 16px;">
                <span style="font-size: 1.1em; font-weight: 700; color: #ccc;">
                    ${current_nav:,.2f}
                </span>
                <span style="font-size: 0.85em; font-weight: 600; color: {delta_color};">
                    {delta_sign}${abs(delta):,.2f} ({delta_sign}{delta_pct:.2f}%)
                </span>
            </div>
        </div>
        <!-- Chart -->
        {svg}
    </div>
    '''

    st.html(html)
