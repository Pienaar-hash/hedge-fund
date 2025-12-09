"""
KPI Strip Component — Single HTML render for top-level metrics.

Renders 8 KPI cards in a horizontal strip:
    NAV, AUM, 24h PnL, Unrealized, Exposure%, Drawdown, ATR, Risk
"""
from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st


def _value_class(value: float, positive_is_good: bool = True) -> str:
    """Return CSS class based on value sign."""
    if value > 0:
        return "positive" if positive_is_good else "negative"
    elif value < 0:
        return "negative" if positive_is_good else "positive"
    return ""


def _status_class(status: str) -> str:
    """Return CSS class for status badges."""
    s = str(status).lower() if status else "normal"
    if s in ("normal", "ok", "low", "healthy", "good", "excellent"):
        return "normal"
    elif s in ("elevated", "medium", "fair", "warning"):
        return "warning"
    return "critical"


def _fmt_usd(val: Any, decimals: int = 0) -> str:
    """Format value as USD string."""
    try:
        num = float(val) if val is not None else 0.0
        return f"${num:,.{decimals}f}"
    except Exception:
        return "$0"


def build_kpi_cards(
    nav_state: Dict[str, Any],
    aum_data: Dict[str, Any],
    kpis: Dict[str, Any],
    risk_snapshot: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Build list of KPI card data for rendering."""
    # Extract values
    nav_usd = float(nav_state.get("nav_usd") or nav_state.get("nav") or nav_state.get("total_equity") or 0)
    total_aum = float(aum_data.get("total_usd") or aum_data.get("total") or nav_usd)
    daily_pnl = float(nav_state.get("realized_pnl_today") or 0)
    unrealized = float(nav_state.get("unrealized_pnl") or 0)
    gross_exp = float(nav_state.get("gross_exposure") or 0)
    drawdown_pct = float(nav_state.get("drawdown_pct") or 0)
    
    # Risk metrics
    risk_block = kpis.get("risk", {})
    atr_regime_raw = risk_snapshot.get("atr_regime") or risk_block.get("atr_regime") or "normal"
    dd_state_raw = risk_snapshot.get("dd_state") or risk_block.get("dd_state") or "normal"
    
    # Normalize nested dicts
    if isinstance(atr_regime_raw, dict):
        atr_regime = str(atr_regime_raw.get("atr_regime") or atr_regime_raw.get("regime") or "normal")
    else:
        atr_regime = str(atr_regime_raw) if atr_regime_raw else "normal"
    
    if isinstance(dd_state_raw, dict):
        dd_state = str(dd_state_raw.get("dd_state") or dd_state_raw.get("state") or "normal")
    else:
        dd_state = str(dd_state_raw) if dd_state_raw else "normal"
    
    # Exposure percentage
    exp_pct = (gross_exp / nav_usd * 100) if nav_usd > 0 else 0
    
    cards = [
        {
            "label": "NAV",
            "value_html": _fmt_usd(nav_usd),
            "value_class": "",
        },
        {
            "label": "AUM",
            "value_html": _fmt_usd(total_aum),
            "value_class": "",
        },
        {
            "label": "24h PnL",
            "value_html": ("+" if daily_pnl >= 0 else "") + _fmt_usd(daily_pnl),
            "value_class": _value_class(daily_pnl),
        },
        {
            "label": "Unrealized",
            "value_html": ("+" if unrealized >= 0 else "") + _fmt_usd(unrealized),
            "value_class": _value_class(unrealized),
        },
        {
            "label": "Exposure",
            "value_html": f"{exp_pct:.0f}%",
            "value_class": "",
        },
        {
            "label": "Drawdown",
            "value_html": f"{drawdown_pct:.2f}%",
            "value_class": "negative" if drawdown_pct > 5 else "",
        },
        {
            "label": "ATR",
            "value_html": f'<span class="status-badge {_status_class(atr_regime)}">{atr_regime.upper()}</span>',
            "value_class": "",
            "is_badge": True,
        },
        {
            "label": "Risk",
            "value_html": f'<span class="status-badge {_status_class(dd_state)}">{dd_state.upper()}</span>',
            "value_class": "",
            "is_badge": True,
        },
    ]
    
    return cards


def render_kpi_strip(
    nav_state: Dict[str, Any],
    aum_data: Dict[str, Any],
    kpis: Dict[str, Any],
    risk_snapshot: Dict[str, Any],
) -> None:
    """
    Render KPI strip as single HTML block.
    
    CRITICAL: Build one HTML string, render with one st.markdown call.
    This prevents raw HTML tags from appearing in the UI.
    """
    cards = build_kpi_cards(nav_state, aum_data, kpis, risk_snapshot)
    
    # Build single HTML string
    html_parts = ['<div class="kpi-strip">']
    
    for card in cards:
        is_badge = card.get("is_badge", False)
        value_class = card.get("value_class", "")
        
        if is_badge:
            # Badge cards render the badge HTML directly
            html_parts.append(f'''
                <div class="kpi-card">
                    <div class="kpi-card-label">{card["label"]}</div>
                    <div class="kpi-card-value">{card["value_html"]}</div>
                </div>
            ''')
        else:
            # Regular cards with optional value class
            cls = f'kpi-card-value {value_class}' if value_class else 'kpi-card-value'
            html_parts.append(f'''
                <div class="kpi-card">
                    <div class="kpi-card-label">{card["label"]}</div>
                    <div class="{cls}">{card["value_html"]}</div>
                </div>
            ''')
    
    html_parts.append('</div>')
    
    # Single render call
    st.html("".join(html_parts))
