"""
KPI Strip Component — Single HTML render for top-level metrics.

Renders 8 KPI cards in a horizontal strip:
    NAV, AUM, 24h PnL, Unrealized, Exposure%, Drawdown, ATR, Risk
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

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
    episode_ledger: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """Build list of KPI card data for rendering."""
    # Extract values
    nav_usd = float(nav_state.get("nav_usd") or nav_state.get("nav") or nav_state.get("total_equity") or 0)
    total_aum = float(aum_data.get("total_usd") or aum_data.get("total") or nav_usd)

    # 24h PnL = NAV delta (current - 24h ago) — NOT episode-windowed PnL.
    # Episode PnL only counts closed trades, missing unrealised gains and
    # mark-to-market changes on holdings. NAV delta is the true portfolio PnL.
    #
    # SPAN AUTHORITY: Only display NAV delta if log span is sufficient.
    # Silent fallback to a stale delta is a structural violation.
    from dashboard.components.nav_pnl import compute_nav_deltas
    _nav_deltas = compute_nav_deltas()
    _span_ok = _nav_deltas.get("span_ok", {})
    gross_exp = float(nav_state.get("gross_exposure") or 0)
    drawdown_pct = float(nav_state.get("drawdown_pct") or 0)
    
    # Risk metrics
    risk_block = kpis.get("risk", {})
    dd_state_raw = risk_snapshot.get("dd_state") or risk_block.get("dd_state") or "normal"
    
    # Normalize nested dict
    if isinstance(dd_state_raw, dict):
        dd_state = str(dd_state_raw.get("dd_state") or dd_state_raw.get("state") or "normal")
    else:
        dd_state = str(dd_state_raw) if dd_state_raw else "normal"
    
    # Exposure percentage
    exp_pct = (gross_exp / nav_usd * 100) if nav_usd > 0 else 0
    
    # All-time PnL from episode ledger (authoritative)
    _at_pnl = 0.0
    if episode_ledger:
        _at_pnl = float(episode_ledger.get("stats", {}).get("total_net_pnl") or 0)
    _win_rate = 0.0
    if episode_ledger:
        _win_rate = float(episode_ledger.get("stats", {}).get("win_rate") or 0)

    cards = [
        {
            "label": "NAV",
            "value_html": _fmt_usd(nav_usd),
            "value_class": "",
        },
        {
            "label": "All-Time PnL",
            "value_html": ("+" if _at_pnl >= 0 else "") + _fmt_usd(_at_pnl),
            "value_class": _value_class(_at_pnl),
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
            "label": "Win Rate",
            "value_html": f"{_win_rate:.1f}%",
            "value_class": "positive" if _win_rate >= 50 else "",
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
    episode_ledger: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Render KPI strip as single HTML block.
    
    CRITICAL: Build one HTML string, render with one st.markdown call.
    This prevents raw HTML tags from appearing in the UI.
    """
    cards = build_kpi_cards(nav_state, aum_data, kpis, risk_snapshot, episode_ledger=episode_ledger)
    
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
