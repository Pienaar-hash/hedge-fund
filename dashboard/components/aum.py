"""
AUM Component — AUM display with allocation breakdown.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st


def render_aum_block(
    nav_state: Dict[str, Any],
    aum_data: Optional[Dict[str, Any]] = None,
) -> None:
    """Render AUM section with allocation doughnut-style visualization."""
    # Section header
    st.html('''
    <div class="section-header">
        <svg class="section-header-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/>
            <path d="M12 2a10 10 0 0 1 10 10"/>
            <line x1="12" y1="12" x2="12" y2="2"/>
            <line x1="12" y1="12" x2="22" y2="12"/>
        </svg>
        <h2>Assets Under Management</h2>
    </div>
    ''')
    
    # Extract values - handle both old and new field names
    nav = float(nav_state.get("nav_usd") or nav_state.get("nav") or nav_state.get("total_equity") or nav_state.get("total_wallet_balance") or 0)
    
    # AUM breakdown - handle load_all_state() format
    if aum_data:
        # New format from load_all_state(): total_usd, slices
        total_aum = float(aum_data.get("total_usd") or aum_data.get("total_aum") or 0)
        
        # Get futures from slices or direct field
        slices = aum_data.get("slices", [])
        futures = 0.0
        treasury = 0.0
        for s in slices:
            label = s.get("label", "")
            usd = float(s.get("usd") or 0)
            if label == "Futures":
                futures = usd
            else:
                treasury += usd  # All non-futures are "treasury"
        
        # Fallback to old field names if slices empty
        if futures == 0 and treasury == 0:
            futures = float(aum_data.get("futures_nav") or nav)
            treasury = float(aum_data.get("treasury_usd") or aum_data.get("off_exchange") or 0)
            total_aum = futures + treasury if total_aum == 0 else total_aum
    else:
        futures = nav
        treasury = 0
        total_aum = nav
    
    # Calculate percentages
    futures_pct = (futures / total_aum * 100) if total_aum > 0 else 100
    treasury_pct = (treasury / total_aum * 100) if total_aum > 0 else 0
    
    # SVG doughnut chart (simple two-segment arc)
    def svg_doughnut(pct1: float, pct2: float, r: int = 40, stroke: int = 12) -> str:
        """Generate a simple two-segment doughnut SVG."""
        if pct1 <= 0:
            return ""
        
        # Circumference
        c = 2 * 3.14159 * r
        
        # Calculate stroke-dasharray for each segment
        seg1 = (pct1 / 100) * c
        seg2 = (pct2 / 100) * c
        
        return f'''
        <svg class="aum-doughnut" viewBox="0 0 100 100" width="100" height="100">
            <!-- Background ring -->
            <circle cx="50" cy="50" r="{r}" fill="none" stroke="#1f2937" stroke-width="{stroke}"/>
            <!-- Futures segment (blue) -->
            <circle cx="50" cy="50" r="{r}" fill="none" stroke="#1a56db" stroke-width="{stroke}"
                    stroke-dasharray="{seg1:.1f} {c:.1f}"
                    stroke-dashoffset="0"
                    transform="rotate(-90 50 50)"/>
            <!-- Treasury segment (teal) -->
            <circle cx="50" cy="50" r="{r}" fill="none" stroke="#14b8a6" stroke-width="{stroke}"
                    stroke-dasharray="{seg2:.1f} {c:.1f}"
                    stroke-dashoffset="-{seg1:.1f}"
                    transform="rotate(-90 50 50)"/>
        </svg>
        '''
    
    doughnut_html = svg_doughnut(futures_pct, treasury_pct)
    
    html = f'''
    <div class="quant-card aum-card">
        <div class="aum-grid">
            <div class="aum-chart-container">
                {doughnut_html}
                <div class="aum-total">
                    <div class="aum-total-label">Total AUM</div>
                    <div class="aum-total-value">${total_aum:,.0f}</div>
                </div>
            </div>
            <div class="aum-breakdown">
                <div class="aum-row">
                    <span class="aum-indicator aum-indicator-futures"></span>
                    <span class="aum-label">Futures NAV</span>
                    <span class="aum-value">${futures:,.0f}</span>
                    <span class="aum-pct text-muted">{futures_pct:.1f}%</span>
                </div>
                <div class="aum-row">
                    <span class="aum-indicator aum-indicator-treasury"></span>
                    <span class="aum-label">Treasury</span>
                    <span class="aum-value">${treasury:,.0f}</span>
                    <span class="aum-pct text-muted">{treasury_pct:.1f}%</span>
                </div>
            </div>
        </div>
    </div>
    '''
    
    st.html(html)
