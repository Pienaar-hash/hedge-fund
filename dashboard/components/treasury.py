"""
Treasury Component — Off-exchange holdings display.
"""
from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st


def render_treasury_block(
    holdings: List[Dict[str, Any]],
    total_usd: float = 0.0,
) -> None:
    """Render treasury / off-exchange holdings with compact table."""
    # Empty state
    if not holdings:
        st.html('''
        <div class="quant-card empty-card">
            <div class="empty-title">No off-exchange holdings</div>
            <div class="empty-subtitle">Configure in config/offexchange_holdings.json</div>
        </div>
        ''')
        return
    
    # Build rows
    rows_html = []
    for h in holdings:
        asset = h.get("asset") or h.get("symbol") or "?"
        qty = float(h.get("quantity") or h.get("qty") or h.get("amount") or 0)
        usd_value = float(h.get("usd_value") or h.get("value_usd") or 0)
        location = h.get("location") or h.get("wallet") or "External"
        
        rows_html.append(f'''
        <tr>
            <td>
                <span class="symbol-name">{asset}</span>
                <span class="text-muted text-xs">{location}</span>
            </td>
            <td class="text-mono text-right">{qty:,.4g}</td>
            <td class="text-mono text-right">${usd_value:,.2f}</td>
        </tr>
        ''')
    
    # Recalculate total if not provided
    if total_usd == 0.0:
        total_usd = sum(float(h.get("usd_value") or h.get("value_usd") or 0) for h in holdings)
    
    html = f'''
    <div class="quant-card">
        <table class="quant-table">
            <thead>
                <tr>
                    <th>Asset</th>
                    <th class="text-right">Quantity</th>
                    <th class="text-right">USD Value</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows_html)}
            </tbody>
            <tfoot>
                <tr>
                    <td colspan="2" class="text-muted">{len(holdings)} holdings</td>
                    <td class="text-right font-weight-700">${total_usd:,.2f}</td>
                </tr>
            </tfoot>
        </table>
    </div>
    '''
    
    st.html(html)
