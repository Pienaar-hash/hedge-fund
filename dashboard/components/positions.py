"""
Positions Component — Clean positions table with compact empty state.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st


def _format_age(age_seconds: Optional[float]) -> str:
    """Format age in human-readable form."""
    if age_seconds is None:
        return "n/a"
    try:
        age = float(age_seconds)
    except Exception:
        return "n/a"
    if age < 60:
        return f"{age:.0f}s"
    if age < 3600:
        return f"{age/60:.1f}m"
    return f"{age/3600:.1f}h"


def render_positions_block(
    positions: List[Dict[str, Any]],
    meta: Dict[str, Any],
) -> None:
    """Render positions block with institutional table styling."""
    # Section header
    st.html('''
    <div class="section-header">
        <svg class="section-header-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="12" y1="20" x2="12" y2="10"/>
            <line x1="18" y1="20" x2="18" y2="4"/>
            <line x1="6" y1="20" x2="6" y2="16"/>
        </svg>
        <h2>Positions</h2>
    </div>
    ''')
    
    # Empty state - compact, not a giant block
    if not positions:
        st.html('''
        <div class="quant-card empty-card">
            <div class="empty-title">No open positions</div>
            <div class="empty-subtitle">Engine is running, awaiting signals.</div>
        </div>
        ''')
        return
    
    # Calculate totals
    total_unrealized = 0.0
    long_count = 0
    short_count = 0
    
    rows_html = []
    for pos in positions:
        symbol = pos.get("symbol", "?")
        side = pos.get("positionSide") or pos.get("side") or "LONG"
        qty = float(pos.get("qty") or pos.get("positionAmt") or 0)
        entry = float(pos.get("entryPrice") or pos.get("entry") or 0)
        mark = float(pos.get("markPrice") or pos.get("mark") or 0)
        unrealized = float(pos.get("unrealized") or pos.get("pnl") or pos.get("unRealizedProfit") or 0)
        
        total_unrealized += unrealized
        is_long = side.upper() == "LONG"
        if is_long:
            long_count += 1
        else:
            short_count += 1
        
        # Direction badge
        dir_class = "normal" if is_long else "critical"
        dir_text = "LONG" if is_long else "SHORT"
        
        # PnL class
        pnl_class = "text-positive" if unrealized >= 0 else "text-negative"
        pnl_sign = "+" if unrealized >= 0 else ""
        
        # PnL percentage
        pnl_pct = ""
        if entry > 0 and qty != 0:
            pct = ((mark - entry) / entry) * 100
            if not is_long:
                pct = -pct
            pnl_pct = f" ({pct:+.2f}%)"
        
        display_symbol = symbol.replace("USDT", "")
        
        rows_html.append(f'''
        <tr>
            <td>
                <span class="symbol-name">{display_symbol}</span>
                <span class="status-badge {dir_class}">{dir_text}</span>
            </td>
            <td class="text-mono text-right">{abs(qty):.4g}</td>
            <td class="text-mono text-right">${entry:,.2f}</td>
            <td class="text-mono text-right">${mark:,.2f}</td>
            <td class="text-right {pnl_class}">
                <span class="font-weight-600">{pnl_sign}${abs(unrealized):,.2f}</span>
                <span class="text-muted text-xs">{pnl_pct}</span>
            </td>
        </tr>
        ''')
    
    # Total row styling
    total_class = "text-positive" if total_unrealized >= 0 else "text-negative"
    total_sign = "+" if total_unrealized >= 0 else ""
    
    # Data age
    age_s = meta.get("data_age_s")
    
    # Single HTML render
    html = f'''
    <div class="quant-card">
        <table class="quant-table">
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th class="text-right">Size</th>
                    <th class="text-right">Entry</th>
                    <th class="text-right">Mark</th>
                    <th class="text-right">Unrealized</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows_html)}
            </tbody>
            <tfoot>
                <tr>
                    <td colspan="4" class="text-muted">{len(positions)} positions ({long_count}L / {short_count}S)</td>
                    <td class="text-right {total_class}">
                        <span class="font-weight-700">{total_sign}${abs(total_unrealized):,.2f}</span>
                    </td>
                </tr>
            </tfoot>
        </table>
        <div class="table-footer text-muted text-xs">Data age: {_format_age(age_s)}</div>
    </div>
    '''
    
    st.html(html)
