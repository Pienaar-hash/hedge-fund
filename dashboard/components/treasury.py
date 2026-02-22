"""
Treasury Component — Off-exchange holdings display.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st


def render_treasury_block(
    holdings: List[Dict[str, Any]],
    total_usd: float = 0.0,
    updated_ts: Optional[str] = None,
) -> None:
    """Render treasury / off-exchange holdings with compact table and staleness indicator."""
    # Empty state
    if not holdings:
        st.html('''
        <div class="quant-card empty-card">
            <div class="empty-title">No off-exchange holdings</div>
            <div class="empty-subtitle">Configure in config/offexchange_holdings.json</div>
        </div>
        ''')
        return
    
    # Calculate staleness
    stale_warning = ""
    if updated_ts:
        try:
            # Parse timestamp
            if isinstance(updated_ts, (int, float)):
                last_update = datetime.fromtimestamp(updated_ts)
            else:
                # ISO format
                ts_str = str(updated_ts).replace("+00:00", "").replace("Z", "")
                last_update = datetime.fromisoformat(ts_str)
            
            age_days = (datetime.now() - last_update).days
            
            if age_days > 7:
                stale_warning = f'''
                <div style="
                    background: #ef444422;
                    border: 1px solid #ef4444;
                    border-radius: 4px;
                    padding: 8px 12px;
                    margin-bottom: 12px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                ">
                    <span style="color: #ef4444; font-size: 0.85em;">
                        ⚠️ <strong>STALE DATA</strong> — Last updated {last_update.strftime('%Y-%m-%d')} ({age_days} days ago)
                    </span>
                    <span style="color: #888; font-size: 0.75em;">
                        Offchain reporting paused
                    </span>
                </div>
                '''
            elif age_days > 1:
                stale_warning = f'''
                <div style="
                    background: #f59e0b22;
                    border: 1px solid #f59e0b;
                    border-radius: 4px;
                    padding: 6px 12px;
                    margin-bottom: 12px;
                    font-size: 0.8em;
                    color: #f59e0b;
                ">
                    Last updated: {last_update.strftime('%Y-%m-%d %H:%M')} ({age_days}d ago)
                </div>
                '''
        except Exception:
            pass
    
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
        {stale_warning}
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
