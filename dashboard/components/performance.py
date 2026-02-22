"""
Performance Component — Replace giant red bar with compact cards + sparklines.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st


def _sparkline_svg(values: List[float], width: int = 80, height: int = 24) -> str:
    """Generate a compact sparkline SVG."""
    if not values or len(values) < 2:
        return ""
    
    min_v = min(values)
    max_v = max(values)
    val_range = max_v - min_v if max_v != min_v else 1
    
    # Normalize and create points
    points = []
    for i, v in enumerate(values):
        x = (i / (len(values) - 1)) * width
        y = height - ((v - min_v) / val_range) * (height - 4)
        points.append(f"{x:.1f},{y:.1f}")
    
    # Determine trend color
    color = "#10b981" if values[-1] >= values[0] else "#ef4444"
    
    return f'''
    <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" class="sparkline">
        <polyline fill="none" stroke="{color}" stroke-width="1.5" points="{' '.join(points)}" />
    </svg>
    '''


def render_performance_block(
    kpis: Dict[str, Any],
    equity_curve: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Render strategy performance with 4 KPI cards."""
    # Section header
    st.html('''
    <div class="section-header">
        <svg class="section-header-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
        </svg>
        <h2>Strategy Performance</h2>
    </div>
    ''')
    
    # Extract KPIs with sensible defaults
    daily_pnl = float(kpis.get("daily_pnl") or kpis.get("pnl_24h") or 0)
    weekly_pnl = float(kpis.get("weekly_pnl") or kpis.get("pnl_7d") or 0)
    monthly_pnl = float(kpis.get("monthly_pnl") or kpis.get("pnl_30d") or 0)
    total_pnl = float(kpis.get("total_pnl") or kpis.get("all_time_pnl") or 0)
    win_rate = float(kpis.get("win_rate") or kpis.get("win_rate_pct") or 0)
    sharpe = float(kpis.get("sharpe") or kpis.get("sharpe_ratio") or 0)
    max_dd = float(kpis.get("max_drawdown") or kpis.get("max_dd") or 0)
    trades = int(kpis.get("total_trades") or kpis.get("trades_count") or 0)
    
    # When PnL windows are suppressed (NAV span too short), show "—" instead of $0
    _suppressed = kpis.get("_nav_span_suppressed_windows", set())
    _daily_suppressed = "24h" in _suppressed
    _weekly_suppressed = "7d" in _suppressed
    _monthly_suppressed = "30d" in _suppressed
    _alltime_suppressed = "all-time" in _suppressed
    
    # Format helper
    def fmt_pnl(val: float, suppressed: bool = False) -> tuple:
        if suppressed and val == 0:
            return "text-muted", "—"
        cls = "text-positive" if val >= 0 else "text-negative"
        sign = "+" if val >= 0 else "-"
        return cls, f"{sign}${abs(val):,.0f}"
    
    # Build cards
    d_cls, d_val = fmt_pnl(daily_pnl, _daily_suppressed)
    w_cls, w_val = fmt_pnl(weekly_pnl, _weekly_suppressed)
    m_cls, m_val = fmt_pnl(monthly_pnl, _monthly_suppressed)
    t_cls, t_val = fmt_pnl(total_pnl, _alltime_suppressed)
    
    win_cls = "text-positive" if win_rate >= 50 else "text-warning"
    sharpe_cls = "text-positive" if sharpe >= 1 else ("text-warning" if sharpe >= 0 else "text-negative")
    
    # Span diagnostic note (injected by layout when NAV windows suppressed)
    span_note = kpis.get("_nav_span_note", "")
    span_html = ""
    if span_note:
        span_html = f'<div style="color:#f59e0b;font-size:11px;margin-top:6px;text-align:center;">⚠ {span_note}</div>'

    # Labels that clarify data source when suppressed
    d_label = "24h PnL" if not _daily_suppressed else "24h PnL<br><span style='font-size:9px;color:#555'>awaiting span</span>"
    w_label = "7d PnL" if not _weekly_suppressed else "7d PnL<br><span style='font-size:9px;color:#555'>awaiting span</span>"
    m_label = "30d PnL" if not _monthly_suppressed else "30d PnL<br><span style='font-size:9px;color:#555'>awaiting span</span>"
    t_label = "All-Time PnL"

    html = f'''
    <div class="quant-card">
        <div class="performance-grid">
            <!-- PnL cards row -->
            <div class="perf-card">
                <div class="perf-label">{d_label}</div>
                <div class="perf-value {d_cls}">{d_val}</div>
            </div>
            <div class="perf-card">
                <div class="perf-label">{w_label}</div>
                <div class="perf-value {w_cls}">{w_val}</div>
            </div>
            <div class="perf-card">
                <div class="perf-label">{m_label}</div>
                <div class="perf-value {m_cls}">{m_val}</div>
            </div>
            <div class="perf-card perf-card-highlight">
                <div class="perf-label">{t_label}</div>
                <div class="perf-value {t_cls}">{t_val}</div>
            </div>
            
            <!-- Stats cards row -->
            <div class="perf-card">
                <div class="perf-label">Win Rate</div>
                <div class="perf-value {win_cls}">{win_rate:.1f}%</div>
            </div>
            <div class="perf-card">
                <div class="perf-label">Sharpe</div>
                <div class="perf-value {sharpe_cls}">{sharpe:.2f}</div>
            </div>
            <div class="perf-card">
                <div class="perf-label">Max DD</div>
                <div class="perf-value text-negative">-{abs(max_dd):.2f}%</div>
            </div>
            <div class="perf-card">
                <div class="perf-label">Trades</div>
                <div class="perf-value">{trades:,}</div>
            </div>
        </div>
        {span_html}
    </div>
    '''
    
    st.html(html)
