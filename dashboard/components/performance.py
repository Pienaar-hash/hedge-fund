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
    """Render strategy performance with 4 KPI cards + mini equity curve."""
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
    
    # Extract equity curve for sparkline
    sparkline_html = ""
    if equity_curve and len(equity_curve) > 1:
        try:
            equity_values = [float(e.get("equity") or e.get("nav") or 0) for e in equity_curve]
            sparkline_html = _sparkline_svg(equity_values, width=120, height=32)
        except Exception:
            pass
    
    # Format helper
    def fmt_pnl(val: float) -> tuple:
        cls = "text-positive" if val >= 0 else "text-negative"
        sign = "+" if val >= 0 else ""
        return cls, f"{sign}${abs(val):,.0f}"
    
    # Build cards
    d_cls, d_val = fmt_pnl(daily_pnl)
    w_cls, w_val = fmt_pnl(weekly_pnl)
    m_cls, m_val = fmt_pnl(monthly_pnl)
    t_cls, t_val = fmt_pnl(total_pnl)
    
    win_cls = "text-positive" if win_rate >= 50 else "text-warning"
    sharpe_cls = "text-positive" if sharpe >= 1 else ("text-warning" if sharpe >= 0 else "text-negative")
    
    html = f'''
    <div class="quant-card">
        <div class="performance-grid">
            <!-- PnL cards row -->
            <div class="perf-card">
                <div class="perf-label">24h PnL</div>
                <div class="perf-value {d_cls}">{d_val}</div>
            </div>
            <div class="perf-card">
                <div class="perf-label">7d PnL</div>
                <div class="perf-value {w_cls}">{w_val}</div>
            </div>
            <div class="perf-card">
                <div class="perf-label">30d PnL</div>
                <div class="perf-value {m_cls}">{m_val}</div>
            </div>
            <div class="perf-card perf-card-highlight">
                <div class="perf-label">All-Time PnL</div>
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
        
        <!-- Equity sparkline -->
        <div class="equity-sparkline-row">
            <span class="sparkline-label">Equity Curve</span>
            {sparkline_html if sparkline_html else '<span class="text-muted text-xs">Insufficient data</span>'}
        </div>
    </div>
    '''
    
    st.html(html)
