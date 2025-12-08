"""
Equity Panel (v7) - Equity curve visualization.

Displays:
- Cumulative equity/PnL line chart
- Underwater (drawdown) curve
- Rolling PnL bar chart
- Returns histogram
- Summary statistics
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# ---------------------------------------------------------------------------
# State file loading
# ---------------------------------------------------------------------------
STATE_DIR = Path(os.getenv("STATE_DIR") or "logs/state")
EQUITY_PATH = STATE_DIR / "equity.json"


def load_equity_state(state_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load equity series from state file."""
    base_dir = state_dir or STATE_DIR
    path = base_dir / "equity.json"
    try:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------
COLOR_PROFIT = "#21c354"
COLOR_LOSS = "#d94a4a"
COLOR_DRAWDOWN = "#ff8c42"
COLOR_NEUTRAL = "#888888"
COLOR_LINE = "#00d4ff"


# ---------------------------------------------------------------------------
# Summary Stats
# ---------------------------------------------------------------------------
def render_equity_stats(equity_data: Dict[str, Any]) -> None:
    """Render equity summary statistics."""
    st.markdown("#### 📊 Performance Summary")
    
    total_pnl = equity_data.get("total_pnl", 0.0)
    mean_pnl = equity_data.get("mean_pnl", 0.0)
    std_pnl = equity_data.get("std_pnl", 0.0)
    max_dd = equity_data.get("max_drawdown", 0.0)
    win_rate = equity_data.get("win_rate", 0.0)
    record_count = equity_data.get("record_count", 0)
    
    # Calculate Sharpe-like ratio (simplified)
    sharpe = (mean_pnl / std_pnl) if std_pnl > 0 else 0.0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pnl_color = COLOR_PROFIT if total_pnl >= 0 else COLOR_LOSS
        st.markdown(
            f"""
            <div style="text-align: center; padding: 10px; border-radius: 8px; 
                        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                        border: 1px solid {pnl_color};">
                <div style="font-size: 12px; color: #888;">Total PnL</div>
                <div style="font-size: 24px; font-weight: bold; color: {pnl_color};">
                    ${total_pnl:+,.2f}
                </div>
                <div style="font-size: 11px; color: #666;">{record_count} trades</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col2:
        wr_color = COLOR_PROFIT if win_rate >= 0.5 else COLOR_LOSS
        st.markdown(
            f"""
            <div style="text-align: center; padding: 10px; border-radius: 8px;
                        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                        border: 1px solid #333;">
                <div style="font-size: 12px; color: #888;">Win Rate</div>
                <div style="font-size: 24px; font-weight: bold; color: {wr_color};">
                    {win_rate * 100:.1f}%
                </div>
                <div style="font-size: 11px; color: #666;">avg ${mean_pnl:+.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col3:
        dd_color = COLOR_PROFIT if max_dd < 0.10 else (COLOR_DRAWDOWN if max_dd < 0.20 else COLOR_LOSS)
        st.markdown(
            f"""
            <div style="text-align: center; padding: 10px; border-radius: 8px;
                        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                        border: 1px solid #333;">
                <div style="font-size: 12px; color: #888;">Max Drawdown</div>
                <div style="font-size: 24px; font-weight: bold; color: {dd_color};">
                    {max_dd * 100:.1f}%
                </div>
                <div style="font-size: 11px; color: #666;">peak-to-trough</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col4:
        sharpe_color = COLOR_PROFIT if sharpe >= 1.0 else (COLOR_NEUTRAL if sharpe >= 0 else COLOR_LOSS)
        st.markdown(
            f"""
            <div style="text-align: center; padding: 10px; border-radius: 8px;
                        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                        border: 1px solid #333;">
                <div style="font-size: 12px; color: #888;">Sharpe Ratio</div>
                <div style="font-size: 24px; font-weight: bold; color: {sharpe_color};">
                    {sharpe:.2f}
                </div>
                <div style="font-size: 11px; color: #666;">mean/std</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Equity Curve Chart
# ---------------------------------------------------------------------------
def render_equity_curve(equity_data: Dict[str, Any]) -> None:
    """Render the main equity curve chart with drawdown overlay."""
    st.markdown("#### 📈 Equity Curve")
    
    timestamps = equity_data.get("timestamps", [])
    equity = equity_data.get("equity", [])
    drawdown = equity_data.get("drawdown", [])
    
    if not timestamps or not equity:
        st.info("No equity data available")
        return
    
    # Convert timestamps to datetime
    dates = [datetime.fromtimestamp(ts) for ts in timestamps]
    
    if HAS_PLOTLY:
        # Create subplot with secondary y-axis for drawdown
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Equity line
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=equity,
                mode='lines',
                name='Equity',
                line=dict(color=COLOR_LINE, width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 212, 255, 0.1)',
            ),
            secondary_y=False,
        )
        
        # Drawdown area (inverted, on secondary axis)
        dd_pct = [d * 100 for d in drawdown]
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=dd_pct,
                mode='lines',
                name='Drawdown %',
                line=dict(color=COLOR_DRAWDOWN, width=1),
                fill='tozeroy',
                fillcolor='rgba(255, 140, 66, 0.2)',
            ),
            secondary_y=True,
        )
        
        fig.update_layout(
            height=350,
            margin=dict(t=20, b=40, l=60, r=60),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', title="Equity ($)"),
            yaxis2=dict(showgrid=False, title="Drawdown (%)", range=[max(dd_pct) * 1.2 if dd_pct else 10, 0]),
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        # Fallback to streamlit line chart
        import pandas as pd
        df = pd.DataFrame({
            'Date': dates,
            'Equity': equity,
        }).set_index('Date')
        st.line_chart(df)


# ---------------------------------------------------------------------------
# Underwater Curve
# ---------------------------------------------------------------------------
def render_underwater_curve(equity_data: Dict[str, Any]) -> None:
    """Render standalone underwater (drawdown) curve."""
    st.markdown("#### 🌊 Underwater Curve")
    
    timestamps = equity_data.get("timestamps", [])
    drawdown = equity_data.get("drawdown", [])
    
    if not timestamps or not drawdown:
        st.info("No drawdown data available")
        return
    
    dates = [datetime.fromtimestamp(ts) for ts in timestamps]
    dd_pct = [d * 100 for d in drawdown]
    
    if HAS_PLOTLY:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=dd_pct,
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(217, 74, 74, 0.3)',
            line=dict(color=COLOR_LOSS, width=2),
            hovertemplate='%{y:.2f}%<extra></extra>',
        ))
        
        fig.update_layout(
            height=200,
            margin=dict(t=10, b=30, l=60, r=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                title="Drawdown %",
                autorange="reversed",
            ),
            showlegend=False,
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        import pandas as pd
        df = pd.DataFrame({
            'Date': dates,
            'Drawdown %': dd_pct,
        }).set_index('Date')
        st.area_chart(df)


# ---------------------------------------------------------------------------
# Rolling PnL Bar Chart
# ---------------------------------------------------------------------------
def render_rolling_pnl(equity_data: Dict[str, Any]) -> None:
    """Render rolling PnL bar chart."""
    st.markdown("#### 📊 Rolling PnL (20-period)")
    
    timestamps = equity_data.get("timestamps", [])
    rolling = equity_data.get("rolling_pnl", [])
    
    if not timestamps or not rolling:
        st.info("No rolling PnL data available")
        return
    
    dates = [datetime.fromtimestamp(ts) for ts in timestamps]
    
    if HAS_PLOTLY:
        colors = [COLOR_PROFIT if v >= 0 else COLOR_LOSS for v in rolling]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=dates,
            y=rolling,
            marker_color=colors,
            hovertemplate='$%{y:,.2f}<extra></extra>',
        ))
        
        fig.update_layout(
            height=200,
            margin=dict(t=10, b=30, l=60, r=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', title="Rolling PnL ($)"),
            showlegend=False,
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        import pandas as pd
        df = pd.DataFrame({
            'Date': dates,
            'Rolling PnL': rolling,
        }).set_index('Date')
        st.bar_chart(df)


# ---------------------------------------------------------------------------
# Returns Histogram
# ---------------------------------------------------------------------------
def render_returns_histogram(equity_data: Dict[str, Any]) -> None:
    """Render histogram of per-trade PnL."""
    st.markdown("#### 📉 Returns Distribution")
    
    pnl = equity_data.get("pnl", [])
    
    if not pnl:
        st.info("No PnL data available")
        return
    
    if HAS_PLOTLY:
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=pnl,
            nbinsx=30,
            marker_color=COLOR_LINE,
            opacity=0.7,
            hovertemplate='$%{x:.2f}: %{y} trades<extra></extra>',
        ))
        
        # Add vertical line at zero
        fig.add_vline(x=0, line_dash="dash", line_color=COLOR_NEUTRAL)
        
        fig.update_layout(
            height=200,
            margin=dict(t=10, b=30, l=60, r=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', title="PnL ($)"),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', title="Count"),
            showlegend=False,
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        import pandas as pd
        df = pd.DataFrame({'PnL': pnl})
        st.bar_chart(df['PnL'].value_counts().sort_index())


# ---------------------------------------------------------------------------
# Main Panel Renderer
# ---------------------------------------------------------------------------
def render_equity_panel(equity_data: Optional[Dict[str, Any]] = None) -> None:
    """
    Render the complete equity panel.
    
    Args:
        equity_data: Pre-loaded equity data, or None to load from state file
    """
    if equity_data is None:
        equity_data = load_equity_state()
    
    if not equity_data or not equity_data.get("timestamps"):
        st.warning("📊 Equity data unavailable - waiting for trade history")
        st.info(
            "The equity curve will populate as trades are executed. "
            "Run `export_equity_series()` from the executor to generate data."
        )
        return
    
    # Render sections
    render_equity_stats(equity_data)
    
    st.markdown("---")
    
    render_equity_curve(equity_data)
    
    # Two-column layout for underwater and rolling
    col1, col2 = st.columns(2)
    
    with col1:
        render_underwater_curve(equity_data)
    
    with col2:
        render_rolling_pnl(equity_data)
    
    st.markdown("---")
    
    render_returns_histogram(equity_data)
    
    # Metadata footer
    ts = equity_data.get("ts")
    if ts:
        age = datetime.now().timestamp() - ts
        st.caption(f"Data age: {age:.0f}s • Window: {equity_data.get('window_days', 30)} days")


# ---------------------------------------------------------------------------
# Compact Renderer (for embedding)
# ---------------------------------------------------------------------------
def render_equity_compact(equity_data: Optional[Dict[str, Any]] = None) -> None:
    """
    Render a compact equity summary suitable for embedding.
    
    Shows only the equity curve and key stats.
    """
    if equity_data is None:
        equity_data = load_equity_state()
    
    if not equity_data or not equity_data.get("timestamps"):
        st.info("📊 Equity curve: No data")
        return
    
    # Quick stats row
    total_pnl = equity_data.get("total_pnl", 0.0)
    max_dd = equity_data.get("max_drawdown", 0.0)
    win_rate = equity_data.get("win_rate", 0.0)
    
    pnl_color = COLOR_PROFIT if total_pnl >= 0 else COLOR_LOSS
    
    st.markdown(
        f"""
        <div style="display: flex; gap: 20px; margin-bottom: 10px;">
            <span>💰 PnL: <b style="color: {pnl_color}">${total_pnl:+,.2f}</b></span>
            <span>📉 Max DD: <b>{max_dd * 100:.1f}%</b></span>
            <span>🎯 Win: <b>{win_rate * 100:.0f}%</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Mini equity curve
    render_equity_curve(equity_data)
