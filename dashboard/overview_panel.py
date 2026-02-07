"""
Dashboard Overview Panel (v7)

Investor-grade overview page with:
- Risk Mode summary
- Regime heatmap
- Router gauge
- Equity curve
- Portfolio exposure summary
- Latest positions
"""
from __future__ import annotations

import streamlit as st
from typing import Any

from dashboard.risk_panel import render_risk_health_card
from dashboard.regime_panel import render_regime_card
from dashboard.router_gauge import render_router_gauge
from dashboard.equity_panel import render_equity_compact, load_equity_state
from dashboard.pnl_attribution_panel import render_pnl_attribution_panel
from dashboard.diagnostics_panel import render_diagnostics_panel
from dashboard.exit_attribution_panel import render_exit_attribution_panel
from dashboard.hybrid_factor_panel import render_hybrid_factor_panel


# ---------------------------------------------------------------------------
# Color scheme (consistent with v7 design)
# ---------------------------------------------------------------------------
COLOR_OK = "#21c354"
COLOR_WARN = "#f2c037"
COLOR_DEFENSIVE = "#d94a4a"
COLOR_HALTED = "#ff0033"
COLOR_NEUTRAL = "#888888"


def _get_risk_mode_color(mode: str) -> str:
    """Return color for risk mode."""
    mode_upper = (mode or "").upper()
    return {
        "OK": COLOR_OK,
        "WARN": COLOR_WARN,
        "DEFENSIVE": COLOR_DEFENSIVE,
        "HALTED": COLOR_HALTED,
    }.get(mode_upper, COLOR_NEUTRAL)


# ---------------------------------------------------------------------------
# Portfolio Exposure Summary
# ---------------------------------------------------------------------------
def render_portfolio_exposure(state: dict[str, Any]) -> None:
    """Render portfolio exposure summary."""
    st.markdown("### 📊 Portfolio Exposure")
    
    # Extract exposure data
    risk_snap = state.get("risk_snapshot", {})
    nav = risk_snap.get("nav", 0.0)
    exposure = risk_snap.get("exposure", 0.0)
    margin_used = risk_snap.get("margin_used", 0.0)
    
    # Calculate metrics
    exposure_pct = (exposure / nav * 100) if nav > 0 else 0.0
    margin_pct = (margin_used / nav * 100) if nav > 0 else 0.0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Total Exposure",
            value=f"${exposure:,.0f}",
            delta=f"{exposure_pct:.1f}% of NAV",
        )
    
    with col2:
        st.metric(
            label="Margin Used",
            value=f"${margin_used:,.0f}",
            delta=f"{margin_pct:.1f}% of NAV",
        )
    
    with col3:
        available = nav - margin_used
        st.metric(
            label="Available Capital",
            value=f"${available:,.0f}",
            delta=f"{100 - margin_pct:.1f}% free",
        )


# ---------------------------------------------------------------------------
# Open Positions Table
# ---------------------------------------------------------------------------
def render_open_positions(state: dict[str, Any]) -> None:
    """Render table of open positions."""
    st.markdown("### 📈 Open Positions")
    
    positions = state.get("positions", [])
    
    if not positions:
        st.info("No open positions")
        return
    
    # Build table data
    rows = []
    for pos in positions:
        symbol = pos.get("symbol", "???")
        side = pos.get("side", "???")
        size = pos.get("size", 0.0)
        entry = pos.get("entry_price", 0.0)
        mark = pos.get("mark_price", 0.0)
        pnl = pos.get("unrealized_pnl", 0.0)
        pnl_pct = pos.get("pnl_pct", 0.0)
        
        # Color code PnL
        pnl_color = COLOR_OK if pnl >= 0 else COLOR_DEFENSIVE
        
        rows.append({
            "Symbol": symbol,
            "Side": side.upper() if side else "???",
            "Size": f"{abs(size):.4f}",
            "Entry": f"${entry:,.2f}",
            "Mark": f"${mark:,.2f}",
            "PnL": f"${pnl:+,.2f}",
            "PnL %": f"{pnl_pct:+.2f}%",
        })
    
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No position data available")


# ---------------------------------------------------------------------------
# Equity Curve Section
# ---------------------------------------------------------------------------
def render_equity_curve_section(state: dict[str, Any]) -> None:
    """Render equity curve section using equity_panel or nav_state series fallback."""
    st.markdown("### 📉 Equity Curve")
    
    # Try to get equity data from state, or load from file
    equity_data = state.get("equity")
    if not equity_data:
        equity_data = load_equity_state()
    
    if equity_data and equity_data.get("timestamps"):
        render_equity_compact(equity_data)
        return

    # Fallback: render from nav_state.json series (always available when
    # executor is running).  This avoids the "no data" message when
    # equity.json doesn't exist but live NAV series does.
    nav_state = state.get("nav", state.get("nav_state", {}))
    series = nav_state.get("series", [])
    if not series:
        st.info(
            "📊 Equity curve will populate as trades are executed. "
            "Run the executor to generate trade history."
        )
        return

    try:
        import plotly.graph_objects as go  # type: ignore[import-untyped]

        timestamps = []
        equities = []
        for pt in series:
            t = pt.get("t")
            eq = pt.get("equity") or pt.get("nav") or pt.get("total_equity")
            if t is not None and eq is not None:
                timestamps.append(str(t))
                equities.append(float(eq))

        if len(equities) < 2:
            st.info("📊 Not enough data points for equity curve yet.")
            return

        pnl = equities[-1] - equities[0]
        pnl_color = "#21c354" if pnl >= 0 else "#d94a4a"
        st.markdown(
            f'<span style="font-size:0.9em;">Session PnL: '
            f'<b style="color:{pnl_color}">${pnl:+,.2f}</b> '
            f'({len(equities)} samples)</span>',
            unsafe_allow_html=True,
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps, y=equities,
            mode="lines", name="Equity",
            line=dict(color="#00d4ff", width=2),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.08)",
        ))
        fig.update_layout(
            height=250, margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(showgrid=False),
            yaxis=dict(title="USDT", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ccc"),
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.info("📊 Install plotly for equity curve visualization.")
    except Exception:
        st.info("📊 Equity curve will populate as trades are executed.")


# ---------------------------------------------------------------------------
# Main Overview Panel
# ---------------------------------------------------------------------------
def render_overview_panel(state: dict[str, Any]) -> None:
    """
    Render the complete Overview panel.
    
    Sections:
    1. Risk Mode + Regime Heatmap (side by side)
    2. Router Gauge
    3. Portfolio Exposure Summary
    4. Open Positions Table
    5. Equity Curve
    6. Attribution
    7. Diagnostics
    """
    st.header("🏦 Portfolio Overview")
    st.caption("Investor-grade system health dashboard")
    
    # -------------------------------------------------------------------------
    # Row 1: Risk Mode + Regime Heatmap
    # -------------------------------------------------------------------------
    col_risk, col_regime = st.columns(2)
    
    with col_risk:
        render_risk_health_card(state)
    
    with col_regime:
        regimes = state.get("regimes", {})
        render_regime_card(regimes)
    
    st.divider()
    
    # -------------------------------------------------------------------------
    # Row 2: Router Gauge
    # -------------------------------------------------------------------------
    router_state = state.get("router_health", {})
    render_router_gauge(router_state)
    
    st.divider()
    
    # -------------------------------------------------------------------------
    # Row 3: Portfolio Exposure
    # -------------------------------------------------------------------------
    render_portfolio_exposure(state)
    
    st.divider()
    
    # -------------------------------------------------------------------------
    # Row 4: Open Positions
    # -------------------------------------------------------------------------
    render_open_positions(state)
    
    st.divider()
    
    # -------------------------------------------------------------------------
    # Row 5: Equity Curve
    # -------------------------------------------------------------------------
    render_equity_curve_section(state)
    
    st.divider()
    render_pnl_attribution_panel(state.get("pnl_attribution", {}))

    st.divider()
    render_exit_attribution_panel(state.get("pnl_attribution", {}))

    st.divider()
    render_hybrid_factor_panel(state.get("pnl_attribution", {}))
    
    st.divider()
    render_diagnostics_panel(
        equity=state.get("equity", {}),
        positions=state.get("positions", []),
        pnl_attribution=state.get("pnl_attribution", {}),
    )


# ---------------------------------------------------------------------------
# Compact Overview (for embedding in other pages)
# ---------------------------------------------------------------------------
def render_overview_compact(state: dict[str, Any]) -> None:
    """
    Render a compact overview suitable for embedding.
    
    Shows only Risk Mode and Regime summary in a single row.
    """
    col1, col2 = st.columns(2)
    
    with col1:
        render_risk_health_card(state)
    
    with col2:
        regimes = state.get("regimes", {})
        render_regime_card(regimes)
