"""
v7.5_C2/C3 — Factor Diagnostics Dashboard Panel.

Provides visual analysis of:
- Per-symbol factor fingerprints
- Factor correlation matrix heatmap
- Factor PnL attribution table
- Factor weights visualization (v7.5_C3)
- Orthogonalization status indicator (v7.5_C3)

This panel is analysis-only for understanding factor behavior.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

from dashboard.state_v7 import (
    load_factor_diagnostics_state,
    load_factor_pnl_state,
    get_factor_correlation_matrix,
    get_factor_volatilities,
    get_factor_pnl_summary,
    get_factor_weights,
    get_factor_ir,
    get_orthogonalization_status,
)


def render_factor_fingerprint_table(
    factor_diagnostics_state: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Render per-symbol factor fingerprint table.
    
    Shows normalized factor values for each symbol in a table format.
    """
    if factor_diagnostics_state is None:
        factor_diagnostics_state = load_factor_diagnostics_state()
    
    per_symbol = factor_diagnostics_state.get("per_symbol", {})
    config = factor_diagnostics_state.get("config", {})
    
    if not per_symbol:
        st.info("No factor diagnostics data available yet.")
        return
    
    # Get factor names from config or first entry
    factor_names = config.get("factors", [])
    if not factor_names and per_symbol:
        first_entry = next(iter(per_symbol.values()), {})
        factor_names = list(first_entry.get("factors", {}).keys())
    
    # Build table data
    rows = []
    for key, data in sorted(per_symbol.items()):
        if not isinstance(data, dict):
            continue
        
        factors = data.get("factors", {})
        direction = data.get("direction", "LONG")
        
        # Parse symbol from key (format: SYMBOL:DIRECTION)
        symbol = key.split(":")[0] if ":" in key else key
        
        row = {
            "Symbol": symbol,
            "Direction": direction,
        }
        
        for factor in factor_names:
            val = factors.get(factor, 0.0)
            row[factor.replace("_", " ").title()] = round(val, 3)
        
        rows.append(row)
    
    if not rows:
        st.info("No factor fingerprints to display.")
        return
    
    st.subheader("📊 Per-Symbol Factor Fingerprints")
    st.caption("Normalized factor values (z-scored) for each symbol. Values near 0 = average, positive = above average, negative = below average.")
    
    # Display as dataframe
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        
        # Apply color styling to factor columns
        factor_cols = [col for col in df.columns if col not in ("Symbol", "Direction")]
        
        def color_factor(val):
            if not isinstance(val, (int, float)):
                return ""
            if val > 1.0:
                return "background-color: #2ecc71; color: black"
            elif val > 0.5:
                return "background-color: #82e0aa; color: black"
            elif val < -1.0:
                return "background-color: #e74c3c; color: white"
            elif val < -0.5:
                return "background-color: #f1948a; color: black"
            return ""
        
        styled = df.style.applymap(color_factor, subset=factor_cols)
        st.dataframe(styled, use_container_width=True)
    except ImportError:
        # Fallback without pandas styling
        st.table(rows)


def render_correlation_heatmap(
    factor_diagnostics_state: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Render factor correlation matrix as a heatmap.
    
    Interpretation:
    - Values near 1.0 = highly correlated (redundant factors)
    - Values near 0.0 = independent factors
    - Negative values = hedging/opposing factors
    """
    if factor_diagnostics_state is None:
        factor_diagnostics_state = load_factor_diagnostics_state()
    
    factors, corr_matrix = get_factor_correlation_matrix(factor_diagnostics_state)
    factor_vols = get_factor_volatilities(factor_diagnostics_state)
    
    if not factors or not corr_matrix:
        st.info("No factor correlation data available yet.")
        return
    
    st.subheader("🔗 Factor Correlation Matrix")
    st.caption("Correlation between factors. |ρ| > 0.7 = redundant, |ρ| < 0.3 = independent, ρ < 0 = hedging.")
    
    try:
        import pandas as pd
        
        # Create correlation dataframe
        corr_df = pd.DataFrame(
            corr_matrix,
            index=factors,
            columns=factors,
        )
        
        # Display with color gradient
        def color_corr(val):
            if not isinstance(val, (int, float)):
                return ""
            if val > 0.7:
                return "background-color: #27ae60; color: white"
            elif val > 0.3:
                return "background-color: #82e0aa; color: black"
            elif val < -0.7:
                return "background-color: #e74c3c; color: white"
            elif val < -0.3:
                return "background-color: #f1948a; color: black"
            return "background-color: #f8f9fa"
        
        styled = corr_df.style.applymap(color_corr).format("{:.3f}")
        st.dataframe(styled, use_container_width=True)
        
        # Display factor volatilities
        if factor_vols:
            st.subheader("📈 Factor Volatilities")
            vol_df = pd.DataFrame([
                {"Factor": f, "Volatility": round(v, 4)}
                for f, v in factor_vols.items()
            ])
            st.dataframe(vol_df, use_container_width=True, hide_index=True)
            
    except ImportError:
        # Fallback display
        st.write("Factors:", factors)
        st.write("Correlation Matrix:", corr_matrix)
        st.write("Factor Volatilities:", factor_vols)


def render_pnl_attribution_table(
    factor_pnl_state: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Render factor PnL attribution table.
    
    Shows how much PnL is attributed to each factor.
    """
    if factor_pnl_state is None:
        factor_pnl_state = load_factor_pnl_state()
    
    summary = get_factor_pnl_summary(factor_pnl_state)
    by_factor = summary.get("by_factor", {})
    pct_by_factor = summary.get("pct_by_factor", {})
    total_pnl = summary.get("total_pnl_usd", 0.0)
    trade_count = summary.get("trade_count", 0)
    window_days = summary.get("window_days", 14)
    
    if not by_factor:
        st.info("No factor PnL attribution data available yet.")
        return
    
    st.subheader("💰 Factor PnL Attribution")
    st.caption(f"PnL attributed to each factor over the last {window_days} days ({trade_count} trades)")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total PnL", f"${total_pnl:,.2f}")
    with col2:
        st.metric("Trades", trade_count)
    with col3:
        st.metric("Window", f"{window_days} days")
    
    # Build table
    rows = []
    for factor in sorted(by_factor.keys()):
        pnl = by_factor.get(factor, 0.0)
        pct = pct_by_factor.get(factor, 0.0)
        
        rows.append({
            "Factor": factor.replace("_", " ").title(),
            "PnL (USD)": round(pnl, 2),
            "% of Total": round(pct, 1),
        })
    
    try:
        import pandas as pd
        
        df = pd.DataFrame(rows)
        
        # Sort by absolute PnL contribution
        df = df.sort_values("PnL (USD)", key=abs, ascending=False)
        
        def color_pnl(val):
            if not isinstance(val, (int, float)):
                return ""
            if val > 0:
                return "color: #27ae60; font-weight: bold"
            elif val < 0:
                return "color: #e74c3c; font-weight: bold"
            return ""
        
        styled = df.style.applymap(color_pnl, subset=["PnL (USD)"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
        
    except ImportError:
        st.table(rows)


def render_factor_weights_table(
    factor_diagnostics_state: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Render factor weights table with Vol, PnL, IR, and Weight columns (v7.5_C3).
    
    Shows auto-computed factor weights based on IR/vol weighting.
    """
    if factor_diagnostics_state is None:
        factor_diagnostics_state = load_factor_diagnostics_state()
    
    # Get status indicators
    ortho_enabled, auto_weight_enabled = get_orthogonalization_status(factor_diagnostics_state)
    
    # Show status indicators
    col1, col2 = st.columns(2)
    with col1:
        if ortho_enabled:
            st.success("✅ Orthogonalization: ENABLED")
        else:
            st.info("ℹ️ Orthogonalization: DISABLED")
    with col2:
        if auto_weight_enabled:
            st.success("✅ Auto-Weighting: ENABLED")
        else:
            st.info("ℹ️ Auto-Weighting: DISABLED")
    
    if not auto_weight_enabled:
        st.info("Auto-weighting is disabled. Enable it in strategy_config.json to see factor weights.")
        return
    
    # Get weights, vols, and IR
    weights = get_factor_weights(factor_diagnostics_state)
    factor_vols = get_factor_volatilities(factor_diagnostics_state)
    factor_ir = get_factor_ir(factor_diagnostics_state)
    
    # Get PnL data
    factor_pnl = load_factor_pnl_state()
    pnl_summary = get_factor_pnl_summary(factor_pnl)
    pnl_by_factor = pnl_summary.get("by_factor", {})
    
    if not weights:
        st.info("No factor weights computed yet.")
        return
    
    st.subheader("⚖️ Factor Weights (Auto-Computed)")
    st.caption("Weights based on vol_inverse_ir mode: higher IR and lower vol → higher weight")
    
    # Build table
    rows = []
    for factor in sorted(weights.keys()):
        w = weights.get(factor, 0.0)
        vol = factor_vols.get(factor, 0.0)
        ir = factor_ir.get(factor, 0.0)
        pnl = pnl_by_factor.get(factor, 0.0)
        
        rows.append({
            "Factor": factor,
            "Weight": f"{w:.1%}",
            "Vol": f"{vol:.4f}",
            "IR": f"{ir:.3f}",
            "PnL": f"${pnl:,.2f}",
        })
    
    if not rows:
        st.info("No factor weight data available.")
        return
    
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Weight bar chart
        st.subheader("📊 Weight Distribution")
        weight_data = {r["Factor"]: weights.get(r["Factor"], 0.0) for r in rows}
        st.bar_chart(weight_data)
        
        # Sum check
        weight_sum = sum(weights.values())
        st.caption(f"Total weight sum: {weight_sum:.1%} (should be ~100%)")
        
    except ImportError:
        for row in rows:
            st.write(f"**{row['Factor']}**: Weight={row['Weight']}, Vol={row['Vol']}, IR={row['IR']}, PnL={row['PnL']}")


def render_factor_panel() -> None:
    """
    Render the complete factor diagnostics panel.
    
    Includes:
    - Per-symbol factor fingerprints
    - Factor correlation matrix
    - Factor PnL attribution
    - Factor weights (v7.5_C3)
    """
    st.header("🧬 Factor Diagnostics (v7.5_C3)")
    st.caption("Multi-factor model analysis: factor fingerprints, correlations, PnL attribution, and auto-weighting.")
    
    # Load state once
    factor_diag = load_factor_diagnostics_state()
    factor_pnl = load_factor_pnl_state()
    
    # Check if data is available
    if not factor_diag and not factor_pnl:
        st.warning(
            "Factor diagnostics data not available. "
            "Ensure executor is running with factor_diagnostics enabled in strategy_config.json."
        )
        return
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Fingerprints", "🔗 Correlations", "💰 PnL Attribution", "⚖️ Weights"])
    
    with tab1:
        render_factor_fingerprint_table(factor_diag)
    
    with tab2:
        render_correlation_heatmap(factor_diag)
    
    with tab3:
        render_pnl_attribution_table(factor_pnl)
    
    with tab4:
        render_factor_weights_table(factor_diag)


# Entry point for importing into main dashboard
__all__ = [
    "render_factor_panel",
    "render_factor_fingerprint_table",
    "render_correlation_heatmap",
    "render_pnl_attribution_table",
    "render_factor_weights_table",
]
