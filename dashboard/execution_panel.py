"""
v7.5_B1/B2 — Execution Quality Panel

Dashboard panel for displaying:
- Per-symbol slippage metrics (EWMA expected vs realized)
- Liquidity bucket assignments
- Router quality scores (v7.5_B2)
- Execution quality summary
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# State directory
STATE_DIR = Path(os.getenv("HEDGE_STATE_DIR") or "logs/state")


def _load_state_file(filename: str) -> Dict[str, Any]:
    """Load a state JSON file."""
    path = STATE_DIR / filename
    if not path.exists():
        return {}
    try:
        with path.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def load_slippage_state() -> Dict[str, Any]:
    """Load slippage.json state file."""
    return _load_state_file("slippage.json")


def load_liquidity_buckets_state() -> Dict[str, Any]:
    """Load liquidity_buckets.json state file (dashboard copy)."""
    return _load_state_file("liquidity_buckets.json")


def load_router_quality_state() -> Dict[str, Any]:
    """Load router_quality.json state file (v7.5_B2)."""
    return _load_state_file("router_quality.json")


def render_slippage_table(slippage_state: Dict[str, Any]) -> None:
    """
    Render a table of per-symbol slippage metrics.
    
    Columns:
    - Symbol
    - EWMA Expected (bps)
    - EWMA Realized (bps)
    - Drift (realized - expected)
    - Trades (count)
    """
    per_symbol = slippage_state.get("per_symbol", {})
    
    if not per_symbol:
        st.info("No slippage data available yet. Metrics will appear after trades execute.")
        return
    
    # Build table data
    rows = []
    for symbol, stats in sorted(per_symbol.items()):
        expected = stats.get("ewma_expected_bps", 0.0)
        realized = stats.get("ewma_realized_bps", 0.0)
        drift = realized - expected
        trade_count = stats.get("trade_count", 0)
        
        rows.append({
            "Symbol": symbol,
            "Expected (bps)": round(expected, 2),
            "Realized (bps)": round(realized, 2),
            "Drift (bps)": round(drift, 2),
            "Trades": trade_count,
        })
    
    # Create DataFrame
    import pandas as pd
    df = pd.DataFrame(rows)
    
    # Style the drift column
    def color_drift(val):
        if val > 3.0:
            return "color: red"
        elif val < -1.0:
            return "color: green"
        return ""
    
    styled = df.style.applymap(color_drift, subset=["Drift (bps)"])
    st.dataframe(styled, use_container_width=True, hide_index=True)


def render_liquidity_buckets_table(buckets_state: Dict[str, Any]) -> None:
    """
    Render a table of liquidity bucket assignments.
    
    Shows symbol -> bucket mapping with bucket properties.
    """
    symbols = buckets_state.get("symbols", {})
    buckets = buckets_state.get("buckets", {})
    
    if not symbols:
        st.info("No liquidity bucket data available.")
        return
    
    # Build bucket summary
    st.markdown("#### Bucket Summary")
    bucket_cols = st.columns(len(buckets) if buckets else 1)
    for i, (bucket_name, bucket_info) in enumerate(sorted(buckets.items())):
        with bucket_cols[i % len(bucket_cols)]:
            st.metric(
                label=bucket_name,
                value=f"{bucket_info.get('symbol_count', 0)} symbols",
                delta=f"max spread: {bucket_info.get('max_spread_bps', 0):.0f} bps",
            )
    
    st.markdown("#### Symbol Assignments")
    
    # Build table data
    rows = []
    for symbol, info in sorted(symbols.items()):
        bucket = info.get("bucket", "GENERIC")
        max_spread = info.get("max_spread_bps", 15.0)
        maker_bias = info.get("default_maker_bias", 0.5)
        
        # Badge color based on bucket
        if bucket == "A_HIGH":
            badge = "🟢"
        elif bucket == "B_MEDIUM":
            badge = "🟡"
        elif bucket == "C_LOW":
            badge = "🟠"
        else:
            badge = "⚪"
        
        rows.append({
            "Symbol": symbol,
            "Bucket": f"{badge} {bucket}",
            "Max Spread (bps)": max_spread,
            "Maker Bias": f"{maker_bias:.0%}",
        })
    
    import pandas as pd
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_execution_summary(
    slippage_state: Dict[str, Any],
    buckets_state: Dict[str, Any],
    router_quality_state: Dict[str, Any] | None = None,
) -> None:
    """Render an execution quality summary."""
    per_symbol = slippage_state.get("per_symbol", {})
    
    if not per_symbol:
        return
    
    # Calculate summary stats
    total_trades = sum(s.get("trade_count", 0) for s in per_symbol.values())
    avg_realized = sum(s.get("ewma_realized_bps", 0) for s in per_symbol.values()) / len(per_symbol) if per_symbol else 0
    avg_expected = sum(s.get("ewma_expected_bps", 0) for s in per_symbol.values()) / len(per_symbol) if per_symbol else 0
    avg_drift = avg_realized - avg_expected
    
    # Router quality summary
    rq_summary = {}
    if router_quality_state:
        rq_summary = router_quality_state.get("summary", {})
    
    # Display summary metrics
    cols = st.columns(5)
    
    with cols[0]:
        st.metric("Total Trades", total_trades)
    
    with cols[1]:
        st.metric("Avg Expected Slippage", f"{avg_expected:.1f} bps")
    
    with cols[2]:
        st.metric("Avg Realized Slippage", f"{avg_realized:.1f} bps")
    
    with cols[3]:
        drift_delta = "worse" if avg_drift > 0 else "better"
        st.metric("Avg Drift", f"{avg_drift:.1f} bps", delta=drift_delta)
    
    with cols[4]:
        if rq_summary:
            avg_rq = rq_summary.get("avg_score", 0.8)
            low_q = rq_summary.get("low_quality_count", 0)
            st.metric(
                "Avg Router Quality",
                f"{avg_rq:.2f}",
                delta=f"{low_q} low" if low_q > 0 else None,
                delta_color="inverse" if low_q > 0 else "off",
            )
        else:
            st.metric("Avg Router Quality", "N/A")


def render_execution_panel() -> None:
    """
    Main entry point for the Execution Quality panel.
    
    Displays:
    1. Execution summary metrics
    2. Slippage table (per symbol)
    3. Router quality view (v7.5_B2)
    4. Liquidity buckets view
    """
    st.header("⚡ Execution Quality")
    
    # Load state files
    slippage_state = load_slippage_state()
    buckets_state = load_liquidity_buckets_state()
    router_quality_state = load_router_quality_state()
    
    # Summary section
    st.markdown("### Summary")
    render_execution_summary(slippage_state, buckets_state, router_quality_state)
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Slippage Metrics", "🎯 Router Quality", "🪣 Liquidity Buckets"])
    
    with tab1:
        st.markdown("### Per-Symbol Slippage (EWMA)")
        st.markdown("""
        Tracks expected vs realized slippage:
        - **Expected**: Estimated from order book depth before execution
        - **Realized**: Actual fill price vs mid price
        - **Drift**: Realized - Expected (positive = worse than expected)
        """)
        render_slippage_table(slippage_state)
    
    with tab2:
        st.markdown("### Router Quality Score")
        st.markdown("""
        Per-symbol execution quality scores (v7.5_B2):
        - **Score**: Overall router quality [0-1] (higher = better)
        - **Bucket**: Liquidity classification (A_HIGH, B_MEDIUM, C_LOW)
        - **Drift**: Realized - Expected slippage (negative = better than expected)
        - **TWAP Skip**: Ratio of skipped TWAP slices
        """)
        render_router_quality_table(router_quality_state)
    
    with tab3:
        st.markdown("### Liquidity Bucket Configuration")
        st.markdown("""
        Symbols are classified into liquidity buckets:
        - **A_HIGH**: Tight spreads (BTC, ETH) - aggressive maker usage
        - **B_MEDIUM**: Normal spreads - balanced routing
        - **C_LOW**: Wide spreads - more taker, smaller sizes
        """)
        render_liquidity_buckets_table(buckets_state)


def render_router_quality_table(router_quality_state: Dict[str, Any]) -> None:
    """
    Render a table of per-symbol router quality scores (v7.5_B2).
    
    Columns:
    - Symbol
    - Router Score (0-1)
    - Bucket (A/B/C)
    - EWMA Exp. (bps)
    - EWMA Real. (bps)
    - Drift (bps)
    - TWAP Skip Ratio
    """
    symbols = router_quality_state.get("symbols", {})
    
    if not symbols:
        st.info("No router quality data available yet. Metrics will appear after trades execute.")
        return
    
    # Build table data
    rows = []
    for symbol, stats in sorted(symbols.items()):
        score = stats.get("score", 0.8)
        bucket = stats.get("bucket", "GENERIC")
        expected = stats.get("ewma_expected_bps", 0.0)
        realized = stats.get("ewma_realized_bps", 0.0)
        drift = stats.get("slippage_drift_bps", realized - expected)
        twap_skip = stats.get("twap_skip_ratio", 0.0)
        trade_count = stats.get("trade_count", 0)
        
        # Quality badge
        if score >= 0.9:
            badge = "🟢"
        elif score >= 0.7:
            badge = "🟡"
        else:
            badge = "🔴"
        
        rows.append({
            "Symbol": symbol,
            "Score": f"{badge} {score:.2f}",
            "Bucket": bucket,
            "Exp. (bps)": round(expected, 2),
            "Real. (bps)": round(realized, 2),
            "Drift (bps)": round(drift, 2),
            "TWAP Skip": f"{twap_skip:.1%}",
            "Trades": trade_count,
        })
    
    # Create DataFrame
    import pandas as pd
    df = pd.DataFrame(rows)
    
    # Style the drift column
    def color_drift(val):
        try:
            v = float(val)
            if v > 3.0:
                return "color: red"
            elif v < -1.0:
                return "color: green"
        except (ValueError, TypeError):
            pass
        return ""
    
    styled = df.style.applymap(color_drift, subset=["Drift (bps)"])
    st.dataframe(styled, use_container_width=True, hide_index=True)


def render_slippage_mini_card() -> None:
    """
    Render a compact slippage summary card for use in overview panels.
    """
    slippage_state = load_slippage_state()
    per_symbol = slippage_state.get("per_symbol", {})
    
    if not per_symbol:
        st.caption("No slippage data")
        return
    
    total_trades = sum(s.get("trade_count", 0) for s in per_symbol.values())
    avg_realized = sum(s.get("ewma_realized_bps", 0) for s in per_symbol.values()) / len(per_symbol)
    
    st.metric(
        label="Avg Slippage",
        value=f"{avg_realized:.1f} bps",
        delta=f"{total_trades} trades",
    )


__all__ = [
    "render_execution_panel",
    "render_slippage_table",
    "render_liquidity_buckets_table",
    "render_router_quality_table",
    "render_slippage_mini_card",
    "load_slippage_state",
    "load_liquidity_buckets_state",
    "load_router_quality_state",
]
