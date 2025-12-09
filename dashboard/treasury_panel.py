"""Treasury panel for v7.6 dashboard."""
from __future__ import annotations

import streamlit as st
from typing import Any, Dict, Optional

from dashboard.state_v7 import (
    load_offchain_assets,
    load_offchain_yield,
    compute_unified_aum,
    compute_treasury_yield_summary,
    get_treasury_health,
    load_nav_state,
)


def _fmt_usd(val: Any, decimals: int = 2) -> str:
    """Format value as USD string."""
    try:
        num = float(val) if val is not None else 0.0
        return f"${num:,.{decimals}f}"
    except Exception:
        return "$0.00"


def _fmt_pct(val: Any, decimals: int = 2) -> str:
    """Format value as percentage string."""
    try:
        num = float(val) if val is not None else 0.0
        return f"{num:.{decimals}f}%"
    except Exception:
        return "0.00%"


def _age_badge(age_s: Optional[float], stale_threshold: float = 300) -> str:
    """Return HTML badge for age indicator."""
    if age_s is None:
        return '<span style="color:#888;">n/a</span>'
    if age_s < stale_threshold:
        color = "#21ba45"  # green
    elif age_s < stale_threshold * 2:
        color = "#f2711c"  # orange
    else:
        color = "#db2828"  # red
    
    if age_s < 60:
        label = f"{age_s:.0f}s"
    elif age_s < 3600:
        label = f"{age_s/60:.1f}m"
    else:
        label = f"{age_s/3600:.1f}h"
    
    return f'<span style="color:{color};font-weight:600;">{label}</span>'


def render_aum_overview(
    nav_state: Optional[Dict[str, Any]] = None,
    offchain_assets: Optional[Dict[str, Any]] = None,
) -> None:
    """Render AUM overview panel with slices."""
    aum = compute_unified_aum(nav_state, offchain_assets)
    
    st.markdown("### 💰 Assets Under Management")
    
    # Main AUM metric
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total AUM",
            _fmt_usd(aum["total_aum_usd"]),
            help="Total Assets Under Management = Futures NAV + External Accounts",
        )
    
    with col2:
        st.metric(
            "Futures NAV",
            _fmt_usd(aum["futures_nav_usd"]),
            help="Trading capital on exchange",
        )
    
    with col3:
        st.metric(
            "External Accounts",
            _fmt_usd(aum["offchain_usd"]),
            help="Holdings on exchanges without API access",
        )
    
    with col4:
        st.metric(
            "Spot Treasury",
            _fmt_usd(aum["spot_treasury_usd"]),
            help="Spot holdings tracked by executor",
        )
    
    # Stale indicators
    stale_flags = aum.get("stale_flags", {})
    if stale_flags.get("nav_stale") or stale_flags.get("offchain_stale"):
        warnings = []
        if stale_flags.get("nav_stale"):
            warnings.append("NAV data is stale")
        if stale_flags.get("offchain_stale"):
            warnings.append("External accounts data is stale")
        st.warning(" | ".join(warnings))


def render_aum_slices(
    nav_state: Optional[Dict[str, Any]] = None,
    offchain_assets: Optional[Dict[str, Any]] = None,
) -> None:
    """Render AUM breakdown as a bar chart."""
    aum = compute_unified_aum(nav_state, offchain_assets)
    slices = aum.get("slices", {})
    
    if not slices:
        return
    
    # Build data for display
    import pandas as pd
    
    data = []
    for slice_key, slice_data in slices.items():
        if slice_data.get("value_usd", 0) > 0:
            data.append({
                "Category": slice_data.get("label", slice_key),
                "Value (USD)": slice_data.get("value_usd", 0),
                "Percentage": slice_data.get("pct", 0),
            })
    
    if data:
        df = pd.DataFrame(data)
        st.dataframe(
            df.style.format({
                "Value (USD)": "${:,.2f}",
                "Percentage": "{:.1f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )


def render_treasury_yield(
    offchain_assets: Optional[Dict[str, Any]] = None,
    offchain_yield: Optional[Dict[str, Any]] = None,
) -> None:
    """Render treasury yield summary panel."""
    yield_summary = compute_treasury_yield_summary(offchain_assets, offchain_yield)
    
    st.markdown("### 📈 Treasury Yield")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Weighted Avg APR",
            _fmt_pct(yield_summary["weighted_avg_apr_pct"]),
            help="Value-weighted average APR across all treasury assets",
        )
    
    with col2:
        st.metric(
            "Daily Yield",
            _fmt_usd(yield_summary["daily_yield_usd"]),
            help="Estimated daily yield from treasury",
        )
    
    with col3:
        st.metric(
            "Monthly Yield",
            _fmt_usd(yield_summary["monthly_yield_usd"]),
            help="Estimated monthly yield from treasury",
        )
    
    with col4:
        st.metric(
            "Annual Yield",
            _fmt_usd(yield_summary["annual_yield_usd"]),
            help="Estimated annual yield from treasury",
        )
    
    # Per-asset breakdown
    per_asset = yield_summary.get("per_asset", {})
    if per_asset:
        st.markdown("#### Per-Asset Yield")
        import pandas as pd
        
        data = []
        for asset_name, asset_data in per_asset.items():
            data.append({
                "Asset": asset_name,
                "Value (USD)": asset_data.get("usd_value", 0),
                "APR": asset_data.get("apr_pct", 0),
                "Daily Yield": asset_data.get("daily_yield_usd", 0),
                "Strategy": asset_data.get("strategy", "unknown"),
            })
        
        if data:
            df = pd.DataFrame(data)
            st.dataframe(
                df.style.format({
                    "Value (USD)": "${:,.2f}",
                    "APR": "{:.2f}%",
                    "Daily Yield": "${:,.4f}",
                }),
                use_container_width=True,
                hide_index=True,
            )


def _load_coingecko_cache() -> Dict[str, float]:
    """Load live prices from coingecko cache."""
    import json
    import os
    cache_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "logs", "cache", "coingecko_cache.json"
    )
    try:
        with open(cache_path, "r") as f:
            cache = json.load(f)
            return cache.get("prices", {})
    except Exception:
        return {}


def render_treasury_assets(
    offchain_assets: Optional[Dict[str, Any]] = None,
) -> None:
    """Render treasury assets detail table with live prices."""
    if offchain_assets is None:
        offchain_assets = load_offchain_assets({})
    
    assets = offchain_assets.get("assets", {})
    if not assets:
        st.info("No external accounts configured")
        return
    
    st.markdown("### 🏦 External Accounts")
    
    # Load live prices from coingecko cache
    live_prices = _load_coingecko_cache()
    
    import pandas as pd
    
    data = []
    for asset_name, asset_data in assets.items():
        if not isinstance(asset_data, dict):
            continue
        qty = float(asset_data.get("qty", 0))
        # Use live price if available, else fallback to static
        live_price = live_prices.get(asset_name.upper())
        if live_price and live_price > 0:
            current_price = live_price
            usd_value = qty * live_price
        else:
            current_price = asset_data.get("current_price_usd") or 0
            usd_value = asset_data.get("usd_value") or 0
        data.append({
            "Asset": asset_name,
            "Quantity": qty,
            "Avg Cost (USD)": asset_data.get("avg_cost_usd", 0),
            "Current Price": current_price,
            "Value (USD)": usd_value,
            "Source": asset_data.get("source", "unknown"),
        })
    
    if data:
        df = pd.DataFrame(data)
        st.dataframe(
            df.style.format({
                "Quantity": "{:.6f}",
                "Avg Cost (USD)": "${:,.2f}",
                "Current Price": "${:,.2f}",
                "Value (USD)": "${:,.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
    
    # Metadata
    metadata = offchain_assets.get("metadata", {})
    if metadata:
        with st.expander("Import Metadata"):
            st.json(metadata)


def render_treasury_health(
    offchain_assets: Optional[Dict[str, Any]] = None,
    offchain_yield: Optional[Dict[str, Any]] = None,
) -> None:
    """Render treasury state health panel."""
    health = get_treasury_health(offchain_assets, offchain_yield)
    
    st.markdown("### 🔍 Treasury State Health")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status = "✅ Healthy" if health["healthy"] else "⚠️ Issues Detected"
        st.markdown(f"**Status:** {status}")
    
    with col2:
        assets_age_html = _age_badge(health["assets_surface_age_s"], stale_threshold=86400)
        st.markdown(f"**Assets Age:** {assets_age_html}", unsafe_allow_html=True)
    
    with col3:
        yield_age_html = _age_badge(health["yield_surface_age_s"], stale_threshold=86400)
        st.markdown(f"**Yield Age:** {yield_age_html}", unsafe_allow_html=True)
    
    # Warnings and missing fields
    warnings = health.get("warnings", [])
    missing = health.get("missing_fields", [])
    
    if warnings:
        st.warning("**Warnings:** " + ", ".join(warnings))
    
    if missing:
        st.error("**Missing Fields:** " + ", ".join(missing))


def render_treasury_panel() -> None:
    """Render complete treasury panel with all sections."""
    # Load data once
    nav_state = load_nav_state({})
    offchain_assets = load_offchain_assets({})
    offchain_yield = load_offchain_yield({})
    
    # AUM Overview
    render_aum_overview(nav_state, offchain_assets)
    st.divider()
    
    # Yield Summary
    render_treasury_yield(offchain_assets, offchain_yield)
    st.divider()
    
    # Assets Detail
    render_treasury_assets(offchain_assets)
    st.divider()
    
    # Health Diagnostics
    render_treasury_health(offchain_assets, offchain_yield)
