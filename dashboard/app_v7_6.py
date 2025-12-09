# mypy: ignore-errors
"""
Hedge Dashboard v7.6 — Streamlit Application

This is the main entry point for the dashboard. It serves as a minimal
orchestrator that:
    1. Configures Streamlit page settings
    2. Loads state from canonical surfaces
    3. Delegates all rendering to layout_v7_6.py

Layout Philosophy:
    NAV → AUM → Risk → Router → Positions → Performance → Treasury → Diagnostics

All rendering logic lives in layout_v7_6.py for maintainability.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import streamlit as st

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# State loaders
from dashboard.state_v7 import (
    load_all_state,
    load_risk_snapshot,
    load_offchain_assets,
    load_offchain_yield,
    validate_surface_health,
    load_engine_metadata,
)
from dashboard.live_helpers import (
    load_expectancy_v6,
    load_router_health_state,
)

# Layout engine
from dashboard.layout_v7_6 import (
    render_header_block,
    render_kpi_strip,
    render_aum_block,
    render_runtime_block,
    render_positions_block,
    render_strategy_block,
    render_treasury_block,
    render_diagnostics_block,
)

# Logging
LOG = logging.getLogger("dash.app")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG.setLevel(logging.INFO)


# =============================================================================
# CSS INJECTION
# =============================================================================

def _inject_css() -> None:
    """Inject institutional quant design system."""
    from pathlib import Path
    
    # Load the quant theme CSS
    css_path = Path(__file__).parent / "static" / "quant_theme.css"
    if css_path.exists():
        css_content = css_path.read_text()
    else:
        css_content = ""
    
    # Inject CSS
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


# =============================================================================
# STATE LOADER
# =============================================================================

def _load_dashboard_state() -> Dict[str, Any]:
    """
    Load all state required for dashboard rendering.
    
    Returns single dict with all state surfaces for easy passing to layout.
    """
    # Core state
    state = load_all_state()
    
    # Extract components
    nav_state = state.get("nav", {})
    aum_data = state.get("aum", {})
    kpis = state.get("kpis", {})
    positions = state.get("positions", [])
    meta = state.get("meta", {})
    
    # Additional surfaces
    risk_snapshot = load_risk_snapshot()
    router_health = load_router_health_state() or {}
    expectancy = load_expectancy_v6() or {}
    offchain_assets = load_offchain_assets({})
    offchain_yield = load_offchain_yield({})
    engine_meta = load_engine_metadata({})
    
    # Compute derived values
    nav_usd = float(nav_state.get("nav_usd") or nav_state.get("nav") or nav_state.get("total_equity") or 0)
    gross_exposure = float(nav_state.get("gross_exposure") or 0)
    
    return {
        "nav_state": nav_state,
        "aum_data": aum_data,
        "kpis": kpis,
        "positions": positions,
        "meta": meta,
        "risk_snapshot": risk_snapshot,
        "router_health": router_health,
        "expectancy": expectancy,
        "offchain_assets": offchain_assets,
        "offchain_yield": offchain_yield,
        "engine_meta": engine_meta,
        "nav_usd": nav_usd,
        "gross_exposure": gross_exposure,
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Main dashboard entry point."""
    # Page config (must be first Streamlit call)
    st.set_page_config(
        page_title="Hedge Dashboard",
        page_icon="dashboard/static/favicon.svg",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    
    # Inject CSS
    _inject_css()
    
    # Load all state
    state = _load_dashboard_state()
    
    # =========================================================================
    # HEADER
    # =========================================================================
    render_header_block(
        nav_state=state["nav_state"],
        engine_meta=state["engine_meta"],
    )
    
    # =========================================================================
    # KPI STRIP (horizontal bar of key metrics)
    # =========================================================================
    render_kpi_strip(
        nav_state=state["nav_state"],
        aum_data=state["aum_data"],
        kpis=state["kpis"],
        risk_snapshot=state["risk_snapshot"],
    )
    
    st.divider()
    
    # =========================================================================
    # AUM DOUGHNUT (primary visual anchor)
    # =========================================================================
    render_aum_block(
        nav_state=state["nav_state"],
        aum_data=state["aum_data"],
        kpis=state["kpis"],
    )
    
    st.divider()
    
    # =========================================================================
    # RUNTIME HEALTH (Risk + Router side by side)
    # =========================================================================
    render_runtime_block(
        risk_snapshot=state["risk_snapshot"],
        router_health=state["router_health"],
        nav_value=state["nav_usd"],
        gross_exposure=state["gross_exposure"],
    )
    
    st.divider()
    
    # =========================================================================
    # POSITIONS & EXECUTION
    # =========================================================================
    render_positions_block(
        positions=state["positions"],
        meta=state["meta"],
    )
    
    st.divider()
    
    # =========================================================================
    # STRATEGY PERFORMANCE
    # =========================================================================
    render_strategy_block(
        expectancy_data=state["expectancy"],
        kpis=state["kpis"],
    )
    
    st.divider()
    
    # =========================================================================
    # TREASURY (collapsed by default)
    # =========================================================================
    with st.expander("Treasury & Off-Exchange Assets", expanded=False):
        render_treasury_block(
            offchain_assets=state["offchain_assets"],
            offchain_yield=state["offchain_yield"],
        )
    
    st.divider()
    
    # =========================================================================
    # DIAGNOSTICS (collapsed by default, at the bottom)
    # =========================================================================
    with st.expander("Diagnostics & Raw State", expanded=False):
        render_diagnostics_block(
            state_summary={
                "nav": state["nav_state"],
                "aum": state["aum_data"],
                "risk": state["risk_snapshot"],
                "router": state["router_health"],
                "engine": state["engine_meta"],
            },
        )


if __name__ == "__main__":
    main()
