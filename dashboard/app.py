# mypy: ignore-errors
"""
Hedge Dashboard — Streamlit Application

This is the main entry point for the dashboard. It serves as a minimal
orchestrator that:
    1. Configures Streamlit page settings
    2. Loads state from canonical surfaces
    3. Delegates all rendering to layout.py

Layout Philosophy:
    NAV → AUM → Risk → Router → Positions → Performance → Treasury → Diagnostics

All rendering logic lives in layout.py for maintainability.
See DASHBOARD_POLICY.md for architecture rules.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
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
    validate_surface_health,
    load_engine_metadata,
)
from dashboard.live_helpers import (
    load_expectancy_v6,
    load_router_health_state,
)

# P0 Widgets — Regime Visibility
from dashboard.components.regime_pressure import (
    load_regime_pressure_state,
    render_regime_pressure_widget,
)
from dashboard.components.sentinel_x import (
    load_sentinel_x_state,
    render_sentinel_x_compact,
)

# Governance Widget — DLE Authority Gate
from dashboard.components.enforcement import (
    load_enforcement_state,
    render_enforcement_widget,
)

# P1 Widgets — Strategy Transparency
from dashboard.components.episode_ledger import (
    load_episode_ledger_state,
    render_episode_ledger_summary,
)
# ARCHIVED 2026-01-29: Hydra status strip disabled
# from dashboard.components.hydra_status import (
#     load_hydra_state,
#     render_hydra_status_strip,
# )

# NAV Composition — Investor Truth Surface
from dashboard.components.nav_composition import (
    load_nav_detail,
    render_nav_composition_panel,
)

# P1 Prediction Telemetry (advisory read-only)
from dashboard.components.prediction_tile import (
    load_prediction_telemetry,
    render_prediction_tile,
)

# D.2 Execution Visibility
from dashboard.components.execution_quality import (
    load_execution_quality_state,
    render_execution_quality_widget,
)
from dashboard.components.pnl_attribution import (
    load_pnl_attribution_state,
    render_pnl_attribution_widget,
)
from dashboard.components.alpha_decay import (
    load_alpha_decay_state,
    render_alpha_decay_widget,
)

# Equity Curve
from dashboard.components.equity_curve import render_equity_curve

# Architecture Health Strip
from dashboard.components.architecture_strip import (
    load_architecture_health,
    render_architecture_strip,
)

# Edge Calibration
from dashboard.components.edge_calibration_panel import (
    render_edge_calibration_panel,
)

# Engine Lift (Hydra vs Legacy)
from dashboard.components.engine_lift_panel import (
    render_engine_lift_panel,
)

# Hydra Score Monotonicity
from dashboard.components.hydra_monotonicity_panel import (
    render_hydra_monotonicity_panel,
)

# Multi-Engine Soak Panel
from dashboard.multi_engine_panel import render_multi_engine_panel

# Layout engine
from dashboard.layout import (
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

# Process start time (captured once at import)
_PROCESS_START_UTC = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _get_build_info() -> Dict[str, str]:
    """Return version, git commit, and process start time (cached per process)."""
    version = "unknown"
    version_path = PROJECT_ROOT / "VERSION"
    if version_path.exists():
        version = version_path.read_text().strip()

    commit = "unknown"
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(PROJECT_ROOT),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        pass

    return {"version": version, "commit": commit, "started": _PROCESS_START_UTC}


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
    engine_meta = load_engine_metadata({})
    
    # P0: Regime visibility surfaces
    regime_pressure = load_regime_pressure_state()
    sentinel_x = load_sentinel_x_state()
    
    # Governance: DLE enforcement surface
    enforcement_state = load_enforcement_state()

    # P1: Strategy transparency surfaces
    episode_ledger = load_episode_ledger_state()
    # ARCHIVED 2026-01-29: Hydra state disabled
    # hydra_state = load_hydra_state()
    hydra_state = {}
    
    # NAV Composition (investor truth surface)
    nav_detail = load_nav_detail()
    
    # P1 Prediction telemetry
    prediction_telemetry = load_prediction_telemetry()
    
    # Architecture health surface
    architecture_health = load_architecture_health()

    # D.2: Execution visibility surfaces
    execution_quality = load_execution_quality_state()
    pnl_attribution = load_pnl_attribution_state()
    alpha_decay = load_alpha_decay_state()
    
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
        "engine_meta": engine_meta,
        "nav_usd": nav_usd,
        "gross_exposure": gross_exposure,
        # P0: Regime visibility
        "regime_pressure": regime_pressure,
        "sentinel_x": sentinel_x,
        # Governance
        "enforcement_state": enforcement_state,
        # P1: Strategy transparency
        "episode_ledger": episode_ledger,
        "hydra_state": hydra_state,
        # NAV Composition
        "nav_detail": nav_detail,
        # Prediction telemetry
        "prediction_telemetry": prediction_telemetry,
        # Architecture health
        "architecture_health": architecture_health,
        # D.2: Execution visibility
        "execution_quality": execution_quality,
        "pnl_attribution": pnl_attribution,
        "alpha_decay": alpha_decay,
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
    # ARCHITECTURE HEALTH (single-glance soak readiness)
    # =========================================================================
    render_architecture_strip(state["architecture_health"])
    
    # =========================================================================
    # EDGE CALIBRATION (predicted edge vs realized return)
    # =========================================================================
    render_edge_calibration_panel(state["episode_ledger"])
    
    # =========================================================================
    # ENGINE LIFT (Hydra vs Legacy outcome comparison)
    # =========================================================================
    render_engine_lift_panel()

    # =========================================================================
    # HYDRA SCORE MONOTONICITY (score ordering quality)
    # =========================================================================
    render_hydra_monotonicity_panel()
    
    # =========================================================================
    # KPI STRIP (horizontal bar of key metrics)
    # =========================================================================
    render_kpi_strip(
        nav_state=state["nav_state"],
        aum_data=state["aum_data"],
        kpis=state["kpis"],
        risk_snapshot=state["risk_snapshot"],
        episode_ledger=state["episode_ledger"],
    )
    
    # =========================================================================
    # P0: REGIME VISIBILITY (Why system is flat)
    # =========================================================================
    col1, col2 = st.columns(2)
    with col1:
        render_regime_pressure_widget(state["regime_pressure"])
    with col2:
        render_sentinel_x_compact(state["sentinel_x"])
    
    st.divider()
    
    # =========================================================================
    # NAV COMPOSITION (Investor Truth Surface — replaces AUM)
    # =========================================================================
    render_nav_composition_panel(
        nav_detail=state["nav_detail"],
        nav_state=state["nav_state"],
        episode_ledger=state["episode_ledger"],
    )
    
    # =========================================================================
    # EQUITY CURVE (NAV over time)
    # =========================================================================
    render_equity_curve()
    
    # =========================================================================
    # MULTI-ENGINE SOAK (Architecture health & migration readiness)
    # =========================================================================
    try:
        render_multi_engine_panel()
    except Exception as exc:
        LOG.warning("multi_engine_panel render failed: %s", exc)
    
    # =========================================================================
    # POSITIONS & EXECUTION
    # =========================================================================
    render_positions_block(
        positions=state["positions"],
        meta=state["meta"],
    )
    
    # =========================================================================
    # P1: STRATEGY TRANSPARENCY (Capital is intentionally idle)
    # =========================================================================
    render_episode_ledger_summary(state["episode_ledger"])
    
    # =========================================================================
    # STRATEGY PERFORMANCE
    # =========================================================================
    render_strategy_block(
        expectancy_data=state["expectancy"],
        kpis=state["kpis"],
        episode_ledger=state["episode_ledger"],
        nav_state=state["nav_state"],
    )
    
    # =========================================================================
    # OPERATIONAL DETAIL (collapsed — internals, only when relevant)
    # =========================================================================
    with st.expander("System Internals", expanded=False):
        render_runtime_block(
            risk_snapshot=state["risk_snapshot"],
            router_health=state["router_health"],
            nav_value=state["nav_usd"],
            gross_exposure=state["gross_exposure"],
        )
        st.divider()
        render_enforcement_widget(state["enforcement_state"])
        # Only show execution detail if there's actual fill data
        _exec_q = state["execution_quality"]
        if _exec_q and _exec_q.get("symbols"):
            st.divider()
            render_execution_quality_widget(_exec_q)
        # Only show PnL attribution if there are recorded trades
        _pnl_attr = state["pnl_attribution"]
        _pnl_count = (_pnl_attr.get("summary", {}).get("record_count", 0) or 0) if _pnl_attr else 0
        if _pnl_count > 0:
            st.divider()
            render_pnl_attribution_widget(_pnl_attr)
        # Only show alpha decay if enabled
        _alpha = state["alpha_decay"]
        if _alpha and _alpha.get("config", {}).get("enabled", False):
            st.divider()
            render_alpha_decay_widget(_alpha)
    
    # =========================================================================
    # DIAGNOSTICS (collapsed by default, at the bottom)
    # =========================================================================
    with st.expander("Diagnostics & Raw State", expanded=False):
        render_prediction_tile(state["prediction_telemetry"])
        st.divider()
        render_diagnostics_block(
            state_summary={
                "nav": state["nav_state"],
                "aum": state["aum_data"],
                "risk": state["risk_snapshot"],
                "router": state["router_health"],
                "engine": state["engine_meta"],
            },
        )

    # =========================================================================
    # BUILD FOOTER
    # =========================================================================
    build = _get_build_info()
    st.markdown(
        f'<div style="text-align:center;color:#555;font-size:0.75rem;padding:1.5rem 0 0.5rem;">'
        f'Build: {build["version"]} &nbsp;·&nbsp; Commit: {build["commit"]}'
        f' &nbsp;·&nbsp; Started: {build["started"]}</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
