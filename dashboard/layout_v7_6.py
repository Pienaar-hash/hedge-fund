"""
Layout Engine v7.6 — Institutional Dashboard Composition

This module provides the render functions called by app_v7_6.py.
Each function wraps the corresponding component module, providing
a clean interface with proper state unpacking.

All visual rendering is delegated to dashboard/components/*.py
which use the single-HTML-render pattern.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

# Import component renderers
from dashboard.components.kpi_strip import render_kpi_strip as _render_kpi_strip
from dashboard.components.aum import render_aum_block as _render_aum_block
from dashboard.components.runtime_health import render_runtime_health_block
from dashboard.components.positions import render_positions_block as _render_positions_block
from dashboard.components.performance import render_performance_block
from dashboard.components.treasury import render_treasury_block as _render_treasury_block
from dashboard.components.diagnostics import render_diagnostics_block as _render_diagnostics_block


# =============================================================================
# HEADER
# =============================================================================

def render_header_block(
    nav_state: Dict[str, Any],
    engine_meta: Dict[str, Any],
) -> None:
    """Render institutional header with system status."""
    # Extract values
    last_update = nav_state.get("updated_at") or nav_state.get("timestamp") or ""
    engine_status = engine_meta.get("status") or "unknown"
    version = engine_meta.get("version") or "v7.6"
    
    # Status badge class
    if engine_status in ("running", "active", "ok"):
        status_class = "normal"
        status_text = "ONLINE"
    elif engine_status in ("degraded", "partial"):
        status_class = "warning"
        status_text = "DEGRADED"
    else:
        status_class = "critical"
        status_text = "OFFLINE"
    
    # Format timestamp if present
    ts_display = ""
    if last_update:
        try:
            from datetime import datetime
            if isinstance(last_update, (int, float)):
                dt = datetime.fromtimestamp(last_update)
                ts_display = dt.strftime("%H:%M:%S")
            else:
                ts_display = str(last_update)[:19]
        except Exception:
            ts_display = str(last_update)
    
    header_html = f'''
    <div class="header">
        <div class="header-left">
            <div class="header-brand">
                <img src="dashboard/static/favicon.svg" class="header-logo" alt="" onerror="this.style.display='none'"/>
                <span class="header-title">GPT Hedge</span>
                <span class="header-version">{version}</span>
            </div>
        </div>
        <div class="header-right">
            <span class="status-badge {status_class}">{status_text}</span>
            <span class="header-timestamp">{ts_display}</span>
        </div>
    </div>
    '''
    
    st.html(header_html)


# =============================================================================
# KPI STRIP
# =============================================================================

def render_kpi_strip(
    nav_state: Dict[str, Any],
    aum_data: Dict[str, Any],
    kpis: Dict[str, Any],
    risk_snapshot: Dict[str, Any],
) -> None:
    """Render horizontal KPI strip with 8 key metrics."""
    _render_kpi_strip(nav_state, aum_data, kpis, risk_snapshot)


# =============================================================================
# AUM BLOCK
# =============================================================================

def render_aum_block(
    nav_state: Dict[str, Any],
    aum_data: Dict[str, Any],
    kpis: Dict[str, Any],
) -> None:
    """Render AUM section with allocation breakdown."""
    _render_aum_block(nav_state, aum_data)


# =============================================================================
# RUNTIME BLOCK (Risk + Router)
# =============================================================================

def render_runtime_block(
    risk_snapshot: Dict[str, Any],
    router_health: Dict[str, Any],
    nav_value: float,
    gross_exposure: float,
) -> None:
    """Render Risk Engine + Router Health side by side."""
    render_runtime_health_block(risk_snapshot, router_health, nav_value, gross_exposure)


# =============================================================================
# POSITIONS BLOCK
# =============================================================================

def render_positions_block(
    positions: List[Dict[str, Any]],
    meta: Dict[str, Any],
) -> None:
    """Render positions table with clean styling."""
    _render_positions_block(positions, meta)


# =============================================================================
# STRATEGY BLOCK
# =============================================================================

def render_strategy_block(
    expectancy_data: Dict[str, Any],
    kpis: Dict[str, Any],
) -> None:
    """Render strategy performance with 4 KPI cards."""
    # Merge expectancy into kpis for the performance component
    merged_kpis = {**kpis}
    if expectancy_data:
        # Map expectancy fields
        merged_kpis.setdefault("win_rate", expectancy_data.get("win_rate"))
        merged_kpis.setdefault("sharpe", expectancy_data.get("sharpe"))
        merged_kpis.setdefault("total_trades", expectancy_data.get("n_trades"))
    
    render_performance_block(merged_kpis, equity_curve=None)


# =============================================================================
# TREASURY BLOCK
# =============================================================================

def render_treasury_block(
    offchain_assets: Dict[str, Any],
    offchain_yield: Dict[str, Any],
) -> None:
    """Render treasury / off-exchange holdings."""
    # Convert offchain_assets dict to list
    # The state file has structure: {assets: {BTC: {...}, ...}, totals: {...}, metadata: {...}}
    holdings = []
    
    # Extract the assets sub-dict (not top-level keys like 'totals', 'metadata')
    assets_dict = offchain_assets.get("assets", {})
    if not assets_dict and isinstance(offchain_assets, dict):
        # Fallback: if no 'assets' key, check if it's the old simple format
        # where asset names are top-level keys (but skip known metadata keys)
        skip_keys = {"updated_ts", "source", "totals", "metadata", "assets"}
        assets_dict = {k: v for k, v in offchain_assets.items() if k not in skip_keys and isinstance(v, dict)}
    
    for asset, data in assets_dict.items():
        if isinstance(data, dict):
            qty = float(data.get("qty") or data.get("quantity") or 0)
            usd_value = data.get("usd_value")
            
            # If no USD value, compute from qty * current_price or qty * avg_cost
            if usd_value is None:
                current_price = data.get("current_price_usd")
                avg_cost = data.get("avg_cost_usd") or data.get("avg_cost")
                if current_price is not None:
                    usd_value = qty * float(current_price)
                elif avg_cost is not None:
                    usd_value = qty * float(avg_cost)
                else:
                    usd_value = 0.0
            
            holdings.append({
                "asset": asset,
                "quantity": qty,
                "usd_value": float(usd_value) if usd_value else 0.0,
                "location": data.get("source") or data.get("location") or "External",
            })
    
    # Calculate total
    total_usd = sum(float(h.get("usd_value") or 0) for h in holdings)
    
    _render_treasury_block(holdings, total_usd)


# =============================================================================
# DIAGNOSTICS BLOCK
# =============================================================================

def render_diagnostics_block(
    state_summary: Dict[str, Any],
) -> None:
    """Render diagnostics panel with state file health."""
    # Build state health list from summary
    state_health = []
    
    # Define expected state files
    state_files = [
        ("nav_state.json", state_summary.get("nav")),
        ("aum_state.json", state_summary.get("aum")),
        ("risk_snapshot.json", state_summary.get("risk")),
        ("router_health.json", state_summary.get("router")),
        ("engine_meta.json", state_summary.get("engine")),
    ]
    
    for name, data in state_files:
        if data is None:
            status = "missing"
            age_s = None
            size = None
        else:
            # Check for staleness based on updated_at
            import time
            updated = data.get("updated_at") or data.get("timestamp")
            if updated:
                try:
                    age_s = time.time() - float(updated)
                    status = "stale" if age_s > 300 else "ok"  # 5 min threshold
                except Exception:
                    age_s = None
                    status = "ok"
            else:
                age_s = None
                status = "ok"
            
            # Estimate size
            import json
            try:
                size = len(json.dumps(data))
            except Exception:
                size = None
        
        state_health.append({
            "name": name,
            "status": status,
            "age_s": age_s,
            "size_bytes": size,
        })
    
    # Executor status (if available in engine meta)
    engine = state_summary.get("engine") or {}
    executor_status = None
    if engine:
        executor_status = {
            "running": engine.get("status") in ("running", "active", "ok"),
            "uptime_s": engine.get("uptime_s") or engine.get("uptime"),
            "last_cycle_s": engine.get("last_cycle_s"),
        }
    
    _render_diagnostics_block(state_health, executor_status)
