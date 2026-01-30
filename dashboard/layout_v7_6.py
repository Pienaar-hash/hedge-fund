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
    version = engine_meta.get("engine_version") or engine_meta.get("version") or "v7.9"
    
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
    
    # Inline SVG logo (Streamlit can't serve static files via HTML img src)
    logo_svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="24" height="24">
        <rect x="4" y="18" width="5" height="10" fill="#1a56db"/>
        <rect x="11" y="12" width="5" height="16" fill="#1a56db"/>
        <rect x="18" y="6" width="5" height="22" fill="#1a56db"/>
        <rect x="25" y="14" width="5" height="14" fill="#1a56db" opacity="0.6"/>
    </svg>'''
    
    header_html = f'''
    <div class="header">
        <div class="header-left">
            <div class="header-brand">
                {logo_svg}
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

def _compute_windowed_pnl(episodes: List[Dict[str, Any]], hours: int) -> float:
    """Compute PnL from episodes closed within the last N hours."""
    from datetime import datetime, timedelta, timezone
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    total = 0.0
    for ep in episodes:
        exit_ts = ep.get("exit_ts", "")
        if not exit_ts:
            continue
        try:
            ts = datetime.fromisoformat(exit_ts.replace("Z", "+00:00"))
            if ts > cutoff:
                total += float(ep.get("net_pnl", 0) or 0)
        except (ValueError, TypeError):
            continue
    return total


def render_strategy_block(
    expectancy_data: Dict[str, Any],
    kpis: Dict[str, Any],
    episode_ledger: Optional[Dict[str, Any]] = None,
    nav_state: Optional[Dict[str, Any]] = None,
) -> None:
    """Render strategy performance with 4 KPI cards.
    
    Data priority:
    1. episode_ledger.stats (authoritative historical)
    2. expectancy_data (real-time estimates)
    3. kpis (fallback)
    """
    # Start with base kpis
    merged_kpis = {**kpis}
    
    # Layer in expectancy data
    if expectancy_data:
        merged_kpis.setdefault("sharpe", expectancy_data.get("sharpe"))
        merged_kpis.setdefault("total_trades", expectancy_data.get("n_trades"))
    
    # Episode ledger is authoritative for historical metrics
    if episode_ledger:
        stats = episode_ledger.get("stats", {})
        episode_count = episode_ledger.get("episode_count", 0)
        episodes = episode_ledger.get("episodes", [])
        
        # Win rate from episode ledger (already percentage, e.g. 6.2)
        if stats.get("win_rate") is not None:
            merged_kpis["win_rate"] = stats.get("win_rate")
        
        # PnL from episode ledger
        if stats.get("total_net_pnl") is not None:
            merged_kpis["total_pnl"] = stats.get("total_net_pnl")
            merged_kpis["all_time_pnl"] = stats.get("total_net_pnl")
        
        # Time-windowed PnL from episodes
        if episodes:
            merged_kpis["daily_pnl"] = _compute_windowed_pnl(episodes, 24)
            merged_kpis["pnl_24h"] = merged_kpis["daily_pnl"]
            merged_kpis["weekly_pnl"] = _compute_windowed_pnl(episodes, 168)
            merged_kpis["pnl_7d"] = merged_kpis["weekly_pnl"]
            merged_kpis["monthly_pnl"] = _compute_windowed_pnl(episodes, 720)
            merged_kpis["pnl_30d"] = merged_kpis["monthly_pnl"]
        
        # Max drawdown from episode ledger (already percentage)
        if stats.get("max_drawdown_pct") is not None:
            merged_kpis["max_drawdown"] = stats.get("max_drawdown_pct")
            merged_kpis["max_dd"] = stats.get("max_drawdown_pct")
        
        # Trade count
        if episode_count:
            merged_kpis["total_trades"] = episode_count
            merged_kpis["trades_count"] = episode_count
        
        # Winners/losers for display
        merged_kpis["winners"] = stats.get("winners", 0)
        merged_kpis["losers"] = stats.get("losers", 0)
    
    # Extract equity curve from NAV state series
    equity_curve = None
    if nav_state:
        series = nav_state.get("series", [])
        if series and len(series) > 1:
            equity_curve = series
    
    render_performance_block(merged_kpis, equity_curve=equity_curve)


# =============================================================================
# TREASURY BLOCK
# =============================================================================

def render_treasury_block(
    offchain_assets: Dict[str, Any],
    offchain_yield: Dict[str, Any],
) -> None:
    """Render treasury / off-exchange holdings with staleness indicator."""
    # Convert offchain_assets dict to list
    # The state file has structure: {assets: {BTC: {...}, ...}, totals: {...}, metadata: {...}}
    holdings = []
    
    # Extract timestamp for staleness check
    updated_ts = offchain_assets.get("updated_ts")
    
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
    
    _render_treasury_block(holdings, total_usd, updated_ts)


# =============================================================================
# DIAGNOSTICS BLOCK
# =============================================================================

def render_diagnostics_block(
    state_summary: Dict[str, Any],
) -> None:
    """Render diagnostics panel with state file health."""
    import time
    import json
    from pathlib import Path
    
    # Build state health list from actual state files
    state_health = []
    
    # Core state files to monitor
    state_files = [
        ("nav.json", Path("logs/state/nav.json")),
        ("nav_state.json", Path("logs/state/nav_state.json")),
        ("risk_snapshot.json", Path("logs/state/risk_snapshot.json")),
        ("router_health.json", Path("logs/state/router_health.json")),
        ("positions_state.json", Path("logs/state/positions_state.json")),
        ("diagnostics.json", Path("logs/state/diagnostics.json")),
        ("sentinel_x.json", Path("logs/state/sentinel_x.json")),
        ("regime_pressure.json", Path("logs/state/regime_pressure.json")),
        ("kpis_v7.json", Path("logs/state/kpis_v7.json")),
        ("engine_metadata.json", Path("logs/state/engine_metadata.json")),
        ("episode_ledger.json", Path("logs/state/episode_ledger.json")),
    ]
    
    for name, path in state_files:
        if not path.exists():
            state_health.append({
                "name": name,
                "status": "missing",
                "age_s": None,
                "size_bytes": None,
            })
            continue
        
        try:
            stat = path.stat()
            size = stat.st_size
            mtime = stat.st_mtime
            age_s = time.time() - mtime
            
            # Status based on age (5 min = stale for most files)
            status = "stale" if age_s > 300 else "ok"
            
            state_health.append({
                "name": name,
                "status": status,
                "age_s": age_s,
                "size_bytes": size,
            })
        except Exception:
            state_health.append({
                "name": name,
                "status": "error",
                "age_s": None,
                "size_bytes": None,
            })
    
    # Executor status from engine meta
    engine = state_summary.get("engine") or {}
    executor_status = None
    if engine:
        executor_status = {
            "running": engine.get("status") in ("running", "active", "ok"),
            "uptime_s": engine.get("uptime_s") or engine.get("uptime"),
            "last_cycle_s": engine.get("last_cycle_s"),
        }
    
    _render_diagnostics_block(state_health, executor_status)
