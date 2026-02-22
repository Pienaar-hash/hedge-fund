"""
NAV Composition Panel — v7.9

Investor-safe NAV breakdown showing WHERE the money is.
Mechanical, not interpretive. Read-only.

Data sources:
- logs/state/nav.json (asset breakdown)
- logs/state/nav_state.json (PnL, exposure)
- logs/state/episode_ledger.json (cycle fees, realized PnL)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st


# ---------------------------------------------------------------------------
# State Loaders
# ---------------------------------------------------------------------------

def load_nav_detail(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load detailed NAV with asset breakdown."""
    p = path or Path("logs/state/nav.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


def load_nav_state(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load NAV state with PnL metrics."""
    p = path or Path("logs/state/nav_state.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


def load_episode_ledger(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load episode ledger for cycle fees and realized PnL."""
    p = path or Path("logs/state/episode_ledger.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


# ---------------------------------------------------------------------------
# Widget Renderer
# ---------------------------------------------------------------------------

def render_nav_composition_panel(
    nav_detail: Optional[Dict[str, Any]] = None,
    nav_state: Optional[Dict[str, Any]] = None,
    episode_ledger: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Render NAV composition breakdown.
    
    Shows:
    - Cash vs Positions (by asset)
    - Unrealized vs Realized PnL
    - Fees paid (cycle-to-date)
    
    Answers: "Where is the money, and what state is it in?"
    No opinions. No forecasts. No attribution.
    """
    if nav_detail is None:
        nav_detail = load_nav_detail()
    if nav_state is None:
        nav_state = load_nav_state()
    if episode_ledger is None:
        episode_ledger = load_episode_ledger()
    
    if not nav_detail:
        return
    
    # Extract NAV total
    nav_total = float(nav_detail.get("nav_usd") or nav_detail.get("nav") or 0)
    
    # Asset breakdown
    assets = nav_detail.get("assets", {})
    if not assets:
        assets = nav_detail.get("nav_detail", {}).get("asset_breakdown", {})
    
    # Separate stablecoins (cash) from non-stables (positions/collateral)
    stablecoins = {"USDT", "USDC", "BUSD", "FDUSD", "TUSD", "DAI"}
    
    cash_total = 0.0
    position_total = 0.0
    cash_assets = {}
    position_assets = {}
    
    for asset, value in assets.items():
        val = float(value) if value else 0
        if val == 0:
            continue
        if asset in stablecoins:
            cash_total += val
            cash_assets[asset] = val
        else:
            position_total += val
            position_assets[asset] = val
    
    # PnL from nav_state
    unrealized_pnl = float(nav_state.get("unrealized_pnl") or 0)
    gross_exposure = float(nav_state.get("gross_exposure") or 0)

    # 24h NAV delta — TRUE portfolio PnL (replaces stale session counter)
    # SPAN AUTHORITY: Only display NAV delta if log span is sufficient.
    from dashboard.components.nav_pnl import compute_nav_deltas
    _nav_deltas = compute_nav_deltas()
    _span_ok = _nav_deltas.get("span_ok", {})
    nav_delta_24h = _nav_deltas.get("pnl_24h", 0.0) if _span_ok.get("24h", False) else 0.0
    
    # Cycle metrics from episode ledger (closed trades only)
    ledger_stats = episode_ledger.get("stats", {})
    cycle_net_pnl = float(ledger_stats.get("total_net_pnl") or 0)
    cycle_fees = float(ledger_stats.get("total_fees") or 0)
    episode_count = episode_ledger.get("episode_count", 0)
    
    # Calculate percentages
    cash_pct = (cash_total / nav_total * 100) if nav_total > 0 else 0
    position_pct = (position_total / nav_total * 100) if nav_total > 0 else 0
    
    # Build asset rows HTML
    def _asset_row(asset: str, value: float, total: float) -> str:
        pct = (value / total * 100) if total > 0 else 0
        return f'''
        <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #1a1d24;">
            <span style="color: #888;">{asset}</span>
            <span style="color: #ccc;">${value:,.2f} <span style="color: #555; font-size: 0.85em;">({pct:.1f}%)</span></span>
        </div>
        '''
    
    cash_rows = "".join(_asset_row(a, v, nav_total) for a, v in sorted(cash_assets.items(), key=lambda x: -x[1]))
    position_rows = "".join(_asset_row(a, v, nav_total) for a, v in sorted(position_assets.items(), key=lambda x: -x[1]))
    
    if not position_rows:
        position_rows = '<div style="color: #555; font-size: 0.85em; padding: 4px 0;">No open positions</div>'
    
    # PnL colors
    unrealized_color = "#22c55e" if unrealized_pnl >= 0 else "#ef4444"
    nav_delta_color = "#22c55e" if nav_delta_24h >= 0 else "#ef4444"
    cycle_pnl_color = "#22c55e" if cycle_net_pnl >= 0 else "#ef4444"
    
    # Build full widget HTML
    html = f'''
    <div style="
        background: linear-gradient(135deg, #1a1d24 0%, #12141a 100%);
        border: 1px solid #2d3139;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    ">
        <!-- Header -->
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px;">
            <div>
                <span style="font-size: 0.75em; color: #888; text-transform: uppercase; letter-spacing: 0.5px;">
                    💰 NAV Composition
                </span>
                <div style="font-size: 1.6em; font-weight: 700; color: #fff; margin-top: 4px;">
                    ${nav_total:,.2f}
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.7em; color: #666;">Source of Truth</div>
                <div style="font-size: 0.85em; color: #888;">Futures Wallet</div>
            </div>
        </div>
        
        <!-- Two Column Layout -->
        <div style="display: flex; gap: 24px;">
            
            <!-- Left: Asset Breakdown -->
            <div style="flex: 1;">
                <!-- Cash Section -->
                <div style="margin-bottom: 16px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-size: 0.85em; font-weight: 600; color: #22c55e;">Cash (Stablecoins)</span>
                        <span style="font-size: 0.9em; font-weight: 600; color: #22c55e;">${cash_total:,.2f} <span style="color: #666; font-size: 0.85em;">({cash_pct:.1f}%)</span></span>
                    </div>
                    <div style="background: #12141a; border-radius: 4px; padding: 8px;">
                        {cash_rows}
                    </div>
                </div>
                
                <!-- Positions Section -->
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-size: 0.85em; font-weight: 600; color: #f59e0b;">Positions (Non-Stable)</span>
                        <span style="font-size: 0.9em; font-weight: 600; color: #f59e0b;">${position_total:,.2f} <span style="color: #666; font-size: 0.85em;">({position_pct:.1f}%)</span></span>
                    </div>
                    <div style="background: #12141a; border-radius: 4px; padding: 8px;">
                        {position_rows}
                    </div>
                </div>
            </div>
            
            <!-- Right: PnL State -->
            <div style="min-width: 200px;">
                <div style="font-size: 0.85em; font-weight: 600; color: #888; margin-bottom: 12px;">PnL State</div>
                
                <!-- Unrealized -->
                <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #1a1d24;">
                    <span style="color: #888; font-size: 0.85em;">Unrealized</span>
                    <span style="color: {unrealized_color}; font-weight: 600;">{'+' if unrealized_pnl >= 0 else ''}${unrealized_pnl:,.2f}</span>
                </div>
                
                <!-- 24h PnL (NAV Delta) -->
                <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #1a1d24;">
                    <span style="color: #888; font-size: 0.85em;">24h PnL (NAV)</span>
                    <span style="color: {nav_delta_color}; font-weight: 600;">{'+' if nav_delta_24h >= 0 else ''}${nav_delta_24h:,.2f}</span>
                </div>
                
                <!-- Closed PnL (Episodes) -->
                <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #1a1d24;">
                    <span style="color: #888; font-size: 0.85em;">Closed PnL (Episodes)</span>
                    <span style="color: {cycle_pnl_color}; font-weight: 600;">{'+' if cycle_net_pnl >= 0 else ''}${cycle_net_pnl:,.2f}</span>
                </div>
                
                <!-- Fees Paid -->
                <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #1a1d24;">
                    <span style="color: #888; font-size: 0.85em;">Fees Paid (Cycle)</span>
                    <span style="color: #ef4444;">-${cycle_fees:,.2f}</span>
                </div>
                
                <!-- Episodes -->
                <div style="display: flex; justify-content: space-between; padding: 8px 0;">
                    <span style="color: #888; font-size: 0.85em;">Episodes Closed</span>
                    <span style="color: #888;">{episode_count}</span>
                </div>
                
                <!-- Exposure -->
                <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #2d3139;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #666; font-size: 0.8em;">Gross Exposure</span>
                        <span style="color: {'#f59e0b' if gross_exposure > 0 else '#555'}; font-size: 0.85em;">${gross_exposure:,.2f}</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    '''
    
    st.html(html)


# ---------------------------------------------------------------------------
# Standalone Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    render_nav_composition_panel()
