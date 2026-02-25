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
# Helpers
# ---------------------------------------------------------------------------

def _build_exit_reasons_html(stats: Dict[str, Any]) -> str:
    """Build compact exit reasons HTML from ledger stats."""
    reasons = stats.get("exit_reasons", {})
    if not reasons:
        return '<span style="color: #555; font-size: 0.85em;">—</span>'
    # Sort by count desc, show top 4
    sorted_reasons = sorted(reasons.items(), key=lambda x: -x[1])[:4]
    total = sum(v for _, v in sorted_reasons)
    parts = []
    for reason, count in sorted_reasons:
        pct = (count / total * 100) if total > 0 else 0
        label = reason.replace("_", " ").title()
        parts.append(
            f'<div style="display:flex;justify-content:space-between;padding:2px 0;">'
            f'<span style="color:#666;font-size:0.8em;">{label}</span>'
            f'<span style="color:#888;font-size:0.8em;">{count} ({pct:.0f}%)</span></div>'
        )
    return "".join(parts)


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
    
    # Cycle metrics from episode ledger (closed trades only)
    ledger_stats = episode_ledger.get("stats", {})
    cycle_net_pnl = float(ledger_stats.get("total_net_pnl") or 0)  # already = gross - fees
    cycle_gross_pnl = float(ledger_stats.get("total_gross_pnl") or 0)
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
            
            <!-- Right: Trading Summary -->
            <div style="min-width: 200px;">
                <div style="font-size: 0.85em; font-weight: 600; color: #888; margin-bottom: 12px;">Trading Summary</div>
                
                <!-- Net PnL (Episodes) — already includes fee deduction -->
                <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #1a1d24;">
                    <span style="color: #888; font-size: 0.85em;">Closed-Trade PnL</span>
                    <span style="color: {cycle_pnl_color}; font-weight: 600;">${cycle_net_pnl:,.2f}</span>
                </div>
                
                <!-- Gross + Fees breakdown (informational) -->
                <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #1a1d24;">
                    <span style="color: #555; font-size: 0.75em;">Gross PnL</span>
                    <span style="color: #666; font-size: 0.85em;">${cycle_gross_pnl:,.2f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #1a1d24;">
                    <span style="color: #555; font-size: 0.75em;">Fees (incl. above)</span>
                    <span style="color: #666; font-size: 0.85em;">-${cycle_fees:,.2f}</span>
                </div>
                
                <!-- Episodes -->
                <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #1a1d24;">
                    <span style="color: #888; font-size: 0.85em;">Episodes</span>
                    <span style="color: #888;">{episode_count}</span>
                </div>
                
                <!-- Exit Reasons -->
                <div style="margin-top: 8px; padding-top: 8px;">
                    <div style="font-size: 0.75em; color: #555; text-transform: uppercase; margin-bottom: 6px;">Exit Reasons</div>
                    {_build_exit_reasons_html(ledger_stats)}
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
