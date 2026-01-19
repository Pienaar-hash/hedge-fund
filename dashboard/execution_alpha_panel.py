"""
Execution Alpha Panel — v7.9_P4

Dashboard panel for Execution Alpha Engine metrics.

Displays:
- Top/bottom symbols by cumulative alpha USD
- Worst tail alpha (p99) symbols
- Per-head execution drag leaderboard
- Regime-conditioned alpha summaries
- Recent alerts

Author: Execution Alpha Engine v7.9_P4
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import streamlit as st
    import pandas as pd
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    st = None
    pd = None

_LOG = logging.getLogger(__name__)

# State file path
_STATE_DIR = Path("logs/state")
_ALPHA_STATE_FILE = _STATE_DIR / "execution_alpha.json"
_EVENTS_FILE = Path("logs/execution/execution_alpha_events.jsonl")


def load_alpha_state() -> Dict[str, Any]:
    """Load execution alpha state from disk."""
    try:
        if _ALPHA_STATE_FILE.exists():
            return json.loads(_ALPHA_STATE_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        _LOG.warning("Failed to load execution_alpha state: %s", e)
    return {"updated_ts": None, "symbols": {}, "heads": {}, "meta": {}}


def load_recent_alerts(hours: int = 24, max_events: int = 50) -> List[Dict[str, Any]]:
    """Load recent alpha alert events from JSONL log."""
    events = []
    try:
        if not _EVENTS_FILE.exists():
            return events
        
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        with open(_EVENTS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    ts_str = event.get("ts", "")
                    if ts_str:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if ts.replace(tzinfo=None) > cutoff:
                            events.append(event)
                except (json.JSONDecodeError, ValueError):
                    continue
        
        return events[-max_events:]
    except Exception as e:
        _LOG.warning("Failed to load alpha events: %s", e)
        return []


def get_panel_data() -> Dict[str, Any]:
    """
    Load all data needed for the panel.
    
    Returns dict with:
    - state: Full execution alpha state
    - alerts: Recent alert events
    - symbol_count: Number of symbols tracked
    - head_count: Number of heads tracked
    - meta: Metadata from state
    """
    state = load_alpha_state()
    alerts = load_recent_alerts()
    
    return {
        "state": state,
        "alerts": alerts,
        "symbol_count": len(state.get("symbols", {})),
        "head_count": len(state.get("heads", {})),
        "meta": state.get("meta", {}),
    }


def render_execution_alpha_panel() -> None:
    """
    Render the execution alpha panel in Streamlit.
    
    Shows:
    - Summary metrics
    - Symbol alpha leaderboard
    - Head alpha leaderboard
    - Regime breakdown
    - Recent alerts
    """
    if not HAS_STREAMLIT:
        return
    
    st.subheader("⚡ Execution Alpha")
    
    data = get_panel_data()
    state = data["state"]
    meta = data["meta"]
    
    if not meta:
        st.info("Execution Alpha not enabled or no data available")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_alpha = meta.get("total_cum_alpha_usd", 0)
        color = "🟢" if total_alpha >= 0 else "🔴"
        st.metric(
            "Total Alpha (USD)",
            f"{color} ${total_alpha:,.2f}",
        )
    
    with col2:
        avg_alpha = meta.get("avg_alpha_bps", 0)
        st.metric("Avg Alpha (bps)", f"{avg_alpha:.1f}")
    
    with col3:
        st.metric("Symbols Tracked", data["symbol_count"])
    
    with col4:
        st.metric("Heads Tracked", data["head_count"])
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Symbol Alpha",
        "🎯 Head Alpha",
        "📈 Regime Breakdown",
        "🚨 Alerts",
    ])
    
    with tab1:
        render_symbol_alpha_table(state)
    
    with tab2:
        render_head_alpha_table(state)
    
    with tab3:
        render_regime_breakdown(state)
    
    with tab4:
        render_alerts(data["alerts"])


def render_symbol_alpha_table(state: Dict[str, Any]) -> None:
    """Render symbol alpha leaderboard."""
    if not HAS_STREAMLIT:
        return
    
    symbols = state.get("symbols", {})
    if not symbols:
        st.info("No symbol alpha data available")
        return
    
    # Convert to dataframe
    rows = []
    for symbol, stats in symbols.items():
        rows.append({
            "Symbol": symbol,
            "Samples": stats.get("samples", 0),
            "Cum Alpha ($)": stats.get("cum_alpha_usd", 0),
            "Avg (bps)": stats.get("avg_alpha_bps", 0),
            "P95 (bps)": stats.get("p95_alpha_bps", 0),
            "P99 (bps)": stats.get("p99_alpha_bps", 0),
            "Drag (bps)": stats.get("avg_drag_bps", 0),
            "Multiplier": stats.get("suggested_multiplier", 1.0),
        })
    
    df = pd.DataFrame(rows)
    
    # Sort by cumulative alpha (worst first)
    df = df.sort_values("Cum Alpha ($)", ascending=True)
    
    # Color code based on alpha
    def style_alpha(val):
        if isinstance(val, (int, float)):
            if val < -10:
                return "color: red"
            elif val < 0:
                return "color: orange"
            elif val > 5:
                return "color: green"
        return ""
    
    st.dataframe(
        df.style.applymap(style_alpha, subset=["Avg (bps)", "P95 (bps)", "P99 (bps)"]),
        use_container_width=True,
        hide_index=True,
    )
    
    # Highlight worst performers
    if len(df) > 0:
        worst = df.iloc[0]
        if worst["Cum Alpha ($)"] < -5:
            st.warning(
                f"⚠️ Worst execution: **{worst['Symbol']}** with "
                f"${worst['Cum Alpha ($)']:.2f} total alpha ({worst['Avg (bps)']:.1f} bps avg)"
            )


def render_head_alpha_table(state: Dict[str, Any]) -> None:
    """Render head alpha leaderboard."""
    if not HAS_STREAMLIT:
        return
    
    heads = state.get("heads", {})
    if not heads:
        st.info("No head alpha data available")
        return
    
    # Convert to dataframe
    rows = []
    for head, stats in heads.items():
        rows.append({
            "Head": head,
            "Samples": stats.get("samples", 0),
            "Cum Alpha ($)": stats.get("cum_alpha_usd", 0),
            "Avg (bps)": stats.get("avg_alpha_bps", 0),
            "P95 (bps)": stats.get("p95_alpha_bps", 0),
            "Drag (bps)": stats.get("avg_drag_bps", 0),
            "Multiplier": stats.get("suggested_multiplier", 1.0),
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values("Cum Alpha ($)", ascending=True)
    
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Show penalty status
    penalties_active = state.get("meta", {}).get("penalties_active", {})
    if any(penalties_active.values()):
        active = [k for k, v in penalties_active.items() if v]
        st.info(f"🔧 Active penalties: {', '.join(active)}")


def render_regime_breakdown(state: Dict[str, Any]) -> None:
    """Render regime-conditioned alpha breakdown."""
    if not HAS_STREAMLIT:
        return
    
    symbols = state.get("symbols", {})
    if not symbols:
        st.info("No regime data available")
        return
    
    # Aggregate regime stats across all symbols
    regime_totals: Dict[str, Dict[str, float]] = {}
    
    for symbol, stats in symbols.items():
        breakdown = stats.get("regime_breakdown", {})
        for regime, regime_stats in breakdown.items():
            if regime not in regime_totals:
                regime_totals[regime] = {"samples": 0, "cum_alpha_usd": 0, "alpha_bps_sum": 0}
            
            regime_totals[regime]["samples"] += regime_stats.get("samples", 0)
            regime_totals[regime]["cum_alpha_usd"] += regime_stats.get("cum_alpha_usd", 0)
            regime_totals[regime]["alpha_bps_sum"] += (
                regime_stats.get("avg_alpha_bps", 0) * regime_stats.get("samples", 0)
            )
    
    # Create summary
    rows = []
    for regime, totals in regime_totals.items():
        samples = totals["samples"]
        avg_bps = totals["alpha_bps_sum"] / samples if samples > 0 else 0
        rows.append({
            "Regime": regime,
            "Samples": samples,
            "Cum Alpha ($)": round(totals["cum_alpha_usd"], 2),
            "Avg Alpha (bps)": round(avg_bps, 2),
        })
    
    if rows:
        df = pd.DataFrame(rows)
        df = df.sort_values("Samples", ascending=False)
        
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Highlight problem regimes
        for _, row in df.iterrows():
            if row["Avg Alpha (bps)"] < -10 and row["Samples"] >= 20:
                st.error(
                    f"🔴 **{row['Regime']}** regime has significant execution drag: "
                    f"{row['Avg Alpha (bps)']:.1f} bps avg over {row['Samples']} fills"
                )


def render_alerts(alerts: List[Dict[str, Any]]) -> None:
    """Render recent alert events."""
    if not HAS_STREAMLIT:
        return
    
    if not alerts:
        st.info("No recent alerts")
        return
    
    st.markdown(f"**{len(alerts)} recent alerts**")
    
    # Group by type
    tail_alerts = [a for a in alerts if a.get("event") == "EXEC_ALPHA_TAIL"]
    drag_alerts = [a for a in alerts if a.get("event") == "EXEC_DRAG_HIGH"]
    
    if tail_alerts:
        st.markdown("#### 🔴 Tail Slippage Events")
        for alert in tail_alerts[-10:]:
            st.markdown(
                f"- **{alert.get('symbol', 'N/A')}**: {alert.get('alpha_bps', 0):.1f} bps "
                f"(${alert.get('alpha_usd', 0):.2f}) in {alert.get('regime', 'UNKNOWN')} regime"
            )
    
    if drag_alerts:
        st.markdown("#### 🟠 High Drag Warnings")
        for alert in drag_alerts[-10:]:
            st.markdown(
                f"- **{alert.get('symbol', 'N/A')}**: {alert.get('avg_drag_bps', 0):.1f} bps avg drag "
                f"over {alert.get('samples', 0)} samples"
            )


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "load_alpha_state",
    "load_recent_alerts",
    "get_panel_data",
    "render_execution_alpha_panel",
    "render_symbol_alpha_table",
    "render_head_alpha_table",
    "render_regime_breakdown",
    "render_alerts",
]
