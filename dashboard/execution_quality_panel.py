"""
Execution Quality Panel — v7.9_P3 (Minotaur)

Dashboard panel for Minotaur Execution Engine quality metrics.

Displays:
- Per-symbol slippage statistics (avg, p95, max)
- Per-symbol fill ratio and TWAP usage
- Current execution regime distribution
- Throttling status and alerts
- Slippage spike events
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
_QUALITY_STATE_FILE = _STATE_DIR / "execution_quality.json"
_EVENTS_FILE = Path("logs/execution/execution_events.jsonl")

# Regime colors for display
REGIME_COLORS = {
    "NORMAL": "🟢",
    "THIN": "🟡",
    "WIDE_SPREAD": "🟠",
    "SPIKE": "🔴",
    "CRUNCH": "⛔",
}


def load_execution_quality_state() -> Dict[str, Any]:
    """Load execution quality state from disk."""
    try:
        if _QUALITY_STATE_FILE.exists():
            return json.loads(_QUALITY_STATE_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        _LOG.warning("Failed to load execution_quality state: %s", e)
    return {"updated_ts": None, "symbols": {}, "meta": {}}


def load_recent_events(hours: int = 24, max_events: int = 100) -> List[Dict[str, Any]]:
    """Load recent execution events from JSONL log."""
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
        
        # Return most recent events
        return events[-max_events:]
        
    except Exception as e:
        _LOG.warning("Failed to load execution events: %s", e)
        return []


def render_execution_quality_panel() -> None:
    """Render the Minotaur Execution Quality panel."""
    if not HAS_STREAMLIT:
        return
    
    st.header("⚡ Execution Quality (Minotaur)")
    
    # Load state
    state = load_execution_quality_state()
    symbols = state.get("symbols", {})
    meta = state.get("meta", {})
    updated_ts = state.get("updated_ts")
    
    # Header metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        enabled = meta.get("enabled", False)
        status = "🟢 Enabled" if enabled else "⚪ Disabled"
        st.metric("Minotaur Status", status)
    
    with col2:
        throttling = meta.get("throttling_active", False)
        throttle_status = "⚠️ Active" if throttling else "✅ Normal"
        st.metric("Throttling", throttle_status)
    
    with col3:
        thin_symbols = meta.get("thin_liquidity_symbols", [])
        st.metric("Thin Liquidity Symbols", len(thin_symbols))
    
    with col4:
        crunch_symbols = meta.get("crunch_symbols", [])
        st.metric("Crunch Symbols", len(crunch_symbols))
    
    # Last update
    if updated_ts:
        st.caption(f"Last updated: {updated_ts}")
    
    st.divider()
    
    # Handle empty state
    if not symbols:
        st.info("No execution quality data available yet. Minotaur will populate this after processing fills.")
        return
    
    # Build DataFrame for symbol quality
    rows = []
    for symbol, stats in symbols.items():
        rows.append({
            "Symbol": symbol,
            "Regime": f"{REGIME_COLORS.get(stats.get('last_regime', 'NORMAL'), '⚪')} {stats.get('last_regime', 'NORMAL')}",
            "Avg Slip (bps)": round(stats.get("avg_slippage_bps", 0), 2),
            "P95 Slip (bps)": round(stats.get("p95_slippage_bps", 0), 2),
            "Max Slip (bps)": round(stats.get("max_slippage_bps", 0), 2),
            "Fill Ratio": f"{stats.get('fill_ratio', 1.0) * 100:.1f}%",
            "TWAP Usage": f"{stats.get('twap_usage_pct', 0) * 100:.1f}%",
            "Trades": stats.get("trade_count", 0),
        })
    
    if rows:
        df = pd.DataFrame(rows)
        
        # Sort by avg slippage (worst first)
        df = df.sort_values("Avg Slip (bps)", ascending=False)
        
        st.subheader("📊 Per-Symbol Quality Metrics")
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Summary stats
        st.subheader("📈 Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_slip = sum(s.get("avg_slippage_bps", 0) for s in symbols.values()) / len(symbols) if symbols else 0
            st.metric("Portfolio Avg Slippage", f"{avg_slip:.2f} bps")
        
        with col2:
            avg_fill = sum(s.get("fill_ratio", 1.0) for s in symbols.values()) / len(symbols) if symbols else 1.0
            st.metric("Portfolio Fill Ratio", f"{avg_fill * 100:.1f}%")
        
        with col3:
            avg_twap = sum(s.get("twap_usage_pct", 0) for s in symbols.values()) / len(symbols) if symbols else 0
            st.metric("Avg TWAP Usage", f"{avg_twap * 100:.1f}%")
    
    # Regime distribution
    st.subheader("🎯 Regime Distribution")
    
    regime_counts = {}
    for stats in symbols.values():
        regime = stats.get("last_regime", "NORMAL")
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
    
    if regime_counts:
        regime_df = pd.DataFrame([
            {"Regime": f"{REGIME_COLORS.get(r, '⚪')} {r}", "Count": c, "Pct": f"{c / len(symbols) * 100:.1f}%"}
            for r, c in sorted(regime_counts.items(), key=lambda x: -x[1])
        ])
        st.dataframe(regime_df, use_container_width=True, hide_index=True)
    
    # Recent events
    st.subheader("📋 Recent Execution Events (24h)")
    
    events = load_recent_events(hours=24, max_events=50)
    
    if events:
        # Filter to interesting events
        interesting_events = [
            e for e in events
            if e.get("event") in ("SLIPPAGE_SPIKE", "THROTTLE", "PLAN_TWAP", "PLAN_STEPPED")
        ]
        
        if interesting_events:
            event_rows = []
            for e in interesting_events[-20:]:  # Last 20
                event_rows.append({
                    "Time": e.get("ts", ""),
                    "Symbol": e.get("symbol", ""),
                    "Event": e.get("event", ""),
                    "Details": _format_event_details(e),
                })
            
            event_df = pd.DataFrame(event_rows)
            st.dataframe(event_df, use_container_width=True, hide_index=True)
        else:
            st.info("No significant execution events in the last 24 hours.")
    else:
        st.info("No execution events logged yet.")
    
    # Alerts section
    _render_alerts(meta, symbols)


def _format_event_details(event: Dict[str, Any]) -> str:
    """Format event details for display."""
    event_type = event.get("event", "")
    
    if event_type == "SLIPPAGE_SPIKE":
        return f"Slippage: {event.get('slippage_bps', 0):.1f} bps (threshold: {event.get('threshold_bps', 0):.1f})"
    elif event_type == "THROTTLE":
        return event.get("reason", "")
    elif event_type in ("PLAN_TWAP", "PLAN_STEPPED"):
        return f"{event.get('slice_count', 1)} slices over {event.get('schedule_seconds', 0)}s, regime={event.get('regime', '')}"
    else:
        return str(event)


def _render_alerts(meta: Dict[str, Any], symbols: Dict[str, Any]) -> None:
    """Render alert section for execution quality issues."""
    alerts = []
    
    # Check throttling
    if meta.get("throttling_active"):
        alerts.append(("⚠️", "Throttling is active — new orders are being rate-limited"))
    
    # Check crunch symbols
    crunch = meta.get("crunch_symbols", [])
    if len(crunch) >= 2:
        alerts.append(("⛔", f"Multiple symbols in CRUNCH regime: {', '.join(crunch)}"))
    
    # Check high slippage symbols
    high_slip = [
        s for s, stats in symbols.items()
        if stats.get("avg_slippage_bps", 0) > 15
    ]
    if high_slip:
        alerts.append(("🔴", f"High average slippage (>15 bps): {', '.join(high_slip)}"))
    
    # Check low fill ratio
    low_fill = [
        s for s, stats in symbols.items()
        if stats.get("fill_ratio", 1.0) < 0.9
    ]
    if low_fill:
        alerts.append(("🟠", f"Low fill ratio (<90%): {', '.join(low_fill)}"))
    
    if alerts:
        st.subheader("🚨 Alerts")
        for icon, message in alerts:
            st.warning(f"{icon} {message}")


def render_compact_execution_quality() -> None:
    """Render a compact version of execution quality for sidebar."""
    if not HAS_STREAMLIT:
        return
    
    state = load_execution_quality_state()
    meta = state.get("meta", {})
    symbols = state.get("symbols", {})
    
    st.sidebar.markdown("### ⚡ Execution Quality")
    
    enabled = meta.get("enabled", False)
    if not enabled:
        st.sidebar.caption("Minotaur: Disabled")
        return
    
    throttling = meta.get("throttling_active", False)
    thin_count = len(meta.get("thin_liquidity_symbols", []))
    crunch_count = len(meta.get("crunch_symbols", []))
    
    status_parts = []
    if throttling:
        status_parts.append("⚠️ Throttle")
    if crunch_count > 0:
        status_parts.append(f"⛔ {crunch_count} crunch")
    if thin_count > 0:
        status_parts.append(f"🟡 {thin_count} thin")
    
    if status_parts:
        st.sidebar.caption(" | ".join(status_parts))
    else:
        st.sidebar.caption("✅ Normal execution")
    
    if symbols:
        avg_slip = sum(s.get("avg_slippage_bps", 0) for s in symbols.values()) / len(symbols)
        st.sidebar.metric("Avg Slippage", f"{avg_slip:.1f} bps", label_visibility="collapsed")


# For testing without Streamlit
def get_panel_data() -> Dict[str, Any]:
    """Get panel data without rendering (for testing)."""
    state = load_execution_quality_state()
    events = load_recent_events(hours=24, max_events=50)
    
    return {
        "state": state,
        "events": events,
        "symbol_count": len(state.get("symbols", {})),
        "meta": state.get("meta", {}),
    }


if __name__ == "__main__":
    # Test data loading
    data = get_panel_data()
    print(f"Symbols: {data['symbol_count']}")
    print(f"Events: {len(data['events'])}")
    print(f"Meta: {data['meta']}")
