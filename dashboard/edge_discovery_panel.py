"""
v7.7_P5 — Edge Discovery Dashboard Panel.

Provides visual analysis of the EdgeScanner surface:
- Top/Weak factors with edge scores
- Top/Weak symbols with hybrid scores and conviction
- Top/Weak categories with momentum and IR
- Regime context (vol_regime, dd_state, risk_mode, router_quality)
- Edge Map visualization (ranked bar charts)

This panel is READ-ONLY — it displays data from edge_insights.json
and does NOT modify any state surfaces.

Note: Dashboard may read state surfaces, but may not write or modify them.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

try:
    import altair as alt
    ALTAIR_AVAILABLE = True
except ImportError:
    ALTAIR_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data Loaders (Read-Only)
# ---------------------------------------------------------------------------


def load_edge_insights(state: Optional[Any] = None) -> Dict[str, Any]:
    """
    Load edge insights from state object or file.

    Args:
        state: Optional state object with edge_insights attribute

    Returns:
        Edge insights dict or empty dict if unavailable
    """
    if state is not None and hasattr(state, "edge_insights"):
        return state.edge_insights or {}

    # Fall back to direct file load
    try:
        from dashboard.state_v7 import load_edge_insights_state
        return load_edge_insights_state() or {}
    except ImportError:
        pass

    return {}


def load_alpha_router_state() -> Dict[str, Any]:
    """
    Load alpha router state from file.
    
    Returns:
        Alpha router state dict or empty dict if unavailable
    """
    try:
        from pathlib import Path
        import json
        path = Path("logs/state/alpha_router_state.json")
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {}


def load_universe_optimizer_state() -> Dict[str, Any]:
    """
    Load universe optimizer state from file.
    
    Returns:
        Universe optimizer state dict or empty dict if unavailable
    """
    try:
        from pathlib import Path
        import json
        path = Path("logs/state/universe_optimizer.json")
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {}


def load_alpha_miner_state() -> Dict[str, Any]:
    """
    Load alpha miner state from file.
    
    v7.8_P4: Autonomous Alpha Miner (Prospector).
    
    Returns:
        Alpha miner state dict or empty dict if unavailable
    """
    try:
        from pathlib import Path
        import json
        path = Path("logs/state/alpha_miner.json")
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {}


def load_cross_pair_state() -> Dict[str, Any]:
    """
    Load cross-pair edges state from file.
    
    v7.8_P5: Cross-Pair Statistical Arbitrage Engine (Crossfire).
    
    Returns:
        Cross-pair state dict or empty dict if unavailable
    """
    try:
        from pathlib import Path
        import json
        path = Path("logs/state/cross_pair_edges.json")
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {}


def load_sentinel_x_state() -> Dict[str, Any]:
    """
    Load Sentinel-X state from file.
    
    v7.8_P6: Sentinel-X Hybrid ML Market Regime Classifier.
    
    Returns:
        Sentinel-X state dict or empty dict if unavailable
    """
    try:
        from pathlib import Path
        import json
        path = Path("logs/state/sentinel_x.json")
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# Regime Context Block
# ---------------------------------------------------------------------------


def render_regime_context(edge_insights: Dict[str, Any]) -> None:
    """
    Render the regime context as a KPI row.

    Shows: vol_regime, dd_state, risk_mode, router_quality
    """
    edge_summary = edge_insights.get("edge_summary", {})
    regime = edge_summary.get("regime", {})

    if not regime:
        st.info("Regime context unavailable.")
        return

    # Extract regime values with defaults
    vol_regime = regime.get("vol_regime", "unknown")
    dd_state = regime.get("dd_state", "unknown")
    risk_mode = regime.get("risk_mode", "normal")
    router_quality = regime.get("router_quality", 0.0)
    current_dd = regime.get("current_dd_pct", 0.0)

    # Color mapping for regimes
    vol_colors = {
        "low": "#21c354",
        "normal": "#7ed957",
        "high": "#f2c037",
        "crisis": "#d94a4a",
    }
    dd_colors = {
        "normal": "#21c354",
        "caution": "#f2c037",
        "drawdown": "#ff8c42",
        "critical": "#d94a4a",
    }
    risk_colors = {
        "normal": "#21c354",
        "caution": "#f2c037",
        "reduced": "#ff8c42",
        "halt": "#d94a4a",
    }

    def get_badge(value: str, color_map: Dict[str, str]) -> str:
        color = color_map.get(value.lower(), "#888")
        return f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:4px;font-weight:600;font-size:0.9em;">{value.upper()}</span>'

    def get_router_badge(quality: float) -> str:
        if quality >= 0.8:
            color = "#21c354"
        elif quality >= 0.6:
            color = "#7ed957"
        elif quality >= 0.4:
            color = "#f2c037"
        else:
            color = "#d94a4a"
        return f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:4px;font-weight:600;font-size:0.9em;">{quality:.0%}</span>'

    st.markdown("### 🎯 Regime Context")

    cols = st.columns(4)

    with cols[0]:
        st.markdown("**Vol Regime**")
        st.markdown(get_badge(vol_regime, vol_colors), unsafe_allow_html=True)

    with cols[1]:
        st.markdown("**DD State**")
        st.markdown(get_badge(dd_state, dd_colors), unsafe_allow_html=True)
        if current_dd > 0:
            st.caption(f"Current: {current_dd:.1%}")

    with cols[2]:
        st.markdown("**Risk Mode**")
        st.markdown(get_badge(risk_mode, risk_colors), unsafe_allow_html=True)

    with cols[3]:
        st.markdown("**Router Quality**")
        st.markdown(get_router_badge(router_quality), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Alpha Router Allocation (v7.8_P2)
# ---------------------------------------------------------------------------


def render_alpha_router_allocation() -> None:
    """
    Render alpha router allocation as a prominent metric.
    
    Shows the current portfolio-level allocation multiplier from the
    Alpha Router (v7.8_P2 "Overmind").
    """
    alpha_state = load_alpha_router_state()
    
    if not alpha_state:
        # Alpha router not enabled or no state yet - don't show section
        return
    
    target_allocation = alpha_state.get("target_allocation", 1.0)
    raw_components = alpha_state.get("raw_components", {})
    updated_ts = alpha_state.get("updated_ts", "")
    
    # Color based on allocation level
    if target_allocation >= 0.85:
        alloc_color = "#21c354"  # green - full gas
        alloc_label = "FULL"
    elif target_allocation >= 0.65:
        alloc_color = "#7ed957"  # light green - cautious
        alloc_label = "ACTIVE"
    elif target_allocation >= 0.45:
        alloc_color = "#f2c037"  # yellow - reduced
        alloc_label = "REDUCED"
    else:
        alloc_color = "#d94a4a"  # red - minimal
        alloc_label = "MINIMAL"
    
    st.markdown("### 🧠 Alpha Router Allocation")
    
    col1, col2, col3 = st.columns([1, 2, 2])
    
    with col1:
        # Big allocation metric
        allocation_html = f'''
        <div style="text-align:center;">
            <div style="font-size:2.5em;font-weight:700;color:{alloc_color};">
                {target_allocation:.0%}
            </div>
            <div style="font-size:0.9em;color:{alloc_color};font-weight:600;">
                {alloc_label}
            </div>
        </div>
        '''
        st.markdown(allocation_html, unsafe_allow_html=True)
    
    with col2:
        # Component breakdown
        st.markdown("**Components**")
        health_base = raw_components.get("health_base", 1.0)
        vol_penalty = raw_components.get("vol_penalty", 1.0)
        dd_penalty = raw_components.get("dd_penalty", 1.0)
        router_penalty = raw_components.get("router_penalty", 1.0)
        meta_adj = raw_components.get("meta_adjustment", 0.0)
        
        st.caption(f"Health Base: {health_base:.0%}")
        st.caption(f"Vol Penalty: {vol_penalty:.0%}")
        st.caption(f"DD Penalty: {dd_penalty:.0%}")
        st.caption(f"Router Penalty: {router_penalty:.0%}")
        if abs(meta_adj) > 0.001:
            sign = "+" if meta_adj > 0 else ""
            st.caption(f"Meta Adj: {sign}{meta_adj:.1%}")
    
    with col3:
        # Context info
        st.markdown("**Context**")
        health_score = raw_components.get("health_score", 0.0)
        vol_regime = raw_components.get("vol_regime", "unknown")
        dd_state = raw_components.get("dd_state", "unknown")
        router_quality = raw_components.get("router_quality", 0.0)
        
        st.caption(f"Health Score: {health_score:.0%}")
        st.caption(f"Vol Regime: {vol_regime}")
        st.caption(f"DD State: {dd_state}")
        st.caption(f"Router Quality: {router_quality:.0%}")
    
    if updated_ts:
        st.caption(f"_Last updated: {updated_ts}_")


def render_universe_optimizer() -> None:
    """
    Render universe optimizer status as a visual section.
    
    Shows the dynamically curated symbol universe from the
    Universe Optimizer (v7.8_P3 "Curator").
    """
    universe_state = load_universe_optimizer_state()
    
    if not universe_state:
        # Universe optimizer not enabled or no state yet - don't show section
        return
    
    allowed_symbols = universe_state.get("allowed_symbols", [])
    symbol_scores = universe_state.get("symbol_scores", {})
    category_scores = universe_state.get("category_scores", {})
    universe_stats = universe_state.get("universe_stats", {})
    updated_ts = universe_state.get("updated_ts", "")
    notes = universe_state.get("notes", [])
    
    current_size = universe_stats.get("current_size", len(allowed_symbols))
    max_size = universe_stats.get("max_size", 20)
    min_size = universe_stats.get("min_size", 4)
    regime_shrink = universe_stats.get("regime_shrink_applied", False)
    dd_shrink = universe_stats.get("dd_shrink_applied", False)
    
    # Color based on universe size vs capacity
    capacity_ratio = current_size / max_size if max_size > 0 else 0
    if capacity_ratio >= 0.75:
        size_color = "#21c354"  # green - expanded
        size_label = "EXPANDED"
    elif capacity_ratio >= 0.50:
        size_color = "#7ed957"  # light green - normal
        size_label = "NORMAL"
    elif capacity_ratio >= 0.30:
        size_color = "#f2c037"  # yellow - contracted
        size_label = "CONTRACTED"
    else:
        size_color = "#d94a4a"  # red - minimal
        size_label = "MINIMAL"
    
    st.markdown("### 🌐 Universe Optimizer")
    
    col1, col2, col3 = st.columns([1, 2, 2])
    
    with col1:
        # Big size metric
        size_html = f'''
        <div style="text-align:center;">
            <div style="font-size:2.5em;font-weight:700;color:{size_color};">
                {current_size}
            </div>
            <div style="font-size:0.9em;color:{size_color};font-weight:600;">
                {size_label}
            </div>
            <div style="font-size:0.8em;color:#888;">
                of {min_size}–{max_size}
            </div>
        </div>
        '''
        st.markdown(size_html, unsafe_allow_html=True)
    
    with col2:
        # Allowed symbols list
        st.markdown("**Active Universe**")
        if allowed_symbols:
            # Show symbols in a compact grid
            symbols_display = ", ".join(allowed_symbols[:8])
            if len(allowed_symbols) > 8:
                symbols_display += f" +{len(allowed_symbols) - 8} more"
            st.caption(symbols_display)
        else:
            st.caption("No symbols active")
        
        # Category diversity
        if category_scores:
            cats = sorted(category_scores.keys())
            st.caption(f"Categories: {', '.join(cats)}")
    
    with col3:
        # Shrinkage status
        st.markdown("**Status**")
        shrink_flags = []
        if regime_shrink:
            shrink_flags.append("⚡ Vol regime shrink")
        if dd_shrink:
            shrink_flags.append("📉 Drawdown shrink")
        if shrink_flags:
            for flag in shrink_flags:
                st.caption(flag)
        else:
            st.caption("✅ No shrinkage active")
        
        # Notes
        if notes:
            st.caption(f"Notes: {len(notes)}")
    
    # Symbol scores table (expandable)
    if symbol_scores:
        with st.expander("Symbol Scores", expanded=False):
            if PANDAS_AVAILABLE:
                import pandas as pd
                scores_data = [
                    {"Symbol": sym, "Score": f"{score:.2f}"}
                    for sym, score in sorted(symbol_scores.items(), key=lambda x: x[1], reverse=True)
                ]
                df = pd.DataFrame(scores_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                for sym, score in sorted(symbol_scores.items(), key=lambda x: x[1], reverse=True)[:10]:
                    st.caption(f"{sym}: {score:.2f}")
    
    if updated_ts:
        st.caption(f"_Last updated: {updated_ts}_")


def render_alpha_miner() -> None:
    """
    Render alpha miner (Prospector) status as a visual section.
    
    Shows candidate symbols discovered from exchange-wide scanning.
    v7.8_P4: Autonomous Alpha Miner (Prospector).
    """
    miner_state = load_alpha_miner_state()
    
    if not miner_state:
        # Alpha miner not enabled or no state yet - don't show section
        return
    
    candidates = miner_state.get("candidates", [])
    symbols_scanned = miner_state.get("symbols_scanned", 0)
    symbols_passed_filter = miner_state.get("symbols_passed_filter", 0)
    cycle_count = miner_state.get("cycle_count", 0)
    updated_ts = miner_state.get("updated_ts", 0)
    notes = miner_state.get("notes", "")
    
    num_candidates = len(candidates)
    
    # Color based on candidate count
    if num_candidates >= 10:
        count_color = "#21c354"  # green - many opportunities
        status_label = "ACTIVE"
    elif num_candidates >= 5:
        count_color = "#7ed957"  # light green - normal
        status_label = "NORMAL"
    elif num_candidates >= 1:
        count_color = "#f2c037"  # yellow - few
        status_label = "LIMITED"
    else:
        count_color = "#888"  # gray - none
        status_label = "IDLE"
    
    st.markdown("### ⛏️ Alpha Miner (Prospector)")
    st.caption("Autonomous discovery of new tradable symbols from exchange-wide scanning.")
    
    col1, col2, col3 = st.columns([1, 2, 2])
    
    with col1:
        # Big candidate count
        count_html = f'''
        <div style="text-align:center;">
            <div style="font-size:2.5em;font-weight:700;color:{count_color};">
                {num_candidates}
            </div>
            <div style="font-size:0.9em;color:{count_color};font-weight:600;">
                {status_label}
            </div>
            <div style="font-size:0.8em;color:#888;">
                candidates
            </div>
        </div>
        '''
        st.markdown(count_html, unsafe_allow_html=True)
    
    with col2:
        # Scan stats
        st.markdown("**Scan Stats**")
        st.caption(f"Scanned: {symbols_scanned}")
        st.caption(f"Passed filter: {symbols_passed_filter}")
        st.caption(f"Cycles: {cycle_count}")
    
    with col3:
        # Top candidate preview
        st.markdown("**Top Candidate**")
        if candidates:
            top = candidates[0]
            top_sym = top.get("symbol", "")
            top_score = top.get("ema_score", 0.0)
            top_reason = top.get("reason", "")[:50]
            st.caption(f"🏆 {top_sym} ({top_score:.2f})")
            st.caption(f"_{top_reason}..._" if len(top.get("reason", "")) > 50 else f"_{top_reason}_")
        else:
            st.caption("No candidates yet")
    
    # Candidates table (expandable)
    if candidates:
        with st.expander(f"All Candidates ({len(candidates)})", expanded=False):
            if PANDAS_AVAILABLE:
                import pandas as pd
                candidates_data = []
                for c in candidates[:20]:  # Limit to 20
                    feat = c.get("features", {})
                    candidates_data.append({
                        "Symbol": c.get("symbol", ""),
                        "Score": f"{c.get('ema_score', 0):.3f}",
                        "Momo (30d)": f"{feat.get('long_momo', 0)*100:.1f}%",
                        "Trend": f"{feat.get('trend_consistency', 0)*100:.0f}%",
                        "Category": feat.get("category_hint", "OTHER"),
                        "Reason": c.get("reason", "")[:40],
                    })
                df = pd.DataFrame(candidates_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                for c in candidates[:10]:
                    sym = c.get("symbol", "")
                    score = c.get("ema_score", 0)
                    reason = c.get("reason", "")
                    st.caption(f"{sym}: {score:.3f} — {reason}")
    
    # Category breakdown
    if candidates:
        category_counts: Dict[str, int] = {}
        for c in candidates:
            feat = c.get("features", {})
            cat = feat.get("category_hint", "OTHER")
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        if category_counts:
            with st.expander("Category Breakdown", expanded=False):
                for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
                    st.caption(f"{cat}: {count}")
    
    if notes:
        st.caption(f"_Notes: {notes}_")
    
    if updated_ts:
        from datetime import datetime
        try:
            dt = datetime.fromtimestamp(updated_ts)
            st.caption(f"_Last scan: {dt.strftime('%Y-%m-%d %H:%M:%S')}_")
        except Exception:
            pass


def render_cross_pair_panel() -> None:
    """
    Render cross-pair statistical arbitrage (Crossfire) panel.
    
    Shows pair edges, spread z-scores, signals, and correlation metrics.
    v7.8_P5: Cross-Pair Statistical Arbitrage Engine.
    """
    cross_pair_state = load_cross_pair_state()
    
    if not cross_pair_state:
        # Cross-pair engine not enabled or no state yet - don't show section
        return
    
    pair_edges = cross_pair_state.get("pair_edges", {})
    pairs_analyzed = cross_pair_state.get("pairs_analyzed", 0)
    pairs_eligible = cross_pair_state.get("pairs_eligible", 0)
    cycle_count = cross_pair_state.get("cycle_count", 0)
    updated_ts = cross_pair_state.get("updated_ts", 0)
    notes = cross_pair_state.get("notes", "")
    
    # Count signals
    enter_count = sum(1 for e in pair_edges.values() if e.get("signal") == "ENTER")
    exit_count = sum(1 for e in pair_edges.values() if e.get("signal") == "EXIT")
    
    # Color based on ENTER signal count
    if enter_count >= 2:
        status_color = "#21c354"  # green - multiple setups
        status_label = "HOT"
    elif enter_count >= 1:
        status_color = "#7ed957"  # light green - one setup
        status_label = "ACTIVE"
    elif pairs_eligible > 0:
        status_color = "#f2c037"  # yellow - pairs ready but no signal
        status_label = "READY"
    else:
        status_color = "#888"  # gray - no eligible pairs
        status_label = "COLD"
    
    st.markdown("### ⚔️ Cross-Pair Engine (Crossfire)")
    st.caption("Statistical arbitrage between correlated pairs — spread z-scores and mean-reversion signals.")
    
    col1, col2, col3 = st.columns([1, 2, 2])
    
    with col1:
        # Signal count display
        count_html = f'''
        <div style="text-align:center;">
            <div style="font-size:2.5em;font-weight:700;color:{status_color};">
                {enter_count}
            </div>
            <div style="font-size:0.9em;color:{status_color};font-weight:600;">
                {status_label}
            </div>
            <div style="font-size:0.8em;color:#888;">
                ENTER signals
            </div>
        </div>
        '''
        st.markdown(count_html, unsafe_allow_html=True)
    
    with col2:
        # Pair stats
        st.markdown("**Pair Stats**")
        st.caption(f"Analyzed: {pairs_analyzed}")
        st.caption(f"Eligible: {pairs_eligible}")
        st.caption(f"EXIT signals: {exit_count}")
    
    with col3:
        # Top signal preview
        st.markdown("**Top Setup**")
        if pair_edges:
            # Find best ENTER signal by edge score
            enter_edges = [
                (k, v) for k, v in pair_edges.items() 
                if v.get("signal") == "ENTER"
            ]
            if enter_edges:
                enter_edges.sort(key=lambda x: x[1].get("ema_score", 0), reverse=True)
                top_key, top_edge = enter_edges[0]
                top_score = top_edge.get("ema_score", 0)
                long_leg = top_edge.get("long_leg", "")
                short_leg = top_edge.get("short_leg", "")
                spread_z = top_edge.get("stats", {}).get("spread_z", 0)
                st.caption(f"🎯 {top_key}")
                st.caption(f"Score: {top_score:.3f} | z: {spread_z:.2f}")
                st.caption(f"Long: {long_leg} | Short: {short_leg}")
            else:
                st.caption("No active ENTER signals")
        else:
            st.caption("No pairs analyzed")
    
    # Pair edges table (expandable)
    if pair_edges:
        with st.expander(f"All Pairs ({len(pair_edges)})", expanded=False):
            if PANDAS_AVAILABLE:
                import pandas as pd
                pairs_data = []
                for key, edge in sorted(
                    pair_edges.items(),
                    key=lambda x: x[1].get("ema_score", 0),
                    reverse=True
                ):
                    stats = edge.get("stats", {})
                    signal = edge.get("signal", "NONE")
                    signal_emoji = "🎯" if signal == "ENTER" else ("🔄" if signal == "EXIT" else "—")
                    pairs_data.append({
                        "Pair": key,
                        "Score": f"{edge.get('ema_score', 0):.3f}",
                        "Z-Score": f"{stats.get('spread_z', 0):.2f}",
                        "Corr": f"{stats.get('corr', 0):.2f}",
                        "Half-Life": f"{stats.get('half_life_est', 0):.0f}",
                        "Signal": f"{signal_emoji} {signal}",
                        "Long": edge.get("long_leg", "—"),
                        "Short": edge.get("short_leg", "—"),
                    })
                df = pd.DataFrame(pairs_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                for key, edge in pair_edges.items():
                    signal = edge.get("signal", "NONE")
                    score = edge.get("ema_score", 0)
                    st.caption(f"{key}: {score:.3f} — {signal}")
    
    # Edge score bar chart
    if pair_edges and ALTAIR_AVAILABLE:
        with st.expander("Edge Score Distribution", expanded=False):
            import pandas as pd
            import altair as alt
            
            chart_data = [
                {"Pair": k, "Edge Score": v.get("ema_score", 0), "Signal": v.get("signal", "NONE")}
                for k, v in pair_edges.items()
            ]
            df = pd.DataFrame(chart_data).sort_values("Edge Score", ascending=False)
            
            color_scale = alt.Scale(
                domain=["ENTER", "EXIT", "NONE"],
                range=["#21c354", "#f2c037", "#888888"]
            )
            
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X("Pair:N", sort="-y"),
                y=alt.Y("Edge Score:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Signal:N", scale=color_scale),
                tooltip=["Pair", "Edge Score", "Signal"]
            ).properties(height=200)
            
            st.altair_chart(chart, use_container_width=True)
    
    if notes:
        st.caption(f"_Notes: {notes}_")
    
    if updated_ts:
        from datetime import datetime
        try:
            dt = datetime.fromtimestamp(updated_ts)
            st.caption(f"_Last scan: {dt.strftime('%Y-%m-%d %H:%M:%S')}_")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# v7.8_P6: Sentinel-X Panel
# ---------------------------------------------------------------------------


def render_sentinel_x_panel() -> None:
    """
    Render Sentinel-X Hybrid ML Market Regime Classifier panel.
    
    Shows regime classification, probabilities, and feature diagnostics.
    v7.8_P6: Sentinel-X integration.
    """
    sentinel_state = load_sentinel_x_state()
    
    if not sentinel_state:
        # Sentinel-X not enabled or no state yet - don't show section
        return
    
    primary_regime = sentinel_state.get("primary_regime", "CHOPPY")
    secondary_regime = sentinel_state.get("secondary_regime")
    smoothed_probs = sentinel_state.get("smoothed_probs", {})
    features = sentinel_state.get("features", {})
    crisis_flag = sentinel_state.get("crisis_flag", False)
    crisis_reason = sentinel_state.get("crisis_reason", "")
    updated_ts = sentinel_state.get("updated_ts", "")
    cycle_count = sentinel_state.get("cycle_count", 0)
    
    # Regime colors
    regime_colors = {
        "TREND_UP": "#21c354",       # Green
        "TREND_DOWN": "#f44",        # Red
        "MEAN_REVERT": "#9370db",    # Purple
        "BREAKOUT": "#f2c037",       # Yellow/Gold
        "CHOPPY": "#888",            # Gray
        "CRISIS": "#ff1744",         # Bright red
    }
    
    # Regime icons
    regime_icons = {
        "TREND_UP": "📈",
        "TREND_DOWN": "📉",
        "MEAN_REVERT": "🔄",
        "BREAKOUT": "💥",
        "CHOPPY": "〰️",
        "CRISIS": "🚨",
    }
    
    primary_color = regime_colors.get(primary_regime, "#888")
    primary_icon = regime_icons.get(primary_regime, "❓")
    
    st.markdown("### 🧠 Sentinel-X (Regime Classifier)")
    st.caption("Hybrid ML market regime detection — probabilities and feature diagnostics.")
    
    col1, col2, col3 = st.columns([1, 2, 2])
    
    with col1:
        # Primary regime display
        crisis_indicator = "🚨 " if crisis_flag else ""
        regime_html = f'''
        <div style="text-align:center;">
            <div style="font-size:2.5em;font-weight:700;color:{primary_color};">
                {crisis_indicator}{primary_icon}
            </div>
            <div style="font-size:1.2em;color:{primary_color};font-weight:600;">
                {primary_regime}
            </div>
            <div style="font-size:0.8em;color:#888;">
                Primary Regime
            </div>
        </div>
        '''
        st.markdown(regime_html, unsafe_allow_html=True)
        
        if secondary_regime:
            sec_color = regime_colors.get(secondary_regime, "#888")
            st.caption(f"Secondary: **{secondary_regime}**")
    
    with col2:
        # Regime probabilities
        st.markdown("**Regime Probabilities**")
        if smoothed_probs:
            for regime in ["TREND_UP", "TREND_DOWN", "MEAN_REVERT", "BREAKOUT", "CHOPPY", "CRISIS"]:
                prob = smoothed_probs.get(regime, 0)
                color = regime_colors.get(regime, "#888")
                icon = regime_icons.get(regime, "")
                bar_width = int(prob * 100)
                st.markdown(
                    f'<div style="margin:2px 0;">'
                    f'<span style="width:100px;display:inline-block;">{icon} {regime}</span>'
                    f'<span style="display:inline-block;width:{bar_width}%;background:{color};height:12px;border-radius:3px;"></span>'
                    f' {prob:.1%}'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.caption("No probabilities available")
    
    with col3:
        # Key features
        st.markdown("**Key Features**")
        if features:
            trend_slope = features.get("trend_slope", 0)
            trend_r2 = features.get("trend_r2", 0)
            vol_z = features.get("vol_regime_z", 0)
            returns_std = features.get("returns_std", 0)
            mean_rev = features.get("mean_reversion_score", 0)
            
            st.caption(f"Trend Slope: {trend_slope:.4f}")
            st.caption(f"Trend R²: {trend_r2:.3f}")
            st.caption(f"Vol Z-Score: {vol_z:.2f}")
            st.caption(f"Returns Std: {returns_std:.4f}")
            st.caption(f"Mean Revert: {mean_rev:.3f}")
        else:
            st.caption("No features available")
    
    # Crisis alert
    if crisis_flag:
        st.error(f"🚨 **CRISIS OVERRIDE ACTIVE**: {crisis_reason}")
    
    # Feature details (expandable)
    if features:
        with st.expander("All Features", expanded=False):
            if PANDAS_AVAILABLE:
                import pandas as pd
                features_data = [
                    {"Feature": k, "Value": f"{v:.4f}" if isinstance(v, float) else str(v)}
                    for k, v in sorted(features.items())
                ]
                df = pd.DataFrame(features_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                for k, v in sorted(features.items()):
                    st.caption(f"{k}: {v}")
    
    # Probability bar chart
    if smoothed_probs and ALTAIR_AVAILABLE:
        with st.expander("Probability Distribution", expanded=False):
            import pandas as pd
            import altair as alt
            
            chart_data = [
                {"Regime": k, "Probability": v}
                for k, v in smoothed_probs.items()
            ]
            df = pd.DataFrame(chart_data)
            
            # Create color list in order
            domain = list(smoothed_probs.keys())
            range_colors = [regime_colors.get(r, "#888") for r in domain]
            
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X("Regime:N", sort=domain),
                y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Regime:N", scale=alt.Scale(domain=domain, range=range_colors), legend=None),
                tooltip=["Regime", "Probability"]
            ).properties(height=200)
            
            st.altair_chart(chart, use_container_width=True)
    
    st.caption(f"_Cycle: {cycle_count} | Updated: {updated_ts}_")


# ---------------------------------------------------------------------------
# v7.8_P7: Alpha Decay Panel
# ---------------------------------------------------------------------------


def load_alpha_decay_state() -> Optional[Dict[str, Any]]:
    """Load alpha decay state from state file."""
    path = Path("logs/state/alpha_decay.json")
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, IOError):
        return None


def render_alpha_decay_panel() -> None:
    """
    Render Alpha Decay & Survival Curves panel.
    
    Shows decay rates, half-lives, and survival probabilities
    for symbols, categories, and factors.
    
    v7.8_P7: Thanatos — The Mortality Model.
    """
    decay_state = load_alpha_decay_state()
    
    if not decay_state:
        # Alpha decay not enabled or no state yet
        return
    
    updated_ts = decay_state.get("updated_ts", "")
    cycle_count = decay_state.get("cycle_count", 0)
    overall_health = decay_state.get("overall_alpha_health", 0.5)
    avg_symbol_survival = decay_state.get("avg_symbol_survival", 0.5)
    avg_category_survival = decay_state.get("avg_category_survival", 0.5)
    avg_factor_survival = decay_state.get("avg_factor_survival", 0.5)
    
    weakest_symbols = decay_state.get("weakest_symbols", [])
    strongest_symbols = decay_state.get("strongest_symbols", [])
    weakest_categories = decay_state.get("weakest_categories", [])
    weakest_factors = decay_state.get("weakest_factors", [])
    
    symbols = decay_state.get("symbols", {})
    categories = decay_state.get("categories", {})
    factors = decay_state.get("factors", {})
    
    # Health color
    if overall_health >= 0.7:
        health_color = "#21c354"  # Green
        health_icon = "💚"
    elif overall_health >= 0.5:
        health_color = "#f2c037"  # Yellow
        health_icon = "💛"
    else:
        health_color = "#ff4444"  # Red
        health_icon = "💔"
    
    st.markdown("### ⏳ Alpha Decay & Survival Curves")
    st.caption("Mortality model — tracking alpha half-lives and decay rates.")
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f'<div style="text-align:center;">'
            f'<div style="font-size:2em;color:{health_color};">{health_icon}</div>'
            f'<div style="font-size:1.5em;font-weight:700;color:{health_color};">{overall_health:.1%}</div>'
            f'<div style="font-size:0.8em;color:#888;">Overall Alpha Health</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    with col2:
        st.metric("Symbol Survival", f"{avg_symbol_survival:.1%}")
    
    with col3:
        st.metric("Category Survival", f"{avg_category_survival:.1%}")
    
    with col4:
        st.metric("Factor Survival", f"{avg_factor_survival:.1%}")
    
    # Summary lists
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("**⚠️ Weakest (Decaying)**")
        if weakest_symbols:
            for sym in weakest_symbols[:5]:
                sym_data = symbols.get(sym, {})
                half_life = sym_data.get("half_life", 0)
                survival = sym_data.get("survival_prob", 0)
                trend = sym_data.get("trend_direction", "stable")
                trend_icon = {"improving": "📈", "declining": "📉", "stable": "➡️"}.get(trend, "➡️")
                st.caption(f"{trend_icon} **{sym}**: HL={half_life:.0f}d, Surv={survival:.0%}")
        if weakest_categories:
            st.caption(f"Categories: {', '.join(weakest_categories[:3])}")
        if weakest_factors:
            st.caption(f"Factors: {', '.join(weakest_factors[:3])}")
    
    with col_right:
        st.markdown("**✅ Strongest (Stable Alpha)**")
        if strongest_symbols:
            for sym in strongest_symbols[:5]:
                sym_data = symbols.get(sym, {})
                half_life = sym_data.get("half_life", 0)
                survival = sym_data.get("survival_prob", 0)
                trend = sym_data.get("trend_direction", "stable")
                trend_icon = {"improving": "📈", "declining": "📉", "stable": "➡️"}.get(trend, "➡️")
                st.caption(f"{trend_icon} **{sym}**: HL={half_life:.0f}d, Surv={survival:.0%}")
    
    # Detailed tables (expandable)
    if symbols:
        with st.expander("Symbol Decay Details", expanded=False):
            if PANDAS_AVAILABLE:
                import pandas as pd
                sym_data = []
                for sym, data in sorted(symbols.items(), key=lambda x: x[1].get("survival_prob", 0)):
                    sym_data.append({
                        "Symbol": sym,
                        "Decay Rate": f"{data.get('decay_rate', 0):.4f}",
                        "Half-Life (d)": f"{data.get('half_life', 0):.1f}",
                        "Survival": f"{data.get('survival_prob', 0):.1%}",
                        "Deterioration": f"{data.get('deterioration_prob', 0):.1%}",
                        "Trend": data.get("trend_direction", "stable"),
                        "Days Since Peak": data.get("days_since_peak", 0),
                    })
                df = pd.DataFrame(sym_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                for sym, data in symbols.items():
                    st.caption(
                        f"{sym}: decay={data.get('decay_rate', 0):.4f}, "
                        f"HL={data.get('half_life', 0):.1f}d, "
                        f"surv={data.get('survival_prob', 0):.1%}"
                    )
    
    if factors:
        with st.expander("Factor Decay Details", expanded=False):
            if PANDAS_AVAILABLE:
                import pandas as pd
                factor_data = []
                for factor, data in sorted(factors.items()):
                    factor_data.append({
                        "Factor": factor,
                        "Decay Rate": f"{data.get('decay_rate', 0):.4f}",
                        "Survival": f"{data.get('survival_prob', 0):.1%}",
                        "Weight Mult": f"{data.get('adjusted_factor_weight_multiplier', 1):.3f}",
                        "Trend": data.get("trend_direction", "stable"),
                        "IR": f"{data.get('ir_rolling', 0):.3f}",
                    })
                df = pd.DataFrame(factor_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                for factor, data in factors.items():
                    st.caption(
                        f"{factor}: surv={data.get('survival_prob', 0):.1%}, "
                        f"mult={data.get('adjusted_factor_weight_multiplier', 1):.3f}"
                    )
    
    if categories:
        with st.expander("Category Decay Details", expanded=False):
            if PANDAS_AVAILABLE:
                import pandas as pd
                cat_data = []
                for cat, data in sorted(categories.items(), key=lambda x: x[1].get("survival_prob", 0)):
                    cat_data.append({
                        "Category": cat,
                        "Decay Rate": f"{data.get('decay_rate', 0):.4f}",
                        "Half-Life (d)": f"{data.get('half_life', 0):.1f}",
                        "Survival": f"{data.get('survival_prob', 0):.1%}",
                        "Symbols": data.get("symbol_count", 0),
                        "Weakest": data.get("weakest_symbol", ""),
                        "Strongest": data.get("strongest_symbol", ""),
                    })
                df = pd.DataFrame(cat_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                for cat, data in categories.items():
                    st.caption(
                        f"{cat}: surv={data.get('survival_prob', 0):.1%}, "
                        f"HL={data.get('half_life', 0):.1f}d"
                    )
    
    st.caption(f"_Cycle: {cycle_count} | Updated: {updated_ts}_")


# ---------------------------------------------------------------------------
# Factor Edge Tables
# ---------------------------------------------------------------------------


def render_factor_edges(edge_insights: Dict[str, Any]) -> None:
    """
    Render top/weak factor tables side by side.
    """
    edge_summary = edge_insights.get("edge_summary", {})
    top_factors = edge_summary.get("top_factors", [])
    weak_factors = edge_summary.get("weak_factors", [])

    st.markdown("### 📊 Factor Edges")

    if not top_factors and not weak_factors:
        st.info("No factor edge data available.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🔥 Top Factors**")
        if top_factors:
            _render_factor_table(top_factors, highlight_positive=True)
        else:
            st.caption("No top factors.")

    with col2:
        st.markdown("**❄️ Weak Factors**")
        if weak_factors:
            _render_factor_table(weak_factors, highlight_positive=False)
        else:
            st.caption("No weak factors.")


def _render_factor_table(factors: List[Dict[str, Any]], highlight_positive: bool = True) -> None:
    """Render a factor table with styling."""
    if not PANDAS_AVAILABLE:
        # Fallback without pandas
        for f in factors:
            factor = f.get("factor", "?")
            ir = f.get("ir", 0)
            pnl = f.get("pnl_contrib", 0)
            weight = f.get("weight", 0)
            edge = f.get("edge_score", 0)
            st.text(f"{factor}: IR={ir:.3f} PnL={pnl:.4f} W={weight:.2f} Edge={edge:.2f}")
        return

    rows = []
    for f in factors:
        rows.append({
            "Factor": f.get("factor", "?"),
            "IR": round(f.get("ir", 0), 3),
            "PnL Contrib": round(f.get("pnl_contrib", 0), 5),
            "Weight": round(f.get("weight", 0), 3),
            "Edge Score": round(f.get("edge_score", 0), 3),
        })

    df = pd.DataFrame(rows)

    # Style edge score column
    def color_edge(val):
        if not isinstance(val, (int, float)):
            return ""
        if highlight_positive:
            if val > 0.5:
                return "background-color: #21c354; color: white"
            elif val > 0:
                return "background-color: #7ed957; color: black"
        else:
            if val < -0.5:
                return "background-color: #d94a4a; color: white"
            elif val < 0:
                return "background-color: #ff8c42; color: black"
        return ""

    styled = df.style.map(color_edge, subset=["Edge Score"])
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Symbol Edge Tables
# ---------------------------------------------------------------------------


def render_symbol_edges(edge_insights: Dict[str, Any]) -> None:
    """
    Render top/weak symbol tables side by side.
    """
    edge_summary = edge_insights.get("edge_summary", {})
    top_symbols = edge_summary.get("top_symbols", [])
    weak_symbols = edge_summary.get("weak_symbols", [])

    st.markdown("### 💎 Symbol Edges")

    if not top_symbols and not weak_symbols:
        st.info("No symbol edge data available.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🔥 Top Symbols**")
        if top_symbols:
            _render_symbol_table(top_symbols, highlight_positive=True)
        else:
            st.caption("No top symbols.")

    with col2:
        st.markdown("**❄️ Weak Symbols**")
        if weak_symbols:
            _render_symbol_table(weak_symbols, highlight_positive=False)
        else:
            st.caption("No weak symbols.")


def _render_symbol_table(symbols: List[Dict[str, Any]], highlight_positive: bool = True) -> None:
    """Render a symbol table with styling."""
    if not PANDAS_AVAILABLE:
        for s in symbols:
            sym = s.get("symbol", "?")
            hybrid = s.get("hybrid_score", 0)
            conv = s.get("conviction", 0)
            pnl = s.get("recent_pnl", 0)
            edge = s.get("edge_score", 0)
            st.text(f"{sym}: H={hybrid:.2f} C={conv:.2f} PnL={pnl:.3f} Edge={edge:.2f}")
        return

    rows = []
    for s in symbols:
        rows.append({
            "Symbol": s.get("symbol", "?"),
            "Hybrid": round(s.get("hybrid_score", 0), 3),
            "Conviction": round(s.get("conviction", 0), 3),
            "Recent PnL": round(s.get("recent_pnl", 0), 4),
            "Direction": s.get("direction", "?"),
            "Edge Score": round(s.get("edge_score", 0), 3),
        })

    df = pd.DataFrame(rows)

    def color_edge(val):
        if not isinstance(val, (int, float)):
            return ""
        if highlight_positive:
            if val > 0.5:
                return "background-color: #21c354; color: white"
            elif val > 0:
                return "background-color: #7ed957; color: black"
        else:
            if val < -0.5:
                return "background-color: #d94a4a; color: white"
            elif val < 0:
                return "background-color: #ff8c42; color: black"
        return ""

    styled = df.style.map(color_edge, subset=["Edge Score"])
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Category Edge Tables
# ---------------------------------------------------------------------------


def render_category_edges(edge_insights: Dict[str, Any]) -> None:
    """
    Render top/weak category tables side by side.
    """
    edge_summary = edge_insights.get("edge_summary", {})
    top_categories = edge_summary.get("top_categories", [])
    weak_categories = edge_summary.get("weak_categories", [])

    st.markdown("### 🏷️ Category Edges")

    if not top_categories and not weak_categories:
        st.info("No category edge data available.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🔥 Top Categories**")
        if top_categories:
            _render_category_table(top_categories, highlight_positive=True)
        else:
            st.caption("No top categories.")

    with col2:
        st.markdown("**❄️ Weak Categories**")
        if weak_categories:
            _render_category_table(weak_categories, highlight_positive=False)
        else:
            st.caption("No weak categories.")


def _render_category_table(categories: List[Dict[str, Any]], highlight_positive: bool = True) -> None:
    """Render a category table with styling."""
    if not PANDAS_AVAILABLE:
        for c in categories:
            cat = c.get("category", "?")
            ir = c.get("ir", 0)
            mom = c.get("momentum", 0)
            pnl = c.get("total_pnl", 0)
            edge = c.get("edge_score", 0)
            st.text(f"{cat}: IR={ir:.2f} Mom={mom:.2f} PnL={pnl:.4f} Edge={edge:.2f}")
        return

    rows = []
    for c in categories:
        rows.append({
            "Category": c.get("category", "?"),
            "IR": round(c.get("ir", 0), 3),
            "Momentum": round(c.get("momentum", 0), 3),
            "Total PnL": round(c.get("total_pnl", 0), 5),
            "Edge Score": round(c.get("edge_score", 0), 3),
        })

    df = pd.DataFrame(rows)

    def color_edge(val):
        if not isinstance(val, (int, float)):
            return ""
        if highlight_positive:
            if val > 0.5:
                return "background-color: #21c354; color: white"
            elif val > 0:
                return "background-color: #7ed957; color: black"
        else:
            if val < -0.5:
                return "background-color: #d94a4a; color: white"
            elif val < 0:
                return "background-color: #ff8c42; color: black"
        return ""

    styled = df.style.map(color_edge, subset=["Edge Score"])
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Edge Map Visualization
# ---------------------------------------------------------------------------


def render_edge_map(edge_insights: Dict[str, Any]) -> None:
    """
    Render an edge map visualization showing ranked edge scores.

    Uses Altair bar charts to show factor, symbol, and category edge rankings.
    """
    st.markdown("### 🗺️ Edge Map")

    factor_edges = edge_insights.get("factor_edges", {})
    symbol_edges = edge_insights.get("symbol_edges", {})
    category_edges = edge_insights.get("category_edges", {})

    if not factor_edges and not symbol_edges and not category_edges:
        st.info("No edge map data available.")
        return

    if not ALTAIR_AVAILABLE or not PANDAS_AVAILABLE:
        st.warning("Altair or Pandas not available for edge map visualization.")
        return

    tabs = st.tabs(["Factors", "Symbols", "Categories"])

    with tabs[0]:
        _render_edge_bar_chart(factor_edges, "Factor", "factor")

    with tabs[1]:
        _render_edge_bar_chart(symbol_edges, "Symbol", "symbol")

    with tabs[2]:
        _render_edge_bar_chart(category_edges, "Category", "category")


def _render_edge_bar_chart(
    edges: Dict[str, Any],
    label: str,
    key_field: str,
) -> None:
    """Render a horizontal bar chart of edge scores."""
    if not edges:
        st.caption(f"No {label.lower()} edge data.")
        return

    # Build data for chart
    data = []
    for name, metrics in edges.items():
        if not isinstance(metrics, dict):
            continue
        edge_score = metrics.get("edge_score", 0)
        data.append({
            label: name,
            "Edge Score": edge_score,
            "Color": "positive" if edge_score >= 0 else "negative",
        })

    if not data:
        st.caption(f"No {label.lower()} data to display.")
        return

    df = pd.DataFrame(data)
    df = df.sort_values("Edge Score", ascending=True)

    # Create horizontal bar chart
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("Edge Score:Q", title="Edge Score"),
        y=alt.Y(f"{label}:N", sort="-x", title=label),
        color=alt.condition(
            alt.datum["Edge Score"] >= 0,
            alt.value("#21c354"),  # positive = green
            alt.value("#d94a4a"),  # negative = red
        ),
        tooltip=[label, "Edge Score"],
    ).properties(
        height=max(200, len(data) * 25),
    )

    st.altair_chart(chart, use_container_width=True)


# ---------------------------------------------------------------------------
# Regime Modifiers (v7.7_P6)
# ---------------------------------------------------------------------------


def render_regime_modifiers(edge_insights: Dict[str, Any]) -> None:
    """
    Render regime modifiers block showing conviction and factor weight adjustments.
    
    v7.7_P6: Shows how regime curves are affecting conviction scoring
    and factor weights.
    """
    regime_adjustments = edge_insights.get("regime_adjustments", {})
    
    if not regime_adjustments:
        return
    
    with st.expander("🎚️ Regime Modifiers (v7.7)", expanded=False):
        st.caption("How regime curves are adjusting conviction and factor weights")
        
        col1, col2 = st.columns(2)
        
        # Conviction modifiers
        conviction = regime_adjustments.get("conviction", {})
        with col1:
            st.markdown("**Conviction Adjustments**")
            if conviction:
                vol_mult = conviction.get("vol_multiplier", 1.0)
                dd_mult = conviction.get("dd_multiplier", 1.0)
                combined = conviction.get("combined_multiplier", 1.0)
                vol_regime = conviction.get("vol_regime", "NORMAL")
                dd_state = conviction.get("dd_state", "NORMAL")
                
                st.metric(
                    "Vol Regime Effect",
                    f"{vol_mult:.2f}x",
                    delta=f"{vol_regime}",
                    delta_color="off" if vol_mult >= 1.0 else "inverse",
                )
                st.metric(
                    "DD State Effect",
                    f"{dd_mult:.2f}x",
                    delta=f"{dd_state}",
                    delta_color="off" if dd_mult >= 1.0 else "inverse",
                )
                st.caption(f"Combined: **{combined:.2f}x**")
            else:
                st.info("Conviction regime data unavailable")
        
        # Factor weight modifiers
        factor_weights = regime_adjustments.get("factor_weights", {})
        with col2:
            st.markdown("**Factor Weight Adjustments**")
            if factor_weights:
                vol_mult = factor_weights.get("vol_multiplier", 1.0)
                dd_mult = factor_weights.get("dd_multiplier", 1.0)
                combined = factor_weights.get("combined_multiplier", 1.0)
                vol_regime = factor_weights.get("vol_regime", "NORMAL")
                dd_state = factor_weights.get("dd_state", "NORMAL")
                
                st.metric(
                    "Vol Regime Effect",
                    f"{vol_mult:.2f}x",
                    delta=f"{vol_regime}",
                    delta_color="off" if vol_mult >= 1.0 else "inverse",
                )
                st.metric(
                    "DD State Effect",
                    f"{dd_mult:.2f}x",
                    delta=f"{dd_state}",
                    delta_color="off" if dd_mult >= 1.0 else "inverse",
                )
                st.caption(f"Combined: **{combined:.2f}x**")
            else:
                st.info("Factor weight regime data unavailable")


# ---------------------------------------------------------------------------
# v7.7_P7: Strategy Health Score Section
# ---------------------------------------------------------------------------


def render_strategy_health(edge_insights: Dict[str, Any]) -> None:
    """
    Render the Strategy Health Score panel (v7.7_P7).

    Displays:
    - Headline health score (prominent gauge)
    - Component health breakdowns (factor, symbol, category, regime, execution)
    - System notes and alerts
    """
    strategy_health = edge_insights.get("strategy_health")

    if not strategy_health:
        # Graceful fallback if strategy_health not available
        return

    st.markdown("### 🏥 Strategy Health Score")

    health_score = strategy_health.get("health_score", 0.0)
    notes = strategy_health.get("notes", [])

    # Color based on health score
    if health_score >= 0.75:
        color = "#21c354"  # Green
        status = "HEALTHY"
    elif health_score >= 0.50:
        color = "#f2c037"  # Yellow
        status = "FAIR"
    else:
        color = "#d94a4a"  # Red
        status = "STRESSED"

    # Headline health score with big number
    col_main, col_status = st.columns([3, 1])

    with col_main:
        st.markdown(
            f'<div style="text-align:center;padding:20px;">'
            f'<span style="font-size:64px;font-weight:bold;color:{color};">{health_score:.0%}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_status:
        st.markdown(
            f'<div style="text-align:center;padding:30px 10px;">'
            f'<span style="background:{color};color:#fff;padding:8px 16px;border-radius:8px;'
            f'font-weight:600;font-size:1.2em;">{status}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Component breakdown
    st.markdown("#### Component Health")

    factor_health = strategy_health.get("factor_health", {})
    symbol_health = strategy_health.get("symbol_health", {})
    category_health = strategy_health.get("category_health", {})
    regime_alignment = strategy_health.get("regime_alignment", {})
    execution_quality = strategy_health.get("execution_quality", {})

    cols = st.columns(5)

    # Factor Health
    with cols[0]:
        strength = factor_health.get("strength_label", "unknown")
        mean_edge = factor_health.get("mean_edge", 0.0)
        pct_neg = factor_health.get("pct_negative", 0.0)

        strength_colors = {"strong": "#21c354", "mixed": "#f2c037", "weak": "#d94a4a", "unknown": "#888"}
        color = strength_colors.get(strength, "#888")

        st.markdown(f"**Factor Health**")
        st.markdown(
            f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:4px;'
            f'font-weight:600;">{strength.upper()}</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"Mean edge: {mean_edge:.3f}")
        st.caption(f"Negative: {pct_neg:.0%}")

    # Symbol Health
    with cols[1]:
        mean_edge = symbol_health.get("mean_edge", 0.0)
        symbol_count = symbol_health.get("symbol_count", 0)

        # Color based on mean edge
        if mean_edge >= 0.3:
            color = "#21c354"
        elif mean_edge >= 0:
            color = "#7ed957"
        elif mean_edge >= -0.3:
            color = "#f2c037"
        else:
            color = "#d94a4a"

        st.markdown(f"**Symbol Health**")
        st.markdown(
            f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:4px;'
            f'font-weight:600;">{mean_edge:.3f}</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"Symbols: {symbol_count}")
        top_syms = symbol_health.get("top_contributors", [])
        if top_syms:
            st.caption(f"Top: {top_syms[0].get('symbol', '?')}")

    # Category Health
    with cols[2]:
        mean_edge = category_health.get("mean_edge", 0.0)
        strongest = category_health.get("strongest_category")
        category_count = category_health.get("category_count", 0)

        if mean_edge >= 0.3:
            color = "#21c354"
        elif mean_edge >= 0:
            color = "#7ed957"
        elif mean_edge >= -0.3:
            color = "#f2c037"
        else:
            color = "#d94a4a"

        st.markdown(f"**Category Health**")
        st.markdown(
            f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:4px;'
            f'font-weight:600;">{mean_edge:.3f}</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"Categories: {category_count}")
        if strongest:
            st.caption(f"Strongest: {strongest}")

    # Regime Alignment
    with cols[3]:
        alignment_score = regime_alignment.get("alignment_score", 0.0)
        in_range = regime_alignment.get("in_expected_range", True)
        vol_regime = regime_alignment.get("vol_regime", "NORMAL")
        dd_state = regime_alignment.get("dd_state", "NORMAL")

        if alignment_score >= 0.8:
            color = "#21c354"
            label = "ALIGNED"
        elif alignment_score >= 0.5:
            color = "#f2c037"
            label = "PARTIAL"
        else:
            color = "#d94a4a"
            label = "MISALIGNED"

        st.markdown(f"**Regime Alignment**")
        st.markdown(
            f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:4px;'
            f'font-weight:600;">{label}</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"Vol: {vol_regime}")
        st.caption(f"DD: {dd_state}")

    # Execution Quality
    with cols[4]:
        bucket = execution_quality.get("quality_bucket", "unknown")
        router_q = execution_quality.get("router_quality", 0.0)
        slippage = execution_quality.get("avg_slippage_bps", 0.0)

        bucket_colors = {"excellent": "#21c354", "good": "#7ed957", "degraded": "#f2c037", "poor": "#d94a4a", "unknown": "#888"}
        color = bucket_colors.get(bucket, "#888")

        st.markdown(f"**Execution Quality**")
        st.markdown(
            f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:4px;'
            f'font-weight:600;">{bucket.upper()}</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"Router: {router_q:.0%}")
        st.caption(f"Slippage: {slippage:.1f} bps")

    # Notes section
    if notes:
        st.markdown("#### System Notes")
        for note in notes[:5]:  # Limit to 5 notes
            if note.startswith("✅"):
                st.success(note)
            elif note.startswith("⚠️"):
                st.warning(note)
            elif note.startswith("🚨"):
                st.error(note)
            else:
                st.info(note)


# ---------------------------------------------------------------------------
# Meta Info
# ---------------------------------------------------------------------------


def render_meta_info(edge_insights: Dict[str, Any]) -> None:
    """Render metadata about the edge insights snapshot."""
    updated_ts = edge_insights.get("updated_ts", "unknown")
    config_echo = edge_insights.get("config_echo", {})

    with st.expander("ℹ️ Edge Insights Metadata", expanded=False):
        st.caption(f"Last Updated: {updated_ts}")

        if config_echo:
            st.json(config_echo)


# ---------------------------------------------------------------------------
# Main Render Function
# ---------------------------------------------------------------------------


def render(state: Optional[Any] = None, edge_insights: Optional[Dict[str, Any]] = None) -> None:
    """
    Render the Edge Discovery panel.

    This panel visualizes the EdgeScanner surface (edge_insights.json):
    - Regime context (vol_regime, dd_state, risk_mode, router_quality)
    - Top/weak factors with edge scores
    - Top/weak symbols with hybrid scores and conviction
    - Top/weak categories with momentum and IR
    - Edge map visualization (bar charts)

    Args:
        state: Optional state object with edge_insights attribute
        edge_insights: Optional pre-loaded edge insights dict
    """
    st.header("🔎 Edge Discovery")
    st.caption("Unified view of where the edge is today — factors, symbols, and categories ranked by composite edge score.")

    # Load edge insights
    if edge_insights is None:
        edge_insights = load_edge_insights(state)

    if not edge_insights:
        st.warning(
            "Edge insights data not available. "
            "The EdgeScanner surface may not have been generated yet."
        )
        st.info("Run the executor intel loop to generate edge_insights.json.")
        return

    # Render sections
    render_regime_context(edge_insights)

    st.markdown("---")

    # v7.8_P2: Alpha Router Allocation (shows if enabled)
    render_alpha_router_allocation()

    st.markdown("---")

    # v7.8_P3: Universe Optimizer (shows if enabled)
    render_universe_optimizer()

    st.markdown("---")

    # v7.8_P4: Alpha Miner (shows if enabled)
    render_alpha_miner()

    st.markdown("---")

    # v7.8_P5: Cross-Pair Engine / Crossfire (shows if enabled)
    render_cross_pair_panel()

    st.markdown("---")

    # v7.8_P6: Sentinel-X Regime Classifier (shows if enabled)
    render_sentinel_x_panel()

    st.markdown("---")

    # v7.8_P7: Alpha Decay & Survival Curves (shows if enabled)
    render_alpha_decay_panel()

    st.markdown("---")

    render_factor_edges(edge_insights)

    st.markdown("---")

    render_symbol_edges(edge_insights)

    st.markdown("---")

    render_category_edges(edge_insights)

    st.markdown("---")

    render_edge_map(edge_insights)

    st.markdown("---")

    # v7.7_P7: Strategy Health Score section (rendered prominently)
    render_strategy_health(edge_insights)

    # v7.7_P6: Regime modifiers section
    render_regime_modifiers(edge_insights)

    render_meta_info(edge_insights)


# Alias for import compatibility with app.py integration
render_edge_discovery_panel = render


# ---------------------------------------------------------------------------
# Standalone Test Entry Point
# ---------------------------------------------------------------------------


def main():
    """Standalone entry point for testing."""
    st.set_page_config(page_title="Edge Discovery", layout="wide")
    render()


if __name__ == "__main__":
    main()
