"""
Hydra Dashboard Panel — v7.9_P1

Dashboard panel for the Hydra Multi-Strategy Execution Engine.
Displays per-head budgets, usage, positions, and merged intents.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

try:
    import streamlit as st
    import pandas as pd
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    st = None
    pd = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_STATE_PATH = Path("logs/state/hydra_state.json")

STRATEGY_HEADS = [
    "TREND",
    "MEAN_REVERT",
    "RELATIVE_VALUE",
    "CATEGORY",
    "VOL_HARVEST",
    "EMERGENT_ALPHA",
]

HEAD_COLORS = {
    "TREND": "#4CAF50",
    "MEAN_REVERT": "#2196F3",
    "RELATIVE_VALUE": "#FF9800",
    "CATEGORY": "#9C27B0",
    "VOL_HARVEST": "#F44336",
    "EMERGENT_ALPHA": "#00BCD4",
}

HEAD_ICONS = {
    "TREND": "📈",
    "MEAN_REVERT": "🔄",
    "RELATIVE_VALUE": "⚖️",
    "CATEGORY": "📊",
    "VOL_HARVEST": "🎯",
    "EMERGENT_ALPHA": "✨",
}


# ---------------------------------------------------------------------------
# State Loading
# ---------------------------------------------------------------------------


def load_hydra_state(state_path: Path | str | None = None) -> Dict[str, Any]:
    """
    Load Hydra state from disk.

    Args:
        state_path: Path to hydra_state.json

    Returns:
        State dict or empty dict if missing/invalid
    """
    path = Path(state_path or DEFAULT_STATE_PATH)
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


# ---------------------------------------------------------------------------
# Panel Rendering
# ---------------------------------------------------------------------------


def render_hydra_panel(state_path: Path | str | None = None) -> None:
    """
    Render the full Hydra dashboard panel.

    Args:
        state_path: Path to hydra_state.json
    """
    if not HAS_STREAMLIT:
        return

    st.subheader("🐍 Hydra Multi-Strategy Engine")

    state = load_hydra_state(state_path)

    if not state:
        st.info("Hydra state not available. Engine may be disabled or not yet run.")
        return

    # Header info
    updated_ts = state.get("updated_ts", "Unknown")
    cycle_count = state.get("cycle_count", 0)
    st.caption(f"Last updated: {updated_ts} | Cycle: {cycle_count}")

    # Show notes/errors if any
    errors = state.get("errors", [])
    if errors:
        with st.expander("⚠️ Errors", expanded=True):
            for err in errors[:5]:
                st.error(err)

    notes = state.get("notes", [])
    if notes:
        with st.expander("📝 Notes", expanded=False):
            for note in notes[:10]:
                st.text(note)

    # Head budgets and usage
    st.markdown("### Head Allocation")

    head_budgets = state.get("head_budgets", {})
    head_usage = state.get("head_usage", {})
    head_positions = state.get("head_positions", {})

    # Build table data
    table_data = []
    for head in STRATEGY_HEADS:
        budget = head_budgets.get(head, 0.0)
        usage = head_usage.get(head, 0.0)
        positions = head_positions.get(head, 0)
        utilization = (usage / budget * 100) if budget > 0 else 0

        table_data.append({
            "Head": f"{HEAD_ICONS.get(head, '')} {head}",
            "Budget (NAV%)": f"{budget * 100:.1f}%",
            "Usage (NAV%)": f"{usage * 100:.2f}%",
            "Utilization": f"{utilization:.1f}%",
            "Positions": positions,
        })

    if table_data:
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Utilization bar chart
    if head_usage:
        st.markdown("### Head Utilization")

        chart_data = []
        for head in STRATEGY_HEADS:
            budget = head_budgets.get(head, 0.0)
            usage = head_usage.get(head, 0.0)
            if budget > 0:
                chart_data.append({
                    "Head": head,
                    "Utilization %": (usage / budget) * 100,
                })

        if chart_data:
            chart_df = pd.DataFrame(chart_data)
            st.bar_chart(chart_df.set_index("Head"), height=250)

    # Merged intents
    merged_intents = state.get("merged_intents", [])

    st.markdown(f"### Merged Intents ({len(merged_intents)})")

    if merged_intents:
        intent_data = []
        for intent in merged_intents[:20]:  # Limit display
            heads_str = ", ".join(intent.get("heads", []))
            contributions = intent.get("head_contributions", {})
            contrib_str = ", ".join(
                f"{h}: {v * 100:.2f}%"
                for h, v in sorted(contributions.items(), key=lambda x: -abs(x[1]))
            )

            intent_data.append({
                "Symbol": intent.get("symbol", ""),
                "Side": intent.get("net_side", "").upper(),
                "NAV%": f"{intent.get('nav_pct', 0) * 100:.2f}%",
                "Score": f"{intent.get('score', 0):.3f}",
                "Heads": heads_str,
                "Contributions": contrib_str,
            })

        if intent_data:
            intent_df = pd.DataFrame(intent_data)
            st.dataframe(intent_df, use_container_width=True, hide_index=True)
    else:
        st.info("No merged intents in current cycle.")

    # Metadata
    meta = state.get("meta", {})
    if meta:
        with st.expander("🔧 Metadata", expanded=False):
            st.json(meta)


def render_hydra_widget(state_path: Path | str | None = None) -> None:
    """
    Render a compact Hydra widget for sidebar or summary views.

    Args:
        state_path: Path to hydra_state.json
    """
    if not HAS_STREAMLIT:
        return

    state = load_hydra_state(state_path)

    if not state:
        st.caption("🐍 Hydra: Disabled")
        return

    head_usage = state.get("head_usage", {})
    merged_intents = state.get("merged_intents", [])
    total_usage = sum(head_usage.values())

    st.markdown("**🐍 Hydra**")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total NAV%", f"{total_usage * 100:.1f}%")

    with col2:
        active_heads = sum(1 for v in head_usage.values() if v > 0)
        st.metric("Active Heads", f"{active_heads}/6")

    with col3:
        st.metric("Intents", len(merged_intents))


def render_hydra_head_breakdown(state_path: Path | str | None = None) -> None:
    """
    Render a detailed head-by-head breakdown.

    Args:
        state_path: Path to hydra_state.json
    """
    if not HAS_STREAMLIT:
        return

    state = load_hydra_state(state_path)

    if not state:
        return

    head_budgets = state.get("head_budgets", {})
    head_usage = state.get("head_usage", {})
    head_positions = state.get("head_positions", {})
    merged_intents = state.get("merged_intents", [])

    # Group intents by head
    intents_by_head: Dict[str, List[Dict]] = {h: [] for h in STRATEGY_HEADS}
    for intent in merged_intents:
        for head in intent.get("heads", []):
            if head in intents_by_head:
                intents_by_head[head].append(intent)

    # Create tabs for each head
    tabs = st.tabs([f"{HEAD_ICONS.get(h, '')} {h}" for h in STRATEGY_HEADS])

    for i, head in enumerate(STRATEGY_HEADS):
        with tabs[i]:
            budget = head_budgets.get(head, 0.0)
            usage = head_usage.get(head, 0.0)
            positions = head_positions.get(head, 0)
            utilization = (usage / budget * 100) if budget > 0 else 0

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Budget", f"{budget * 100:.1f}%")
            with col2:
                st.metric("Usage", f"{usage * 100:.2f}%")
            with col3:
                st.metric("Utilization", f"{utilization:.0f}%")
            with col4:
                st.metric("Positions", positions)

            # Show intents for this head
            head_intents = intents_by_head.get(head, [])
            if head_intents:
                st.markdown("**Intents:**")
                for intent in head_intents[:10]:
                    contribution = intent.get("head_contributions", {}).get(head, 0)
                    symbol = intent.get("symbol", "")
                    side = intent.get("net_side", "")
                    st.text(f"  {symbol} {side.upper()} | Contribution: {contribution * 100:.2f}%")
            else:
                st.caption("No intents for this head")


# ---------------------------------------------------------------------------
# Hydra PnL Panel — v7.9_P2
# ---------------------------------------------------------------------------

DEFAULT_PNL_STATE_PATH = Path("logs/state/hydra_pnl.json")


def load_hydra_pnl_state(state_path: Path | str | None = None) -> Dict[str, Any]:
    """
    Load Hydra PnL state from disk.

    Args:
        state_path: Path to hydra_pnl.json

    Returns:
        State dict or empty dict if missing/invalid
    """
    path = Path(state_path or DEFAULT_PNL_STATE_PATH)
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


def render_hydra_pnl_panel(state_path: Path | str | None = None) -> None:
    """
    Render the Hydra PnL Attribution panel.

    Args:
        state_path: Path to hydra_pnl.json
    """
    if not HAS_STREAMLIT:
        return

    st.subheader("📊 Hydra PnL Attribution")

    state = load_hydra_pnl_state(state_path)

    if not state:
        st.info("Hydra PnL state not available. Engine may be disabled or not yet run.")
        return

    # Header info
    updated_ts = state.get("updated_ts", "Unknown")
    meta = state.get("meta", {})
    st.caption(f"Last updated: {updated_ts}")

    # Summary metrics
    total_realized = meta.get("total_realized_pnl", 0.0)
    total_unrealized = meta.get("total_unrealized_pnl", 0.0)
    total_equity = meta.get("total_equity", 0.0)
    max_dd = meta.get("max_head_drawdown", 0.0)
    heads_killed = meta.get("heads_killed", 0)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Realized PnL", f"${total_realized:,.2f}")
    with col2:
        st.metric("Unrealized PnL", f"${total_unrealized:,.2f}")
    with col3:
        st.metric("Total Equity", f"${total_equity:,.2f}")
    with col4:
        st.metric("Max Head DD", f"{max_dd * 100:.1f}%")
    with col5:
        st.metric("Heads Killed", f"{heads_killed}/6")

    # Head-level stats table
    st.markdown("### Head Performance")

    heads = state.get("heads", {})
    table_data = []

    for head in STRATEGY_HEADS:
        head_stats = heads.get(head, {})
        equity = head_stats.get("equity", 0.0)
        dd = head_stats.get("drawdown", 0.0)
        win_rate = head_stats.get("win_rate", 0.0)
        trades = head_stats.get("trades", 0)
        throttle = head_stats.get("throttle_scale", 1.0)
        kill_switch = head_stats.get("kill_switch_active", False)
        cooldown = head_stats.get("cooldown_remaining", 0)

        status = "🔴 KILLED" if kill_switch else ("🟡 THROTTLED" if throttle < 1.0 else "🟢 ACTIVE")

        table_data.append({
            "Head": f"{HEAD_ICONS.get(head, '')} {head}",
            "Equity": f"${equity:,.2f}",
            "Drawdown": f"{dd * 100:.1f}%",
            "Win Rate": f"{win_rate * 100:.1f}%",
            "Trades": trades,
            "Throttle": f"{throttle * 100:.0f}%",
            "Status": status,
            "Cooldown": cooldown if kill_switch else "-",
        })

    if table_data:
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Equity bar chart
    st.markdown("### Head Equity Distribution")

    chart_data = []
    for head in STRATEGY_HEADS:
        head_stats = heads.get(head, {})
        equity = head_stats.get("equity", 0.0)
        chart_data.append({
            "Head": head,
            "Equity": equity,
        })

    if chart_data:
        chart_df = pd.DataFrame(chart_data)
        st.bar_chart(chart_df.set_index("Head"), height=250)

    # Drawdown chart
    st.markdown("### Head Drawdowns")

    dd_data = []
    for head in STRATEGY_HEADS:
        head_stats = heads.get(head, {})
        dd = head_stats.get("drawdown", 0.0)
        dd_data.append({
            "Head": head,
            "Drawdown %": dd * 100,
        })

    if dd_data:
        dd_df = pd.DataFrame(dd_data)
        st.bar_chart(dd_df.set_index("Head"), height=200)


def render_hydra_pnl_widget(state_path: Path | str | None = None) -> None:
    """
    Render a compact Hydra PnL widget for sidebar or summary views.

    Args:
        state_path: Path to hydra_pnl.json
    """
    if not HAS_STREAMLIT:
        return

    state = load_hydra_pnl_state(state_path)

    if not state:
        st.caption("📊 PnL: Disabled")
        return

    meta = state.get("meta", {})

    total_equity = meta.get("total_equity", 0.0)
    max_dd = meta.get("max_head_drawdown", 0.0)
    heads_killed = meta.get("heads_killed", 0)

    st.markdown("**📊 Head PnL**")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Equity", f"${total_equity:,.0f}")

    with col2:
        st.metric("Max DD", f"{max_dd * 100:.1f}%")

    with col3:
        active = 6 - heads_killed
        st.metric("Active", f"{active}/6")


def render_hydra_pnl_head_detail(
    head: str,
    state_path: Path | str | None = None,
) -> None:
    """
    Render detailed PnL stats for a single head.

    Args:
        head: Strategy head name
        state_path: Path to hydra_pnl.json
    """
    if not HAS_STREAMLIT:
        return

    state = load_hydra_pnl_state(state_path)

    if not state:
        st.info("PnL data not available")
        return

    heads = state.get("heads", {})
    head_stats = heads.get(head, {})

    if not head_stats:
        st.info(f"No PnL data for {head}")
        return

    icon = HEAD_ICONS.get(head, "")
    st.markdown(f"### {icon} {head} PnL Detail")

    # Main metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        realized = head_stats.get("realized_pnl", 0.0)
        st.metric("Realized PnL", f"${realized:,.2f}")

    with col2:
        unrealized = head_stats.get("unrealized_pnl", 0.0)
        st.metric("Unrealized PnL", f"${unrealized:,.2f}")

    with col3:
        equity = head_stats.get("equity", 0.0)
        st.metric("Equity", f"${equity:,.2f}")

    with col4:
        max_eq = head_stats.get("max_equity", 0.0)
        st.metric("Max Equity", f"${max_eq:,.2f}")

    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        dd = head_stats.get("drawdown", 0.0)
        st.metric("Drawdown", f"{dd * 100:.1f}%")

    with col2:
        win_rate = head_stats.get("win_rate", 0.0)
        st.metric("Win Rate", f"{win_rate * 100:.1f}%")

    with col3:
        avg_r = head_stats.get("avg_R", 0.0)
        st.metric("Avg R-Multiple", f"{avg_r:.2f}")

    with col4:
        trades = head_stats.get("trades", 0)
        wins = head_stats.get("wins", 0)
        st.metric("Trades (Wins)", f"{trades} ({wins})")

    # Throttle/kill status
    col1, col2, col3 = st.columns(3)

    with col1:
        throttle = head_stats.get("throttle_scale", 1.0)
        st.metric("Throttle Scale", f"{throttle * 100:.0f}%")

    with col2:
        kill = head_stats.get("kill_switch_active", False)
        status = "🔴 ACTIVE" if kill else "🟢 OFF"
        st.metric("Kill Switch", status)

    with col3:
        cooldown = head_stats.get("cooldown_remaining", 0)
        st.metric("Cooldown Cycles", cooldown)

    # Exposure
    exposure = head_stats.get("gross_exposure", 0.0)
    veto_count = head_stats.get("veto_count", 0)
    st.caption(f"Gross Exposure: {exposure * 100:.1f}% | Veto Count: {veto_count}")


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Hydra state
    "load_hydra_state",
    "render_hydra_panel",
    "render_hydra_widget",
    "render_hydra_head_breakdown",
    # Hydra PnL
    "load_hydra_pnl_state",
    "render_hydra_pnl_panel",
    "render_hydra_pnl_widget",
    "render_hydra_pnl_head_detail",
    # Constants
    "HEAD_COLORS",
    "HEAD_ICONS",
]
