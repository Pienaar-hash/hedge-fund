"""
Cerberus Router Dashboard Panel — v7.8_P8

Displays multi-strategy portfolio router head multipliers and trends.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# State Loader
# ---------------------------------------------------------------------------


def load_cerberus_state(
    path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Load Cerberus state from file.

    Returns empty dict if file doesn't exist or is invalid.
    """
    p = Path(path or "logs/state/cerberus_state.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_trend_icon(direction: str) -> str:
    """Get icon for trend direction."""
    if direction == "up":
        return "📈"
    elif direction == "down":
        return "📉"
    else:
        return "➡️"


def _get_multiplier_color(multiplier: float) -> str:
    """Get color indicator for multiplier value."""
    if multiplier >= 1.3:
        return "🟢"  # Strong
    elif multiplier >= 1.1:
        return "🔵"  # Above average
    elif multiplier >= 0.9:
        return "⚪"  # Neutral
    elif multiplier >= 0.7:
        return "🟡"  # Below average
    else:
        return "🔴"  # Weak


def _get_head_description(head: str) -> str:
    """Get human-readable description for a strategy head."""
    descriptions = {
        "TREND": "Directional momentum / trend following",
        "MEAN_REVERT": "Mean reversion / reversal strategies",
        "RELATIVE_VALUE": "Crossfire pair trades (stat arb)",
        "CATEGORY": "Category rotation / sector tilt",
        "VOL_HARVEST": "Volatility harvesting / sizing",
        "EMERGENT_ALPHA": "Prospector / universe expansion",
    }
    return descriptions.get(head, head)


# ---------------------------------------------------------------------------
# Panel Renderer
# ---------------------------------------------------------------------------


def render_cerberus_panel(
    cerberus_state: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Render the Cerberus Multi-Strategy Router panel.

    Args:
        cerberus_state: Cerberus state dict (or None to load from file)
    """
    st.subheader("🐕‍🦺 Cerberus — Multi-Strategy Portfolio Router")

    # Load state if not provided
    if cerberus_state is None:
        cerberus_state = load_cerberus_state()

    if not cerberus_state:
        st.info(
            "No Cerberus state available yet. "
            "Enable `cerberus_router.enabled` in strategy_config.json and wait for the next intel cycle."
        )
        return

    # Header metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        regime = cerberus_state.get("regime", "UNKNOWN")
        st.metric("Current Regime", regime)

    with col2:
        health = _safe_float(cerberus_state.get("overall_health", 0.5))
        st.metric("Strategy Health", f"{health:.1%}")

    with col3:
        decay_survival = _safe_float(cerberus_state.get("avg_decay_survival", 1.0))
        st.metric("Avg Alpha Survival", f"{decay_survival:.1%}")

    with col4:
        cycle_count = cerberus_state.get("cycle_count", 0)
        st.metric("Cycle Count", cycle_count)

    st.markdown("---")

    # Head multipliers table
    st.markdown("### Strategy Head Multipliers")

    head_state = cerberus_state.get("head_state", {})
    heads = head_state.get("heads", {})

    if not heads:
        st.warning("No head state available.")
        return

    # Build dataframe
    rows: List[Dict[str, Any]] = []
    for head_name, metrics in heads.items():
        if not isinstance(metrics, dict):
            continue
        
        multiplier = _safe_float(metrics.get("multiplier", 1.0), 1.0)
        ema_score = _safe_float(metrics.get("ema_score", 0.5), 0.5)
        signal_score = _safe_float(metrics.get("signal_score", 0.5), 0.5)
        trend_dir = metrics.get("trend_direction", "flat")
        sample_count = metrics.get("sample_count", 0)

        rows.append({
            "Head": head_name,
            "Description": _get_head_description(head_name),
            "Multiplier": multiplier,
            "Color": _get_multiplier_color(multiplier),
            "EMA Score": ema_score,
            "Signal Score": signal_score,
            "Trend": _get_trend_icon(trend_dir),
            "Samples": sample_count,
        })

    if rows:
        df = pd.DataFrame(rows)
        df = df.sort_values("Multiplier", ascending=False)

        # Display with formatting
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Multiplier": st.column_config.NumberColumn(format="%.3f"),
                "EMA Score": st.column_config.NumberColumn(format="%.3f"),
                "Signal Score": st.column_config.NumberColumn(format="%.3f"),
            },
        )

        # Multiplier bar chart
        st.markdown("### Head Multiplier Distribution")
        chart_data = df[["Head", "Multiplier"]].set_index("Head")
        st.bar_chart(chart_data, use_container_width=True)

    st.markdown("---")

    # Signal components breakdown
    st.markdown("### Signal Components (Last Update)")

    # Show components for each head
    expander = st.expander("View Component Details", expanded=False)
    with expander:
        for head_name, metrics in heads.items():
            if not isinstance(metrics, dict):
                continue

            st.markdown(f"**{head_name}**")

            comp_cols = st.columns(5)
            with comp_cols[0]:
                st.caption(f"Regime: {_safe_float(metrics.get('regime_component', 1.0)):.2f}")
            with comp_cols[1]:
                st.caption(f"Decay: {_safe_float(metrics.get('decay_component', 1.0)):.2f}")
            with comp_cols[2]:
                st.caption(f"Meta: {_safe_float(metrics.get('meta_component', 1.0)):.2f}")
            with comp_cols[3]:
                st.caption(f"Edge: {_safe_float(metrics.get('edge_component', 0.5)):.2f}")
            with comp_cols[4]:
                st.caption(f"Health: {_safe_float(metrics.get('health_component', 0.5)):.2f}")

    st.markdown("---")

    # Notes and metadata
    notes = cerberus_state.get("notes", [])
    if notes:
        st.markdown("### Recent Notes")
        for note in notes[:5]:
            st.caption(f"• {note}")

    errors = cerberus_state.get("errors", [])
    if errors:
        st.markdown("### Recent Errors")
        for error in errors[:3]:
            st.error(error)

    # Metadata
    meta = cerberus_state.get("meta", {})
    if meta:
        with st.expander("Metadata", expanded=False):
            st.json(meta)


# ---------------------------------------------------------------------------
# Compact Widget
# ---------------------------------------------------------------------------


def render_cerberus_widget(
    cerberus_state: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Render a compact Cerberus widget for embedding in other panels.

    Shows just the head multipliers as a small table.
    """
    if cerberus_state is None:
        cerberus_state = load_cerberus_state()

    if not cerberus_state:
        st.caption("Cerberus: Not enabled")
        return

    head_state = cerberus_state.get("head_state", {})
    heads = head_state.get("heads", {})

    if not heads:
        st.caption("Cerberus: No head state")
        return

    # Compact display
    cols = st.columns(len(heads))
    for idx, (head_name, metrics) in enumerate(heads.items()):
        if not isinstance(metrics, dict):
            continue
        multiplier = _safe_float(metrics.get("multiplier", 1.0), 1.0)
        trend = metrics.get("trend_direction", "flat")
        icon = _get_trend_icon(trend)

        with cols[idx % len(cols)]:
            st.metric(
                label=head_name[:4],
                value=f"{multiplier:.2f}",
                delta=icon,
            )


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "load_cerberus_state",
    "render_cerberus_panel",
    "render_cerberus_widget",
]
