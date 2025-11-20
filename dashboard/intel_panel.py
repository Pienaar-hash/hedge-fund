from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st


def _risk_mode_badge(mode: Optional[str]) -> str:
    label = (mode or "unknown").lower()
    color_map = {
        "normal": "#21ba45",
        "cautious": "#f2c037",
        "defensive": "#db2828",
    }
    color = color_map.get(label, "#95a5a6")
    return (
        f'<span style="display:inline-block;padding:0.25em 0.7em;'
        f'border-radius:0.6em;font-weight:700;color:#fff;background:{color};text-transform:uppercase;">{label}</span>'
    )


def _expectancy_table(expectancy: Optional[Dict[str, Any]]) -> None:
    symbols = expectancy.get("symbols") if isinstance(expectancy, dict) else None
    if not isinstance(symbols, dict) or not symbols:
        st.info("Expectancy data unavailable.")
        return
    rows = []
    for symbol, stats in symbols.items():
        if not isinstance(stats, dict):
            continue
        rows.append(
            {
                "symbol": symbol,
                "expectancy": stats.get("expectancy"),
                "hit_rate": stats.get("hit_rate"),
                "avg_win": stats.get("avg_win"),
                "avg_loss": stats.get("avg_loss"),
            }
        )
    if not rows:
        st.info("No expectancy rows to display.")
        return
    df = pd.DataFrame(rows)
    df = df.sort_values("expectancy", ascending=False, na_position="last").reset_index(drop=True)
    st.dataframe(df.head(50), use_container_width=True, height=360)


def _scores_table(scores: Optional[Dict[str, Any]]) -> None:
    symbols = scores.get("symbols") if isinstance(scores, dict) else None
    if not isinstance(symbols, list) or not symbols:
        st.info("Symbol scores unavailable.")
        return
    rows = []
    for entry in symbols:
        if not isinstance(entry, dict):
            continue
        rows.append(
            {
                "symbol": entry.get("symbol"),
                "score": entry.get("score"),
                "router": (entry.get("components") or {}).get("router"),
                "expectancy": (entry.get("components") or {}).get("expectancy"),
                "slippage": (entry.get("components") or {}).get("slippage_penalty"),
            }
        )
    if not rows:
        st.info("No scores to display.")
        return
    df = pd.DataFrame(rows)
    df = df.sort_values("score", ascending=False, na_position="last").reset_index(drop=True)
    st.dataframe(df.head(50), use_container_width=True, height=360)


def render_intel_panel(
    expectancy: Optional[Dict[str, Any]],
    scores: Optional[Dict[str, Any]],
    risk_alloc: Optional[Dict[str, Any]],
) -> None:
    st.markdown("### Intel (v6)")

    risk_mode = None
    if isinstance(risk_alloc, dict):
        global_block = risk_alloc.get("global")
        if isinstance(global_block, dict):
            risk_mode = global_block.get("risk_mode")
    badge_html = _risk_mode_badge(risk_mode)
    st.markdown(f"**Risk Mode:** {badge_html}", unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Expectancy (per symbol)")
        _expectancy_table(expectancy)

    with right:
        st.markdown("#### Symbol Scores")
        _scores_table(scores)


__all__ = ["render_intel_panel"]
