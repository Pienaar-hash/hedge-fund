from __future__ import annotations

import streamlit as st
from typing import Any, Dict

from .utils.attribution_loaders import (
    get_carry_regimes,
    get_hybrid_deciles,
    get_trend_strength_buckets,
)


def _to_rows_from_bucket(bucket: Dict[str, Any], label: str) -> list[dict[str, Any]]:
    rows = []
    for key, stats in sorted(bucket.items(), key=lambda kv: kv[0]):
        rows.append(
            {
                label: key,
                "Trades": stats.get("trade_count", 0),
                "Total PnL": stats.get("total_pnl", 0.0),
                "Avg PnL": stats.get("avg_pnl", 0.0),
            }
        )
    return rows


def render_hybrid_factor_panel(snapshot: Dict[str, Any]) -> None:
    st.header("Hybrid Factor Attribution")

    deciles = get_hybrid_deciles(snapshot)
    trend_buckets = get_trend_strength_buckets(snapshot)
    carry_buckets = get_carry_regimes(snapshot)

    if not deciles and not trend_buckets and not carry_buckets:
        st.info("No hybrid factor attribution data yet (no vol_target trades with hybrid metadata).")
        return

    if deciles:
        st.subheader("Hybrid Score Deciles")
        st.dataframe(_to_rows_from_bucket(deciles, "Decile"), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if trend_buckets:
            st.subheader("Trend Strength Buckets")
            st.dataframe(_to_rows_from_bucket(trend_buckets, "Trend Strength"), use_container_width=True)
    with col2:
        if carry_buckets:
            st.subheader("Carry Regimes")
            st.dataframe(_to_rows_from_bucket(carry_buckets, "Carry Regime"), use_container_width=True)
