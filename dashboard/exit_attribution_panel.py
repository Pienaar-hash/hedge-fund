from __future__ import annotations

import streamlit as st
from typing import Any, Dict

from .utils.attribution_loaders import (
    get_exits_by_strategy,
    get_exits_by_symbol,
    get_exits_regimes,
    get_exits_summary,
)


def render_exit_summary_card(snapshot: Dict[str, Any]) -> None:
    summary = get_exits_summary(snapshot)
    total = summary["total_exits"]
    tp = summary["tp_hits"]
    sl = summary["sl_hits"]

    st.subheader("Exit Summary")

    if total == 0:
        st.info("No exits recorded yet.")
        return

    cols = st.columns(4)
    cols[0].metric("Total Exits", total)
    cols[1].metric("TP Hits", tp)
    cols[2].metric("SL Hits", sl)
    cols[3].metric("TP Ratio", f"{summary['tp_ratio']*100:.1f}%")

    col_rr = st.columns(3)
    col_rr[0].metric("Avg R:R (TP)", f"{summary['avg_rr_tp']:.2f}" if summary["avg_rr_tp"] is not None else "n/a")
    col_rr[1].metric("Avg R:R (SL)", f"{summary['avg_rr_sl']:.2f}" if summary["avg_rr_sl"] is not None else "n/a")
    col_rr[2].metric("Avg Exit PnL", f"{summary['avg_exit_pnl']:.2f}")


def render_exit_by_strategy(snapshot: Dict[str, Any]) -> None:
    by_strat = get_exits_by_strategy(snapshot)
    if not by_strat:
        return

    st.subheader("Exits by Strategy")
    rows = []
    for strat, stats in by_strat.items():
        rows.append(
            {
                "Strategy": strat,
                "Exits": stats.get("total_exits", 0),
                "TP Hits": stats.get("tp_hits", 0),
                "SL Hits": stats.get("sl_hits", 0),
                "TP Ratio": stats.get("tp_ratio", 0.0),
                "Avg Exit PnL": stats.get("avg_exit_pnl", 0.0),
                "Avg R:R (TP)": stats.get("avg_rr_tp"),
                "Avg R:R (SL)": stats.get("avg_rr_sl"),
            }
        )

    if rows:
        st.dataframe(rows, use_container_width=True)


def render_exit_by_symbol(snapshot: Dict[str, Any]) -> None:
    by_sym = get_exits_by_symbol(snapshot)
    if not by_sym:
        return

    st.subheader("Exits by Symbol")

    rows = []
    for sym, stats in by_sym.items():
        rows.append(
            {
                "Symbol": sym,
                "Exits": stats.get("total_exits", 0),
                "TP Hits": stats.get("tp_hits", 0),
                "SL Hits": stats.get("sl_hits", 0),
                "TP Ratio": stats.get("tp_ratio", 0.0),
                "Avg Exit PnL": stats.get("avg_exit_pnl", 0.0),
            }
        )

    if rows:
        st.dataframe(rows, use_container_width=True)


def render_exit_regime_heatmaps(snapshot: Dict[str, Any]) -> None:
    regimes = get_exits_regimes(snapshot)
    if not regimes:
        return

    st.subheader("Exits by Regime")

    for name, bucket in regimes.items():
        st.markdown(f"**{name.upper()} Regimes**")
        rows = []
        for key, stats in bucket.items():
            rows.append(
                {
                    "Regime": key,
                    "Exits": stats.get("total_exits", 0),
                    "TP Hits": stats.get("tp_hits", 0),
                    "SL Hits": stats.get("sl_hits", 0),
                    "TP Ratio": stats.get("tp_ratio", 0.0),
                    "Total Exit PnL": stats.get("total_exit_pnl", 0.0),
                }
            )
        if rows:
            st.dataframe(rows, use_container_width=True)


def render_exit_attribution_panel(snapshot: Dict[str, Any]) -> None:
    st.header("Exit Attribution")

    render_exit_summary_card(snapshot)
    render_exit_by_strategy(snapshot)
    render_exit_by_symbol(snapshot)
    render_exit_regime_heatmaps(snapshot)
