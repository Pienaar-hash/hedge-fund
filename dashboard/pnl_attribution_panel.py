from __future__ import annotations

import pandas as pd
import streamlit as st
import altair as alt
from datetime import datetime
from typing import Any, Dict

from dashboard.utils.attribution_loaders import load_pnl_attribution, safe_get_block

COLOR_GREEN = "#21c354"
COLOR_GOLD = "#f2c037"
COLOR_ORANGE = "#ff8c42"
COLOR_RED = "#d94a4a"
COLOR_OK_TINT = "#21c35420"
COLOR_WARN_TINT = "#f2c03720"
COLOR_DEF_TINT = "#d94a4a20"
COLOR_HALT_TINT = "#ff003320"


def _safe_number(value: Any) -> float:
    try:
        num = float(value)
        if num != num:  # NaN check
            return 0.0
        return num
    except Exception:
        return 0.0


def _render_table_and_bar(rows: list[dict], value_key: str, label_key: str) -> None:
    if not rows:
        st.info("No data available")
        return
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    chart_df = df[[label_key, value_key]].sort_values(value_key)
    st.bar_chart(chart_df.set_index(label_key), use_container_width=True)


def render_symbol_attribution(snapshot: Dict[str, Any]) -> None:
    st.subheader("Per-Symbol Attribution")
    per_symbol = safe_get_block(snapshot, "per_symbol", {}) or {}
    rows = []
    for sym, vals in per_symbol.items():
        rows.append(
            {
                "Symbol": sym,
                "Realized": _safe_number(vals.get("realized_pnl")),
                "Unrealized": _safe_number(vals.get("unrealized_pnl")),
                "Total": _safe_number(vals.get("total_pnl")),
                "Trades": int(vals.get("trade_count", 0) or 0),
            }
        )
    _render_table_and_bar(rows, "Total", "Symbol")


def render_strategy_attribution(snapshot: Dict[str, Any]) -> None:
    st.subheader("Per-Strategy Attribution")
    per_strategy = safe_get_block(snapshot, "per_strategy", {}) or {}
    rows = []
    for name, vals in per_strategy.items():
        rows.append(
            {
                "Strategy": name,
                "Realized": _safe_number(vals.get("realized_pnl")),
                "Unrealized": _safe_number(vals.get("unrealized_pnl")),
                "Total": _safe_number(vals.get("total_pnl")),
                "Trades": int(vals.get("trade_count", 0) or 0),
            }
        )
    _render_table_and_bar(rows, "Total", "Strategy")


def _regime_rows(regime_block: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    colors = [COLOR_GREEN, COLOR_GOLD, COLOR_ORANGE, COLOR_RED]
    for idx in ["0", "1", "2", "3"]:
        vals = regime_block.get(idx) or {}
        rows.append(
            {
                "Regime": idx,
                "Realized": _safe_number(vals.get("realized")),
                "Unrealized": _safe_number(vals.get("unrealized")),
                "Total": _safe_number(vals.get("total")),
                "Trades": int(vals.get("trade_count", 0) or 0),
                "Color": colors[int(idx)],
            }
        )
    return rows


def render_regime_attribution(snapshot: Dict[str, Any]) -> None:
    st.subheader("Regime Attribution")
    per_regime = safe_get_block(snapshot, "per_regime", {}) or {}
    atr_block = safe_get_block(per_regime, "atr", {}) or {}
    dd_block = safe_get_block(per_regime, "dd", {}) or {}
    col_atr, col_dd = st.columns(2)
    with col_atr:
        st.caption("ATR Regime")
        rows = _regime_rows(atr_block)
        if rows and any(r["Total"] != 0 or r["Trades"] for r in rows):
            df = pd.DataFrame(rows)
            chart = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X("Regime:O"),
                    y=alt.Y("Total:Q"),
                    color=alt.Color("Color:N", scale=None),
                )
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No ATR regime data")
    with col_dd:
        st.caption("Drawdown Regime")
        rows = _regime_rows(dd_block)
        if rows and any(r["Total"] != 0 or r["Trades"] for r in rows):
            df = pd.DataFrame(rows)
            chart = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X("Regime:O"),
                    y=alt.Y("Total:Q"),
                    color=alt.Color("Color:N", scale=None),
                )
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No DD regime data")


def render_risk_mode_attribution(snapshot: Dict[str, Any]) -> None:
    st.subheader("Risk Mode Attribution")
    per_risk = safe_get_block(snapshot, "per_risk_mode", {}) or {}
    tint_map = {"OK": COLOR_OK_TINT, "WARN": COLOR_WARN_TINT, "DEFENSIVE": COLOR_DEF_TINT, "HALTED": COLOR_HALT_TINT}
    cols = st.columns(4)
    modes = ["OK", "WARN", "DEFENSIVE", "HALTED"]
    for col, mode in zip(cols, modes):
        vals = per_risk.get(mode) or {}
        with col:
            st.markdown(
                f"""
                <div style="padding:0.8rem;border-radius:8px;background:{tint_map.get(mode, '#f0f0f0')};">
                    <div style="font-weight:700;">{mode}</div>
                    <div>Realized: {_safe_number(vals.get('realized')):,.2f}</div>
                    <div>Unrealized: {_safe_number(vals.get('unrealized')):,.2f}</div>
                    <div>Total: {_safe_number(vals.get('total')):,.2f}</div>
                    <div>Trades: {int(vals.get('trade_count', 0) or 0)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_daily_pnl_strip(snapshot: Dict[str, Any]) -> None:
    st.subheader("Daily PnL")
    per_day = safe_get_block(snapshot, "per_day", {}) or {}
    if not per_day:
        st.info("No daily PnL yet")
        return
    day_keys = sorted(per_day.keys())
    spans = []
    for key in day_keys:
        vals = per_day.get(key) or {}
        total = _safe_number(vals.get("total"))
        color = COLOR_GREEN if total >= 0 else COLOR_RED
        tooltip = (
            f"Realized: {_safe_number(vals.get('realized')):,.2f} | "
            f"Unrealized: {_safe_number(vals.get('unrealized')):,.2f}"
        )
        spans.append(
            f'<span title="{tooltip}" style="display:inline-block;margin-right:6px;'
            f'padding:6px 10px;border-radius:6px;background:{color}20;color:{color};'
            f'font-weight:700;">{key}: {total:+.2f}</span>'
        )
    st.markdown(" ".join(spans), unsafe_allow_html=True)


def render_summary_block(snapshot: Dict[str, Any]) -> None:
    st.subheader("Summary")
    summary = safe_get_block(snapshot, "summary", {}) or {}
    total = _safe_number(summary.get("total_pnl"))
    realized = _safe_number(summary.get("total_realized"))
    unrealized = _safe_number(summary.get("total_unrealized"))
    win_rate = _safe_number(summary.get("win_rate"))
    record_count = int(summary.get("record_count", 0) or 0)
    ts = summary.get("ts")
    ts_label = datetime.utcfromtimestamp(ts).isoformat() if isinstance(ts, (int, float)) else "n/a"
    cols = st.columns(3)
    cols[0].metric("Total PnL", f"{total:,.2f}")
    cols[1].metric("Realized", f"{realized:,.2f}")
    cols[2].metric("Unrealized", f"{unrealized:,.2f}")
    st.caption(f"Win Rate: {win_rate:.2%} · Trades: {record_count} · Snapshot: {ts_label}")


def render_pnl_attribution_panel(snapshot: Dict[str, Any] | None = None) -> None:
    snap = snapshot if snapshot is not None else load_pnl_attribution()
    st.header("PnL Attribution")
    if not snap:
        st.info("Attribution snapshot not available yet.")
        return
    render_summary_block(snap)
    st.divider()
    render_symbol_attribution(snap)
    st.divider()
    render_strategy_attribution(snap)
    st.divider()
    render_regime_attribution(snap)
    st.divider()
    render_risk_mode_attribution(snap)
    st.divider()
    render_daily_pnl_strip(snap)
