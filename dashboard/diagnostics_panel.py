from __future__ import annotations

import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Any, Dict, List

from dashboard.utils.diagnostics_loaders import (
    load_equity_state,
    load_positions_state,
    load_pnl_attribution_state,
)
from dashboard.state_v7 import get_ledger_consistency_status

COLOR_GREEN = "#21c354"
COLOR_GOLD = "#f2c037"
COLOR_ORANGE = "#ff8c42"
COLOR_RED = "#d94a4a"


def _safe_number(value: Any) -> float:
    try:
        num = float(value)
        if num != num:  # NaN
            return 0.0
        return num
    except Exception:
        return 0.0


def render_correlation_heatmap(equity_snapshot: Dict[str, Any], pnl_attribution: Dict[str, Any]) -> None:
    st.subheader("Symbol Correlation")
    per_day = pnl_attribution.get("per_day") if isinstance(pnl_attribution, dict) else None
    per_symbol = pnl_attribution.get("per_symbol") if isinstance(pnl_attribution, dict) else None

    if per_day and per_symbol:
        # Build per-symbol daily totals (fallback: total_pnl if available)
        symbols = list(per_symbol.keys())
        day_keys = sorted(per_day.keys())
        if symbols and day_keys:
            data = {sym: [] for sym in symbols}
            for day in day_keys:
                for sym in symbols:
                    sym_pnl = 0.0
                    # If per-day attribution by symbol isn't available, leave zero
                    sym_pnl = _safe_number(per_symbol.get(sym, {}).get("total_pnl"))
                    data[sym].append(sym_pnl)
            df = pd.DataFrame(data, index=day_keys)
            corr = df.corr()
            st.dataframe(corr, use_container_width=True)
            return

    if per_symbol:
        # Minimal fallback: diagonal correlation from totals
        symbols = list(per_symbol.keys())
        if symbols:
            corr = pd.DataFrame(0.0, index=symbols, columns=symbols)
            for sym in symbols:
                corr.loc[sym, sym] = 1.0
            st.dataframe(corr, use_container_width=True)
            return

    st.info("Not enough data to compute symbol correlations yet.")


def render_daily_return_strip(equity_snapshot: Dict[str, Any]) -> None:
    st.subheader("Daily Returns")
    timestamps = equity_snapshot.get("timestamps") or []
    equity_series = equity_snapshot.get("equity") or []
    if not timestamps or not equity_series or len(timestamps) != len(equity_series):
        st.info("Daily returns unavailable.")
        return
    day_points: Dict[str, List[float]] = {}
    for ts, eq in zip(timestamps, equity_series):
        try:
            day = datetime.utcfromtimestamp(float(ts)).strftime("%Y-%m-%d")
        except Exception:
            continue
        day_points.setdefault(day, []).append(_safe_number(eq))
    rows = []
    for day, vals in sorted(day_points.items()):
        if not vals:
            continue
        daily_ret = vals[-1] - vals[0]
        rows.append((day, daily_ret))
    if not rows:
        st.info("Daily returns unavailable.")
        return
    spans = []
    for day, ret in rows:
        color = COLOR_GREEN if ret >= 0 else COLOR_RED
        spans.append(
            f'<span style="display:inline-block;margin-right:6px; padding:6px 10px;'
            f'border-radius:6px;background:{color}20;color:{color};font-weight:700;">'
            f'{day}: {ret:+.2f}</span>'
        )
    st.markdown(" ".join(spans), unsafe_allow_html=True)


def render_symbol_drawdown_snapshot(equity_snapshot: Dict[str, Any], positions: List[Dict[str, Any]]) -> None:
    st.subheader("Drawdown & Positions")
    drawdown = equity_snapshot.get("drawdown") or []
    current_dd = _safe_number(drawdown[-1]) if drawdown else 0.0
    st.metric("Current Drawdown", f"{current_dd*100:.2f}%")

    if not positions:
        st.info("No positions available.")
        return
    sorted_positions = sorted(
        positions,
        key=lambda row: _safe_number(row.get("unrealized_pnl")),
    )[:5]
    rows = []
    for pos in sorted_positions:
        rows.append(
            {
                "Symbol": pos.get("symbol", "???"),
                "Unrealized PnL": _safe_number(pos.get("unrealized_pnl")),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_regime_pnl_bars(pnl_attribution: Dict[str, Any]) -> None:
    st.subheader("Regime PnL")
    per_regime = pnl_attribution.get("per_regime") if isinstance(pnl_attribution, dict) else None
    if not per_regime:
        st.info("Regime attribution not available.")
        return
    colors = [COLOR_GREEN, COLOR_GOLD, COLOR_ORANGE, COLOR_RED]
    for key in ("atr", "dd"):
        block = per_regime.get(key) if isinstance(per_regime, dict) else None
        if not block:
            continue
        rows = []
        for idx in ["0", "1", "2", "3"]:
            vals = block.get(idx) or {}
            rows.append({"Regime": idx, "Total": _safe_number(vals.get("total")), "Color": colors[int(idx)]})
        st.caption(f"{key.upper()} regime")
        df = pd.DataFrame(rows)
        st.bar_chart(df.set_index("Regime")["Total"], use_container_width=True)


def render_ledger_consistency(consistency: Dict[str, Any] | None = None) -> None:
    """Render ledger consistency status (v7.4_C3)."""
    st.subheader("Position Ledger Status")
    
    if consistency is None:
        try:
            consistency = get_ledger_consistency_status()
        except Exception:
            st.warning("Unable to load ledger consistency status")
            return
    
    status = consistency.get("status", "unknown")
    message = consistency.get("message", "")
    
    if status == "ok":
        st.success(message)
    elif status == "partial":
        st.warning(message)
    elif status == "error":
        st.error(message)
    else:
        st.info(f"Ledger status: {status}")
    
    # Show details
    cols = st.columns(3)
    cols[0].metric("Positions", consistency.get("num_positions", 0))
    cols[1].metric("Ledger Entries", consistency.get("num_ledger", 0))
    cols[2].metric("With TP/SL", consistency.get("num_with_tp_sl", 0))


def render_diagnostics_panel(
    equity: Dict[str, Any] | None = None,
    positions: List[Dict[str, Any]] | None = None,
    pnl_attribution: Dict[str, Any] | None = None,
) -> None:
    st.header("Portfolio Diagnostics v7")
    eq = equity if equity is not None else load_equity_state()
    pos = positions if positions is not None else load_positions_state()
    pnl = pnl_attribution if pnl_attribution is not None else load_pnl_attribution_state()

    render_symbol_drawdown_snapshot(eq, pos)
    st.divider()
    render_ledger_consistency()
    st.divider()
    render_daily_return_strip(eq)
    st.divider()
    render_correlation_heatmap(eq, pnl)
    st.divider()
    render_regime_pnl_bars(pnl)
