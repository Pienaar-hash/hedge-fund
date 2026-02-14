"""
PnL Attribution Widget — Breakdown by Symbol, Strategy, Risk Mode

Displays realized/unrealized PnL split across dimensions the executor
actually writes: per_symbol, per_strategy, per_risk_mode.

Data source: logs/state/pnl_attribution.json
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STATE_PATH = Path("logs/state/pnl_attribution.json")

_RISK_MODE_TINTS: Dict[str, str] = {
    "OK": "#21c35420",
    "WARN": "#f2c03720",
    "DEFENSIVE": "#d94a4a20",
    "HALTED": "#ff003320",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_num(v: Any) -> float:
    try:
        n = float(v)
        return 0.0 if n != n else n  # NaN guard
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# State Loader
# ---------------------------------------------------------------------------

def load_pnl_attribution_state(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load PnL attribution snapshot."""
    p = path or _STATE_PATH
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def render_pnl_attribution_widget(state: Dict[str, Any]) -> None:
    """Render PnL attribution panel."""
    st.header("PnL Attribution")

    if not state:
        st.info("PnL attribution state not available.")
        return

    # ── Summary KPIs ──────────────────────────────────────────────────────
    summary = state.get("summary", {})
    total = _safe_num(summary.get("total_pnl"))
    realized = _safe_num(summary.get("total_realized"))
    unrealized = _safe_num(summary.get("total_unrealized"))
    win_rate = _safe_num(summary.get("win_rate"))
    record_count = int(summary.get("record_count", 0) or 0)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        color = "normal" if total >= 0 else "inverse"
        st.metric("Total PnL", f"${total:,.2f}", delta_color=color)
    with c2:
        st.metric("Realized", f"${realized:,.2f}")
    with c3:
        st.metric("Unrealized", f"${unrealized:,.2f}")
    with c4:
        st.metric("Win Rate", f"{win_rate:.1%}" if record_count else "—")

    ts = summary.get("ts")
    if isinstance(ts, (int, float)):
        ts_label = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    else:
        ts_label = "n/a"
    st.caption(f"Trades: {record_count} · Snapshot: {ts_label}")

    # ── Per-Symbol ────────────────────────────────────────────────────────
    per_symbol = state.get("per_symbol", {})
    if per_symbol:
        st.subheader("Per-Symbol")
        rows = []
        for sym, v in sorted(per_symbol.items()):
            rows.append({
                "Symbol": sym,
                "Realized": round(_safe_num(v.get("realized_pnl")), 2),
                "Unrealized": round(_safe_num(v.get("unrealized_pnl")), 2),
                "Total": round(_safe_num(v.get("total_pnl")), 2),
                "Trades": int(v.get("trade_count", 0) or 0),
            })
        _render_table(rows)

    # ── Per-Strategy ──────────────────────────────────────────────────────
    per_strategy = state.get("per_strategy", {})
    if per_strategy:
        st.subheader("Per-Strategy")
        rows = []
        for name, v in sorted(per_strategy.items()):
            rows.append({
                "Strategy": name,
                "Realized": round(_safe_num(v.get("realized_pnl")), 2),
                "Unrealized": round(_safe_num(v.get("unrealized_pnl")), 2),
                "Total": round(_safe_num(v.get("total_pnl")), 2),
                "Trades": int(v.get("trade_count", 0) or 0),
            })
        _render_table(rows)

    # ── Per-Risk-Mode ─────────────────────────────────────────────────────
    per_risk = state.get("per_risk_mode", {})
    if per_risk:
        st.subheader("Risk Mode Breakdown")
        modes = ["OK", "WARN", "DEFENSIVE", "HALTED"]
        cols = st.columns(len(modes))
        for col, mode in zip(cols, modes):
            v = per_risk.get(mode, {})
            r = _safe_num(v.get("realized"))
            u = _safe_num(v.get("unrealized"))
            t = _safe_num(v.get("total"))
            tc = int(v.get("trade_count", 0) or 0)
            tint = _RISK_MODE_TINTS.get(mode, "#f0f0f020")
            with col:
                st.html(
                    f'<div style="padding:0.6rem;border-radius:8px;background:{tint};">'
                    f'<div style="font-weight:700;">{mode}</div>'
                    f"<div>Realized: {r:,.2f}</div>"
                    f"<div>Unrealized: {u:,.2f}</div>"
                    f"<div>Total: {t:,.2f}</div>"
                    f"<div>Trades: {tc}</div>"
                    f"</div>"
                )

    # ── Empty-state fallback ──────────────────────────────────────────────
    if not per_symbol and not per_strategy and record_count == 0:
        st.info("No PnL data yet. Attribution populates once trades are recorded.")


def _render_table(rows: list[dict]) -> None:
    """Render a list of dicts as a dataframe (with pandas fallback)."""
    if not rows:
        return
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    except ImportError:
        for r in rows:
            st.text(str(r))
