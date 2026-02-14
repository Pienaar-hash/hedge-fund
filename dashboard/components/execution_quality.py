"""
Execution Quality Widget — Minotaur Fill Metrics

Displays fill slippage, throttling status, thin-liquidity alerts,
and per-symbol execution regime.

Data source: logs/state/execution_quality.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STATE_PATH = Path("logs/state/execution_quality.json")

REGIME_ICONS: Dict[str, str] = {
    "NORMAL": "🟢",
    "VOLATILE": "🟡",
    "THIN": "🟠",
    "CRISIS": "🔴",
}


# ---------------------------------------------------------------------------
# State Loader
# ---------------------------------------------------------------------------

def load_execution_quality_state(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load Minotaur execution quality state."""
    p = path or _STATE_PATH
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def render_execution_quality_widget(state: Dict[str, Any]) -> None:
    """Render Minotaur execution quality panel."""
    st.header("Execution Quality")

    if not state:
        st.info("Execution quality state not available.")
        return

    meta = state.get("meta", {})
    symbols = state.get("symbols", {})
    updated_ts = state.get("updated_ts", "")

    # ── Top-level status row ──────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        enabled = meta.get("enabled", False)
        st.metric("Minotaur", "Enabled" if enabled else "Disabled")

    with c2:
        throttling = meta.get("throttling_active", False)
        st.metric("Throttling", "ACTIVE" if throttling else "Normal")

    with c3:
        thin: List[str] = meta.get("thin_liquidity_symbols", [])
        st.metric("Thin Liquidity", len(thin))

    with c4:
        crunch: List[str] = meta.get("crunch_symbols", [])
        st.metric("Crunch Symbols", len(crunch))

    # Slippage target
    target_bps = meta.get("max_slippage_target_bps")
    if target_bps is not None:
        st.caption(f"Max slippage target: {target_bps} bps")

    # ── Alerts ────────────────────────────────────────────────────────────
    if meta.get("throttling_active"):
        st.warning("Minotaur throttling is active — execution sizing is being reduced.")
    if thin:
        st.warning(f"Thin liquidity: {', '.join(thin)}")
    if crunch:
        st.warning(f"Crunch (slippage spike): {', '.join(crunch)}")

    # ── Per-symbol table ──────────────────────────────────────────────────
    if not symbols:
        st.info("No per-symbol execution data yet. Metrics appear after fills are processed.")
        if updated_ts:
            st.caption(f"Last updated: {updated_ts}")
        return

    st.subheader("Per-Symbol Quality")

    rows = []
    for sym, s in sorted(symbols.items()):
        regime = s.get("last_regime", "NORMAL")
        rows.append({
            "Symbol": sym,
            "Regime": f"{REGIME_ICONS.get(regime, '⚪')} {regime}",
            "Avg Slip (bps)": round(s.get("avg_slippage_bps", 0), 2),
            "P95 Slip (bps)": round(s.get("p95_slippage_bps", 0), 2),
            "Fill Ratio %": round(s.get("fill_ratio", 1.0) * 100, 1),
            "TWAP %": round(s.get("twap_usage_pct", 0) * 100, 1),
            "Trades": s.get("trade_count", 0),
        })

    try:
        import pandas as pd
        df = pd.DataFrame(rows).sort_values("Avg Slip (bps)", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)
    except ImportError:
        for r in rows:
            st.text(f"{r['Symbol']:12s}  slip={r['Avg Slip (bps)']:6.2f}  fill={r['Fill Ratio %']:5.1f}%  trades={r['Trades']}")

    # ── Portfolio summary ─────────────────────────────────────────────────
    if symbols:
        n = len(symbols)
        avg_slip = sum(s.get("avg_slippage_bps", 0) for s in symbols.values()) / n
        avg_fill = sum(s.get("fill_ratio", 1.0) for s in symbols.values()) / n
        sc1, sc2 = st.columns(2)
        with sc1:
            st.metric("Portfolio Avg Slippage", f"{avg_slip:.2f} bps")
        with sc2:
            st.metric("Portfolio Fill Ratio", f"{avg_fill * 100:.1f}%")

    if updated_ts:
        st.caption(f"Last updated: {updated_ts}")
