"""
Alpha Decay Widget — Position Age Penalty Surface

Shows whether alpha decay is active and, when populated,
per-symbol decay multipliers that reduce sizing on stale signals.

Data source: logs/state/alpha_decay.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STATE_PATH = Path("logs/state/alpha_decay.json")


# ---------------------------------------------------------------------------
# State Loader
# ---------------------------------------------------------------------------

def load_alpha_decay_state(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load alpha decay state."""
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

def render_alpha_decay_widget(state: Dict[str, Any]) -> None:
    """Render alpha decay status and per-symbol multipliers."""
    st.header("Alpha Decay")

    if not state:
        st.info("Alpha decay state not available.")
        return

    config = state.get("config", {})
    symbols = state.get("symbols", {})
    enabled = config.get("enabled", False)

    # ── Config strip ──────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Status", "Enabled" if enabled else "Disabled")
    with c2:
        half_life = config.get("half_life_minutes", "—")
        st.metric("Half-Life", f"{half_life} min" if half_life != "—" else "—")
    with c3:
        floor = config.get("min_decay_multiplier", "—")
        if isinstance(floor, (int, float)):
            st.metric("Min Multiplier", f"{floor:.2f}")
        else:
            st.metric("Min Multiplier", "—")

    if not enabled:
        st.info("Alpha decay is currently disabled. Signals are not penalized for age.")
        return

    # ── Per-symbol multipliers ────────────────────────────────────────────
    if not symbols:
        st.info("No active decay entries. Multipliers populate when positions have aging signals.")
        return

    st.subheader("Per-Symbol Decay")
    rows = []
    for sym, data in sorted(symbols.items()):
        if isinstance(data, dict):
            mult = data.get("multiplier", 1.0)
            age_min = data.get("age_minutes", 0)
        else:
            mult = float(data) if data else 1.0
            age_min = 0

        rows.append({
            "Symbol": sym,
            "Multiplier": round(mult, 3),
            "Age (min)": round(age_min, 1),
            "Penalty": f"{(1 - mult) * 100:.1f}%",
        })

    try:
        import pandas as pd
        df = pd.DataFrame(rows).sort_values("Multiplier", ascending=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
    except ImportError:
        for r in rows:
            st.text(f"{r['Symbol']:12s}  mult={r['Multiplier']:.3f}  age={r['Age (min)']:.0f}min")

    # Highlight severe decay
    severe = [r for r in rows if r["Multiplier"] < 0.5]
    if severe:
        names = ", ".join(r["Symbol"] for r in severe)
        st.warning(f"Severe decay (<50% multiplier): {names}")
