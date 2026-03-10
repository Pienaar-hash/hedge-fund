"""
Engine Lift Panel — Hydra vs Legacy outcome comparison.

Shows outcome-based CEL (Conditional Edge Lift) computed from
realized returns on trades selected by each engine.

Data source: logs/state/engine_lift.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

_STATE_PATH = Path("logs/state/engine_lift.json")

try:
    import altair as alt
    import pandas as pd
    _HAS_ALTAIR = True
except ImportError:
    _HAS_ALTAIR = False


def _load_engine_lift() -> Dict[str, Any]:
    try:
        return json.loads(_STATE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _cel_color(v: Optional[float]) -> str:
    if v is None:
        return "#666"
    if v > 0.001:
        return "#21ba45"
    if v >= -0.001:
        return "#f2c037"
    return "#db2828"


def render_engine_lift_panel(lift_data: Optional[Dict[str, Any]] = None) -> None:
    """Render Hydra vs Legacy comparison panel."""
    data = lift_data or _load_engine_lift()
    if not data:
        return

    h_count = data.get("hydra_count", 0)
    l_count = data.get("legacy_count", 0)
    if h_count + l_count < 10:
        return  # not enough data

    h_mean = data.get("hydra_mean_return", 0)
    l_mean = data.get("legacy_mean_return", 0)
    cel = data.get("outcome_cel")
    plift = data.get("participation_lift")
    h_pnl = data.get("hydra_pnl_sum", 0)
    l_pnl = data.get("legacy_pnl_sum", 0)

    cel_str = f"{cel * 100:+.3f}%" if cel is not None else "—"
    color = _cel_color(cel)

    st.markdown(
        f"#### Hydra vs Legacy &nbsp;&nbsp;"
        f"<span style='font-size:0.85rem; color:{color};'>Outcome CEL = {cel_str}</span>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        # Summary metrics
        html = f"""
        <div style="background:#1a1a2e;border:1px solid #2a2a4a;border-radius:6px;
                    padding:12px 16px;font-family:monospace;font-size:0.82rem;">
          <div style="display:flex;gap:8px;margin-bottom:8px;">
            <span style="color:#21ba45;font-weight:600;width:120px;">Hydra</span>
            <span style="color:#ccc;">{h_mean*100:+.4f}% avg</span>
            <span style="color:#888;">n={h_count}</span>
            <span style="color:#888;">${h_pnl:+.2f}</span>
          </div>
          <div style="display:flex;gap:8px;margin-bottom:8px;">
            <span style="color:#f2c037;font-weight:600;width:120px;">Legacy</span>
            <span style="color:#ccc;">{l_mean*100:+.4f}% avg</span>
            <span style="color:#888;">n={l_count}</span>
            <span style="color:#888;">${l_pnl:+.2f}</span>
          </div>
          <div style="border-top:1px solid #333;padding-top:6px;display:flex;gap:16px;">
            <span>
              <span style="color:#aaa;">CEL</span>&nbsp;
              <span style="color:{color};font-weight:700;">{cel_str}</span>
            </span>
            <span>
              <span style="color:#aaa;">Participation</span>&nbsp;
              <span style="color:#ccc;">{f'{plift:.2f}x' if plift is not None else '—'}</span>
            </span>
          </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

    with col2:
        if _HAS_ALTAIR:
            # Bar chart: mean return per engine
            df = pd.DataFrame([
                {"engine": "Hydra", "mean_return": h_mean * 100, "count": h_count},
                {"engine": "Legacy", "mean_return": l_mean * 100, "count": l_count},
            ])
            bars = (
                alt.Chart(df)
                .mark_bar(opacity=0.8, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("engine:N", title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("mean_return:Q", title="Mean Return (%)"),
                    color=alt.Color(
                        "engine:N",
                        scale=alt.Scale(domain=["Hydra", "Legacy"], range=["#21ba45", "#f2c037"]),
                        legend=None,
                    ),
                    tooltip=[
                        "engine:N",
                        alt.Tooltip("mean_return:Q", format="+.4f", title="Mean Return %"),
                        "count:Q",
                    ],
                )
            )
            text = (
                alt.Chart(df)
                .mark_text(dy=-10, fontSize=11, color="#ccc")
                .encode(
                    x="engine:N",
                    y="mean_return:Q",
                    text=alt.Text("mean_return:Q", format="+.3f"),
                )
            )
            zero = (
                alt.Chart(pd.DataFrame({"y": [0]}))
                .mark_rule(color="#555", strokeWidth=0.5)
                .encode(y="y:Q")
            )
            chart = (bars + text + zero).properties(height=200, title="Mean Return per Engine (%)")
            st.altair_chart(chart, use_container_width=True)
