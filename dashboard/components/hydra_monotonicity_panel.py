"""
Hydra Score Monotonicity Panel — Does higher score → higher return?

Bar chart of Hydra score buckets vs mean realized return,
plus Spearman rank correlation summary.

Data source: logs/state/hydra_monotonicity.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

_STATE_PATH = Path("logs/state/hydra_monotonicity.json")
_EPISODE_PATH = Path("logs/state/episode_ledger.json")

try:
    import altair as alt
    import pandas as pd
    _HAS_ALTAIR = True
except ImportError:
    _HAS_ALTAIR = False


def _load_monotonicity() -> Dict[str, Any]:
    try:
        return json.loads(_STATE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _slope_color(slope: str) -> str:
    if slope == "upward":
        return "#21ba45"
    if slope == "flat":
        return "#f2c037"
    if slope == "inverted":
        return "#db2828"
    return "#666"


def _spearman_color(v: Optional[float]) -> str:
    if v is None:
        return "#666"
    if v > 0.2:
        return "#21ba45"
    if v >= 0:
        return "#f2c037"
    return "#db2828"


def render_hydra_monotonicity_panel(
    data: Optional[Dict[str, Any]] = None,
) -> None:
    """Render Hydra score monotonicity bar chart + Spearman summary."""
    data = data or _load_monotonicity()
    if not data:
        return

    buckets = data.get("buckets", [])
    spearman = data.get("spearman")
    slope = data.get("slope", "unknown")
    n = data.get("n", 0)

    if n < 5 or not buckets:
        return

    sp_str = f"{spearman:+.3f}" if spearman is not None else "—"
    sp_color = _spearman_color(spearman)
    sl_color = _slope_color(slope)

    st.markdown(
        f"#### Hydra Score Monotonicity &nbsp;&nbsp;"
        f"<span style='font-size:0.85rem; color:{sp_color};'>"
        f"Spearman ρ = {sp_str}</span>"
        f"&nbsp;&nbsp;<span style='font-size:0.75rem; color:{sl_color};'>"
        f"({slope})</span>"
        f"&nbsp;&nbsp;<span style='color:#666;font-size:0.7rem;'>(n={n})</span>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        # Text table
        rows = ""
        for b in buckets:
            ret = b["mean_return"]
            pct = f"{ret * 100:+.3f}%"
            bar_len = max(0, min(20, int(abs(ret) * 2000)))
            bar_char = "█" if ret >= 0 else "░"
            color = "#21ba45" if ret >= 0 else "#db2828"
            rows += (
                f'<div style="display:flex;gap:8px;margin-bottom:3px;">'
                f'<span style="color:#aaa;width:100px;font-size:0.78rem;">{b["range"]}</span>'
                f'<span style="color:{color};width:170px;font-size:0.78rem;">'
                f'{bar_char * bar_len} {pct}</span>'
                f'<span style="color:#666;font-size:0.68rem;">n={b["n"]}</span>'
                f'</div>'
            )

        html = f"""
        <div style="background:#1a1a2e;border:1px solid #2a2a4a;border-radius:6px;
                    padding:12px 16px;font-family:monospace;">
          <div style="color:#888;font-size:0.7rem;margin-bottom:8px;">
            Score Bucket → Average Return
          </div>
          {rows}
          <div style="border-top:1px solid #333;padding-top:8px;margin-top:8px;
                      display:flex;gap:16px;">
            <span>
              <span style="color:#aaa;font-size:0.75rem;">Spearman ρ</span>&nbsp;
              <span style="color:{sp_color};font-weight:700;">{sp_str}</span>
            </span>
            <span>
              <span style="color:#aaa;font-size:0.75rem;">Slope</span>&nbsp;
              <span style="color:{sl_color};font-weight:600;">{slope}</span>
            </span>
          </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

    with col2:
        if not _HAS_ALTAIR:
            return

        df = pd.DataFrame(buckets)
        df["return_pct"] = df["mean_return"] * 100
        df["bar_color"] = df["mean_return"].apply(
            lambda v: "#21ba45" if v >= 0 else "#db2828"
        )

        chart = (
            alt.Chart(df)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("range:N", sort=None, title="Hydra Score Bucket"),
                y=alt.Y("return_pct:Q", title="Avg Return (%)"),
                color=alt.Color(
                    "bar_color:N",
                    scale=None,
                ),
                tooltip=[
                    alt.Tooltip("range:N", title="Score Range"),
                    alt.Tooltip("return_pct:Q", title="Avg Return %", format="+.3f"),
                    alt.Tooltip("n:Q", title="Count"),
                    alt.Tooltip("mean_score:Q", title="Mean Score", format=".4f"),
                ],
            )
            .properties(height=250, title="Score → Return (monotonicity check)")
        )

        st.altair_chart(chart, use_container_width=True)

    # --- Quintile Return Spread (Q5 - Q1) ---
    quintiles = data.get("quintiles", [])
    q5_q1 = data.get("q5_q1_spread")
    if quintiles and q5_q1 is not None:
        q5_q1_pct = q5_q1 * 100
        if q5_q1_pct > 0.40:
            q_color = "#21ba45"
            q_label = "strong separation"
        elif q5_q1_pct > 0.15:
            q_color = "#21ba45"
            q_label = "useful signal"
        elif q5_q1_pct >= 0:
            q_color = "#f2c037"
            q_label = "weak / marginal"
        else:
            q_color = "#db2828"
            q_label = "inverted"

        q_rows = ""
        for q in quintiles:
            ret_pct = q["mean_return"] * 100
            bar_len = max(0, min(20, int(abs(ret_pct) * 40)))
            bar_char = "\u2588" if ret_pct >= 0 else "\u2591"
            color = "#21ba45" if ret_pct >= 0 else "#db2828"
            q_rows += (
                f'<div style="display:flex;gap:8px;margin-bottom:3px;">'
                f'<span style="color:#aaa;width:30px;font-weight:600;font-size:0.78rem;">{q["label"]}</span>'
                f'<span style="color:{color};width:180px;font-size:0.78rem;">'
                f'{bar_char * bar_len} {ret_pct:+.3f}%</span>'
                f'<span style="color:#666;font-size:0.68rem;">n={q["n"]}</span>'
                f'</div>'
            )

        q_html = f"""
        <div style="background:#1a1a2e;border:1px solid #2a2a4a;border-radius:6px;
                    padding:12px 16px;font-family:monospace;margin-top:12px;">
          <div style="color:#888;font-size:0.7rem;margin-bottom:8px;">
            Hydra Quintile Spread
          </div>
          {q_rows}
          <div style="border-top:1px solid #333;padding-top:8px;margin-top:8px;
                      display:flex;gap:16px;">
            <span>
              <span style="color:#aaa;font-size:0.75rem;">Q5 \u2212 Q1</span>&nbsp;
              <span style="color:{q_color};font-weight:700;">{q5_q1_pct:+.2f}%</span>
            </span>
            <span>
              <span style="color:#aaa;font-size:0.75rem;">Signal</span>&nbsp;
              <span style="color:{q_color};font-weight:600;">{q_label}</span>
            </span>
          </div>
        </div>
        """
        st.markdown(q_html, unsafe_allow_html=True)

    # --- Score vs Return scatter with quintile rails ---
    if _HAS_ALTAIR and quintiles and len(quintiles) == 5:
        try:
            ep_data = json.loads(_EPISODE_PATH.read_text())
            scored = []
            for ep in ep_data.get("episodes", []):
                sc = float(ep.get("hybrid_score") or 0)
                if sc <= 0:
                    continue
                entry = float(ep.get("avg_entry_price") or 0)
                exit_ = float(ep.get("avg_exit_price") or 0)
                if entry <= 0 or exit_ <= 0:
                    continue
                side = str(ep.get("side", "")).upper()
                if side == "LONG":
                    ret = (exit_ - entry) / entry * 100
                elif side == "SHORT":
                    ret = (entry - exit_) / entry * 100
                else:
                    continue
                head = str(ep.get("regime_at_entry", "") or "").upper()
                scored.append({"hybrid_score": sc, "realized_return": round(ret, 4), "head": head})
            if len(scored) >= 10:
                df_sc = pd.DataFrame(scored)
                scatter = (
                    alt.Chart(df_sc)
                    .mark_circle(size=40, opacity=0.6)
                    .encode(
                        x=alt.X("hybrid_score:Q", title="Hydra Score"),
                        y=alt.Y("realized_return:Q", title="Realized Return (%)"),
                        color=alt.condition(
                            "datum.realized_return > 0",
                            alt.value("#2ecc71"),
                            alt.value("#e74c3c"),
                        ),
                        tooltip=[
                            alt.Tooltip("hybrid_score:Q", title="Score", format=".4f"),
                            alt.Tooltip("realized_return:Q", title="Return %", format="+.3f"),
                            alt.Tooltip("head:N", title="Regime"),
                        ],
                    )
                )
                zero_rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#555", strokeDash=[4, 4]).encode(y="y:Q")
                # Quintile boundary rails from bucket edges
                q_bounds = [quintiles[i]["mean_score"] + (quintiles[i + 1]["mean_score"] - quintiles[i]["mean_score"]) / 2 for i in range(4)]
                rails = alt.Chart(pd.DataFrame({"x": q_bounds})).mark_rule(color="#888", strokeDash=[4, 4]).encode(x="x:Q")
                full_chart = (scatter + zero_rule + rails).properties(height=280, title="Score → Return Map")
                st.altair_chart(full_chart, use_container_width=True)
        except (OSError, json.JSONDecodeError, ValueError, KeyError):
            pass

    # Per-head monotonicity table
    per_head = data.get("per_head", [])
    contamination = data.get("head_contamination", False)
    if per_head:
        contam_badge = ""
        if contamination:
            contam_badge = (
                '&nbsp;&nbsp;<span style="background:#db2828;color:#fff;'
                'padding:1px 6px;border-radius:3px;font-weight:600;'
                'font-size:0.7rem;">HEAD CONTAMINATION</span>'
            )

        st.markdown(
            f"##### Per-Head Monotonicity{contam_badge}",
            unsafe_allow_html=True,
        )

        head_rows = ""
        for h in per_head:
            h_sp = h.get("spearman")
            h_slope = h.get("slope", "unknown")
            h_n = h.get("n", 0)
            h_color = _spearman_color(h_sp)
            h_sl_color = _slope_color(h_slope)
            h_sp_str = f"{h_sp:+.3f}" if h_sp is not None else "—"
            head_rows += (
                f'<div style="display:flex;gap:12px;margin-bottom:3px;">'
                f'<span style="color:#ccc;width:110px;font-weight:600;'
                f'font-size:0.78rem;">{h["head"]}</span>'
                f'<span style="color:{h_color};width:70px;font-size:0.78rem;'
                f'font-weight:600;">{h_sp_str}</span>'
                f'<span style="color:{h_sl_color};width:70px;'
                f'font-size:0.78rem;">{h_slope}</span>'
                f'<span style="color:#666;font-size:0.68rem;">n={h_n}</span>'
                f'</div>'
            )
        # Add global row for comparison
        gl_sp_str = f"{spearman:+.3f}" if spearman is not None else "—"
        head_rows += (
            f'<div style="display:flex;gap:12px;margin-top:4px;'
            f'border-top:1px solid #333;padding-top:4px;">'
            f'<span style="color:#888;width:110px;font-weight:600;'
            f'font-size:0.78rem;">ALL</span>'
            f'<span style="color:{sp_color};width:70px;font-size:0.78rem;'
            f'font-weight:600;">{gl_sp_str}</span>'
            f'<span style="color:{sl_color};width:70px;'
            f'font-size:0.78rem;">{slope}</span>'
            f'<span style="color:#666;font-size:0.68rem;">n={n}</span>'
            f'</div>'
        )

        head_html = f"""
        <div style="background:#1a1a2e;border:1px solid #2a2a4a;border-radius:6px;
                    padding:12px 16px;font-family:monospace;">
          <div style="display:flex;gap:12px;margin-bottom:6px;">
            <span style="color:#666;width:110px;font-size:0.68rem;">HEAD</span>
            <span style="color:#666;width:70px;font-size:0.68rem;">SPEARMAN</span>
            <span style="color:#666;width:70px;font-size:0.68rem;">SLOPE</span>
            <span style="color:#666;font-size:0.68rem;">COUNT</span>
          </div>
          {head_rows}
        </div>
        """
        st.markdown(head_html, unsafe_allow_html=True)
