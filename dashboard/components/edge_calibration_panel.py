"""
Edge Calibration Panel — Predicted Edge vs Realized Return chart.

Live dashboard view of conviction model calibration.
Reads episode_ledger.json, computes per-trade predicted edge vs realized return,
renders scatter + bucket calibration using altair (already in the venv).

Data source: logs/state/episode_ledger.json + logs/state/edge_calibration.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

_LEDGER_PATH = Path("logs/state/episode_ledger.json")
_ERR_PATH = Path("logs/state/edge_calibration.json")

try:
    import altair as alt
    import pandas as pd
    _HAS_ALTAIR = True
except ImportError:
    _HAS_ALTAIR = False


def _safe_float(val: Any) -> float:
    if val is None:
        return 0.0
    try:
        v = float(val)
        return v if math.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _load_calibration_points(episodes: List[Dict]) -> List[Dict[str, float]]:
    """Extract (predicted_edge, realized_return) pairs from episodes."""
    points = []
    for ep in episodes:
        edge = _safe_float(ep.get("expected_edge"))
        if not (edge > 0 and math.isfinite(edge)):
            conv = _safe_float(ep.get("conviction_score"))
            edge = max(0.0, conv - 0.5)
        if edge <= 0:
            continue

        entry_px = _safe_float(ep.get("avg_entry_price"))
        exit_px = _safe_float(ep.get("avg_exit_price"))
        if entry_px <= 0 or exit_px <= 0:
            continue

        side = str(ep.get("side", "")).upper()
        if side == "LONG":
            realized = (exit_px - entry_px) / entry_px
        elif side == "SHORT":
            realized = (entry_px - exit_px) / entry_px
        else:
            continue

        points.append({
            "predicted_edge": edge,
            "realized_return": realized,
            "symbol": ep.get("symbol", "?"),
            "side": side,
        })
    return points


def _bucket_data(points: List[Dict], n_buckets: int = 5) -> List[Dict]:
    """Group points into predicted-edge buckets with mean realized return."""
    if not points:
        return []
    sorted_pts = sorted(points, key=lambda p: p["predicted_edge"])
    bucket_size = max(1, len(sorted_pts) // n_buckets)
    buckets = []
    for i in range(0, len(sorted_pts), bucket_size):
        chunk = sorted_pts[i : i + bucket_size]
        pred_vals = [p["predicted_edge"] for p in chunk]
        real_vals = [p["realized_return"] for p in chunk]
        lo, hi = min(pred_vals), max(pred_vals)
        buckets.append({
            "bucket": f"{lo:.1%}–{hi:.1%}",
            "predicted_edge": sum(pred_vals) / len(pred_vals),
            "realized_return": sum(real_vals) / len(real_vals),
            "count": len(chunk),
        })
    return buckets


def render_edge_calibration_panel(episode_ledger: Optional[Dict] = None) -> None:
    """Render edge calibration scatter + bucket chart."""
    if not _HAS_ALTAIR:
        return

    # Load episodes
    if episode_ledger is None:
        try:
            episode_ledger = json.loads(_LEDGER_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            return
    episodes = episode_ledger.get("episodes", [])
    if not episodes:
        return

    points = _load_calibration_points(episodes)
    if len(points) < 5:
        return  # not enough data to chart

    # Load ERR summary
    err_val = None
    try:
        ec = json.loads(_ERR_PATH.read_text())
        err_val = ec.get("err")
    except (OSError, json.JSONDecodeError):
        pass

    err_label = f"{err_val:.2f}" if err_val is not None else "—"
    err_color = (
        "🟢" if err_val is not None and 0.8 <= err_val <= 1.2
        else "🟡" if err_val is not None and 0.6 <= err_val <= 1.4
        else "🔴"
    ) if err_val is not None else "⚪"

    st.markdown(
        f"#### Edge Calibration &nbsp;&nbsp;"
        f"<span style='font-size:0.85rem;'>{err_color} ERR = {err_label} (n={len(points)})</span>",
        unsafe_allow_html=True,
    )

    df = pd.DataFrame(points)
    max_edge = df["predicted_edge"].max()

    # Ideal line data
    ideal_df = pd.DataFrame({
        "predicted_edge": [0, max_edge],
        "realized_return": [0, max_edge],
    })

    col1, col2 = st.columns(2)

    with col1:
        # ── Scatter: predicted vs realized ──────────────────────
        scatter = (
            alt.Chart(df)
            .mark_circle(size=40, opacity=0.5)
            .encode(
                x=alt.X("predicted_edge:Q", title="Predicted Edge", scale=alt.Scale(domain=[0, max_edge * 1.1])),
                y=alt.Y("realized_return:Q", title="Realized Return"),
                color=alt.Color("side:N", scale=alt.Scale(domain=["LONG", "SHORT"], range=["#21ba45", "#db2828"])),
                tooltip=["symbol:N", "side:N",
                         alt.Tooltip("predicted_edge:Q", format=".4f"),
                         alt.Tooltip("realized_return:Q", format=".4f")],
            )
        )
        ideal_line = (
            alt.Chart(ideal_df)
            .mark_line(strokeDash=[5, 5], color="#888")
            .encode(x="predicted_edge:Q", y="realized_return:Q")
        )
        zero_line = (
            alt.Chart(pd.DataFrame({"y": [0]}))
            .mark_rule(color="#555", strokeWidth=0.5)
            .encode(y="y:Q")
        )
        chart = (scatter + ideal_line + zero_line).properties(
            height=280, title="Predicted Edge vs Realized Return"
        )
        st.altair_chart(chart, use_container_width=True)

    with col2:
        # ── Bucket calibration ──────────────────────────────────
        buckets = _bucket_data(points)
        if buckets:
            bdf = pd.DataFrame(buckets)
            bars = (
                alt.Chart(bdf)
                .mark_bar(opacity=0.7)
                .encode(
                    x=alt.X("bucket:N", title="Predicted Edge Bucket", sort=None),
                    y=alt.Y("realized_return:Q", title="Mean Realized Return"),
                    color=alt.condition(
                        alt.datum.realized_return > 0,
                        alt.value("#21ba45"),
                        alt.value("#db2828"),
                    ),
                    tooltip=[
                        "bucket:N",
                        alt.Tooltip("predicted_edge:Q", title="Avg Predicted", format=".4f"),
                        alt.Tooltip("realized_return:Q", title="Avg Realized", format=".4f"),
                        "count:Q",
                    ],
                )
            )
            ideal_dots = (
                alt.Chart(bdf)
                .mark_point(color="#888", size=30, shape="diamond")
                .encode(
                    x=alt.X("bucket:N", sort=None),
                    y=alt.Y("predicted_edge:Q"),
                    tooltip=[alt.Tooltip("predicted_edge:Q", title="Ideal", format=".4f")],
                )
            )
            bucket_chart = (bars + ideal_dots).properties(
                height=280, title="Bucket Calibration"
            )
            st.altair_chart(bucket_chart, use_container_width=True)
