"""v6 pipeline parity view."""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import streamlit as st


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _classify_parity(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map v6 compare summary into a discrete status.

    Inputs (from pipeline_v6_compare_summary.json):
      - sample_size
      - min_sample_size
      - is_warmup
      - veto_mismatch_pct
      - size_diff_stats.{mean,p50,p95}
      - sizing_diff_stats.{p50,p95,upsize_count,sample_size}
    """
    sample_size = int(summary.get("sample_size") or 0)
    min_sample_size = int(summary.get("min_sample_size") or 0)
    is_warmup = bool(summary.get("is_warmup"))
    warmup_reason = summary.get("warmup_reason")

    sizing = summary.get("sizing_diff_stats") or {}
    p95 = abs(_safe_float(sizing.get("p95")))
    upsize_count = int(sizing.get("upsize_count") or 0)

    if is_warmup or sample_size < max(1, min_sample_size):
        return {
            "level": "warmup",
            "label": "WARMUP",
            "color": "#9ca3af",
            "reason": warmup_reason
            or f"Shadow compare warming up (sample={sample_size}, min={min_sample_size}).",
        }

    if upsize_count > 0 or p95 > 0.10:
        return {
            "level": "red",
            "label": "RED",
            "color": "#b91c1c",
            "reason": f"Upsized trades detected (count={upsize_count}) or p95 sizing diff={p95:.3f} > 0.10.",
        }

    if p95 > 0.05:
        return {
            "level": "amber",
            "label": "AMBER",
            "color": "#d97706",
            "reason": f"Sizing p95={p95:.3f} in [0.05, 0.10] — check parity before enabling v6.",
        }

    return {
        "level": "green",
        "label": "GREEN",
        "color": "#16a34a",
        "reason": f"Sizing p95={p95:.3f} and no upsizes detected.",
    }


def render_pipeline_parity(
    shadow_head: Dict[str, Any],
    compare_summary: Dict[str, Any],
) -> None:
    """
    Render the v6 pipeline shadow compare status in the Overview tab.
    """
    st.subheader("Pipeline v6 — Shadow Compare")

    if not isinstance(compare_summary, dict) or not compare_summary:
        st.caption("No pipeline v6 compare summary yet.")
        return

    status = _classify_parity(compare_summary)

    color = status["color"]
    label = status["label"]
    reason = status["reason"]

    st.markdown(
        f"""
        <div style="
            padding:0.6rem 0.9rem;
            border-radius:0.75rem;
            border:1px solid rgba(15,23,42,0.08);
            background:rgba(15,23,42,0.02);
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:0.75rem;
        ">
          <div style="font-weight:600;">
             Pipeline parity status:
             <span style="
                 display:inline-flex;
                 align-items:center;
                 justify-content:center;
                 padding:0.15rem 0.6rem;
                 border-radius:999px;
                 background:{color};
                 color:#f9fafb;
                 font-size:0.85rem;
                 font-weight:700;
                 margin-left:0.35rem;
             ">{label}</span>
          </div>
          <div style="font-size:0.85rem;color:#4b5563;max-width:520px;">
             {reason}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    basics = {
        "sample_size": int(compare_summary.get("sample_size") or 0),
        "min_sample_size": int(compare_summary.get("min_sample_size") or 0),
        "veto_mismatch_pct": _safe_float(compare_summary.get("veto_mismatch_pct")),
    }
    size_diff = compare_summary.get("size_diff_stats") or {}
    sizing_diff = compare_summary.get("sizing_diff_stats") or {}
    slip_diff = compare_summary.get("slippage_diff_bps") or {}

    df_rows = [
        {
            "metric": "sample_size",
            "value": basics["sample_size"],
        },
        {
            "metric": "min_sample_size",
            "value": basics["min_sample_size"],
        },
        {
            "metric": "veto_mismatch_pct",
            "value": f"{basics['veto_mismatch_pct']:.2f}",
        },
        {
            "metric": "size_diff_p95",
            "value": f"{_safe_float(size_diff.get('p95')):.4f}",
        },
        {
            "metric": "sizing_diff_p95",
            "value": f"{_safe_float(sizing_diff.get('p95')):.4f}",
        },
        {
            "metric": "sizing_upsize_count",
            "value": int(sizing_diff.get("upsize_count") or 0),
        },
        {
            "metric": "slippage_diff_p95_bps",
            "value": f"{_safe_float(slip_diff.get('p95')):.2f}",
        },
    ]
    st.table(pd.DataFrame(df_rows))

    if shadow_head:
        st.caption("Pipeline shadow head (raw)")
        st.json(shadow_head, expanded=False)


__all__ = ["render_pipeline_parity"]
