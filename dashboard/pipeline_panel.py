from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st


def _badge(label: str, color: str) -> str:
    return (
        f'<span style="display:inline-block;padding:0.25em 0.7em;'
        f'border-radius:0.6em;font-weight:700;color:#fff;background:{color};">{label}</span>'
    )


def _warmup_badge(sample_size: Optional[float]) -> str:
    if sample_size is None or sample_size < 50:
        return _badge("WARMUP", "#f2c037")
    return _badge("STEADY STATE", "#21ba45")


def _fmt(value: Any, suffix: str = "", precision: int = 2) -> str:
    try:
        num = float(value)
    except Exception:
        return "n/a"
    fmt = f"{{:,.{precision}f}}{suffix}"
    return fmt.format(num)


def render_pipeline_parity(shadow: Optional[Dict[str, Any]], compare: Optional[Dict[str, Any]]) -> None:
    st.markdown("### Pipeline Parity (v6)")

    shadow = shadow or {}
    compare = compare or {}

    sample_size = compare.get("sample_size") if isinstance(compare, dict) else None
    badge_html = _warmup_badge(sample_size if isinstance(sample_size, (int, float)) else None)
    st.markdown(badge_html, unsafe_allow_html=True)

    if not shadow and not compare:
        st.info("No data yet.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Shadow Head")
        last_decision = shadow.get("last_decision") if isinstance(shadow, dict) else None
        total = shadow.get("total") if isinstance(shadow, dict) else None
        allowed = shadow.get("allowed") if isinstance(shadow, dict) else None
        vetoed = shadow.get("vetoed") if isinstance(shadow, dict) else None
        st.metric("Last Decision", last_decision or "n/a")
        st.metric("Total", int(total) if isinstance(total, (int, float)) else "n/a")
        st.metric("Allowed", int(allowed) if isinstance(allowed, (int, float)) else "n/a")
        st.metric("Vetoed", int(vetoed) if isinstance(vetoed, (int, float)) else "n/a")

    with col2:
        st.markdown("#### Compare Summary")
        st.metric("Sample Size", int(sample_size) if isinstance(sample_size, (int, float)) else "n/a")
        veto_mismatch = compare.get("veto_mismatch_pct") if isinstance(compare, dict) else None
        st.metric("Veto Mismatch %", _fmt(veto_mismatch, suffix="%"))

        size_diff = compare.get("size_diff_stats") if isinstance(compare.get("size_diff_stats"), dict) else {}
        slip_diff = compare.get("slippage_diff_bps") if isinstance(compare.get("slippage_diff_bps"), dict) else {}

        st.caption("Size Diff (bps)")
        st.write(
            f"p50: {_fmt(size_diff.get('p50'))}, p95: {_fmt(size_diff.get('p95'))}, mean: {_fmt(size_diff.get('mean'))}"
        )

        st.caption("Slippage Diff (bps)")
        st.write(
            f"mean: {_fmt(slip_diff.get('mean'))}, p50: {_fmt(slip_diff.get('p50'))}, p95: {_fmt(slip_diff.get('p95'))}"
        )


__all__ = ["render_pipeline_parity"]
