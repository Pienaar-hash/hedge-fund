"""v6 intel panel."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


def _clamp_01(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except Exception:
        return None
    if v != v:  # NaN
        return None
    return max(0.0, min(1.0, v))


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def render_intel_panel(
    expectancy_v6: Dict[str, Any],
    symbol_scores_v6: Dict[str, Any],
    risk_allocator_v6: Dict[str, Any],
) -> None:
    """
    v6 Intel panel with normalized/clamped scores.
    """
    st.subheader("Intel v6 — Symbol Scores & Expectancy")

    scores_list = symbol_scores_v6.get("symbols") if isinstance(symbol_scores_v6, dict) else None
    if not isinstance(scores_list, list):
        scores_list = []

    exp_map = expectancy_v6.get("symbols") if isinstance(expectancy_v6, dict) else {}
    if not isinstance(exp_map, dict):
        exp_map = {}

    rows: List[Dict[str, Any]] = []
    for entry in scores_list:
        if not isinstance(entry, dict):
            continue
        symbol = entry.get("symbol")
        if not symbol:
            continue
        sym = str(symbol).upper()

        exp_entry = exp_map.get(sym) if isinstance(exp_map.get(sym), dict) else {}
        expectancy_raw = (exp_entry or {}).get("expectancy")
        hit_raw = (exp_entry or {}).get("hit_rate")
        dd_raw = (exp_entry or {}).get("max_drawdown")

        rows.append(
            {
                "symbol": sym,
                "score_raw": _safe_float(entry.get("score")),
                "score": _clamp_01(entry.get("score")),
                "expectancy_raw": _safe_float(expectancy_raw),
                "expectancy": _clamp_01(expectancy_raw),
                "hit_rate_raw": _safe_float(hit_raw),
                "hit_rate": _clamp_01(hit_raw),
                "max_drawdown": _safe_float(dd_raw),
            }
        )

    if not rows:
        st.info("No v6 intel snapshots yet — wait for INTEL_V6 to publish.")
        return

    df = pd.DataFrame(rows)
    df.sort_values("score", ascending=False, inplace=True, ignore_index=True)

    display_cols = [
        "symbol",
        "score",
        "expectancy",
        "hit_rate",
        "max_drawdown",
        "score_raw",
        "expectancy_raw",
        "hit_rate_raw",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    st.markdown("### Top Symbols (normalized to 0–1)")
    st.dataframe(df[display_cols].head(50), use_container_width=True, height=360)

    st.markdown("---")
    st.subheader("Risk Allocator v6 — Suggestions")

    if not isinstance(risk_allocator_v6, dict) or not risk_allocator_v6:
        st.caption("No v6 allocator suggestions yet.")
        return

    alloc_symbols = risk_allocator_v6.get("symbols")
    allocations = risk_allocator_v6.get("allocations")
    generated_ts = risk_allocator_v6.get("generated_ts") or risk_allocator_v6.get("ts")

    if isinstance(alloc_symbols, list):
        alloc_rows: List[Dict[str, Any]] = []
        for entry in alloc_symbols:
            if not isinstance(entry, dict):
                continue
            sym = str(entry.get("symbol") or "").upper()
            if not sym:
                continue
            row = {"symbol": sym}
            for key in ("weight", "target_notional", "max_exposure_pct", "dd_state"):
                if key in entry:
                    row[key] = entry.get(key)
            alloc_rows.append(row)
        if alloc_rows:
            st.dataframe(pd.DataFrame(alloc_rows), use_container_width=True, height=260)
        else:
            st.json(risk_allocator_v6)
    elif isinstance(allocations, dict):
        st.json(allocations)
    else:
        st.json(risk_allocator_v6)

    if generated_ts:
        st.caption(f"Allocator snapshot ts={generated_ts}")


__all__ = ["render_intel_panel"]
