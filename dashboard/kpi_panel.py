"""v7 KPI panel utilities for the dashboard Overview page."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import streamlit as st

from dashboard.nav_helpers import safe_float, safe_format

st.markdown(
    """
    <style>
    .stMetric, .css-1ht1j8u, .css-16idsys, [data-testid="stMetricValue"] {
        overflow: visible !important;
        text-overflow: initial !important;
        white-space: nowrap !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

STATE_DIR = Path(os.getenv("STATE_DIR") or "logs/state")
KPI_V7_STATE_PATH = Path(os.getenv("KPI_V7_STATE_PATH") or (STATE_DIR / "kpis_v7.json"))


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists() or path.stat().st_size <= 0:
            return {}
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def load_kpis(state_dir: str | None = None) -> Dict[str, Any]:
    base_dir = Path(state_dir) if state_dir else STATE_DIR
    path = Path(os.getenv("KPI_V7_STATE_PATH") or (base_dir / "kpis_v7.json"))
    return _load_json(path)


def _metric_value(value: Any, fmt: str = "auto") -> str:
    if value is None:
        return "–"
    try:
        if fmt == "percent":
            return f"{float(value):.2%}"
        if fmt == "float":
            return f"{float(value):.2f}"
    except Exception:
        return str(value)
    return str(value)


def render_kpis_overview(kpis: Dict[str, Any]) -> None:
    if not isinstance(kpis, dict):
        kpis = {}
    
    # Extract values from the kpis structure
    risk_block = kpis.get("risk") or {}
    router_block = kpis.get("router") or {}
    atr_block = kpis.get("atr") or {}
    dd_block = kpis.get("drawdown") or {}
    fee_pnl = kpis.get("fee_pnl") or {}
    
    dd_state = risk_block.get("dd_state") or "normal"
    atr_regime = atr_block.get("atr_regime") or "unknown"
    fee_ratio = safe_float(kpis.get("fee_pnl_ratio") or fee_pnl.get("fee_pnl_ratio"))
    router_quality = router_block.get("quality") or router_block.get("policy_quality") or "n/a"
    
    maker_fill_ratio = safe_float(router_block.get("maker_fill_ratio") or router_block.get("maker_fill_rate"))
    fallback_ratio = safe_float(router_block.get("fallback_ratio"))
    drawdown_pct = safe_float(dd_block.get("dd_pct") or risk_block.get("dd_pct"))
    atr_ratio = safe_float(atr_block.get("median_ratio"))
    
    dd_label = dd_state
    if drawdown_pct is not None:
        try:
            dd_pct_fmt = safe_format(drawdown_pct, nd=2)
            dd_label = f"{dd_state} ({dd_pct_fmt}%)"
        except Exception:
            pass

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Drawdown", dd_label)
    col2.metric("ATR Regime", atr_regime or "–")
    col3.metric("Fee / PnL", safe_format(fee_ratio, nd=3) if fee_ratio is not None else "–")
    col4.metric("Router Quality", router_quality or "–")

    cols2 = st.columns(2)
    cols2[0].metric("Maker Fill Ratio", safe_format(maker_fill_ratio, nd=2) if maker_fill_ratio is not None else "–")
    cols2[1].metric("Fallback Ratio", safe_format(fallback_ratio, nd=2) if fallback_ratio is not None else "–")
    
    if atr_ratio is not None:
        st.caption(f"ATR Ratio (median): {safe_format(atr_ratio, nd=3)}")
