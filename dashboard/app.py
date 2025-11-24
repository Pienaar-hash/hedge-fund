# mypy: ignore-errors
"""Streamlit dashboard for v6 runtime."""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local helpers
from dashboard.dashboard_utils import (
    parse_nav_to_df_and_kpis,
    positions_sorted,
    get_env_badge,
)
from dashboard.nav_helpers import load_nav_state, load_synced_state, nav_state_age_seconds
from dashboard.live_helpers import (
    get_nav_snapshot,
    load_runtime_probe,
    load_expectancy_v6,
    load_symbol_scores_v6,
    load_risk_allocator_v6,
    load_router_policy_v6,
    load_router_health_state,
    load_router_suggestions_v6,
    load_shadow_head,
    load_compare_summary,
    load_kpis_v7,
    execution_kpis,
)
from dashboard.intel_panel import render_intel_panel
from dashboard.pipeline_panel import render_pipeline_parity
from dashboard.router_policy import render_router_policy_panel
from dashboard.router_health import load_router_health, is_empty_router_health

LOG = logging.getLogger("dash.app")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG.setLevel(logging.INFO)


def _env_label() -> str:
    return os.getenv("ENV", os.getenv("HEDGE_ENV", "prod"))


def _flag_badge(enabled: bool) -> str:
    color = "#21ba45" if enabled else "#db2828"
    label = "ON" if enabled else "OFF"
    return (
        f'<span style="display:inline-block;padding:0.2em 0.6em;border-radius:0.6em;'
        f'font-weight:700;color:#fff;background:{color};">{label}</span>'
    )


def _safe_panel(label: str, func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - render safety
        LOG.exception("[dash] %s panel failed", label)
        st.error(f"{label} unavailable: {exc}")
        return None


def _format_age_seconds(age_seconds: Optional[float]) -> str:
    if age_seconds is None:
        return "n/a"
    try:
        age = float(age_seconds)
    except Exception:
        return "n/a"
    if age < 60:
        return f"{age:.1f}s"
    if age < 3600:
        return f"{age/60:.1f}m"
    return f"{age/3600:.1f}h"


def render_runtime_strip(runtime_probe: Dict[str, Any]) -> None:
    if not runtime_probe:
        st.info("Runtime probe unavailable.")
        return
    flags_payload = runtime_probe.get("flags") if isinstance(runtime_probe.get("flags"), dict) else {}

    def _flag_value(key: str, aliases: Tuple[str, ...] = ()) -> bool:
        if key in flags_payload:
            return bool(flags_payload.get(key))
        for alias in aliases:
            if alias in runtime_probe:
                return bool(runtime_probe.get(alias))
        return False

    flags = {
        "INTEL_V6": _flag_value("INTEL_V6", ("intel_v6_enabled",)),
        "RISK_ENGINE_V6": _flag_value("RISK_ENGINE_V6", ("risk_v6_enabled", "risk_engine_v6_enabled")),
        "PIPELINE_V6_SHADOW": _flag_value("PIPELINE_V6_SHADOW", ("pipeline_v6_enabled",)),
        "ROUTER_AUTOTUNE_V6": _flag_value("ROUTER_AUTOTUNE_V6", ("router_autotune_v6_enabled",)),
        "FEEDBACK_ALLOCATOR_V6": _flag_value("FEEDBACK_ALLOCATOR_V6", ("feedback_allocator_v6_enabled",)),
    }
    engine_version = runtime_probe.get("engine_version") or "n/a"
    nav_age_ms = runtime_probe.get("nav_age_ms")
    loop_latency_ms = runtime_probe.get("loop_latency_ms")
    generated_at = runtime_probe.get("generated_at") or runtime_probe.get("ts") or "n/a"

    flags_html = " · ".join([f"{name}: {_flag_badge(bool(value))}" for name, value in flags.items()])
    st.markdown(
        f"""
        <div style="
            display:flex;align-items:center;gap:0.8rem;
            padding:0.6rem 0.9rem;
            background:rgba(15,23,42,0.04);
            border:1px solid rgba(15,23,42,0.08);
            border-radius:10px;
            margin-bottom:0.5rem;
        ">
            <div style="font-weight:700;">Runtime v6 · engine {engine_version}</div>
            <div style="font-weight:600;">{flags_html}</div>
            <div>nav_age_ms: {nav_age_ms if nav_age_ms is not None else 'n/a'}</div>
            <div>loop_latency_ms: {loop_latency_ms if loop_latency_ms is not None else 'n/a'}</div>
            <div>generated_at: {generated_at}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header(nav_doc: Dict[str, Any], nav_source: str, runtime_probe: Dict[str, Any]) -> None:
    nav_df, kpis = parse_nav_to_df_and_kpis(nav_doc or {})
    nav_value = None
    if isinstance(kpis.get("total_equity"), (int, float)):
        nav_value = float(kpis["total_equity"])
    elif not nav_df.empty:
        try:
            nav_value = float(nav_df["equity"].iloc[-1])
        except Exception:
            nav_value = None
    nav_age = nav_state_age_seconds(nav_doc)
    nav_age_display = _format_age_seconds(nav_age)
    env_label, env_color = get_env_badge(False)

    st.markdown(
        f"""
        <div style="display:flex;align-items:center;justify-content:space-between;
                    padding:0.8rem 1rem;border:1px solid #e5e7eb;border-radius:10px;
                    background:linear-gradient(90deg, #f8fafc 0%, #f1f5f9 100%);margin-bottom:0.75rem;">
            <div>
                <div style="font-size:1.6rem;font-weight:700;">NAV: {nav_value:,.2f} USD</div>
                <div style="color:#475569;">Source: {nav_source} · Age: {nav_age_display}</div>
            </div>
            <div style="display:flex;align-items:center;gap:0.5rem;">
                <span style="background:{env_color};color:#fff;padding:0.4rem 0.8rem;border-radius:999px;font-weight:700;">{env_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpis_v7_panel(kpis: Dict[str, Any]) -> None:
    st.subheader("v7 Risk KPIs")
    if not kpis:
        st.info("v7 KPIs unavailable (waiting for executor publish).")
        return
    cols = st.columns(4)
    cols[0].metric("Drawdown State", str(kpis.get("dd_state") or "unknown"))
    cols[1].metric("ATR Regime", str(kpis.get("atr_regime") or "unknown"))
    fee_ratio = kpis.get("fee_pnl_ratio")
    try:
        fee_ratio_fmt = f"{float(fee_ratio):.2f}x" if fee_ratio is not None else "n/a"
    except Exception:
        fee_ratio_fmt = "n/a"
    cols[2].metric("Fee / PnL", fee_ratio_fmt)
    cols[3].metric("Router Quality", str(kpis.get("router_quality") or "unknown"))
    atr_entries = kpis.get("atr", {}).get("symbols") if isinstance(kpis.get("atr"), dict) else []
    if atr_entries:
        st.caption("Per-symbol ATR regime (last publish)")
        try:
            df = pd.DataFrame(atr_entries)
            if not df.empty and "symbol" in df.columns:
                df = df.set_index("symbol")
            st.dataframe(df, use_container_width=True, height=220)
        except Exception:
            st.json(atr_entries)


def render_positions_table(pos_doc: Dict[str, Any], kpis_v7: Dict[str, Any] | None = None) -> None:
    items = (pos_doc or {}).get("items") or []
    if not isinstance(items, list) or not items:
        st.info("No open positions.")
        return
    enriched: List[Dict[str, Any]] = []
    for pos in positions_sorted(items):
        row = dict(pos)
        try:
            symbol = str(row.get("symbol") or "").upper()
        except Exception:
            symbol = ""
        if symbol:
            try:
                kpi = execution_kpis(symbol, kpis_v7=kpis_v7 or {})
            except Exception:
                kpi = {}
            row.setdefault("dd_state", kpi.get("dd_state_symbol") or kpi.get("dd_state"))
            row.setdefault("dd_today_pct", kpi.get("dd_today_pct"))
            row.setdefault("atr_regime", kpi.get("atr_regime_symbol") or kpi.get("atr_regime"))
            row.setdefault("atr_ratio", kpi.get("atr_ratio_symbol") or kpi.get("atr_ratio"))
        enriched.append(row)
    try:
        df = pd.DataFrame(enriched)
    except Exception:
        df = pd.DataFrame(enriched or items)
    if df.empty:
        st.info("No open positions.")
        return
    st.dataframe(df, use_container_width=True, height=420)


def main() -> None:
    st.set_page_config(page_title="Hedge — v6 Dashboard", layout="wide")
    st.title("📊 Hedge — Portfolio Dashboard (v6)")

    runtime_probe = load_runtime_probe()
    render_runtime_strip(runtime_probe)

    nav_doc, nav_source = load_nav_state()
    synced_state = load_synced_state()
    kpis_v7 = load_kpis_v7()
    pos_doc, pos_source = nav_doc, nav_source  # default
    pos_state = Path(os.getenv("POSITIONS_STATE_PATH") or (PROJECT_ROOT / "logs/state/positions_state.json"))
    try:
        if pos_state.exists():
            pos_doc = json.loads(pos_state.read_text())
            pos_source = pos_state.name
    except Exception:
        pos_doc = {}

    expectancy_v6 = load_expectancy_v6()
    symbol_scores_v6 = load_symbol_scores_v6()
    risk_allocator_v6 = load_risk_allocator_v6()
    router_policy_v6 = load_router_policy_v6()
    router_suggestions_v6 = load_router_suggestions_v6()
    pipeline_shadow_head = load_shadow_head()
    pipeline_compare_summary = load_compare_summary()
    router_health_state = load_router_health_state()
    router_health = load_router_health(snapshot=router_health_state)

    render_header(nav_doc, nav_source, runtime_probe)
    render_kpis_v7_panel(kpis_v7)

    tabs = st.tabs(
        [
            "Overview",
            "Intel",
            "Pipeline",
            "Router",
            "Positions",
        ]
    )

    with tabs[0]:
        st.subheader("NAV & Freshness")
        nav_age = nav_state_age_seconds(nav_doc)
        nav_age_display = _format_age_seconds(nav_age)
        st.metric("NAV Source", nav_source)
        st.metric("NAV Age", nav_age_display)
        if synced_state:
            st.caption("Synced state present.")
        st.markdown("---")
        st.subheader("Runtime Flags")
        flags = runtime_probe.get("flags") if isinstance(runtime_probe.get("flags"), dict) else {}
        st.json(flags or runtime_probe, expanded=False)

    with tabs[1]:
        st.subheader("Intel v6")
        _safe_panel("Intel Panel", render_intel_panel, expectancy_v6, symbol_scores_v6, risk_allocator_v6)

    with tabs[2]:
        st.subheader("Pipeline v6")
        _safe_panel("Pipeline Parity", render_pipeline_parity, pipeline_shadow_head, pipeline_compare_summary)
        st.markdown("Raw Compare Summary")
        st.json(pipeline_compare_summary, expanded=False)

    with tabs[3]:
        st.subheader("Router Health")
        if router_health and not is_empty_router_health(router_health):
            st.metric("Symbols", int(router_health.summary.get("count", 0)))
            if not router_health.per_symbol.empty:
                st.dataframe(router_health.per_symbol, use_container_width=True, height=360)
            else:
                st.info("No router per-symbol metrics yet.")
        else:
            st.info("Router health unavailable.")

        st.markdown("---")
        st.subheader("Router Policy Suggestions")
        # Normalize suggestions to avoid NoneType errors
        suggestions_clean = router_suggestions_v6 if isinstance(router_suggestions_v6, dict) else {}
        current_policy = router_policy_v6
        if (not current_policy) and isinstance(router_health_state, dict):
            # fallback: use policies embedded in router_health symbols
            symbols = router_health_state.get("symbols")
            if isinstance(symbols, list):
                current_policy = {"symbols": symbols}
        _safe_panel("Router Policy", render_router_policy_panel, current_policy or {}, suggestions_clean, False)
        st.caption("Raw suggestions and policy state")
        st.json({"policy": router_policy_v6, "suggestions": router_suggestions_v6}, expanded=False)

    with tabs[4]:
        st.subheader("Positions")
        st.caption(f"positions_source={pos_source}")
        render_positions_table(pos_doc, kpis_v7=kpis_v7)


if __name__ == "__main__":
    main()
