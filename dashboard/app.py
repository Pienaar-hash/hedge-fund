# mypy: ignore-errors
"""Streamlit dashboard for v6 runtime."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import altair as alt

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local helpers
from dashboard.dashboard_utils import (
    positions_sorted,
    get_env_badge,
)
from dashboard.nav_helpers import (
    safe_format,
    safe_float,
)
from dashboard.live_helpers import (
    load_runtime_probe,
    load_expectancy_v6,
    load_symbol_scores_v6,
    load_risk_allocator_v6,
    load_router_policy_v6,
    load_router_health_state,
    load_router_suggestions_v6,
    load_shadow_head,
    load_compare_summary,
)
from dashboard.intel_panel import render_intel_panel, render_hybrid_scores_panel, render_carry_scores_panel, render_vol_regimes_panel
from dashboard.pipeline_panel import render_pipeline_parity
from dashboard.router_policy import render_router_policy_panel
from dashboard.router_health import load_router_health, is_empty_router_health
from dashboard import kpi_panel
from dashboard.state_v7 import (
    load_all_state,
    load_funding_snapshot,
    load_basis_snapshot,
    load_hybrid_scores,
    load_vol_regimes,
    load_runtime_diagnostics_state,
)
from dashboard.research_panel import render_research_panel
from dashboard.risk_panel import (
    render_risk_health_card,
    load_risk_snapshot,
    render_exit_pipeline_status,
    render_veto_heatmap,
    render_liveness_status,
)
from dashboard.regime_panel import render_regime_card, load_regimes_snapshot
from dashboard.router_gauge import render_router_gauge
from dashboard.trader_toys import (
    inject_trader_toys_css,
    render_liquid_gauge,
    render_radial_progress,
    render_mini_bar,
    render_pulse_indicator,
    render_gauge_panel,
)
LOG = logging.getLogger("dash.app")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG.setLevel(logging.INFO)

st.markdown(
    """
    <style>
    .stMetric, .css-1ht1j8u, .css-16idsys, [data-testid="stMetricValue"] {
        overflow: visible !important;
        text-overflow: initial !important;
        white-space: nowrap !important;
    }
    .metric-title { font-size: 0.9rem !important; font-weight: 600 !important; }
    .metric-value { font-size: 1.3rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


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


def _load_nav_history() -> List[Dict]:
    """Load NAV history from nav_log.json for sparkline chart.
    Only returns mainnet data (after cutover from testnet).
    """
    import json
    nav_log_path = PROJECT_ROOT / "logs" / "nav_log.json"
    try:
        if nav_log_path.exists():
            with open(nav_log_path) as f:
                data = json.load(f)
            # Filter to only mainnet data (NAV < 6000 indicates mainnet)
            # This excludes testnet data where NAV was ~11k
            mainnet_data = [p for p in data if p.get('nav', 0) < 6000]
            # Return last 100 points for sparkline
            return mainnet_data[-100:] if len(mainnet_data) > 100 else mainnet_data
    except Exception:
        pass
    return []


def _load_risk_snapshot() -> Dict:
    """Load detailed risk snapshot for symbol cards."""
    import json
    path = PROJECT_ROOT / "logs" / "state" / "risk_snapshot.json"
    try:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


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


def render_header(nav: Dict[str, Any]) -> None:
    nav_value = safe_float(nav.get("nav_usd")) or 0.0
    nav_age_display = _format_age_seconds(nav.get("age_s"))
    nav_source = nav.get("source") or "nav_state.json"
    env_label, env_color = get_env_badge(False)

    st.markdown(
        f"""
        <div style="display:flex;align-items:center;justify-content:space-between;
                    padding:0.8rem 1rem;border:1px solid #444;border-radius:10px;
                    background:#1e1e1e;margin-bottom:0.75rem;">
            <div>
                <div style="font-size:1.6rem;font-weight:700;color:#fff;">NAV: {nav_value:,.2f} USD</div>
                <div style="color:#888;">Source: {nav_source} · Age: {nav_age_display}</div>
            </div>
            <div style="display:flex;align-items:center;gap:0.5rem;">
                <span style="background:{env_color};color:#fff;padding:0.4rem 0.8rem;border-radius:999px;font-weight:700;">{env_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpis_v7_panel(kpis: Dict[str, Any], router: Dict[str, Any]) -> None:
    """Render v7 Risk KPIs from execution_health and router_health data."""
    st.subheader("v7 Risk KPIs")
    
    # Load execution_health for richer data
    import json
    from collections import Counter
    
    exec_health_path = PROJECT_ROOT / "logs" / "state" / "execution_health.json"
    exec_health = {}
    try:
        if exec_health_path.exists():
            with open(exec_health_path) as f:
                exec_health = json.load(f)
    except Exception:
        pass
    
    # Load router_health for aggregate stats
    router_health_path = PROJECT_ROOT / "logs" / "state" / "router_health.json"
    router_health_data = {}
    try:
        if router_health_path.exists():
            with open(router_health_path) as f:
                router_health_data = json.load(f)
    except Exception:
        pass
    
    # Get symbols from execution_health
    symbols_data = exec_health.get("symbols", [])
    rh_symbols = router_health_data.get("symbols", [])
    rh_summary = router_health_data.get("summary", {})
    
    if not symbols_data and not kpis:
        st.info("v7 KPIs unavailable (waiting for executor publish).")
        return
    
    # Aggregate stats from symbols
    if symbols_data:
        dd_states = [s.get("risk", {}).get("dd_state", "normal") for s in symbols_data]
        atr_regimes = [s.get("vol", {}).get("atr_regime") for s in symbols_data]
        atr_regimes = [r for r in atr_regimes if r]  # filter None
        atr_ratios = [s.get("vol", {}).get("atr_ratio") for s in symbols_data if s.get("vol", {}).get("atr_ratio") is not None]
        router_qualities = [s.get("router", {}).get("policy_quality", "ok") for s in symbols_data]
        size_factors = [s.get("sizing", {}).get("final_size_factor") for s in symbols_data if s.get("sizing", {}).get("final_size_factor") is not None]
        sharpe_states = [s.get("risk", {}).get("sharpe_state", "neutral") for s in symbols_data]
        
        dd_state = Counter(dd_states).most_common(1)[0][0] if dd_states else "normal"
        atr_regime = Counter(atr_regimes).most_common(1)[0][0] if atr_regimes else "normal"
        sharpe_state = Counter(sharpe_states).most_common(1)[0][0] if sharpe_states else "neutral"
        
        # Median ATR ratio
        median_atr_ratio = None
        if atr_ratios:
            sorted_ratios = sorted(atr_ratios)
            mid = len(sorted_ratios) // 2
            median_atr_ratio = sorted_ratios[mid] if len(sorted_ratios) % 2 == 1 else (sorted_ratios[mid-1] + sorted_ratios[mid]) / 2
        
        avg_size_factor = sum(size_factors) / len(size_factors) if size_factors else 1.0
    else:
        dd_state = kpis.get("risk", {}).get("dd_state") or "normal"
        atr_regime = kpis.get("risk", {}).get("atr_regime") or "normal"
        median_atr_ratio = kpis.get("risk", {}).get("atr_ratio")
        sharpe_state = "neutral"
        avg_size_factor = 1.0
    
    # Router health from summary
    quality_counts = rh_summary.get("quality_counts", {})
    total_symbols = rh_summary.get("count", len(symbols_data))
    broken_count = quality_counts.get("broken", 0)
    degraded_count = quality_counts.get("degraded", 0)
    ok_count = total_symbols - broken_count - degraded_count
    
    if broken_count > 0:
        router_summary = f"{ok_count}/{total_symbols} ok"
        router_color = "inverse"
    elif degraded_count > 0:
        router_summary = f"{ok_count}/{total_symbols} ok"
        router_color = "off"
    else:
        router_summary = "all ok"
        router_color = "normal"
    
    # Compute execution stats from router_health symbols (only those with actual trades)
    slippages = [s.get("slippage_p50") for s in rh_symbols if s.get("slippage_p50") is not None]
    fallbacks = [s.get("fallback_rate") for s in rh_symbols if s.get("fallback_rate") is not None]
    maker_fills = [s.get("maker_fill_rate") for s in rh_symbols if s.get("maker_fill_rate") is not None]
    
    symbols_with_trades = len(slippages)
    avg_slip = sum(slippages) / len(slippages) if slippages else None
    avg_fallback = sum(fallbacks) / len(fallbacks) if fallbacks else None
    avg_maker = sum(maker_fills) / len(maker_fills) if maker_fills else None
    
    # First row - Volatility & Risk States
    cols = st.columns(4)
    
    # ATR regime with color coding
    atr_color = {"quiet": "🟢", "normal": "🟡", "elevated": "🟠", "hot": "🔴"}.get(atr_regime, "⚪")
    atr_display = f"{atr_color} {atr_regime}"
    if median_atr_ratio is not None:
        atr_display += f" ({median_atr_ratio:.2f}x)"
    cols[0].metric("ATR Regime", atr_display)
    
    # DD state with color
    dd_color = {"normal": "🟢", "cautious": "🟡", "elevated": "🟠", "critical": "🔴"}.get(dd_state, "⚪")
    cols[1].metric("DD State", f"{dd_color} {dd_state}")
    
    # Sharpe state
    sharpe_color = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}.get(sharpe_state, "⚪")
    cols[2].metric("Sharpe State", f"{sharpe_color} {sharpe_state}")
    
    # Size factor
    cols[3].metric("Avg Size Factor", f"{avg_size_factor:.0%}")
    
    # Second row - Router & Execution Quality
    cols2 = st.columns(4)
    cols2[0].metric("Router Health", router_summary)
    
    # Only show execution stats if we have actual trade data
    if symbols_with_trades > 0:
        cols2[1].metric("Avg Slip (bps)", f"{avg_slip:.1f}" if avg_slip else "—")
        cols2[2].metric("Maker Fill", f"{avg_maker*100:.0f}%" if avg_maker is not None else "—")
        cols2[3].metric("Fallback Rate", f"{avg_fallback*100:.0f}%" if avg_fallback is not None else "—")
    else:
        cols2[1].metric("Avg Slip (bps)", "—", help="No trades yet")
        cols2[2].metric("Maker Fill", "—", help="No trades yet")
        cols2[3].metric("Fallback Rate", "—", help="No trades yet")
    
    # Per-symbol table from execution_health
    if symbols_data:
        st.caption("Per-symbol Execution Health")
        rows = []
        for s in symbols_data:
            sym = s.get("symbol", "?").replace("USDT", "")
            router = s.get("router", {})
            risk = s.get("risk", {})
            sizing = s.get("sizing", {})
            vol = s.get("vol", {})
            
            slip = router.get('slip_q50')
            fallback = router.get('fallback_ratio')
            atr_ratio = vol.get('atr_ratio')
            atr_regime = vol.get('atr_regime', '?')
            
            rows.append({
                "Symbol": sym,
                "Router": router.get("policy_quality", "ok"),
                "Risk": risk.get("dd_state", "normal"),
                "ATR": f"{atr_regime} ({atr_ratio:.2f}x)" if atr_ratio else atr_regime,
                "Sharpe": risk.get("sharpe_state", "neutral"),
                "Size": f"{sizing.get('final_size_factor', 1.0)*100:.0f}%",
                "Slip": f"{slip:.1f}" if slip else "—",
                "FB%": f"{fallback*100:.0f}%" if fallback is not None else "—",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=220)


def _load_raw_positions() -> List[Dict]:
    """Load positions from the raw source that has actual prices."""
    import json
    # Try state/positions.json first (has entryPrice, markPrice)
    path1 = PROJECT_ROOT / "logs" / "state" / "positions.json"
    try:
        if path1.exists():
            with open(path1) as f:
                data = json.load(f)
            rows = data.get("rows", [])
            if rows:
                return rows
    except Exception:
        pass
    # Fallback to logs/positions.json
    path2 = PROJECT_ROOT / "logs" / "positions.json"
    try:
        if path2.exists():
            with open(path2) as f:
                data = json.load(f)
            return data.get("items", [])
    except Exception:
        pass
    return []


def render_positions_table(positions: List[Dict[str, Any]]) -> None:
    # Use raw positions data which has actual prices
    raw_positions = _load_raw_positions()
    
    if not raw_positions:
        st.info("No open positions.")
        return
    
    enriched = []
    for pos in raw_positions:
        qty = float(pos.get("qty") or pos.get("size") or 0)
        if abs(qty) > 0.0001:
            entry = float(pos.get("entryPrice") or pos.get("entry") or pos.get("entry_price") or 0)
            mark = float(pos.get("markPrice") or pos.get("mark") or pos.get("mark_price") or 0)
            notional = abs(qty * mark) if mark else 0
            side = pos.get("side") or pos.get("positionSide") or ("SHORT" if qty < 0 else "LONG")
            
            pnl = pos.get("unrealized") or pos.get("pnl") or pos.get("pnl_usd")
            # Calculate PnL if not provided or is zero
            if (pnl is None or pnl == 0) and entry and mark:
                if side == "SHORT":
                    pnl = (entry - mark) * abs(qty)  # Profit when mark < entry
                else:
                    pnl = (mark - entry) * abs(qty)  # Profit when mark > entry
            
            leverage = pos.get("leverage")
            
            row = {
                "Symbol": pos.get("symbol", "?").replace("USDT", ""),
                "Side": pos.get("side") or pos.get("positionSide") or ("SHORT" if qty < 0 else "LONG"),
                "Qty": abs(qty),
                "Entry": f"${entry:,.2f}" if entry else "—",
                "Mark": f"${mark:,.2f}" if mark else "—",
                "Notional": f"${notional:,.2f}",
                "PnL": f"${pnl:,.2f}" if pnl is not None else "—",
                "Lev": f"{int(leverage)}x" if leverage else "—",
            }
            enriched.append(row)
    
    if not enriched:
        st.info("No open positions.")
        return
    
    df = pd.DataFrame(enriched)
    st.dataframe(df, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Hedge — v6 Dashboard", layout="wide")
    
    # Inject trader toys CSS
    inject_trader_toys_css()
    
    st.title("📊 Hedge — Portfolio Dashboard (v7)")

    state = load_all_state()
    nav = state.get("nav", {})
    aum = state.get("aum", {})
    kpis = state.get("kpis", {})
    positions = state.get("positions", [])
    router = state.get("router", {})
    meta = state.get("meta", {})
    runtime_probe = load_runtime_probe()
    expectancy_v6 = load_expectancy_v6()
    symbol_scores_v6 = load_symbol_scores_v6()
    risk_allocator_v6 = load_risk_allocator_v6()
    router_policy_v6 = load_router_policy_v6()
    router_suggestions_v6 = load_router_suggestions_v6()
    pipeline_shadow_head = load_shadow_head()
    pipeline_compare_summary = load_compare_summary()
    router_health_state = load_router_health_state()
    router_health = load_router_health(snapshot=router_health_state)
    
    # Load NAV history for sparkline
    nav_history = _load_nav_history()

    render_header(nav)

    tabs = st.tabs(["Overview (v7)", "Research & Optimizations", "Advanced (v6)"])

    with tabs[0]:
        st.subheader("Overview (v7)")
        risk_block = kpis.get("risk") or {}
        router_kpis = kpis.get("router") or {}
        
        # First row: NAV, AUM, Exposure, PnL
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        col1.metric("NAV (USD)", safe_format(nav.get("nav_usd")))
        nav_zar = nav.get("nav_zar")
        if nav_zar:
            col1.caption(f"NAV ZAR {safe_format(nav_zar, nd=0)}")
        
        col2.metric("AUM (USD)", safe_format(aum.get("total_usd")))
        aum_zar = aum.get("total_zar")
        if aum_zar:
            col2.caption(f"AUM ZAR {safe_format(aum_zar, nd=0)}")
        
        # Calculate gross exposure from raw positions (nav_state may be stale)
        raw_positions = _load_raw_positions()
        calculated_gross = 0.0
        calculated_net = 0.0
        for pos in raw_positions:
            qty = float(pos.get("qty") or pos.get("size") or 0)
            mark = pos.get("markPrice") or pos.get("mark") or pos.get("mark_price") or 0
            notional = abs(qty * mark) if mark else 0
            calculated_gross += notional
            calculated_net += qty * mark if mark else 0
        
        # Use calculated if available, else fall back to nav_state
        gross_exp = calculated_gross if calculated_gross > 0 else safe_float(nav.get("gross_exposure"))
        net_exp = calculated_net if calculated_gross > 0 else safe_float(nav.get("net_exposure"))
        col3.metric("Gross Exposure", f"${gross_exp:,.0f}" if gross_exp else "$0")
        col3.caption(f"Net: ${net_exp:,.0f}" if net_exp else "Net: $0")
        
        realized_pnl = safe_float(nav.get("realized_pnl_today"))
        unrealized_pnl = safe_float(nav.get("unrealized_pnl"))
        
        # If unrealized is 0 but we have positions, calculate from entry vs mark prices
        if (unrealized_pnl == 0 or unrealized_pnl is None) and raw_positions:
            calc_unrealized = 0.0
            for pos in raw_positions:
                qty = float(pos.get("qty") or pos.get("size") or pos.get("positionAmt") or 0)
                entry = float(pos.get("entryPrice") or pos.get("entry") or 0)
                mark = float(pos.get("markPrice") or pos.get("mark") or pos.get("mark_price") or 0)
                side = pos.get("positionSide", "LONG")
                if qty != 0 and entry > 0 and mark > 0:
                    if side == "SHORT":
                        calc_unrealized += (entry - mark) * abs(qty)
                    else:
                        calc_unrealized += (mark - entry) * abs(qty)
            if calc_unrealized != 0:
                unrealized_pnl = calc_unrealized
        
        col4.metric("PnL Today", safe_format(realized_pnl))
        unrealized_color = "#00cc00" if (unrealized_pnl or 0) >= 0 else "#ff4444"
        col4.caption(f"Unrealized: <span style='color:{unrealized_color}'>{safe_format(unrealized_pnl)}</span>", unsafe_allow_html=True)
        
        # Second row: Drawdown, ATR, Router metrics
        col_r1, col_r2, col_r3, col_r4 = st.columns([1, 1, 1, 1])
        
        # Drawdown: stored value is already a percentage (e.g., 0.04 = 0.04%)
        drawdown_pct = safe_float(nav.get("drawdown_pct")) or 0.0
        if drawdown_pct < 0.01:  # Less than 0.01% - truly at peak
            dd_label = "0.00%"
            dd_caption = "At peak"
        elif drawdown_pct < 0.1:  # Tiny drawdown - show with more precision
            dd_label = f"-{drawdown_pct:.3f}%"
            dd_caption = "Near peak"
        else:
            dd_label = f"-{drawdown_pct:.2f}%"
            dd_caption = risk_block.get("dd_state") or "normal"
        col_r1.metric("Drawdown", dd_label)
        col_r1.caption(dd_caption)
        
        atr_ratio = safe_float(risk_block.get("atr_ratio"))
        atr_label = safe_format(atr_ratio, nd=3) if atr_ratio is not None else "n/a"
        col_r2.metric("ATR Ratio", atr_label)
        col_r2.caption(risk_block.get("atr_regime") or "n/a")
        
        router_quality = router.get("quality") or router_kpis.get("quality") or router_kpis.get("policy_quality") or "n/a"
        col_r3.metric("Router Quality", str(router_quality))
        
        maker_fill_rate = safe_float(router_kpis.get("maker_fill_rate") or router.get("maker_fill_rate"))
        fallback_ratio = safe_float(router_kpis.get("fallback_ratio") or router.get("fallback_ratio"))
        maker_label = safe_format(maker_fill_rate, nd=2) if maker_fill_rate is not None else "0.00"
        fallback_label = safe_format(fallback_ratio, nd=2) if fallback_ratio is not None else "n/a"
        col_r4.metric("Maker / Fallback", f"{maker_label} / {fallback_label}")
        
        st.caption(f"Source: {nav.get('source') or 'nav_state.json'} • Age: {_format_age_seconds(nav.get('age_s'))}")

        # NAV Sparkline Chart
        if nav_history and len(nav_history) > 5:
            import plotly.graph_objects as go
            from datetime import datetime
            
            nav_vals = [p.get('nav', 0) for p in nav_history]
            times = [datetime.fromtimestamp(p.get('t', 0)) for p in nav_history]
            
            # Determine line color based on trend
            trend_color = "#00cc00" if nav_vals[-1] >= nav_vals[0] else "#ff4444"
            
            fig_spark = go.Figure()
            fig_spark.add_trace(go.Scatter(
                x=times, y=nav_vals,
                mode='lines',
                fill='tozeroy',
                fillcolor=f'rgba({int(trend_color[1:3], 16)}, {int(trend_color[3:5], 16)}, {int(trend_color[5:7], 16)}, 0.1)',
                line=dict(color=trend_color, width=2),
                hovertemplate='%{y:,.2f}<extra></extra>'
            ))
            fig_spark.update_layout(
                height=120,
                margin=dict(t=10, b=10, l=10, r=10),
                xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_spark, use_container_width=True, config={'displayModeBar': False})

        # ===== TRADER TOYS: Risk Gauges =====
        st.markdown("#### 🎯 Risk Gauges")
        
        # Calculate metrics for gauges
        nav_value = safe_float(nav.get("nav_usd")) or 1
        exposure_pct = (gross_exp / nav_value * 100) if nav_value > 0 else 0
        
        # Margin used estimation (notional / (nav * max_leverage))
        max_leverage = 4  # From config
        margin_capacity = nav_value * max_leverage
        margin_used_pct = (gross_exp / margin_capacity * 100) if margin_capacity > 0 else 0
        
        # Risk capacity = inverse of exposure utilization
        max_exposure_pct = 150  # From risk_limits
        risk_capacity_pct = max(0, 100 - (exposure_pct / max_exposure_pct * 100))
        
        gauge_cols = st.columns(4)
        
        with gauge_cols[0]:
            # Exposure gauge
            exp_color = "green" if exposure_pct < 80 else ("yellow" if exposure_pct < 120 else "red")
            st.markdown(f"""
            <div class="liquid-gauge {exp_color}" style="width:85px;height:85px;margin:0 auto;">
                <span class="gauge-value">{exposure_pct:.0f}%</span>
                <span class="gauge-label">Exposure</span>
            </div>
            <style>.liquid-gauge.{exp_color}::before {{ height: {min(100, exposure_pct/1.5):.0f}%; }}</style>
            """, unsafe_allow_html=True)
        
        with gauge_cols[1]:
            # Margin used gauge
            margin_color = "green" if margin_used_pct < 50 else ("yellow" if margin_used_pct < 80 else "red")
            st.markdown(f"""
            <div class="liquid-gauge {margin_color}" style="width:85px;height:85px;margin:0 auto;">
                <span class="gauge-value">{margin_used_pct:.0f}%</span>
                <span class="gauge-label">Margin</span>
            </div>
            <style>.liquid-gauge.{margin_color}::before {{ height: {min(100, margin_used_pct):.0f}%; }}</style>
            """, unsafe_allow_html=True)
        
        with gauge_cols[2]:
            # Drawdown gauge (inverse - lower is better)
            dd_value = drawdown_pct if drawdown_pct else 0
            dd_color = "green" if dd_value < 5 else ("yellow" if dd_value < 15 else "red")
            dd_fill = min(100, dd_value * 3.33)  # Scale to 30% max
            st.markdown(f"""
            <div class="liquid-gauge {dd_color}" style="width:85px;height:85px;margin:0 auto;">
                <span class="gauge-value">{dd_value:.1f}%</span>
                <span class="gauge-label">Drawdown</span>
            </div>
            <style>.liquid-gauge.{dd_color}::before {{ height: {dd_fill:.0f}%; }}</style>
            """, unsafe_allow_html=True)
        
        with gauge_cols[3]:
            # Risk capacity gauge (higher is better)
            cap_color = "red" if risk_capacity_pct < 30 else ("yellow" if risk_capacity_pct < 60 else "green")
            st.markdown(f"""
            <div class="liquid-gauge {cap_color}" style="width:85px;height:85px;margin:0 auto;">
                <span class="gauge-value">{risk_capacity_pct:.0f}%</span>
                <span class="gauge-label">Capacity</span>
            </div>
            <style>.liquid-gauge.{cap_color}::before {{ height: {min(100, risk_capacity_pct):.0f}%; }}</style>
            """, unsafe_allow_html=True)

        # ===== RISK HEALTH CARD (v7) =====
        st.markdown("---")
        risk_snapshot = load_risk_snapshot()
        if risk_snapshot:
            render_risk_health_card(
                snapshot=risk_snapshot,
                nav_value=nav_value,
                gross_exposure=gross_exp,
                cap_drawdown=0.30,  # 30% max drawdown cap
                cap_daily_loss=0.10,  # 10% daily loss cap
            )
        diagnostics_state = load_runtime_diagnostics_state()
        if diagnostics_state:
            render_veto_heatmap(diagnostics_state)
            render_exit_pipeline_status(diagnostics_state)
            render_liveness_status(diagnostics_state)

        # ===== REGIME HEATMAP (v7) =====
        st.markdown("---")
        st.markdown("#### 🌡️ ATR/DD Regime Heatmap")
        regimes_state = load_regimes_snapshot()
        if regimes_state:
            render_regime_card(regimes_state)
        else:
            st.info("Regime data unavailable - waiting for state sync")

        # ===== ROUTER HEALTH GAUGE (v7) =====
        st.markdown("---")
        render_router_gauge(router_health_state or {})

        st.markdown("---")
        
        # AUM Section
        total_usd = aum.get('total_usd', 0)
        total_zar = aum.get('total_zar', 0)
        total_pnl = aum.get("total_pnl_usd", 0.0)
        pnl_pct = (total_pnl / total_usd * 100) if total_usd > 0 else 0
        pnl_color = "#00cc00" if total_pnl >= 0 else "#ff4444"
        pnl_arrow = "▲" if total_pnl >= 0 else "▼"
        slices = aum.get("slices") or []
        
        # Animated AUM Header with glowing effect
        st.markdown(f'''
        <div style="display:flex;gap:20px;align-items:stretch;margin:15px 0;">
            <div style="flex:1;background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);border-radius:16px;padding:20px;border:1px solid #333;box-shadow:0 0 20px rgba(0,200,255,0.1);">
                <div style="font-size:12px;color:#888;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;">💰 Total AUM</div>
                <div style="font-size:32px;font-weight:700;color:#00d4ff;text-shadow:0 0 20px rgba(0,212,255,0.4);">${total_usd:,.0f}</div>
                <div style="font-size:14px;color:#666;margin-top:4px;">ZAR {total_zar:,.0f}</div>
            </div>
            <div style="flex:1;background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);border-radius:16px;padding:20px;border:1px solid #333;box-shadow:0 0 20px rgba({"0,255,100" if total_pnl >= 0 else "255,100,100"},0.1);">
                <div style="font-size:12px;color:#888;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;">📈 Total PnL</div>
                <div style="font-size:32px;font-weight:700;color:{pnl_color};text-shadow:0 0 20px {pnl_color}66;">{pnl_arrow} ${abs(total_pnl):,.2f}</div>
                <div style="font-size:14px;color:{pnl_color};margin-top:4px;">{pnl_pct:+.2f}% return</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Donut and breakdown side by side - balanced layout
        col_chart, col_list = st.columns([1, 1])
        
        with col_chart:
            if slices:
                import plotly.graph_objects as go
                labels = [s.get('label', '?') for s in slices]
                values = [s.get('usd', 0) for s in slices]
                # Premium color palette matching asset colors
                color_map = {"Futures": "#00d4ff", "BTC": "#f7931a", "ETH": "#627eea", "SOL": "#00ffa3", "USDT": "#26a17b", "USDC": "#2775ca", "XAUT": "#ffd700"}
                colors = [color_map.get(l, "#888888") for l in labels]
                
                fig = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.65,  # Larger hole for true ring appearance
                    textinfo='label+percent',
                    textposition='inside',
                    textfont=dict(size=12, color='#fff'),
                    hovertemplate='<b>%{label}</b><br>$%{value:,.0f}<br>%{percent}<extra></extra>',
                    marker=dict(
                        colors=colors,
                        line=dict(color='#1a1a2e', width=3)
                    ),
                    pull=[0.02] * len(labels)  # Slight pull for 3D effect
                )])
                fig.update_layout(
                    height=380,
                    margin=dict(t=20, b=20, l=20, r=20),
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    annotations=[]  # Remove central label - true ring style
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col_list:
            st.markdown("#### Asset Breakdown")
            # Build a lookup for off-exchange details (has avg_cost)
            offex_details = {d.get('symbol'): d for d in aum.get('offexchange_details', [])}
            
            # Asset type icons
            asset_icons = {"Futures": "⚡", "BTC": "₿", "ETH": "⟠", "SOL": "◎", "USDT": "💵", "USDC": "💵", "XAUT": "🥇"}
            asset_colors = {"Futures": "#00d4ff", "BTC": "#f7931a", "ETH": "#627eea", "SOL": "#00ffa3", "USDT": "#26a17b", "USDC": "#2775ca", "XAUT": "#ffd700"}
            
            for s in slices:
                label = s.get('label', '?')
                usd_val = s.get('usd', 0)
                pct_of_total = (usd_val / total_usd * 100) if total_usd > 0 else 0
                
                # Get avg_cost for off-exchange assets
                detail = offex_details.get(label, {})
                avg_cost = detail.get('avg_cost')
                qty = detail.get('qty')
                
                # Asset styling
                icon = asset_icons.get(label, "💎")
                bar_color = asset_colors.get(label, "#888")
                
                # Build the line - Futures uses all-time net PnL, others use unrealized PnL
                extra_info = ""
                if label == "Futures":
                    alltime_pnl = s.get('alltime_realized_pnl', 0)
                    alltime_fees = s.get('alltime_fees', 0)
                    alltime_notional = s.get('alltime_notional', 0)
                    alltime_trades = s.get('alltime_trades', 0)
                    net_pnl = alltime_pnl - alltime_fees
                    pnl_pct = (net_pnl / usd_val * 100) if usd_val > 0 else 0
                    arrow = "▲" if net_pnl >= 0 else "▼"
                    color = "#00cc00" if net_pnl >= 0 else "#ff4444"
                    if alltime_trades > 0:
                        extra_info = f"<div style='font-size:11px;color:#666;margin-top:4px;'>📊 {alltime_trades} trades • ${alltime_notional:,.0f} vol • ${alltime_fees:.2f} fees</div>"
                    pnl_usd = net_pnl
                    pnl_pct_asset = pnl_pct
                else:
                    pnl_usd = s.get("pnl_usd", 0)
                    pnl_pct_asset = s.get("pnl_pct", 0)
                    arrow = "▲" if pnl_usd >= 0 else "▼"
                    color = "#00cc00" if pnl_usd >= 0 else "#ff4444"
                    if avg_cost and qty:
                        extra_info = f"<div style='font-size:11px;color:#666;margin-top:4px;'>{qty:.4g} @ ${avg_cost:,.2f}</div>"
                
                # Render asset card with allocation bar
                st.markdown(f'''
                <div style="background:linear-gradient(135deg,#1a1a2e 0%,#0f0f1a 100%);border-radius:10px;padding:12px 15px;margin:8px 0;border-left:3px solid {bar_color};">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <span style="font-size:18px;font-weight:600;color:#fff;">{icon} {label}</span>
                            <span style="font-size:14px;color:#888;margin-left:10px;">${usd_val:,.0f}</span>
                        </div>
                        <div style="text-align:right;">
                            <span style="color:{color};font-weight:bold;font-size:14px;">{arrow} ${abs(pnl_usd):,.2f}</span>
                            <span style="color:{color};font-size:12px;margin-left:5px;">({pnl_pct_asset:+.1f}%)</span>
                        </div>
                    </div>
                    <div style="height:4px;background:#222;border-radius:2px;margin-top:8px;overflow:hidden;">
                        <div style="height:100%;width:{pct_of_total}%;background:linear-gradient(90deg,{bar_color}88,{bar_color});border-radius:2px;"></div>
                    </div>
                    <div style="font-size:10px;color:#555;margin-top:4px;">{pct_of_total:.1f}% of portfolio</div>
                    {extra_info}
                </div>
                ''', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Trading Performance Section - Visual charts instead of tables
        expectancy_data = load_expectancy_v6()
        exp_age_s = None
        if expectancy_data:
            exp_ts = expectancy_data.get("updated_ts")
            if exp_ts:
                import time
                exp_age_s = time.time() - float(exp_ts)
        exp_age_str = _format_age_seconds(exp_age_s) if exp_age_s is not None else "n/a"
        lookback = expectancy_data.get("lookback_hours", 48) if expectancy_data else 48
        sample_count = expectancy_data.get("sample_count", 0) if expectancy_data else 0
        
        st.markdown(f"### Trading Performance <span style='font-size:12px; color:#888; font-weight:normal'>(last {lookback}h • {sample_count} completed trades • updated {exp_age_str} ago)</span>", unsafe_allow_html=True)
        
        # Quick stats row
        if expectancy_data and "symbols" in expectancy_data:
            symbols_exp = expectancy_data["symbols"]
            total_trades = sum(s.get("count", 0) for s in symbols_exp.values())
            avg_win_rate = sum(s.get("hit_rate", 0) for s in symbols_exp.values()) / len(symbols_exp) * 100 if symbols_exp else 0
            avg_expectancy = sum(s.get("expectancy", 0) for s in symbols_exp.values()) / len(symbols_exp) if symbols_exp else 0
            total_pnl_trades = sum(s.get("expectancy", 0) * s.get("count", 0) for s in symbols_exp.values())
            
            exp_color = "#00cc00" if avg_expectancy >= 0 else "#ff4444"
            pnl_trade_color = "#00cc00" if total_pnl_trades >= 0 else "#ff4444"
            wr_color = "#00cc00" if avg_win_rate >= 55 else ("#ffaa00" if avg_win_rate >= 45 else "#ff4444")
            
            st.markdown(f'''
            <div style="display:flex;gap:12px;margin:15px 0;">
                <div style="flex:1;background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:12px;padding:15px;text-align:center;border:1px solid #333;">
                    <div style="font-size:11px;color:#888;text-transform:uppercase;">Trades</div>
                    <div style="font-size:28px;font-weight:700;color:#00d4ff;">{total_trades}</div>
                </div>
                <div style="flex:1;background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:12px;padding:15px;text-align:center;border:1px solid #333;">
                    <div style="font-size:11px;color:#888;text-transform:uppercase;">Win Rate</div>
                    <div style="font-size:28px;font-weight:700;color:{wr_color};">{avg_win_rate:.0f}%</div>
                </div>
                <div style="flex:1;background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:12px;padding:15px;text-align:center;border:1px solid #333;">
                    <div style="font-size:11px;color:#888;text-transform:uppercase;">Avg EV/Trade</div>
                    <div style="font-size:28px;font-weight:700;color:{exp_color};">${avg_expectancy:.2f}</div>
                </div>
                <div style="flex:1;background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:12px;padding:15px;text-align:center;border:1px solid #333;">
                    <div style="font-size:11px;color:#888;text-transform:uppercase;">Est. PnL</div>
                    <div style="font-size:28px;font-weight:700;color:{pnl_trade_color};">${total_pnl_trades:+,.0f}</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        col_perf_left, col_perf_right = st.columns([1, 1])
        
        with col_perf_left:
            # Expectancy Bar Chart by Symbol (horizontal for better readability)
            if expectancy_data and "symbols" in expectancy_data:
                import plotly.graph_objects as go
                
                symbols = list(expectancy_data["symbols"].keys())
                # Clean symbol names
                symbol_labels = [s.replace("USDT", "") for s in symbols]
                win_rates = [expectancy_data["symbols"][s].get("hit_rate", 0) * 100 for s in symbols]
                expectancies = [expectancy_data["symbols"][s].get("expectancy", 0) for s in symbols]
                trade_counts = [expectancy_data["symbols"][s].get("count", 0) for s in symbols]
                
                # Color by expectancy (green = positive, red = negative)
                colors = ["#00cc00" if e >= 0 else "#ff4444" for e in expectancies]
                
                # Premium gradient colors based on value
                bar_colors = []
                for e in expectancies:
                    if e >= 0.5:
                        bar_colors.append('#00ff88')  # Bright green
                    elif e >= 0:
                        bar_colors.append('#00cc66')  # Green
                    elif e >= -0.5:
                        bar_colors.append('#ff6b35')  # Orange-red
                    else:
                        bar_colors.append('#ff4444')  # Red
                
                fig_exp = go.Figure()
                fig_exp.add_trace(go.Bar(
                    y=symbol_labels,
                    x=expectancies,
                    orientation='h',
                    marker=dict(
                        color=bar_colors,
                        line=dict(color='rgba(255,255,255,0.1)', width=1)
                    ),
                    text=[f'${e:+.2f}' for e in expectancies],
                    textposition='outside',
                    textfont=dict(size=11, color='#ccc'),
                    hovertemplate='<b>%{y}</b><br>EV: $%{x:.2f} per trade<extra></extra>'
                ))
                
                # Add win rate annotations on the bars
                for i, (sym, wr, tc) in enumerate(zip(symbol_labels, win_rates, trade_counts)):
                    fig_exp.add_annotation(
                        x=0, y=i,
                        text=f'{wr:.0f}% ({tc})',
                        showarrow=False,
                        font=dict(size=9, color='#888'),
                        xanchor='right' if expectancies[i] >= 0 else 'left',
                        xshift=-5 if expectancies[i] >= 0 else 5
                    )
                
                fig_exp.update_layout(
                    title=dict(text='📊 Expectancy by Symbol', font=dict(size=14, color='#fff'), x=0),
                    height=220,
                    margin=dict(t=45, b=25, l=55, r=55),
                    xaxis=dict(
                        title=dict(text='$ per trade', font=dict(size=10, color='#666')),
                        zeroline=True,
                        zerolinecolor='#444',
                        zerolinewidth=2,
                        gridcolor='rgba(255,255,255,0.05)',
                        tickfont=dict(color='#888')
                    ),
                    yaxis=dict(title='', tickfont=dict(color='#ccc', size=11)),
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(26,26,46,0.5)',
                    bargap=0.3
                )
                st.plotly_chart(fig_exp, use_container_width=True)
            else:
                st.info("No trade history yet.")
        
        with col_perf_right:
            # Hour-of-day Heatmap
            if expectancy_data and "hours" in expectancy_data:
                import plotly.graph_objects as go
                
                hour_data = expectancy_data["hours"]
                hours = sorted([int(h) for h in hour_data.keys()])
                
                if hours:
                    # Create performance data for each hour
                    hour_labels = [f"{h:02d}:00" for h in hours]
                    returns = [hour_data[str(h)].get("avg_return", 0) for h in hours]
                    counts = [hour_data[str(h)].get("count", 0) for h in hours]
                    
                    # Premium gradient colors based on return value
                    bar_colors = []
                    for r in returns:
                        if r >= 1.0:
                            bar_colors.append('#00ff88')  # Bright green
                        elif r >= 0:
                            bar_colors.append('#00cc66')  # Green
                        elif r >= -1.0:
                            bar_colors.append('#ff6b35')  # Orange
                        else:
                            bar_colors.append('#ff4444')  # Red
                    
                    fig_hours = go.Figure()
                    fig_hours.add_trace(go.Bar(
                        y=hour_labels,
                        x=returns,
                        orientation='h',
                        marker=dict(
                            color=bar_colors,
                            line=dict(color='rgba(255,255,255,0.1)', width=1)
                        ),
                        text=[f'${r:+.2f}' for r in returns],
                        textposition='outside',
                        textfont=dict(size=10, color='#ccc'),
                        hovertemplate='<b>%{y} UTC</b><br>Avg Return: $%{x:.2f}<br>Trades: ' + str(counts) + '<extra></extra>'
                    ))
                    
                    # Add trade count annotations
                    for i, (h, c) in enumerate(zip(hour_labels, counts)):
                        if c > 0:
                            fig_hours.add_annotation(
                                x=0, y=i,
                                text=f'({c})',
                                showarrow=False,
                                font=dict(size=9, color='#666'),
                                xanchor='right' if returns[i] >= 0 else 'left',
                                xshift=-5 if returns[i] >= 0 else 5
                            )
                    
                    bar_height = max(220, len(hours) * 24)
                    fig_hours.update_layout(
                        title=dict(text='🕐 Avg Return by Hour (UTC)', font=dict(size=14, color='#fff'), x=0),
                        height=bar_height,
                        margin=dict(t=45, b=25, l=55, r=55),
                        xaxis=dict(
                            title=dict(text='$ Return', font=dict(size=10, color='#666')),
                            zeroline=True,
                            zerolinecolor='#444',
                            zerolinewidth=2,
                            gridcolor='rgba(255,255,255,0.05)',
                            tickfont=dict(color='#888')
                        ),
                        yaxis=dict(title='', autorange='reversed', tickfont=dict(color='#ccc', size=10)),
                        showlegend=False,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(26,26,46,0.5)',
                        bargap=0.25
                    )
                    st.plotly_chart(fig_hours, use_container_width=True)
                    
                    # Best/Worst hours with visual styling
                    best_hour = max(hour_data.items(), key=lambda x: x[1].get("avg_return", 0))
                    worst_hour = min(hour_data.items(), key=lambda x: x[1].get("avg_return", 0))
                    best_ret = best_hour[1].get("avg_return", 0)
                    worst_ret = worst_hour[1].get("avg_return", 0)
                    st.markdown(f'''
                    <div style="display:flex;justify-content:space-around;margin-top:10px;">
                        <div style="text-align:center;">
                            <span style="color:#00ff88;font-size:20px;">🏆</span>
                            <span style="color:#00ff88;font-weight:bold;">{best_hour[0]}:00</span>
                            <span style="color:#888;font-size:12px;"> (+${best_ret:.2f})</span>
                        </div>
                        <div style="text-align:center;">
                            <span style="color:#ff4444;font-size:20px;">⚠️</span>
                            <span style="color:#ff4444;font-weight:bold;">{worst_hour[0]}:00</span>
                            <span style="color:#888;font-size:12px;"> (${worst_ret:.2f})</span>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.info("No hourly data yet.")
            else:
                st.info("Hourly performance coming soon.")
        
        st.markdown("---")
        
        # Position status with pulse indicators
        pos_count = len(raw_positions) if raw_positions else 0
        if pos_count > 0:
            total_pnl = sum(
                float(p.get("unrealized") or p.get("pnl") or 0)
                for p in raw_positions
            )
            pnl_status = "green" if total_pnl >= 0 else "red"
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:15px;margin-bottom:10px;">
                <span style="font-weight:600;font-size:16px;">Open Positions</span>
                <span class="pulse-dot {pnl_status}"></span>
                <span style="color:#888;font-size:14px;">{pos_count} active</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("**Open Positions**")
        
        render_positions_table(positions)
        st.caption(f"Data age: {_format_age_seconds(meta.get('data_age_s'))}")
        
        # Per-symbol risk snapshot - Visual Cards
        st.markdown("---")
        st.markdown("### Symbol Health")
        try:
            risk_snapshot = _load_risk_snapshot()
            if risk_snapshot and "symbols" in risk_snapshot:
                symbols_data = risk_snapshot["symbols"]
                
                # Create a grid of symbol cards
                num_symbols = len(symbols_data)
                cols_per_row = min(4, num_symbols) if num_symbols > 0 else 4
                
                for i in range(0, num_symbols, cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < num_symbols:
                            sym_data = symbols_data[idx]
                            symbol = sym_data.get("symbol", "?").replace("USDT", "")
                            
                            # Get status data
                            risk_state = sym_data.get("risk", {}).get("dd_state", "normal")
                            router_quality = sym_data.get("router", {}).get("policy_quality", "ok")
                            atr_regime = sym_data.get("vol", {}).get("atr_regime", "unknown")
                            atr_ratio = sym_data.get("vol", {}).get("atr_ratio")
                            size_factor = sym_data.get("sizing", {}).get("final_size_factor", 1.0)
                            slip_q50 = sym_data.get("router", {}).get("slip_q50")
                            
                            # Color coding
                            def status_color(val, good_vals=["normal", "ok", "low"]):
                                if val in good_vals:
                                    return "#00cc00"
                                elif val in ["elevated", "poor", "medium"]:
                                    return "#ffaa00"
                                else:
                                    return "#ff4444"
                            
                            def status_emoji(val, good_vals=["normal", "ok", "low"]):
                                if val in good_vals:
                                    return "🟢"
                                elif val in ["elevated", "poor", "medium", "unknown"]:
                                    return "🟡"
                                else:
                                    return "🔴"
                            
                            def atr_color(regime):
                                if regime == "quiet":
                                    return "#00aaff"  # Blue - low vol
                                elif regime == "normal":
                                    return "#00cc00"  # Green
                                elif regime == "hot":
                                    return "#ffaa00"  # Orange
                                elif regime == "panic":
                                    return "#ff4444"  # Red
                                return "#888"
                            
                            risk_emoji = status_emoji(risk_state)
                            router_emoji = status_emoji(router_quality)
                            
                            # Build card HTML
                            slip_text = f"{slip_q50:.1f}bps" if slip_q50 is not None else "n/a"
                            size_color = "#00cc00" if size_factor >= 0.9 else ("#ffaa00" if size_factor >= 0.7 else "#ff4444")
                            atr_text = f"{atr_regime} ({atr_ratio:.2f}x)" if atr_ratio else atr_regime
                            
                            with col:
                                # Calculate progress bar percentages
                                risk_pct = {"low": 25, "normal": 30, "medium": 55, "elevated": 70, "high": 85, "extreme": 100}.get(risk_state.lower(), 50)
                                router_pct = {"excellent": 95, "ok": 80, "good": 75, "fair": 50, "poor": 25, "degraded": 20, "cold": 10}.get(router_quality.lower(), 50)
                                size_pct = min(100, int(size_factor * 100))
                                
                                risk_bar_color = {"low": "#00ff88", "normal": "#00ff88", "medium": "#ffcc00", "elevated": "#ff9900", "high": "#ff6b35", "extreme": "#ff0040"}.get(risk_state.lower(), "#888")
                                router_bar_color = {"excellent": "#00ff88", "ok": "#00ff88", "good": "#00d4ff", "fair": "#ffcc00", "poor": "#ff6b35", "degraded": "#ff4444", "cold": "#888"}.get(router_quality.lower(), "#888")
                                
                                risk_c = status_color(risk_state)
                                router_c = status_color(router_quality)
                                atr_c = atr_color(atr_regime)
                                
                                # Build card in plain format for safer rendering
                                card_html = f'''<div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);border-radius:12px;padding:16px;margin:4px 0;border:1px solid #333;box-shadow:0 4px 15px rgba(0,0,0,0.3);">
<div style="font-size:20px;font-weight:bold;margin-bottom:12px;"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{size_color};box-shadow:0 0 8px {size_color};margin-right:8px;"></span>{symbol}</div>
<div style="font-size:13px;margin:8px 0;">{risk_emoji} Risk: <span style="color:{risk_c}">{risk_state}</span></div>
<div style="height:4px;background:#222;border-radius:2px;margin:4px 0;overflow:hidden;"><div style="height:100%;width:{risk_pct}%;background:linear-gradient(90deg,{risk_bar_color}88,{risk_bar_color});border-radius:2px;"></div></div>
<div style="font-size:13px;margin:8px 0;">{router_emoji} Router: <span style="color:{router_c}">{router_quality}</span></div>
<div style="height:4px;background:#222;border-radius:2px;margin:4px 0;overflow:hidden;"><div style="height:100%;width:{router_pct}%;background:linear-gradient(90deg,{router_bar_color}88,{router_bar_color});border-radius:2px;"></div></div>
<div style="font-size:13px;margin:8px 0;">📊 ATR: <span style="color:{atr_c}">{atr_text}</span></div>
<div style="font-size:13px;margin:8px 0;color:#888;">Slip: {slip_text}</div>
<div style="font-size:13px;margin-top:10px;">⚖️ Size: <span style="color:{size_color};font-weight:bold">{size_factor:.0%}</span></div>
<div style="height:6px;background:#222;border-radius:3px;margin:4px 0;overflow:hidden;"><div style="height:100%;width:{size_pct}%;background:linear-gradient(90deg,{size_color}88,{size_color});border-radius:3px;"></div></div>
</div>'''
                                st.markdown(card_html, unsafe_allow_html=True)
            else:
                st.info("Risk snapshot unavailable.")
        except Exception as e:
            st.error(f"Error loading risk snapshot: {e}")
        
        st.markdown("---")
        render_kpis_v7_panel(kpis, router)

    with tabs[1]:
        render_research_panel()

    with tabs[2]:
        st.subheader("Advanced (v6)")
        st.caption("Legacy v6 telemetry — visible for internal diagnostics only.")
        with st.expander("Runtime (v6 telemetry)", expanded=False):
            render_runtime_strip(runtime_probe)
        st.metric("NAV Source", nav.get("source") or "nav_state.json")
        st.metric("NAV Age", _format_age_seconds(nav.get("age_s")))
        st.markdown("---")
        with st.expander("Vol Regimes (v7.4)", expanded=False):
            vol_regimes = load_vol_regimes()
            _safe_panel("Vol Regimes Panel", render_vol_regimes_panel, vol_regimes)
        with st.expander("Hybrid Scores (v7.4)", expanded=False):
            hybrid_scores = load_hybrid_scores()
            _safe_panel("Hybrid Scores Panel", render_hybrid_scores_panel, hybrid_scores)
        with st.expander("Carry Components (v7.4)", expanded=False):
            funding_snap = load_funding_snapshot()
            basis_snap = load_basis_snapshot()
            _safe_panel("Carry Scores Panel", render_carry_scores_panel, funding_snap, basis_snap)
        with st.expander("Intel v6", expanded=False):
            _safe_panel("Intel Panel", render_intel_panel, expectancy_v6, symbol_scores_v6, risk_allocator_v6)
        with st.expander("Pipeline v6", expanded=False):
            _safe_panel("Pipeline Parity", render_pipeline_parity, pipeline_shadow_head, pipeline_compare_summary)
            st.markdown("Raw Compare Summary")
            st.json(pipeline_compare_summary, expanded=False)
        with st.expander("Router v6", expanded=False):
            if router_health and not is_empty_router_health(router_health):
                st.metric("Symbols", int(router_health.summary.get("count", 0)))
                if not router_health.per_symbol.empty:
                    df_router = router_health.per_symbol.copy()
                    if "value" in df_router.columns:
                        try:
                            df_router["value"] = pd.to_numeric(df_router["value"], errors="coerce").fillna(0.0).astype(float)
                        except Exception:
                            df_router["value"] = df_router["value"]
                    for col in df_router.columns:
                        col_lower = str(col).lower()
                        if col_lower in {
                            "value",
                            "qty",
                            "pnl",
                            "notional",
                            "dd_pct",
                            "atr_ratio",
                            "maker_fill_ratio",
                            "fallback_ratio",
                            "slip_q25",
                            "slip_q50",
                            "slip_q75",
                            "reject_rate",
                        }:
                            df_router[col] = df_router[col].apply(safe_float)
                    st.dataframe(df_router, use_container_width=True, height=360)
                else:
                    st.info("No router per-symbol metrics yet.")
            else:
                st.info("Router health unavailable.")
            st.markdown("---")
            st.subheader("Router Policy Suggestions")
            suggestions_clean = router_suggestions_v6 if isinstance(router_suggestions_v6, dict) else {}
            current_policy = router_policy_v6
            if (not current_policy) and isinstance(router_health_state, dict):
                symbols = router_health_state.get("symbols")
                if isinstance(symbols, list):
                    current_policy = {"symbols": symbols}
            _safe_panel("Router Policy", render_router_policy_panel, current_policy or {}, suggestions_clean, False)
            st.caption("Raw suggestions and policy state")
            st.json({"policy": router_policy_v6, "suggestions": router_suggestions_v6}, expanded=False)
        with st.expander("Positions v6", expanded=False):
            st.caption(f"positions_source={nav.get('source') or 'nav_state.json'}")
            render_positions_table(positions)
    
    # Rich Footer (outside tabs so it appears on all tabs)
    st.markdown("---")
    import time
    from datetime import datetime
    
    footer_parts = []
    
    # Version - use actual version number, not branch name
    try:
        version_path = PROJECT_ROOT / "VERSION"
        if version_path.exists():
            version = version_path.read_text().strip()
            # If it's a branch name like "v7-risk-tuning", extract just version part
            if version.startswith("v") and "-" in version:
                version = version.split("-")[0]  # "v7"
            elif not version.startswith("v"):
                version = f"v{version}"
            footer_parts.append(f"📦 {version}")
    except Exception:
        pass
    
    # Environment
    import os
    env_label = os.getenv("ENV") or os.getenv("HEDGE_ENV") or "unknown"
    binance_testnet = os.getenv("BINANCE_TESTNET", "0").strip() in ("1", "true", "True")
    env_icon = "🧪" if binance_testnet or "test" in env_label.lower() else "🟢"
    footer_parts.append(f"{env_icon} {env_label}")
    
    # Data freshness
    state_files = {
        "nav": PROJECT_ROOT / "logs" / "state" / "nav_state.json",
        "pos": PROJECT_ROOT / "logs" / "state" / "positions.json",
    }
    now = time.time()
    max_age = 0
    for name, path in state_files.items():
        try:
            if path.exists():
                import json
                data = json.loads(path.read_text())
                ts = data.get("updated_ts") or data.get("updated") or data.get("ts")
                if ts:
                    age = now - float(ts)
                    max_age = max(max_age, age)
        except Exception:
            pass
    
    if max_age > 0:
        if max_age < 60:
            footer_parts.append(f"🟢 Live ({int(max_age)}s)")
        elif max_age < 300:
            footer_parts.append(f"🟡 Recent ({int(max_age/60)}m)")
        else:
            footer_parts.append(f"🔴 Stale ({int(max_age/60)}m)")
    
    # Symbols enabled - check universe array structure
    try:
        universe_path = PROJECT_ROOT / "config" / "pairs_universe.json"
        if universe_path.exists():
            import json
            data = json.loads(universe_path.read_text())
            # New format: has "universe" array
            if "universe" in data and isinstance(data["universe"], list):
                universe_list = data["universe"]
                enabled = sum(1 for s in universe_list if isinstance(s, dict) and s.get("enabled"))
                total = len(universe_list)
            else:
                # Old format: flat dict
                enabled = sum(1 for s in data.values() if isinstance(s, dict) and s.get("enabled"))
                total = len([v for v in data.values() if isinstance(v, dict)])
            footer_parts.append(f"📊 {enabled}/{total} symbols")
    except Exception:
        pass
    
    # Current time
    footer_parts.append(f"🕐 {datetime.utcnow().strftime('%H:%M:%S')} UTC")
    
    st.markdown(
        f"<div style='text-align:center; padding:15px; color:#888; font-size:14px; border-top:1px solid #333; margin-top:20px;'>"
        f"{' • '.join(footer_parts)}"
        f"</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
