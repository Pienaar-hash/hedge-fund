#!/usr/bin/env python3
"""Hedge Streamlit overview dashboard.

Features
--------
- Firestore-first data with automatic local fallbacks.
- Trading/Portfolio NAV KPI cards with drawdown over selectable window.
- Reserve quick tiles, risk summary, open positions table.
- Recent veto reason distribution and tail log.
- Per-symbol KPIs (hit rate, average size, slippage, last signal/veto).
- Environment knobs surfaced as read-only controls.
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from dashboard.data_sources import DashboardData, load_dashboard_data, read_nav_snapshot


try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

if load_dotenv:
    load_dotenv()


WINDOW_OPTIONS = {
    "24h": 24 * 3600,
    "7d": 7 * 24 * 3600,
    "30d": 30 * 24 * 3600,
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV = os.getenv("ENV", "prod")
TESTNET = os.getenv("BINANCE_TESTNET", "0") == "1"
ENV_KEY = f"{ENV}{'-testnet' if TESTNET else ''}"


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def humanize_seconds(delta: Optional[float]) -> str:
    if delta is None or math.isnan(delta):
        return "—"
    delta = max(0.0, float(delta))
    if delta < 60:
        return f"{delta:.0f}s ago"
    if delta < 3600:
        return f"{delta/60:.1f}m ago"
    if delta < 86400:
        return f"{delta/3600:.1f}h ago"
    return f"{delta/86400:.1f}d ago"


def humanize_ts(ts: Optional[float]) -> str:
    if ts is None:
        return "—"
    return humanize_seconds(time.time() - float(ts))


def fmt_usd(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        val = float(value)
    except Exception:
        return "—"
    if math.isnan(val):
        return "—"
    return f"{val:,.{digits}f}"


def fmt_pct(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        val = float(value)
    except Exception:
        return "—"
    if math.isnan(val):
        return "—"
    return f"{val:.{digits}f}%"


def fmt_usd_delta(value: Optional[float], digits: int = 0) -> Optional[str]:
    if value is None:
        return None
    try:
        val = float(value)
    except Exception:
        return None
    if math.isclose(val, 0.0, abs_tol=10 ** (-digits - 2)):
        val = 0.0
    sign = "+" if val > 0 else ""
    return f"{sign}{val:,.{digits}f}"


def _max_drawdown_pct(series: List[Dict[str, float]]) -> Optional[float]:
    if not series:
        return None
    peak = None
    max_dd = 0.0
    for point in sorted(series, key=lambda x: x["ts"]):
        nav = float(point.get("nav", 0.0))
        if nav <= 0:
            continue
        if peak is None or nav > peak:
            peak = nav
        if peak and nav < peak:
            dd = (peak - nav) / peak * 100.0
            max_dd = max(max_dd, dd)
    return max_dd if max_dd > 0 else 0.0


def compute_nav_metrics(nav_series: List[Dict[str, float]], window_sec: int, snapshot: Dict[str, Any]) -> Dict[str, Optional[float]]:
    now = time.time()
    window_series = [row for row in nav_series if row.get("ts") and row["ts"] >= now - window_sec]
    if len(window_series) < 2:
        window_series = nav_series[-min(len(nav_series), 200):]

    nav_now = window_series[-1]["nav"] if window_series else snapshot.get("trading_nav_usd")
    nav_ref = window_series[0]["nav"] if window_series else nav_now

    delta = None
    delta_pct = None
    if nav_now is not None and nav_ref:
        delta = float(nav_now) - float(nav_ref)
        if nav_ref != 0:
            delta_pct = (delta / float(nav_ref)) * 100.0

    drawdown_pct = _max_drawdown_pct(window_series)
    if drawdown_pct is None and snapshot.get("drawdown") is not None:
        try:
            drawdown_pct = float(snapshot.get("drawdown"))
        except Exception:
            drawdown_pct = None

    return {
        "nav_now": float(nav_now) if nav_now is not None else None,
        "nav_ref": float(nav_ref) if nav_ref is not None else None,
        "delta": delta,
        "delta_pct": delta_pct,
        "drawdown_pct": drawdown_pct,
    }


def nav_dataframe(nav_series: List[Dict[str, float]], window_sec: int) -> pd.DataFrame:
    if not nav_series:
        return pd.DataFrame(columns=["ts", "nav"])
    now = time.time()
    rows = [row for row in nav_series if row.get("ts") and row["ts"] >= now - window_sec]
    if not rows:
        rows = nav_series
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = df.set_index("ts").sort_index()
    return df


def build_reserve_map(snapshot: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    reserves = snapshot.get("reserves", []) or []
    fees = snapshot.get("fee_wallets", []) or []
    out: Dict[str, Dict[str, Any]] = {}
    for rec in reserves:
        if not isinstance(rec, dict):
            continue
        asset = str(rec.get("asset") or rec.get("symbol") or "").upper()
        if asset:
            out[asset] = rec
    for rec in fees:
        if not isinstance(rec, dict):
            continue
        asset = str(rec.get("asset") or rec.get("symbol") or "").upper()
        if asset:
            out[f"{asset}_FEE"] = rec
    return out


def _float_env(env_value: Optional[str]) -> Optional[float]:
    if env_value is None:
        return None
    try:
        return float(env_value)
    except Exception:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def load_risk_limits_config() -> Dict[str, Any]:
    path = PROJECT_ROOT / "config" / "risk_limits.json"
    try:
        raw = path.read_text()
    except Exception:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def compute_risk_summary(
    positions: List[Dict[str, Any]],
    snapshot: Dict[str, Any],
    env_knobs: Dict[str, Any],
    nav_series: List[Dict[str, float]],
    risk_events: List[Dict[str, Any]],
) -> Dict[str, Optional[float | str]]:
    trading_nav = snapshot.get("trading_nav_usd")
    if trading_nav is None and nav_series:
        trading_nav = nav_series[-1].get("nav")
    trading_nav = float(trading_nav) if trading_nav is not None else None

    gross = 0.0
    for pos in positions:
        try:
            gross += float(pos.get("notional") or 0.0)
        except Exception:
            continue

    cap_override = _float_env(env_knobs.get("CAP_OVERRIDE_PCT"))
    cfg = load_risk_limits_config()
    global_cfg = cfg.get("global") if isinstance(cfg, dict) else None
    if not global_cfg and isinstance(cfg, dict):
        global_cfg = cfg  # backwards compatibility
    cfg_cap = None
    try:
        cfg_cap = float((global_cfg or {}).get("max_gross_exposure_pct", 0.0))
    except Exception:
        cfg_cap = None
    cap_pct = cap_override if cap_override is not None else cfg_cap

    cap_notional = None
    headroom = None
    used_pct = None
    if cap_pct is not None and trading_nav is not None:
        cap_notional = trading_nav * (cap_pct / 100.0)
        headroom = cap_notional - gross
        used_pct = (gross / cap_notional * 100.0) if cap_notional else None

    kill_reasons = {"kill_switch", "kill_switch_triggered", "kill_switch_drawdown"}
    kill_event = next((evt for evt in risk_events if evt.get("reason") in kill_reasons), None)
    if kill_event:
        kill_status = f"Triggered {humanize_ts(kill_event.get('ts'))}"
    else:
        kill_status = "Clear"

    return {
        "trading_nav": trading_nav,
        "gross": gross,
        "cap_pct": cap_pct,
        "cap_notional": cap_notional,
        "headroom": headroom,
        "used_pct": used_pct,
        "kill_status": kill_status,
    }


def veto_distribution(risk_events: List[Dict[str, Any]], window_sec: int = 24 * 3600) -> pd.DataFrame:
    now = time.time()
    records: Dict[str, int] = {}
    for evt in risk_events:
        ts = evt.get("ts")
        if not ts or ts < now - window_sec:
            continue
        reason = str(evt.get("reason") or "other").lower()
        records[reason] = records.get(reason, 0) + 1
    df = pd.DataFrame({"reason": list(records.keys()), "count": list(records.values())})
    return df.sort_values("count", ascending=False)


def veto_tail(risk_events: List[Dict[str, Any]], limit: int = 12) -> pd.DataFrame:
    if not risk_events:
        return pd.DataFrame(columns=["time", "symbol", "reason", "notional"])
    rows = []
    for evt in risk_events[:limit]:
        ts = evt.get("ts")
        when = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S") if ts else "—"
        rows.append(
            {
                "time": when,
                "symbol": evt.get("symbol"),
                "reason": evt.get("reason"),
                "notional": evt.get("notional") or evt.get("gross"),
            }
        )
    return pd.DataFrame(rows)


def compute_symbol_kpis(
    positions: List[Dict[str, Any]],
    trades: List[Dict[str, Any]],
    signals: List[Dict[str, Any]],
    risk_events: List[Dict[str, Any]],
) -> pd.DataFrame:
    now = time.time()
    sym_set = {str(pos.get("symbol")) for pos in positions if pos.get("symbol")}
    sym_set.update(str(trade.get("symbol")) for trade in trades if trade.get("symbol"))
    sym_set.update(str(sig.get("symbol")) for sig in signals if sig.get("symbol"))
    sym_set.update(str(evt.get("symbol")) for evt in risk_events if evt.get("symbol"))
    symbols = sorted(s for s in sym_set if s and s != "None")

    rows = []
    for sym in symbols:
        trades_sym = [t for t in trades if t.get("symbol") == sym]
        trades_24h = [t for t in trades_sym if t.get("ts") and t["ts"] >= now - 24 * 3600]
        trades_7d = [t for t in trades_sym if t.get("ts") and t["ts"] >= now - 7 * 24 * 3600]

        def _hit_rate(rows: List[Dict[str, Any]]) -> Optional[float]:
            wins = 0
            total = 0
            for r in rows:
                pnl = r.get("pnl")
                if pnl is None:
                    continue
                total += 1
                if float(pnl) > 0:
                    wins += 1
            if total == 0:
                return None
            return wins / total * 100.0

        def _avg_notional(rows: List[Dict[str, Any]]) -> Optional[float]:
            vals = [abs(float(r.get("notional"))) for r in rows if r.get("notional")]
            if not vals:
                return None
            return sum(vals) / len(vals)

        def _avg_slippage(rows: List[Dict[str, Any]]) -> Optional[float]:
            vals = [float(r.get("slippage_bps")) for r in rows if r.get("slippage_bps")]
            if not vals:
                return None
            return sum(vals) / len(vals)

        last_signal_ts = next((sig.get("ts") for sig in signals if sig.get("symbol") == sym), None)
        last_veto = next((evt.get("reason") for evt in risk_events if evt.get("symbol") == sym), None)
        open_position = next((pos for pos in positions if pos.get("symbol") == sym), None)
        open_notional = open_position.get("notional") if open_position else None

        rows.append(
            {
                "symbol": sym,
                "hit_rate_24h": _hit_rate(trades_24h),
                "hit_rate_7d": _hit_rate(trades_7d),
                "avg_trade_size_usd": _avg_notional(trades_7d),
                "avg_slippage_bps": _avg_slippage(trades_7d),
                "last_signal": humanize_ts(last_signal_ts),
                "last_veto_reason": last_veto or "—",
                "open_notional_usd": open_notional,
            }
        )

    if not rows:
        return pd.DataFrame(columns=[
            "symbol",
            "hit_rate_24h",
            "hit_rate_7d",
            "avg_trade_size_usd",
            "avg_slippage_bps",
            "last_signal",
            "last_veto_reason",
            "open_notional_usd",
        ])

    df = pd.DataFrame(rows)
    df = df.sort_values(by="symbol")
    df["hit_rate_24h"] = df["hit_rate_24h"].apply(lambda x: fmt_pct(x, 1) if x is not None else "—")
    df["hit_rate_7d"] = df["hit_rate_7d"].apply(lambda x: fmt_pct(x, 1) if x is not None else "—")
    df["avg_trade_size_usd"] = df["avg_trade_size_usd"].apply(lambda x: fmt_usd(x, 0))
    df["avg_slippage_bps"] = df["avg_slippage_bps"].apply(lambda x: f"{x:.1f}" if x is not None else "—")
    df["open_notional_usd"] = df["open_notional_usd"].apply(lambda x: fmt_usd(x, 0))
    return df


@st.cache_data(ttl=300, show_spinner=False)
def load_walkforward_metrics() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    metrics_path = PROJECT_ROOT / "logs" / "walkforward_metrics.csv"
    summary_path = metrics_path.with_suffix(".json")
    if not metrics_path.exists():
        return pd.DataFrame(), {}
    try:
        df = pd.read_csv(metrics_path)
    except Exception:
        df = pd.DataFrame()
    summary: Dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            summary = {}
    return df, summary


# ---------------------------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------------------------


st.set_page_config(page_title="Hedge — Overview", layout="wide")
st.title("Hedge — Overview")

data: DashboardData = load_dashboard_data()
snapshot = data.nav_snapshot.data or {}
snapshot_prev = read_nav_snapshot("logs/nav_snapshot.json") or {}
nav_series = data.nav_series.data or []
positions = data.positions.data or []
trades = data.trades.data or []
risk_events = data.risk_events.data or []
signals = data.signals.data or []
screener_tail_lines = data.screener_tail.data or []
env_knobs = data.env_risk_knobs.data or {}

window_label = st.radio("Drawdown window", list(WINDOW_OPTIONS.keys()), index=0, horizontal=True)
window_sec = WINDOW_OPTIONS[window_label]

include_spot_flag = bool(snapshot.get("include_spot_in_trading_nav", False))
st.toggle(
    "Show Spot in Trading NAV",
    value=include_spot_flag,
    disabled=True,
    help="Mirrors INCLUDE_SPOT_IN_TRADING_NAV flag. Toggle is read-only.",
)

nav_metrics = compute_nav_metrics(nav_series, window_sec, snapshot)

source_caption = (
    f"NAV • {data.nav_series.source}"
    + (f" ({data.nav_series.detail})" if data.nav_series.detail else "")
    + f" | Positions • {data.positions.source}"
    + (f" ({data.positions.detail})" if data.positions.detail else "")
    + f" | Risk • {data.risk_events.source}"
    + (f" ({data.risk_events.detail})" if data.risk_events.detail else "")
)
st.caption(source_caption)

if os.getenv("FIRESTORE_ENABLED", "0") != "1":
    st.info("Firestore disabled; using local snapshots and logs.")

if data.nav_series.errors:
    st.warning("NAV series load issues: " + "; ".join(data.nav_series.errors))

top_cols = st.columns(4)
top_cols[0].metric(
    "Trading NAV (USDT)",
    fmt_usd(nav_metrics["nav_now"]),
    None if nav_metrics["delta"] is None else f"{fmt_usd(nav_metrics['delta'])} / {fmt_pct(nav_metrics['delta_pct'])}",
    help="Futures NAV plus spot if INCLUDE_SPOT_IN_TRADING_NAV=1.",
)

portfolio_nav = snapshot.get("portfolio_nav_usd")
if portfolio_nav is None and nav_metrics["nav_now"] is not None:
    spot_total = snapshot.get("spot_total_usd") or 0.0
    reserves_total = snapshot.get("reserves_total_usd") or 0.0
    fee_total = snapshot.get("fee_wallets_total_usd") or 0.0
    portfolio_nav = nav_metrics["nav_now"] + float(spot_total) + float(reserves_total) + float(fee_total)

top_cols[1].metric(
    "Portfolio NAV (USDT)",
    fmt_usd(portfolio_nav),
    fmt_usd((portfolio_nav or 0.0) - (nav_metrics["nav_ref"] or 0.0)) if portfolio_nav and nav_metrics["nav_ref"] else None,
    help="Trading NAV plus spot balances, reserves, and fee wallets.",
)

top_cols[2].metric(
    f"Drawdown ({window_label})",
    fmt_pct(nav_metrics["drawdown_pct"]),
    help="Peak-to-trough drawdown over the selected window.",
)

open_positions_cnt = sum(1 for pos in positions if pos.get("qty"))
top_cols[3].metric(
    "Open Positions",
    str(open_positions_cnt),
    help="Count of non-zero futures positions.",
)

reserve_map = build_reserve_map(snapshot)
prev_reserve_map = build_reserve_map(snapshot_prev)
reserve_cols = st.columns(3)
btc_tile = reserve_map.get("BTC")
xaut_tile = reserve_map.get("XAUT")
bnb_tile = reserve_map.get("BNB_FEE") or reserve_map.get("BNB")


def _reserve_delta(curr_key: str, prev_key: Optional[str] = None) -> Optional[float]:
    curr = reserve_map.get(curr_key)
    prev = prev_reserve_map.get(prev_key or curr_key)
    curr_usd = curr.get("usd") if isinstance(curr, dict) else None
    prev_usd = prev.get("usd") if isinstance(prev, dict) else None
    if curr_usd is None or prev_usd is None:
        return None
    try:
        return float(curr_usd) - float(prev_usd)
    except Exception:
        return None

reserve_cols[0].metric(
    "Reserves • BTC",
    fmt_usd((btc_tile or {}).get("usd")),
    fmt_usd_delta(_reserve_delta("BTC")),
    help="Off-exchange BTC reserves valued in USDT.",
)
reserve_cols[1].metric(
    "Reserves • XAUT",
    fmt_usd((xaut_tile or {}).get("usd")),
    fmt_usd_delta(_reserve_delta("XAUT")),
    help="Gold-backed reserves (XAUT) valued in USDT.",
)
reserve_cols[2].metric(
    "Fee Wallet • BNB",
    fmt_usd((bnb_tile or {}).get("usd")),
    fmt_usd_delta(_reserve_delta("BNB_FEE", "BNB") or _reserve_delta("BNB")),
    help="BNB fee wallet valuation in USDT.",
)

st.divider()

nav_df = nav_dataframe(nav_series, window_sec)
if nav_df.empty:
    st.info("NAV series unavailable. Ensure Firestore state/nav or local nav.jsonl is populated.")
else:
    st.line_chart(nav_df["nav"], use_container_width=True)
    st.caption(f"NAV updated {humanize_ts(data.nav_series.updated_at)}")

st.divider()

st.subheader("Walk-forward Performance")
wf_df, wf_summary = load_walkforward_metrics()
if wf_df.empty:
    st.info("Run scripts/walkforward_eval.py to populate walk-forward metrics.")
else:
    metric_cols = st.columns(3)
    metric_cols[0].metric("Avg Sharpe", f"{wf_summary.get('avg_sharpe', float(wf_df['sharpe'].mean())):.2f}")
    metric_cols[1].metric("Avg Calmar", f"{wf_summary.get('avg_calmar', float(wf_df['calmar'].mean())):.2f}")
    metric_cols[2].metric("Avg PSR", f"{wf_summary.get('avg_psr', float(wf_df['psr'].mean())):.2f}")
    st.dataframe(wf_df, use_container_width=True, height=260)
    st.caption(f"Walk-forward windows: {wf_summary.get('windows', len(wf_df))}")

st.divider()

positions_header = st.container()
positions_header.subheader("Open Positions")
if positions:
    df_positions = pd.DataFrame(positions)
    df_positions_display = df_positions.rename(
        columns={
            "symbol": "Symbol",
            "side": "Side",
            "qty": "Qty",
            "entry_price": "Entry",
            "mark_price": "Mark",
            "pnl": "PnL",
            "leverage": "Lev",
            "notional": "Notional",
        }
    )
    st.dataframe(df_positions_display, use_container_width=True, height=320)
else:
    st.info("No open positions.")
st.caption(f"Positions updated {humanize_ts(data.positions.updated_at)}")

st.divider()

risk_summary = compute_risk_summary(positions, snapshot, env_knobs, nav_series, risk_events)
risk_cols = st.columns(4)
risk_cols[0].metric("Open Gross", fmt_usd(risk_summary["gross"], 0))
risk_cols[1].metric(
    "Cap (NAV %)",
    fmt_pct(risk_summary["cap_pct"], 1),
    fmt_usd(risk_summary.get("cap_notional"), 0) if risk_summary.get("cap_notional") else None,
)
risk_cols[2].metric(
    "Headroom",
    fmt_usd(risk_summary.get("headroom"), 0),
    fmt_pct(risk_summary.get("used_pct"), 1) if risk_summary.get("used_pct") is not None else None,
)
risk_cols[3].metric("Kill Switch", risk_summary.get("kill_status", "—"))
st.caption(f"Risk updated {humanize_ts(data.risk_events.updated_at)}")

st.divider()

veto_df = veto_distribution(risk_events)
if veto_df.empty:
    st.info("No veto reasons recorded in the last 24h.")
else:
    chart = pd.DataFrame(veto_df).set_index("reason")
    st.bar_chart(chart, use_container_width=True)

st.subheader("Recent veto tail")
tail_df = veto_tail(risk_events)
if tail_df.empty:
    st.info("No recent veto events.")
else:
    st.dataframe(tail_df, use_container_width=True, height=240)

st.divider()

st.subheader("Per-symbol KPIs")
symbol_df = compute_symbol_kpis(positions, trades, signals, risk_events)
if symbol_df.empty:
    st.info("Symbol KPIs unavailable. Need recent trades or signals.")
else:
    st.dataframe(symbol_df, use_container_width=True, height=360)
st.caption(
    "Trades updated "
    + humanize_ts(data.trades.updated_at)
    + " • Signals updated "
    + humanize_ts(data.signals.updated_at)
)

st.divider()

st.subheader("Screener tail")
if screener_tail_lines:
    tail_text = "\n".join(screener_tail_lines[-200:])
    st.code(tail_text or "(empty)")
else:
    st.info("No screener tail logs present.")
st.caption(f"Screener tail updated {humanize_ts(data.screener_tail.updated_at)}")

st.divider()

with st.expander("Environment & Data Health", expanded=False):
    st.write(f"ENV: {ENV} • ENV_KEY: {ENV_KEY}")
    st.write(f"Firestore enabled: {'yes' if os.getenv('FIRESTORE_ENABLED', '0') == '1' else 'no'}")
    if env_knobs:
        st.write("Risk knobs:")
        st.json(env_knobs)
    else:
        st.write("Risk knobs unavailable.")

    summary_rows = [
        ("NAV", data.nav_series.source, data.nav_series.detail, humanize_ts(data.nav_series.updated_at)),
        ("Positions", data.positions.source, data.positions.detail, humanize_ts(data.positions.updated_at)),
        ("Trades", data.trades.source, data.trades.detail, humanize_ts(data.trades.updated_at)),
        ("Risk", data.risk_events.source, data.risk_events.detail, humanize_ts(data.risk_events.updated_at)),
        ("Signals", data.signals.source, data.signals.detail, humanize_ts(data.signals.updated_at)),
    ]
    df_summary = pd.DataFrame(summary_rows, columns=["Dataset", "Source", "Detail", "Updated"])
    st.table(df_summary)

    all_errors: List[str] = []
    for dataset in (
        data.nav_series,
        data.positions,
        data.trades,
        data.risk_events,
        data.signals,
        data.screener_tail,
        data.veto_events,
    ):
        all_errors.extend(dataset.errors or [])
    all_errors = [err for err in all_errors if err]
    if all_errors:
        st.error("Data load warnings:\n" + "\n".join(all_errors))
    else:
        st.success("No data load warnings.")
