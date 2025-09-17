#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations
"""Streamlit dashboard (single "Overview" tab), read-only.
Firestore-first; falls back to local files for NAV.
Shows: KPIs, NAV chart, Positions, Signals (5), Trade log (5), Screener tail (10), BTC reserve.
"""
# Streamlit dashboard (single "Overview" tab), read-only.
# Firestore-first; falls back to local files for NAV.

# ---- tolerant dotenv ----
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import os
import json
import time
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st
from pathlib import Path

# Diagnostics container populated during data loads
_DIAG: Dict[str, Any] = {
    "source": None,
    "fs_project": None,
    "col_nav": None,
    "col_positions": None,
    "col_trades": None,
    "col_risk": None,
}

# ---- single env context ----
ENV = os.getenv("ENV", "prod")
TESTNET = os.getenv("BINANCE_TESTNET", "0") == "1"
ENV_KEY = f"{ENV}{'-testnet' if TESTNET else ''}"

RESERVE_BTC = float(os.getenv("RESERVE_BTC", "0.025"))
# Local-only log files when Firestore is not selected
LOG_PATH = os.getenv("EXECUTOR_LOG", f"logs/{ENV_KEY}/screener_tail.log")
TAIL_LINES = 10
RISK_CFG_PATH = os.getenv("RISK_LIMITS_CONFIG", "config/risk_limits.json")

# ---------- helpers ----------
def btc_24h_change() -> float | None:
    import requests
    # Use TESTNET flag from ENV wiring; default to mainnet
    base = "https://testnet.binancefuture.com" if TESTNET else "https://fapi.binance.com"
    try:
        r = requests.get(base + "/fapi/v1/ticker/24hr", params={"symbol":"BTCUSDT"}, timeout=6)
        r.raise_for_status()
        return float(r.json().get("priceChangePercent", 0.0))
    except Exception:
        try:
            r = requests.get("https://api.binance.com/api/v3/ticker/24hr", params={"symbol":"BTCUSDT"}, timeout=6)
            r.raise_for_status()
            return float(r.json().get("priceChangePercent", 0.0))
        except Exception:
            return None

def _load_json(path: str, default=None):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default

# Firestore (best effort)
def _fs_client():
    """Return Firestore client if libs + credentials are available.
    Firestore is authoritative only if:
    - google.cloud.firestore imports OK
    - And credentials file path is present: FIREBASE_CREDS_PATH or GOOGLE_APPLICATION_CREDENTIALS
    """
    try:
        from google.cloud import firestore  # type: ignore
    except Exception:
        return None

    creds_path = os.getenv("FIREBASE_CREDS_PATH") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path and Path(creds_path).exists():
        try:
            return firestore.Client()
        except Exception:
            return None
    return None

# --- Firestore paths helper (correct sibling collections) ---
def fs_paths(db):
    ROOT_DOC = db.collection("hedge").document(ENV)
    STATE_COL = ROOT_DOC.collection("state")
    return {
        "root_doc": ROOT_DOC,
        "state_col": STATE_COL,
        "nav_doc": STATE_COL.document("nav"),
        "positions_doc": STATE_COL.document("positions"),
        "trades_col": ROOT_DOC.collection("trades"),
        "risk_col": ROOT_DOC.collection("risk"),
    }

def _fs_get_state(doc: str):
    """Back-compat read for single-doc state if present under old path.
    This is best-effort and namespacing by ENV only. Prefer collection helpers below.
    """
    cli = _fs_client()
    if not cli:
        return None
    try:
        snap = cli.document(f"hedge/{ENV}/state/{doc}").get()
        return snap.to_dict() if getattr(snap, "exists", False) else None
    except Exception:
        return None

# ---- Firestore helpers (authoritative when selected) ----
def _fs_pick_collection(cli, candidates: List[str]) -> Optional[str]:
    """Pick the first collection that appears to have docs. Returns name or None.
    Document choice in Doctor panel later.
    """
    for name in candidates:
        try:
            it = cli.collection(name).limit(1).stream()
            for _ in it:
                return name
        except Exception:
            continue
    return None

def _filter_env_fields(doc: Dict[str, Any]) -> bool:
    """Client-side filter: if env/testnet fields exist, they must match.
    Always applied for generic collections.
    """
    d_env = doc.get("env")
    d_tn = doc.get("testnet")
    if d_env is not None and str(d_env) != ENV:
        return False
    if d_tn is not None and bool(d_tn) != TESTNET:
        return False
    return True

def _is_mixed_namespaces(docs: List[Dict[str, Any]]) -> bool:
    env_vals = set()
    tn_vals = set()
    for d in docs:
        if "env" in d:
            env_vals.add(str(d.get("env")))
        if "testnet" in d:
            tn_vals.add(bool(d.get("testnet")))
    return (len(env_vals) > 1) or (len(tn_vals) > 1)

def _get_firestore_project_id() -> Optional[str]:
    try:
        from google.auth import default  # type: ignore
        creds, proj = default()
        return proj
    except Exception:
        return None

def _load_risk_cfg():
    try:
        return json.loads(Path(RISK_CFG_PATH).read_text())
    except Exception:
        return {}

# ---------- sources ----------
def _select_data_source() -> Dict[str, Any]:
    """Pick single data source for this render and cache useful paths.
    - If Firestore client available AND creds path present -> use Firestore exclusively.
    - Else use local namespaced files under state/{ENV_KEY} and logs/{ENV_KEY}.
    """
    cli = _fs_client()
    source = "firestore" if cli else "local"
    local_paths = {
        "nav": f"state/{ENV_KEY}/nav_log.json",
        "risk": f"logs/{ENV_KEY}/risk.log",
        "screener": f"logs/{ENV_KEY}/screener_tail.log",
    }
    _DIAG["source"] = source
    if source == "firestore":
        _DIAG["fs_project"] = _get_firestore_project_id()
    return {"source": source, "cli": cli, "local": local_paths}

_DS = _select_data_source()

def load_nav_series() -> List[Dict[str,Any]]:
    # Authoritative selection: Firestore (nested doc) OR Local, not both
    if _DS["source"] == "firestore" and _DS.get("cli"):
        cli = _DS["cli"]
        try:
            p = fs_paths(cli)
            doc = p["nav_doc"].get()
            if not getattr(doc, "exists", False):
                series: List[Dict[str, Any]] = []
            else:
                data = doc.to_dict() or {}
                series = []
                for _, v in (data.items() if isinstance(data, dict) else []):
                    if not isinstance(v, dict):
                        continue
                    try:
                        nav = float(v.get("nav", 0.0))
                    except Exception:
                        nav = 0.0
                    t = v.get("t") or v.get("ts") or v.get("time")
                    if hasattr(t, "timestamp"):
                        try:
                            ts = float(t.timestamp())
                        except Exception:
                            ts = None
                    elif isinstance(t, (int, float)):
                        ts = float(t) / 1000.0 if float(t) > 1e12 else float(t)
                    else:
                        ts = None
                    if ts is None:
                        continue
                    series.append({"ts": ts, "nav": nav})
                series.sort(key=lambda r: r["ts"])  # oldest->newest
            _DIAG["col_nav"] = f"hedge/{ENV}/state/nav"
            return series
        except Exception:
            return []
    # Local fallback (single file)
    p = _DS["local"]["nav"]
    js = _load_json(p, [])
    if isinstance(js, dict):
        return js.get("rows") or js.get("series") or js.get("nav") or []
    return js if isinstance(js, list) else []

def load_positions() -> List[Dict[str,Any]]:
    """Return ONLY open futures positions (non-zero qty). Columns:
    symbol, side, qty, entryPrice, markPrice, unrealizedPnl, leverage, updatedAt
    Read from single document hedge/{ENV}/state/positions. Accept {rows:[..]} or mapping.
    """
    rows: List[Dict[str, Any]] = []
    if _DS["source"] == "firestore" and _DS.get("cli"):
        cli = _DS["cli"]
        try:
            p = fs_paths(cli)
            snap = p["positions_doc"].get()
            data = snap.to_dict() if getattr(snap, "exists", False) else None
            if isinstance(data, dict):
                if isinstance(data.get("rows"), list):
                    rows = list(data.get("rows") or [])
                else:
                    rows = []
                    for sym, rec in data.items():
                        if isinstance(rec, dict):
                            rec = {**rec}
                            rec.setdefault("symbol", sym)
                            rows.append(rec)
            _DIAG["col_positions"] = f"hedge/{ENV}/state/positions"
        except Exception:
            rows = []
    else:
        rows = []

    out: List[Dict[str, Any]] = []
    for r in rows:
        try:
            qty_val = r.get("positionAmt") if r.get("positionAmt") is not None else r.get("qty")
            qty = float(qty_val or 0)
            if abs(qty) <= 0:
                continue
            entry = float(r.get("entryPrice") or r.get("avgEntryPrice") or r.get("entry_price") or 0)
            mark = float(r.get("markPrice") or r.get("mark") or r.get("lastPrice") or r.get("mark_price") or 0)
            upnl = float(r.get("unrealizedPnl") or r.get("uPnl") or r.get("unrealized") or r.get("u_pnl") or 0)
            lev = float(r.get("leverage") or r.get("lev") or 0)
            ts = r.get("updatedAt") or r.get("ts") or r.get("time")
            side = "LONG" if qty > 0 else "SHORT"
            out.append({
                "symbol": r.get("symbol"),
                "side": side,
                "qty": abs(qty),
                "entryPrice": entry,
                "markPrice": mark,
                "unrealizedPnl": upnl,
                "leverage": lev,
                "updatedAt": ts,
            })
        except Exception:
            continue
    # sort by updatedAt/ts desc
    def _sort_key(x: Dict[str, Any]):
        ts = x.get("updatedAt") or x.get("ts") or x.get("time")
        return _to_epoch_seconds(ts)
    out.sort(key=_sort_key, reverse=True)
    return out

def _literal_tail(tag: str, n: int) -> List[str]:
    """Return last n lines from local screener log containing the literal tag.
    Only used when local files are the selected source.
    """
    if _DS["source"] != "local":
        return []
    try:
        with open(_DS["local"]["screener"], "r", errors="ignore") as f:
            lines = f.readlines()[-10000:]
    except Exception:
        return []
    hits = [ln.rstrip("\n") for ln in lines if tag in ln]
    return hits[-n:]

def load_signals_table(limit: int = 100) -> pd.DataFrame:
    """Signals from local screener tail only (when local source).
    Parse `[screener->executor] {...}` and keep last 24h sorted desc.
    """
    import ast
    rows: List[Dict[str, Any]] = []
    if _DS["source"] != "local":
        return pd.DataFrame(rows)
    tag = "[screener->executor]"
    raw = _literal_tail(tag, 200)
    now = time.time()
    for ln in raw:
        try:
            payload = ln.split(tag, 1)[1].strip() if tag in ln else None
            if not payload:
                continue
            obj: Optional[Dict[str,Any]] = None
            if payload.startswith("{"):
                try:
                    obj = json.loads(payload)
                except Exception:
                    try:
                        obj = ast.literal_eval(payload)
                    except Exception:
                        obj = None
            if not isinstance(obj, dict):
                continue
            ts = obj.get("timestamp") or obj.get("t") or obj.get("time")
            t_epoch = _to_epoch_seconds(ts)
            if not is_recent(t_epoch, 24*3600):
                continue
            rows.append({
                "ts": t_epoch,
                "symbol": obj.get("symbol"),
                "tf": obj.get("timeframe"),
                "signal": obj.get("signal"),
                "price": obj.get("price"),
                "cap": obj.get("capital_per_trade"),
                "lev": obj.get("leverage"),
            })
        except Exception:
            continue
    rows.sort(key=lambda x: x.get("ts") or 0.0, reverse=True)
    return pd.DataFrame(rows[:limit])

def load_trade_log(limit: int = 100) -> pd.DataFrame:
    """Trades (last 24h) — Use hedge/{ENV}/trades only. Local returns empty.
    Apply env/testnet filter if fields exist. Order desc.
    """
    rows: List[Dict[str, Any]] = []
    if _DS["source"] == "firestore" and _DS.get("cli"):
        cli = _DS["cli"]
        try:
            p = fs_paths(cli)
            col = p["trades_col"]
            docs = list(col.order_by("ts", direction="DESCENDING").limit(1000).stream())
            for d in docs:
                x = d.to_dict() or {}
                # env/testnet filter if present
                if (x.get("env") is not None and str(x.get("env")) != ENV) or (
                    x.get("testnet") is not None and bool(x.get("testnet")) != TESTNET
                ):
                    continue
                ts = _to_epoch_seconds(x.get("ts") or x.get("time"))
                if not is_recent(ts, 24*3600):
                    continue
                rows.append({
                    "ts": ts,
                    "symbol": x.get("symbol"),
                    "side": x.get("side"),
                    "qty": x.get("qty"),
                    "price": x.get("price"),
                    "pnl": x.get("pnl"),
                })
            _DIAG["col_trades"] = f"hedge/{ENV}/trades"
        except Exception:
            rows = []
        rows.sort(key=lambda x: x.get("ts") or 0.0, reverse=True)
        return pd.DataFrame(rows[:limit])
    # Local (no trades file in this simplified app)
    return pd.DataFrame(rows[:limit])

def load_blocked_orders(limit: int = 200) -> pd.DataFrame:
    """Risk blocks (last 24h) — Use hedge/{ENV}/risk only for Firestore.
    Local: parse risk log if any; else empty.
    """
    rows: List[Dict[str, Any]] = []
    if _DS["source"] == "firestore" and _DS.get("cli"):
        cli = _DS["cli"]
        try:
            p = fs_paths(cli)
            col = p["risk_col"]
            docs = list(col.order_by("ts", direction="DESCENDING").limit(2000).stream())
            for d in docs:
                x = d.to_dict() or {}
                # env/testnet filter if present
                if (x.get("env") is not None and str(x.get("env")) != ENV) or (
                    x.get("testnet") is not None and bool(x.get("testnet")) != TESTNET
                ):
                    continue
                ts = _to_epoch_seconds(x.get("ts") or x.get("t") or x.get("time"))
                if not is_recent(ts, 24*3600):
                    continue
                # keep only blocked if phase exists; otherwise keep all
                if "phase" in x and str(x.get("phase")) != "blocked":
                    continue
                rows.append({
                    "ts": ts,
                    "symbol": x.get("symbol"),
                    "side": x.get("side"),
                    "reason": x.get("reason"),
                    "notional": x.get("notional"),
                    "open_qty": x.get("open_qty"),
                    "gross": x.get("gross"),
                    "nav": x.get("nav"),
                })
            _DIAG["col_risk"] = f"hedge/{ENV}/risk"
        except Exception:
            rows = []
        rows.sort(key=lambda x: x.get("ts") or 0.0, reverse=True)
        return pd.DataFrame(rows[:limit])
    # Local fallback: parse risk log if configured
    if _DS["source"] == "local":
        try:
            with open(_DS["local"]["risk"], "r", errors="ignore") as f:
                lines = f.readlines()[-5000:]
            for ln in lines:
                try:
                    x = json.loads(ln.strip())
                except Exception:
                    continue
                if (x.get("phase") or "") != "blocked":
                    continue
                t_epoch = _to_epoch_seconds(x.get("t") or x.get("ts") or x.get("time"))
                if not is_recent(t_epoch, 24*3600):
                    continue
                rows.append({
                    "ts": t_epoch,
                    "symbol": x.get("symbol"),
                    "side": x.get("side"),
                    "reason": x.get("reason"),
                    "notional": x.get("notional"),
                    "open_qty": x.get("open_qty"),
                    "gross": x.get("gross"),
                    "nav": x.get("nav"),
                })
        except Exception:
            rows = []
    rows.sort(key=lambda x: x.get("ts") or 0.0, reverse=True)
    return pd.DataFrame(rows[:limit])

# ---- time helpers / recency ----
NOW = time.time()

def _to_epoch_seconds(ts: Any) -> float:
    if ts is None:
        return 0.0
    if isinstance(ts, (int, float)):
        x = float(ts)
        return x / 1000.0 if x > 1e12 else x
    try:
        return pd.to_datetime(str(ts), utc=True).timestamp()
    except Exception:
        return 0.0

def is_recent(ts: float, window_sec: int) -> bool:
    if not ts:
        return False
    return (NOW - float(ts)) <= float(window_sec)

def humanize_ago(ts: Optional[float]) -> str:
    if not ts:
        return "(no recent data)"
    delta = max(0.0, NOW - ts)
    if delta < 60:
        return f"{int(delta)}s ago"
    if delta < 3600:
        return f"{int(delta//60)}m ago"
    if delta < 86400:
        return f"{int(delta//3600)}h ago"
    return f"{int(delta//86400)}d ago"

# ---------- KPIs ----------
def compute_nav_kpis(series: List[Dict[str,Any]]) -> Dict[str,Any]:
    if not series:
        return {"nav": None, "nav_24h": None, "delta": 0.0, "delta_pct": 0.0, "dd": None}
    def _ts(v):
        t = v.get("t") or v.get("ts") or v.get("time")
        # Allow epoch seconds/ms or ISO strings
        if isinstance(t, (int,float)):
            x = float(t)
            if x > 1e12:   # ms
                return x/1000.0
            return x       # seconds
        if isinstance(t, str):
            try:
                return pd.to_datetime(t, utc=True).timestamp()
            except Exception:
                return 0.0
        return 0.0
    def _val(v):
        for k in ("nav","value","equity","v"):
            if k in v:
                try:
                    return float(v[k])
                except Exception:
                    pass
        return None

    rows = [( _ts(x), _val(x)) for x in series if isinstance(x, dict)]
    rows = [(t,v) for (t,v) in rows if t and v is not None]
    if not rows:
        return {"nav": None, "nav_24h": None, "delta": 0.0, "delta_pct": 0.0, "dd": None}
    rows.sort(key=lambda x: x[0])
    nav_now = rows[-1][1]
    cutoff = time.time() - 24*3600
    past_vals = [v for (t,v) in rows if t <= cutoff]
    nav_24h = past_vals[-1] if past_vals else nav_now
    peak = max(v for _,v in rows)
    dd = 0.0 if not peak else (peak - nav_now) / peak * 100.0
    delta = nav_now - nav_24h
    delta_pct = (delta / nav_24h * 100.0) if nav_24h else 0.0
    return {"nav": nav_now, "nav_24h": nav_24h, "delta": delta, "delta_pct": delta_pct, "dd": dd}

# ---------- UI ----------
st.set_page_config(page_title="Hedge — Overview", layout="wide")
st.title("Hedge — Overview")

# Source banner with namespace
source_label = "Firestore" if _DS["source"] == "firestore" else "Local"
_top_banner = st.empty()
fs_marker = "N/A"
try:
    if _DS["source"] == "firestore":
        from scripts.fs_doctor import run as fs_check  # type: ignore

        ok, _info = fs_check(ENV)
        fs_marker = "OK" if ok else "Mixed"
except Exception:
    fs_marker = "N/A"

_top_banner.caption(f"Source: {source_label} • ENV_KEY: {ENV_KEY} • FS: {fs_marker}")

# KPIs


series = load_nav_series()
k = compute_nav_kpis(series)
# KPI helpers derived from positions (defined after functions above)
pos_now = load_positions()
open_cnt = sum(1 for p in pos_now if abs(p.get('qty') or 0) > 0)

c1,c2,c3,c4 = st.columns(4)
c1.metric("NAV (USDT)", f"{k['nav']:,.2f}" if k['nav'] is not None else "—", f"{k['delta']:,.2f} / {k['delta_pct']:+.2f}%")
c2.metric("Drawdown", f"{k['dd']:.2f}%" if k['dd'] is not None else "—")
btc_delta = btc_24h_change()
c3.metric("Reserve (BTC)", f"{RESERVE_BTC}", (f"{btc_delta:+.2f}% (24h)" if btc_delta is not None else None))
c4.metric("Open positions", str(open_cnt))

# Risk Status KPI (open gross vs cap)
risk_cfg = _load_risk_cfg()
max_gross_pct = float(((risk_cfg.get("global") or {}).get("max_gross_nav_pct") or 0.0))
open_gross = 0.0
for p in pos_now:
    try:
        open_gross += abs(float(p.get("qty") or 0.0)) * abs(float(p.get("entryPrice") or 0.0))
    except Exception:
        pass
nav_now = k.get('nav') or 0.0
cap = (float(nav_now) * (max_gross_pct/100.0)) if (nav_now and max_gross_pct>0) else 0.0
used_pct = (open_gross / cap * 100.0) if cap else 0.0
c5, = st.columns(1)
c5.metric("Risk Status", f"{open_gross:,.0f} / {cap:,.0f}", f"{used_pct:.1f}% used")

# NAV chart
if series:
    def _parse_ts_val(x):
        t = x.get("t") or x.get("ts") or x.get("time")
        v = x.get("nav") or x.get("value") or x.get("equity") or x.get("v")
        # pandas parse
        if isinstance(t, (int,float)):
            tnum=float(t)
            if tnum>1e12:  # ms
                ts=pd.to_datetime(tnum, unit="ms", utc=True)
            else:
                ts=pd.to_datetime(tnum, unit="s", utc=True)
        else:
            ts=pd.to_datetime(str(t), utc=True, errors="coerce")
        try:
            val=float(v)
        except Exception:
            val=None
        return ts, val

    df = pd.DataFrame([_parse_ts_val(x) for x in series], columns=["ts","nav"]).dropna()
    if not df.empty:
        st.line_chart(df.set_index("ts")["nav"])
    else:
        st.info("NAV series present but could not parse timestamps/values.")
else:
    st.info("No NAV series in Firestore; showing KPIs only (fallback mode).")

st.markdown("---")

# Positions
st.subheader("Positions")
pos = pos_now
if pos:
    dfp = pd.DataFrame(pos).sort_values(by=["symbol","side"])
    st.dataframe(dfp, use_container_width=True, height=260)
else:
    st.write("No open positions")

# Signals
st.subheader("Signals (last 24h)")
df_sig = load_signals_table(100)
if not df_sig.empty:
    newest_sig = float(df_sig["ts"].max()) if "ts" in df_sig else None
    st.caption(f"Last updated: {humanize_ago(newest_sig)}")
    if newest_sig and not is_recent(newest_sig, 1800):
        st.warning("Signals data may be stale (>30m)")
    st.dataframe(df_sig, use_container_width=True, height=210)
else:
    st.write("No recent signals")

# Trade log
st.subheader("Trade Log (last 24h)")
df_tr = load_trade_log(100)
if not df_tr.empty:
    newest_tr = float(df_tr["ts"].max()) if "ts" in df_tr else None
    st.caption(f"Last updated: {humanize_ago(newest_tr)}")
    if newest_tr and not is_recent(newest_tr, 1800):
        st.warning("Trades data may be stale (>30m)")
    st.dataframe(df_tr, use_container_width=True, height=210)
else:
    st.write("No recent trades")

# Risk blocks
st.subheader("Risk Blocks (last 24h)")
df_rb = load_blocked_orders(200)
if not df_rb.empty:
    newest_rb = float(df_rb["ts"].max()) if "ts" in df_rb else None
    st.caption(f"Last updated: {humanize_ago(newest_rb)}")
    if newest_rb and not is_recent(newest_rb, 1800):
        st.warning("Risk blocks may be stale (>30m)")
    st.dataframe(df_rb, use_container_width=True, height=210)
else:
    st.write("No recent risk blocks")

# Screener tail (last 10 lines) — literal tags, no regex
st.subheader("Screener Tail (recent)")
tail = []
if _DS["source"] == "local":
    # cap at 200 lines
    for tag in ("[screener]", "[decision]", "[screener->executor]"):
        tail.extend(_literal_tail(tag, 200))
    tail = tail[-200:]
    # Best-effort freshness from any timestamp-like token
    newest_ts: Optional[float] = None
    for ln in reversed(tail):
        # try to find a timestamp in json-ish payload
        try:
            if "{" in ln and "}" in ln:
                payload = ln.split("{",1)[1]
                payload = "{" + payload.split("}",1)[0] + "}"
                obj = json.loads(payload)
                cand = _to_epoch_seconds(obj.get("timestamp") or obj.get("t") or obj.get("time"))
                if cand:
                    newest_ts = cand
                    break
        except Exception:
            continue
    st.caption(f"Last updated: {humanize_ago(newest_ts)}")
    if newest_ts and not is_recent(newest_ts, 1800):
        st.warning("Screener tail may be stale (>30m)")
st.code("\n".join(tail) if tail else "(empty)")

# Compact recency banner (NAV/Trades/Risk)
try:
    nav_latest_ts = None
    if series:
        try:
            nav_latest_ts = max(
                _to_epoch_seconds(x.get("ts") or x.get("t") or x.get("time"))
                for x in series if isinstance(x, dict)
            )
        except Exception:
            nav_latest_ts = None
    tr_latest_ts = float(df_tr["ts"].max()) if isinstance(df_tr, pd.DataFrame) and not df_tr.empty and "ts" in df_tr else None
    rb_latest_ts = float(df_rb["ts"].max()) if isinstance(df_rb, pd.DataFrame) and not df_rb.empty and "ts" in df_rb else None
    _top_banner.caption(
        f"Source: {'Firestore' if _DS['source']=='firestore' else 'Local'} • ENV_KEY: {ENV_KEY} "
        f"• FS: {fs_marker} • NAV: {humanize_ago(nav_latest_ts)} • Trades: {humanize_ago(tr_latest_ts)} • Risk: {humanize_ago(rb_latest_ts)}"
    )
except Exception:
    # Keep original simple banner if anything goes wrong
    _top_banner.caption(f"Source: {'Firestore' if _DS['source']=='firestore' else 'Local'} • ENV_KEY: {ENV_KEY}")

# Keep banner simple per acceptance

# ---- Doctor panel ----
with st.expander("Doctor", expanded=False):
    try:
        st.write(f"Source: {'Firestore' if _DS['source']=='firestore' else 'Local'} • ENV_KEY: {ENV_KEY}")
        # Paths
        st.write(f"Paths: hedge/{ENV}/state/nav, hedge/{ENV}/state/positions, hedge/{ENV}/trades, hedge/{ENV}/risk")
        pos_cnt = len(pos)
        tr_cnt = int(len(df_tr)) if isinstance(df_tr, pd.DataFrame) else 0
        rb_cnt = int(len(df_rb)) if isinstance(df_rb, pd.DataFrame) else 0
        nav_latest_ts = None
        if series:
            try:
                nav_latest_ts = max(_to_epoch_seconds(x.get('t') or x.get('ts') or x.get('time')) for x in series if isinstance(x, dict))
            except Exception:
                nav_latest_ts = None
        st.write(f"Counts: positions={pos_cnt}, nav={len(series)}, trades(24h)={tr_cnt}, risk(24h)={rb_cnt}")
        st.write(f"Newest NAV ts: {humanize_ago(nav_latest_ts)}")
        # Which collections used (if any)
        if _DS["source"] == "firestore":
            if _DIAG.get("col_trades"):
                st.write(f"Trades collection: {_DIAG.get('col_trades')}")
            if _DIAG.get("col_risk"):
                st.write(f"Risk collection: {_DIAG.get('col_risk')}")
        # Tier badges/counts
        try:
            tiers = _load_json("config/symbol_tiers.json", {}) or {}
            tier_counts = {k: (len(v) if isinstance(v, list) else 0) for k, v in tiers.items()}
            # Open positions by tier: map symbol->tier
            sym2tier = {}
            for t, arr in tiers.items():
                if isinstance(arr, list):
                    for s in arr:
                        sym2tier[str(s).upper()] = t
            pos_by_tier = {}
            for r in pos:
                try:
                    t = sym2tier.get(str(r.get("symbol")).upper(), "OTHER")
                    pos_by_tier[t] = pos_by_tier.get(t, 0) + 1
                except Exception:
                    continue
            st.write(
                "Tiers: "
                + ", ".join(f"{k}:{tier_counts.get(k,0)}" for k in ("CORE","SATELLITE","TACTICAL","ALT-EXT") if k in tier_counts)
            )
            if pos_by_tier:
                st.write(
                    "Open positions by tier: "
                    + ", ".join(f"{k}:{v}" for k, v in pos_by_tier.items())
                )
        except Exception:
            pass
    except Exception:
        st.write("(doctor diagnostics unavailable)")

# --- Exit plans loader (Firestore-first; local fallback) ---
def load_exit_plans() -> pd.DataFrame:
    plans = _fs_get_state("exit_plans")
    rows = []
    if isinstance(plans, dict) and "rows" in plans:
        rows = plans["rows"]
    elif isinstance(plans, list):
        rows = plans
    if not rows:
        local = _load_json("exit_plans.json", [])
        rows = (local.get("rows") if isinstance(local, dict) else local) or []

    out = []
    now = time.time()
    for r in rows:
        try:
            ts = r.get("created_ts") or r.get("ts") or r.get("time")
            if isinstance(ts,(int,float)):
                t_epoch = float(ts)
            else:
                try:
                    t_epoch = pd.to_datetime(ts, utc=True).timestamp()
                except Exception:
                    t_epoch = 0.0
            out.append({
                "symbol": r.get("symbol"),
                "side": r.get("side") or r.get("positionSide"),
                "entry_px": float(r.get("entry_px") or r.get("entryPrice") or 0),
                "sl_px": float(r.get("sl_px") or 0),
                "tp_px": float(r.get("tp_px") or 0),
                "age_min": round((now - t_epoch)/60.0, 1) if t_epoch else None
            })
        except Exception:
            continue
    return pd.DataFrame(out)

# --- UI block (fail-soft) ---
st.subheader("Exit plans (open)")
try:
    dfep = load_exit_plans()
    if not dfep.empty:
        st.dataframe(dfep, use_container_width=True, height=210)
    else:
        st.write("No exit plans recorded yet.")
except Exception as e:
    st.write(f"Exit plans unavailable: {e}")
