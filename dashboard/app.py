#!/usr/bin/env python3
from __future__ import annotations
"""Streamlit dashboard (single "Overview" tab), read-only.
Firestore-first; falls back to local files for NAV.
Shows: KPIs, NAV chart, Positions, Signals (5), Trade log (5), Screener tail (10), BTC reserve.
"""
# Streamlit dashboard (single "Overview" tab), read-only.
# Firestore-first; falls back to local files for NAV.

# ---- tolerant dotenv ----
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

import os, json, time, pathlib
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

ENV = os.getenv("ENV","prod")
RESERVE_BTC = float(os.getenv("RESERVE_BTC", "0.025"))
LOG_PATH = os.getenv("EXECUTOR_LOG", "/var/log/hedge-executor.out.log")
TAIL_LINES = 10

# ---------- helpers ----------
def btc_24h_change() -> float | None:
    import requests
    tn = os.getenv("BINANCE_TESTNET","1") in ("1","true","True")
    base = "https://testnet.binancefuture.com" if tn else "https://fapi.binance.com"
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
    try:
        from google.cloud import firestore  # type: ignore
        return firestore.Client()
    except Exception:
        return None

def _fs_get_state(doc: str):
    cli = _fs_client()
    if not cli: return None
    try:
        snap = cli.document(f"hedge/{ENV}/state/{doc}").get()
        return snap.to_dict() if getattr(snap, "exists", False) else None
    except Exception:
        return None

# ---------- sources ----------
def load_nav_series() -> List[Dict[str,Any]]:
    # Firestore first
    nav = _fs_get_state("nav")
    if isinstance(nav, dict):
        for k in ("series","nav_series","nav"):
            if isinstance(nav.get(k), list):
                return nav[k]
    if isinstance(nav, list):
        return nav
    # Local fallback
    peak = _load_json("peak_state.json", {})
    if isinstance(peak, dict):
        for k in ("nav_series","nav"):
            if isinstance(peak.get(k), list):
                return peak[k]
    return []

def load_positions() -> List[Dict[str,Any]]:
    pos = _fs_get_state("positions")
    rows: List[Dict[str,Any]] = []
    if isinstance(pos, dict) and isinstance(pos.get("rows"), list):
        rows = pos["rows"]
    elif isinstance(pos, list):
        rows = pos
    out = []
    for r in rows:
        try:
            out.append({
                "symbol": r.get("symbol"),
                "side": r.get("positionSide") or r.get("side") or "BOTH",
                "qty": float(r.get("qty") or r.get("positionAmt") or 0),
                "entryPrice": float(r.get("entryPrice") or 0),
                "unrealized": float(r.get("unrealized") or r.get("uPnl") or 0),
                "leverage": float(r.get("leverage") or 0),
            })
        except Exception:
            continue
    return out

def _literal_tail(tag: str, n: int) -> List[str]:
    """Return last n lines in LOG_PATH containing the literal tag (no regex)."""
    try:
        with open(LOG_PATH, "r", errors="ignore") as f:
            lines = f.readlines()[-4000:]
    except Exception:
        return []
    hits = [ln.rstrip("\n") for ln in lines if tag in ln]
    return hits[-n:]

def load_signals_table(n: int = 5) -> pd.DataFrame:
    """Parse `[screener->executor] {dict or json}` lines without regex."""
    import ast
    tag = "[screener->executor]"
    raw = _literal_tail(tag, 80)
    rows = []
    for ln in raw[-n:]:
        try:
            if tag not in ln:
                continue
            payload = ln.split(tag, 1)[1].strip()
            if not payload:
                continue
            obj: Optional[Dict[str,Any]] = None
            if payload.startswith("{"):
                # Try JSON first (double quotes), then Python-literal (single quotes)
                try:
                    obj = json.loads(payload)
                except Exception:
                    try:
                        obj = ast.literal_eval(payload)
                    except Exception:
                        obj = None
            if not isinstance(obj, dict):
                continue
            rows.append({
                "time": obj.get("timestamp") or obj.get("t") or obj.get("time"),
                "symbol": obj.get("symbol"),
                "tf": obj.get("timeframe"),
                "signal": obj.get("signal"),
                "price": obj.get("price"),
                "cap": obj.get("capital_per_trade"),
                "lev": obj.get("leverage"),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)

def load_trade_log(n: int = 5) -> pd.DataFrame:
    tl = _load_json("trade_log.json", [])
    rows = []
    if isinstance(tl, list) and tl:
        for x in tl[-n:]:
            try:
                rows.append({
                    "time": x.get("time") or x.get("ts"),
                    "symbol": x.get("symbol"),
                    "side": x.get("side"),
                    "qty": x.get("qty"),
                    "price": x.get("price"),
                    "pnl": x.get("pnl"),
                })
            except Exception:
                continue
        return pd.DataFrame(rows)
    # Fallback: show ORDER_REQ lines from executor log
    tag = "[executor] ORDER_REQ"
    raw = _literal_tail(tag, 40)[-n:]
    for ln in raw:
        rows.append({"time":"-", "symbol":"-", "side":"-", "qty":"-", "price":"-", "pnl":"-", "raw": ln})
    return pd.DataFrame(rows)

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
                try: return float(v[k])
                except Exception: pass
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
st.caption(f"ENV: {ENV} · Read-only dashboard")

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
    st.write("No positions available in Firestore.")

# Signals
st.subheader("Signals (latest 5)")
st.dataframe(load_signals_table(5), use_container_width=True, height=210)

# Trade log
st.subheader("Trade Log (latest 5)")
st.dataframe(load_trade_log(5), use_container_width=True, height=210)

st.subheader("Exit plans (open)")
dfep = load_exit_plans()
if not dfep.empty:
    dfep = dfep.sort_values(by=["symbol","side"]) 
    st.dataframe(dfep, use_container_width=True, height=220)
else:
    st.write("No open exit plans.")

st.dataframe(load_trade_log(5), use_container_width=True, height=210)

# Screener tail (last 10 lines) — literal tags, no regex
st.subheader("Screener Tail (last 10)")
tail = []
for tag in ("[screener]", "[decision]", "[screener->executor]"):
    tail.extend(_literal_tail(tag, TAIL_LINES))
tail = tail[-TAIL_LINES:]
st.code("\n".join(tail) if tail else "(empty)")

def load_exit_plans():
    """Fetch open exit plans from Firestore. Fallback to exit_plans.json if present."""
    import time, pandas as pd
    rows = []
    cli = _fs_client()
    now = time.time()
    if cli:
        try:
            col = cli.collection(f"hedge/{ENV}/state")
            for ref in col.list_documents():
                doc_id = getattr(ref, "id", "")
                if not str(doc_id).startswith("exits_"):
                    continue
                d = (ref.get().to_dict() or {})
                if not d or d.get("closed_ts"):
                    continue
                rows.append({
                    "symbol": d.get("symbol"),
                    "side": d.get("positionSide") or d.get("side"),
                    "entry_px": float(d.get("entry_px") or 0.0),
                    "sl_px": float(d.get("sl_px") or 0.0),
                    "tp_px": float(d.get("tp_px") or 0.0),
                    "age_min": round(max(0.0, (now - float(d.get("entry_ts") or now)) / 60.0), 1),
                })
        except Exception:
            pass
    if not rows:
        # local fallback
        loc = _load_json("exit_plans.json", [])
        if isinstance(loc, list):
            for d in loc:
                try:
                    rows.append({
                        "symbol": d.get("symbol"),
                        "side": d.get("positionSide") or d.get("side"),
                        "entry_px": float(d.get("entry_px") or 0.0),
                        "sl_px": float(d.get("sl_px") or 0.0),
                        "tp_px": float(d.get("tp_px") or 0.0),
                        "age_min": float(d.get("age_min") or 0.0),
                    })
                except Exception:
                    continue
    import pandas as pd
    return pd.DataFrame(rows)

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
