#!/usr/bin/env python3
# Streamlit dashboard (single Overview tab).
# Read-only and fast. Firestore-first; fall back to local files.
# Shows: KPIs, NAV chart, Positions, Signals, Trade log (5), Screener tail (10), Reserve.

from __future__ import annotations
import os, json, time, pathlib, re
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

ENV = os.getenv("ENV","prod")
RESERVE_BTC = 0.013
LOG_PATH = "/var/log/hedge-executor.out.log"
TAIL_LINES = 10

# --- helpers: safe JSON load ---
def _load_json(path: str, default=None):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default

# --- Firestore (best effort) ---
def _fs_client():
    try:
        from google.cloud import firestore
        return firestore.Client()
    except Exception:
        return None

def _fs_get_state(doc: str):
    cli = _fs_client()
    if not cli: return None
    try:
        snap = cli.document(f"hedge/{ENV}/state/{doc}").get()
        return snap.to_dict() if snap.exists else None
    except Exception:
        return None

# --- Data sources ---
def load_nav_series() -> List[Dict[str,Any]]:
    # Firestore first
    nav = _fs_get_state("nav")
    if isinstance(nav, dict) and "series" in nav:
        return nav["series"]
    if isinstance(nav, list):
        return nav
    # Local fallback
    peak = _load_json("peak_state.json", {})
    if isinstance(peak, dict) and "nav" in peak and isinstance(peak["nav"], list):
        return peak["nav"]
    return []

def load_positions() -> List[Dict[str,Any]]:
    pos = _fs_get_state("positions")
    if isinstance(pos, dict) and "rows" in pos:
        rows = pos["rows"]
    elif isinstance(pos, list):
        rows = pos
    else:
        rows = []
    # ensure normalized columns
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

def _parse_log_tail(pattern: str, n: int) -> List[str]:
    try:
        with open(LOG_PATH, "r", errors="ignore") as f:
            lines = f.readlines()[-2000:]  # bounded read
    except Exception:
        return []
    matcher = re.compile(pattern)
    hits = [ln.strip() for ln in lines if matcher.search(ln)]
    return hits[-n:]

def load_signals_table(n: int = 5) -> pd.DataFrame:
    pat = r"\[screener->executor\]\s+(.*)"
    raw = _parse_log_tail(pat, 50)
    rows = []
    for ln in raw[-n:]:
        try:
            jtxt = ln.split("]", 1)[1].strip()
            obj = json.loads(jtxt)
            rows.append({
                "time": obj.get("timestamp"),
                "symbol": obj.get("symbol"),
                "tf": obj.get("timeframe"),
                "signal": obj.get("signal"),
                "price": obj.get("price"),
                "cap": obj.get("capital_per_trade"),
                "lev": obj.get("leverage")
            })
        except Exception:
            continue
    return pd.DataFrame(rows)

def load_trade_log(n: int = 5) -> pd.DataFrame:
    # Prefer local trade_log.json if present
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
    # Fallback: show ORDER_REQ lines (not fills, but pipeline visibility)
    pat = r"\[executor\]\s+ORDER_REQ.*"
    raw = _parse_log_tail(pat, 40)[-n:]
    for ln in raw:
        rows.append({"time":"-", "symbol":"-", "side":"-", "qty":"-", "price":"-", "pnl":"-", "raw": ln})
    return pd.DataFrame(rows)

# --- NAV KPIs ---
def compute_nav_kpis(series: List[Dict[str,Any]]) -> Dict[str,Any]:
    # accept items with keys {t/nav} or {ts/value/equity}
    if not series:
        return {"nav": None, "nav_24h": None, "delta": 0.0, "delta_pct": 0.0, "dd": None}
    def _ts(v):
        return float(v.get("t") or v.get("ts") or v.get("time") or 0.0)
    def _val(v):
        for k in ("nav","value","equity"):
            if k in v: return float(v[k])
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
    dd = 0.0 if nav_now is None else (0.0 if peak == 0 else (peak - nav_now) / peak * 100.0)
    delta = nav_now - nav_24h
    delta_pct = (delta / nav_24h * 100.0) if nav_24h else 0.0
    return {"nav": nav_now, "nav_24h": nav_24h, "delta": delta, "delta_pct": delta_pct, "dd": dd}

# === UI ===
st.set_page_config(page_title="Hedge — Overview", layout="wide")
st.title("Hedge — Overview")
st.caption(f"ENV: {ENV} · Read-only dashboard")

# KPIs
series = load_nav_series()
k = compute_nav_kpis(series)
c1,c2,c3,c4 = st.columns(4)
c1.metric("NAV (USDT)", f"{k['nav']:,.2f}" if k['nav'] is not None else "—", f"{k['delta']:,.2f} / {k['delta_pct']:+.2f}%")
c2.metric("Drawdown", f"{k['dd']:.2f}%" if k['dd'] is not None else "—")
c3.metric("Reserve (BTC)", f"{RESERVE_BTC}")
c4.metric("Tail lines", f"{TAIL_LINES}")

# NAV chart
if series:
    df = pd.DataFrame([{"ts": (x.get("t") or x.get("ts") or x.get("time")), "nav": (x.get("nav") or x.get("value") or x.get("equity"))} for x in series])
    df["ts"] = pd.to_datetime(df["ts"], unit="s", errors="coerce")
    df = df.dropna()
    st.line_chart(df.set_index("ts")["nav"])
else:
    st.info("No NAV series in Firestore; showing KPIs only (fallback mode).")

st.markdown("---")

# Positions
st.subheader("Positions")
pos = load_positions()
if pos:
    dfp = pd.DataFrame(pos)
    dfp = dfp.sort_values(by=["symbol","side"])
    st.dataframe(dfp, use_container_width=True, height=260)
else:
    st.write("No positions available in Firestore.")

# Signals
st.subheader("Signals (latest 5)")
st.dataframe(load_signals_table(5), use_container_width=True, height=210)

# Trade log
st.subheader("Trade Log (latest 5)")
st.dataframe(load_trade_log(5), use_container_width=True, height=210)

# Screener tail (last 10 lines)
st.subheader("Screener Tail (last 10)")
tail = []
for tag in (r"\[screener\]", r"\[decision\]", r"\[screener->executor\]"):
    tail.extend(_parse_log_tail(tag, TAIL_LINES))
# keep last 10 only
tail = tail[-TAIL_LINES:]
st.code("\n".join(tail) if tail else "(empty)")
