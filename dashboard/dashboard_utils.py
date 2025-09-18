import math
import requests

import os
import json
from typing import Any, Dict, List, Tuple

import pandas as pd

# Optional Streamlit caching if running under Streamlit; safe no-op otherwise
try:
    import streamlit as st
    _HAVE_ST = True
except Exception:  # pragma: no cover
    _HAVE_ST = False
    class _Dummy:
        def cache_resource(self, **kwargs):
            def deco(fn):
                return fn
            return deco
    st = _Dummy()

# ---------------------------- Firestore --------------------------------------
_DB = None

@st.cache_resource(show_spinner=False)
def get_firestore_connection():
    """Return a cached Firestore client using FIREBASE_CREDS_PATH if provided.

    Expects GOOGLE Cloud Firestore library to be available. If credentials
    are not provided, it will try Application Default Credentials.
    """
    global _DB
    if _DB is not None:
        return _DB
    try:
        from google.cloud import firestore  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("google-cloud-firestore not installed: pip install google-cloud-firestore") from e

    creds_path = os.getenv("FIREBASE_CREDS_PATH")
    if creds_path and os.path.exists(creds_path):
        os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", creds_path)
    _DB = firestore.Client()
    return _DB


def _doc_path(env: str, name: str) -> Tuple[str, str, str, str]:
    """Canonical path used by the project: hedge/{env}/state/{name}."""
    return ("hedge", env, "state", name)


def fetch_state_document(name: str, env: str = "prod") -> Dict[str, Any]:
    """Load a state document from Firestore. Returns {} if not found.

    Path: hedge/{env}/state/{name}
    """
    db = get_firestore_connection()
    c1, d1, c2, d2 = _doc_path(env, name)
    snap = (
        db.collection(c1).document(d1).collection(c2).document(d2).get()
    )
    return snap.to_dict() or {}

# ---------------------------- NAV helpers ------------------------------------

def _points_from_nav_doc(nav_doc: Dict[str, Any]) -> List[Tuple[int, float]]:
    """Best-effort extraction of [(ts, equity)] pairs from various shapes."""
    # Supported keys: points, nav, equity_curve, series
    candidates = (
        nav_doc.get("points")
        or nav_doc.get("nav")
        or nav_doc.get("equity_curve")
        or nav_doc.get("series")
        or []
    )
    pts: List[Tuple[int, float]] = []
    if isinstance(candidates, list):
        for x in candidates:
            if isinstance(x, dict):
                ts = x.get("ts") or x.get("t") or x.get("time")
                eq = x.get("equity") or x.get("v") or x.get("value")
            elif isinstance(x, (list, tuple)) and len(x) >= 2:
                ts, eq = x[0], x[1]
            else:
                continue
            try:
                ts_i = int(float(ts))
                eq_f = float(eq)
                pts.append((ts_i, eq_f))
            except Exception:
                continue
    return pts


def parse_nav_to_df_and_kpis(nav_doc: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Return (DataFrame, KPIs) where DF has an `equity` column indexed by datetime.

    KPIs has keys: total_equity, peak_equity, drawdown, unrealized_pnl, realized_pnl
    """
    pts = _points_from_nav_doc(nav_doc)
    if not pts:
        df = pd.DataFrame(columns=["equity"])  # empty
        kpis = dict(
            total_equity=float(nav_doc.get("total_equity", 0.0)),
            peak_equity=float(nav_doc.get("peak_equity", 0.0)),
            drawdown=float(nav_doc.get("drawdown", 0.0)),
            unrealized_pnl=float(nav_doc.get("unrealized_pnl", 0.0)),
            realized_pnl=float(nav_doc.get("realized_pnl", 0.0)),
        )
        return df, kpis

    # Normalize to DataFrame
    ts = [t/1000 if t > 10_000_000_000 else t for t, _ in pts]  # ms -> s if needed
    eq = [v for _, v in pts]
    idx = pd.to_datetime(ts, unit="s", utc=True)
    df = pd.DataFrame({"equity": eq}, index=idx).sort_index()

    total_equity = float(eq[-1])
    peak_equity = float(max(eq)) if eq else total_equity
    drawdown = 0.0 if peak_equity <= 0 else max(0.0, (peak_equity - total_equity) / peak_equity)

    kpis = dict(
        total_equity=total_equity,
        peak_equity=float(nav_doc.get("peak_equity", peak_equity)),
        drawdown=float(nav_doc.get("drawdown", drawdown)),
        unrealized_pnl=float(nav_doc.get("unrealized_pnl", 0.0)),
        realized_pnl=float(nav_doc.get("realized_pnl", 0.0)),
    )
    return df, kpis

# ---------------------------- Misc helpers -----------------------------------

def positions_sorted(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort open positions by descending notional (abs)."""
    def _abs_notional(x: Dict[str, Any]) -> float:
        try:
            return abs(float(x.get("notional", 0.0)))
        except Exception:
            return 0.0
    return sorted(items or [], key=_abs_notional, reverse=True)


def read_trade_log_tail(path: str, tail: int = 10) -> List[Dict[str, Any]]:
    """Read last N trades from either JSON array or JSON Lines file.
    Returns empty list if file missing or malformed.
    """
    try:
        with open(path, "r") as f:
            txt = f.read().strip()
    except Exception:
        return []

    if not txt:
        return []

    # Try JSON array first
    try:
        data = json.loads(txt)
        if isinstance(data, list):
            return data[-tail:]
    except Exception:
        pass

    # Fallback: JSON Lines
    rows: List[Dict[str, Any]] = []
    for ln in txt.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            obj = json.loads(ln)
            if isinstance(obj, dict):
                rows.append(obj)
        except Exception:
            continue
    return rows[-tail:]


def fmt_ccy(x: Any) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return str(x)


def fmt_pct(x: Any) -> str:
    try:
        return f"{float(x)*100:.2f}%"
    except Exception:
        return str(x)

# --- Price helpers (read-only) ------------------------------------------------

def get_env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default

def fetch_mark_price_usdt(symbol: str = "BTCUSDT", timeout: float = 4.0) -> float:
    """
    Read-only quote for display purposes.
    Prefers futures mark price; falls back to spot ticker if needed.
    """
    # 1) Futures testnet mark (premiumIndex)
    fut = "https://testnet.binancefuture.com/fapi/v1/premiumIndex"
    try:
        r = requests.get(fut, params={"symbol": symbol}, timeout=timeout)
        if r.ok:
            px = float(r.json().get("markPrice", 0.0))
            if math.isfinite(px) and px > 0:
                return px
    except Exception:
        pass

    # 2) Spot testnet/backup (display-only)
    spot = "https://testnet.binance.vision/api/v3/ticker/price"
    try:
        r = requests.get(spot, params={"symbol": symbol}, timeout=timeout)
        if r.ok:
            px = float(r.json().get("price", 0.0))
            if math.isfinite(px) and px > 0:
                return px
    except Exception:
        pass
    return 0.0
