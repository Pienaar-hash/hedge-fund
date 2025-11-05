import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

import pandas as pd
import requests

from utils.firestore_client import get_db as _get_firestore_db
from execution.mirror_builders import build_mirror_payloads

LOG = logging.getLogger("dash.utils")

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
        def cache_data(self, **kwargs):
            def deco(fn):
                return fn
            return deco
    st = _Dummy()

# ---------------------------- Firestore --------------------------------------

@st.cache_resource(show_spinner=False)
def _build_firestore_connection():
    """Return a cached Firestore client via utils.firestore_client.

    Raises RuntimeError when Firestore is unavailable so Streamlit does not cache failures.
    """
    try:
        db = _get_firestore_db(strict=False)
    except Exception:
        raise
    if getattr(db, "_is_noop", False):
        raise RuntimeError("Firestore returned noop client")
    return db


def get_firestore_connection():
    """Best-effort Firestore client that retries initialization on subsequent calls."""
    try:
        return _build_firestore_connection()
    except Exception as exc:
        LOG.debug("[dash] firestore connection unavailable: %s", exc)
        return None


def _doc_path(env: str, name: str) -> Tuple[str, str, str, str]:
    """Canonical path used by the project: hedge/{env}/state/{name}."""
    return ("hedge", env, "state", name)


@st.cache_data(ttl=5, show_spinner=False)
def fetch_state_document(name: str, env: str = "prod") -> Dict[str, Any]:
    """Load a state document from Firestore. Returns {} if not found.

    Path: hedge/{env}/state/{name}
    """
    db = get_firestore_connection()
    if db is None:
        return {}
    c1, d1, c2, d2 = _doc_path(env, name)
    try:
        snap = (
            db.collection(c1)
            .document(d1)
            .collection(c2)
            .document(d2)
            .get(timeout=3.0)
        )
        return snap.to_dict() or {}
    except Exception as exc:
        LOG.debug("[dash] state fetch failed %s/%s: %s", env, name, exc)
        return {}


@st.cache_data(ttl=5, show_spinner=False)
def fetch_telemetry_health(env: str = "prod") -> Dict[str, Any]:
    """Return sync_state telemetry health document (hedge/{env}/telemetry/health)."""
    db = get_firestore_connection()
    if db is None:
        return {}
    try:
        snap = (
            db.collection("hedge")
            .document(env)
            .collection("telemetry")
            .document("health")
            .get(timeout=3.0)
        )
        if hasattr(snap, "to_dict"):
            data = snap.to_dict() or {}
            return data if isinstance(data, dict) else {}
    except Exception as exc:
        LOG.debug("[dash] telemetry health fetch failed for %s: %s", env, exc)
    return {}


@st.cache_data(ttl=5, show_spinner=False)
def load_exec_snapshot(kind: str, env: str = "prod") -> Dict[str, Any]:
    """Return router/trade/signal snapshot preferring Firestore mirror with local fallback."""
    kind_norm = str(kind or "").strip().lower()
    db = get_firestore_connection()
    if db is not None and kind_norm:
        try:
            if kind_norm == "signals":
                doc_ref = (
                    db.collection("hedge")
                    .document(env)
                    .collection("signals")
                    .document("latest")
                )
            else:
                doc_ref = (
                    db.collection("hedge")
                    .document(env)
                    .collection("executions")
                    .document(kind_norm)
                )
            snap = doc_ref.get(timeout=3.0)
            data = snap.to_dict() or {}
            if isinstance(data, dict) and data.get("items"):
                data.setdefault("source", "firestore")
                return data
        except Exception as exc:
            LOG.debug("[dash] exec snapshot fetch failed (%s/%s): %s", env, kind_norm, exc)
    try:
        payloads = build_mirror_payloads(Path(__file__).resolve().parents[1] / "logs")
    except Exception as exc:
        LOG.debug("[dash] exec snapshot local fallback failed: %s", exc)
        return {}
    mapping = {
        "router": payloads.router,
        "trades": payloads.trades,
        "signals": payloads.signals,
    }
    items = mapping.get(kind_norm or "router", [])
    if not items:
        return {}
    return {
        "items": items,
        "count": len(items),
        "ts_iso": datetime.now(timezone.utc).isoformat(),
        "source": "local",
    }


def _to_float(value: Any) -> float:
    try:
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return 0.0
        return float(value)
    except Exception:
        return 0.0


def load_treasury_cache(payload: Any | None = None, path: str | None = None) -> Dict[str, Any]:
    """Return normalized treasury cache with assets, totals, and timestamp."""

    if payload is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        target = path or os.path.join(base_dir, "logs", "treasury.json")
        try:
            with open(target, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            payload = {}

    if not isinstance(payload, dict):
        return {"assets": [], "total_usd": 0.0, "updated_at": None}

    assets: List[Dict[str, Any]] = []

    raw_assets = payload.get("assets") if isinstance(payload.get("assets"), list) else None
    if isinstance(raw_assets, list):
        for item in raw_assets:
            if not isinstance(item, dict):
                continue
            symbol = str(
                item.get("Asset")
                or item.get("asset")
                or item.get("symbol")
                or item.get("code")
                or ""
            ).upper()
            if not symbol:
                continue
            units = _to_float(
                item.get("Units")
                or item.get("units")
                or item.get("qty")
                or item.get("balance")
                or item.get("amount")
            )
            usd_val = _to_float(
                item.get("USD Value")
                or item.get("usd_value")
                or item.get("value_usd")
                or item.get("usd")
            )
            assets.append({"Asset": symbol, "Units": units, "USD Value": usd_val})
    else:
        for key, value in payload.items():
            key_lower = str(key).lower()
            if key_lower in {"total_usd", "updated_at", "source"}:
                continue
            symbol = str(key).upper()
            units: float
            usd_val: float
            if isinstance(value, dict):
                units = _to_float(
                    value.get("qty")
                    or value.get("Units")
                    or value.get("units")
                    or value.get("balance")
                    or value.get("amount")
                )
                usd_val = _to_float(
                    value.get("usd_value")
                    or value.get("USD Value")
                    or value.get("value_usd")
                    or value.get("usd")
                )
            else:
                units = _to_float(value)
                usd_val = units
            if symbol:
                assets.append({"Asset": symbol, "Units": units, "USD Value": usd_val})

    total_usd = _to_float(payload.get("total_usd")) if isinstance(payload, dict) else 0.0
    if total_usd <= 0 and assets:
        total_usd = sum(_to_float(item.get("USD Value")) for item in assets)

    return {
        "assets": assets,
        "total_usd": float(total_usd),
        "updated_at": payload.get("updated_at"),
        "raw": payload,
    }


def load_nav_snapshot(*, base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Return the first available NAV snapshot for the dashboard doctor view."""

    root = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parents[1]
    candidates = [
        root / "logs" / "cache" / "nav_confirmed.json",
        root / "logs" / "nav_snapshot.json",
        root / "logs" / "nav_log.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            continue
        if isinstance(payload, dict):
            payload.setdefault("source", path.name)
            return payload
    return {}


def load_recent_logs(kind: str, limit: int = 200) -> Any:
    """
    Render the tail of logs/<kind>_log.json inside Streamlit.

    Falls back to a text area when the payload cannot be parsed as a tabular list.
    Automatically schedules a refresh every 15 seconds (no-op if st_autorefresh unavailable).
    """
    if not _HAVE_ST:
        raise RuntimeError("Streamlit context required for load_recent_logs")

    try:
        from streamlit_extras.st_autorefresh import st_autorefresh  # type: ignore

        st_autorefresh(interval=15_000, key=f"log-refresh-{kind}")
    except Exception:  # pragma: no cover - optional dependency
        pass

    base_dir = Path(__file__).resolve().parents[1]
    path = base_dir / "logs" / f"{kind}_log.json"
    if not path.exists():
        st.warning(f"log file not found: {path}")
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = None

    if isinstance(payload, list) and payload:
        rows = payload[-limit:]
        try:
            df = pd.DataFrame(rows)
        except ValueError:
            df = pd.DataFrame({"value": rows})
        if df.empty:
            st.info("No recent entries.")
            return df
        st.dataframe(df, use_container_width=True, height=min(600, 60 + len(df) * 28))
        return df

    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        st.error(f"Unable to read log file: {exc}")
        return None

    lines = text.splitlines()
    tail = "\n".join(lines[-limit:]) if lines else ""
    st.text_area(
        f"{kind} log tail",
        tail or "(empty)",
        height=320,
    )
    return tail


def compute_total_nav_cached() -> Tuple[Dict[str, Any], float]:
    """
    Compute NAV from cached files instead of hitting live APIs.
    Reads:
      - logs/nav_log.json       → Binance Futures NAV
      - logs/spot_state.json    → Binance Spot balances (if exists)
      - logs/treasury.json      → Off-exchange reserves
      - logs/polymarket.json    → Polymarket holdings
    Returns (sources_dict, total_nav) where sources_dict has per-source NAV/asset data.
    """

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    def _abs_path(rel: str) -> str:
        return os.path.join(base_dir, rel)

    def safe_load(rel_path: str, label: str) -> Tuple[Any, bool]:
        path = _abs_path(rel_path)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f), False
            except Exception:
                return {}, False
        print(f"[dash] missing cache file: {label}", flush=True)
        return {}, True

    nav_data, nav_missing = safe_load("logs/nav_log.json", "nav_log.json")
    spot_data, spot_missing = safe_load("logs/spot_state.json", "spot_state.json")
    treasury_data, treasury_missing = safe_load("logs/treasury.json", "treasury.json")
    treasury_cache = load_treasury_cache(treasury_data)
    poly_data, poly_missing = safe_load("logs/polymarket.json", "polymarket.json")

    def extract_equity(payload: Any) -> float:
        if isinstance(payload, dict):
            try:
                return float(
                    payload.get("total_equity")
                    or payload.get("equity")
                    or payload.get("nav")
                    or payload.get("total_usd")
                    or 0.0
                )
            except Exception:
                return 0.0
        if isinstance(payload, list) and payload:
            last = payload[-1]
            if isinstance(last, dict):
                try:
                    return float(
                        last.get("equity")
                        or last.get("nav")
                        or last.get("total_equity")
                        or 0.0
                    )
                except Exception:
                    return 0.0
        return 0.0

    futures_nav = extract_equity(nav_data)

    def extract_assets(payload: Any) -> List[Dict[str, float | str]]:
        assets: List[Dict[str, float | str]] = []
        if isinstance(payload, dict):
            if isinstance(payload.get("assets"), list):
                for item in payload.get("assets") or []:
                    if not isinstance(item, dict):
                        continue
                    symbol = str(
                        item.get("symbol")
                        or item.get("asset")
                        or item.get("Asset")
                        or item.get("code")
                        or ""
                    ).upper()
                    units = _to_float(
                        item.get("balance")
                        or item.get("amount")
                        or item.get("qty")
                        or item.get("units")
                        or item.get("Units")
                        or item.get("free")
                    )
                    usd_val = _to_float(
                        item.get("usd_value")
                        or item.get("value_usd")
                        or item.get("USD Value")
                        or item.get("usd")
                        or item.get("notional_usd")
                        or item.get("val_usdt")
                    )
                    if symbol:
                        assets.append(
                            {
                                "Asset": symbol,
                                "Units": units,
                                "USD Value": usd_val,
                            }
                        )
                return assets
            if isinstance(payload.get("balances"), list):
                for item in payload.get("balances") or []:
                    if not isinstance(item, dict):
                        continue
                    symbol = str(item.get("asset") or item.get("symbol") or "").upper()
                    units = _to_float(
                        item.get("balance")
                        or item.get("free")
                        or item.get("amount")
                        or item.get("qty")
                        or item.get("units")
                    )
                    usd_val = _to_float(
                        item.get("usd_value")
                        or item.get("value_usd")
                        or item.get("usd")
                        or item.get("val_usdt")
                    )
                    if symbol:
                        assets.append(
                            {"Asset": symbol, "Units": units, "USD Value": usd_val}
                        )
                return assets
            for key, value in payload.items():
                key_lower = str(key).lower()
                if key_lower in {"total_usd", "updated_at", "source"}:
                    continue
                if key.lower() == "total_usd":
                    continue
                symbol = str(key).upper()
                if isinstance(value, dict):
                    units = _to_float(
                        value.get("qty")
                        or value.get("amount")
                        or value.get("balance")
                        or value.get("units")
                        or value.get("Units")
                        or value.get("free")
                    )
                    usd_val = _to_float(
                        value.get("usd_value")
                        or value.get("value_usd")
                        or value.get("usd")
                        or value.get("val_usdt")
                    )
                else:
                    units = _to_float(value)
                    usd_val = units
                assets.append({"Asset": symbol, "Units": units, "USD Value": usd_val})
        return assets

    spot_assets = extract_assets(spot_data)
    treasury_assets = treasury_cache.get("assets") or extract_assets(treasury_data)
    poly_assets = extract_assets(poly_data)

    def sum_usd(assets: List[Dict[str, Any]]) -> float:
        total = 0.0
        for item in assets:
            try:
                total += float(item.get("USD Value", 0.0))
            except Exception:
                continue
        return total

    spot_nav = float(spot_data.get("total_usd", 0.0) or 0.0)
    if spot_nav <= 0 and spot_assets:
        spot_nav = sum_usd(spot_assets)

    treasury_nav = float(treasury_cache.get("total_usd") or treasury_data.get("total_usd", 0.0) or 0.0)
    if treasury_nav <= 0 and treasury_assets:
        treasury_nav = sum_usd(treasury_assets)
    treasury_updated_at = treasury_cache.get("updated_at")

    poly_nav = float(
        (poly_data.get("total_usd") if isinstance(poly_data, dict) else 0.0) or 0.0
    )
    if poly_nav <= 0 and poly_assets:
        poly_nav = sum_usd(poly_assets)

    total_nav = futures_nav + spot_nav + treasury_nav + poly_nav

    print(
        (
            "[dash] NAV composition: "
            f"futures={futures_nav:.2f}, spot={spot_nav:.2f}, "
            f"treasury={treasury_nav:.2f}, poly={poly_nav:.2f} "
            f"→ total={total_nav:.2f}"
        ),
        flush=True,
    )

    def _label_assets(items: List[Dict[str, Any]], source_name: str) -> List[Dict[str, Any]]:
        labeled: List[Dict[str, Any]] = []
        for entry in items:
            entry = dict(entry)
            entry.setdefault("Asset", source_name.upper())
            entry["Source"] = source_name
            labeled.append(entry)
        return labeled

    combined_assets = (
        _label_assets(spot_assets, "Spot")
        + _label_assets(treasury_assets, "Treasury")
        + _label_assets(poly_assets, "Poly")
    )

    sources: Dict[str, Dict[str, Any]] = {
        "futures": {
            "label": "Futures",
            "nav": futures_nav,
            "assets": [],
            "missing": nav_missing,
        },
        "spot": {
            "label": "Spot",
            "nav": spot_nav,
            "assets": _label_assets(spot_assets, "Spot"),
            "missing": spot_missing,
        },
        "treasury": {
            "label": "Treasury",
            "nav": treasury_nav,
            "assets": _label_assets(treasury_assets, "Treasury"),
            "missing": treasury_missing,
            "updated_at": treasury_updated_at,
        },
        "poly": {
            "label": "Poly",
            "nav": poly_nav,
            "assets": _label_assets(poly_assets, "Poly"),
            "missing": poly_missing,
        },
        "combined": {
            "label": "All",
            "nav": total_nav,
            "assets": combined_assets,
            "missing": spot_missing and treasury_missing and poly_missing,
        },
    }

    return sources, total_nav

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
import math

def get_env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default

def get_env_badge(testnet_flag: Any) -> Tuple[str, str]:
    """Return (label, background_hex) for environment badges."""
    value = str(testnet_flag).strip().lower()
    if value in {"1", "true", "yes", "on", "testnet"}:
        return "TESTNET", "#ea580c"
    return "LIVE", "#059669"


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
