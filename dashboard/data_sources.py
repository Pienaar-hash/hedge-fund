from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


try:  # Streamlit may not be present under unit tests
    import streamlit as st
except Exception:  # pragma: no cover - local tooling fallback
    class _StubStreamlit:  # minimal interface for cache decorators
        def cache_data(self, **kwargs):
            def _decorator(fn):
                return fn

            return _decorator

        def cache_resource(self, **kwargs):
            def _decorator(fn):
                return fn

            return _decorator

    st = _StubStreamlit()  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NAV_SNAPSHOT = PROJECT_ROOT / "state" / "nav_snapshot.json"
DEFAULT_NAV_HISTORY = PROJECT_ROOT / "logs" / "nav.jsonl"
DEFAULT_POSITIONS_LOG = PROJECT_ROOT / "logs" / "positions.jsonl"
DEFAULT_SCREENER_LOG = PROJECT_ROOT / "logs" / "screener_tail.log"
DEFAULT_VETO_PATTERN = "veto_exec_*.json"

LOCAL_NAV = os.environ.get("LOCAL_NAV_PATH", str(DEFAULT_NAV_HISTORY))
FIRESTORE_ENABLED = os.environ.get("FIRESTORE_ENABLED", "0") == "1"
FIRESTORE_COLLECTION = os.environ.get("FIRESTORE_NAV_COL", "nav_snapshots")
STALE_AFTER_SEC = int(os.environ.get("DASH_STALE_AFTER_SEC", "120"))

try:  # optional Firestore helper import (shared cache in dashboard_utils)
    from utils.firestore_client import get_db as _get_db
except Exception:  # pragma: no cover - optional dependency fallback
    _get_db = None


ENV = os.getenv("ENV", "prod")
TESTNET = os.getenv("BINANCE_TESTNET", "0") == "1"
ENV_KEY = f"{ENV}{'-testnet' if TESTNET else ''}"


@dataclass
class DataSet:
    data: Any
    source: str
    updated_at: Optional[float] = None
    detail: Optional[str] = None
    errors: List[str] = field(default_factory=list)


@dataclass
class DashboardData:
    nav_snapshot: DataSet
    nav_series: DataSet
    positions: DataSet
    trades: DataSet
    risk_events: DataSet
    signals: DataSet
    screener_tail: DataSet
    veto_events: DataSet
    env_risk_knobs: DataSet


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def _load_json_file(path: Path) -> Tuple[Any, Optional[str]]:
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None, "missing"
    except Exception as exc:
        return None, f"read_error:{exc}"  # pragma: no cover - defensive

    text = text.strip()
    if not text:
        return None, "empty"
    try:
        return json.loads(text), None
    except Exception as exc:
        return None, f"json_error:{exc}"


def _read_json_lines(path: Path, limit: Optional[int] = None) -> Tuple[List[Any], Optional[str]]:
    try:
        raw = path.read_text().splitlines()
    except FileNotFoundError:
        return [], "missing"
    except Exception as exc:
        return [], f"read_error:{exc}"  # pragma: no cover

    if limit is not None and limit > 0:
        raw = raw[-limit:]

    rows: List[Any] = []
    for ln in raw:
        ln = ln.strip()
        if not ln:
            continue
        try:
            rows.append(json.loads(ln))
        except Exception:
            continue
    return rows, None


def _to_epoch_seconds(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        val = float(value)
        if val > 1e12:  # ms
            return val / 1000.0
        return val
    if isinstance(value, str) and value:
        txt = value.strip()
        if not txt:
            return None
        if txt.isdigit():
            try:
                return _to_epoch_seconds(float(txt))
            except Exception:
                return None
        from datetime import datetime, timezone

        for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                dt = datetime.strptime(txt, fmt)
                if dt.tzinfo is None:
                    return dt.replace(tzinfo=timezone.utc).timestamp()
                return dt.timestamp()
            except Exception:
                continue
        try:
            dt = datetime.fromisoformat(txt.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception:
            return None
    # Firestore Timestamp like objects expose .timestamp()
    ts_attr = getattr(value, "timestamp", None)
    if callable(ts_attr):
        try:
            return float(ts_attr())
        except Exception:
            return None
    return None


def _first_not_none(values: Iterable[Any]) -> Any:
    for v in values:
        if v is not None:
            return v
    return None


def _firestore_enabled() -> bool:
    return os.getenv("FIRESTORE_ENABLED", "0") == "1"


@st.cache_resource(show_spinner=False)
def get_firestore_client():  # pragma: no cover - only executed with Firestore enabled
    if not _firestore_enabled():
        return None
    try:
        from google.cloud import firestore  # type: ignore
    except Exception:
        return None

    creds_path = os.getenv("FIREBASE_CREDS_PATH") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path:
        creds = Path(creds_path)
        if creds.exists():
            os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", str(creds))
        else:
            return None
    try:
        return firestore.Client()
    except Exception:
        return None


def _firestore_paths(db):
    root = db.collection("hedge").document(ENV)
    state_col = root.collection("state")
    return {
        "root": root,
        "state": state_col,
        "nav_doc": state_col.document("nav"),
        "positions_doc": state_col.document("positions"),
        "signals_col": root.collection("signals"),
        "screener_tail_col": root.collection("screener_tail"),
        "trades_col": root.collection("trades"),
        "risk_col": root.collection("risk"),
        "alt_positions_col": root.collection("positions"),
    }


def _env_filter(doc: Dict[str, Any]) -> bool:
    env_val = doc.get("env")
    if env_val is not None and str(env_val) != ENV:
        return False
    tn_val = doc.get("testnet")
    if tn_val is not None and bool(tn_val) != TESTNET:
        return False
    return True


def _normalize_nav_series(payload: Any) -> List[Dict[str, Any]]:
    series: List[Dict[str, Any]] = []
    if isinstance(payload, dict):
        for key in ("series", "points", "rows", "nav"):
            if isinstance(payload.get(key), list):
                payload = payload.get(key)
                break
        else:
            payload = list(payload.values())
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            ts = _first_not_none((item.get("ts"), item.get("t"), item.get("time")))
            nav_val = _first_not_none((item.get("nav"), item.get("equity"), item.get("value")))
            ts_epoch = _to_epoch_seconds(ts)
            try:
                nav_float = None if nav_val is None else float(nav_val)
            except Exception:
                nav_float = None
            if ts_epoch is None or nav_float is None:
                continue
            series.append({"ts": ts_epoch, "nav": nav_float})
    series.sort(key=lambda x: x["ts"])
    return series


def _normalize_position_rows(raw: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    candidates: Iterable[Any]
    if isinstance(raw, dict):
        if isinstance(raw.get("rows"), list):
            candidates = raw.get("rows") or []
        elif isinstance(raw.get("items"), list):
            candidates = raw.get("items") or []
        else:
            candidates = raw.values()
    elif isinstance(raw, list):
        candidates = raw
    else:
        candidates = []

    for rec in candidates:
        if not isinstance(rec, dict):
            continue
        symbol = str(rec.get("symbol") or rec.get("pair") or "").upper()
        if not symbol:
            continue
        qty_raw = _first_not_none((rec.get("qty"), rec.get("size"), rec.get("positionAmt"), rec.get("amount")))
        try:
            qty = float(qty_raw or 0.0)
        except Exception:
            qty = 0.0
        if math.isclose(qty, 0.0, abs_tol=1e-9):
            continue
        entry_raw = _first_not_none((rec.get("entryPrice"), rec.get("avgEntryPrice"), rec.get("entry_price")))
        mark_raw = _first_not_none((rec.get("markPrice"), rec.get("mark_price"), rec.get("lastPrice")))
        pnl_raw = _first_not_none((rec.get("unrealizedPnl"), rec.get("pnl"), rec.get("unrealized")))
        lev_raw = _first_not_none((rec.get("leverage"), rec.get("lev")))
        ts_raw = _first_not_none((rec.get("updatedAt"), rec.get("ts"), rec.get("time")))

        try:
            entry = float(entry_raw) if entry_raw is not None else None
        except Exception:
            entry = None
        try:
            mark = float(mark_raw) if mark_raw is not None else None
        except Exception:
            mark = None
        try:
            pnl = float(pnl_raw) if pnl_raw is not None else None
        except Exception:
            pnl = None
        try:
            lev = float(lev_raw) if lev_raw is not None else None
        except Exception:
            lev = None

        price_for_notional = _first_not_none((mark, entry)) or 0.0
        notional = abs(qty) * float(price_for_notional)
        side = rec.get("side") or rec.get("positionSide")
        if not side:
            side = "LONG" if qty > 0 else "SHORT"

        rows.append(
            {
                "symbol": symbol,
                "side": side,
                "qty": abs(qty),
                "entry_price": entry,
                "mark_price": mark,
                "pnl": pnl,
                "leverage": lev,
                "ts": _to_epoch_seconds(ts_raw),
                "notional": notional,
            }
        )
    rows.sort(key=lambda x: x.get("notional", 0.0), reverse=True)
    return rows


def _normalize_trade_rows(raw: Iterable[Any], window_cutoff: Optional[float] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    now = time.time()
    for rec in raw:
        if not isinstance(rec, dict):
            continue
        ts = _to_epoch_seconds(_first_not_none((rec.get("ts"), rec.get("time"), rec.get("timestamp"))))
        if ts is None:
            continue
        if window_cutoff is not None and ts < window_cutoff:
            continue
        if ts > now + 3600:  # guard against malformed future timestamps
            continue
        symbol = rec.get("symbol")
        if not symbol:
            continue
        qty_raw = _first_not_none((rec.get("qty"), rec.get("quantity"), rec.get("size")))
        price_raw = _first_not_none((rec.get("price"), rec.get("fillPrice")))
        pnl_raw = rec.get("pnl") or rec.get("realizedPnl")
        notional_raw = rec.get("notional")
        try:
            qty = float(qty_raw) if qty_raw is not None else None
        except Exception:
            qty = None
        try:
            price = float(price_raw) if price_raw is not None else None
        except Exception:
            price = None
        if notional_raw is None and qty is not None and price is not None:
            notional_raw = abs(qty * price)
        try:
            notional = float(notional_raw) if notional_raw is not None else None
        except Exception:
            notional = None
        try:
            pnl = float(pnl_raw) if pnl_raw is not None else None
        except Exception:
            pnl = None

        out.append(
            {
                "symbol": str(symbol).upper(),
                "side": rec.get("side"),
                "qty": qty,
                "price": price,
                "pnl": pnl,
                "notional": notional,
                "ts": ts,
                "slippage_bps": rec.get("slippage_bps") or rec.get("slippageBps"),
            }
        )
    out.sort(key=lambda x: x["ts"], reverse=True)
    return out


def _normalize_risk_rows(raw: Iterable[Any], window_cutoff: Optional[float] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rec in raw:
        if not isinstance(rec, dict):
            continue
        ts = _to_epoch_seconds(_first_not_none((rec.get("ts"), rec.get("t"), rec.get("time"))))
        if ts is None:
            continue
        if window_cutoff is not None and ts < window_cutoff:
            continue
        reason = rec.get("reason") or rec.get("veto")
        symbol = rec.get("symbol")
        rows.append(
            {
                "symbol": symbol,
                "reason": reason,
                "side": rec.get("side"),
                "ts": ts,
                "phase": rec.get("phase"),
                "notional": rec.get("notional") or rec.get("gross"),
                "gross": rec.get("gross"),
                "env": rec.get("env"),
            }
        )
    rows.sort(key=lambda x: x["ts"], reverse=True)
    return rows


def _normalize_signal_rows(raw: Iterable[Any], window_cutoff: Optional[float] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rec in raw:
        if not isinstance(rec, dict):
            continue
        ts = _to_epoch_seconds(_first_not_none((rec.get("ts"), rec.get("t"), rec.get("timestamp"), rec.get("time"))))
        if ts is None:
            continue
        if window_cutoff is not None and ts < window_cutoff:
            continue
        rows.append(
            {
                "symbol": rec.get("symbol"),
                "signal": rec.get("signal"),
                "timeframe": rec.get("timeframe") or rec.get("tf"),
                "side": rec.get("side"),
                "price": rec.get("price"),
                "cap": rec.get("capital_per_trade") or rec.get("cap"),
                "leverage": rec.get("leverage") or rec.get("lev"),
                "ts": ts,
            }
        )
    rows.sort(key=lambda x: x["ts"], reverse=True)
    return rows


def _local_nav_history() -> DataSet:
    path = DEFAULT_NAV_HISTORY
    rows, err = _read_json_lines(path, limit=5000)
    series = _normalize_nav_series(rows)
    updated = series[-1]["ts"] if series else None
    detail = str(path.relative_to(PROJECT_ROOT)) if path.exists() else None
    errors = [err] if err else []
    return DataSet(data=series, source="local", updated_at=updated, detail=detail, errors=errors)


def _local_positions() -> DataSet:
    data, err = _load_json_file(DEFAULT_POSITIONS_LOG)
    rows = _normalize_position_rows(data)
    updated = rows[0]["ts"] if rows else None
    detail = str(DEFAULT_POSITIONS_LOG.relative_to(PROJECT_ROOT)) if DEFAULT_POSITIONS_LOG.exists() else None
    errors = [err] if err else []
    return DataSet(data=rows, source="local", updated_at=updated, detail=detail, errors=errors)


def _local_risk_events() -> DataSet:
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    veto_glob = PROJECT_ROOT / "logs" / DEFAULT_VETO_PATTERN
    for path in sorted(veto_glob.parent.glob(veto_glob.name)):
        data, err = _load_json_file(path)
        if err:
            errors.append(f"{path.name}:{err}")
            continue
        if isinstance(data, dict):
            rows.extend(_normalize_risk_rows([data]))
        elif isinstance(data, list):
            rows.extend(_normalize_risk_rows(data))
    rows.sort(key=lambda x: x["ts"], reverse=True)
    return DataSet(data=rows, source="local", updated_at=rows[0]["ts"] if rows else None, detail="logs/veto_exec_*.json", errors=errors)


def _local_screener_tail(limit: int = 200) -> DataSet:
    try:
        lines = DEFAULT_SCREENER_LOG.read_text().splitlines()
    except Exception:
        lines = []
    tail = lines[-limit:]
    updated: Optional[float] = None
    for ln in reversed(tail):
        if "{" not in ln or "}" not in ln:
            continue
        try:
            payload = ln.split("{", 1)[1]
            payload = payload.split("}", 1)[0]
            obj = json.loads("{" + payload + "}")
        except Exception:
            continue
        updated = _to_epoch_seconds(_first_not_none((obj.get("ts"), obj.get("t"), obj.get("timestamp"))))
        if updated:
            break
    detail = str(DEFAULT_SCREENER_LOG.relative_to(PROJECT_ROOT)) if DEFAULT_SCREENER_LOG.exists() else None
    return DataSet(data=tail, source="local", updated_at=updated, detail=detail)


def _local_nav_snapshot() -> DataSet:
    data, err = _load_json_file(DEFAULT_NAV_SNAPSHOT)
    if not isinstance(data, dict):
        data = {}
    ts = _to_epoch_seconds(data.get("ts")) if isinstance(data, dict) else None
    detail = str(DEFAULT_NAV_SNAPSHOT.relative_to(PROJECT_ROOT)) if DEFAULT_NAV_SNAPSHOT.exists() else None
    errors = [err] if err else []
    return DataSet(data=data, source="local", updated_at=ts, detail=detail, errors=errors)


@st.cache_data(ttl=15, show_spinner=False)
def read_nav_snapshot(path: str = "state/nav_snapshot.json") -> Dict[str, Any]:
    data, _ = _load_json_file(_resolve_path(path))
    return data if isinstance(data, dict) else {}


@st.cache_data(ttl=15, show_spinner=False)
def load_nav_series() -> DataSet:
    db = get_firestore_client()
    if db is not None:
        try:
            paths = _firestore_paths(db)
            snap = paths["nav_doc"].get()
            if getattr(snap, "exists", False):
                payload = snap.to_dict() or {}
                series = _normalize_nav_series(payload)
                updated = _to_epoch_seconds(payload.get("updated_at")) or (snap.update_time.timestamp() if getattr(snap, "update_time", None) else None)
                if not updated and series:
                    updated = series[-1]["ts"]
                if series:
                    return DataSet(data=series, source="firestore", updated_at=updated, detail=f"hedge/{ENV}/state/nav")
        except Exception as exc:  # pragma: no cover - defensive
            return DataSet(data=[], source="firestore", updated_at=None, detail=f"hedge/{ENV}/state/nav", errors=[str(exc)])

    return _local_nav_history()


@st.cache_data(ttl=15, show_spinner=False)
def load_positions() -> DataSet:
    db = get_firestore_client()
    if db is not None:
        try:
            paths = _firestore_paths(db)
            snap = paths["positions_doc"].get()
            payload = snap.to_dict() if getattr(snap, "exists", False) else None
            rows = _normalize_position_rows(payload)
            updated = None
            if isinstance(payload, dict):
                updated = _to_epoch_seconds(payload.get("updated_at"))
            if not updated and rows:
                updated = rows[0].get("ts")
            if rows:
                return DataSet(data=rows, source="firestore", updated_at=updated, detail=f"hedge/{ENV}/state/positions")
            # fallback to alt collection if document empty
            alt_rows: List[Dict[str, Any]] = []
            try:
                col = paths["alt_positions_col"]
                docs = list(col.order_by("ts", direction="DESCENDING").limit(200).stream())
                alt_rows = _normalize_position_rows([d.to_dict() for d in docs if _env_filter(d.to_dict() or {})])
            except Exception:
                alt_rows = []
            if alt_rows:
                updated = alt_rows[0].get("ts") if alt_rows else None
                return DataSet(data=alt_rows, source="firestore", updated_at=updated, detail=f"hedge/{ENV}/positions")
        except Exception as exc:  # pragma: no cover - defensive
            return DataSet(data=[], source="firestore", updated_at=None, detail=f"hedge/{ENV}/state/positions", errors=[str(exc)])

    return _local_positions()


@st.cache_data(ttl=15, show_spinner=False)
def load_trades(hours: int = 168) -> DataSet:
    cutoff = time.time() - hours * 3600
    db = get_firestore_client()
    if db is not None:
        try:
            paths = _firestore_paths(db)
            docs = paths["trades_col"].order_by("ts", direction="DESCENDING").limit(2000).stream()
            payload = []
            for doc in docs:
                data = doc.to_dict() or {}
                if not _env_filter(data):
                    continue
                payload.append(data)
            rows = _normalize_trade_rows(payload, window_cutoff=cutoff)
            updated = rows[0]["ts"] if rows else None
            return DataSet(data=rows, source="firestore", updated_at=updated, detail=f"hedge/{ENV}/trades")
        except Exception as exc:  # pragma: no cover - defensive
            return DataSet(data=[], source="firestore", updated_at=None, detail=f"hedge/{ENV}/trades", errors=[str(exc)])

    return DataSet(data=[], source="local", updated_at=None, detail=None)


@st.cache_data(ttl=15, show_spinner=False)
def load_risk_events(hours: int = 168) -> DataSet:
    cutoff = time.time() - hours * 3600
    db = get_firestore_client()
    if db is not None:
        try:
            paths = _firestore_paths(db)
            docs = paths["risk_col"].order_by("ts", direction="DESCENDING").limit(2000).stream()
            payload = []
            for doc in docs:
                data = doc.to_dict() or {}
                if not _env_filter(data):
                    continue
                payload.append(data)
            rows = _normalize_risk_rows(payload, window_cutoff=cutoff)
            updated = rows[0]["ts"] if rows else None
            return DataSet(data=rows, source="firestore", updated_at=updated, detail=f"hedge/{ENV}/risk")
        except Exception as exc:  # pragma: no cover - defensive
            return DataSet(data=[], source="firestore", updated_at=None, detail=f"hedge/{ENV}/risk", errors=[str(exc)])

    return _local_risk_events()


@st.cache_data(ttl=15, show_spinner=False)
def load_signals(hours: int = 168) -> DataSet:
    cutoff = time.time() - hours * 3600
    db = get_firestore_client()
    if db is not None:
        try:
            paths = _firestore_paths(db)
            docs = paths["signals_col"].order_by("ts", direction="DESCENDING").limit(2000).stream()
            payload = []
            for doc in docs:
                data = doc.to_dict() or {}
                if not _env_filter(data):
                    continue
                payload.append(data)
            rows = _normalize_signal_rows(payload, window_cutoff=cutoff)
            updated = rows[0]["ts"] if rows else None
            return DataSet(data=rows, source="firestore", updated_at=updated, detail=f"hedge/{ENV}/signals")
        except Exception as exc:  # pragma: no cover - defensive
            return DataSet(data=[], source="firestore", updated_at=None, detail=f"hedge/{ENV}/signals", errors=[str(exc)])

    # Local fallback from screener tail log
    tail = _local_screener_tail().data
    rows: List[Dict[str, Any]] = []
    for ln in tail:
        if "{" not in ln or "}" not in ln:
            continue
        try:
            payload = ln.split("{", 1)[1]
            payload = "{" + payload.split("}", 1)[0] + "}"
            obj = json.loads(payload)
        except Exception:
            continue
        rows.extend(_normalize_signal_rows([obj], window_cutoff=cutoff))
    rows.sort(key=lambda x: x["ts"], reverse=True)
    updated = rows[0]["ts"] if rows else None
    return DataSet(data=rows, source="local", updated_at=updated, detail="logs/screener_tail.log")


@st.cache_data(ttl=15, show_spinner=False)
def load_screener_tail() -> DataSet:
    db = get_firestore_client()
    if db is not None:
        try:
            paths = _firestore_paths(db)
            docs = paths["screener_tail_col"].order_by("ts", direction="DESCENDING").limit(200).stream()
            payload = []
            for doc in docs:
                data = doc.to_dict() or {}
                if not _env_filter(data):
                    continue
                payload.append(data)
            rows = _normalize_signal_rows(payload, window_cutoff=None)
            updated = rows[0]["ts"] if rows else None
            lines = [json.dumps(row) for row in rows]
            return DataSet(data=lines, source="firestore", updated_at=updated, detail=f"hedge/{ENV}/screener_tail")
        except Exception:
            pass
    return _local_screener_tail()


def _env_risk_knobs() -> DataSet:
    keys = [
        "CAP_OVERRIDE_PCT",
        "HEADROOM_FRACTION",
        "MIN_ENTRY_NOTIONAL_USD",
        "COOLDOWN_SECS",
        "RISK_FAIL_CLOSED",
    ]
    data = {k: os.getenv(k) for k in keys if os.getenv(k) is not None}
    return DataSet(data=data, source="env", updated_at=None, detail="env")


@st.cache_data(ttl=15, show_spinner=False)
def load_dashboard_data() -> DashboardData:
    nav_snapshot = _local_nav_snapshot()
    nav_series = load_nav_series()
    positions = load_positions()
    trades = load_trades()
    risk_events = load_risk_events()
    signals = load_signals()
    screener_tail = load_screener_tail()
    veto_events = _local_risk_events() if risk_events.source != "firestore" else risk_events
    env_knobs = _env_risk_knobs()
    return DashboardData(
        nav_snapshot=nav_snapshot,
        nav_series=nav_series,
        positions=positions,
        trades=trades,
        risk_events=risk_events,
        signals=signals,
        screener_tail=screener_tail,
        veto_events=veto_events,
        env_risk_knobs=env_knobs,
    )


def _read_local_jsonl_tail(path: str, tail: int = 1) -> List[Dict[str, Any]]:
    path_obj = _resolve_path(path)
    if not path_obj.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with path_obj.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:  # pragma: no cover - defensive I/O handling
        return []
    return rows[-tail:]


def _firestore_latest() -> Dict[str, Any]:
    if not (FIRESTORE_ENABLED and _get_db):
        return {}
    try:
        db = _get_db()
        doc = db.collection(FIRESTORE_COLLECTION).document("latest").get()
        return doc.to_dict() or {}
    except Exception:  # pragma: no cover - defensive against Firestore outage
        return {}


def _mark_staleness(snap: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    ts = snap.get("ts") or snap.get("timestamp") or 0
    try:
        ts_float = float(ts)
    except Exception:
        ts_float = 0.0
    if ts_float > 10_000_000_000:
        ts_float /= 1000.0
    age = time.time() - ts_float if ts_float else float("inf")
    snap["age_sec"] = max(0.0, age if math.isfinite(age) else float("inf"))
    is_stale = age > STALE_AFTER_SEC if math.isfinite(age) else True
    snap["is_stale"] = is_stale
    return snap, is_stale


@lru_cache(maxsize=1)
def _cached_latest_nav(_nonce: int) -> Dict[str, Any]:
    firestore_doc = _firestore_latest()
    if firestore_doc:
        snap, _ = _mark_staleness(dict(firestore_doc))
        snap["source"] = "firestore"
        return snap

    local_tail = _read_local_jsonl_tail(LOCAL_NAV, tail=1)
    if local_tail:
        snap, _ = _mark_staleness(dict(local_tail[0]))
        snap["source"] = "local"
        return snap

    return {
        "equity": 0.0,
        "balance": 0.0,
        "positions": [],
        "source": "empty",
        "ts": 0,
        "is_stale": True,
        "age_sec": float("inf"),
    }


def get_latest_nav() -> Dict[str, Any]:
    nonce = int(time.time() // 30)
    return _cached_latest_nav(nonce)


def _safe_rr(entry: Any, mark: Any) -> float:
    try:
        entry_f = float(entry)
        mark_f = float(mark)
        if entry_f == 0:
            return 0.0
        return (mark_f - entry_f) / entry_f
    except Exception:
        return 0.0


def per_symbol_kpis(positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    snapshot = []
    now = time.time()
    for pos in positions or []:
        try:
            last_update = float(pos.get("last_update", 0) or 0)
        except Exception:
            last_update = 0.0
        age = now - last_update if last_update else float("inf")
        snapshot.append(
            {
                "symbol": pos.get("symbol"),
                "side": pos.get("side"),
                "size": pos.get("size", pos.get("positionAmt", 0.0)),
                "entry": pos.get("entry", pos.get("entryPrice")),
                "mark": pos.get("mark", pos.get("markPrice")),
                "unrealized": pos.get("unrealized", pos.get("unRealizedProfit", 0.0)),
                "realized": pos.get("realized", pos.get("realizedPnl", 0.0)),
                "rr": _safe_rr(pos.get("entry", pos.get("entryPrice")), pos.get("mark", pos.get("markPrice"))),
                "veto_tail": pos.get("veto_tail", []),
                "stale": age > STALE_AFTER_SEC if math.isfinite(age) else True,
                "age_sec": age if math.isfinite(age) else float("inf"),
            }
        )
    return snapshot
