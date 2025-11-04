#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone

# Publishes read-only state to Firestore (positions + NAV).
# - Loads .env from repo root (override=True) so ad-hoc runs see keys.
# - Filters positions to non-zero qty and to symbols in pairs_universe.json (if present).
# - Debounces writes (executor can call this every loop safely).
import os
import pathlib
import time
from typing import Any, Dict, List, Optional

from utils.firestore_client import get_db

# ----- robust .env load -----
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env", override=True)
except Exception:
    pass

def _resolve_env(default: str = "dev") -> str:
    value = (os.getenv("ENV") or os.getenv("ENVIRONMENT") or "").strip()
    return value or default


ENV = _resolve_env()
if ENV.lower() == "prod":
    allow_prod = os.getenv("ALLOW_PROD_WRITE", "0").strip().lower()
    if allow_prod not in {"1", "true", "yes"}:
        raise RuntimeError(
            "Refusing to publish state with ENV=prod. Set ALLOW_PROD_WRITE=1 to override explicitly."
        )
FS_ROOT = f"hedge/{ENV}/state"
LOG_DIR = ROOT_DIR / "logs"
EXEC_LOG_DIR = LOG_DIR / "execution"
_EXEC_STATS_CACHE: Dict[str, Any] = {"ts": 0.0, "data": None}


def _ensure_keys() -> None:
    """Backstop parse of .env in case python-dotenv wasn't available."""
    if os.getenv("BINANCE_API_KEY") and os.getenv("BINANCE_API_SECRET"):
        return
    env_path = ROOT_DIR / ".env"
    if not env_path.exists():
        return
    for ln in env_path.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#") or "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_ensure_keys()


def _normalize_status(value: Any) -> str:
    if not value:
        return "UNKNOWN"
    try:
        text = str(value).upper()
    except Exception:
        return "UNKNOWN"
    if text == "CANCELLED":
        return "CANCELED"
    return text


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_timestamp(record: Dict[str, Any]) -> Optional[float]:
    candidates = (
        record.get("ts"),
        record.get("timestamp"),
        record.get("time"),
        record.get("t"),
        record.get("local_ts"),
    )
    for value in candidates:
        if value is None:
            continue
        if isinstance(value, (int, float)):
            try:
                return float(value)
            except Exception:
                continue
        if isinstance(value, str):
            val = value.strip()
            if not val:
                continue
            try:
                return float(val)
            except ValueError:
                pass
            try:
                iso_val = val.replace("Z", "+00:00") if val.endswith("Z") else val
                dt = datetime.fromisoformat(iso_val)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.timestamp()
            except Exception:
                continue
    return None


def _to_iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except Exception:
        return None


def _iter_recent_records(path: pathlib.Path, cutoff: float):
    try:
        with path.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        return
    except Exception:
        return
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        if not isinstance(record, dict):
            continue
        ts = _extract_timestamp(record)
        if ts is None:
            continue
        if ts < cutoff:
            break
        yield record


def _last_heartbeats() -> Dict[str, Optional[str]]:
    path = EXEC_LOG_DIR / "sync_heartbeats.jsonl"
    latest: Dict[str, float] = {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                if not isinstance(record, dict):
                    continue
                service = record.get("service")
                if not service:
                    continue
                ts = _extract_timestamp(record)
                if ts is None:
                    continue
                if ts >= latest.get(service, float("-inf")):
                    latest[service] = ts
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return {svc: _to_iso(ts) for svc, ts in latest.items()}


def _compute_exec_stats() -> Dict[str, Any]:
    now_ts = time.time()
    if (
        isinstance(_EXEC_STATS_CACHE.get("ts"), (int, float))
        and _EXEC_STATS_CACHE.get("data") is not None
        and (now_ts - float(_EXEC_STATS_CACHE["ts"])) < 30.0
    ):
        return _EXEC_STATS_CACHE["data"]

    try:
        cutoff = now_ts - 86400.0
        attempted = 0
        ack_ids: set[str] = set()
        fill_ids: set[str] = set()
        successful_ids: set[str] = set()
        veto_counter: Counter[str] = Counter()

        attempt_path = EXEC_LOG_DIR / "orders_attempted.jsonl"
        for _ in _iter_recent_records(attempt_path, cutoff) or []:
            attempted += 1

        executed_path = EXEC_LOG_DIR / "orders_executed.jsonl"
        legacy_counter = 0
        for record in _iter_recent_records(executed_path, cutoff) or []:
            if not isinstance(record, dict):
                continue
            event_type = str(record.get("event_type") or record.get("event") or "").lower()
            order_id = record.get("orderId") or record.get("order_id")
            client_id = record.get("clientOrderId") or record.get("client_order_id")
            identifier: str
            if order_id or client_id:
                identifier = str(order_id or client_id)
            else:
                legacy_counter += 1
                identifier = f"legacy_{legacy_counter}"
            if event_type == "order_ack":
                ack_ids.add(identifier)
                continue
            if event_type == "order_fill":
                fill_ids.add(identifier)
                status = _normalize_status(record.get("status"))
                executed_qty = (
                    _safe_float(record.get("executedQty"))
                    or _safe_float(record.get("qty"))
                )
                if executed_qty and executed_qty > 0 and status in {"FILLED", "PARTIALLY_FILLED"}:
                    successful_ids.add(identifier)
                continue
            if event_type == "order_close":
                continue
            # Legacy fallback: treat as executed fill (and ack) entry
            ack_ids.add(identifier)
            fill_ids.add(identifier)
            status = _normalize_status(record.get("status"))
            executed_qty = (
                _safe_float(record.get("executedQty"))
                or _safe_float(record.get("qty"))
            )
            if executed_qty and executed_qty > 0 and status in {"FILLED", "SUCCESS"}:
                successful_ids.add(identifier)

        veto_path = EXEC_LOG_DIR / "risk_vetoes.jsonl"
        for record in _iter_recent_records(veto_path, cutoff) or []:
            reason = record.get("veto_reason") or record.get("reason") or "unknown"
            veto_counter[str(reason)] += 1

        executed = len(fill_ids)
        successful = len(successful_ids)
        ack_count = len(ack_ids) if ack_ids else executed
        fill_rate = 0.0
        denominator = attempted if attempted > 0 else (ack_count if ack_count > 0 else executed)
        numerator = successful if attempted > 0 else successful
        if denominator > 0:
            fill_rate = float(numerator) / float(denominator)

        last_hb = _last_heartbeats()
        stats = {
            "attempted_24h": attempted,
            "executed_24h": executed,
            "vetoes_24h": sum(veto_counter.values()),
            "fill_rate": fill_rate,
            "top_vetoes": [
                {"reason": reason, "count": count}
                for reason, count in veto_counter.most_common(5)
            ],
            "last_heartbeats": {
                "executor_live": last_hb.get("executor_live"),
                "sync_daemon": last_hb.get("sync_daemon"),
            },
        }
    except Exception:
        stats = {
            "attempted_24h": 0,
            "executed_24h": 0,
            "vetoes_24h": 0,
            "fill_rate": 0.0,
            "top_vetoes": [],
            "last_heartbeats": {
                "executor_live": None,
                "sync_daemon": None,
            },
        }

    _EXEC_STATS_CACHE["ts"] = now_ts
    _EXEC_STATS_CACHE["data"] = stats
    return stats


def _firestore_enabled() -> bool:
    return os.environ.get("FIRESTORE_ENABLED", "1") != "0"


# ----- Firestore client -----
def _fs_client():
    if not _firestore_enabled():
        raise RuntimeError("firestore_disabled")
    return get_db()


# ----- Exchange helpers (import late so .env is loaded) -----
def get_positions(symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    from execution.exchange_utils import _req

    params = {}
    if symbol:
        params["symbol"] = symbol
    return _req("GET", "/fapi/v2/positionRisk", signed=True, params=params).json()


def get_account() -> Dict[str, Any]:
    from execution.exchange_utils import _req

    return _req("GET", "/fapi/v2/account", signed=True).json()


# ----- Normalizers / publishers -----
def normalize_positions(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Optional universe whitelist
    universe = None
    try:
        u = json.load(open(str(ROOT_DIR / "config/pairs_universe.json")))
        universe = set(u.get("symbols", []))
    except Exception:
        pass

    rows: List[Dict[str, Any]] = []
    for p in raw or []:
        try:
            sym = p.get("symbol")
            if universe and sym not in universe:
                continue
            qty = float(p.get("qty", p.get("positionAmt", 0)) or 0.0)
            if qty == 0.0:
                continue
            rows.append(
                {
                    "symbol": sym,
                    "positionSide": p.get("positionSide") or "BOTH",
                    "qty": qty,
                    "entryPrice": float(p.get("entryPrice") or 0),
                    "leverage": float(p.get("leverage") or 0),
                    "uPnl": float(
                        p.get("unRealizedProfit", p.get("unrealized", 0)) or 0
                    ),
                }
            )
        except Exception:
            pass
    return rows


def publish_positions(rows: List[Dict[str, Any]]) -> None:
    stats = _compute_exec_stats()
    payload = {"rows": rows, "updated": time.time(), "exec_stats": stats}
    if not _firestore_enabled():
        _append_local_jsonl("positions", payload)
        return
    cli = _fs_client()
    cli.document(f"{FS_ROOT}/positions").set(payload, merge=True)


def compute_nav() -> float:
    acc = get_account()
    # Prefer totalMarginBalance (wallet + uPnL)
    tmb = acc.get("totalMarginBalance")
    if tmb is not None:
        return float(tmb)
    twb = float(acc.get("totalWalletBalance", 0) or 0)
    tup = float(acc.get("totalUnrealizedProfit", 0) or 0)
    return twb + tup


def publish_nav_value(
    nav: float, min_interval_s: int = 60, max_points: int = 20000
) -> None:
    now = time.time()
    stats = _compute_exec_stats()
    if not _firestore_enabled():
        _append_local_jsonl("nav", {"t": now, "nav": float(nav), "exec_stats": stats})
        return
    cli = _fs_client()
    doc = cli.document(f"{FS_ROOT}/nav")
    snap = doc.get()
    data = snap.to_dict() if getattr(snap, "exists", False) else {}
    series = data.get("series") or []
    # Debounce
    if series:
        last = series[-1]
        try:
            last_t = float(last.get("t") or last.get("ts") or last.get("time") or 0)
        except Exception:
            last_t = 0.0
        if now - last_t < min_interval_s:
            return
    series.append({"t": now, "nav": float(nav)})
    if len(series) > max_points:
        series = series[-max_points:]
    doc.set({"series": series, "updated": now, "exec_stats": stats}, merge=True)


class StatePublisher:
    """Hash/debounce publisher for positions from the executor loop."""

    def __init__(self, interval_s: int = 60):
        self.interval_s = interval_s
        self._h: Optional[str] = None
        self._t: float = 0.0

    def maybe_publish_positions(self, rows: List[Dict[str, Any]]) -> None:
        body = json.dumps(rows, sort_keys=True, default=str).encode()
        h = hashlib.sha256(body).hexdigest()
        now = time.time()
        if h != self._h or (now - self._t) >= self.interval_s:
            publish_positions(rows)
            self._h = h
            self._t = now


# ----- Audit/event helpers -----
def _fs_client_safe():
    try:
        return _fs_client()
    except Exception:
        return None


def _append_local_jsonl(name: str, event: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"{name}.jsonl"
    with path.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(event) + '\n')


def _audit_append(doc_name: str, event: dict, max_len: int = 500) -> None:
    now = time.time()
    ev = dict(event)
    ev.setdefault("t", now)
    if not _firestore_enabled():
        _append_local_jsonl(doc_name, ev)
        return
    cli = _fs_client_safe()
    if not cli:
        _append_local_jsonl(doc_name, ev)
        return
    doc = cli.document(f"{FS_ROOT}/{doc_name}")
    try:
        snap = doc.get()
        data = snap.to_dict() if getattr(snap, "exists", False) else {}
    except Exception:
        data = {}
    hist = list(data.get("history", []))
    hist.append(ev)
    hist = hist[-max_len:]
    try:
        doc.set({"last": ev, "history": hist, "updated": now}, merge=True)
    except Exception:
        pass


def publish_intent_audit(intent: dict) -> None:
    ev = dict(intent)
    ev.setdefault("type", "intent")
    _audit_append("audit_intents", ev)


def publish_order_audit(symbol: str, event: dict) -> None:
    ev = dict(event)
    ev.setdefault("symbol", symbol)
    ev.setdefault("type", "order")
    _audit_append(f"audit_orders_{symbol}", ev)


def publish_close_audit(symbol: str, position_side: str, event: dict) -> None:
    ev = dict(event)
    ev.setdefault("symbol", symbol)
    ev.setdefault("positionSide", position_side)
    ev.setdefault("type", "close")
    _audit_append(f"audit_closes_{symbol}_{position_side}", ev)


if __name__ == "__main__":
    # Preflight: ensure keys & perms visible to THIS process
    from execution.exchange_utils import _req

    try:
        _ = _req("GET", "/fapi/v1/time")
        _ = _req("GET", "/fapi/v2/balance", signed=True)
    except Exception as e:
        print("preflight_error:", e)
        raise SystemExit(1)

    raw = get_positions()
    rows = normalize_positions(raw)
    publish_positions(rows)
    try:
        nav = compute_nav()
        publish_nav_value(nav)
        print(f"published positions: {len(rows)}, nav: {nav}")
    except Exception as e:
        print("nav_publish_warn:", e)


# ----- Exit plan publish/fetch -----
def publish_exit_plan(symbol: str, position_side: str, plan: dict) -> None:
    now = time.time()
    body = dict(plan)
    body["updated"] = now
    if not _firestore_enabled():
        _append_local_jsonl(f"exits_{symbol}_{position_side}", body)
        return
    cli = _fs_client()
    cli.document(f"{FS_ROOT}/exits_{symbol}_{position_side}").set(body, merge=True)


def get_exit_plan(symbol: str, position_side: str) -> dict | None:
    if not _firestore_enabled():
        return None
    cli = _fs_client()
    snap = cli.document(f"{FS_ROOT}/exits_{symbol}_{position_side}").get()
    return snap.to_dict() if getattr(snap, "exists", False) else None


def publish_exit_event(symbol: str, position_side: str, event: dict) -> None:
    now = time.time()
    ev = dict(event)
    ev.setdefault("t", now)
    if not _firestore_enabled():
        _append_local_jsonl(f"exit_event_{symbol}_{position_side}", ev)
        return
    cli = _fs_client()
    doc = cli.document(f"{FS_ROOT}/exit_event_{symbol}_{position_side}")
    snap = doc.get()
    data = snap.to_dict() if getattr(snap, "exists", False) else {}
    hist = list(data.get("history", []))
    hist.append(ev)
    doc.set({"last": ev, "history": hist[-200:], "updated": now}, merge=True)
