#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json

# Publishes read-only state to Firestore (positions + NAV).
# - Loads .env from repo root (override=True) so ad-hoc runs see keys.
# - Filters positions to non-zero qty and to symbols in pairs_universe.json (if present).
# - Debounces writes (executor can call this every loop safely).
import os
import pathlib
import time
from typing import Any, Dict, List, Optional

# ----- robust .env load -----
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env", override=True)
except Exception:
    pass

ENV = os.getenv("ENV", "prod")
FS_ROOT = f"hedge/{ENV}/state"
CREDS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or str(
    ROOT_DIR / "config/firebase_creds.json"
)


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


# ----- Firestore client -----
def _fs_client():
    from google.cloud import firestore

    if os.path.exists(CREDS_PATH):
        from google.oauth2 import service_account

        info = json.load(open(CREDS_PATH))
        creds = service_account.Credentials.from_service_account_file(CREDS_PATH)
        return firestore.Client(project=info.get("project_id"), credentials=creds)
    return firestore.Client()


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
    cli = _fs_client()
    cli.document(f"{FS_ROOT}/positions").set(
        {
            "rows": rows,
            "updated": time.time(),
        },
        merge=True,
    )


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
    cli = _fs_client()
    doc = cli.document(f"{FS_ROOT}/nav")
    snap = doc.get()
    data = snap.to_dict() if getattr(snap, "exists", False) else {}
    series = data.get("series") or []
    now = time.time()
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
    doc.set({"series": series, "updated": now}, merge=True)


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


def _audit_append(doc_name: str, event: dict, max_len: int = 500) -> None:
    cli = _fs_client_safe()
    if not cli:
        return
    now = time.time()
    doc = cli.document(f"{FS_ROOT}/{doc_name}")
    try:
        snap = doc.get()
        data = snap.to_dict() if getattr(snap, "exists", False) else {}
    except Exception:
        data = {}
    hist = list(data.get("history", []))
    ev = dict(event)
    ev.setdefault("t", now)
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
    cli = _fs_client()
    now = time.time()
    body = dict(plan)
    body["updated"] = now
    cli.document(f"{FS_ROOT}/exits_{symbol}_{position_side}").set(body, merge=True)


def get_exit_plan(symbol: str, position_side: str) -> dict | None:
    cli = _fs_client()
    snap = cli.document(f"{FS_ROOT}/exits_{symbol}_{position_side}").get()
    return snap.to_dict() if getattr(snap, "exists", False) else None


def publish_exit_event(symbol: str, position_side: str, event: dict) -> None:
    cli = _fs_client()
    now = time.time()
    doc = cli.document(f"{FS_ROOT}/exit_event_{symbol}_{position_side}")
    snap = doc.get()
    data = snap.to_dict() if getattr(snap, "exists", False) else {}
    hist = list(data.get("history", []))
    event = dict(event)
    event.setdefault("t", now)
    hist.append(event)
    doc.set({"last": event, "history": hist[-200:], "updated": now}, merge=True)
