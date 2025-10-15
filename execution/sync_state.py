from __future__ import annotations

import json

# execution/sync_state.py — Phase‑4.1 “Stability & Signals” (hardened sync)
#
# What this does
#  - Reads local files (nav_log.json, peak_state.json, synced_state.json)
#  - Applies cutoff filtering to NAV history if configured
#  - Guards against zero-equity rows and empty tails
#  - Computes exposure KPIs from positions
#  - Derives peak from best available source (file, rows, existing Firestore doc)
#  - Upserts compact docs to Firestore:
#       hedge/{ENV}/state/nav
#       hedge/{ENV}/state/positions
#       hedge/{ENV}/state/leaderboard
#
# Env knobs
#  ENV=prod|dev
#  NAV_CUTOFF_ISO="2025-08-01T00:00:00+00:00"   # preferred explicit cutoff
#  NAV_CUTOFF_SECAGO=86400                       # or relative cutoff in seconds
#  SYNC_INTERVAL_SEC=20
#
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

# ---------------- Firestore import (supports multiple layouts) ---------------
try:
    # legacy: project root
    from firestore_client import with_firestore  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    try:
        # sibling package layout
        from utils.firestore_client import with_firestore  # type: ignore
    except ModuleNotFoundError:
        # monorepo package layout
        from hedge_fund.utils.firestore_client import with_firestore  # type: ignore

# ------------------------------- Files ---------------------------------------
NAV_LOG: str = "nav_log.json"
PEAK_STATE: str = "peak_state.json"
SYNCED_STATE: str = "synced_state.json"

# ------------------------------ Settings ------------------------------------
MAX_POINTS: int = 500  # dashboard series cap

_FIRESTORE_FAIL_COUNT = 0
_FAILURE_THRESHOLD = 5


# ------------------------------- Utilities ----------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _cutoff_dt() -> Optional[datetime]:
    """Cutoff datetime from env: NAV_CUTOFF_ISO (preferred) or NAV_CUTOFF_SECAGO."""
    iso = os.getenv("NAV_CUTOFF_ISO")
    if iso:
        dt = _parse_iso(iso)
        if dt:
            return dt
    sec = os.getenv("NAV_CUTOFF_SECAGO")
    if sec:
        try:
            s = int(sec)
            return datetime.now(timezone.utc) - timedelta(seconds=s)
        except Exception:
            pass
    return None


# ----------------------------- File readers ---------------------------------


def _read_nav_rows(path: str) -> List[Dict[str, Any]]:
    """Read nav_log.json list and normalize timestamps to key 't'."""
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            cut = _cutoff_dt()
            for p in data:
                # normalize timestamp key for series
                ts = p.get("timestamp") or p.get("t")
                if ts and "t" not in p:
                    p = {**p, "t": ts}
                # enforce cutoff if configured
                if cut:
                    dt = _parse_iso(p.get("t") or "")
                    if dt and dt < cut:
                        continue
                rows.append(p)
    except Exception:
        pass
    return rows


def _read_peak_file(path: str) -> float:
    if not os.path.exists(path):
        return 0.0
    try:
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f) or {}
        return float(j.get("peak_equity") or 0.0)
    except Exception:
        return 0.0


def _read_positions_snapshot(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"items": [], "updated_at": _now_iso()}
    try:
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f) or {}
        items = j.get("items") or []
        return {"items": items, "updated_at": j.get("updated_at") or _now_iso()}
    except Exception:
        return {"items": [], "updated_at": _now_iso()}


# --- KPI tail reader for nav card metrics
def _read_nav_tail_metrics(path: str) -> Dict[str, float]:
    """Return last point's metrics for nav KPIs (safe defaults)."""
    out = {
        "total_equity": 0.0,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "drawdown": 0.0,
    }
    if not os.path.exists(path):
        return out
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            last = data[-1]
            out["total_equity"] = float(last.get("equity", 0.0))
            out["realized_pnl"] = float(last.get("realized", 0.0))
            out["unrealized_pnl"] = float(last.get("unrealized", 0.0))
            out["drawdown"] = float(last.get("drawdown_pct", 0.0))
    except Exception:
        pass
    return out


# --------------------------- Filtering / metrics -----------------------------


def _is_good_nav_row(p: Dict[str, Any]) -> bool:
    try:
        eq = float(p.get("equity") or 0.0)
    except Exception:
        eq = 0.0
    if p.get("heartbeat_reason") == "exchange_unhealthy":
        return False
    return eq > 0.0


def _compute_peak_from_rows(rows: List[Dict[str, Any]]) -> float:
    try:
        return max((float(p.get("equity") or 0.0) for p in rows), default=0.0)
    except Exception:
        return 0.0


# --------------------------- Position normalization -------------------------


def _normalize_positions_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    now = _now_iso()
    for it in items or []:
        try:
            qty = float(it.get("qty", 0.0))
            entry = float(it.get("entry", it.get("entry_price", 0.0)))
            mark = float(
                it.get("mark_price", it.get("mark", it.get("latest_price", 0.0)))
            )
            pnl = (mark - entry) * qty  # recompute; don't trust inbound 'pnl'
            side = it.get("side") or ("LONG" if qty >= 0 else "SHORT")
            out.append(
                {
                    "symbol": it.get("symbol"),
                    "side": side,
                    "qty": qty,
                    "entry_price": entry,
                    "mark_price": mark,
                    "pnl": pnl,
                    "leverage": int(float(it.get("leverage", 1))),
                    "notional": abs(qty) * mark,
                    "ts": it.get("updated_at", now),
                }
            )
        except Exception:
            continue
    return out


# ------------------------------ Exposure KPIs --------------------------------


def _exposure_from_positions(items: List[Dict[str, Any]]) -> Dict[str, float]:
    gross = net = 0.0
    largest = 0.0
    for it in items or []:
        try:
            qty = float(it.get("qty", 0.0))
            price = float(
                it.get("mark_price", it.get("mark", it.get("latest_price", 0.0)))
            )
            pv = abs(qty) * price
            gross += pv
            net += qty * price
            largest = max(largest, pv)
        except Exception:
            continue
    return {
        "gross_exposure": gross,
        "net_exposure": net,
        "largest_position_value": largest,
    }


# ----------------------------- Firestore helpers ----------------------------


def _get_env() -> str:
    return os.getenv("ENV", "dev")


def _nav_doc_ref(db):
    return (
        db.collection("hedge").document(_get_env()).collection("state").document("nav")
    )


def _pos_doc_ref(db):
    return (
        db.collection("hedge")
        .document(_get_env())
        .collection("state")
        .document("positions")
    )


def _lb_doc_ref(db):
    return (
        db.collection("hedge")
        .document(_get_env())
        .collection("state")
        .document("leaderboard")
    )


def _health_doc_ref(db):
    return (
        db.collection("hedge")
        .document(_get_env())
        .collection("telemetry")
        .document("health")
    )


def _maybe_fetch_nav_doc(db) -> Dict[str, Any]:
    try:
        snap = _nav_doc_ref(db).get()
        if snap and snap.exists:
            return snap.to_dict() or {}
    except Exception:
        pass
    return {}


def _commit_nav(db, rows: List[Dict[str, Any]], peak: float) -> Dict[str, Any]:
    tail = _read_nav_tail_metrics(NAV_LOG)  # include KPIs for dashboard cards
    if not rows:
        payload = {
            "series": [],
            "peak_equity": float(peak),
            "updated_at": _now_iso(),
            **tail,
        }
    else:
        slim = rows[-MAX_POINTS:]
        payload = {
            "series": slim,
            "peak_equity": float(peak),
            "updated_at": slim[-1].get("t", _now_iso()),
            **tail,
        }
    _nav_doc_ref(db).set(payload, merge=True)
    return payload


def _commit_positions(db, positions: Dict[str, Any]) -> Dict[str, Any]:
    items = positions.get("items") or []
    norm = _normalize_positions_items(items)  # normalize to dashboard schema
    exp = _exposure_from_positions(norm)
    payload = {"items": norm, "updated_at": _now_iso(), **exp}
    _pos_doc_ref(db).set(payload, merge=True)
    return payload


def _commit_leaderboard(db, positions: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal leaderboard: aggregate notional and pnl by symbol."""
    items = positions.get("items") or []
    agg: Dict[str, Dict[str, float]] = {}
    for it in items:
        sym = it.get("symbol") or "UNKNOWN"
        agg.setdefault(sym, {"notional": 0.0, "pnl": 0.0})
        try:
            agg[sym]["notional"] += float(it.get("notional", 0.0))
            agg[sym]["pnl"] += float(it.get("pnl", 0.0))
        except Exception:
            pass
    leaderboard = sorted(
        [
            {"symbol": k, "notional": v["notional"], "pnl": v["pnl"]}
            for k, v in agg.items()
        ],
        key=lambda r: r["notional"],
        reverse=True,
    )
    payload = {"items": leaderboard, "updated_at": _now_iso()}
    _lb_doc_ref(db).set(payload, merge=True)
    return payload


def _publish_health(db, ok: bool, last_error: str) -> None:
    payload = {
        "firestore_ok": bool(ok),
        "last_error": str(last_error or ""),
        "ts": time.time(),
    }
    try:
        _health_doc_ref(db).set(payload, merge=True)
    except Exception as exc:
        print(f"[sync] WARN telemetry_write_failed: {exc}", flush=True)


# ------------------------------ Public API ----------------------------------


def _sync_once_with_db(db) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    # NAV rows with cutoff + zero-equity filter
    rows = [p for p in _read_nav_rows(NAV_LOG) if _is_good_nav_row(p)]

    # Tail equity guard: avoid writing garbage when executor/sync is cold
    tail_kpis = _read_nav_tail_metrics(NAV_LOG)
    if tail_kpis.get("total_equity", 0.0) <= 0.0:
        # still log positions exposure to nav doc (merge), but skip full write
        try:
            pos_snap = _read_positions_snapshot(SYNCED_STATE)
            exp = _exposure_from_positions(pos_snap.get("items") or [])
            if exp:
                _nav_doc_ref(db).set(exp | {"updated_at": _now_iso()}, merge=True)
        except Exception:
            pass
        return {}, {}, {}

    # Peak: choose the best available source in this order:
    #  1) existing Firestore nav doc (peak_equity)
    #  2) file peak_state.json
    #  3) computed from filtered rows
    nav_existing = _maybe_fetch_nav_doc(db)
    peak_doc = 0.0
    try:
        peak_doc = float(nav_existing.get("peak_equity") or 0.0)
    except Exception:
        pass
    peak_file = _read_peak_file(PEAK_STATE)
    peak_rows = _compute_peak_from_rows(rows)

    # If we're filtering history (cutoff active), prefer rows-based peak (local regime)
    cutoff_active = bool(os.getenv("NAV_CUTOFF_ISO") or os.getenv("NAV_CUTOFF_SECAGO"))
    if cutoff_active:
        peak = (
            max(peak_rows, peak_doc, peak_file)
            if peak_rows > 0
            else max(peak_doc, peak_file)
        )
    else:
        peak = max(peak_doc, peak_file, peak_rows)

    # Positions snapshot (executor populates synced_state.json)
    pos_snap = _read_positions_snapshot(SYNCED_STATE)

    # Firestore upserts
    nav_payload = _commit_nav(db, rows, peak)

    # attach exposure locally for prints and persist to the same doc
    exposure = _exposure_from_positions(pos_snap.get("items") or [])
    if exposure:
        nav_payload.update(exposure)
        _nav_doc_ref(db).set(exposure, merge=True)

    pos_payload = _commit_positions(db, pos_snap)
    lb_payload = _commit_leaderboard(db, pos_snap)

    # Console log
    try:
        updated_at = nav_payload.get("updated_at")
        print(
            f"[sync] upsert ok: points={len(nav_payload.get('series') or [])} "
            f"peak={nav_payload.get('peak_equity')} at={updated_at}",
            flush=True,
        )
    except Exception:
        pass

    return nav_payload, pos_payload, lb_payload


def sync_once() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Read files -> filter/compute -> upsert Firestore once.
    Returns (nav_payload, positions_payload, leaderboard_payload).
    """
    global _FIRESTORE_FAIL_COUNT
    try:
        with with_firestore() as db:
            result = _sync_once_with_db(db)
            _publish_health(db, True, "")
            _FIRESTORE_FAIL_COUNT = 0
            return result
    except Exception as exc:
        _FIRESTORE_FAIL_COUNT += 1
        print(
            f"[sync] WARN firestore_sync_error count={_FIRESTORE_FAIL_COUNT} err={exc}",
            flush=True,
        )
        if _FIRESTORE_FAIL_COUNT >= _FAILURE_THRESHOLD:
            print("[sync] ERROR firestore_degraded", flush=True)
        try:
            with with_firestore() as db:
                _publish_health(db, False, str(exc))
        except Exception:
            pass
        raise


# --------------------------------- Runner -----------------------------------


def _interval_seconds() -> int:
    try:
        return int(os.getenv("SYNC_INTERVAL_SEC", "20"))
    except Exception:
        return 20


def main_loop() -> None:
    env = os.getenv("ENV", "dev")
    print(
        "[sync] starting: ENV=%s interval=%ss files=(%s, %s, %s)"
        % (env, _interval_seconds(), NAV_LOG, PEAK_STATE, SYNCED_STATE),
        flush=True,
    )
    while True:
        try:
            sync_once()
        except Exception as e:
            print(f"[sync] ERROR: {e}", flush=True)
        time.sleep(_interval_seconds())


if __name__ == "__main__":
    main_loop()
