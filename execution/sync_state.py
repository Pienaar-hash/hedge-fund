"""
Granular Firestore sync API for investor-facing state.

Contracts (must match dashboard readers exactly):

leaderboard doc:
{
  "updated_at": ISO-UTC str,
  "items": [
    {"strategy": str, "cagr": float, "sharpe": float, "mdd": float,
     "win_rate": float, "trades": int, "pnl": float, "equity": float, "rank": int}
  ]
}

nav doc:
{
  "updated_at": ISO-UTC str,
  "series": [{"ts": ISO-UTC str, "equity": float}],
  "total_equity": float,
  "realized_pnl": float,
  "unrealized_pnl": float,
  "peak_equity": float,
  "drawdown": float
}

positions doc:
{
  "updated_at": ISO-UTC str,
  "items": [
    {"symbol": str, "side": str, "qty": float, "entry_price": float,
     "mark_price": float, "pnl": float, "leverage": int, "notional": float, "ts": ISO-UTC str}
  ]
}
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping

# ---------- helpers ----------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_doc(db, env: str, name: str):
    return (
        db.collection("hedge")
        .document(env)
        .collection("state")
        .document(name)
    )


def _merge_set(doc_ref, payload: Mapping[str, Any]) -> None:
    # All state writes are idempotent upserts
    doc_ref.set(dict(payload), merge=True)


# ---------- shape guards (lenient but protective) ----------

def _coerce_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _coerce_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def _ensure_iso(ts: Any) -> str:
    if isinstance(ts, str):
        return ts
    if isinstance(ts, (int, float)):
        # assume seconds epoch
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    if isinstance(ts, datetime):
        return ts.astimezone(timezone.utc).isoformat()
    return _now_iso()


# ---------- public API ----------

def sync_leaderboard(db, data: Dict[str, Any], env: str) -> None:
    """Upsert leaderboard state.
    Expected: {"items": [{strategy,cagr,sharpe,mdd,win_rate,trades,pnl,equity,rank}], ...}
    """
    raw_items = data.get("items", []) or []
    items: List[Dict[str, Any]] = []
    for row in raw_items:
        if not isinstance(row, Mapping):
            continue
        items.append(
            {
                "strategy": str(row.get("strategy", "")),
                "cagr": _coerce_float(row.get("cagr")),
                "sharpe": _coerce_float(row.get("sharpe")),
                "mdd": _coerce_float(row.get("mdd")),
                "win_rate": _coerce_float(row.get("win_rate")),
                "trades": _coerce_int(row.get("trades")),
                "pnl": _coerce_float(row.get("pnl")),
                "equity": _coerce_float(row.get("equity")),
                "rank": _coerce_int(row.get("rank", 0)),
            }
        )

    payload = {
        "updated_at": _now_iso(),
        "items": items,
    }
    # allow passthrough of optional fields if provided
    for k in ("note", "as_of"):
        if k in data:
            payload[k] = data[k]

    _merge_set(_state_doc(db, env, "leaderboard"), payload)


def sync_nav(db, data: Dict[str, Any], env: str) -> None:
    """Upsert NAV state.
    Expected keys: series, total_equity, realized_pnl, unrealized_pnl, peak_equity, drawdown
    """
    raw_series = data.get("series", []) or []
    series: List[Dict[str, Any]] = []
    for pt in raw_series:
        if not isinstance(pt, Mapping):
            continue
        series.append({"ts": _ensure_iso(pt.get("ts")), "equity": _coerce_float(pt.get("equity"))})

    payload = {
        "updated_at": _now_iso(),
        "series": series,
        "total_equity": _coerce_float(data.get("total_equity")),
        "realized_pnl": _coerce_float(data.get("realized_pnl")),
        "unrealized_pnl": _coerce_float(data.get("unrealized_pnl")),
        "peak_equity": _coerce_float(data.get("peak_equity")),
        "drawdown": _coerce_float(data.get("drawdown")),
    }

    _merge_set(_state_doc(db, env, "nav"), payload)


def sync_positions(db, positions: Dict[str, Any], env: str) -> None:
    """Upsert positions state.
    Expected: {"items": [{symbol, side, qty, entry_price, mark_price, pnl, leverage, notional, ts}]}
    """
    raw_items = positions.get("items", []) or []
    items: List[Dict[str, Any]] = []
    for row in raw_items:
        if not isinstance(row, Mapping):
            continue
        items.append(
            {
                "symbol": str(row.get("symbol", "")),
                "side": str(row.get("side", "")).upper() or "FLAT",
                "qty": _coerce_float(row.get("qty")),
                "entry_price": _coerce_float(row.get("entry_price")),
                "mark_price": _coerce_float(row.get("mark_price")),
                "pnl": _coerce_float(row.get("pnl")),
                "leverage": _coerce_int(row.get("leverage", 1)),
                "notional": _coerce_float(row.get("notional")),
                "ts": _ensure_iso(row.get("ts")),
            }
        )

    payload = {
        "updated_at": _now_iso(),
        "items": items,
    }

    _merge_set(_state_doc(db, env, "positions"), payload)


__all__ = [
    "sync_leaderboard",
    "sync_nav",
    "sync_positions",
]
