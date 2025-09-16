#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Tuple

try:
    from utils.firestore_client import get_db  # type: ignore
except Exception:
    get_db = None  # type: ignore


def _paths(db, env: str) -> Dict[str, Any]:
    root = db.collection("hedge").document(env)
    state = root.collection("state")
    return {
        "nav": state.document("nav"),
        "positions": state.document("positions"),
        "trades": root.collection("trades"),
        "risk": root.collection("risk"),
    }


def _to_epoch(v: Any) -> float:
    try:
        if isinstance(v, (int, float)):
            x = float(v)
            return x / 1000.0 if x > 1e12 else x
        # Firestore timestamp has .timestamp()
        if hasattr(v, "timestamp"):
            return float(v.timestamp())
    except Exception:
        pass
    return 0.0


def run(env: str) -> Tuple[bool, Dict[str, Any]]:
    if get_db is None:
        return False, {"error": "firestore libs unavailable"}
    db = get_db()
    p = _paths(db, env)
    out: Dict[str, Any] = {}

    # NAV
    nav_doc = p["nav"].get()
    nav_series = []
    if getattr(nav_doc, "exists", False):
        data = nav_doc.to_dict() or {}
        for _, v in data.items():
            if not isinstance(v, dict):
                continue
            ts = _to_epoch(v.get("t") or v.get("time") or v.get("ts"))
            nav_series.append(ts)
    out["nav_count"] = len(nav_series)
    out["nav_newest_ts"] = max(nav_series) if nav_series else None

    # Positions
    pos_doc = p["positions"].get()
    pos_rows = 0
    if getattr(pos_doc, "exists", False):
        d = pos_doc.to_dict() or {}
        if isinstance(d.get("rows"), list):
            pos_rows = len(d.get("rows", []))
    out["positions_count"] = pos_rows

    # Trades (24h)
    now = time.time()
    tr_rows: List[float] = []
    docs = list(p["trades"].order_by("ts", direction="DESCENDING").limit(1000).stream())
    env_vals = set()
    tn_vals = set()
    for d in docs:
        x = d.to_dict() or {}
        t = _to_epoch(x.get("ts") or x.get("time") or x.get("t"))
        if (now - t) <= 24 * 3600:
            tr_rows.append(t)
        if "env" in x:
            env_vals.add(str(x.get("env")))
        if "testnet" in x:
            tn_vals.add(bool(x.get("testnet")))
    out["trades_24h"] = len(tr_rows)

    # Risk (24h)
    rk_rows: List[float] = []
    docs_r = list(p["risk"].order_by("ts", direction="DESCENDING").limit(1000).stream())
    for d in docs_r:
        x = d.to_dict() or {}
        t = _to_epoch(x.get("ts") or x.get("time") or x.get("t"))
        if (now - t) <= 24 * 3600:
            rk_rows.append(t)
        if "env" in x:
            env_vals.add(str(x.get("env")))
        if "testnet" in x:
            tn_vals.add(bool(x.get("testnet")))
    out["risk_24h"] = len(rk_rows)

    # Mixing detection
    mixed = (len(env_vals) > 1) or (len(tn_vals) > 1)
    out["mixed"] = bool(mixed)
    return (not mixed), out


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Firestore doctor: counts and mixing detector")
    ap.add_argument("--env", default=os.getenv("ENV", "prod"))
    args = ap.parse_args(argv)
    ok, info = run(args.env)
    print({"ok": ok, **info})
    # Exit non-zero if mixing detected and ENV=prod
    if not ok and str(args.env) == "prod":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

