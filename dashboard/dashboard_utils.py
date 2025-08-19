
import os
import sys
from typing import Any, Dict, List, Tuple
from pathlib import Path

import pandas as pd

# Ensure absolute project root in sys.path for imports
PROJECT_ROOT = "/root/hedge-fund"
if PROJECT_ROOT not in sys.path and os.path.isdir(PROJECT_ROOT):
    sys.path.insert(0, PROJECT_ROOT)

# Firestore client (supports multiple layouts via utils.firestore_client)
try:
    from utils.firestore_client import get_db  # repo's utils/
except ModuleNotFoundError:
    # fallback to root-level or package layout
    from firestore_client import get_db  # type: ignore


# --------------------------- Firestore helpers -------------------------------
def get_firestore_connection():
    return get_db()

def fetch_state_document(doc_name: str, env: str = None) -> Dict[str, Any]:
    """Fetch hedge/{ENV}/state/{doc_name} (no 'live' fallback here—keep it explicit)."""
    db = get_db()
    env = env or os.getenv("ENV", "prod")
    ref = db.collection("hedge").document(env).collection("state").document(doc_name)
    snap = ref.get()
    return snap.to_dict() or {} if snap.exists else {}

# --------------------------- Formatting utils --------------------------------
def fmt_ccy(v: Any) -> str:
    try:
        return f"{float(v):,.2f}"
    except Exception:
        return "0.00"

def fmt_pct(v: Any) -> str:
    try:
        return f"{float(v) * 100:+.2f}%"
    except Exception:
        return "+0.00%"

# --------------------------- NAV parsing -------------------------------------
def _coalesce(*vals, default=None):
    for v in vals:
        if v is not None:
            return v
    return default

def parse_nav_to_df_and_kpis(nav: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Accepts nav doc with:
      series: list of dicts like {"t": iso, "equity": float} (preferred)
              or list[list] [[iso, equity]] (tolerated)
              or dict {iso: equity}
    Returns (df indexed by t, kpis dict)
    """
    series = nav.get("series")
    rows: List[List[Any]] = []

    if isinstance(series, list) and series and isinstance(series[0], dict):
        rows = [[r.get("t"), r.get("equity")] for r in series]
    elif isinstance(series, list):
        rows = series
    elif isinstance(series, dict):
        rows = [[t, v] for t, v in sorted(series.items())]

    if rows:
        df = pd.DataFrame(rows, columns=["t","equity"])
        df["t"] = pd.to_datetime(df["t"], utc=True, errors="coerce")
        df = df.dropna(subset=["t"]).sort_values("t").set_index("t")
    else:
        df = pd.DataFrame(columns=["equity"])

    kpis = {
        "total_equity": float(nav.get("total_equity") or (df["equity"].iloc[-1] if not df.empty else 0.0)),
        "peak_equity": float(_coalesce(nav.get("peak_equity"), nav.get("peak"), 0.0)),
        "realized_pnl": float(nav.get("realized_pnl") or 0.0),
        "unrealized_pnl": float(nav.get("unrealized_pnl") or 0.0),
        "drawdown": float(nav.get("drawdown") or 0.0),
        "updated_at": _coalesce(nav.get("updated_at"), "—"),
    }
    return df, kpis

# --------------------------- Positions utils ---------------------------------
def positions_sorted(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort by absolute notional desc; compute notional if missing."""
    def _notional(it: Dict[str, Any]) -> float:
        try:
            if "notional" in it and it["notional"] is not None:
                return float(it["notional"])
            qty = float(it.get("qty", 0.0))
            px = float(it.get("mark_price", it.get("latest_price", 0.0)))
            return abs(qty) * float(px)
        except Exception:
            return 0.0
    return sorted(items or [], key=lambda it: abs(_notional(it)), reverse=True)

# --------------------------- Trade log reader --------------------------------
def read_trade_log_tail(path: str = "trade_log.json", tail: int = 10) -> List[Dict[str, Any]]:
    """Read last N entries from a local JSONL or JSON trade log.
    Supports:
      - JSON lines (one trade per line)
      - JSON array of trade dicts
    Normalizes keys to: ts, symbol, side, qty, price, notional, realized_pnl, comment, strategy
    """
    p = Path(path)
    if not p.exists():
        return []
    try:
        text = p.read_text(encoding="utf-8")
        trades: List[Dict[str, Any]] = []
        if text.strip().startswith("["):
            import json
            data = json.loads(text)
            if isinstance(data, list):
                trades = data
        else:
            # JSONL
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    import json
                    trades.append(json.loads(line))
                except Exception:
                    continue
        # normalize & tail
        def _norm(t: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "ts": t.get("ts") or t.get("timestamp") or t.get("time"),
                "symbol": t.get("symbol"),
                "side": (t.get("side") or "").upper(),
                "qty": float(t.get("qty", t.get("quantity", 0.0)) or 0.0),
                "price": float(t.get("price") or t.get("fill_price") or 0.0),
                "notional": float(t.get("notional", 0.0)) if t.get("notional") is not None else (float(t.get("qty", 0.0)) * float(t.get("price") or 0.0)),
                "realized_pnl": float(t.get("realized_pnl", t.get("pnl", 0.0)) or 0.0),
                "comment": t.get("comment") or t.get("reason"),
                "strategy": t.get("strategy") or t.get("strategy_name"),
            }
        normed = [_norm(t) for t in trades]
        # sort desc by ts if present
        try:
            normed.sort(key=lambda r: r.get("ts") or "", reverse=True)
        except Exception:
            pass
        return normed[:tail]
    except Exception:
        return []
