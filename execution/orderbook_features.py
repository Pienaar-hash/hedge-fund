from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Tuple

try:
    from .exchange_utils import _S, _BASE  # requests session/base
except Exception:  # pragma: no cover - offline
    _S = None
    _BASE = "https://fapi.binance.com"

_OB_CACHE: Dict[str, Tuple[float, Any]] = {}
_OB_TTL = 2.5  # seconds, rate-limit REST depth snapshots
_GATE_ENABLED = os.getenv("ORDERBOOK_GATE_ENABLED", "1").lower() not in {"0", "false", "no"}


def _fetch_depth(symbol: str, limit: int = 20) -> Optional[Dict[str, Any]]:
    if _S is None:  # offline/no-requests available
        return None
    url = f"{_BASE}/fapi/v1/depth?symbol={symbol}&limit={int(limit)}"
    r = _S.get(url, timeout=4)
    if not r.ok:
        return None
    return r.json()


def get_depth_snapshot(symbol: str, limit: int = 20) -> Optional[Dict[str, Any]]:
    now = time.time()
    key = f"{symbol}:{limit}"
    if key in _OB_CACHE:
        ts, data = _OB_CACHE[key]
        if (now - ts) <= _OB_TTL:
            return data
    data = _fetch_depth(symbol, limit=limit)
    _OB_CACHE[key] = (now, data)
    return data


def topn_imbalance(symbol: str, limit: int = 20) -> float:
    """Return simple buy-sell imbalance in top N levels: (bid_volume - ask_volume)/(sum).
    Positive => bid pressure; Negative => ask pressure. 0.0 on error.
    """
    try:
        d = get_depth_snapshot(symbol, limit=limit)
        if not d:
            return 0.0
        bids = sum(float(x[1]) for x in d.get("bids", [])[:limit])
        asks = sum(float(x[1]) for x in d.get("asks", [])[:limit])
        tot = max(1e-9, bids + asks)
        return (bids - asks) / tot
    except Exception:
        return 0.0


def evaluate_entry_gate(symbol: str, side: str, enabled: bool = True) -> Tuple[bool, Dict[str, Any]]:
    """Return (veto, info) based on simple imbalance gate.
    Veto on adverse pressure; allow slight boost when aligned.
    info = {"metric": float, "boost": float}
    """
    info: Dict[str, Any] = {"metric": 0.0, "boost": 0.0}
    effective = (
        enabled
        and _GATE_ENABLED
        and os.getenv("ORDERBOOK_GATE_DISABLED", "0").lower() not in {"1", "true", "yes"}
    )
    if not effective:
        return False, info
    m = topn_imbalance(symbol, limit=20)
    info["metric"] = m
    # thresholds tuned for micro-notional; keep conservative
    veto = False
    boost = 0.0
    side = str(side).upper()
    adverse_threshold = 0.30
    align_threshold = 0.40
    if side in ("BUY", "LONG") and m < -adverse_threshold:  # adverse ask pressure
        veto = True
    elif side in ("SELL", "SHORT") and m > adverse_threshold:  # adverse bid pressure
        veto = True
    else:
        # aligned slight boost (<= +5%)
        if side in ("BUY", "LONG") and m > align_threshold:
            boost = min(0.05, m)
        if side in ("SELL", "SHORT") and m < -align_threshold:
            boost = min(0.05, -m)
    info["boost"] = float(boost)
    return bool(veto), info
