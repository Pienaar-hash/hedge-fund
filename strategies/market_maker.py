from __future__ import annotations

import os
import time
from typing import Any, Dict, Iterable, List


def _b(x: str) -> bool:
    return str(x).strip().lower() in ("1", "true", "yes", "y", "on")


def generate_mm_intents() -> Iterable[Dict[str, Any]]:
    """Optional very small passive market maker for BTCUSDT.
    OFF by default; enable with env MARKET_MAKER_ENABLED=1.
    Produces best-effort intents for BUY/SELL reduceOnly=False with tiny cap.
    The executor DRY_RUN and risk engine provide guardrails.
    """
    if not _b(os.getenv("MARKET_MAKER_ENABLED", "0")):
        return []
    sym = os.getenv("MM_SYMBOL", "BTCUSDT")
    cap = float(os.getenv("MM_CAP", "5.0") or 5.0)
    lev = float(os.getenv("MM_LEV", "20") or 20.0)
    spread_thr = float(os.getenv("MM_SPREAD_THR", "0.05") or 0.05)  # 5 bps

    try:
        from execution.exchange_utils import get_price
    except Exception:
        def get_price(_s: str) -> float:  # type: ignore
            return 0.0

    px = float(get_price(sym))
    if px <= 0:
        return []

    # Simplified inside-spread check using Mark Price proxy if available in env
    try:
        mark = float(os.getenv("MM_MARK_PX", str(px)))
    except Exception:
        mark = px
    # bps spread estimate
    spr_bps = abs(px - mark) / px * 10000.0
    if spr_bps < spread_thr * 10000.0:
        return []

    t = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    # Place both sides tiny quotes (executor/risk may block) — passive effect
    return [
        {
            "timestamp": t,
            "symbol": sym,
            "signal": "BUY",
            "capital_per_trade": cap,
            "leverage": lev,
            "positionSide": "LONG",
            "reduceOnly": False,
            "source": "market_maker",
        },
        {
            "timestamp": t,
            "symbol": sym,
            "signal": "SELL",
            "capital_per_trade": cap,
            "leverage": lev,
            "positionSide": "SHORT",
            "reduceOnly": False,
            "source": "market_maker",
        },
    ]

