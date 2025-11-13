"""
Rolling expectancy derived from recent trade logs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from execution import trade_logs

LOOKBACK_N = 300


def rolling_expectancy(symbol: str, lookback: int = LOOKBACK_N) -> Optional[float]:
    """
    Mean realized PnL per trade (USDT) over the lookback window for a symbol.
    Returns None when there is insufficient data.
    """
    trades: List[Dict[str, Any]] = trade_logs.get_recent_trades(
        symbol=symbol,
        limit=lookback,
    )
    if not trades or len(trades) < 10:
        return None
    values = []
    for trade in trades:
        try:
            values.append(float(trade.get("realized_pnl", 0.0)))
        except (TypeError, ValueError):
            continue
    if len(values) < 10:
        return None
    return float(np.mean(values))
