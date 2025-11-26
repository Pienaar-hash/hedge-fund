from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

LotKey = Tuple[str, str]


@dataclass
class Lot:
    side: str
    qty: float
    price: float


@dataclass
class Fill:
    symbol: str
    side: str
    qty: float
    price: float
    fee: float = 0.0
    position_side: Optional[str] = None
    reduce_only: bool = False


@dataclass
class CloseResult:
    symbol: str
    position_side: str
    closed_qty: float
    realized_pnl: float
    fees: float
    position_before: float
    position_after: float


class PositionTracker:
    """FIFO position tracker that yields realized PnL for reducing fills."""

    _EPS = 1e-9

    def __init__(self) -> None:
        self._lots: Dict[LotKey, Deque[Lot]] = defaultdict(deque)

    @staticmethod
    def _normalize_side(side: str) -> str:
        value = (side or "").upper()
        if value not in {"BUY", "SELL"}:
            raise ValueError(f"invalid side: {side}")
        return value

    @staticmethod
    def _normalize_position_side(position_side: Optional[str]) -> str:
        if not position_side:
            return "BOTH"
        value = position_side.upper()
        if value not in {"LONG", "SHORT", "BOTH"}:
            return "BOTH"
        return value

    def _key(self, symbol: str, position_side: Optional[str]) -> LotKey:
        return (symbol.upper(), self._normalize_position_side(position_side))

    @staticmethod
    def _net_position(lots: Deque[Lot]) -> float:
        net = 0.0
        for lot in lots:
            sign = 1.0 if lot.side == "BUY" else -1.0
            net += sign * lot.qty
        return net

    def apply_fill(self, fill: Fill) -> Optional[CloseResult]:
        qty = float(fill.qty or 0.0)
        if qty <= 0:
            return None
        side = self._normalize_side(fill.side)
        key = self._key(fill.symbol, fill.position_side)
        lots = self._lots[key]
        position_before = self._net_position(lots)

        opposite = "SELL" if side == "BUY" else "BUY"
        remaining = qty
        realized = 0.0
        closed_qty = 0.0
        fees_closed = 0.0

        while remaining > self._EPS and lots and lots[0].side == opposite:
            lot = lots[0]
            matched = min(remaining, lot.qty)
            if side == "SELL":
                realized += (fill.price - lot.price) * matched
            else:
                realized += (lot.price - fill.price) * matched
            lot.qty -= matched
            remaining -= matched
            closed_qty += matched
            if fill.qty:
                fees_closed += fill.fee * (matched / fill.qty)
            if lot.qty <= self._EPS:
                lots.popleft()
            else:
                lots[0] = lot

        if remaining > self._EPS:
            lots.append(Lot(side=side, qty=remaining, price=fill.price))

        position_after = self._net_position(lots)
        if closed_qty <= self._EPS:
            return None

        return CloseResult(
            symbol=fill.symbol,
            position_side=self._normalize_position_side(fill.position_side),
            closed_qty=closed_qty,
            realized_pnl=realized,
            fees=fees_closed,
            position_before=position_before,
            position_after=position_after,
        )


# ---------------------------------------------------------------------------
# Log-backed helpers used by execution.utils.metrics

LOG_DIR = Path(os.getenv("EXEC_LOG_DIR") or "logs/execution")
EXECUTED_PATH = Path(os.getenv("ORDER_EVENTS_PATH") or (LOG_DIR / "orders_executed.jsonl"))
READ_LIMIT = int(os.getenv("EXEC_LOG_MAX_ROWS", "5000") or 5000)


def _to_float(value: Any) -> Optional[float]:
    try:
        if value in (None, "", "nan"):
            return None
        return float(value)
    except Exception:
        pass
    # Try parsing ISO timestamp strings
    if isinstance(value, str) and "T" in value:
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.timestamp()
        except Exception:
            pass
    return None


def _read_jsonl(path: Path, limit: int) -> List[Dict[str, Any]]:
    if not path.exists() or limit <= 0:
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except Exception:
        return []
    if len(rows) > limit:
        rows = rows[-limit:]
    return rows


def _recent_executed(window_days: int) -> List[Dict[str, Any]]:
    records = _read_jsonl(EXECUTED_PATH, READ_LIMIT)
    if not records:
        return []
    cutoff = time.time() - max(window_days, 0) * 86400.0
    filtered: List[Dict[str, Any]] = []
    for rec in records:
        ts = rec.get("ts") or rec.get("ts_fill_last") or rec.get("timestamp")
        ts_val = _to_float(ts)
        if ts_val is None:
            continue
        if ts_val < cutoff:
            continue
        rec["_ts"] = ts_val
        rec["symbol"] = str(rec.get("symbol") or "").upper()
        filtered.append(rec)
    return filtered


def _realized_pnl_value(record: Dict[str, Any]) -> Optional[float]:
    for key in ("realized_pnl", "realized_pnl_usd", "realizedPnlUsd", "pnl_usd", "pnl"):
        val = record.get(key)
        num = _to_float(val)
        if num is not None:
            return num
    return None


def _fee_value(record: Dict[str, Any]) -> Optional[float]:
    for key in ("fee_total", "fees", "fee", "commission"):
        val = record.get(key)
        num = _to_float(val)
        if num is not None:
            return num
    return None


def get_gross_realized(symbol: Optional[str] = None, window_days: int = 7) -> float:
    events = _recent_executed(window_days)
    total = 0.0
    for rec in events:
        if symbol and rec.get("symbol") != symbol.upper():
            continue
        pnl = _realized_pnl_value(rec)
        if pnl is not None:
            total += pnl
    return total


def get_fees(symbol: Optional[str] = None, window_days: int = 7) -> float:
    """
    Compatibility shim for dashboards. Returns 0.0 on any failure so callers
    never crash even if fee tracking is unavailable.
    """
    try:
        events = _recent_executed(window_days)
        total = 0.0
        for rec in events:
            if symbol and rec.get("symbol") != symbol.upper():
                continue
            fee = _fee_value(rec)
            if fee is not None:
                total += fee
        return total
    except Exception:
        return 0.0


def get_symbol_stats(symbol: str, window_days: int = 7) -> Dict[str, float]:
    events = _recent_executed(window_days)
    returns: List[float] = []
    sym = symbol.upper()
    for rec in events:
        if rec.get("symbol") != sym:
            continue
        pnl = _realized_pnl_value(rec)
        qty = _to_float(rec.get("executedQty") or rec.get("qty"))
        price = _to_float(rec.get("avgPrice") or rec.get("price"))
        notional = None
        if qty is not None and price is not None:
            notional = abs(qty * price)
        if pnl is None:
            continue
        if notional and notional > 0:
            returns.append(pnl / notional)
        else:
            returns.append(pnl)
    if not returns:
        return {"mean": 0.0, "std": 0.0}
    arr = np.array(returns, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {"mean": mean, "std": std}


def get_symbol_atr(symbol: str, lookback_bars: int = 50, median_only: bool = False) -> Dict[str, float]:
    """
    Approximate ATR using executed prices when no dedicated ATR feed exists.
    """
    events = _recent_executed(window_days=7)
    prices: List[float] = []
    sym = symbol.upper()
    for rec in events:
        if rec.get("symbol") != sym:
            continue
        px = _to_float(rec.get("avgPrice") or rec.get("price"))
        if px is not None and px > 0:
            prices.append(px)
    if len(prices) < 2:
        return {"atr_pct": 0.0, "atr_pct_median": 0.0}
    arr = np.array(prices[-lookback_bars:], dtype=float)
    diffs = np.abs(np.diff(arr))
    if diffs.size == 0:
        return {"atr_pct": 0.0, "atr_pct_median": 0.0}
    atr = float(diffs.mean())
    atr_med = float(np.median(diffs))
    mid = float(np.mean(arr))
    atr_pct = (atr / mid) * 100 if mid > 0 else 0.0
    atr_pct_median = (atr_med / mid) * 100 if mid > 0 else 0.0
    return {"atr_pct": atr_pct, "atr_pct_median": atr_pct_median}


def get_hourly_expectancy(symbol: str, window_days: int = 7) -> Dict[int, float]:
    events = _recent_executed(window_days)
    buckets: Dict[int, List[float]] = {}
    sym = symbol.upper()
    for rec in events:
        if rec.get("symbol") != sym:
            continue
        pnl = _realized_pnl_value(rec)
        ts = rec.get("_ts")
        if pnl is None or ts is None:
            continue
        hour = time.gmtime(ts).tm_hour
        buckets.setdefault(hour, []).append(pnl)
    return {hour: float(sum(vals) / len(vals)) for hour, vals in buckets.items() if vals}
