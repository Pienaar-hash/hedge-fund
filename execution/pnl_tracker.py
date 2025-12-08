from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Tuple

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


def _load_trades(path: Path | None = None) -> List[Dict[str, Any]]:
    target = Path(path) if path else TRADES_LOG_PATH
    return _read_jsonl(target, READ_LIMIT)


def _load_positions(path: Path | None = None) -> List[Dict[str, Any]]:
    target = Path(path) if path else POSITIONS_STATE_PATH
    try:
        if not target.exists():
            return []
        text = target.read_text(encoding="utf-8")
        payload = json.loads(text)
    except Exception:
        return []
    if isinstance(payload, dict):
        rows = payload.get("rows")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
        return []
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def _load_state_json(path: Path | None) -> Dict[str, Any]:
    target = Path(path) if path else Path()
    try:
        if not target.exists():
            return {}
        with target.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _init_bucket() -> Dict[str, Any]:
    return {"realized": 0.0, "unrealized": 0.0, "total": 0.0, "trade_count": 0}


def _bucket_hybrid_score(score: Any) -> Optional[str]:
    try:
        if score is None:
            return None
        s = float(score)
    except Exception:
        return None
    s = max(0.0, min(1.0, s))
    decile = int(s * 10.0)
    if decile == 10:
        decile = 9
    return str(decile)


def _bucket_trend_strength(strength: Any) -> Optional[str]:
    try:
        if strength is None:
            return None
        s = float(strength)
    except Exception:
        return None
    s = max(0.0, min(1.0, s))
    if s < 0.33:
        return "weak"
    if s < 0.66:
        return "medium"
    return "strong"


def _bucket_carry_regime(*, carry_long: Any, carry_short: Any, eps: float = 1e-6) -> str:
    try:
        l = float(carry_long)
    except Exception:
        l = 0.0
    try:
        s = float(carry_short)
    except Exception:
        s = 0.0
    if l > s + eps:
        return "long_carry"
    if s > l + eps:
        return "short_carry"
    return "neutral"


def _update_factor_buckets(
    factors_state: Dict[str, Any],
    *,
    hybrid_decile: Optional[str],
    trend_bucket: Optional[str],
    carry_regime: Optional[str],
    realized_pnl: float,
) -> None:
    """Update regimes.factors buckets in-place."""

    def _bump(bucket_name: str, key: Optional[str]) -> None:
        if key is None:
            return
        bucket = factors_state.setdefault(bucket_name, {})
        slot = bucket.setdefault(
            key,
            {
                "trade_count": 0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
            },
        )
        slot["trade_count"] += 1
        slot["total_pnl"] += float(realized_pnl)

    _bump("hybrid_score_decile", hybrid_decile)
    _bump("trend_strength_bucket", trend_bucket)
    _bump("carry_regime", carry_regime)


def _finalize_factor_buckets(factors_state: Dict[str, Any]) -> None:
    for bucket in factors_state.values():
        for slot in bucket.values():
            n = slot.get("trade_count", 0)
            if n > 0:
                slot["avg_pnl"] = slot.get("total_pnl", 0.0) / float(n)
            else:
                slot["avg_pnl"] = 0.0


def _update_exit_buckets(
    exits_state: Dict[str, Any],
    *,
    reason: str,
    strategy: Optional[str],
    symbol: Optional[str],
    realized_pnl: float,
    rr: Optional[float],
    atr_regime: Optional[int],
    dd_regime: Optional[int],
    risk_mode: Optional[str],
    day_key: Optional[str],
) -> None:
    """
    Mutate exits_state with summary, strategy, symbol, and regime aggregations.
    """
    exits_state.setdefault(
        "summary",
        {
            "total_exits": 0,
            "tp_hits": 0,
            "sl_hits": 0,
            "tp_ratio": 0.0,
            "avg_rr_tp": None,
            "avg_rr_sl": None,
            "avg_exit_pnl": 0.0,
            "total_exit_pnl": 0.0,
            "_rr_tp_sum": 0.0,
            "_rr_tp_count": 0,
            "_rr_sl_sum": 0.0,
            "_rr_sl_count": 0,
        },
    )

    summary = exits_state["summary"]
    summary["total_exits"] += 1
    summary["total_exit_pnl"] += realized_pnl

    if reason == "tp":
        summary["tp_hits"] += 1
        if rr is not None:
            summary["_rr_tp_sum"] += rr
            summary["_rr_tp_count"] += 1
    elif reason == "sl":
        summary["sl_hits"] += 1
        if rr is not None:
            summary["_rr_sl_sum"] += rr
            summary["_rr_sl_count"] += 1

    if strategy:
        by_strat = exits_state.setdefault("by_strategy", {})
        strat_bucket = by_strat.setdefault(
            strategy,
            {
                "total_exits": 0,
                "tp_hits": 0,
                "sl_hits": 0,
                "tp_ratio": 0.0,
                "avg_rr_tp": None,
                "avg_rr_sl": None,
                "avg_exit_pnl": 0.0,
                "total_exit_pnl": 0.0,
                "_rr_tp_sum": 0.0,
                "_rr_tp_count": 0,
                "_rr_sl_sum": 0.0,
                "_rr_sl_count": 0,
            },
        )
        strat_bucket["total_exits"] += 1
        strat_bucket["total_exit_pnl"] += realized_pnl
        if reason == "tp":
            strat_bucket["tp_hits"] += 1
            if rr is not None:
                strat_bucket["_rr_tp_sum"] += rr
                strat_bucket["_rr_tp_count"] += 1
        elif reason == "sl":
            strat_bucket["sl_hits"] += 1
            if rr is not None:
                strat_bucket["_rr_sl_sum"] += rr
                strat_bucket["_rr_sl_count"] += 1

    if symbol:
        by_sym = exits_state.setdefault("by_symbol", {})
        sym_bucket = by_sym.setdefault(
            symbol,
            {
                "total_exits": 0,
                "tp_hits": 0,
                "sl_hits": 0,
                "tp_ratio": 0.0,
                "avg_exit_pnl": 0.0,
                "total_exit_pnl": 0.0,
            },
        )
        sym_bucket["total_exits"] += 1
        sym_bucket["total_exit_pnl"] += realized_pnl
        if reason == "tp":
            sym_bucket["tp_hits"] += 1
        elif reason == "sl":
            sym_bucket["sl_hits"] += 1

    regimes_state = exits_state.setdefault("regimes", {})

    def _bump_regime(bucket_name: str, key: Any) -> None:
        if key is None:
            return
        bucket = regimes_state.setdefault(bucket_name, {})
        slot = bucket.setdefault(
            str(key),
            {
                "total_exits": 0,
                "tp_hits": 0,
                "sl_hits": 0,
                "total_exit_pnl": 0.0,
            },
        )
        slot["total_exits"] += 1
        slot["total_exit_pnl"] += realized_pnl
        if reason == "tp":
            slot["tp_hits"] += 1
        elif reason == "sl":
            slot["sl_hits"] += 1

    _bump_regime("atr", atr_regime)
    _bump_regime("dd", dd_regime)
    _bump_regime("risk_mode", risk_mode)
    _bump_regime("day", day_key)


def _finalize_exits_state(exits_state: Dict[str, Any]) -> None:
    if not exits_state:
        return

    def _finalize_bucket(bucket: Dict[str, Any]) -> None:
        total = bucket.get("total_exits", 0)
        tp = bucket.get("tp_hits", 0)
        if total > 0:
            bucket["tp_ratio"] = tp / float(total)
            bucket["avg_exit_pnl"] = bucket.get("total_exit_pnl", 0.0) / float(total)
        else:
            bucket["tp_ratio"] = 0.0
            bucket["avg_exit_pnl"] = 0.0

        tp_sum = bucket.pop("_rr_tp_sum", 0.0)
        tp_count = bucket.pop("_rr_tp_count", 0)
        sl_sum = bucket.pop("_rr_sl_sum", 0.0)
        sl_count = bucket.pop("_rr_sl_count", 0)
        bucket["avg_rr_tp"] = (tp_sum / tp_count) if tp_count > 0 else None
        bucket["avg_rr_sl"] = (sl_sum / sl_count) if sl_count > 0 else None

    if "summary" in exits_state:
        _finalize_bucket(exits_state["summary"])

    for strat_bucket in exits_state.get("by_strategy", {}).values():
        _finalize_bucket(strat_bucket)

    for sym_bucket in exits_state.get("by_symbol", {}).values():
        total = sym_bucket.get("total_exits", 0)
        tp = sym_bucket.get("tp_hits", 0)
        if total > 0:
            sym_bucket["tp_ratio"] = tp / float(total)
            sym_bucket["avg_exit_pnl"] = sym_bucket.get("total_exit_pnl", 0.0) / float(total)
        else:
            sym_bucket["tp_ratio"] = 0.0
            sym_bucket["avg_exit_pnl"] = 0.0

    for regime_bucket in exits_state.get("regimes", {}).values():
        for slot in regime_bucket.values():
            total = slot.get("total_exits", 0)
            tp = slot.get("tp_hits", 0)
            if total > 0:
                slot["tp_ratio"] = tp / float(total)
            else:
                slot["tp_ratio"] = 0.0


def _regime_key(value: Any) -> str | None:
    try:
        idx = int(value)
    except Exception:
        return None
    if 0 <= idx <= 3:
        return str(idx)
    return None


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


# ---------------------------------------------------------------------------
# Equity Series Export (v7)
# ---------------------------------------------------------------------------

STATE_DIR = Path(os.getenv("STATE_DIR") or "logs/state")
EQUITY_PATH = STATE_DIR / "equity.json"
TRADES_LOG_PATH = Path(os.getenv("TRADES_LOG_PATH") or (LOG_DIR / "trades.jsonl"))
POSITIONS_STATE_PATH = Path(os.getenv("POSITIONS_STATE_PATH") or (STATE_DIR / "positions.json"))
PNL_ATTRIBUTION_PATH = STATE_DIR / "pnl_attribution.json"
REGIMES_PATH = STATE_DIR / "regimes.json"
RISK_SNAPSHOT_PATH = STATE_DIR / "risk_snapshot.json"


def _compute_equity_series(
    records: List[Dict[str, Any]],
    initial_equity: float = 0.0,
) -> Dict[str, List[float]]:
    """
    Compute equity series from PnL records.
    
    Returns dict with:
    - timestamps: Unix timestamps
    - equity: Cumulative equity values
    - pnl: Per-record PnL values
    - drawdown: Drawdown fraction at each point
    """
    if not records:
        return {
            "timestamps": [],
            "equity": [],
            "pnl": [],
            "drawdown": [],
        }
    
    # Sort by timestamp
    sorted_records = sorted(records, key=lambda r: r.get("_ts", 0))
    
    timestamps: List[float] = []
    equity_values: List[float] = []
    pnl_values: List[float] = []
    drawdown_values: List[float] = []
    
    cumulative = initial_equity
    peak = initial_equity
    
    for rec in sorted_records:
        ts = rec.get("_ts")
        if ts is None:
            continue
        
        pnl = _realized_pnl_value(rec)
        if pnl is None:
            pnl = 0.0
        
        cumulative += pnl
        peak = max(peak, cumulative)
        
        # Calculate drawdown fraction
        if peak > 0:
            dd_frac = (peak - cumulative) / peak
        else:
            dd_frac = 0.0
        
        timestamps.append(float(ts))
        equity_values.append(cumulative)
        pnl_values.append(pnl)
        drawdown_values.append(dd_frac)
    
    return {
        "timestamps": timestamps,
        "equity": equity_values,
        "pnl": pnl_values,
        "drawdown": drawdown_values,
    }


def _compute_rolling_returns(
    pnl_values: List[float],
    window: int = 20,
) -> List[float]:
    """Compute rolling sum of PnL over a window."""
    if not pnl_values or window <= 0:
        return []
    
    rolling: List[float] = []
    for i in range(len(pnl_values)):
        start = max(0, i - window + 1)
        window_sum = sum(pnl_values[start:i + 1])
        rolling.append(window_sum)
    
    return rolling


def export_equity_series(
    window_days: int = 30,
    initial_equity: float = 0.0,
) -> Dict[str, Any]:
    """
    Export equity series to logs/state/equity.json.
    
    Args:
        window_days: Number of days of history to include
        initial_equity: Starting equity value (default 0 for pure PnL tracking)
    
    Returns:
        The equity series dict that was written
    """
    records = _recent_executed(window_days)
    series = _compute_equity_series(records, initial_equity)
    
    # Add rolling returns
    series["rolling_pnl"] = _compute_rolling_returns(series["pnl"], window=20)
    
    # Add metadata
    series["ts"] = time.time()
    series["window_days"] = window_days
    series["initial_equity"] = initial_equity
    series["record_count"] = len(series["timestamps"])
    
    # Compute summary stats
    if series["pnl"]:
        pnl_arr = np.array(series["pnl"], dtype=float)
        series["total_pnl"] = float(pnl_arr.sum())
        series["mean_pnl"] = float(pnl_arr.mean())
        series["std_pnl"] = float(pnl_arr.std(ddof=1)) if len(pnl_arr) > 1 else 0.0
        series["max_drawdown"] = float(max(series["drawdown"])) if series["drawdown"] else 0.0
        series["win_rate"] = float((pnl_arr > 0).sum() / len(pnl_arr)) if len(pnl_arr) > 0 else 0.0
    else:
        series["total_pnl"] = 0.0
        series["mean_pnl"] = 0.0
        series["std_pnl"] = 0.0
        series["max_drawdown"] = 0.0
        series["win_rate"] = 0.0
    
    # Ensure state directory exists
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Write to file
    try:
        with EQUITY_PATH.open("w", encoding="utf-8") as handle:
            json.dump(series, handle, indent=2)
    except Exception:
        pass  # Fail silently - don't break execution
    
    return series


def load_equity_series() -> Dict[str, Any]:
    """
    Load equity series from logs/state/equity.json.
    
    Returns empty dict if file doesn't exist or is invalid.
    """
    try:
        if not EQUITY_PATH.exists():
            return {}
        with EQUITY_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def build_pnl_attribution_snapshot(
    trades: Iterable[Dict[str, Any]],
    positions: Iterable[Dict[str, Any]],
    now_ts: float | None = None,
    *,
    regimes: Mapping[str, Any] | None = None,
    risk_snapshot: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    now = float(now_ts or time.time())
    symbol_agg: Dict[str, Dict[str, Any]] = {}
    strategy_agg: Dict[str, Dict[str, Any]] = {}
    total_realized = 0.0
    total_unrealized = 0.0
    record_count = 0
    win_count = 0

    def _ensure_symbol(sym: str) -> Dict[str, Any]:
        if sym not in symbol_agg:
            symbol_agg[sym] = {"realized_pnl": 0.0, "unrealized_pnl": 0.0, "trade_count": 0}
        return symbol_agg[sym]

    def _ensure_strategy(name: str) -> Dict[str, Any]:
        if name not in strategy_agg:
            strategy_agg[name] = {"realized_pnl": 0.0, "unrealized_pnl": 0.0, "trade_count": 0}
        return strategy_agg[name]

    regimes_data = regimes if regimes is not None else _load_state_json(REGIMES_PATH)
    risk_data = risk_snapshot if risk_snapshot is not None else _load_state_json(RISK_SNAPSHOT_PATH)
    atr_idx = _regime_key(regimes_data.get("atr_regime"))
    dd_idx = _regime_key(regimes_data.get("dd_regime"))

    per_regime_atr: Dict[str, Dict[str, Any]] = (
        {str(i): _init_bucket() for i in range(4)} if atr_idx is not None else {}
    )
    per_regime_dd: Dict[str, Dict[str, Any]] = (
        {str(i): _init_bucket() for i in range(4)} if dd_idx is not None else {}
    )
    risk_mode_value = str(risk_data.get("risk_mode") or "OK").upper()
    per_risk_mode = {mode: _init_bucket() for mode in ("OK", "WARN", "DEFENSIVE", "HALTED")}
    per_day: Dict[str, Dict[str, Any]] = {}
    exits_state: Dict[str, Any] = {}
    factors_state: Dict[str, Any] = {}

    for trade in trades or []:
        sym = str(trade.get("symbol") or "").upper()
        if not sym:
            continue
        metadata = trade.get("metadata") if isinstance(trade.get("metadata"), Mapping) else {}
        exit_meta = metadata.get("exit") if isinstance(metadata, Mapping) and isinstance(metadata.get("exit"), Mapping) else {}
        strat = str(trade.get("strategy") or "unknown").lower()
        realized = _realized_pnl_value(trade) or 0.0
        total_realized += realized
        record_count += 1
        if realized > 0:
            win_count += 1
        sym_entry = _ensure_symbol(sym)
        strat_entry = _ensure_strategy(strat)
        sym_entry["realized_pnl"] += realized
        sym_entry["trade_count"] += 1
        strat_entry["realized_pnl"] += realized
        strat_entry["trade_count"] += 1
        if atr_idx is not None:
            bucket = per_regime_atr.get(str(atr_idx))
            bucket["realized"] += realized
            bucket["trade_count"] += 1
        if dd_idx is not None:
            bucket = per_regime_dd.get(str(dd_idx))
            bucket["realized"] += realized
            bucket["trade_count"] += 1
        bucket = per_risk_mode.get(risk_mode_value)
        bucket["realized"] += realized
        bucket["trade_count"] += 1
        ts = _to_float(trade.get("ts"))
        day_key = None
        if ts is not None:
            day_key = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
            day_bucket = per_day.setdefault(day_key, _init_bucket())
            day_bucket["realized"] += realized
            day_bucket["trade_count"] += 1
        reason_val = None
        if isinstance(exit_meta, Mapping):
            reason_val = exit_meta.get("reason")
        if reason_val is None and isinstance(metadata, Mapping):
            reason_val = metadata.get("exit_reason") or metadata.get("exitReason")
        reason = str(reason_val).lower() if isinstance(reason_val, str) else None
        if reason in {"tp", "sl"}:
            rr_val = None
            vt_meta = metadata.get("vol_target") if isinstance(metadata, Mapping) else None
            if isinstance(vt_meta, Mapping):
                tp_sl_meta = vt_meta.get("tp_sl")
                if isinstance(tp_sl_meta, Mapping):
                    rr_val = tp_sl_meta.get("reward_risk")
            if rr_val is None and isinstance(exit_meta, Mapping):
                rr_val = exit_meta.get("reward_risk")
            rr = _to_float(rr_val) if rr_val is not None else None
            strat_value = exit_meta.get("source_strategy") if isinstance(exit_meta, Mapping) else None
            if not strat_value and isinstance(metadata, Mapping):
                strat_value = metadata.get("strategy")
            if not strat_value:
                strat_value = trade.get("strategy")
            strategy_name = str(strat_value).lower() if strat_value else None
            _update_exit_buckets(
                exits_state,
                reason=reason,
                strategy=strategy_name,
                symbol=sym,
                realized_pnl=realized,
                rr=rr,
                atr_regime=int(atr_idx) if atr_idx is not None else None,
                dd_regime=int(dd_idx) if dd_idx is not None else None,
                risk_mode=risk_mode_value,
                day_key=day_key,
            )
        # Hybrid factor attribution (vol_target only)
        vt_meta = metadata.get("vol_target") if isinstance(metadata, Mapping) and str(metadata.get("strategy") or "").lower() == "vol_target" else {}
        if isinstance(vt_meta, Mapping) and (vt_meta.get("hybrid") or vt_meta.get("trend") or vt_meta.get("carry")):
            hybrid = vt_meta.get("hybrid") if isinstance(vt_meta.get("hybrid"), Mapping) else {}
            trend_meta = vt_meta.get("trend") if isinstance(vt_meta.get("trend"), Mapping) else {}
            carry_meta = vt_meta.get("carry") if isinstance(vt_meta.get("carry"), Mapping) else {}
            components = hybrid.get("components") if isinstance(hybrid, Mapping) and isinstance(hybrid.get("components"), Mapping) else {}
            hybrid_score = hybrid.get("hybrid_score")
            trend_strength = components.get("trend_strength") if isinstance(components, Mapping) else None
            if trend_strength is None and isinstance(trend_meta, Mapping):
                trend_strength = trend_meta.get("strength")
            carry_long = components.get("carry_long") if isinstance(components, Mapping) else None
            carry_short = components.get("carry_short") if isinstance(components, Mapping) else None
            if carry_long is None and isinstance(carry_meta, Mapping):
                carry_long = carry_meta.get("score_long")
            if carry_short is None and isinstance(carry_meta, Mapping):
                carry_short = carry_meta.get("score_short")
            _update_factor_buckets(
                factors_state,
                hybrid_decile=_bucket_hybrid_score(hybrid_score),
                trend_bucket=_bucket_trend_strength(trend_strength),
                carry_regime=_bucket_carry_regime(carry_long=carry_long, carry_short=carry_short),
                realized_pnl=realized,
            )

    for pos in positions or []:
        sym = str(pos.get("symbol") or "").upper()
        if not sym:
            continue
        unrealized = _to_float(pos.get("unrealized_pnl")) or 0.0
        total_unrealized += unrealized
        sym_entry = _ensure_symbol(sym)
        sym_entry["unrealized_pnl"] += unrealized
        if atr_idx is not None:
            bucket = per_regime_atr.get(str(atr_idx))
            bucket["unrealized"] += unrealized
        if dd_idx is not None:
            bucket = per_regime_dd.get(str(dd_idx))
            bucket["unrealized"] += unrealized
        bucket = per_risk_mode.get(risk_mode_value)
        bucket["unrealized"] += unrealized
        today_key = datetime.utcfromtimestamp(now).strftime("%Y-%m-%d")
        today_bucket = per_day.setdefault(today_key, _init_bucket())
        today_bucket["unrealized"] += unrealized

    for entry in symbol_agg.values():
        entry["total_pnl"] = entry.get("realized_pnl", 0.0) + entry.get("unrealized_pnl", 0.0)

    for entry in strategy_agg.values():
        entry["total_pnl"] = entry.get("realized_pnl", 0.0) + entry.get("unrealized_pnl", 0.0)

    for bucket in per_regime_atr.values():
        bucket["total"] = bucket["realized"] + bucket["unrealized"]
    for bucket in per_regime_dd.values():
        bucket["total"] = bucket["realized"] + bucket["unrealized"]
    for bucket in per_risk_mode.values():
        bucket["total"] = bucket["realized"] + bucket["unrealized"]
    for bucket in per_day.values():
        bucket["total"] = bucket["realized"] + bucket["unrealized"]

    summary = {
        "total_realized": total_realized,
        "total_unrealized": total_unrealized,
        "total_pnl": total_realized + total_unrealized,
        "win_rate": float(win_count) / record_count if record_count else 0.0,
        "record_count": record_count,
        "ts": now,
    }

    snapshot = {
        "per_symbol": {sym: dict(vals) for sym, vals in symbol_agg.items()},
        "per_strategy": {name: dict(vals) for name, vals in strategy_agg.items()},
        "summary": summary,
    }
    if per_regime_atr or per_regime_dd:
        snapshot["per_regime"] = {"atr": per_regime_atr, "dd": per_regime_dd}
    if per_risk_mode:
        snapshot["per_risk_mode"] = per_risk_mode
    if per_day:
        snapshot["per_day"] = per_day
    if exits_state:
        _finalize_exits_state(exits_state)
        snapshot["exits"] = exits_state
    if factors_state:
        _finalize_factor_buckets(factors_state)
        regimes_block = snapshot.setdefault("regimes", {})
        regimes_block["factors"] = factors_state
    return snapshot


def export_pnl_attribution_state(
    *,
    trades_path: Path | str | None = None,
    positions_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> Dict[str, Any]:
    trades_target = Path(trades_path) if trades_path else TRADES_LOG_PATH
    positions_target = Path(positions_path) if positions_path else POSITIONS_STATE_PATH
    output_target = Path(output_path) if output_path else PNL_ATTRIBUTION_PATH

    trades = _load_trades(trades_target)
    positions = _load_positions(positions_target)
    snapshot = build_pnl_attribution_snapshot(trades, positions)

    output_target.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_target.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2)
    tmp.replace(output_target)
    return snapshot
