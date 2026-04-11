"""
Futures S2 Proxy — minimal parallel sleeve for transfer-of-edge testing.

Tests ONE question:

    Does PM calibration mispricing predict short-horizon futures returns?

Signal:  extract_s2_signal() → edge = p_model − p_market
Rule:    edge > +3pp → LONG, edge < −3pp → SHORT, else NO_TRADE
Horizon: Fixed 15-min hold (entry_ts + 900s), no thesis/regime/doctrine exits
Risk:    DD kill switch, stale-quote denial.  No Sentinel-X, no Hydra.

All events tagged ``execution_mode: FUTURES_S2_PROXY``.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROUND_DURATION_S: int = 900
ENTRY_OFFSET_S: int = 30
ENTRY_WINDOW_S: int = 120  # 2-min entry window (tighter than PM's 10 min)
EXECUTION_MODE: str = "FUTURES_S2_PROXY"
TAKER_FEE_RATE_DEFAULT: float = 0.0004

TRADE_LOG_PATH = Path("logs/execution/futures_s2_proxy_trades.jsonl")
STATE_PATH = Path("logs/state/futures_s2_proxy_state.json")

# ---------------------------------------------------------------------------
# Price region classifier (reused from S2)
# ---------------------------------------------------------------------------
def _price_region(p_market: float) -> str:
    if p_market < 0.15:
        return "extreme_low"
    if p_market < 0.30:
        return "low"
    if p_market < 0.45:
        return "mid_low"
    if p_market < 0.55:
        return "center"
    if p_market < 0.70:
        return "mid_high"
    if p_market < 0.85:
        return "high"
    return "extreme_high"


# ---------------------------------------------------------------------------
# Round boundary helpers
# ---------------------------------------------------------------------------
def _round_start_unix(ts_unix: float) -> float:
    return ts_unix - (ts_unix % ROUND_DURATION_S)


def _make_round_id(round_start_unix: float) -> str:
    dt = datetime.fromtimestamp(round_start_unix, tz=timezone.utc)
    return f"FS2_{dt.strftime('%Y%m%d_%H%M')}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ts_to_unix(ts_iso: str) -> float:
    try:
        return datetime.fromisoformat(ts_iso).timestamp()
    except Exception:
        return time.time()


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class OpenFuturesTrade:
    """An in-flight futures position awaiting fixed-horizon exit."""
    round_id: str
    symbol: str
    side: str           # "LONG" or "SHORT"
    entry_price: float
    entry_ts: str
    entry_ts_unix: float
    exit_ts_unix: float  # entry_ts_unix + hold_duration_s
    qty: float
    notional_usd: float
    order_id: str
    signal_snapshot: Dict[str, Any]
    # Entry timing decomposition (unix timestamps)
    round_open_ts: float = 0.0
    entry_signal_ready_ts: float = 0.0
    entry_order_submitted_ts: float = 0.0
    entry_fill_ts: float = 0.0


@dataclass
class FuturesTradeOutcome:
    """Resolved outcome for a futures S2 proxy trade."""
    round_id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    qty: float
    notional_usd: float
    hold_duration_s: float
    gross_pnl: float
    fee_usd: float
    net_pnl: float
    log_return: float
    entry_ts: str
    exit_ts: str
    exit_delay_s: float
    entry_order_id: str
    exit_order_id: str
    signal_snapshot: Dict[str, Any]
    # Exit timing decomposition (unix timestamps)
    hold_expiry_ts: float = 0.0
    exit_decision_ts: float = 0.0
    exit_order_submitted_ts: float = 0.0
    exit_fill_confirmed_ts: float = 0.0
    # Entry timing carried forward for full decomposition
    entry_signal_ready_ts: float = 0.0
    entry_order_submitted_ts: float = 0.0
    entry_fill_ts: float = 0.0


# ---------------------------------------------------------------------------
# Trade log writer (append-only)
# ---------------------------------------------------------------------------
class _TradeWriter:
    def __init__(self, path: Path = TRADE_LOG_PATH):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: Dict[str, Any]) -> None:
        try:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        except Exception as exc:
            logger.warning("futures_s2_proxy: trade write failed: %s", exc)


# ---------------------------------------------------------------------------
# Daemon thread for dedicated proxy cadence
# ---------------------------------------------------------------------------
class _ProxyTickThread(threading.Thread):
    """Daemon thread that ticks the proxy runner at a dedicated cadence."""

    def __init__(self, runner: "FuturesS2ProxyRunner", interval: float = 5.0) -> None:
        super().__init__(daemon=True, name="futures-s2-proxy-tick")
        self._runner = runner
        self._interval = interval
        self._stop_event = threading.Event()

    def run(self) -> None:
        logger.info("futures_s2_proxy: daemon started (interval=%.1fs)", self._interval)
        while not self._stop_event.is_set():
            try:
                now_iso = datetime.now(timezone.utc).isoformat()
                self._runner.tick(now_iso)
            except Exception as exc:
                logger.warning("futures_s2_proxy: daemon tick failed: %s", exc)
            self._stop_event.wait(self._interval)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        self.join(timeout=timeout)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
class FuturesS2ProxyRunner:
    """
    Minimal futures sleeve — tests PM-edge transfer to futures returns.

    Called once per executor tick via ``tick(now_ts)``.
    Can also run on a dedicated daemon thread via ``start_daemon()``.
    """

    def __init__(
        self,
        *,
        limits: Dict[str, Any],
        model: Any,
        trade_log_path: Path = TRADE_LOG_PATH,
        state_path: Optional[Path] = None,
    ) -> None:
        self._limits = limits
        self._model = model
        self._writer = _TradeWriter(trade_log_path)
        self._state_path = state_path or STATE_PATH

        # Config
        self._symbols: List[str] = limits.get("symbols", ["BTCUSDT"])
        self._notional_usd: float = float(limits.get("notional_usd", 200))
        self._hold_s: int = int(limits.get("hold_duration_s", ROUND_DURATION_S))
        self._max_concurrent: int = int(limits.get("max_concurrent_per_symbol", 1))
        self._min_edge: float = float(limits.get("min_edge", 0.03))
        self._taker_fee: float = float(limits.get("taker_fee_rate", TAKER_FEE_RATE_DEFAULT))
        self._max_quote_age: float = float(limits.get("max_quote_age_s", 90))
        self._dd_kill_usd: float = float(limits.get("dd_kill_usd", -500))

        # Variant mode: "normal", "invert", "short_only", "threshold"
        self._variant_mode: str = str(limits.get("variant_mode", "normal"))
        self._variant_min_edge_abs: float = float(limits.get("variant_min_edge_abs", 0.05))

        # State
        self._open_trades: List[OpenFuturesTrade] = []
        self._processed_round_ids: set[str] = set()
        self._cumulative_pnl: float = 0.0
        self._total_entries: int = 0
        self._total_exits: int = 0
        self._sum_log_return: float = 0.0
        self._win_count: int = 0
        self._dd_kill_active: bool = False

        # Heartbeat counters (session-level, reset on restart)
        self._tick_count: int = 0
        self._entries_attempted: int = 0
        self._denials: Dict[str, int] = {}

        # Thread safety
        self._lock = threading.Lock()
        self._daemon: Optional[_ProxyTickThread] = None

        # Restore state from disk
        self._load_state()

        logger.info(
            "futures_s2_proxy: init variant_mode=%s min_edge=%.4f variant_min_edge_abs=%.4f",
            self._variant_mode, self._min_edge, self._variant_min_edge_abs,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tick(self, now_ts: str) -> bool:
        with self._lock:
            now_unix = _ts_to_unix(now_ts)
            acted = False
            self._tick_count += 1

            # Exit first (free slots)
            if self._maybe_exit_all(now_unix, now_ts):
                acted = True

            # Then enter (only if slots available and DD kill not active)
            if not self._dd_kill_active:
                for symbol in self._symbols:
                    if self._maybe_enter(symbol, now_unix, now_ts):
                        acted = True

            # Heartbeat: persist state every tick so dashboard shows liveness
            self._write_state()

            return acted

    def start_daemon(self, interval: float = 5.0) -> None:
        """Start a dedicated daemon thread that ticks at *interval* seconds."""
        if self._daemon is not None and self._daemon.is_alive():
            return  # already running
        self._daemon = _ProxyTickThread(self, interval=interval)
        self._daemon.start()

    def daemon_alive(self) -> bool:
        """Return True if the daemon thread is running."""
        return self._daemon is not None and self._daemon.is_alive()

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    def _maybe_enter(self, symbol: str, now_unix: float, now_ts: str) -> bool:
        # Round boundary check
        round_start = _round_start_unix(now_unix)
        round_id = _make_round_id(round_start)

        if round_id in self._processed_round_ids:
            return False

        # Entry window: first 2 minutes after offset
        window_start = round_start + ENTRY_OFFSET_S
        window_end = window_start + ENTRY_WINDOW_S
        if not (window_start <= now_unix <= window_end):
            return False

        # Mark processed (even if we skip — prevents re-evaluation)
        self._processed_round_ids.add(round_id)
        self._prune_processed_ids(now_unix)

        # Concurrent position check
        open_for_symbol = sum(1 for t in self._open_trades if t.symbol == symbol)
        if open_for_symbol >= self._max_concurrent:
            self._log_no_trade(round_id, symbol, now_ts, "max_concurrent_reached")
            return False

        # DD kill check
        if self._cumulative_pnl <= self._dd_kill_usd:
            self._dd_kill_active = True
            self._log_no_trade(round_id, symbol, now_ts, "DD_KILL")
            logger.warning(
                "futures_s2_proxy: DD KILL active — cumulative_pnl=%.2f <= %.2f",
                self._cumulative_pnl, self._dd_kill_usd,
            )
            return False

        # Extract S2 signal
        from execution.binary_lab_s2_signals import extract_s2_signal

        signal = extract_s2_signal(
            self._model,
            per_round_usd=self._notional_usd,
            min_edge_threshold=self._min_edge,
        )
        _entry_signal_ready_ts = time.time()
        if signal is None:
            self._log_no_trade(round_id, symbol, now_ts, "signal_unavailable")
            return False

        # Stale quote check
        if signal.quote_age_s > self._max_quote_age:
            self._log_no_trade(round_id, symbol, now_ts, "stale_quote",
                               signal=signal)
            return False

        # Edge → direction (pure translation, no regime/conviction)
        edge = signal.edge_yes
        if edge >= self._min_edge:
            direction = "LONG"
        elif edge <= -self._min_edge:
            direction = "SHORT"
        else:
            self._log_no_trade(round_id, symbol, now_ts, "edge_below_threshold",
                               signal=signal)
            return False

        # ---- Variant mode transforms ----
        if self._variant_mode == "invert":
            direction = "SHORT" if direction == "LONG" else "LONG"
        elif self._variant_mode == "short_only":
            if direction == "LONG":
                self._log_no_trade(round_id, symbol, now_ts,
                                   "variant_long_disabled", signal=signal)
                return False
        elif self._variant_mode == "threshold":
            if abs(edge) < self._variant_min_edge_abs:
                self._log_no_trade(round_id, symbol, now_ts,
                                   "variant_edge_below_abs_threshold",
                                   signal=signal)
                return False
        elif self._variant_mode == "invert_threshold":
            direction = "SHORT" if direction == "LONG" else "LONG"
            if abs(edge) < self._variant_min_edge_abs:
                self._log_no_trade(round_id, symbol, now_ts,
                                   "variant_edge_below_abs_threshold",
                                   signal=signal)
                return False

        # Freeze signal snapshot for attribution
        snapshot: Dict[str, Any] = {
            "p_market": signal.p_yes_mid,
            "p_model": signal.p_model_yes,
            "edge": round(edge, 6),
            "quote_source": signal.quote_reconstruction_mode,
            "price_region": _price_region(signal.p_yes_mid),
            "spread": signal.spread,
            "calibration_confident": signal.calibration_confident,
            "quote_age_s": signal.quote_age_s,
            "round_id": round_id,
            "symbol": symbol,
            "variant_mode": self._variant_mode,
        }

        # Get current price and compute qty
        try:
            from execution.exchange_utils import get_price
            price = get_price(symbol)
        except Exception:
            price = None

        if price is None or price <= 0:
            self._log_no_trade(round_id, symbol, now_ts, "price_unavailable",
                               signal=signal)
            return False

        raw_qty = self._notional_usd / price

        try:
            from execution.exchange_precision import normalize_qty
            qty = normalize_qty(symbol, raw_qty)
        except Exception:
            qty = raw_qty

        if qty <= 0:
            self._log_no_trade(round_id, symbol, now_ts, "qty_zero_after_normalize",
                               signal=signal)
            return False

        # Place MARKET order on testnet
        self._entries_attempted += 1
        order_side = "BUY" if direction == "LONG" else "SELL"
        _entry_order_submitted_ts = time.time()
        try:
            from execution.exchange_utils import send_order
            resp = send_order(
                symbol=symbol,
                side=order_side,
                type="MARKET",
                quantity=qty,
                positionSide=direction,
            )
            _entry_fill_ts = time.time()
        except Exception as exc:
            logger.warning("futures_s2_proxy: send_order failed: %s", exc)
            self._log_no_trade(round_id, symbol, now_ts,
                               f"order_failed: {exc}", signal=signal)
            return False

        # Parse response
        order_id = str(resp.get("orderId", ""))
        status = resp.get("status", "")

        # Fix: float-first-then-fallback.  Raw string "0.00000000" is
        # truthy so ``float(x or fallback)`` skips the fallback.
        _raw_fill_price = float(resp.get("avgPrice", 0) or 0)
        fill_price = _raw_fill_price if _raw_fill_price > 0 else price
        _raw_fill_qty = float(resp.get("executedQty", 0) or 0)
        fill_qty = _raw_fill_qty if _raw_fill_qty > 0 else qty

        logger.info(
            "futures_s2_proxy: order resp %s %s — status=%s "
            "executedQty=%s avgPrice=%s orderId=%s raw_qty=%.6f price=%.2f",
            symbol, order_side, status,
            resp.get("executedQty"), resp.get("avgPrice"), order_id,
            raw_qty, price,
        )

        if _raw_fill_qty <= 0 and status != "DRY_RUN":
            logger.info(
                "futures_s2_proxy: exchange returned zero fill for %s — "
                "using requested qty=%.6f as paper fill "
                "(testnet no-liquidity fallback)",
                symbol, qty,
            )

        if fill_qty <= 0 and status != "DRY_RUN":
            self._log_no_trade(round_id, symbol, now_ts, "fill_qty_zero",
                               signal=signal)
            return False

        actual_notional = fill_price * fill_qty

        # Build OpenFuturesTrade
        trade = OpenFuturesTrade(
            round_id=round_id,
            symbol=symbol,
            side=direction,
            entry_price=fill_price,
            entry_ts=now_ts,
            entry_ts_unix=now_unix,
            exit_ts_unix=now_unix + self._hold_s,
            qty=fill_qty,
            notional_usd=round(actual_notional, 4),
            order_id=order_id,
            signal_snapshot=snapshot,
            round_open_ts=round_start,
            entry_signal_ready_ts=_entry_signal_ready_ts,
            entry_order_submitted_ts=_entry_order_submitted_ts,
            entry_fill_ts=_entry_fill_ts,
        )
        self._open_trades.append(trade)
        self._total_entries += 1

        # Entry timing diagnostics
        entry_window_position_s = round(now_unix - round_start, 1)

        # Log ENTRY
        self._writer.write({
            "event_type": "ENTRY",
            "execution_mode": EXECUTION_MODE,
            "ts": now_ts,
            "ts_unix": round(now_unix, 3),
            "round_id": round_id,
            "symbol": symbol,
            "side": direction,
            "order_side": order_side,
            "entry_price": fill_price,
            "qty": fill_qty,
            "notional_usd": trade.notional_usd,
            "order_id": order_id,
            "order_status": status,
            "signal_snapshot": snapshot,
            "hold_duration_s": self._hold_s,
            "variant_mode": self._variant_mode,
            "entry_window_position_s": entry_window_position_s,
            "round_open_ts": round(round_start, 3),
            "entry_signal_ready_ts": round(_entry_signal_ready_ts, 3),
            "entry_order_submitted_ts": round(_entry_order_submitted_ts, 3),
            "entry_fill_ts": round(_entry_fill_ts, 3),
            "cumulative_pnl": round(self._cumulative_pnl, 4),
            "total_entries": self._total_entries,
        })

        # Persist state
        self._write_state()

        logger.info(
            "futures_s2_proxy: ENTRY %s %s %s @ %.2f qty=%.4f edge=%.4f region=%s",
            round_id, symbol, direction, fill_price, fill_qty,
            edge, snapshot["price_region"],
        )
        return True

    # ------------------------------------------------------------------
    # Exit (fixed-horizon only)
    # ------------------------------------------------------------------

    def _maybe_exit_all(self, now_unix: float, now_ts: str) -> bool:
        exited_any = False
        still_open: List[OpenFuturesTrade] = []

        for trade in self._open_trades:
            if now_unix >= trade.exit_ts_unix:
                outcome = self._close_trade(trade, now_ts, now_unix)
                if outcome is not None:
                    self._emit_round_closed(outcome)
                    exited_any = True
                else:
                    # Close failed — keep for retry next tick
                    still_open.append(trade)
            else:
                still_open.append(trade)

        self._open_trades = still_open
        return exited_any

    def _close_trade(
        self, trade: OpenFuturesTrade, now_ts: str, now_unix: float,
    ) -> Optional[FuturesTradeOutcome]:
        """Close position via MARKET reduceOnly order."""
        _exit_decision_ts = time.time()
        exit_side = "SELL" if trade.side == "LONG" else "BUY"
        exit_order_id = ""
        _exit_order_submitted_ts = time.time()

        try:
            from execution.exchange_utils import send_order
            resp = send_order(
                symbol=trade.symbol,
                side=exit_side,
                type="MARKET",
                quantity=trade.qty,
                positionSide=trade.side,
                reduceOnly=True,
            )
            _exit_fill_confirmed_ts = time.time()
            exit_order_id = str(resp.get("orderId", ""))
        except Exception as exc:
            _exit_fill_confirmed_ts = time.time()
            logger.warning(
                "futures_s2_proxy: close_trade send_order failed %s: %s "
                "— using current price as paper exit (testnet fallback)",
                trade.round_id, exc,
            )
            resp = {}

        _raw_exit_price = float(resp.get("avgPrice", 0) or 0)
        exit_price = _raw_exit_price

        # Fallback: if avgPrice is 0 (dry run / testnet no-fill), get current price
        if exit_price <= 0:
            try:
                from execution.exchange_utils import get_price
                exit_price = get_price(trade.symbol)
            except Exception:
                exit_price = trade.entry_price  # worst case: flat

        # PnL computation
        direction_sign = 1.0 if trade.side == "LONG" else -1.0
        gross_pnl = (exit_price - trade.entry_price) * trade.qty * direction_sign
        fee_usd = (trade.entry_price * trade.qty + exit_price * trade.qty) * self._taker_fee
        net_pnl = gross_pnl - fee_usd

        # Log return: ln(exit/entry) * direction_sign - round_trip_fee
        if trade.entry_price > 0 and exit_price > 0:
            log_ret = math.log(exit_price / trade.entry_price) * direction_sign - 2 * self._taker_fee
        else:
            log_ret = 0.0

        hold_duration = now_unix - trade.entry_ts_unix
        exit_delay = now_unix - trade.exit_ts_unix  # >0 means late

        if exit_delay > 30:
            logger.warning(
                "futures_s2_proxy: EXIT DELAY %s — %.1fs late (target=%ds actual=%.1fs)",
                trade.round_id, exit_delay, self._hold_s, hold_duration,
            )

        return FuturesTradeOutcome(
            round_id=trade.round_id,
            symbol=trade.symbol,
            side=trade.side,
            entry_price=trade.entry_price,
            exit_price=exit_price,
            qty=trade.qty,
            notional_usd=trade.notional_usd,
            hold_duration_s=round(hold_duration, 1),
            gross_pnl=round(gross_pnl, 6),
            fee_usd=round(fee_usd, 6),
            net_pnl=round(net_pnl, 6),
            log_return=round(log_ret, 8),
            exit_delay_s=round(exit_delay, 1),
            entry_ts=trade.entry_ts,
            exit_ts=now_ts,
            entry_order_id=trade.order_id,
            exit_order_id=exit_order_id,
            signal_snapshot=trade.signal_snapshot,
            hold_expiry_ts=trade.exit_ts_unix,
            exit_decision_ts=round(_exit_decision_ts, 3),
            exit_order_submitted_ts=round(_exit_order_submitted_ts, 3),
            exit_fill_confirmed_ts=round(_exit_fill_confirmed_ts, 3),
            entry_signal_ready_ts=trade.entry_signal_ready_ts,
            entry_order_submitted_ts=trade.entry_order_submitted_ts,
            entry_fill_ts=trade.entry_fill_ts,
        )

    # ------------------------------------------------------------------
    # Log emission
    # ------------------------------------------------------------------

    def _emit_round_closed(self, outcome: FuturesTradeOutcome) -> None:
        self._cumulative_pnl += outcome.net_pnl
        self._total_exits += 1
        self._sum_log_return += outcome.log_return
        if outcome.net_pnl > 0:
            self._win_count += 1

        self._writer.write({
            "event_type": "ROUND_CLOSED",
            "execution_mode": EXECUTION_MODE,
            "ts": outcome.exit_ts,
            "ts_unix": round(_ts_to_unix(outcome.exit_ts), 3),
            "round_id": outcome.round_id,
            "symbol": outcome.symbol,
            "side": outcome.side,
            "entry_price": outcome.entry_price,
            "exit_price": outcome.exit_price,
            "qty": outcome.qty,
            "notional_usd": outcome.notional_usd,
            "hold_duration_s": outcome.hold_duration_s,
            "exit_delay_s": outcome.exit_delay_s,
            "gross_pnl": outcome.gross_pnl,
            "fee_usd": outcome.fee_usd,
            "net_pnl": outcome.net_pnl,
            "log_return": outcome.log_return,
            "entry_order_id": outcome.entry_order_id,
            "exit_order_id": outcome.exit_order_id,
            "signal_snapshot": outcome.signal_snapshot,
            "hold_expiry_ts": outcome.hold_expiry_ts,
            "exit_decision_ts": outcome.exit_decision_ts,
            "exit_order_submitted_ts": outcome.exit_order_submitted_ts,
            "exit_fill_confirmed_ts": outcome.exit_fill_confirmed_ts,
            "entry_signal_ready_ts": outcome.entry_signal_ready_ts,
            "entry_order_submitted_ts": outcome.entry_order_submitted_ts,
            "entry_fill_ts": outcome.entry_fill_ts,
            "cumulative_pnl": round(self._cumulative_pnl, 4),
            "total_exits": self._total_exits,
            "variant_mode": self._variant_mode,
            "realized_win_rate": round(
                self._win_count / self._total_exits, 4,
            ) if self._total_exits > 0 else 0.0,
            "mean_log_return": round(
                self._sum_log_return / self._total_exits, 8,
            ) if self._total_exits > 0 else 0.0,
        })

        # Persist state after every resolution
        self._write_state()

        logger.info(
            "futures_s2_proxy: CLOSED %s %s %s pnl=%.4f cum=%.2f log_ret=%.6f exit_delay=%.1fs",
            outcome.round_id, outcome.symbol, outcome.side,
            outcome.net_pnl, self._cumulative_pnl, outcome.log_return,
            outcome.exit_delay_s,
        )

    def _log_no_trade(
        self,
        round_id: str,
        symbol: str,
        now_ts: str,
        reason: str,
        *,
        signal: Any = None,
    ) -> None:
        self._denials[reason] = self._denials.get(reason, 0) + 1
        record: Dict[str, Any] = {
            "event_type": "NO_TRADE",
            "execution_mode": EXECUTION_MODE,
            "ts": now_ts,
            "round_id": round_id,
            "symbol": symbol,
            "deny_reason": reason,
            "cumulative_pnl": round(self._cumulative_pnl, 4),
            "dd_kill_active": self._dd_kill_active,
            "variant_mode": self._variant_mode,
        }
        if signal is not None:
            record["signal_snapshot"] = {
                "p_market": signal.p_yes_mid,
                "p_model": signal.p_model_yes,
                "edge": round(signal.edge_yes, 6),
                "quote_source": signal.quote_reconstruction_mode,
                "price_region": _price_region(signal.p_yes_mid),
                "spread": signal.spread,
                "calibration_confident": signal.calibration_confident,
                "quote_age_s": signal.quote_age_s,
            }
        self._writer.write(record)

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _write_state(self) -> None:
        state = {
            "cumulative_pnl": round(self._cumulative_pnl, 4),
            "open_trades": [
                {
                    "round_id": t.round_id,
                    "symbol": t.symbol,
                    "side": t.side,
                    "entry_price": t.entry_price,
                    "entry_ts": t.entry_ts,
                    "entry_ts_unix": t.entry_ts_unix,
                    "exit_ts_unix": t.exit_ts_unix,
                    "qty": t.qty,
                    "notional_usd": t.notional_usd,
                    "order_id": t.order_id,
                    "signal_snapshot": t.signal_snapshot,
                    "round_open_ts": t.round_open_ts,
                    "entry_signal_ready_ts": t.entry_signal_ready_ts,
                    "entry_order_submitted_ts": t.entry_order_submitted_ts,
                    "entry_fill_ts": t.entry_fill_ts,
                }
                for t in self._open_trades
            ],
            "total_entries": self._total_entries,
            "total_exits": self._total_exits,
            "realized_win_rate": round(
                self._win_count / self._total_exits, 4,
            ) if self._total_exits > 0 else 0.0,
            "mean_log_return": round(
                self._sum_log_return / self._total_exits, 8,
            ) if self._total_exits > 0 else 0.0,
            "dd_kill_active": self._dd_kill_active,
            "updated_ts": _now_iso(),
            # Heartbeat fields (session-level, reset on restart)
            "tick_count": self._tick_count,
            "entries_attempted": self._entries_attempted,
            "denials_by_reason": dict(self._denials),
        }
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, default=str)
            tmp.replace(self._state_path)
        except Exception as exc:
            logger.warning("futures_s2_proxy: state write failed: %s", exc)

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            with self._state_path.open("r", encoding="utf-8") as f:
                state = json.load(f)
            self._cumulative_pnl = float(state.get("cumulative_pnl", 0.0))
            self._total_entries = int(state.get("total_entries", 0))
            self._total_exits = int(state.get("total_exits", 0))
            self._dd_kill_active = bool(state.get("dd_kill_active", False))

            # Restore running stats
            win_rate = float(state.get("realized_win_rate", 0.0))
            self._win_count = round(win_rate * self._total_exits) if self._total_exits > 0 else 0
            self._sum_log_return = float(state.get("mean_log_return", 0.0)) * self._total_exits

            # Restore open trades
            for raw in state.get("open_trades", []):
                self._open_trades.append(OpenFuturesTrade(
                    round_id=raw["round_id"],
                    symbol=raw["symbol"],
                    side=raw["side"],
                    entry_price=float(raw["entry_price"]),
                    entry_ts=raw["entry_ts"],
                    entry_ts_unix=float(raw["entry_ts_unix"]),
                    exit_ts_unix=float(raw["exit_ts_unix"]),
                    qty=float(raw["qty"]),
                    notional_usd=float(raw["notional_usd"]),
                    order_id=raw["order_id"],
                    signal_snapshot=raw.get("signal_snapshot", {}),
                    round_open_ts=float(raw.get("round_open_ts", 0.0)),
                    entry_signal_ready_ts=float(raw.get("entry_signal_ready_ts", 0.0)),
                    entry_order_submitted_ts=float(raw.get("entry_order_submitted_ts", 0.0)),
                    entry_fill_ts=float(raw.get("entry_fill_ts", 0.0)),
                ))

            logger.info(
                "futures_s2_proxy: state restored — cum_pnl=%.2f entries=%d exits=%d open=%d dd_kill=%s",
                self._cumulative_pnl, self._total_entries, self._total_exits,
                len(self._open_trades), self._dd_kill_active,
            )
        except Exception as exc:
            logger.warning("futures_s2_proxy: state load failed: %s", exc)

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def _prune_processed_ids(self, now_unix: float) -> None:
        cutoff = now_unix - 7200
        self._processed_round_ids = {
            rid for rid in self._processed_round_ids
            if rid >= _make_round_id(cutoff)
        }
