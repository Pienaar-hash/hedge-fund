"""
Binary Lab S1 — shadow execution adapter + runner.

Architecture (from S1 Clean Forward Test Design):

    Signal → Eligibility Gate → SimulatedFill → StateReducer → TradeLog

Same state machine, same limits, same kill rules.
Only the execution adapter is swapped: real exchange API → deterministic
fill simulator.  Every event is tagged ``execution_mode: SHADOW`` so it
can never be confused with live.

Frozen parameters (immutable for the 30-day window):

    ENTRY_OFFSET_S      = 30      seconds after round start
    ENTRY_WINDOW_S      = 120     tolerance for executor tick jitter
    SLIPPAGE_BUFFER_BPS = 5.0     conservative entry/exit slippage
    FEE_BPS             = 4.0     Binance futures taker fee per side
    ROUND_DURATION_S    = 900     15 minutes
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frozen constants — locked for the 30-day window
# ---------------------------------------------------------------------------
ROUND_DURATION_S: int = 900           # 15 minutes
ENTRY_OFFSET_S: int = 30             # fixed offset from round start
ENTRY_WINDOW_S: int = 120            # tolerance for tick jitter
SLIPPAGE_BUFFER_BPS: float = 5.0     # conservative fill adjustment
FEE_BPS: float = 4.0                 # Binance futures taker fee per side
DEFAULT_SYMBOL: str = "BTCUSDT"

# ---------------------------------------------------------------------------
# Log paths
# ---------------------------------------------------------------------------
TRADE_LOG_PATH = Path("logs/execution/binary_lab_trades.jsonl")

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class SimulatedFill:
    """Result of a deterministic fill simulation."""
    fill_price: float
    mid_price: float
    best_bid: float
    best_ask: float
    slippage_bps: float
    ts: str                # ISO 8601

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fill_price": round(self.fill_price, 8),
            "mid_price": round(self.mid_price, 8),
            "best_bid": round(self.best_bid, 8),
            "best_ask": round(self.best_ask, 8),
            "slippage_bps": round(self.slippage_bps, 4),
            "ts": self.ts,
        }


@dataclass
class OpenRound:
    """An in-flight shadow round waiting for resolution."""
    round_id: str
    symbol: str
    direction: str              # "UP" | "DOWN"
    entry_fill: SimulatedFill
    notional_usd: float
    entry_ts: str               # ISO 8601
    entry_ts_unix: float
    resolution_ts_unix: float   # when to resolve
    conviction_band: str
    conviction_score: float
    regime: str
    regime_confidence: float


@dataclass
class RoundOutcome:
    """Resolved outcome for a shadow round."""
    round_id: str
    symbol: str
    direction: str
    outcome: str                # "WIN" | "LOSS"
    pnl_usd: float
    fee_usd: float
    gross_pnl_usd: float
    entry_price: float
    exit_price: float
    notional_usd: float
    conviction_band: str
    conviction_score: float
    regime: str
    regime_confidence: float
    entry_ts: str
    exit_ts: str


# ---------------------------------------------------------------------------
# Simulated execution adapter
# ---------------------------------------------------------------------------

class SimulatedExecutionAdapter:
    """
    Deterministic fill simulator.  No exchange API calls for order
    submission — only read-only price snapshots.

    Fill model (conservative):
        UP  entry: best_ask + slippage_buffer
        UP  exit:  best_bid - slippage_buffer
        DOWN entry: best_bid - slippage_buffer
        DOWN exit:  best_ask + slippage_buffer
    """

    def __init__(self, slippage_buffer_bps: float = SLIPPAGE_BUFFER_BPS):
        self._slippage_bps = slippage_buffer_bps

    def _fetch_book_snapshot(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Get best bid/ask from the exchange (read-only).

        Falls back to ``get_price()`` if orderbook is unavailable.
        """
        try:
            from execution.exchange_utils import get_orderbook, get_price

            book = get_orderbook(symbol, limit=5)
            bids = book.get("bids", [])
            asks = book.get("asks", [])

            if bids and asks:
                bb = float(bids[0][0])
                ba = float(asks[0][0])
                return {"best_bid": bb, "best_ask": ba, "mid": (bb + ba) / 2}

            # Fallback: ticker price
            price = get_price(symbol)
            if price and price > 0:
                return {"best_bid": price, "best_ask": price, "mid": price}
        except Exception as exc:
            logger.warning("shadow_fill: book snapshot failed for %s: %s", symbol, exc)
        return None

    def simulate_fill(
        self,
        symbol: str,
        direction: str,
        *,
        is_entry: bool = True,
    ) -> Optional[SimulatedFill]:
        """
        Simulate a deterministic fill.

        Entry UP  → fill at best_ask + buffer (worst-case for buyer)
        Entry DOWN → fill at best_bid - buffer (worst-case for seller)
        Exit  UP  → fill at best_bid - buffer (worst-case exit for long)
        Exit  DOWN → fill at best_ask + buffer (worst-case exit for short)
        """
        snap = self._fetch_book_snapshot(symbol)
        if snap is None:
            return None

        bb = snap["best_bid"]
        ba = snap["best_ask"]
        mid = snap["mid"]
        slip_mult = self._slippage_bps / 10_000

        if direction == "UP":
            fill = (ba * (1 + slip_mult)) if is_entry else (bb * (1 - slip_mult))
        else:  # DOWN
            fill = (bb * (1 - slip_mult)) if is_entry else (ba * (1 + slip_mult))

        now = datetime.now(timezone.utc).isoformat()
        return SimulatedFill(
            fill_price=fill,
            mid_price=mid,
            best_bid=bb,
            best_ask=ba,
            slippage_bps=self._slippage_bps,
            ts=now,
        )


# ---------------------------------------------------------------------------
# Trade log writer (append-only)
# ---------------------------------------------------------------------------

class BinaryLabTradeWriter:
    """Append-only JSONL writer for binary_lab_trades.jsonl."""

    def __init__(self, path: Path = TRADE_LOG_PATH):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self.write_failure_count: int = 0

    def write(self, record: Dict[str, Any]) -> None:
        try:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        except Exception as exc:
            self.write_failure_count += 1
            logger.warning("binary_lab_trade_writer: write failed: %s", exc)


# ---------------------------------------------------------------------------
# Round ID + boundary helpers
# ---------------------------------------------------------------------------

def _round_start_unix(ts_unix: float) -> float:
    """Compute the start of the 15-min round containing *ts_unix*."""
    return ts_unix - (ts_unix % ROUND_DURATION_S)


def _make_round_id(round_start_unix: float) -> str:
    """Deterministic round ID from the round start timestamp."""
    dt = datetime.fromtimestamp(round_start_unix, tz=timezone.utc)
    return f"R_{dt.strftime('%Y%m%d_%H%M')}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ts_to_unix(ts_iso: str) -> float:
    try:
        return datetime.fromisoformat(ts_iso).timestamp()
    except Exception:
        return time.time()


# ---------------------------------------------------------------------------
# Shadow runner
# ---------------------------------------------------------------------------

class BinaryLabShadowRunner:
    """
    Shadow orchestrator — called once per executor tick.

    Manages:
        - 15-min round boundary detection
        - Eligibility gating (via :mod:`binary_lab_signals`)
        - Entry simulation via :class:`SimulatedExecutionAdapter`
        - Round resolution at maturity
        - State machine feeding (reuses :class:`BinaryLabRuntimeWriter`)
        - Trade log emission (append-only JSONL)

    All events are tagged ``execution_mode: SHADOW``.
    """

    def __init__(
        self,
        *,
        limits: Dict[str, Any],
        writer: Any = None,          # BinaryLabRuntimeWriter (optional; for state emission)
        trade_log_path: Path = TRADE_LOG_PATH,
        symbol: str = DEFAULT_SYMBOL,
        sentinel_path: Optional[Path] = None,
        scores_path: Optional[Path] = None,
    ) -> None:
        self._limits = limits
        self._runtime_writer = writer
        self._trade_writer = BinaryLabTradeWriter(trade_log_path)
        self._adapter = SimulatedExecutionAdapter()
        self._symbol = symbol
        self._sentinel_path = sentinel_path
        self._scores_path = scores_path

        self._open_rounds: List[OpenRound] = []
        self._processed_round_ids: set[str] = set()

        cap = limits.get("capital") or {}
        self._per_round_usd: float = float(cap.get("per_round_usd", 20.0))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def open_round_count(self) -> int:
        return len(self._open_rounds)

    def tick(self, now_ts: str) -> bool:
        """
        Main entry point — called once per executor cycle (~60s).

        Returns True if any state change occurred (entry or resolution).
        """
        now_unix = _ts_to_unix(now_ts)
        changed = False

        # Phase 1: resolve matured rounds
        resolved = self._resolve_matured_rounds(now_unix, now_ts)
        if resolved:
            changed = True

        # Phase 2: check for new round entry
        entered = self._maybe_enter_round(now_unix, now_ts)
        if entered:
            changed = True

        # Phase 3: emit NO_TRADE events for skipped rounds
        self._log_skipped_rounds(now_unix, now_ts)

        return changed

    # ------------------------------------------------------------------
    # Phase 1: Resolve matured rounds
    # ------------------------------------------------------------------

    def _resolve_matured_rounds(self, now_unix: float, now_ts: str) -> bool:
        resolved_any = False
        still_open: List[OpenRound] = []

        for rnd in self._open_rounds:
            if now_unix >= rnd.resolution_ts_unix:
                outcome = self._resolve_round(rnd, now_ts)
                if outcome is not None:
                    self._emit_round_closed(outcome)
                    resolved_any = True
            else:
                still_open.append(rnd)

        self._open_rounds = still_open
        return resolved_any

    def _resolve_round(self, rnd: OpenRound, now_ts: str) -> Optional[RoundOutcome]:
        """Compute PnL for a matured round."""
        exit_fill = self._adapter.simulate_fill(
            rnd.symbol, rnd.direction, is_entry=False,
        )
        if exit_fill is None:
            logger.warning("shadow: exit fill failed for %s — recording as LOSS", rnd.round_id)
            # Conservative: treat missing exit as a loss
            return RoundOutcome(
                round_id=rnd.round_id,
                symbol=rnd.symbol,
                direction=rnd.direction,
                outcome="LOSS",
                pnl_usd=0.0,
                fee_usd=0.0,
                gross_pnl_usd=0.0,
                entry_price=rnd.entry_fill.fill_price,
                exit_price=rnd.entry_fill.fill_price,  # flat assumption
                notional_usd=rnd.notional_usd,
                conviction_band=rnd.conviction_band,
                conviction_score=rnd.conviction_score,
                regime=rnd.regime,
                regime_confidence=rnd.regime_confidence,
                entry_ts=rnd.entry_ts,
                exit_ts=now_ts,
            )

        entry_p = rnd.entry_fill.fill_price
        exit_p = exit_fill.fill_price
        qty = rnd.notional_usd / entry_p if entry_p > 0 else 0.0

        # PnL computation
        if rnd.direction == "UP":
            gross = (exit_p - entry_p) * qty
        else:  # DOWN
            gross = (entry_p - exit_p) * qty

        fee_per_side = rnd.notional_usd * (FEE_BPS / 10_000)
        total_fee = fee_per_side * 2  # entry + exit
        net = gross - total_fee
        outcome = "WIN" if net > 0 else "LOSS"

        return RoundOutcome(
            round_id=rnd.round_id,
            symbol=rnd.symbol,
            direction=rnd.direction,
            outcome=outcome,
            pnl_usd=round(net, 8),
            fee_usd=round(total_fee, 8),
            gross_pnl_usd=round(gross, 8),
            entry_price=round(entry_p, 8),
            exit_price=round(exit_p, 8),
            notional_usd=rnd.notional_usd,
            conviction_band=rnd.conviction_band,
            conviction_score=rnd.conviction_score,
            regime=rnd.regime,
            regime_confidence=rnd.regime_confidence,
            entry_ts=rnd.entry_ts,
            exit_ts=exit_fill.ts,
        )

    # ------------------------------------------------------------------
    # Phase 2: Enter new round
    # ------------------------------------------------------------------

    def _maybe_enter_round(self, now_unix: float, now_ts: str) -> bool:
        """Check if we are in the entry window for a new 15-min round."""
        round_start = _round_start_unix(now_unix)
        round_id = _make_round_id(round_start)

        # Already processed this round?
        if round_id in self._processed_round_ids:
            return False

        # Are we in the entry window?
        window_start = round_start + ENTRY_OFFSET_S
        window_end = window_start + ENTRY_WINDOW_S
        if not (window_start <= now_unix <= window_end):
            return False

        # Mark processed (regardless of eligibility)
        self._processed_round_ids.add(round_id)
        # Reclaim memory: drop round IDs older than 2 hours
        cutoff = now_unix - 7200
        self._processed_round_ids = {
            rid for rid in self._processed_round_ids
            if rid >= _make_round_id(cutoff)
        }

        # Extract signal
        from execution.binary_lab_signals import extract_signal, check_eligibility

        sig_kwargs: Dict[str, Any] = {"symbol": self._symbol}
        if self._sentinel_path:
            sig_kwargs["sentinel_path"] = self._sentinel_path
        if self._scores_path:
            sig_kwargs["scores_path"] = self._scores_path

        signal = extract_signal(**sig_kwargs)
        if signal is None:
            self._log_no_trade(round_id, now_ts, "signal_unavailable")
            return False

        # Get current state for gate check
        state = self._get_current_state()
        elig = check_eligibility(
            signal,
            self._limits,
            current_nav_usd=state.get("current_nav_usd", 2000.0),
            open_positions=len(self._open_rounds),
            freeze_intact=state.get("freeze_intact", True),
        )

        if not elig.eligible:
            self._log_no_trade(round_id, now_ts, elig.deny_reason or "ineligible", signal=signal)
            return False

        # Simulate entry
        assert signal.direction is not None
        fill = self._adapter.simulate_fill(self._symbol, signal.direction, is_entry=True)
        if fill is None:
            self._log_no_trade(round_id, now_ts, "fill_simulation_failed", signal=signal)
            return False

        rnd = OpenRound(
            round_id=round_id,
            symbol=self._symbol,
            direction=signal.direction,
            entry_fill=fill,
            notional_usd=self._per_round_usd,
            entry_ts=now_ts,
            entry_ts_unix=now_unix,
            resolution_ts_unix=round_start + ROUND_DURATION_S,
            conviction_band=signal.conviction_band,
            conviction_score=signal.conviction_score,
            regime=signal.regime,
            regime_confidence=signal.regime_confidence,
        )
        self._open_rounds.append(rnd)

        # Log entry event
        self._trade_writer.write({
            "event_type": "ENTRY",
            "execution_mode": "SHADOW",
            "ts": now_ts,
            "ts_ms": int(now_unix * 1000),
            "round_id": round_id,
            "market_slug": self._symbol,
            "horizon_s": ROUND_DURATION_S,
            "side": signal.direction,
            "intent_direction": signal.direction,
            "p_fill": round(fill.fill_price, 8),
            "notional_usd": self._per_round_usd,
            "conviction_band": signal.conviction_band,
            "conviction_score": signal.conviction_score,
            "regime": signal.regime,
            "regime_confidence": signal.regime_confidence,
            "status": "filled",
            "fill_snapshot": fill.to_dict(),
        })

        logger.info(
            "shadow: ENTRY %s %s %s @ %.2f  band=%s  regime=%s",
            round_id, self._symbol, signal.direction,
            fill.fill_price, signal.conviction_band, signal.regime,
        )
        return True

    # ------------------------------------------------------------------
    # State access helpers
    # ------------------------------------------------------------------

    def _get_current_state(self) -> Dict[str, Any]:
        """Read current sleeve state from the runtime writer or defaults."""
        if self._runtime_writer is not None:
            state = self._runtime_writer.state
            if state is not None:
                return {
                    "current_nav_usd": state.current_nav_usd,
                    "freeze_intact": state.freeze_intact,
                }
        return {"current_nav_usd": 2000.0, "freeze_intact": True}

    # ------------------------------------------------------------------
    # Round-closed emission
    # ------------------------------------------------------------------

    def _emit_round_closed(self, outcome: RoundOutcome) -> None:
        """Log to trade JSONL + feed state machine via RuntimeWriter."""
        # 1. Append to binary_lab_trades.jsonl
        self._trade_writer.write({
            "event_type": "ROUND_CLOSED",
            "execution_mode": "SHADOW",
            "ts": outcome.exit_ts,
            "ts_ms": int(_ts_to_unix(outcome.exit_ts) * 1000),
            "round_id": outcome.round_id,
            "market_slug": outcome.symbol,
            "horizon_s": ROUND_DURATION_S,
            "side": outcome.direction,
            "intent_direction": outcome.direction,
            "p_fill": outcome.entry_price,
            "p_exit": outcome.exit_price,
            "notional_usd": outcome.notional_usd,
            "fee_usd": outcome.fee_usd,
            "conviction_band": outcome.conviction_band,
            "conviction_score": outcome.conviction_score,
            "regime": outcome.regime,
            "regime_confidence": outcome.regime_confidence,
            "status": "resolved",
            "resolved_outcome": outcome.outcome,
            "pnl_usd": outcome.pnl_usd,
            "gross_pnl_usd": outcome.gross_pnl_usd,
        })

        # 2. Feed the state machine (via RuntimeWriter if available)
        if self._runtime_writer is not None:
            try:
                from execution.binary_lab_runtime import RuntimeLoopContext
                from execution.binary_lab_executor import BinaryLabEventType, BinaryLabMode

                ctx = RuntimeLoopContext(
                    now_ts=outcome.exit_ts,
                    open_positions=len(self._open_rounds),
                    trade_taken=True,
                    outcome=outcome.outcome,
                    conviction_band=outcome.conviction_band,
                    pnl_usd=outcome.pnl_usd,
                    size_usd=outcome.notional_usd,
                    event_type_override=BinaryLabEventType.ROUND_CLOSED,
                    mode=BinaryLabMode.PAPER,  # SHADOW maps to PAPER for the state machine
                )
                self._runtime_writer.tick(ctx)
            except Exception as exc:
                logger.warning("shadow: state machine feed failed: %s", exc)

        logger.info(
            "shadow: RESOLVED %s %s %s → %s  pnl=%.4f  band=%s",
            outcome.round_id, outcome.symbol, outcome.direction,
            outcome.outcome, outcome.pnl_usd, outcome.conviction_band,
        )

    # ------------------------------------------------------------------
    # No-trade logging
    # ------------------------------------------------------------------

    def _log_no_trade(
        self,
        round_id: str,
        now_ts: str,
        deny_reason: str,
        *,
        signal: Optional[Any] = None,
    ) -> None:
        record: Dict[str, Any] = {
            "event_type": "NO_TRADE",
            "execution_mode": "SHADOW",
            "ts": now_ts,
            "ts_ms": int(_ts_to_unix(now_ts) * 1000),
            "round_id": round_id,
            "market_slug": self._symbol,
            "horizon_s": ROUND_DURATION_S,
            "status": "no_trade",
            "eligibility": False,
            "deny_reason": deny_reason,
        }
        if signal is not None:
            record.update({
                "regime": signal.regime,
                "regime_confidence": signal.regime_confidence,
                "conviction_band": signal.conviction_band,
                "conviction_score": signal.conviction_score,
                "side": signal.direction,
            })
        self._trade_writer.write(record)

    def _log_skipped_rounds(self, now_unix: float, now_ts: str) -> None:
        """Detect and log any rounds that the entry window has closed for."""
        # No-op: skipped rounds are recorded in _maybe_enter_round when
        # eligibility fails.  This is a placeholder for future round-gap
        # detection if needed.
        pass

    # ------------------------------------------------------------------
    # Metrics (for dashboard)
    # ------------------------------------------------------------------

    def get_shadow_metrics(self) -> Dict[str, Any]:
        """Return current shadow sleeve metrics for observability."""
        return {
            "open_rounds": len(self._open_rounds),
            "processed_round_count": len(self._processed_round_ids),
            "open_round_ids": [r.round_id for r in self._open_rounds],
        }
