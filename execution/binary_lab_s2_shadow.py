"""
Binary Lab S2 — shadow execution runner (probability-model binary rounds).

Architecture:

    CLOB Health → S2 Signal → Eligibility Gate → Binary Fill → Resolution → State Reducer → Trade Log

Fork of ``binary_lab_shadow.py`` (S1) with these structural changes:

1. **Fill model**: S1 uses Binance orderbook (continuous prices).  S2 uses CLOB
   probability quotes (0-1 range).  Entry cost = ask-side probability.  No
   continuous exit — **binary resolution**: payout is 0 or 1 at expiry.

2. **Signal source**: ``extract_s2_signal()`` + ``check_s2_eligibility()`` replace
   S1's regime/conviction pipeline.

3. **Resolution**: At expiry, check BTC price vs round reference level.  If
   BTC ended above → YES resolves to 1.

4. **Baseline vs model**: Both tracks permanently logged.

5. **Executable pricing**: Entry cost uses ask, not mid.

All events tagged ``execution_mode: SHADOW``.

Frozen parameters (immutable for the 30-day window):

    ENTRY_OFFSET_S      = 30
    ENTRY_WINDOW_S      = 600
    ROUND_DURATION_S    = 900
    POLYMARKET_FEE_RATE = 0.02
    SETTLEMENT_SOURCE   = "binance_shadow_reference"
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------------
ROUND_DURATION_S: int = 900
ENTRY_OFFSET_S: int = 30
ENTRY_WINDOW_S: int = 600
POLYMARKET_FEE_RATE: float = 0.02
SETTLEMENT_SOURCE: str = "binance_shadow_reference"
DEFAULT_SYMBOL: str = "BTCUSDT"

# ---------------------------------------------------------------------------
# Log paths
# ---------------------------------------------------------------------------
TRADE_LOG_PATH = Path("logs/execution/binary_lab_s2_trades.jsonl")
PAPER_TRADE_LOG_PATH = Path("logs/execution/binary_lab_s2_paper_trades.jsonl")

# ---------------------------------------------------------------------------
# Edge bucket mapping — Amendment 8 (pure numeric labels)
# ---------------------------------------------------------------------------
def _edge_to_bucket(abs_edge: float) -> str:
    if abs_edge < 0.02:
        return "edge_0_2pp"
    if abs_edge < 0.05:
        return "edge_2_5pp"
    if abs_edge < 0.08:
        return "edge_5_8pp"
    return "edge_8pp_plus"


def _price_region(p_market: float) -> str:
    """Classify market price into a region for PnL attribution."""
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
# Data types
# ---------------------------------------------------------------------------
@dataclass
class OpenRound:
    """An in-flight S2 shadow round waiting for binary resolution."""
    round_id: str
    notional_usd: float
    entry_ts: str
    entry_ts_unix: float
    resolution_ts_unix: float
    # Trade decision
    trade_side: str                 # "YES" | "NO"
    # Quotes frozen at entry — Amendment 4
    p_yes_bid: float
    p_yes_ask: float
    p_yes_mid: float
    p_no_bid: float
    p_no_ask: float
    # Baseline + model — Amendment 1
    p_baseline_yes: float
    p_model_yes: float
    edge_yes: float
    baseline_edge: float
    # Executable pricing — Amendment 2
    entry_cost: float
    executable_edge: float
    expected_value_usd: float
    # Settlement — Amendment 5
    reference_btc_price: float
    # Metadata
    model_version: str
    features: Dict[str, float]
    quote_age_s: float
    calibration_active: bool
    calibration_confident: bool
    edge_bucket: str
    spread: float
    quote_source: str = "mid_plus_mean_spread"  # "clob_live" or "mid_plus_mean_spread"
    reconstructed_entry_cost: Optional[float] = None  # what entry_cost WOULD be under reconstruction
    config_hash: Optional[str] = None
    freeze_intact: bool = True


@dataclass
class PassiveRound:
    """Untraded round kept for calibration observation only."""
    round_id: str
    features: Dict[str, float]
    reference_btc_price: float
    resolution_ts_unix: float
    p_baseline_yes: float


@dataclass
class RoundOutcome:
    """Resolved outcome for an S2 shadow round."""
    round_id: str
    trade_side: str
    outcome: str                    # "WIN" | "LOSS"
    outcome_yes: bool
    payout: int                     # 0 or 1
    pnl_usd: float
    fee_usd: float
    gross_pnl_usd: float
    notional_usd: float
    # Quotes
    p_yes_bid: float
    p_yes_ask: float
    p_yes_mid: float
    p_no_bid: float
    p_no_ask: float
    # Baseline + model — Amendment 1
    p_baseline_yes: float
    p_model_yes: float
    edge_yes: float
    baseline_edge: float
    # Executable
    entry_cost: float
    executable_edge: float
    expected_value_usd: float
    # Settlement — Amendment 5
    reference_btc_price: float
    settlement_btc_price: float
    settlement_source: str
    # Metrics
    brier_component: float
    baseline_brier_component: float
    # Metadata
    model_version: str
    quote_age_s: float
    calibration_active: bool
    calibration_confident: bool
    edge_bucket: str
    spread: float
    entry_ts: str
    exit_ts: str
    features: Dict[str, float]
    quote_source: str = "mid_plus_mean_spread"
    reconstructed_entry_cost: Optional[float] = None
    slippage_vs_reconstructed: Optional[float] = None  # entry_cost - reconstructed_entry_cost
    price_region: str = "center"
    config_hash: Optional[str] = None
    freeze_intact: bool = True


# ---------------------------------------------------------------------------
# Trade log writer (append-only)
# ---------------------------------------------------------------------------

class BinaryLabS2TradeWriter:
    """Append-only JSONL writer for binary_lab_s2_trades.jsonl."""

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
            logger.warning("s2_trade_writer: write failed: %s", exc)


# ---------------------------------------------------------------------------
# Round boundary helpers
# ---------------------------------------------------------------------------

def _round_start_unix(ts_unix: float) -> float:
    return ts_unix - (ts_unix % ROUND_DURATION_S)


def _make_round_id(round_start_unix: float) -> str:
    dt = datetime.fromtimestamp(round_start_unix, tz=timezone.utc)
    return f"S2_R_{dt.strftime('%Y%m%d_%H%M')}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ts_to_unix(ts_iso: str) -> float:
    try:
        return datetime.fromisoformat(ts_iso).timestamp()
    except Exception:
        return time.time()


def _polymarket_fee(entry_cost: float, notional_usd: float) -> float:
    """Polymarket taker fee: rate * min(p, 1-p) * notional."""
    return POLYMARKET_FEE_RATE * min(entry_cost, 1.0 - entry_cost) * notional_usd


# ---------------------------------------------------------------------------
# Shadow runner
# ---------------------------------------------------------------------------

class BinaryLabS2ShadowRunner:
    """
    S2 shadow orchestrator — called once per executor tick.

    Manages:
        - 15-min round boundary detection
        - Probability signal extraction + eligibility gating
        - Binary fill simulation (ask-side pricing)
        - Binary resolution at maturity (BTC above/below reference)
        - State machine feeding (reuses BinaryLabRuntimeWriter)
        - Trade log emission (append-only JSONL)
        - Calibration model updates per resolution

    All events are tagged ``execution_mode: SHADOW``.
    """

    def __init__(
        self,
        *,
        limits: Dict[str, Any],
        model: Any,                  # BinaryProbabilityModel
        writer: Any = None,          # BinaryLabRuntimeWriter (optional)
        trade_log_path: Path = TRADE_LOG_PATH,
        symbol: str = DEFAULT_SYMBOL,
        config_hash: Optional[str] = None,
        paper_mode: bool = False,
    ) -> None:
        self._limits = limits
        self._model = model
        self._runtime_writer = writer
        self._trade_writer = BinaryLabS2TradeWriter(trade_log_path)
        self._symbol = symbol
        self._config_hash = config_hash
        self._paper_mode = paper_mode
        self._execution_mode = "PAPER_TRADE" if paper_mode else "SHADOW"

        # Paper trade writer — separate log for execution validation
        if paper_mode:
            self._paper_writer = BinaryLabS2TradeWriter(PAPER_TRADE_LOG_PATH)
        else:
            self._paper_writer = None

        self._open_rounds: List[OpenRound] = []
        self._passive_rounds: List[PassiveRound] = []
        self._processed_round_ids: set[str] = set()

        cap = limits.get("capital") or {}
        self._per_round_usd: float = float(cap.get("per_round_usd", 30.0))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def open_round_count(self) -> int:
        return len(self._open_rounds)

    def tick(self, now_ts: str) -> bool:
        """
        Main entry point — called once per executor cycle (~60s).
        Returns True if any state change occurred.
        """
        now_unix = _ts_to_unix(now_ts)
        changed = False

        # Phase 1: resolve matured rounds (traded)
        if self._resolve_matured_rounds(now_unix, now_ts):
            changed = True

        # Phase 1b: resolve matured passive rounds (untraded — calibration only)
        self._resolve_passive_rounds(now_unix)

        # Phase 2: enter new round
        if self._maybe_enter_round(now_unix, now_ts):
            changed = True

        # Persist model calibration stats
        try:
            self._model.save_state()
        except Exception:
            pass

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

    def _resolve_passive_rounds(self, now_unix: float) -> None:
        """Resolve untraded rounds for calibration observation only."""
        still_pending: List[PassiveRound] = []
        for pr in self._passive_rounds:
            if now_unix < pr.resolution_ts_unix:
                still_pending.append(pr)
                continue
            try:
                from execution.exchange_utils import get_price
                settlement_price = get_price(self._symbol)
            except Exception:
                settlement_price = None
            if settlement_price is None or settlement_price <= 0:
                continue
            outcome_yes = settlement_price > pr.reference_btc_price
            self._model.update_observation(pr.features, outcome_yes)
            logger.debug(
                "s2_shadow: passive observation %s outcome_yes=%s (n=%d)",
                pr.round_id, outcome_yes, self._model.n_observations,
            )
        self._passive_rounds = still_pending

    def _resolve_round(self, rnd: OpenRound, now_ts: str) -> Optional[RoundOutcome]:
        """Binary resolution: check BTC price vs reference."""
        # Fetch settlement price — Amendment 5
        try:
            from execution.exchange_utils import get_price
            settlement_price = get_price(self._symbol)
        except Exception as exc:
            logger.warning("s2_shadow: settlement price fetch failed: %s", exc)
            settlement_price = None

        if settlement_price is None or settlement_price <= 0:
            logger.warning(
                "s2_shadow: no settlement price for %s — recording as LOSS",
                rnd.round_id,
            )
            settlement_price = rnd.reference_btc_price  # flat

        # Binary outcome
        outcome_yes = settlement_price > rnd.reference_btc_price

        # Payout: did our side win?
        if rnd.trade_side == "YES":
            won = outcome_yes
        else:  # NO
            won = not outcome_yes

        payout = 1 if won else 0

        # PnL
        gross_pnl = rnd.notional_usd * (payout - rnd.entry_cost)
        fee = _polymarket_fee(rnd.entry_cost, rnd.notional_usd)
        net_pnl = gross_pnl - fee
        result = "WIN" if net_pnl > 0 else "LOSS"

        # Brier scores — Amendment 1
        outcome_int = 1 if outcome_yes else 0
        brier = (rnd.p_model_yes - outcome_int) ** 2
        baseline_brier = (rnd.p_baseline_yes - outcome_int) ** 2

        # Feed model — calibration update
        self._model.update_observation(rnd.features, outcome_yes)

        return RoundOutcome(
            round_id=rnd.round_id,
            trade_side=rnd.trade_side,
            outcome=result,
            outcome_yes=outcome_yes,
            payout=payout,
            pnl_usd=round(net_pnl, 8),
            fee_usd=round(fee, 8),
            gross_pnl_usd=round(gross_pnl, 8),
            notional_usd=rnd.notional_usd,
            p_yes_bid=rnd.p_yes_bid,
            p_yes_ask=rnd.p_yes_ask,
            p_yes_mid=rnd.p_yes_mid,
            p_no_bid=rnd.p_no_bid,
            p_no_ask=rnd.p_no_ask,
            p_baseline_yes=rnd.p_baseline_yes,
            p_model_yes=rnd.p_model_yes,
            edge_yes=rnd.edge_yes,
            baseline_edge=rnd.baseline_edge,
            entry_cost=rnd.entry_cost,
            executable_edge=rnd.executable_edge,
            expected_value_usd=rnd.expected_value_usd,
            reference_btc_price=rnd.reference_btc_price,
            settlement_btc_price=round(settlement_price, 2),
            settlement_source=SETTLEMENT_SOURCE,
            brier_component=round(brier, 8),
            baseline_brier_component=round(baseline_brier, 8),
            model_version=rnd.model_version,
            quote_age_s=rnd.quote_age_s,
            calibration_active=rnd.calibration_active,
            calibration_confident=rnd.calibration_confident,
            edge_bucket=rnd.edge_bucket,
            spread=rnd.spread,
            entry_ts=rnd.entry_ts,
            exit_ts=now_ts,
            features=rnd.features,
            quote_source=rnd.quote_source,
            reconstructed_entry_cost=rnd.reconstructed_entry_cost,
            slippage_vs_reconstructed=(
                round(rnd.entry_cost - rnd.reconstructed_entry_cost, 6)
                if rnd.reconstructed_entry_cost is not None
                else None
            ),
            price_region=_price_region(rnd.p_yes_mid),
            config_hash=rnd.config_hash,
            freeze_intact=rnd.freeze_intact,
        )

    # ------------------------------------------------------------------
    # Phase 2: Enter new round
    # ------------------------------------------------------------------

    def _maybe_enter_round(self, now_unix: float, now_ts: str) -> bool:
        round_start = _round_start_unix(now_unix)
        round_id = _make_round_id(round_start)

        # Already processed?
        if round_id in self._processed_round_ids:
            return False

        # In entry window?
        window_start = round_start + ENTRY_OFFSET_S
        window_end = window_start + ENTRY_WINDOW_S
        if not (window_start <= now_unix <= window_end):
            return False

        # Mark processed
        self._processed_round_ids.add(round_id)
        cutoff = now_unix - 7200
        self._processed_round_ids = {
            rid for rid in self._processed_round_ids
            if rid >= _make_round_id(cutoff)
        }

        # Time remaining in round
        time_remaining_s = (round_start + ROUND_DURATION_S) - now_unix

        # Extract S2 signal
        from execution.binary_lab_s2_signals import extract_s2_signal, check_s2_eligibility

        entry_gate = self._limits.get("entry_gate") or {}
        min_edge = float(entry_gate.get("min_edge_threshold", 0.03))

        signal = extract_s2_signal(
            self._model,
            per_round_usd=self._per_round_usd,
            min_edge_threshold=min_edge,
        )
        if signal is None:
            self._log_no_trade(round_id, now_ts, "signal_unavailable")
            return False

        # Friction-erased-edge: log as dedicated skip — Amendment 6
        if signal.skip_reason == "SKIP_FRICTION_ERASED_EDGE":
            self._log_no_trade(
                round_id, now_ts, "SKIP_FRICTION_ERASED_EDGE", signal=signal,
            )
            return False

        # Invalid quote reconstruction: log as dedicated skip
        if signal.skip_reason == "SKIP_INVALID_QUOTE_RECONSTRUCTION":
            self._log_no_trade(
                round_id, now_ts, "SKIP_INVALID_QUOTE_RECONSTRUCTION", signal=signal,
            )
            return False

        # State for gate check
        state = self._get_current_state()

        elig = check_s2_eligibility(
            signal,
            self._limits,
            current_nav_usd=state.get("current_nav_usd", 900.0),
            open_positions=len(self._open_rounds),
            freeze_intact=state.get("freeze_intact", True),
            time_remaining_s=time_remaining_s,
        )

        if not elig.eligible:
            self._log_no_trade(
                round_id, now_ts, elig.deny_reason or "ineligible", signal=signal,
            )
            return False

        # Fetch reference BTC price — Amendment 5
        try:
            from execution.exchange_utils import get_price
            ref_price = get_price(self._symbol)
        except Exception:
            ref_price = None

        if ref_price is None or ref_price <= 0:
            self._log_no_trade(round_id, now_ts, "reference_price_unavailable", signal=signal)
            return False

        # Build OpenRound
        rnd = OpenRound(
            round_id=round_id,
            notional_usd=self._per_round_usd,
            entry_ts=now_ts,
            entry_ts_unix=now_unix,
            resolution_ts_unix=round_start + ROUND_DURATION_S,
            trade_side=signal.trade_side,
            p_yes_bid=signal.p_yes_bid,
            p_yes_ask=signal.p_yes_ask,
            p_yes_mid=signal.p_yes_mid,
            p_no_bid=signal.p_no_bid,
            p_no_ask=signal.p_no_ask,
            p_baseline_yes=signal.p_baseline_yes,
            p_model_yes=signal.p_model_yes,
            edge_yes=signal.edge_yes,
            baseline_edge=signal.baseline_edge,
            entry_cost=signal.entry_cost,
            executable_edge=signal.executable_edge,
            expected_value_usd=signal.expected_value_usd,
            reference_btc_price=round(ref_price, 2),
            model_version=signal.model_version,
            features=signal.features,
            quote_age_s=signal.quote_age_s,
            calibration_active=signal.calibration_active,
            calibration_confident=signal.calibration_confident,
            edge_bucket=_edge_to_bucket(abs(signal.edge_yes)),
            spread=signal.spread,
            quote_source=signal.quote_reconstruction_mode,
            reconstructed_entry_cost=self._compute_reconstructed_entry_cost(signal),
            config_hash=self._config_hash,
            freeze_intact=state.get("freeze_intact", True),
        )
        self._open_rounds.append(rnd)

        # Log ENTRY
        self._trade_writer.write({
            "event_type": "ENTRY",
            "execution_mode": self._execution_mode,
            "ts": now_ts,
            "ts_ms": int(now_unix * 1000),
            "round_id": round_id,
            "market_slug": self._symbol,
            "horizon_s": ROUND_DURATION_S,
            "trade_side": signal.trade_side,
            "p_yes_bid": signal.p_yes_bid,
            "p_yes_ask": signal.p_yes_ask,
            "p_yes_mid": signal.p_yes_mid,
            "p_no_bid": signal.p_no_bid,
            "p_no_ask": signal.p_no_ask,
            "p_baseline_yes": signal.p_baseline_yes,
            "p_model_yes": signal.p_model_yes,
            "edge_yes": signal.edge_yes,
            "baseline_edge": signal.baseline_edge,
            "entry_cost": signal.entry_cost,
            "executable_edge": signal.executable_edge,
            "expected_value_usd": signal.expected_value_usd,
            "notional_usd": self._per_round_usd,
            "reference_btc_price": round(ref_price, 2),
            "quote_age_s": signal.quote_age_s,
            "quote_reconstruction_mode": signal.quote_reconstruction_mode,
            "quote_source": rnd.quote_source,
            "reconstructed_entry_cost": rnd.reconstructed_entry_cost,
            "spread": signal.spread,
            "edge_bucket": rnd.edge_bucket,
            "calibration_active": signal.calibration_active,
            "calibration_confident": signal.calibration_confident,
            "model_version": signal.model_version,
            "config_hash": self._config_hash,
            "freeze_intact": rnd.freeze_intact,
            "status": "filled",
        })

        logger.info(
            "s2_shadow: ENTRY %s %s side=%s cost=%.4f edge=%.4f bucket=%s",
            round_id, self._symbol, signal.trade_side,
            signal.entry_cost, signal.executable_edge, rnd.edge_bucket,
        )
        return True

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    def _get_current_state(self) -> Dict[str, Any]:
        if self._runtime_writer is not None:
            state = self._runtime_writer.state
            if state is not None:
                return {
                    "current_nav_usd": state.current_nav_usd,
                    "freeze_intact": state.freeze_intact,
                }
        return {"current_nav_usd": 900.0, "freeze_intact": True}

    @staticmethod
    def _compute_reconstructed_entry_cost(signal: Any) -> Optional[float]:
        """Compute what entry_cost would be under mid+spread reconstruction.

        When quote_source is already 'mid_plus_mean_spread', returns signal.entry_cost.
        When quote_source is 'clob_live', reconstructs from mid + spread to measure delta.
        """
        if signal.quote_reconstruction_mode != "clob_live":
            return signal.entry_cost  # already using reconstruction — no delta

        from execution.binary_lab_s2_signals import _reconstruct_quotes
        quotes = _reconstruct_quotes(signal.p_yes_mid, signal.spread)
        if quotes is None:
            return None

        if signal.trade_side == "YES":
            return quotes["p_yes_ask"]
        elif signal.trade_side == "NO":
            return quotes["p_no_ask"]
        return None

    # ------------------------------------------------------------------
    # Round-closed emission
    # ------------------------------------------------------------------

    def _emit_round_closed(self, outcome: RoundOutcome) -> None:
        """Log to JSONL + feed state machine."""
        # PnL field semantics (locked):
        #   gross_pnl_usd = notional * (payout - entry_cost)
        #   fee_usd       = polymarket taker fee
        #   pnl_usd       = gross_pnl_usd - fee_usd
        self._trade_writer.write({
            "event_type": "ROUND_CLOSED",
            "execution_mode": self._execution_mode,
            "ts": outcome.exit_ts,
            "ts_ms": int(_ts_to_unix(outcome.exit_ts) * 1000),
            "round_id": outcome.round_id,
            "market_slug": self._symbol,
            "horizon_s": ROUND_DURATION_S,
            "trade_side": outcome.trade_side,
            "p_yes_bid": outcome.p_yes_bid,
            "p_yes_ask": outcome.p_yes_ask,
            "p_yes_mid": outcome.p_yes_mid,
            "p_no_bid": outcome.p_no_bid,
            "p_no_ask": outcome.p_no_ask,
            "p_baseline_yes": outcome.p_baseline_yes,
            "p_model_yes": outcome.p_model_yes,
            "edge_yes": outcome.edge_yes,
            "baseline_edge": outcome.baseline_edge,
            "entry_cost": outcome.entry_cost,
            "executable_edge": outcome.executable_edge,
            "expected_value_usd": outcome.expected_value_usd,
            "notional_usd": outcome.notional_usd,
            "fee_usd": outcome.fee_usd,
            "reference_btc_price": outcome.reference_btc_price,
            "settlement_btc_price": outcome.settlement_btc_price,
            "settlement_source": outcome.settlement_source,
            "quote_age_s": outcome.quote_age_s,
            "quote_reconstruction_mode": outcome.quote_source,
            "quote_source": outcome.quote_source,
            "reconstructed_entry_cost": outcome.reconstructed_entry_cost,
            "slippage_vs_reconstructed": outcome.slippage_vs_reconstructed,
            "price_region": outcome.price_region,
            "spread": outcome.spread,
            "edge_bucket": outcome.edge_bucket,
            "calibration_active": outcome.calibration_active,
            "calibration_confident": outcome.calibration_confident,
            "model_version": outcome.model_version,
            "config_hash": outcome.config_hash,
            "freeze_intact": outcome.freeze_intact,
            "status": "resolved",
            "resolved_outcome": outcome.outcome,
            "outcome_yes": outcome.outcome_yes,
            "payout": outcome.payout,
            "pnl_usd": outcome.pnl_usd,
            "gross_pnl_usd": outcome.gross_pnl_usd,
            "net_pnl_usd": outcome.pnl_usd,
            "brier_component": outcome.brier_component,
            "baseline_brier_component": outcome.baseline_brier_component,
        })

        # Paper trade log — enhanced record for execution validation
        if self._paper_writer is not None:
            self._paper_writer.write({
                "event_type": "PAPER_ROUND_CLOSED",
                "execution_mode": "PAPER_TRADE",
                "ts": outcome.exit_ts,
                "entry_ts": outcome.entry_ts,
                "round_id": outcome.round_id,
                "trade_side": outcome.trade_side,
                "quote_source": outcome.quote_source,
                "p_yes_bid": outcome.p_yes_bid,
                "p_yes_ask": outcome.p_yes_ask,
                "p_yes_mid": outcome.p_yes_mid,
                "entry_cost": outcome.entry_cost,
                "reconstructed_entry_cost": outcome.reconstructed_entry_cost,
                "slippage_vs_reconstructed": outcome.slippage_vs_reconstructed,
                "price_region": outcome.price_region,
                "edge_yes": outcome.edge_yes,
                "executable_edge": outcome.executable_edge,
                "edge_bucket": outcome.edge_bucket,
                "spread": outcome.spread,
                "outcome": outcome.outcome,
                "outcome_yes": outcome.outcome_yes,
                "payout": outcome.payout,
                "pnl_usd": outcome.pnl_usd,
                "gross_pnl_usd": outcome.gross_pnl_usd,
                "fee_usd": outcome.fee_usd,
                "notional_usd": outcome.notional_usd,
                "reference_btc_price": outcome.reference_btc_price,
                "settlement_btc_price": outcome.settlement_btc_price,
                "brier_component": outcome.brier_component,
                "calibration_confident": outcome.calibration_confident,
            })

        # Feed state machine (edge_bucket as conviction_band) — Amendment 8
        if self._runtime_writer is not None:
            try:
                from execution.binary_lab_runtime import RuntimeLoopContext
                from execution.binary_lab_executor import BinaryLabEventType, BinaryLabMode

                ctx = RuntimeLoopContext(
                    now_ts=outcome.exit_ts,
                    open_positions=len(self._open_rounds),
                    trade_taken=True,
                    outcome=outcome.outcome,
                    conviction_band=outcome.edge_bucket,
                    pnl_usd=outcome.pnl_usd,
                    size_usd=outcome.notional_usd,
                    event_type_override=BinaryLabEventType.ROUND_CLOSED,
                    mode=BinaryLabMode.PAPER,
                )
                self._runtime_writer.tick(ctx)
            except Exception as exc:
                logger.warning("s2_shadow: state machine feed failed: %s", exc)

        # Save calibration state after each resolution
        try:
            self._model.save_state()
        except Exception as exc:
            logger.warning("s2_shadow: calibration save failed: %s", exc)

        logger.info(
            "s2_shadow: RESOLVED %s side=%s → %s pnl=%.4f brier=%.4f baseline_brier=%.4f",
            outcome.round_id, outcome.trade_side, outcome.outcome,
            outcome.pnl_usd, outcome.brier_component, outcome.baseline_brier_component,
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
        signal: Any = None,
    ) -> None:
        record: Dict[str, Any] = {
            "event_type": "NO_TRADE",
            "execution_mode": self._execution_mode,
            "ts": now_ts,
            "ts_ms": int(_ts_to_unix(now_ts) * 1000),
            "round_id": round_id,
            "market_slug": self._symbol,
            "horizon_s": ROUND_DURATION_S,
            "status": "no_trade",
            "eligibility": False,
            "deny_reason": deny_reason,
            "skip_reason": deny_reason,
        }
        if signal is not None:
            record.update({
                "p_yes_mid": signal.p_yes_mid,
                "p_baseline_yes": signal.p_baseline_yes,
                "p_model_yes": signal.p_model_yes,
                "edge_yes": signal.edge_yes,
                "baseline_edge": signal.baseline_edge,
                "spread": signal.spread,
                "quote_age_s": signal.quote_age_s,
                "quote_reconstruction_mode": signal.quote_reconstruction_mode,
                "trade_side": signal.trade_side,
                "calibration_active": signal.calibration_active,
                "calibration_confident": signal.calibration_confident,
                "model_version": signal.model_version,
            })
        self._trade_writer.write(record)

        # Enqueue passive round for calibration observation
        if signal is not None and hasattr(signal, "features") and signal.features:
            try:
                from execution.exchange_utils import get_price
                ref_price = get_price(self._symbol)
            except Exception:
                ref_price = None
            if ref_price is not None and ref_price > 0:
                now_unix = _ts_to_unix(now_ts)
                round_start = _round_start_unix(now_unix)
                self._passive_rounds.append(PassiveRound(
                    round_id=round_id,
                    features=dict(signal.features),
                    reference_btc_price=ref_price,
                    resolution_ts_unix=round_start + ROUND_DURATION_S,
                    p_baseline_yes=signal.p_baseline_yes,
                ))

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_shadow_metrics(self) -> Dict[str, Any]:
        return {
            "open_rounds": len(self._open_rounds),
            "passive_rounds": len(self._passive_rounds),
            "processed_round_count": len(self._processed_round_ids),
            "open_round_ids": [r.round_id for r in self._open_rounds],
            "model_n_observations": self._model.n_observations,
            "calibration_active": self._model.calibration_active,
            "calibration_confident": self._model.calibration_confident,
        }
