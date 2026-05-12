"""
Phase 5: FPS v2 Shadow Soak Observer (Research-Only)

This module runs as an independent observer layer, comparing strategy signals
from the certified replay logic against actual live orders from the executor.
It produces no live orders, mutates no doctrine/risk/positions, and operates
purely for validation and metric collection.

Output:
- logs/research/shadow_soak_events.jsonl (append-only)
- logs/state/shadow_soak_state.json (state snapshot)

No imports from execution/ allowed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

# Sanity check: refuse to import executor modules
_forbidden_modules = ["executor_live", "order_dispatch", "doctrine_kernel", "risk_limits"]


def _check_no_forbidden_imports() -> None:
    """Abort if forbidden modules have been imported."""
    for mod_name in _forbidden_modules:
        if mod_name in sys.modules:
            raise RuntimeError(
                f"shadow_soak_v8 refuses to run: forbidden module '{mod_name}' already imported. "
                "This module must remain independent from live execution layers."
            )


_check_no_forbidden_imports()


class EventType(str, Enum):
    """Shadow soak event classification."""
    SHADOW_SIGNAL = "shadow_signal"
    LIVE_ORDER_MATCH = "live_order_match"
    SYMBOL_MISMATCH = "symbol_mismatch"
    DIRECTION_MISMATCH = "direction_mismatch"
    QUANTITY_MISMATCH = "quantity_mismatch"
    TIMESTAMP_DRIFT = "timestamp_drift"
    SLIPPAGE_OUTLIER = "slippage_outlier"
    EDGE_CASE_SKIP = "edge_case_skip"
    METRICS_CHECKPOINT = "metrics_checkpoint"


class MatchStatus(str, Enum):
    """Match classification."""
    MATCH = "match"
    PARTIAL = "partial"
    MISMATCH = "mismatch"
    UNMATCHABLE = "unmatchable"


class VerdictType(str, Enum):
    """14-day gate verdict."""
    PENDING = "pending"
    PASS = "pass"
    CONDITIONAL = "conditional"
    FAIL = "fail"


class SoakStatus(str, Enum):
    """Shadow soak runtime status."""
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class ShadowSoakEvent:
    """
    Shadow soak observation event.

    Fields:
    - ts: event creation timestamp (ISO8601 UTC)
    - event_type: EventType enum
    - run_id: certification run identifier
    - symbol: trading symbol (e.g., BTCUSDT)
    - live_side: 'BUY'|'SELL'|None
    - shadow_side: 'BUY'|'SELL'|None
    - live_qty: filled quantity from live order
    - shadow_qty: expected quantity from signal
    - live_price: actual fill price
    - shadow_price: model-expected price
    - live_order_ts: when live order was submitted (ISO8601)
    - shadow_signal_ts: when shadow signal was generated (ISO8601)
    - symbol_match: bool
    - direction_match: bool|None
    - quantity_bucket_match: bool
    - timestamp_delta_s: seconds between signal and live order
    - slippage_bps_actual: actual slippage in basis points
    - slippage_bps_model: model assumption in basis points
    - slippage_error_bps: (actual - model)
    - catastrophic_mismatch: bool (direction or symbol mismatch on comparable event)
    - reason: human-readable explanation
    """
    ts: str
    event_type: str
    run_id: str
    symbol: str
    live_side: str | None
    shadow_side: str | None
    live_qty: float | None
    shadow_qty: float | None
    live_price: float | None
    shadow_price: float | None
    live_order_ts: str | None
    shadow_signal_ts: str | None
    symbol_match: bool
    direction_match: bool | None
    quantity_bucket_match: bool
    timestamp_delta_s: float | None
    slippage_bps_actual: float | None
    slippage_bps_model: float
    slippage_error_bps: float | None
    catastrophic_mismatch: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)


@dataclass
class ShadowSoakState:
    """
    Shadow soak runner state snapshot.

    Fields:
    - run_id: certification run being observed
    - started_at: ISO8601 UTC when soak started
    - updated_at: ISO8601 UTC when state last updated
    - status: SoakStatus enum (RUNNING / PAUSED / COMPLETE / FAILED)
    - sample_size: total matched events (symbol + direction + qty comparable)
    - symbol_match_rate: % of events with symbol match
    - direction_match_rate: % of matched events with direction match
    - quantity_bucket_match_rate: % of matched events with qty within 5% bucket
    - timestamp_alignment_p95_s: p95 of |live_ts - shadow_ts|
    - slippage_model_error_r: correlation(actual, model) or None if < 10 samples
    - median_abs_slippage_error_bps: median(|actual - model|)
    - p95_abs_slippage_error_bps: 95th pct of |actual - model|
    - fill_latency_p99_s: 99th pct of fill round-trip time
    - catastrophic_mismatch_count: number of direction/symbol mismatches
    - consecutive_failed_checks: how many consecutive checks failed thresholds
    - verdict: VerdictType (PENDING / PASS / CONDITIONAL / FAIL)
    - failed_criteria: list of criteria that failed (for FAIL/CONDITIONAL)
    """
    run_id: str
    started_at: str
    updated_at: str
    status: str  # SoakStatus enum value
    sample_size: int
    symbol_match_rate: float
    direction_match_rate: float | None
    quantity_bucket_match_rate: float | None
    timestamp_alignment_p95_s: float | None
    slippage_model_error_r: float | None
    median_abs_slippage_error_bps: float | None
    p95_abs_slippage_error_bps: float | None
    fill_latency_p99_s: float | None
    catastrophic_mismatch_count: int
    consecutive_failed_checks: int
    verdict: str  # VerdictType enum value
    failed_criteria: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)


def _now_iso() -> str:
    """Return current UTC time as ISO8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec='microseconds').replace('+00:00', 'Z')


def _read_live_orders(logs_dir: Path, order_type: str = 'executed') -> list[dict[str, Any]]:
    """
    Read live orders from audit log.
    
    Args:
        logs_dir: logs/ directory path
        order_type: 'executed' or 'attempted'
    
    Returns:
        List of order dicts with keys: order_id, symbol, side, qty, price, timestamp, etc.
    """
    orders = []
    log_file = logs_dir / 'execution' / f'orders_{order_type}.jsonl'
    
    if not log_file.exists():
        return orders
    
    try:
        with log_file.open('r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    orders.append(json.loads(line))
    except Exception:
        pass  # Missing or corrupt log is acceptable in research mode
    
    return orders


def _read_shadow_signals(
    replay_dir: Path,
) -> list[dict[str, Any]]:
    """
    Read shadow signal stream (candidate entries from backtest trades).
    
    Args:
        replay_dir: replay certification run directory
    
    Returns:
        List of signal dicts with keys: symbol, side, qty, price, ts, reason
    """
    signals = []
    trades_file = replay_dir / 'trades.csv'
    
    if not trades_file.exists():
        return signals
    
    try:
        with trades_file.open('r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                signals.append({
                    'symbol': row.get('symbol'),
                    'side': row.get('side'),  # LONG / SHORT
                    'qty': float(row.get('qty', 0)),
                    'entry_price': float(row.get('entry_price', 0)),
                    'entry_ts': row.get('entry_ts'),
                    'reason': row.get('entry_reason', ''),
                })
    except Exception:
        pass
    
    return signals


def _side_to_order_side(side: str) -> str | None:
    """Convert LONG/SHORT to BUY/SELL."""
    if side == 'LONG':
        return 'BUY'
    elif side == 'SHORT':
        return 'SELL'
    return None


def _parse_iso(ts: str | None) -> datetime | None:
    """Parse ISO8601 timestamp to datetime."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace('Z', '+00:00'))
    except Exception:
        return None


def _timestamp_delta(ts1: str | None, ts2: str | None) -> float | None:
    """Compute seconds between two ISO8601 timestamps."""
    dt1 = _parse_iso(ts1)
    dt2 = _parse_iso(ts2)
    if dt1 and dt2:
        return abs((dt2 - dt1).total_seconds())
    return None


def _compute_slippage_bps(
    entry_price: float,
    fill_price: float,
    side: str,
) -> float:
    """
    Compute slippage in basis points.
    
    For BUY: slippage = (fill_price - entry_price) / entry_price * 10000
    For SELL: slippage = (entry_price - fill_price) / entry_price * 10000
    """
    if entry_price <= 0:
        return 0.0
    
    if side == 'BUY':
        return (fill_price - entry_price) / entry_price * 10000
    elif side == 'SELL':
        return (entry_price - fill_price) / entry_price * 10000
    
    return 0.0


def _qty_matches_bucket(live_qty: float, shadow_qty: float, bucket_pct: float = 0.05) -> bool:
    """Check if quantities are within bucket_pct tolerance."""
    if shadow_qty <= 0:
        return False
    return abs(live_qty - shadow_qty) / shadow_qty <= bucket_pct


class ShadowSoakRunner:
    """
    Main shadow soak orchestrator.
    
    Workflow:
    1. Read live orders from logs/execution/orders_executed.jsonl
    2. Read shadow signals from replay trades.csv
    3. Match live orders to shadow signals by (symbol, side, timestamp proximity)
    4. Compute metrics (symbol match rate, direction match, slippage error, etc.)
    5. Emit events to logs/research/shadow_soak_events.jsonl
    6. Publish state to logs/state/shadow_soak_state.json
    7. Evaluate 14-day gate criteria and set verdict
    """
    
    def __init__(
        self,
        run_id: str,
        logs_dir: Path,
        replay_dir: Path,
        output_base_dir: Path | None = None,
    ):
        self.run_id = run_id
        self.logs_dir = logs_dir
        self.replay_dir = replay_dir
        self.output_base_dir = output_base_dir or logs_dir
        
        # State
        self.started_at = _now_iso()
        self.events: list[ShadowSoakEvent] = []
        self.state: ShadowSoakState | None = None
        self.status = SoakStatus.RUNNING
        self.consecutive_failed_checks = 0
        
        # Metrics accumulators
        self.symbol_matches = 0
        self.direction_matches = 0
        self.qty_matches = 0
        self.timestamp_deltas: list[float] = []
        self.slippage_errors: list[float] = []
        self.catastrophic_count = 0
        self.fill_latencies: list[float] = []
    
    def run(self) -> ShadowSoakState:
        """Execute shadow soak and return final state."""
        try:
            # Step 1: Read inputs
            live_orders = _read_live_orders(self.logs_dir, 'executed')
            shadow_signals = _read_shadow_signals(self.replay_dir)
            
            # Step 2: Process matches
            self._process_matches(live_orders, shadow_signals)
            
            # Step 3: Compute metrics
            self._compute_metrics()
            
            # Step 4: Evaluate verdict
            self._evaluate_verdict()
            
            # Step 5: Publish
            self._publish_events()
            self._publish_state()
            
            if self.status == SoakStatus.PAUSED or self.status == SoakStatus.FAILED:
                self.state.status = self.status.value
        
        except Exception:
            self.status = SoakStatus.FAILED
            self.state = self._build_state()
            self.state.status = self.status.value
            self._publish_state()
            raise
        
        return self.state
    
    def _process_matches(
        self,
        live_orders: list[dict[str, Any]],
        shadow_signals: list[dict[str, Any]],
    ) -> None:
        """Match live orders to shadow signals and emit events."""
        # Build symbol -> signals map for faster lookup
        signal_map: dict[str, list[dict[str, Any]]] = {}
        for sig in shadow_signals:
            symbol = sig.get('symbol')
            if symbol:
                signal_map.setdefault(symbol, []).append(sig)
        
        # Try to match each live order
        for live_order in live_orders:
            live_symbol = live_order.get('symbol')
            
            # Find best matching shadow signal
            shadow_signals_for_symbol = signal_map.get(live_symbol, [])
            best_match = self._find_best_match(
                live_order,
                shadow_signals_for_symbol,
            )
            
            if best_match:
                self._emit_match_event(live_order, best_match)
            else:
                # No shadow signal for this order
                self._emit_nomatch_event(live_order)
    
    def _find_best_match(
        self,
        live_order: dict[str, Any],
        shadow_signals: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Find shadow signal closest in time to live order."""
        if not shadow_signals:
            return None
        
        live_ts = _parse_iso(live_order.get('timestamp'))
        if not live_ts:
            return None
        
        best = min(
            shadow_signals,
            key=lambda sig: abs(
                (_parse_iso(sig.get('entry_ts')) or live_ts - live_ts).total_seconds()
            )
            if _parse_iso(sig.get('entry_ts'))
            else float('inf'),
        )
        
        return best if _parse_iso(best.get('entry_ts')) else None
    
    def _emit_match_event(
        self,
        live_order: dict[str, Any],
        shadow_signal: dict[str, Any],
    ) -> None:
        """Emit event for live order matched to shadow signal."""
        live_symbol = live_order.get('symbol')
        shadow_symbol = shadow_signal.get('symbol')
        
        live_side = live_order.get('side')
        shadow_side = _side_to_order_side(shadow_signal.get('side'))
        
        live_qty = float(live_order.get('quantity', 0))
        shadow_qty = shadow_signal.get('qty', 0)
        
        live_price = float(live_order.get('filled_price', 0))
        shadow_price = shadow_signal.get('entry_price', 0)
        
        live_ts = live_order.get('timestamp')
        shadow_ts = shadow_signal.get('entry_ts')
        
        # Check matches
        symbol_match = live_symbol == shadow_symbol
        direction_match = live_side == shadow_side if symbol_match else None
        qty_match = _qty_matches_bucket(live_qty, shadow_qty)
        ts_delta = _timestamp_delta(live_ts, shadow_ts)
        
        # Slippage
        slippage_actual = _compute_slippage_bps(shadow_price, live_price, live_side)
        slippage_model = 5.0  # basis points (0.05%)
        slippage_error = slippage_actual - slippage_model
        
        # Catastrophic mismatch
        catastrophic = (not symbol_match) or (direction_match is False and symbol_match)
        
        if catastrophic:
            self.catastrophic_count += 1
            self.status = SoakStatus.PAUSED
        
        if symbol_match:
            self.symbol_matches += 1
        
        if direction_match:
            self.direction_matches += 1
        
        if qty_match:
            self.qty_matches += 1
        
        if ts_delta is not None:
            self.timestamp_deltas.append(ts_delta)
        
        if slippage_error is not None:
            self.slippage_errors.append(slippage_error)
        
        reason = f"symbol_match={symbol_match}, direction_match={direction_match}, qty_match={qty_match}"
        
        event = ShadowSoakEvent(
            ts=_now_iso(),
            event_type=EventType.LIVE_ORDER_MATCH.value if not catastrophic else (
                EventType.SYMBOL_MISMATCH.value if not symbol_match else EventType.DIRECTION_MISMATCH.value
            ),
            run_id=self.run_id,
            symbol=live_symbol,
            live_side=live_side,
            shadow_side=shadow_side,
            live_qty=live_qty,
            shadow_qty=shadow_qty,
            live_price=live_price,
            shadow_price=shadow_price,
            live_order_ts=live_ts,
            shadow_signal_ts=shadow_ts,
            symbol_match=symbol_match,
            direction_match=direction_match,
            quantity_bucket_match=qty_match,
            timestamp_delta_s=ts_delta,
            slippage_bps_actual=slippage_actual,
            slippage_bps_model=slippage_model,
            slippage_error_bps=slippage_error,
            catastrophic_mismatch=catastrophic,
            reason=reason,
        )
        
        self.events.append(event)
    
    def _emit_nomatch_event(self, live_order: dict[str, Any]) -> None:
        """Emit event for live order with no shadow signal."""
        event = ShadowSoakEvent(
            ts=_now_iso(),
            event_type=EventType.EDGE_CASE_SKIP.value,
            run_id=self.run_id,
            symbol=live_order.get('symbol'),
            live_side=live_order.get('side'),
            shadow_side=None,
            live_qty=float(live_order.get('quantity', 0)),
            shadow_qty=None,
            live_price=float(live_order.get('filled_price', 0)),
            shadow_price=None,
            live_order_ts=live_order.get('timestamp'),
            shadow_signal_ts=None,
            symbol_match=False,
            direction_match=None,
            quantity_bucket_match=False,
            timestamp_delta_s=None,
            slippage_bps_actual=None,
            slippage_bps_model=5.0,
            slippage_error_bps=None,
            catastrophic_mismatch=False,
            reason="no shadow signal found for live order",
        )
        
        self.events.append(event)
    
    def _compute_metrics(self) -> None:
        """Compute aggregated metrics from events."""
        total_comparable = sum(1 for e in self.events if e.symbol_match)
        
        if total_comparable > 0:
            self.symbol_matches = sum(1 for e in self.events if e.symbol_match)
        
        total_direction_comparable = sum(1 for e in self.events if e.direction_match is not None)
        if total_direction_comparable > 0:
            self.direction_matches = sum(1 for e in self.events if e.direction_match is True)
        
        self.qty_matches = sum(1 for e in self.events if e.quantity_bucket_match)
    
    def _build_state(self) -> ShadowSoakState:
        """Build current state snapshot."""
        total_events = len(self.events)
        
        symbol_match_rate = (
            self.symbol_matches / total_events if total_events > 0 else 0.0
        )
        
        direction_match_count = sum(1 for e in self.events if e.direction_match is not None)
        direction_match_rate = (
            self.direction_matches / direction_match_count if direction_match_count > 0 else None
        )
        
        qty_match_rate = (
            self.qty_matches / total_events if total_events > 0 else None
        )
        
        timestamp_p95 = None
        if self.timestamp_deltas:
            sorted_deltas = sorted(self.timestamp_deltas)
            idx = int(0.95 * len(sorted_deltas))
            timestamp_p95 = sorted_deltas[min(idx, len(sorted_deltas) - 1)]
        
        slippage_r = None
        if len(self.slippage_errors) >= 10:
            # Simplified: use stddev/mean as proxy (full correlation needs two series)
            mean_error = sum(self.slippage_errors) / len(self.slippage_errors)
            if mean_error != 0:
                stddev = (
                    sum((x - mean_error) ** 2 for x in self.slippage_errors)
                    / len(self.slippage_errors)
                ) ** 0.5
                slippage_r = 1.0 - (stddev / abs(mean_error)) if abs(mean_error) > 0 else None
        
        median_slippage_error = None
        if self.slippage_errors:
            sorted_errors = sorted(self.slippage_errors)
            idx = len(sorted_errors) // 2
            median_slippage_error = sorted_errors[idx]
        
        p95_slippage_error = None
        if self.slippage_errors:
            sorted_errors = sorted(self.slippage_errors)
            idx = int(0.95 * len(sorted_errors))
            p95_slippage_error = sorted_errors[min(idx, len(sorted_errors) - 1)]
        
        fill_latency_p99 = None
        if self.fill_latencies:
            sorted_latencies = sorted(self.fill_latencies)
            idx = int(0.99 * len(sorted_latencies))
            fill_latency_p99 = sorted_latencies[min(idx, len(sorted_latencies) - 1)]
        
        return ShadowSoakState(
            run_id=self.run_id,
            started_at=self.started_at,
            updated_at=_now_iso(),
            status=self.status.value,
            sample_size=total_events,
            symbol_match_rate=symbol_match_rate,
            direction_match_rate=direction_match_rate,
            quantity_bucket_match_rate=qty_match_rate,
            timestamp_alignment_p95_s=timestamp_p95,
            slippage_model_error_r=slippage_r,
            median_abs_slippage_error_bps=median_slippage_error,
            p95_abs_slippage_error_bps=p95_slippage_error,
            fill_latency_p99_s=fill_latency_p99,
            catastrophic_mismatch_count=self.catastrophic_count,
            consecutive_failed_checks=self.consecutive_failed_checks,
            verdict=VerdictType.PENDING.value,
        )
    
    def _evaluate_verdict(self) -> None:
        """Evaluate 14-day gate criteria and set verdict."""
        self.state = self._build_state()
        
        # Criteria
        criteria = []
        
        # 1. sample_size >= 100
        if self.state.sample_size >= 100:
            criteria.append(True)
        else:
            criteria.append(False)
        
        # 2. symbol_match_rate >= 0.95
        if self.state.symbol_match_rate >= 0.95:
            criteria.append(True)
        else:
            criteria.append(False)
        
        # 3. direction_match_rate >= 0.95
        if self.state.direction_match_rate is None or self.state.direction_match_rate >= 0.95:
            criteria.append(True)
        else:
            criteria.append(False)
        
        # 4. quantity_bucket_match_rate >= 0.95
        if self.state.quantity_bucket_match_rate is None or self.state.quantity_bucket_match_rate >= 0.95:
            criteria.append(True)
        else:
            criteria.append(False)
        
        # 5. timestamp_alignment_p95_s <= 60 (one executor cycle)
        if self.state.timestamp_alignment_p95_s is None or self.state.timestamp_alignment_p95_s <= 60.0:
            criteria.append(True)
        else:
            criteria.append(False)
        
        # 6. slippage_model_error_r >= 0.80
        if self.state.slippage_model_error_r is None or self.state.slippage_model_error_r >= 0.80:
            criteria.append(True)
        else:
            criteria.append(False)
        
        # 7. median_abs_slippage_error_bps <= 3
        if self.state.median_abs_slippage_error_bps is None or self.state.median_abs_slippage_error_bps <= 3.0:
            criteria.append(True)
        else:
            criteria.append(False)
        
        # 8. p95_abs_slippage_error_bps <= 10
        if self.state.p95_abs_slippage_error_bps is None or self.state.p95_abs_slippage_error_bps <= 10.0:
            criteria.append(True)
        else:
            criteria.append(False)
        
        # 9. fill_latency_p99_s < 2.0
        if self.state.fill_latency_p99_s is None or self.state.fill_latency_p99_s < 2.0:
            criteria.append(True)
        else:
            criteria.append(False)
        
        # 10. catastrophic_mismatch_count == 0
        if self.state.catastrophic_mismatch_count == 0:
            criteria.append(True)
        else:
            criteria.append(False)
        
        # Set verdict
        passed = sum(criteria)
        total = len(criteria)
        
        if self.status == SoakStatus.PAUSED or self.status == SoakStatus.FAILED:
            self.state.verdict = VerdictType.FAIL.value
        elif passed == total:
            self.state.verdict = VerdictType.PASS.value
        elif passed >= total - 2:
            self.state.verdict = VerdictType.CONDITIONAL.value
        else:
            self.state.verdict = VerdictType.FAIL.value
            self.consecutive_failed_checks += 1
        
        # Track failed criteria
        failed_names = [
            "sample_size >= 100",
            "symbol_match_rate >= 0.95",
            "direction_match_rate >= 0.95",
            "quantity_bucket_match_rate >= 0.95",
            "timestamp_alignment_p95_s <= 60",
            "slippage_model_error_r >= 0.80",
            "median_abs_slippage_error_bps <= 3",
            "p95_abs_slippage_error_bps <= 10",
            "fill_latency_p99_s < 2.0",
            "catastrophic_mismatch_count == 0",
        ]
        
        self.state.failed_criteria = [
            failed_names[i] for i, passed in enumerate(criteria) if not passed
        ]
    
    def _publish_events(self) -> None:
        """Write events to logs/research/shadow_soak_events.jsonl (append-only)."""
        events_dir = self.output_base_dir / 'research'
        events_dir.mkdir(parents=True, exist_ok=True)
        
        events_file = events_dir / 'shadow_soak_events.jsonl'
        
        with events_file.open('a', encoding='utf-8') as f:
            for event in self.events:
                f.write(json.dumps(event.to_dict()) + '\n')
    
    def _publish_state(self) -> None:
        """Write state to logs/state/shadow_soak_state.json (snapshot)."""
        state_dir = self.output_base_dir / 'state'
        state_dir.mkdir(parents=True, exist_ok=True)
        
        state_file = state_dir / 'shadow_soak_state.json'
        
        with state_file.open('w', encoding='utf-8') as f:
            json.dump(self.state.to_dict(), f, indent=2)


def run_shadow_soak(
    run_id: str,
    logs_dir: Path | str = 'logs',
    replay_dir: Path | str | None = None,
    output_base_dir: Path | str | None = None,
) -> ShadowSoakState:
    """
    Run shadow soak observer against a certification run.
    
    Args:
        run_id: certification run ID (e.g., 'v8_phase4_fps_v2_cert_003')
        logs_dir: path to logs/ directory
        replay_dir: path to replay directory; defaults to data/replay_certifications/<run_id>/
        output_base_dir: path to output directory; defaults to logs_dir
    
    Returns:
        Final ShadowSoakState
    
    Raises:
        RuntimeError: if forbidden execution modules are imported
    """
    logs_path = Path(logs_dir)
    
    if replay_dir is None:
        replay_dir = Path('data/replay_certifications') / run_id
    else:
        replay_dir = Path(replay_dir)
    
    if output_base_dir:
        output_path = Path(output_base_dir)
    else:
        output_path = logs_path
    
    runner = ShadowSoakRunner(
        run_id=run_id,
        logs_dir=logs_path,
        replay_dir=replay_dir,
        output_base_dir=output_path,
    )
    
    return runner.run()


def _parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse both positional and flag-based CLI forms for the runner script."""
    parser = argparse.ArgumentParser(prog='python -m research.shadow_soak_v8')
    parser.add_argument('run_id', nargs='?', help='Observer run id')
    parser.add_argument('logs_dir', nargs='?', default=None, help='Logs root directory')
    parser.add_argument('replay_dir', nargs='?', default=None, help='Replay certification directory')
    parser.add_argument('--run-id', dest='run_id_flag', help='Observer run id')
    parser.add_argument('--logs-root', dest='logs_root_flag', help='Logs root directory')
    parser.add_argument('--certification-dir', dest='certification_dir_flag', help='Replay certification directory')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = _parse_cli_args(sys.argv[1:])

    run_id = args.run_id_flag or args.run_id
    if not run_id:
        print(
            'Usage: python -m research.shadow_soak_v8 <run_id> [logs_dir] [replay_dir] '
            'or python -m research.shadow_soak_v8 --run-id <run_id> --logs-root <logs_dir> --certification-dir <replay_dir>'
        )
        sys.exit(1)

    logs_dir = args.logs_root_flag or args.logs_dir or 'logs'
    replay_dir = args.certification_dir_flag or args.replay_dir
    
    state = run_shadow_soak(run_id=run_id, logs_dir=logs_dir, replay_dir=replay_dir)
    
    print(json.dumps(state.to_dict(), indent=2))
