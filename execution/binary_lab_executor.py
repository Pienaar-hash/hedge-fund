"""
Binary Lab executor state machine (deterministic core).

This module intentionally contains no exchange I/O. It enforces governance and
risk transitions for the Binary Lab sleeve as pure state transitions:

    next_state, actions = reduce_event(current_state, event, limits)

Primary sources mapped:
- config/binary_lab_limits.json
- ops/BINARY_LAB_WINDOW_2026-02-XX.md
- ops/BINARY_LAB_DAILY_CHECKPOINT.md
- docs/PHASE_P1_PREDICTION_ADVISORY_DOCTRINE.md
- config/dataset_admission.json
- v7_manifest.json (binary_lab_state surface)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Tuple


class BinaryLabStatus(str, Enum):
    DISABLED = "DISABLED"
    NOT_DEPLOYED = "NOT_DEPLOYED"
    ACTIVE = "ACTIVE"
    TERMINATED = "TERMINATED"
    COMPLETED = "COMPLETED"


class BinaryLabMode(str, Enum):
    PAPER = "PAPER"
    LIVE = "LIVE"


class BinaryLabEventType(str, Enum):
    ACTIVATE = "ACTIVATE"
    ROUND_CLOSED = "ROUND_CLOSED"
    DAILY_CHECKPOINT = "DAILY_CHECKPOINT"
    TERMINATE = "TERMINATE"


@dataclass
class BandStats:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl_usd: float = 0.0

    def add_trade(self, outcome: str, pnl_usd: float) -> None:
        self.trades += 1
        if outcome == "WIN":
            self.wins += 1
        elif outcome == "LOSS":
            self.losses += 1
        self.pnl_usd += float(pnl_usd)

    def to_dict(self) -> Dict[str, Any]:
        ev = (self.pnl_usd / self.trades) if self.trades > 0 else None
        return {
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "pnl_usd": round(float(self.pnl_usd), 8),
            "ev_usd": None if ev is None else round(float(ev), 8),
        }


@dataclass
class BinaryLabState:
    sleeve_id: str
    status: BinaryLabStatus
    day: int
    day_total: int
    start_nav_usd: float
    current_nav_usd: float
    pnl_usd: float
    kill_distance_usd: float
    kill_breached: bool
    total_trades: int
    wins: int
    losses: int
    by_conviction_band: Dict[str, BandStats] = field(default_factory=dict)
    rule_violations: int = 0
    freeze_intact: bool = True
    config_hash: Optional[str] = None
    mode: BinaryLabMode = BinaryLabMode.PAPER
    termination_reason: Optional[str] = None
    last_event_ts: Optional[str] = None
    last_checkpoint_utc_date: Optional[str] = None

    def copy(self) -> "BinaryLabState":
        return dataclasses.replace(
            self,
            by_conviction_band={k: dataclasses.replace(v) for k, v in self.by_conviction_band.items()},
        )

    @property
    def win_rate(self) -> Optional[float]:
        if self.total_trades <= 0:
            return None
        return self.wins / self.total_trades

    def to_payload(self) -> Dict[str, Any]:
        return {
            "sleeve_id": self.sleeve_id,
            "status": self.status.value,
            "day": int(self.day),
            "day_total": int(self.day_total),
            "capital": {
                "current_nav_usd": round(float(self.current_nav_usd), 8),
                "pnl_usd": round(float(self.pnl_usd), 8),
            },
            "kill_line": {
                "distance_usd": round(float(self.kill_distance_usd), 8),
                "breached": bool(self.kill_breached),
            },
            "metrics": {
                "total_trades": int(self.total_trades),
                "wins": int(self.wins),
                "losses": int(self.losses),
                "win_rate": None if self.win_rate is None else round(float(self.win_rate), 8),
                "by_conviction_band": {
                    band: stats.to_dict() for band, stats in sorted(self.by_conviction_band.items())
                },
            },
            "rule_violations": int(self.rule_violations),
            "freeze_intact": bool(self.freeze_intact),
            "config_hash": self.config_hash,
            "mode": self.mode.value,
            "termination_reason": self.termination_reason,
            "last_checkpoint_utc_date": self.last_checkpoint_utc_date,
            "updated_ts": self.last_event_ts,
        }


def state_from_payload(payload: Mapping[str, Any]) -> BinaryLabState:
    """
    Hydrate BinaryLabState from persisted payload.
    Raises ValueError on invalid payload.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("payload_not_mapping")

    cap = payload.get("capital") if isinstance(payload.get("capital"), Mapping) else {}
    kill = payload.get("kill_line") if isinstance(payload.get("kill_line"), Mapping) else {}
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else {}
    bands_raw = metrics.get("by_conviction_band") if isinstance(metrics.get("by_conviction_band"), Mapping) else {}

    try:
        status = BinaryLabStatus(str(payload.get("status") or BinaryLabStatus.NOT_DEPLOYED.value))
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("invalid_status") from exc

    try:
        mode = BinaryLabMode(str(payload.get("mode") or BinaryLabMode.PAPER.value))
    except Exception:
        mode = BinaryLabMode.PAPER

    bands: Dict[str, BandStats] = {}
    for band, row in bands_raw.items():
        if not isinstance(row, Mapping):
            continue
        bands[str(band)] = BandStats(
            trades=int(float(row.get("trades") or 0)),
            wins=int(float(row.get("wins") or 0)),
            losses=int(float(row.get("losses") or 0)),
            pnl_usd=float(row.get("pnl_usd") or 0.0),
        )

    start_nav = float(payload.get("start_nav_usd") or 0.0)
    current_nav = float(cap.get("current_nav_usd") or 0.0)
    pnl_usd = float(cap.get("pnl_usd") or 0.0)
    if start_nav <= 0.0:
        start_nav = current_nav - pnl_usd

    return BinaryLabState(
        sleeve_id=str(payload.get("sleeve_id") or "binary_lab_s1"),
        status=status,
        day=int(float(payload.get("day") or 0)),
        day_total=int(float(payload.get("day_total") or 30)),
        start_nav_usd=start_nav,
        current_nav_usd=current_nav,
        pnl_usd=pnl_usd,
        kill_distance_usd=float(kill.get("distance_usd") or 0.0),
        kill_breached=bool(kill.get("breached")),
        total_trades=int(float(metrics.get("total_trades") or 0)),
        wins=int(float(metrics.get("wins") or 0)),
        losses=int(float(metrics.get("losses") or 0)),
        by_conviction_band=bands,
        rule_violations=int(float(payload.get("rule_violations") or 0)),
        freeze_intact=bool(payload.get("freeze_intact", True)),
        config_hash=(None if payload.get("config_hash") is None else str(payload.get("config_hash"))),
        mode=mode,
        termination_reason=(None if payload.get("termination_reason") is None else str(payload.get("termination_reason"))),
        last_event_ts=(None if payload.get("updated_ts") is None else str(payload.get("updated_ts"))),
        last_checkpoint_utc_date=(
            None if payload.get("last_checkpoint_utc_date") is None else str(payload.get("last_checkpoint_utc_date"))
        ),
    )


@dataclass(frozen=True)
class BinaryLabEvent:
    event_type: BinaryLabEventType
    ts: Optional[str] = None
    config_hash: Optional[str] = None
    mode: BinaryLabMode = BinaryLabMode.PAPER
    activation_gate_go: bool = False
    horizon_minutes: Optional[int] = None
    prediction_phase: Optional[str] = None
    dataset_states: Optional[Dict[str, str]] = None
    trade_taken: bool = False
    outcome: Optional[str] = None
    conviction_band: Optional[str] = None
    pnl_usd: float = 0.0
    size_usd: Optional[float] = None
    open_positions: Optional[int] = None
    same_direction_stacking: Optional[bool] = None
    martingale_detected: Optional[bool] = None
    size_escalation_detected: Optional[bool] = None
    freeze_broken: Optional[bool] = None
    core_nav_contaminated: Optional[bool] = None
    reason: Optional[str] = None


@dataclass(frozen=True)
class TransitionResult:
    state: BinaryLabState
    accepted: bool
    deny_reason: Optional[str]
    actions: Tuple[str, ...]


def initialize_state(limits: Dict[str, Any], *, day_total: int = 30) -> BinaryLabState:
    meta = limits.get("_meta") or {}
    cap = limits.get("capital") or {}
    kill_cfg = limits.get("kill_conditions") or {}

    sleeve_id = str(meta.get("sleeve_id") or "binary_lab_s1")
    start_nav = float(cap.get("sleeve_total_usd") or 0.0)
    kill_nav = float(kill_cfg.get("kill_nav_usd") or 0.0)

    return BinaryLabState(
        sleeve_id=sleeve_id,
        status=BinaryLabStatus.NOT_DEPLOYED,
        day=0,
        day_total=int(day_total),
        start_nav_usd=start_nav,
        current_nav_usd=start_nav,
        pnl_usd=0.0,
        kill_distance_usd=(start_nav - kill_nav),
        kill_breached=False,
        total_trades=0,
        wins=0,
        losses=0,
    )


def _as_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _kill_eval(state: BinaryLabState, limits: Dict[str, Any]) -> Tuple[bool, float]:
    cap = limits.get("capital") or {}
    kill_cfg = limits.get("kill_conditions") or {}
    start_nav = _as_float(cap.get("sleeve_total_usd"), state.start_nav_usd)

    kill_nav = _as_float(kill_cfg.get("kill_nav_usd"), 0.0)
    dd_abs_limit = _as_float(kill_cfg.get("sleeve_drawdown_usd"), 0.0)
    dd_pct_limit = _as_float(kill_cfg.get("sleeve_drawdown_pct"), 0.0)

    dd_abs = max(0.0, start_nav - state.current_nav_usd)
    dd_pct = (dd_abs / start_nav) if start_nav > 0 else 0.0
    breached = False
    if kill_nav > 0.0 and state.current_nav_usd <= kill_nav:
        breached = True
    if dd_abs_limit > 0.0 and dd_abs >= dd_abs_limit:
        breached = True
    if dd_pct_limit > 0.0 and dd_pct >= dd_pct_limit:
        breached = True

    distance = state.current_nav_usd - kill_nav
    return breached, distance


def _check_live_governance(event: BinaryLabEvent) -> Optional[str]:
    phase = str(event.prediction_phase or "").strip()
    ds = event.dataset_states or {}
    pm_snapshot_state = str(ds.get("polymarket_snapshot") or "")
    pm_feed_state = str(ds.get("prediction_polymarket_feed") or "")

    if phase != "P2_PRODUCTION":
        return "prediction_phase_not_p2"
    if pm_snapshot_state != "PRODUCTION_ELIGIBLE":
        return "polymarket_snapshot_not_production_eligible"
    if pm_feed_state != "PRODUCTION_ELIGIBLE":
        return "prediction_polymarket_feed_not_production_eligible"
    return None


def _register_violation(state: BinaryLabState, reason: str, actions: List[str]) -> None:
    state.rule_violations += 1
    state.termination_reason = reason
    state.status = BinaryLabStatus.TERMINATED
    actions.append("TERMINATE_IMMEDIATELY")
    actions.append("CLOSE_ALL_POSITIONS")


def _extract_utc_date(ts: Optional[str]) -> Optional[str]:
    if not ts:
        return None
    s = str(ts).strip()
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    return None


def reduce_event(
    current: BinaryLabState,
    event: BinaryLabEvent,
    limits: Dict[str, Any],
) -> TransitionResult:
    """
    Apply one event to the binary-lab state machine.

    Deterministic rules:
    - Any rule violation terminates immediately.
    - Any kill-line breach terminates immediately.
    - 30-day freeze/hash mismatch sets freeze_intact=false and terminates.
    - LIVE activation requires P2 + production-eligible prediction datasets.
    """
    state = current.copy()
    actions: List[str] = []
    state.last_event_ts = event.ts

    if state.status in (BinaryLabStatus.TERMINATED, BinaryLabStatus.COMPLETED):
        return TransitionResult(state=state, accepted=False, deny_reason="terminal_state", actions=tuple(actions))

    meta = limits.get("_meta") or {}
    cap = limits.get("capital") or {}
    pos_rules = limits.get("position_rules") or {}
    horizon_cfg = limits.get("time_horizon") or {}

    if event.event_type == BinaryLabEventType.ACTIVATE:
        if state.status != BinaryLabStatus.NOT_DEPLOYED:
            return TransitionResult(state=state, accepted=False, deny_reason="already_activated", actions=tuple(actions))
        if not event.activation_gate_go:
            return TransitionResult(state=state, accepted=False, deny_reason="activation_gate_no_go", actions=tuple(actions))

        requested_horizon = _as_int(event.horizon_minutes, 0)
        locked_horizon = _as_int(meta.get("time_horizon_minutes") or horizon_cfg.get("round_minutes"), 0)
        if requested_horizon != locked_horizon:
            return TransitionResult(state=state, accepted=False, deny_reason="horizon_mismatch", actions=tuple(actions))

        if event.mode == BinaryLabMode.LIVE:
            reason = _check_live_governance(event)
            if reason:
                return TransitionResult(state=state, accepted=False, deny_reason=reason, actions=tuple(actions))
        else:
            ds = event.dataset_states or {}
            if str(ds.get("polymarket_snapshot") or "") == "REJECTED":
                return TransitionResult(state=state, accepted=False, deny_reason="polymarket_snapshot_rejected", actions=tuple(actions))
            if str(ds.get("prediction_polymarket_feed") or "") == "REJECTED":
                return TransitionResult(state=state, accepted=False, deny_reason="prediction_polymarket_feed_rejected", actions=tuple(actions))

        state.status = BinaryLabStatus.ACTIVE
        state.mode = event.mode
        state.config_hash = event.config_hash
        actions.append("LOCK_CONFIG_HASH")
        return TransitionResult(state=state, accepted=True, deny_reason=None, actions=tuple(actions))

    if state.status != BinaryLabStatus.ACTIVE:
        return TransitionResult(state=state, accepted=False, deny_reason="not_active", actions=tuple(actions))

    if event.event_type == BinaryLabEventType.TERMINATE:
        state.status = BinaryLabStatus.TERMINATED
        state.termination_reason = event.reason or "manual_termination"
        actions.append("CLOSE_ALL_POSITIONS")
        return TransitionResult(state=state, accepted=True, deny_reason=None, actions=tuple(actions))

    if event.event_type == BinaryLabEventType.ROUND_CLOSED:
        if event.trade_taken:
            # Position/conduct rule checks
            max_concurrent = _as_int(pos_rules.get("max_concurrent"), 0)
            if event.open_positions is not None and max_concurrent > 0 and int(event.open_positions) > max_concurrent:
                _register_violation(state, "max_concurrent_breach", actions)

            size_target = _as_float(cap.get("per_round_usd"), 0.0)
            if event.size_usd is not None and size_target > 0 and abs(float(event.size_usd) - size_target) > 1e-9:
                _register_violation(state, "per_round_size_breach", actions)

            if bool(event.same_direction_stacking) and not bool(pos_rules.get("same_direction_stacking", False)):
                _register_violation(state, "same_direction_stacking_breach", actions)

            if bool(event.martingale_detected) and not bool(pos_rules.get("martingale", False)):
                _register_violation(state, "martingale_breach", actions)

            if bool(event.size_escalation_detected) and not bool(pos_rules.get("size_escalation_after_wins", False)):
                _register_violation(state, "size_escalation_breach", actions)

            outcome = str(event.outcome or "").upper()
            band = str(event.conviction_band or "unknown")
            pnl = float(event.pnl_usd)

            state.total_trades += 1
            if outcome == "WIN":
                state.wins += 1
            elif outcome == "LOSS":
                state.losses += 1
            state.pnl_usd += pnl
            state.current_nav_usd += pnl

            stats = state.by_conviction_band.get(band)
            if stats is None:
                stats = BandStats()
                state.by_conviction_band[band] = stats
            stats.add_trade(outcome, pnl)

        breached, distance = _kill_eval(state, limits)
        state.kill_breached = breached
        state.kill_distance_usd = distance
        if breached:
            state.status = BinaryLabStatus.TERMINATED
            state.termination_reason = "kill_line_breached"
            actions.append("TERMINATE_IMMEDIATELY")
            actions.append("CLOSE_ALL_POSITIONS")

        return TransitionResult(state=state, accepted=True, deny_reason=None, actions=tuple(actions))

    if event.event_type == BinaryLabEventType.DAILY_CHECKPOINT:
        if event.config_hash and state.config_hash and event.config_hash != state.config_hash:
            state.freeze_intact = False
            _register_violation(state, "config_hash_mismatch", actions)

        max_concurrent = _as_int(pos_rules.get("max_concurrent"), 0)
        if event.open_positions is not None and max_concurrent > 0 and int(event.open_positions) > max_concurrent:
            _register_violation(state, "max_concurrent_breach", actions)

        if bool(event.freeze_broken):
            state.freeze_intact = False
            _register_violation(state, "freeze_broken", actions)

        if bool(event.core_nav_contaminated):
            _register_violation(state, "core_nav_contaminated", actions)

        checkpoint_day = _extract_utc_date(event.ts)
        is_duplicate_checkpoint = (
            checkpoint_day is not None
            and state.last_checkpoint_utc_date is not None
            and checkpoint_day == state.last_checkpoint_utc_date
        )

        if not is_duplicate_checkpoint:
            state.day = min(int(state.day) + 1, int(state.day_total))
            if checkpoint_day is not None:
                state.last_checkpoint_utc_date = checkpoint_day
            if state.day >= state.day_total and state.status == BinaryLabStatus.ACTIVE:
                state.status = BinaryLabStatus.COMPLETED
                actions.append("WINDOW_COMPLETE")
        else:
            actions.append("CHECKPOINT_DUPLICATE_NOOP")

        breached, distance = _kill_eval(state, limits)
        state.kill_breached = breached
        state.kill_distance_usd = distance
        if breached and state.status == BinaryLabStatus.ACTIVE:
            state.status = BinaryLabStatus.TERMINATED
            state.termination_reason = "kill_line_breached"
            actions.append("TERMINATE_IMMEDIATELY")
            actions.append("CLOSE_ALL_POSITIONS")

        return TransitionResult(state=state, accepted=True, deny_reason=None, actions=tuple(actions))

    return TransitionResult(state=state, accepted=False, deny_reason="unsupported_event", actions=tuple(actions))
