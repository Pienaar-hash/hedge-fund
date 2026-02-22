"""
Binary Lab runtime writer (state surface emitter).

This module wires the pure reducer (`execution.binary_lab_executor`) into a
minimal runtime shim that writes `logs/state/binary_lab_state.json`.

Design constraints:
- One-way flow only: runtime context -> reducer -> state file
- No order placement / no exchange side-effects
- Fail-closed when limits hash cannot be proven
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from execution.binary_lab_executor import (
    BinaryLabEvent,
    BinaryLabEventType,
    BinaryLabMode,
    BinaryLabState,
    BinaryLabStatus,
    TransitionResult,
    initialize_state,
    reduce_event,
    state_from_payload,
)

DEFAULT_LIMITS_PATH = Path("config/binary_lab_limits.json")
DEFAULT_DATASET_ADMISSION_PATH = Path("config/dataset_admission.json")
DEFAULT_STATE_PATH = Path("logs/state/binary_lab_state.json")


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    with tmp_path.open("w", encoding="utf-8") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(path)


def _fallback_disabled_state(reason: str, ts: Optional[str], config_hash: Optional[str]) -> BinaryLabState:
    state = BinaryLabState(
        sleeve_id="binary_lab_s1",
        status=BinaryLabStatus.DISABLED,
        day=0,
        day_total=30,
        start_nav_usd=0.0,
        current_nav_usd=0.0,
        pnl_usd=0.0,
        kill_distance_usd=0.0,
        kill_breached=False,
        total_trades=0,
        wins=0,
        losses=0,
        config_hash=config_hash,
        mode=BinaryLabMode.PAPER,
        termination_reason=reason,
        freeze_intact=False,
        rule_violations=0,
        last_event_ts=ts,
    )
    return state


def _build_disabled_from_limits(
    limits: Dict[str, Any],
    *,
    reason: str,
    ts: Optional[str],
    config_hash: Optional[str],
) -> BinaryLabState:
    state = initialize_state(limits)
    state.status = BinaryLabStatus.DISABLED
    state.freeze_intact = False
    state.termination_reason = reason
    state.config_hash = config_hash
    state.last_event_ts = ts
    return state


def _load_dataset_states(
    admission_path: Path,
    dataset_ids: Optional[tuple[str, ...]] = None,
) -> Dict[str, str]:
    ids = dataset_ids or ("polymarket_snapshot", "prediction_polymarket_feed")
    if not admission_path.exists():
        return {ds: "REJECTED" for ds in ids}

    adm = _read_json(admission_path)
    datasets = adm.get("datasets") or {}
    out: Dict[str, str] = {}
    for ds in ids:
        entry = datasets.get(ds) or {}
        out[ds] = str(entry.get("state") or "REJECTED")
    return out


@dataclass(frozen=True)
class RuntimeLoopContext:
    now_ts: str
    open_positions: int = 0
    activate: bool = False
    activation_gate_go: bool = False
    mode: BinaryLabMode = BinaryLabMode.PAPER
    prediction_phase: str = "P1_ADVISORY"
    horizon_minutes: int = 15
    trade_taken: bool = False
    outcome: Optional[str] = None
    conviction_band: Optional[str] = None
    pnl_usd: float = 0.0
    size_usd: Optional[float] = None
    freeze_broken: bool = False
    core_nav_contaminated: bool = False
    event_type_override: Optional[BinaryLabEventType] = None

    def event_type(self) -> BinaryLabEventType:
        if self.event_type_override is not None:
            return self.event_type_override
        if self.activate:
            return BinaryLabEventType.ACTIVATE
        if self.trade_taken:
            return BinaryLabEventType.ROUND_CLOSED
        return BinaryLabEventType.DAILY_CHECKPOINT


class BinaryLabRuntimeWriter:
    """
    Minimal runtime shim for Binary Lab state emission.
    """

    def __init__(
        self,
        *,
        limits_path: Path = DEFAULT_LIMITS_PATH,
        dataset_admission_path: Path = DEFAULT_DATASET_ADMISSION_PATH,
        state_path: Path = DEFAULT_STATE_PATH,
        expected_limits_hash: Optional[str] = None,
    ) -> None:
        self._limits_path = Path(limits_path)
        self._dataset_admission_path = Path(dataset_admission_path)
        self._state_path = Path(state_path)
        self._expected_limits_hash = (
            expected_limits_hash
            if expected_limits_hash is not None
            else os.getenv("BINARY_LAB_LIMITS_HASH")
        )
        self._limits: Optional[Dict[str, Any]] = None
        self._limits_hash: Optional[str] = None
        self._state: Optional[BinaryLabState] = None

    @property
    def state(self) -> Optional[BinaryLabState]:
        return self._state

    @property
    def limits_hash(self) -> Optional[str]:
        return self._limits_hash

    def _write_state(self) -> None:
        if self._state is None:
            return
        _atomic_write_json(self._state_path, self._state.to_payload())

    def _set_disabled(self, reason: str, ts: Optional[str], config_hash: Optional[str]) -> BinaryLabState:
        if self._limits is not None:
            self._state = _build_disabled_from_limits(
                self._limits,
                reason=reason,
                ts=ts,
                config_hash=config_hash,
            )
        else:
            self._state = _fallback_disabled_state(reason, ts, config_hash)
        self._write_state()
        return self._state

    def boot(self, now_ts: str) -> BinaryLabState:
        """
        Initialize runtime state and emit first payload.

        Fail-closed:
        - missing limits file -> DISABLED
        - unset expected hash -> DISABLED
        - hash mismatch -> DISABLED
        """
        if not self._limits_path.exists():
            return self._set_disabled("limits_missing", now_ts, None)

        try:
            self._limits = _read_json(self._limits_path)
            self._limits_hash = _sha256_file(self._limits_path)
        except Exception:
            return self._set_disabled("limits_load_failed", now_ts, None)

        if not self._expected_limits_hash:
            return self._set_disabled("limits_hash_unset", now_ts, self._limits_hash)

        if self._limits_hash != self._expected_limits_hash:
            return self._set_disabled("limits_hash_mismatch", now_ts, self._limits_hash)

        # Resume from persisted state when available to preserve restart invariance.
        if self._state_path.exists():
            try:
                persisted = _read_json(self._state_path)
                loaded = state_from_payload(persisted)
            except Exception:
                return self._set_disabled("persisted_state_invalid", now_ts, self._limits_hash)

            persisted_hash = loaded.config_hash
            if persisted_hash is not None and persisted_hash != self._limits_hash:
                return self._set_disabled("persisted_state_hash_mismatch", now_ts, self._limits_hash)
            loaded.config_hash = self._limits_hash
            loaded.last_event_ts = now_ts
            self._state = loaded
        else:
            self._state = initialize_state(self._limits)
            self._state.config_hash = self._limits_hash
            self._state.last_event_ts = now_ts
        self._write_state()
        return self._state

    def tick(self, ctx: RuntimeLoopContext) -> TransitionResult:
        """
        Consume one deterministic loop context and emit updated state surface.
        """
        if self._state is None:
            self.boot(ctx.now_ts)

        if self._state is None:
            # Defensive: should never happen
            disabled = self._set_disabled("runtime_uninitialized", ctx.now_ts, self._limits_hash)
            return TransitionResult(
                state=disabled,
                accepted=False,
                deny_reason="runtime_uninitialized",
                actions=(),
            )

        if self._state.status == BinaryLabStatus.DISABLED:
            self._state.last_event_ts = ctx.now_ts
            self._write_state()
            return TransitionResult(
                state=self._state,
                accepted=False,
                deny_reason="runtime_disabled",
                actions=(),
            )

        if self._limits is None or self._limits_hash is None:
            disabled = self._set_disabled("limits_not_loaded", ctx.now_ts, None)
            return TransitionResult(
                state=disabled,
                accepted=False,
                deny_reason="limits_not_loaded",
                actions=(),
            )

        if self._expected_limits_hash != self._limits_hash:
            disabled = self._set_disabled("limits_hash_mismatch", ctx.now_ts, self._limits_hash)
            return TransitionResult(
                state=disabled,
                accepted=False,
                deny_reason="limits_hash_mismatch",
                actions=(),
            )

        dataset_states = _load_dataset_states(self._dataset_admission_path)
        event = BinaryLabEvent(
            event_type=ctx.event_type(),
            ts=ctx.now_ts,
            config_hash=self._limits_hash,
            mode=ctx.mode,
            activation_gate_go=ctx.activation_gate_go,
            horizon_minutes=ctx.horizon_minutes,
            prediction_phase=ctx.prediction_phase,
            dataset_states=dataset_states,
            trade_taken=ctx.trade_taken,
            outcome=ctx.outcome,
            conviction_band=ctx.conviction_band,
            pnl_usd=ctx.pnl_usd,
            size_usd=ctx.size_usd,
            open_positions=ctx.open_positions,
            freeze_broken=ctx.freeze_broken,
            core_nav_contaminated=ctx.core_nav_contaminated,
        )
        result = reduce_event(self._state, event, self._limits)
        self._state = result.state
        self._write_state()
        return result
