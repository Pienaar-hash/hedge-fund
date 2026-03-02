"""
Churn Guard — Anti-Churn Safety Rails (v7.9-E2)

Structural gate that prevents fee-destructive rapid cycling.
This is NOT alpha tuning — it is a safety rail consistent with
"churn is the enemy" doctrine.

Two independent gates:
  1. min_hold_seconds: No exits until position has been held for N seconds
     (except CRISIS_OVERRIDE and SEATBELT — catastrophe protection overrides)
  2. cooldown_seconds: After exiting a symbol+side, no re-entry for N seconds

Both are configurable via runtime.yaml:
    churn_guard:
      min_hold_seconds: 120     # 2 minutes minimum hold
      cooldown_seconds: 300     # 5 minutes after exit before re-entry
      crisis_override: true     # CRISIS_OVERRIDE bypasses min_hold

State is held in-memory (resets on executor restart — intentional).
All vetoes are logged to doctrine_events.jsonl.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

LOG = logging.getLogger("churn_guard")

# ── Defaults (overridden by runtime.yaml) ──────────────────────────────────
DEFAULT_MIN_HOLD_SECONDS: float = 120.0    # 2-minute minimum hold
DEFAULT_COOLDOWN_SECONDS: float = 300.0    # 5-minute post-exit cooldown
DEFAULT_CRISIS_OVERRIDE: bool = True       # CRISIS/SEATBELT bypasses hold

# Exit reasons that bypass min_hold (catastrophe + regime safety).
# Includes both raw doctrine_kernel enum values AND canonical exit_reason_map
# names so bypass works regardless of which layer supplies the reason string.
# See config/exit_reason_map.yaml for the raw→canonical mapping.
HOLD_BYPASS_REASONS = frozenset({
    "CRISIS_OVERRIDE",             # raw (doctrine_kernel ExitReason)
    "CRISIS",                      # canonical (exit_reason_map)
    "SEATBELT",                    # raw + canonical (same in both layers)
    "STOP_LOSS_SEATBELT",          # raw (doctrine_kernel ExitReason)
    "REGIME_FLIP",                 # raw (doctrine_kernel ExitReason)
    "REGIME_CHANGE",               # canonical (exit_reason_map)
    "REGIME_CONFIDENCE_COLLAPSE",  # raw (doctrine_kernel ExitReason)
})


@dataclass
class ChurnConfig:
    """Churn guard configuration — loaded from runtime.yaml."""

    min_hold_seconds: float = DEFAULT_MIN_HOLD_SECONDS
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS
    crisis_override: bool = DEFAULT_CRISIS_OVERRIDE


@dataclass
class ChurnState:
    """In-memory churn tracking state per symbol+side."""

    # {(symbol, side): entry_ts} — when position was entered
    entry_times: Dict[str, float] = field(default_factory=dict)
    # {(symbol, side): exit_ts} — when position was last exited
    exit_times: Dict[str, float] = field(default_factory=dict)


# ── Module-level singleton (reset on executor restart) ─────────────────────
_state = ChurnState()


def _key(symbol: str, side: str) -> str:
    """Canonical key for symbol+side tracking."""
    return f"{str(symbol).upper()}_{str(side).upper()}"


def load_churn_config(runtime_cfg: Optional[Dict[str, Any]] = None) -> ChurnConfig:
    """Load churn guard config from runtime.yaml or dict.

    Reads from ``runtime_cfg["churn_guard"]`` if present,
    otherwise falls back to defaults.
    """
    if runtime_cfg is None:
        try:
            import yaml
            from pathlib import Path

            path = Path("config/runtime.yaml")
            if path.exists():
                with open(path, "r") as fh:
                    runtime_cfg = yaml.safe_load(fh) or {}
        except Exception:
            runtime_cfg = {}

    cg = runtime_cfg.get("churn_guard") if runtime_cfg else None
    if not cg or not isinstance(cg, dict):
        return ChurnConfig()

    return ChurnConfig(
        min_hold_seconds=float(cg.get("min_hold_seconds", DEFAULT_MIN_HOLD_SECONDS)),
        cooldown_seconds=float(cg.get("cooldown_seconds", DEFAULT_COOLDOWN_SECONDS)),
        crisis_override=bool(cg.get("crisis_override", DEFAULT_CRISIS_OVERRIDE)),
    )


def record_entry(symbol: str, side: str, ts: Optional[float] = None) -> None:
    """Record that a position was entered (for min_hold tracking)."""
    k = _key(symbol, side)
    _state.entry_times[k] = ts if ts is not None else time.time()
    LOG.debug("[churn_guard] entry recorded: %s at %.1f", k, _state.entry_times[k])


def record_exit(symbol: str, side: str, ts: Optional[float] = None) -> None:
    """Record that a position was exited (for cooldown tracking)."""
    k = _key(symbol, side)
    _state.exit_times[k] = ts if ts is not None else time.time()
    # Clear entry time — position is now closed
    _state.entry_times.pop(k, None)
    LOG.debug("[churn_guard] exit recorded: %s at %.1f", k, _state.exit_times[k])


def check_exit_allowed(
    symbol: str,
    side: str,
    exit_reason: str = "",
    now: Optional[float] = None,
    config: Optional[ChurnConfig] = None,
) -> tuple[bool, str]:
    """Check if an exit is allowed by the min_hold gate.

    Returns ``(allowed, reason)``.  If ``allowed`` is False, the exit
    should be vetoed and the reason logged to doctrine_events.

    CRISIS_OVERRIDE and SEATBELT exits always bypass this gate when
    ``config.crisis_override`` is True (default).
    """
    if config is None:
        config = load_churn_config()

    ts = now if now is not None else time.time()
    reason_upper = str(exit_reason).upper()

    # Catastrophe protection always passes
    if config.crisis_override and reason_upper in HOLD_BYPASS_REASONS:
        return True, ""

    k = _key(symbol, side)
    entry_ts = _state.entry_times.get(k)

    if entry_ts is None:
        # No recorded entry — allow (position may predate this executor session)
        return True, ""

    held_seconds = ts - entry_ts
    if held_seconds < config.min_hold_seconds:
        reason = (
            f"min_hold_veto: {symbol} {side} held {held_seconds:.0f}s "
            f"< min_hold {config.min_hold_seconds:.0f}s "
            f"(exit_reason={exit_reason})"
        )
        LOG.info("[churn_guard] %s", reason)
        return False, reason

    return True, ""


def check_entry_allowed(
    symbol: str,
    side: str,
    now: Optional[float] = None,
    config: Optional[ChurnConfig] = None,
) -> tuple[bool, str]:
    """Check if a new entry is allowed by the cooldown gate.

    Returns ``(allowed, reason)``.  If ``allowed`` is False, the entry
    should be vetoed and the reason logged.
    """
    if config is None:
        config = load_churn_config()

    ts = now if now is not None else time.time()
    k = _key(symbol, side)
    exit_ts = _state.exit_times.get(k)

    if exit_ts is None:
        return True, ""

    elapsed = ts - exit_ts
    if elapsed < config.cooldown_seconds:
        remaining = config.cooldown_seconds - elapsed
        reason = (
            f"cooldown_veto: {symbol} {side} exited {elapsed:.0f}s ago, "
            f"cooldown {config.cooldown_seconds:.0f}s ({remaining:.0f}s remaining)"
        )
        LOG.info("[churn_guard] %s", reason)
        return False, reason

    # Cooldown expired — clear state
    _state.exit_times.pop(k, None)
    return True, ""


def bootstrap_from_positions(
    positions: list,
    fill_log_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Seed churn state from existing exchange positions on startup.

    For every open position, sets an ``entry_time`` so that the min-hold
    gate is active immediately — converting the restart fail-open into
    fail-closed.

    Strategy:
      1. If *fill_log_path* is given, scan for the **last entry fill** for
         each symbol+side and use its timestamp.
      2. Otherwise, fall back to ``now()`` (conservative — min-hold starts
         from restart time).

    Returns a summary dict for logging.
    """
    import datetime as _dt

    now = time.time()
    seeded: Dict[str, float] = {}

    # ── Optional: build last-fill map from execution log ───────────────
    last_fill_ts: Dict[str, float] = {}
    if fill_log_path:
        try:
            import json as _json
            from pathlib import Path as _Path

            p = _Path(fill_log_path)
            if p.exists():
                with open(p, "r") as fh:
                    for raw in fh:
                        try:
                            rec = _json.loads(raw)
                        except Exception:
                            continue
                        if rec.get("event_type") != "order_fill":
                            continue
                        qty = float(rec.get("executedQty") or rec.get("qty") or 0)
                        if qty <= 0:
                            continue
                        # Only consider entries (not reduce_only)
                        if rec.get("reduceOnly"):
                            continue
                        sym = str(rec.get("symbol", "")).upper()
                        ps = str(rec.get("positionSide", "")).upper()
                        if not sym or not ps:
                            continue
                        # Parse timestamp
                        ts_str = rec.get("ts_fill_first") or rec.get("ts") or ""
                        try:
                            ts_val = _dt.datetime.fromisoformat(str(ts_str)).timestamp()
                        except Exception:
                            ts_val = 0.0
                        k = _key(sym, ps)
                        if ts_val > last_fill_ts.get(k, 0.0):
                            last_fill_ts[k] = ts_val
        except Exception as exc:
            LOG.warning("[churn_guard] bootstrap fill scan failed: %s", exc)

    # ── Seed entry_times for every active position ─────────────────────
    for pos in positions or []:
        sym = str(pos.get("symbol", "")).upper()
        ps = str(pos.get("positionSide", "")).upper()
        qty = float(pos.get("positionAmt") or pos.get("qty") or 0)
        if not sym or abs(qty) == 0:
            continue
        # Derive positionSide from qty sign if ps is BOTH/missing
        if ps in ("", "BOTH"):
            ps = "LONG" if qty > 0 else "SHORT"

        k = _key(sym, ps)
        if k in _state.entry_times:
            continue  # Already seeded (e.g. by a prior call)

        ts = last_fill_ts.get(k) or now
        _state.entry_times[k] = ts
        seeded[k] = ts
        LOG.info(
            "[churn_guard] bootstrap: %s entry_time=%.1f (age=%.0fs, source=%s)",
            k,
            ts,
            now - ts,
            "fill_log" if k in last_fill_ts else "now",
        )

    return {
        "positions_seen": len(positions or []),
        "seeded": len(seeded),
        "keys": list(seeded.keys()),
    }


def get_state_snapshot() -> Dict[str, Any]:
    """Return a read-only snapshot of churn guard state (for diagnostics)."""
    now = time.time()
    entries = {}
    for k, ts in _state.entry_times.items():
        entries[k] = {"entry_ts": ts, "held_seconds": round(now - ts, 1)}

    cooldowns = {}
    for k, ts in _state.exit_times.items():
        cooldowns[k] = {"exit_ts": ts, "elapsed_seconds": round(now - ts, 1)}

    return {
        "active_entries": entries,
        "active_cooldowns": cooldowns,
        "ts": now,
    }


def reset_state() -> None:
    """Reset all in-memory state. Used by tests and on executor restart."""
    _state.entry_times.clear()
    _state.exit_times.clear()
