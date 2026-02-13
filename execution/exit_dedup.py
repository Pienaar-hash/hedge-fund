"""
Exit Intent Dedup — Prevent redundant exit order spam (v7.9-E2.2)

The exit scanner fires on every loop cycle (~60s). Without dedup,
the same REGIME_FLIP exit fires every iteration for a position that
hasn't fully wound down yet, producing dozens of redundant fills.

This module keeps an in-memory map of recent exit intents and blocks
re-sends within a configurable TTL unless bypass conditions are met.

Key: ``(symbol, positionSide, exit_reason)``
TTL: ``exit_dedup_ttl_seconds`` (default 300s)

Bypass conditions:
  1. exit_reason is CRISIS_OVERRIDE or SEATBELT (catastrophe protection)
  2. Position qty grew materially since last send (>10%)
  3. TTL has expired

State is in-memory only — resets on executor restart (safe).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

LOG = logging.getLogger("exit_dedup")

# ── Defaults ───────────────────────────────────────────────────────────────
DEFAULT_EXIT_DEDUP_TTL: float = 300.0      # 5 minutes
DEFAULT_QTY_CHANGE_THRESHOLD: float = 0.10  # 10% position size change

# Exit reasons that always bypass dedup (catastrophe protection)
DEDUP_BYPASS_REASONS = frozenset({"CRISIS_OVERRIDE", "SEATBELT"})


@dataclass
class ExitDedupConfig:
    """Configuration for exit intent dedup."""

    ttl_seconds: float = DEFAULT_EXIT_DEDUP_TTL
    qty_change_threshold: float = DEFAULT_QTY_CHANGE_THRESHOLD
    enabled: bool = True


@dataclass
class _ExitRecord:
    """Record of a recently sent exit intent."""

    last_sent_ts: float
    last_qty: float


# ── Module state ───────────────────────────────────────────────────────────
_sent: Dict[Tuple[str, str, str], _ExitRecord] = {}


def _key(symbol: str, pos_side: str, exit_reason: str) -> Tuple[str, str, str]:
    return (
        str(symbol).upper(),
        str(pos_side).upper(),
        str(exit_reason).upper(),
    )


def load_exit_dedup_config(
    runtime_cfg: Optional[Dict[str, Any]] = None,
) -> ExitDedupConfig:
    """Load exit dedup config from runtime.yaml or dict."""
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

    sec = runtime_cfg.get("exit_dedup") if runtime_cfg else None
    if not sec or not isinstance(sec, dict):
        return ExitDedupConfig()

    return ExitDedupConfig(
        ttl_seconds=float(sec.get("ttl_seconds", DEFAULT_EXIT_DEDUP_TTL)),
        qty_change_threshold=float(
            sec.get("qty_change_threshold", DEFAULT_QTY_CHANGE_THRESHOLD)
        ),
        enabled=bool(sec.get("enabled", True)),
    )


def check_exit_dedup(
    symbol: str,
    pos_side: str,
    exit_reason: str,
    current_qty: float = 0.0,
    now: Optional[float] = None,
    config: Optional[ExitDedupConfig] = None,
) -> Tuple[bool, str]:
    """Check if an exit intent should be sent or suppressed.

    Returns ``(allowed, reason)``.  If ``allowed`` is False, the exit
    intent should be suppressed (not sent to exchange).
    """
    if config is None:
        config = load_exit_dedup_config()

    if not config.enabled:
        return True, ""

    ts = now if now is not None else time.time()
    reason_upper = str(exit_reason).upper()

    # Catastrophe protection always passes
    if reason_upper in DEDUP_BYPASS_REASONS:
        return True, ""

    k = _key(symbol, pos_side, exit_reason)
    prev = _sent.get(k)

    if prev is None:
        # First time seeing this exit — allow
        return True, ""

    elapsed = ts - prev.last_sent_ts
    if elapsed >= config.ttl_seconds:
        # TTL expired — allow
        return True, ""

    # Check if position grew materially
    if current_qty > 0 and prev.last_qty > 0:
        change_pct = abs(current_qty - prev.last_qty) / prev.last_qty
        if change_pct > config.qty_change_threshold:
            return True, ""

    reason = (
        f"exit_dedup: {symbol} {pos_side} {exit_reason} "
        f"sent {elapsed:.0f}s ago, TTL={config.ttl_seconds:.0f}s "
        f"({config.ttl_seconds - elapsed:.0f}s remaining)"
    )
    LOG.info("[exit_dedup] %s", reason)
    return False, reason


def record_exit_sent(
    symbol: str,
    pos_side: str,
    exit_reason: str,
    qty: float = 0.0,
    now: Optional[float] = None,
) -> None:
    """Record that an exit intent was sent to exchange."""
    ts = now if now is not None else time.time()
    k = _key(symbol, pos_side, exit_reason)
    _sent[k] = _ExitRecord(last_sent_ts=ts, last_qty=abs(qty))
    LOG.debug("[exit_dedup] recorded: %s qty=%.6f", k, abs(qty))


def clear_for_symbol(symbol: str) -> None:
    """Clear all dedup records for a symbol (e.g. after full close)."""
    sym = str(symbol).upper()
    to_remove = [k for k in _sent if k[0] == sym]
    for k in to_remove:
        del _sent[k]


def get_state_snapshot() -> Dict[str, Any]:
    """Return a read-only snapshot for diagnostics."""
    now = time.time()
    records = {}
    for (sym, ps, reason), rec in _sent.items():
        k = f"{sym}_{ps}_{reason}"
        records[k] = {
            "last_sent_ts": rec.last_sent_ts,
            "last_qty": rec.last_qty,
            "elapsed_s": round(now - rec.last_sent_ts, 1),
        }
    return {"active_dedup_entries": records, "ts": now}


def reset_state() -> None:
    """Reset all state. Used by tests."""
    _sent.clear()
