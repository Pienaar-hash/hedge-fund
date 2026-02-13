"""Testnet Reset Guard — detect and handle Binance testnet account resets.

Binance Futures Testnet performs periodic full account resets that wipe
positions, balances, and fill history.  This module detects those resets
and enforces the protocol defined in ``docs/TESTNET_RESET_PROTOCOL.md``.

Position in the executor loop::

    fetch_exchange_state()
      ↓
    **reset_guard.check_for_testnet_reset()**   ← HERE
      ↓
    nav_update()
      ↓
    signal_generation() → execution

Design constraints:

* **Fail-closed** — if detection is ambiguous, halt the cycle.
* **Never triggers in production** — hard ``env`` guard.
* **Never modifies historical logs** — only archives and creates new state.
* **Debounced** — will not re-trigger within ``RESET_DEBOUNCE_S`` of last reset.

Logging targets:

* ``logs/execution/environment_events.jsonl`` — append-only reset events
* ``logs/state/environment_meta.json`` — current cycle watermark (dashboard-readable)
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from execution.log_utils import get_logger, log_event

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Binance Futures Testnet resets wallet to this value.
DEFAULT_TESTNET_BALANCE: float = float(
    os.getenv("TESTNET_DEFAULT_BALANCE", "10000.0")
)

# Balance must be within this tolerance of the default to count as "reset balance".
BALANCE_MATCH_TOLERANCE: float = float(
    os.getenv("TESTNET_BALANCE_TOLERANCE", "1.0")
)

# Delta between exchange balance and last logged NAV must exceed this to trigger.
RESET_DELTA_THRESHOLD: float = float(
    os.getenv("TESTNET_RESET_THRESHOLD", "0.50")
)  # 50% relative change

# Minimum seconds between consecutive reset triggers (debounce).
RESET_DEBOUNCE_S: float = float(
    os.getenv("TESTNET_RESET_DEBOUNCE_S", "300")
)

# Paths
_ENV_EVENTS_LOG = "logs/execution/environment_events.jsonl"
_ENV_META_PATH = Path("logs/state/environment_meta.json")
_NAV_STATE_PATH = Path("logs/state/nav_state.json")
_EPISODE_LEDGER_PATH = Path("logs/state/episode_ledger.json")
_POSITIONS_STATE_PATH = Path("logs/state/positions_state.json")
_ARCHIVE_BASE = Path("archive")

_env_logger = get_logger(_ENV_EVENTS_LOG)

# Module-level debounce timestamp.
_last_reset_ts: float = 0.0


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ResetResult:
    """Outcome of a testnet reset check."""

    triggered: bool = False
    reason: Optional[str] = None
    pre_reset_nav: Optional[float] = None
    post_reset_balance: Optional[float] = None
    cycle_id: Optional[str] = None
    archive_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_last_logged_nav() -> float:
    """Read the most recent NAV from the persisted nav_state.json.

    Returns 0.0 if the file is missing, empty, or unparseable.
    """
    try:
        if not _NAV_STATE_PATH.exists():
            return 0.0
        with open(_NAV_STATE_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # Try total_equity first (most reliable), fall back to nav
        for key in ("total_equity", "nav_usd", "nav"):
            val = data.get(key)
            if val is not None:
                fval = float(val)
                if fval > 0:
                    return fval
        # Try last series entry
        series = data.get("series")
        if isinstance(series, list) and series:
            last = series[-1]
            if isinstance(last, dict):
                for key in ("total_equity", "nav", "equity"):
                    val = last.get(key)
                    if val is not None:
                        fval = float(val)
                        if fval > 0:
                            return fval
        return 0.0
    except Exception:
        return 0.0


def _read_env_meta() -> Dict[str, Any]:
    """Load the current environment_meta.json, or empty dict on failure."""
    try:
        if _ENV_META_PATH.exists():
            with open(_ENV_META_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _next_cycle_id() -> str:
    """Generate the next CYCLE_TEST_NNN identifier."""
    meta = _read_env_meta()
    current = meta.get("cycle_id", "")
    # Parse existing cycle number
    if current.startswith("CYCLE_TEST_"):
        try:
            num = int(current.split("_")[-1])
            return f"CYCLE_TEST_{num + 1:03d}"
        except (ValueError, IndexError):
            pass
    return "CYCLE_TEST_001"


def _archive_state(cycle_id: str) -> str:
    """Archive current state files into a timestamped directory.

    Returns the archive path as a string.
    """
    ts_label = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    archive_dir = _ARCHIVE_BASE / f"{cycle_id}_{ts_label}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    for src in (_NAV_STATE_PATH, _EPISODE_LEDGER_PATH, _POSITIONS_STATE_PATH):
        if src.exists():
            try:
                shutil.copy2(str(src), str(archive_dir / src.name))
            except Exception:
                pass  # Best-effort; do not block on archive failure
    return str(archive_dir)


def _write_env_meta(cycle_id: str, reset_ts: float) -> None:
    """Write the environment watermark for dashboard readability."""
    meta = _read_env_meta()
    meta["cycle_id"] = cycle_id
    meta["last_testnet_reset_ts"] = reset_ts
    meta["last_testnet_reset_iso"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(reset_ts)
    )
    meta["updated_ts"] = time.time()

    _ENV_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _ENV_META_PATH.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, sort_keys=True)
        tmp.replace(_ENV_META_PATH)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def _balance_matches_default(balance: float) -> bool:
    """True if balance is within tolerance of the known testnet default."""
    return abs(balance - DEFAULT_TESTNET_BALANCE) <= BALANCE_MATCH_TOLERANCE


def _delta_exceeds_threshold(balance: float, last_nav: float) -> bool:
    """True if the relative jump from last_nav to balance exceeds the threshold."""
    if last_nav <= 0:
        return False
    return abs(balance - last_nav) / last_nav > RESET_DELTA_THRESHOLD


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_for_testnet_reset(
    *,
    exchange_balance: float,
    exchange_positions: Sequence[Any],
    env: str,
    last_logged_nav: Optional[float] = None,
) -> ResetResult:
    """Check whether a testnet reset has occurred and handle it if so.

    Parameters
    ----------
    exchange_balance : float
        Total wallet balance returned by the exchange.
    exchange_positions : sequence
        List of open positions from the exchange.
    env : str
        Environment identifier.  Must contain ``"testnet"`` (case-insensitive)
        for the guard to evaluate.  Any other value results in an immediate
        no-trigger return (or CRITICAL if a balance anomaly is detected).
    last_logged_nav : float or None
        Override for the last persisted NAV.  If *None*, reads from
        ``logs/state/nav_state.json``.

    Returns
    -------
    ResetResult
        ``.triggered`` is True if a reset was detected **and** handled.
    """
    global _last_reset_ts

    is_testnet = "testnet" in str(env).lower()

    # ── PRODUCTION GUARD ──────────────────────────────────────────────
    # In production, a sudden balance jump is a CRITICAL incident.
    if not is_testnet:
        if last_logged_nav is None:
            last_logged_nav = _read_last_logged_nav()
        if (
            last_logged_nav > 0
            and exchange_balance > 0
            and _delta_exceeds_threshold(exchange_balance, last_logged_nav)
        ):
            log_event(_env_logger, "BALANCE_ANOMALY_PRODUCTION", {
                "exchange_balance": exchange_balance,
                "last_logged_nav": last_logged_nav,
                "delta_pct": round(
                    abs(exchange_balance - last_logged_nav) / last_logged_nav, 4
                ),
                "action": "CRITICAL_LOGGED",
            })
        return ResetResult(triggered=False, reason="not_testnet")

    # ── TESTNET EVALUATION ────────────────────────────────────────────

    if last_logged_nav is None:
        last_logged_nav = _read_last_logged_nav()

    # Condition 1: Balance matches known testnet default
    if not _balance_matches_default(exchange_balance):
        return ResetResult(triggered=False, reason="balance_not_default")

    # Condition 2: No open positions
    positions = list(exchange_positions) if exchange_positions else []
    active = [
        p for p in positions
        if isinstance(p, dict)
        and float(p.get("positionAmt", p.get("qty", 0)) or 0) != 0
    ]
    if active:
        return ResetResult(triggered=False, reason="positions_open")

    # Condition 3: Prior NAV was meaningfully different from default
    if last_logged_nav <= 0:
        return ResetResult(triggered=False, reason="no_prior_nav")

    if not _delta_exceeds_threshold(exchange_balance, last_logged_nav):
        return ResetResult(triggered=False, reason="delta_below_threshold")

    # Condition 4: Debounce — don't re-fire within window
    now = time.time()
    if _last_reset_ts > 0 and (now - _last_reset_ts) < RESET_DEBOUNCE_S:
        return ResetResult(triggered=False, reason="debounce")

    # ── ALL CONDITIONS MET: TRIGGER RESET PROTOCOL ────────────────────

    _last_reset_ts = now
    new_cycle_id = _next_cycle_id()
    archive_path = _archive_state(new_cycle_id)

    # Log the immutable reset event
    event_payload = {
        "event": "TESTNET_RESET",
        "environment": "binance_futures_testnet",
        "pre_reset_nav": round(last_logged_nav, 4),
        "post_reset_balance": round(exchange_balance, 4),
        "cycle_id": new_cycle_id,
        "archive_path": archive_path,
    }
    log_event(_env_logger, "TESTNET_RESET", event_payload)

    # Write environment watermark
    try:
        _write_env_meta(new_cycle_id, now)
    except Exception:
        pass  # Watermark is advisory; failure must not block halt

    return ResetResult(
        triggered=True,
        reason="testnet_reset_detected",
        pre_reset_nav=last_logged_nav,
        post_reset_balance=exchange_balance,
        cycle_id=new_cycle_id,
        archive_path=archive_path,
    )


def get_current_cycle_id() -> str:
    """Return the current test cycle ID from environment_meta, or 'CYCLE_TEST_001'."""
    meta = _read_env_meta()
    return meta.get("cycle_id", "CYCLE_TEST_001")


def reset_debounce_state() -> None:
    """Clear the internal debounce timer.  For testing only."""
    global _last_reset_ts
    _last_reset_ts = 0.0
