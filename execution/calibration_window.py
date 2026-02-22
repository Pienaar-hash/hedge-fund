"""
Calibration Window — Episode-Capped Trading Gate (v7.9-CW)
==========================================================

Enforces a bounded calibration window: the executor may trade until
``episode_cap`` new episodes are completed, then automatically halts
via KILL_SWITCH.

This module is **observability + enforcement**:
  - Reads ``calibration_window`` section from ``config/runtime.yaml``
  - Counts episodes created since ``start_ts``
  - When cap is reached, sets ``os.environ["KILL_SWITCH"] = "1"``
  - Logs halt event to ``logs/doctrine_events.jsonl``

Design constraints:
  - Fail-closed: if config missing or malformed, window is INACTIVE (no trades)
  - Uses existing KILL_SWITCH mechanism (no executor modification)
  - Episode count derived from episode_ledger (authoritative)
  - Idempotent: multiple calls after cap produce no duplicate logs

Activation (dual-key):
  1. Add ``calibration_window`` section to ``config/runtime.yaml`` with ``enabled: true``
  2. Set ``CALIBRATION_WINDOW_ACK=1`` in supervisor environment
  3. Set ``DRY_RUN=0`` in supervisor environment
  4. Restart executor

Both keys are required.  Config alone cannot activate the window.

Deactivation:
  - Remove ``calibration_window`` section or set ``enabled: false``
  - Or: unset ``CALIBRATION_WINDOW_ACK``
  - Or: ``KILL_SWITCH=1`` fires automatically when cap is reached
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

RUNTIME_YAML = Path("config/runtime.yaml")
DOCTRINE_LOG = Path("logs/doctrine_events.jsonl")
EPISODE_LEDGER_PATH = Path("logs/state/episode_ledger.json")
CALIBRATION_ACK_ENV = "CALIBRATION_WINDOW_ACK"

# Sentinel: once we fire the kill switch, don't log again
_kill_switch_fired: bool = False
# Throttle: emit periodic status at most every N calls
_STATUS_LOG_INTERVAL: int = 60  # every ~60 loop iterations
_status_log_counter: int = 0


def _load_calibration_config() -> Optional[Dict[str, Any]]:
    """Load calibration_window section from runtime.yaml.

    Returns None if section is missing or ``enabled`` is false.
    """
    try:
        with open(RUNTIME_YAML) as f:
            cfg = yaml.safe_load(f) or {}
        cw = cfg.get("calibration_window")
        if not isinstance(cw, dict):
            return None
        if not cw.get("enabled", False):
            return None
        # Dual-key: env ACK must also be set to prevent accidental activation
        ack = os.environ.get(CALIBRATION_ACK_ENV, "0").strip().lower()
        if ack not in ("1", "true", "yes", "on"):
            logger.debug(
                "[calibration_window] config enabled but %s not set — inactive",
                CALIBRATION_ACK_ENV,
            )
            return None
        return cw
    except Exception as exc:
        logger.warning("[calibration_window] failed to load config: %s", exc)
        return None


def _parse_iso_ts(ts_str: str) -> Optional[datetime]:
    """Parse ISO 8601 timestamp string to datetime (UTC).

    Handles:
      - ``2026-02-22T00:00:00Z``
      - ``2026-02-22T00:00:00+00:00``
      - ``2026-02-22T00:00:00.123456+00:00``

    Returns None on parse failure.
    """
    if not ts_str:
        return None
    try:
        # Python 3.7+ fromisoformat handles +00:00 but not Z
        cleaned = ts_str.replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned)
    except (ValueError, TypeError):
        return None


def _count_episodes_since(start_ts: str) -> int:
    """Count completed episodes with exit_ts >= start_ts.

    Uses proper datetime comparison (not lexicographic) to handle
    mixed ISO formats (Z vs +00:00, fractional seconds).
    """
    start_dt = _parse_iso_ts(start_ts)
    if start_dt is None:
        logger.warning("[calibration_window] invalid start_ts: %r", start_ts)
        return 0
    try:
        with open(EPISODE_LEDGER_PATH) as f:
            data = json.load(f)
        episodes = data.get("episodes", [])
        count = 0
        for ep in episodes:
            exit_dt = _parse_iso_ts(ep.get("exit_ts", ""))
            if exit_dt is not None and exit_dt >= start_dt:
                count += 1
        return count
    except FileNotFoundError:
        return 0
    except Exception as exc:
        logger.warning("[calibration_window] episode count failed: %s", exc)
        return 0


def _log_doctrine_event(event: Dict[str, Any]) -> None:
    """Append event to doctrine_events.jsonl."""
    try:
        DOCTRINE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(DOCTRINE_LOG, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as exc:
        logger.error("[calibration_window] doctrine log write failed: %s", exc)


def _get_portfolio_dd_pct() -> float:
    """Read current portfolio drawdown percentage from risk state.

    Returns a *positive* float representing percentage drawdown (e.g. 0.03 = 3%).
    Returns 0.0 on any error (fail-open for drawdown reads; fail-closed is handled
    by the episode cap and KILL_SWITCH).
    """
    try:
        from execution.risk_limits import drawdown_snapshot
        snap = drawdown_snapshot()
        dd_info = snap.get("drawdown") if isinstance(snap, dict) else {}
        dd_pct = (dd_info.get("pct") if isinstance(dd_info, dict) else None)
        if dd_pct is None:
            dd_pct = snap.get("dd_pct", 0.0) if isinstance(snap, dict) else 0.0
        return abs(float(dd_pct or 0.0))
    except Exception as exc:
        logger.debug("[calibration_window] drawdown read failed: %s", exc)
        return 0.0


def check_calibration_window() -> Dict[str, Any]:
    """Check calibration window state.  Called from executor loop.

    Returns a status dict (for telemetry/logging):
      - active: bool — whether a calibration window is configured
      - episode_cap: int
      - episodes_completed: int
      - episodes_remaining: int
      - halted: bool — whether KILL_SWITCH was triggered
      - drawdown_kill_pct: float
      - start_ts: str

    Side effects:
      - Sets ``os.environ["KILL_SWITCH"] = "1"`` when cap reached
      - Logs doctrine event on first halt
    """
    global _kill_switch_fired

    cw = _load_calibration_config()
    if cw is None:
        return {"active": False}

    episode_cap = int(cw.get("episode_cap", 30))
    start_ts = str(cw.get("start_ts", ""))
    drawdown_kill_pct = float(cw.get("drawdown_kill_pct", 0.05))

    if not start_ts:
        logger.warning("[calibration_window] no start_ts configured — inactive")
        return {"active": False}

    episodes_completed = _count_episodes_since(start_ts)
    episodes_remaining = max(0, episode_cap - episodes_completed)
    halted = episodes_completed >= episode_cap

    # --- Drawdown kill check ---
    dd_pct = _get_portfolio_dd_pct()
    dd_breached = drawdown_kill_pct > 0 and dd_pct >= drawdown_kill_pct
    if dd_breached:
        halted = True

    status = {
        "active": True,
        "episode_cap": episode_cap,
        "episodes_completed": episodes_completed,
        "episodes_remaining": episodes_remaining,
        "halted": halted,
        "drawdown_kill_pct": drawdown_kill_pct,
        "current_dd_pct": dd_pct,
        "dd_breached": dd_breached,
        "start_ts": start_ts,
    }

    if halted and not _kill_switch_fired:
        os.environ["KILL_SWITCH"] = "1"
        _kill_switch_fired = True
        now = datetime.now(timezone.utc).isoformat()
        halt_reason = (
            f"drawdown {dd_pct:.4f} >= kill threshold {drawdown_kill_pct:.4f}"
            if dd_breached
            else f"episode cap {episodes_completed}/{episode_cap} reached"
        )
        _log_doctrine_event({
            "ts": now,
            "event": "CALIBRATION_WINDOW_HALT",
            "source": "calibration_window",
            "episode_cap": episode_cap,
            "episodes_completed": episodes_completed,
            "dd_pct": dd_pct,
            "drawdown_kill_pct": drawdown_kill_pct,
            "dd_breached": dd_breached,
            "start_ts": start_ts,
            "action": f"KILL_SWITCH activated — {halt_reason}",
        })
        logger.info(
            "[calibration_window] HALT — %s. KILL_SWITCH=1.",
            halt_reason,
        )

    # --- Periodic status emission (throttled) ---
    global _status_log_counter
    _status_log_counter += 1
    if _status_log_counter >= _STATUS_LOG_INTERVAL:
        _status_log_counter = 0
        logger.info(
            "[calibration_window] status: %d/%d episodes, dd=%.4f/%.4f, halted=%s",
            episodes_completed,
            episode_cap,
            dd_pct,
            drawdown_kill_pct,
            halted,
        )

    return status


def log_calibration_boot_status() -> None:
    """Log calibration window state at executor startup.

    Emits a single INFO line with the full config snapshot so operators
    can confirm the window is ACTIVE or INACTIVE from the first log line.
    """
    cw = _load_calibration_config()
    if cw is None:
        ack = os.environ.get(CALIBRATION_ACK_ENV, "0")
        # Distinguish between "off" and "config-without-ack"
        try:
            with open(RUNTIME_YAML) as f:
                raw = yaml.safe_load(f) or {}
            raw_cw = raw.get("calibration_window", {})
            if isinstance(raw_cw, dict) and raw_cw.get("enabled"):
                logger.warning(
                    "[calibration_window] BOOT: config enabled=true but %s=%s — INACTIVE (dual-key missing)",
                    CALIBRATION_ACK_ENV, ack,
                )
                return
        except Exception:
            pass
        logger.info("[calibration_window] BOOT: INACTIVE (disabled or missing config)")
        return

    episode_cap = int(cw.get("episode_cap", 30))
    start_ts = str(cw.get("start_ts", ""))
    dd_kill = float(cw.get("drawdown_kill_pct", 0.05))
    sizing = cw.get("per_trade_nav_pct")

    # Compute sizing cap in USD for the log line
    sizing_usd_str = "N/A"
    if sizing is not None:
        try:
            from execution.nav import nav_health_snapshot
            snap = nav_health_snapshot()
            nav_usd = float((snap or {}).get("nav_usd", 0.0) or 0.0)
            if nav_usd > 0:
                sizing_usd_str = f"${float(sizing) * nav_usd:.2f}"
        except Exception:
            pass

    logger.info(
        "[calibration_window] BOOT: ACTIVE — cap=%d, start=%s, dd_kill=%.4f, "
        "sizing=%.4f (%s), ack=%s",
        episode_cap,
        start_ts or "(not set)",
        dd_kill,
        float(sizing or 0),
        sizing_usd_str,
        os.environ.get(CALIBRATION_ACK_ENV, "0"),
    )


def get_calibration_sizing_override() -> Optional[float]:
    """Return per_trade_nav_pct override if calibration window is active.

    Returns None if no calibration window, or the configured
    ``per_trade_nav_pct`` override (typically reduced sizing).
    """
    cw = _load_calibration_config()
    if cw is None:
        return None
    return cw.get("per_trade_nav_pct")
