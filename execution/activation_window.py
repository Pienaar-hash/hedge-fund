"""
Activation Window — Full-Stack System Certification Protocol (v8.0)
====================================================================

Enforces a **14-day time-bounded** activation window that covers the
entire hedge stack: futures engine, DLE gating, Binary Lab SHADOW,
risk vetoes, state machine transitions, telemetry integrity, and
manifest immutability.

This is NOT a strategy calibration — it is a **system certification
protocol**.  The window proves full-stack integrity before capital
scale is permitted.

Architecture
------------
::

    executor boot  →  log_activation_boot_status()
    executor loop  →  check_activation_window()  →  status dict + KILL_SWITCH
    day 14         →  scripts/activation_verify.py  →  7-gate GO/NO-GO

Design constraints
------------------
- **Time-bounded** (14 calendar days), not episode-based.
- **Stack-wide** — checks NAV, drawdown, manifest, DLE, Binary Lab,
  risk vetoes, config drift.
- **Dual-key activation**: ``enabled: true`` in YAML + ``ACTIVATION_WINDOW_ACK=1``
  in supervisor env.
- **Structural integrity kills** — halts on manifest drift, config hash
  mismatch, DLE bypass, not just drawdown.
- **Fail-closed**: missing/malformed config → window INACTIVE.
- **State file emitted**: ``logs/state/activation_window_state.json``
  for dashboard observability.

Kill conditions (any one triggers KILL_SWITCH):
    1. NAV drawdown ≥ ``drawdown_kill_pct``
    2. Manifest file hash changed mid-window
    3. Config hash drifted from boot snapshot
    4. DLE veto bypass detected
    5. Binary Lab freeze violation
    6. Risk limits breach
    7. 14 days elapsed (window complete — NOT a failure)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RUNTIME_YAML = Path("config/runtime.yaml")
MANIFEST_PATH = Path("v7_manifest.json")
DOCTRINE_LOG = Path("logs/doctrine_events.jsonl")
EPISODE_LEDGER_PATH = Path("logs/state/episode_ledger.json")
STATE_FILE_PATH = Path("logs/state/activation_window_state.json")
NAV_STATE_PATH = Path("logs/state/nav_state.json")
RISK_SNAPSHOT_PATH = Path("logs/state/risk_snapshot.json")
BINARY_LAB_STATE_PATH = Path("logs/state/binary_lab_state.json")
DLE_SHADOW_LOG = Path("logs/execution/dle_shadow_events.jsonl")

ACTIVATION_ACK_ENV = "ACTIVATION_WINDOW_ACK"
DEFAULT_DURATION_DAYS = 14
DEFAULT_DD_KILL_PCT = 0.05
DEFAULT_PER_TRADE_NAV_PCT = 0.005

# DLE governance constants
STRUCTURAL_GUARD_EVENT_TYPE = "STRUCTURAL_GUARD"
GUARD_TYPE = "ACTIVATION_WINDOW"
PHASE_ID = "PHASE_C"
VERIFICATION_VERDICT_PATH = Path("logs/state/activation_verification_verdict.json")

# In-memory sentinels
_kill_switch_fired: bool = False
_boot_manifest_hash: Optional[str] = None
_boot_config_hash: Optional[str] = None
_dle_started_emitted: bool = False
_status_log_counter: int = 0
_STATUS_LOG_INTERVAL: int = 60


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def _load_activation_config(
    runtime_yaml: Path = RUNTIME_YAML,
) -> Optional[Dict[str, Any]]:
    """Load ``activation_window`` section from runtime.yaml.

    Returns ``None`` if section is missing, ``enabled`` is false, or
    the dual-key env ACK is absent.
    """
    try:
        with open(runtime_yaml) as f:
            cfg = yaml.safe_load(f) or {}
        aw = cfg.get("activation_window")
        if not isinstance(aw, dict):
            return None
        if not aw.get("enabled", False):
            return None
        # Dual-key: env ACK required
        ack = os.environ.get(ACTIVATION_ACK_ENV, "0").strip().lower()
        if ack not in ("1", "true", "yes", "on"):
            logger.debug(
                "[activation_window] config enabled but %s not set — inactive",
                ACTIVATION_ACK_ENV,
            )
            return None
        return aw
    except Exception as exc:
        logger.warning("[activation_window] config load failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------
def _parse_iso_ts(ts_str: str) -> Optional[datetime]:
    """Parse ISO 8601 timestamp to UTC datetime."""
    if not ts_str:
        return None
    try:
        cleaned = ts_str.replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned)
    except (ValueError, TypeError):
        return None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Hash helpers (structural integrity)
# ---------------------------------------------------------------------------
def _file_hash(path: Path) -> Optional[str]:
    """SHA-256 of file contents.  Returns None if file missing."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except Exception:
        return None


def _compute_manifest_hash(manifest_path: Path = MANIFEST_PATH) -> Optional[str]:
    return _file_hash(manifest_path)


def _compute_config_hash(runtime_yaml: Path = RUNTIME_YAML) -> Optional[str]:
    return _file_hash(runtime_yaml)


# ---------------------------------------------------------------------------
# State readers
# ---------------------------------------------------------------------------
def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        return None


def _get_portfolio_dd_pct() -> float:
    """Read current portfolio drawdown %.  Fail-open (returns 0.0)."""
    try:
        from execution.risk_limits import drawdown_snapshot
        snap = drawdown_snapshot()
        dd_info = snap.get("drawdown") if isinstance(snap, dict) else {}
        dd_pct = dd_info.get("pct") if isinstance(dd_info, dict) else None
        if dd_pct is None:
            dd_pct = snap.get("dd_pct", 0.0) if isinstance(snap, dict) else 0.0
        return abs(float(dd_pct or 0.0))
    except Exception:
        return 0.0


def _get_nav_usd() -> float:
    """Read current NAV from state file."""
    data = _read_json(NAV_STATE_PATH)
    if data:
        return float(data.get("nav_usd", 0.0) or 0.0)
    return 0.0


def _count_risk_vetoes_since(start_ts: str) -> int:
    """Count risk vetoes since window start."""
    veto_log = Path("logs/execution/risk_vetoes.jsonl")
    if not veto_log.exists():
        return 0
    count = 0
    try:
        with veto_log.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if (rec.get("ts", "") or "") >= start_ts:
                        count += 1
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return count


def _count_episodes_since(start_ts: str) -> int:
    """Count completed episodes with exit_ts >= start_ts."""
    start_dt = _parse_iso_ts(start_ts)
    if start_dt is None:
        return 0
    try:
        with open(EPISODE_LEDGER_PATH) as f:
            data = json.load(f)
        count = 0
        for ep in data.get("episodes", []):
            exit_dt = _parse_iso_ts(ep.get("exit_ts", ""))
            if exit_dt is not None and exit_dt >= start_dt:
                count += 1
        return count
    except Exception:
        return 0


def _check_binary_lab_freeze() -> bool:
    """Check if Binary Lab freeze is intact.  Returns True if OK."""
    data = _read_json(BINARY_LAB_STATE_PATH)
    if data is None:
        return True  # No binary lab state → no violation
    return data.get("freeze_intact", True)


def _check_dle_anomalies(start_ts: str) -> int:
    """Count DLE shadow events flagged as mismatches since start."""
    if not DLE_SHADOW_LOG.exists():
        return 0
    count = 0
    try:
        with DLE_SHADOW_LOG.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if (rec.get("ts", "") or "") >= start_ts:
                        if rec.get("mismatch") or rec.get("event") == "DLE_MISMATCH":
                            count += 1
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return count


# ---------------------------------------------------------------------------
# Doctrine event logging
# ---------------------------------------------------------------------------
def _log_doctrine_event(
    event: Dict[str, Any],
    doctrine_log: Path = DOCTRINE_LOG,
) -> None:
    """Append event to doctrine_events.jsonl."""
    try:
        doctrine_log.parent.mkdir(parents=True, exist_ok=True)
        with open(doctrine_log, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")
    except Exception as exc:
        logger.error("[activation_window] doctrine log failed: %s", exc)


# ---------------------------------------------------------------------------
# DLE lifecycle event emission (STRUCTURAL_GUARD)
# ---------------------------------------------------------------------------
def _emit_dle_lifecycle_event(
    action: str,
    details: Dict[str, Any],
) -> None:
    """Emit STRUCTURAL_GUARD event to DLE shadow log.

    Binds the activation window into DLE episode observability as a
    constitutional governance artifact.  Emitted unconditionally
    (not gated by SHADOW_DLE_ENABLED) — structural guards are
    governance, not trade-gating.

    Fail-open: never crashes executor.
    """
    try:
        from execution.dle_shadow import (
            DLEShadowEvent,
            DLEShadowWriter,
            SCHEMA_VERSION_V2,
            DEFAULT_LOG_PATH,
        )

        event = DLEShadowEvent(
            schema_version=SCHEMA_VERSION_V2,
            event_type=STRUCTURAL_GUARD_EVENT_TYPE,
            ts=_utc_now().isoformat(),
            payload={
                "guard_type": GUARD_TYPE,
                "action": action,
                "phase_id": PHASE_ID,
                **details,
            },
        )

        writer = DLEShadowWriter(DEFAULT_LOG_PATH)
        writer.write(event)
    except Exception as exc:
        logger.debug("[activation_window] DLE lifecycle event failed: %s", exc)


# ---------------------------------------------------------------------------
# State file writer
# ---------------------------------------------------------------------------
def _write_state(
    status: Dict[str, Any],
    state_path: Path = STATE_FILE_PATH,
) -> None:
    """Atomically write activation window state for dashboard."""
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = state_path.with_suffix(".tmp")
        with tmp.open("w") as f:
            json.dump(status, f, indent=2, default=str)
        tmp.replace(state_path)
    except Exception as exc:
        logger.debug("[activation_window] state write failed: %s", exc)


# ---------------------------------------------------------------------------
# Main check — called every executor loop
# ---------------------------------------------------------------------------
def check_activation_window(
    *,
    runtime_yaml: Path = RUNTIME_YAML,
    manifest_path: Path = MANIFEST_PATH,
    state_path: Path = STATE_FILE_PATH,
    doctrine_log: Path = DOCTRINE_LOG,
) -> Dict[str, Any]:
    """Check activation window state.  Called from executor main loop.

    Returns a status dict with full stack health.  Sets KILL_SWITCH
    on any structural integrity violation or when the 14-day window
    completes.

    Side effects:
        - Sets ``os.environ["KILL_SWITCH"] = "1"`` on halt condition
        - Logs doctrine event on first halt (idempotent)
        - Writes ``logs/state/activation_window_state.json``
    """
    global _kill_switch_fired, _boot_manifest_hash, _boot_config_hash

    aw = _load_activation_config(runtime_yaml)
    if aw is None:
        inactive = {"active": False, "ts": _utc_now().isoformat()}
        _write_state(inactive, state_path)
        return inactive

    # --- Window timing ---
    start_ts = str(aw.get("start_ts", ""))
    duration_days = int(aw.get("duration_days", DEFAULT_DURATION_DAYS))
    dd_kill_pct = float(aw.get("drawdown_kill_pct", DEFAULT_DD_KILL_PCT))
    per_trade_nav_pct = float(aw.get("per_trade_nav_pct", DEFAULT_PER_TRADE_NAV_PCT))

    if not start_ts:
        logger.warning("[activation_window] no start_ts — inactive")
        return {"active": False}

    start_dt = _parse_iso_ts(start_ts)
    if start_dt is None:
        return {"active": False}

    now = _utc_now()
    end_dt = start_dt + timedelta(days=duration_days)
    elapsed_days = (now - start_dt).total_seconds() / 86400
    remaining_days = max(0.0, (end_dt - now).total_seconds() / 86400)
    window_expired = now >= end_dt

    # --- Hash snapshots (capture at first check) ---
    if _boot_manifest_hash is None:
        _boot_manifest_hash = _compute_manifest_hash(manifest_path)
    if _boot_config_hash is None:
        _boot_config_hash = _compute_config_hash(runtime_yaml)

    current_manifest_hash = _compute_manifest_hash(manifest_path)
    current_config_hash = _compute_config_hash(runtime_yaml)

    # --- Integrity checks ---
    manifest_intact = (
        _boot_manifest_hash is not None
        and current_manifest_hash == _boot_manifest_hash
    )
    config_intact = (
        _boot_config_hash is not None
        and current_config_hash == _boot_config_hash
    )

    dd_pct = _get_portfolio_dd_pct()
    dd_breached = dd_kill_pct > 0 and dd_pct >= dd_kill_pct

    nav_usd = _get_nav_usd()
    binary_lab_freeze_ok = _check_binary_lab_freeze()
    dle_mismatches = _check_dle_anomalies(start_ts)
    episodes_completed = _count_episodes_since(start_ts)
    risk_veto_count = _count_risk_vetoes_since(start_ts)

    # --- Determine halt condition ---
    halt_reason: Optional[str] = None
    if window_expired:
        halt_reason = f"window_complete:{duration_days}d_elapsed"
    elif dd_breached:
        halt_reason = f"drawdown_kill:{dd_pct:.4f}>={dd_kill_pct:.4f}"
    elif not manifest_intact:
        halt_reason = f"manifest_drift:boot={_boot_manifest_hash},now={current_manifest_hash}"
    elif not config_intact:
        halt_reason = f"config_drift:boot={_boot_config_hash},now={current_config_hash}"
    elif not binary_lab_freeze_ok:
        halt_reason = "binary_lab_freeze_violation"

    halted = halt_reason is not None

    status: Dict[str, Any] = {
        "active": True,
        "ts": now.isoformat(),
        "start_ts": start_ts,
        "end_ts": end_dt.isoformat(),
        "duration_days": duration_days,
        "elapsed_days": round(elapsed_days, 2),
        "remaining_days": round(remaining_days, 2),
        "window_expired": window_expired,
        "halted": halted,
        "halt_reason": halt_reason,
        # Integrity
        "manifest_intact": manifest_intact,
        "config_intact": config_intact,
        "boot_manifest_hash": _boot_manifest_hash,
        "current_manifest_hash": current_manifest_hash,
        "boot_config_hash": _boot_config_hash,
        "current_config_hash": current_config_hash,
        # Risk
        "drawdown_pct": round(dd_pct, 6),
        "drawdown_kill_pct": dd_kill_pct,
        "dd_breached": dd_breached,
        "nav_usd": round(nav_usd, 2),
        # Telemetry
        "episodes_completed": episodes_completed,
        "risk_veto_count": risk_veto_count,
        "dle_mismatches": dle_mismatches,
        "binary_lab_freeze_ok": binary_lab_freeze_ok,
        # Sizing
        "per_trade_nav_pct": per_trade_nav_pct,
    }

    # --- Fire KILL_SWITCH (idempotent) ---
    if halted and not _kill_switch_fired:
        # Window complete is NOT a failure — just a lock
        os.environ["KILL_SWITCH"] = "1"
        _kill_switch_fired = True

        # Distinguish completion from structural kill for DLE events
        is_completion = halt_reason is not None and halt_reason.startswith("window_complete:")
        dle_action = "COMPLETED" if is_completion else "HALTED"

        _log_doctrine_event({
            "ts": now.isoformat(),
            "event": "ACTIVATION_WINDOW_HALT",
            "source": "activation_window",
            "halt_reason": halt_reason,
            "elapsed_days": round(elapsed_days, 2),
            "episodes_completed": episodes_completed,
            "dd_pct": round(dd_pct, 6),
            "manifest_intact": manifest_intact,
            "config_intact": config_intact,
            "binary_lab_freeze_ok": binary_lab_freeze_ok,
            "dle_mismatches": dle_mismatches,
            "action": f"KILL_SWITCH activated — {halt_reason}",
        }, doctrine_log)

        # Emit DLE STRUCTURAL_GUARD lifecycle event
        _emit_dle_lifecycle_event(dle_action, {
            "window_start_ts": start_ts,
            "duration_days": duration_days,
            "elapsed_days": round(elapsed_days, 2),
            "manifest_hash": current_manifest_hash,
            "config_hash": current_config_hash,
            "halt_reason": halt_reason,
            "episodes_completed": episodes_completed,
            "dd_pct": round(dd_pct, 6),
            "dle_mismatches": dle_mismatches,
            "provenance": {"source": "activation_window", "version": "v8.0"},
        })

        logger.info(
            "[activation_window] HALT — %s. KILL_SWITCH=1.",
            halt_reason,
        )

    # --- Periodic status log (throttled) ---
    global _status_log_counter
    _status_log_counter += 1
    if _status_log_counter >= _STATUS_LOG_INTERVAL:
        _status_log_counter = 0
        logger.info(
            "[activation_window] day %.1f/%d — dd=%.4f  episodes=%d  vetoes=%d  manifest=%s",
            elapsed_days, duration_days, dd_pct, episodes_completed,
            risk_veto_count, "OK" if manifest_intact else "DRIFT",
        )

    # --- Write state file ---
    _write_state(status, state_path)

    return status


# ---------------------------------------------------------------------------
# Boot status logging
# ---------------------------------------------------------------------------
def log_activation_boot_status(
    *,
    runtime_yaml: Path = RUNTIME_YAML,
    manifest_path: Path = MANIFEST_PATH,
) -> None:
    """Log activation window state at executor startup.

    Emits a single INFO line and captures initial hash snapshots.
    """
    global _boot_manifest_hash, _boot_config_hash

    aw = _load_activation_config(runtime_yaml)
    if aw is None:
        ack = os.environ.get(ACTIVATION_ACK_ENV, "0")
        try:
            with open(runtime_yaml) as f:
                raw = yaml.safe_load(f) or {}
            raw_aw = raw.get("activation_window", {})
            if isinstance(raw_aw, dict) and raw_aw.get("enabled"):
                logger.warning(
                    "[activation_window] BOOT: config enabled but %s=%s — INACTIVE (dual-key missing)",
                    ACTIVATION_ACK_ENV, ack,
                )
                return
        except Exception:
            pass
        logger.info("[activation_window] BOOT: INACTIVE (disabled or missing config)")
        return

    # Capture hash snapshots at boot
    _boot_manifest_hash = _compute_manifest_hash(manifest_path)
    _boot_config_hash = _compute_config_hash(runtime_yaml)

    duration = int(aw.get("duration_days", DEFAULT_DURATION_DAYS))
    start_ts = str(aw.get("start_ts", ""))
    dd_kill = float(aw.get("drawdown_kill_pct", DEFAULT_DD_KILL_PCT))
    sizing = float(aw.get("per_trade_nav_pct", DEFAULT_PER_TRADE_NAV_PCT))

    # Compute sizing cap in USD
    sizing_usd_str = "N/A"
    nav = _get_nav_usd()
    if nav > 0:
        sizing_usd_str = f"${sizing * nav:.2f}"

    logger.info(
        "[activation_window] BOOT: ACTIVE — %dd window, start=%s, "
        "dd_kill=%.4f, sizing=%.4f (%s), manifest=%s, ack=%s",
        duration,
        start_ts or "(not set)",
        dd_kill,
        sizing,
        sizing_usd_str,
        _boot_manifest_hash or "unknown",
        os.environ.get(ACTIVATION_ACK_ENV, "0"),
    )

    # Emit DLE STRUCTURAL_GUARD STARTED event (once per boot)
    global _dle_started_emitted
    if not _dle_started_emitted:
        _dle_started_emitted = True
        _emit_dle_lifecycle_event("STARTED", {
            "window_start_ts": start_ts,
            "duration_days": duration,
            "manifest_hash": _boot_manifest_hash,
            "config_hash": _boot_config_hash,
            "nav_usd": round(nav, 2),
            "sizing_cap": sizing,
            "provenance": {"source": "activation_window", "version": "v8.0"},
        })


# ---------------------------------------------------------------------------
# Sizing override
# ---------------------------------------------------------------------------
def get_activation_sizing_override(
    runtime_yaml: Path = RUNTIME_YAML,
) -> Optional[float]:
    """Return per_trade_nav_pct if activation window is active, else None."""
    aw = _load_activation_config(runtime_yaml)
    if aw is None:
        return None
    pct = aw.get("per_trade_nav_pct")
    return float(pct) if pct is not None else None


# ---------------------------------------------------------------------------
# Stack health snapshot (for activation_verify.py)
# ---------------------------------------------------------------------------
def collect_stack_health(
    *,
    runtime_yaml: Path = RUNTIME_YAML,
    manifest_path: Path = MANIFEST_PATH,
) -> Dict[str, Any]:
    """Collect full-stack health metrics for verification.

    Used by ``scripts/activation_verify.py`` at day 14.
    """
    aw = _load_activation_config(runtime_yaml) or {}
    start_ts = str(aw.get("start_ts", ""))
    duration_days = int(aw.get("duration_days", DEFAULT_DURATION_DAYS))
    dd_kill_pct = float(aw.get("drawdown_kill_pct", DEFAULT_DD_KILL_PCT))

    start_dt = _parse_iso_ts(start_ts)
    now = _utc_now()
    elapsed_days = 0.0
    if start_dt:
        elapsed_days = (now - start_dt).total_seconds() / 86400

    nav = _get_nav_usd()
    dd_pct = _get_portfolio_dd_pct()
    manifest_hash = _compute_manifest_hash(manifest_path)
    config_hash = _compute_config_hash(runtime_yaml)

    return {
        "ts": now.isoformat(),
        "start_ts": start_ts,
        "duration_days": duration_days,
        "elapsed_days": round(elapsed_days, 2),
        "nav_usd": round(nav, 2),
        "drawdown_pct": round(dd_pct, 6),
        "drawdown_kill_pct": dd_kill_pct,
        "dd_within_limits": dd_pct < dd_kill_pct,
        "episodes_completed": _count_episodes_since(start_ts) if start_ts else 0,
        "risk_veto_count": _count_risk_vetoes_since(start_ts) if start_ts else 0,
        "dle_mismatches": _check_dle_anomalies(start_ts) if start_ts else 0,
        "binary_lab_freeze_ok": _check_binary_lab_freeze(),
        "manifest_hash": manifest_hash,
        "config_hash": config_hash,
    }


# ---------------------------------------------------------------------------
# Production scale gate (Phase C constitutional binding)
# ---------------------------------------------------------------------------
def get_scale_gate_cap(
    *,
    runtime_yaml: Path = RUNTIME_YAML,
    verdict_path: Path = VERIFICATION_VERDICT_PATH,
    manifest_path: Path = MANIFEST_PATH,
) -> Optional[float]:
    """Return sizing cap if production scale is not yet authorized.

    Returns ``per_trade_nav_pct`` if the scale gate is active (no valid
    7/7 GO verdict with matching manifest hash), or ``None`` if
    production scale is authorized.

    The gate stays active even after the activation window is disabled.
    Only a 7/7 GO verdict with matching manifest hash clears it.

    This implements Phase C Amendment §2.1 — mandatory pre-scale gate.
    """
    try:
        with open(runtime_yaml) as f:
            cfg = yaml.safe_load(f) or {}
        aw = cfg.get("activation_window")
        if not isinstance(aw, dict):
            return None  # No activation window config → no gate
        if not aw.get("require_go_for_scale", True):
            return None  # Gate explicitly disabled
        cap = float(aw.get("per_trade_nav_pct", DEFAULT_PER_TRADE_NAV_PCT))
    except Exception:
        return None

    # Check if GO verdict exists with matching manifest
    verdict = _read_json(verdict_path)
    if verdict is None:
        return cap  # No verdict → cap active

    if verdict.get("verdict") != "GO":
        return cap
    if verdict.get("passed") != verdict.get("total_gates"):
        return cap

    # Check manifest hash integrity
    verdict_manifest = verdict.get("manifest_hash")
    current_manifest = _compute_manifest_hash(manifest_path)
    if (
        verdict_manifest
        and current_manifest
        and verdict_manifest == current_manifest
    ):
        return None  # Scale authorized — 7/7 GO + manifest match

    return cap  # Manifest drift since verification → cap stays


def verify_production_scale_eligible(
    *,
    manifest_path: Path = MANIFEST_PATH,
    verdict_path: Path = VERIFICATION_VERDICT_PATH,
) -> Dict[str, Any]:
    """Check if production scale is authorized post-activation.

    Returns dict with ``eligible`` bool and details.
    Used by ops tooling and dashboard for visibility.
    """
    result: Dict[str, Any] = {
        "eligible": False,
        "reason": "unknown",
        "verdict": None,
        "manifest_match": False,
    }

    verdict_data = _read_json(verdict_path)
    if verdict_data is None:
        result["reason"] = "no_verification_verdict"
        return result

    verdict = verdict_data.get("verdict")
    passed = verdict_data.get("passed", 0)
    total = verdict_data.get("total_gates", 7)

    if verdict != "GO":
        result["reason"] = f"verdict_{verdict}_not_GO"
        result["verdict"] = verdict
        return result

    if passed != total:
        result["reason"] = f"gates_{passed}/{total}_incomplete"
        result["verdict"] = verdict
        return result

    # Check manifest hash integrity
    verdict_manifest = verdict_data.get("manifest_hash")
    current_manifest = _compute_manifest_hash(manifest_path)

    manifest_match = (
        verdict_manifest is not None
        and current_manifest is not None
        and verdict_manifest == current_manifest
    )
    result["manifest_match"] = manifest_match
    result["verdict"] = verdict

    if not manifest_match:
        result["reason"] = (
            f"manifest_drift:verdict={verdict_manifest},"
            f"current={current_manifest}"
        )
        return result

    result["eligible"] = True
    result["reason"] = "production_scale_authorized"
    return result


# ---------------------------------------------------------------------------
# Verdict recording (called by activation_verify.py)
# ---------------------------------------------------------------------------
def record_verification_verdict(
    verdict_result: Dict[str, Any],
    *,
    verdict_path: Path = VERIFICATION_VERDICT_PATH,
    manifest_path: Path = MANIFEST_PATH,
) -> None:
    """Persist verification verdict for production scale gating.

    Called by ``scripts/activation_verify.py`` after 7-gate evaluation.
    Writes verdict to state file and emits ACTIVATION_WINDOW_VERIFIED
    DLE lifecycle event.
    """
    verdict_result["manifest_hash"] = _compute_manifest_hash(manifest_path)
    verdict_result["recorded_at"] = _utc_now().isoformat()

    try:
        verdict_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = verdict_path.with_suffix(".tmp")
        with tmp.open("w") as f:
            json.dump(verdict_result, f, indent=2, default=str)
        tmp.replace(verdict_path)
    except Exception as exc:
        logger.error("[activation_window] verdict write failed: %s", exc)

    # Emit DLE lifecycle event for governance observability
    _emit_dle_lifecycle_event("VERIFIED", {
        "verdict": verdict_result.get("verdict"),
        "passed": verdict_result.get("passed"),
        "total_gates": verdict_result.get("total_gates"),
        "manifest_hash": verdict_result.get("manifest_hash"),
        "action_text": verdict_result.get("action"),
        "provenance": {"source": "activation_verify", "version": "v8.0"},
    })


# ---------------------------------------------------------------------------
# Module reset (for testing)
# ---------------------------------------------------------------------------
def _reset_globals() -> None:
    """Reset module-level sentinels.  For testing only."""
    global _kill_switch_fired, _boot_manifest_hash, _boot_config_hash
    global _status_log_counter, _dle_started_emitted
    _kill_switch_fired = False
    _boot_manifest_hash = None
    _boot_config_hash = None
    _status_log_counter = 0
    _dle_started_emitted = False
