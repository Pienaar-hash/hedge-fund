"""Execution determinism guard — runtime invariant checker.

Reads /proc metrics to detect when the executor's environment has degraded
below execution-grade.  Swap-induced latency, memory pressure stalls, and
involuntary context-switch spikes violate the deterministic execution
assumption required by Doctrine.

Usage inside executor_live::

    from execution.determinism_guard import check_determinism, DeterminismSnapshot

    snap = check_determinism(executor_pid=os.getpid())
    if snap.degraded:
        # block new entries, log DLE event
        ...

This module is **fail-open**: if any /proc read fails, the corresponding
check is skipped (never false-positive).  It never raises, never blocks
execution, and has zero external dependencies.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds (overridable via env for tuning without code change)
# ---------------------------------------------------------------------------
EXECUTOR_SWAP_KB_THRESHOLD = int(os.getenv("DETERMINISM_SWAP_KB", "10240"))  # 10 MB
SYSTEM_SWAP_MB_THRESHOLD = int(os.getenv("DETERMINISM_SYS_SWAP_MB", "400"))  # 400 MB
MEM_PSI_THRESHOLD = float(os.getenv("DETERMINISM_MEM_PSI", "1.0"))  # avg10 > 1%
AVAIL_MEM_PCT_FLOOR = int(os.getenv("DETERMINISM_AVAIL_PCT", "15"))  # < 15%

# Check interval — don't read /proc every loop iteration
CHECK_INTERVAL_S = float(os.getenv("DETERMINISM_CHECK_INTERVAL_S", "30"))

# Hysteresis — after DEGRADED, require this many seconds of clean readings
# before returning to OK.  Prevents rapid toggling at threshold boundaries.
HYSTERESIS_CLEAR_S = float(os.getenv("DETERMINISM_HYSTERESIS_S", "60"))

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeterminismSnapshot:
    """Point-in-time execution determinism assessment."""

    ts: float
    degraded: bool
    violations: List[str]

    # Raw metrics (present if readable, None if /proc unavailable)
    executor_swap_kb: Optional[int] = None
    executor_rss_kb: Optional[int] = None
    system_swap_used_mb: Optional[int] = None
    mem_psi_avg10: Optional[float] = None
    avail_mem_pct: Optional[int] = None

    # Hysteresis: True when violations cleared but hold period not elapsed
    held_degraded: bool = False

    def as_dict(self) -> dict:
        return {
            "ts": self.ts,
            "degraded": self.degraded,
            "violations": self.violations,
            "executor_swap_kb": self.executor_swap_kb,
            "executor_rss_kb": self.executor_rss_kb,
            "system_swap_used_mb": self.system_swap_used_mb,
            "mem_psi_avg10": self.mem_psi_avg10,
            "avail_mem_pct": self.avail_mem_pct,
        }

    def for_episode(self) -> dict:
        """Compact environment context for embedding in entry intents.

        This is the Episode ↔ Environment linkage: attach to every entry
        so that post-hoc analysis can separate model error from
        environment-induced execution error.
        """
        return {
            "determinism": "DEGRADED" if self.degraded else "OK",
            "vm_swap_kb": self.executor_swap_kb,
            "sys_swap_mb": self.system_swap_used_mb,
            "mem_psi_avg10": self.mem_psi_avg10,
            "mem_avail_pct": self.avail_mem_pct,
            "violations": len(self.violations),
        }

    def snapshot_hash(self) -> str:
        """Deterministic 16-char hex hash of the full snapshot state.

        Provides a hard link from episode → exact environment state at
        decision time.  Uses sorted-key canonical JSON so the hash is
        stable across Python dict ordering.
        """
        return compute_snapshot_hash(self.as_dict())


# ---------------------------------------------------------------------------
# Cached state (avoid /proc reads every cycle)
# ---------------------------------------------------------------------------
_last_check_ts: float = 0.0
_last_snapshot: Optional[DeterminismSnapshot] = None
_first_clean_ts: float = 0.0  # When violations first cleared (hysteresis)
_proc_read_failures: int = 0  # Cumulative /proc read failures


def _reset_cache() -> None:
    """Reset cached state (testing only)."""
    global _last_check_ts, _last_snapshot, _first_clean_ts, _proc_read_failures
    _last_check_ts = 0.0
    _last_snapshot = None
    _first_clean_ts = 0.0
    _proc_read_failures = 0


def get_proc_read_failures() -> int:
    """Return cumulative count of /proc read failures (observability)."""
    return _proc_read_failures


def compute_snapshot_hash(state: dict) -> str:
    """Deterministic 16-char SHA-256 prefix of canonical JSON.

    Used for episode ↔ environment referential linkage.  The hash is
    derived from the exact state dict seen at decision time, not from
    individual fields, guaranteeing reconstruction fidelity.
    """
    payload = json.dumps(state, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# /proc readers (each returns None on failure — never raises)
# ---------------------------------------------------------------------------


def _read_proc_status(pid: int) -> tuple[Optional[int], Optional[int]]:
    """Read VmSwap and VmRSS from /proc/<pid>/status.

    Returns (swap_kb, rss_kb) or (None, None).
    """
    global _proc_read_failures
    try:
        status = Path(f"/proc/{pid}/status").read_text()
        swap_kb: Optional[int] = None
        rss_kb: Optional[int] = None
        for line in status.splitlines():
            if line.startswith("VmSwap:"):
                swap_kb = int(line.split()[1])
            elif line.startswith("VmRSS:"):
                rss_kb = int(line.split()[1])
        return swap_kb, rss_kb
    except Exception:
        _proc_read_failures += 1
        return None, None


def _read_system_swap() -> Optional[int]:
    """Read system swap used in MB from /proc/meminfo.

    Returns swap_used_mb or None.
    """
    global _proc_read_failures
    try:
        meminfo = Path("/proc/meminfo").read_text()
        total = free = 0
        for line in meminfo.splitlines():
            if line.startswith("SwapTotal:"):
                total = int(line.split()[1])
            elif line.startswith("SwapFree:"):
                free = int(line.split()[1])
        used_kb = max(0, total - free)
        return used_kb // 1024
    except Exception:
        _proc_read_failures += 1
        return None


def _read_mem_psi_avg10() -> Optional[float]:
    """Read memory PSI some avg10 from /proc/pressure/memory.

    Returns avg10 percentage or None.
    """
    global _proc_read_failures
    try:
        text = Path("/proc/pressure/memory").read_text()
        for line in text.splitlines():
            if line.startswith("some "):
                for part in line.split():
                    if part.startswith("avg10="):
                        return float(part.split("=")[1])
        return None
    except Exception:
        _proc_read_failures += 1
        return None


def _read_avail_mem_pct() -> Optional[int]:
    """Read MemAvailable as percentage of MemTotal.

    Returns integer percentage or None.
    """
    global _proc_read_failures
    try:
        meminfo = Path("/proc/meminfo").read_text()
        total = avail = 0
        for line in meminfo.splitlines():
            if line.startswith("MemTotal:"):
                total = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                avail = int(line.split()[1])
        if total == 0:
            return None
        return (avail * 100) // total
    except Exception:
        _proc_read_failures += 1
        return None


# ---------------------------------------------------------------------------
# Main check
# ---------------------------------------------------------------------------


def check_determinism(
    executor_pid: Optional[int] = None,
    force: bool = False,
) -> DeterminismSnapshot:
    """Assess execution determinism from /proc metrics.

    Returns a cached snapshot if called within CHECK_INTERVAL_S of the
    previous check (unless ``force=True``).  This keeps /proc I/O to
    ~2 reads per minute rather than per-cycle.

    Fail-open: if all /proc reads fail, returns degraded=False
    (unknown is not a veto).
    """
    global _last_check_ts, _last_snapshot, _first_clean_ts

    now = time.time()
    if not force and _last_snapshot is not None and (now - _last_check_ts) < CHECK_INTERVAL_S:
        return _last_snapshot

    pid = executor_pid or os.getpid()
    violations: List[str] = []

    # 1. Executor process swap
    swap_kb, rss_kb = _read_proc_status(pid)
    if swap_kb is not None and swap_kb > EXECUTOR_SWAP_KB_THRESHOLD:
        violations.append(
            f"EXECUTOR_SWAP: {swap_kb // 1024}MB swapped (threshold: {EXECUTOR_SWAP_KB_THRESHOLD // 1024}MB)"
        )

    # 2. System swap usage
    sys_swap_mb = _read_system_swap()
    if sys_swap_mb is not None and sys_swap_mb > SYSTEM_SWAP_MB_THRESHOLD:
        violations.append(
            f"SYSTEM_SWAP: {sys_swap_mb}MB used (threshold: {SYSTEM_SWAP_MB_THRESHOLD}MB)"
        )

    # 3. Memory PSI stalls
    mem_psi = _read_mem_psi_avg10()
    if mem_psi is not None and mem_psi > MEM_PSI_THRESHOLD:
        violations.append(
            f"MEMORY_PSI: avg10={mem_psi:.2f}% (threshold: {MEM_PSI_THRESHOLD}%)"
        )

    # 4. Available memory headroom
    avail_pct = _read_avail_mem_pct()
    if avail_pct is not None and avail_pct < AVAIL_MEM_PCT_FLOOR:
        violations.append(
            f"LOW_MEMORY: {avail_pct}% available (floor: {AVAIL_MEM_PCT_FLOOR}%)"
        )

    # --- Hysteresis logic ---
    # Raw check says violations exist right now
    raw_degraded = len(violations) > 0

    if raw_degraded:
        # Currently violating — reset the clean timer
        _first_clean_ts = 0.0
        effective_degraded = True
    else:
        # Currently clean — but was previously degraded?
        was_degraded = _last_snapshot is not None and _last_snapshot.degraded
        if was_degraded:
            if _first_clean_ts == 0.0:
                # First clean reading after degradation — start hysteresis timer
                _first_clean_ts = now
            clean_duration = now - _first_clean_ts
            if clean_duration < HYSTERESIS_CLEAR_S:
                # Not clean long enough — hold DEGRADED
                effective_degraded = True
                violations = [
                    f"HYSTERESIS_HOLD: clean for {clean_duration:.0f}s "
                    f"(need {HYSTERESIS_CLEAR_S:.0f}s)"
                ]
            else:
                # Clean long enough — release
                effective_degraded = False
                _first_clean_ts = 0.0
        else:
            effective_degraded = False

    snap = DeterminismSnapshot(
        ts=now,
        degraded=effective_degraded,
        violations=violations,
        executor_swap_kb=swap_kb,
        executor_rss_kb=rss_kb,
        system_swap_used_mb=sys_swap_mb,
        mem_psi_avg10=mem_psi,
        avail_mem_pct=avail_pct,
        held_degraded=effective_degraded and not raw_degraded,
    )

    _last_check_ts = now
    _last_snapshot = snap
    return snap
