"""
Typed executor state container (Phase 5).

Centralises mutable per-run state that was previously scattered across
module-level globals in ``executor_live.py``.  Fields are introduced
incrementally — only those already wired through call-sites live here.

Design rules
------------
* New code must receive ``state`` explicitly — do NOT read ``_STATE``
  directly from nested helpers.
* Fields are added one-at-a-time: declare → wire readers/writers → delete
  old global.
* This dataclass is *not* frozen — fields are mutated in-place during the
  main loop.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from execution.position_cache import PositionCache


def _default_run_id() -> str:
    return os.getenv("EXECUTOR_RUN_ID") or str(uuid.uuid4())


def _default_engine_version() -> str:
    try:
        from execution.versioning import read_version
        return read_version(default="v7.6")
    except Exception:
        return "v7.6"


@dataclass
class ExecutorState:
    """Container for mutable executor-run state.

    Only fields that have been fully or partially wired through explicit
    ``state`` parameters belong here.  Do **not** add fields speculatively.
    """

    # ── core identity ─────────────────────────────────────────────────
    position_cache: PositionCache
    run_id: str = field(default_factory=_default_run_id)
    engine_version: str = field(default_factory=_default_engine_version)

    # ── lifecycle timestamps ──────────────────────────────────────────
    executor_start_ts: float = field(default_factory=time.time)
    last_cycle_ts: float = 0.0

    # ── disk pressure guard ───────────────────────────────────────────
    last_disk_check_ts: float = 0.0
    last_disk_alert_ts: float = 0.0

    # ── deferred fields (declared, NOT yet wired — Phase 6) ──────────
    last_signal_pull: float = 0.0
    last_queue_depth: int = 0
    last_episode_ledger_rebuild_ts: float = 0.0
    last_doctrine_block_log_ts: float = 0.0

    # ── deferred fill collection (Phase 7 throughput sprint) ──────────
    pending_fill_handles: list = field(default_factory=list)

    # ── convenience mutators ──────────────────────────────────────────

    def touch_cycle(self) -> None:
        """Stamp current wall-clock time as last cycle start."""
        self.last_cycle_ts = time.time()
