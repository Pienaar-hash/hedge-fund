"""
Hydra Funnel Telemetry — tracks intent survival through pipeline stages.

Answers: "Is Hydra being observed or being filtered before it can be measured?"

Stages tracked:
  generated     → Hydra intents produced by run_hydra_pipeline
  post_merge    → Hydra intents surviving merge with legacy
  submitted     → Hydra intents passing doctor check into _send_order
  post_doctrine → Hydra intents passing doctrine gate (inside _send_order)
  executed      → Hydra intents reaching exchange dispatch (order acked)

Visibility Rate = executed / generated
Per-regime breakdown reveals where suppression occurs.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

LOG = logging.getLogger(__name__)
_FUNNEL_PATH = Path("logs/state/hydra_funnel.json")

STAGES = ("generated", "post_merge", "submitted", "post_doctrine", "executed")


class HydraFunnel:
    """Accumulator for Hydra intent survival through pipeline stages."""

    def __init__(self) -> None:
        self._counts: Dict[str, int] = defaultdict(int)
        self._regime_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._lock = threading.Lock()
        self._last_flush_ts = 0.0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, stage: str, count: int = 1, regime: str = "") -> None:
        """Record *count* intents reaching *stage*, optionally keyed by regime."""
        if count <= 0:
            return
        with self._lock:
            self._counts[stage] += count
            if regime:
                self._regime_counts[regime][stage] += count

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-safe snapshot of the funnel state."""
        with self._lock:
            generated = self._counts.get("generated", 0)
            post_merge = self._counts.get("post_merge", 0)
            submitted = self._counts.get("submitted", 0)
            post_doctrine = self._counts.get("post_doctrine", 0)
            executed = self._counts.get("executed", 0)

            vis_rate = executed / generated if generated > 0 else 0.0
            merge_win_rate = post_merge / generated if generated > 0 else 0.0

            regime_vis: Dict[str, Any] = {}
            for regime, stages in self._regime_counts.items():
                rg = stages.get("generated", 0)
                re = stages.get("executed", 0)
                rpm = stages.get("post_merge", 0)
                regime_vis[regime] = {
                    "stages": dict(stages),
                    "visibility_rate": round(re / rg, 4) if rg > 0 else 0.0,
                    "merge_win_rate": round(rpm / rg, 4) if rg > 0 else 0.0,
                }

            return {
                "stages": {
                    "generated": generated,
                    "post_merge": post_merge,
                    "submitted": submitted,
                    "post_doctrine": post_doctrine,
                    "executed": executed,
                },
                "visibility_rate": round(vis_rate, 4),
                "merge_win_rate": round(merge_win_rate, 4),
                "regime_visibility": regime_vis,
                "updated_ts": time.time(),
            }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def flush(self, path: Optional[Path] = None) -> None:
        """Write snapshot to JSON state file."""
        p = path or _FUNNEL_PATH
        try:
            snap = self.snapshot()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(snap, indent=2))
        except Exception as exc:
            LOG.debug("[hydra_funnel] flush failed: %s", exc)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_FUNNEL = HydraFunnel()


def get_funnel() -> HydraFunnel:
    """Return the module-level funnel singleton."""
    return _FUNNEL


def record(stage: str, count: int = 1, regime: str = "") -> None:
    """Convenience: record on the global singleton."""
    _FUNNEL.record(stage, count, regime=regime)


def snapshot() -> Dict[str, Any]:
    """Convenience: snapshot from the global singleton."""
    return _FUNNEL.snapshot()


def flush(path: Optional[Path] = None) -> None:
    """Convenience: flush global singleton to disk."""
    _FUNNEL.flush(path)
