"""
Edge Realization Ratio (ERR) — Conviction calibration metric.

Answers: "Is the conviction model's predicted edge showing up in realized PnL?"

    ERR = Σ realized_return / Σ predicted_edge

Interpretation:
    ~1.0  → Model calibrated
    <0.7  → Model overestimates edge
    >1.3  → Model underestimates edge

Data source: episode_ledger.json (no new telemetry layer needed).
State output: logs/state/edge_calibration.json
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional

LOG = logging.getLogger(__name__)

_STATE_PATH = os.path.join("logs", "state", "edge_calibration.json")


def compute_err_from_episodes(
    episodes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute Edge Realization Ratio from episode ledger entries.

    For each episode with a usable edge estimate:
      predicted_edge  = episode["expected_edge"] if present,
                        else max(0, conviction_score - 0.5)
      realized_return = (exit - entry) / entry  (sign-adjusted for side)

    Returns dict with err, count, pred_sum, real_sum, ts.
    """
    pred_sum = 0.0
    real_sum = 0.0
    count = 0

    for ep in episodes:
        # ── Predicted edge ──────────────────────────────────────────
        edge = _safe_float(ep.get("expected_edge"))
        if not (edge > 0 and math.isfinite(edge)):
            # Fallback: derive from conviction_score
            conv = _safe_float(ep.get("conviction_score"))
            edge = max(0.0, conv - 0.5)
        if edge <= 0 or not math.isfinite(edge):
            continue  # no usable edge estimate

        # ── Realized return ─────────────────────────────────────────
        entry_px = _safe_float(ep.get("avg_entry_price"))
        exit_px = _safe_float(ep.get("avg_exit_price"))
        if entry_px <= 0 or exit_px <= 0:
            continue

        side = str(ep.get("side", "")).upper()
        if side == "LONG":
            realized = (exit_px - entry_px) / entry_px
        elif side == "SHORT":
            realized = (entry_px - exit_px) / entry_px
        else:
            continue

        pred_sum += edge
        real_sum += realized
        count += 1

    err = (real_sum / pred_sum) if pred_sum > 0 else None

    return {
        "err": round(err, 4) if err is not None else None,
        "count": count,
        "pred_sum": round(pred_sum, 8),
        "real_sum": round(real_sum, 8),
        "ts": time.time(),
    }


def persist_snapshot(
    episodes: List[Dict[str, Any]],
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute ERR and write to state file for dashboard consumption."""
    dest = path or _STATE_PATH
    snap = compute_err_from_episodes(episodes)

    try:
        tmp = dest + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(snap, fh, indent=2)
        os.replace(tmp, dest)
    except OSError as exc:
        LOG.debug("[edge_calibration] persist failed: %s", exc)

    return snap


def _safe_float(val: Any) -> float:
    if val is None:
        return 0.0
    try:
        v = float(val)
        return v if math.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0
