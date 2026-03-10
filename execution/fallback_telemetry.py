"""Fallback swap telemetry — lightweight in-process accumulator.

Tracks running sums for fallback edge delta, primary rejection gap,
fallback rate, and Hydra quality metrics over a rolling 24h window.
Zero I/O on record; snapshot() is O(1).  Safe across executor restarts
(counters simply reset to zero).

v7.9_P3c
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

_LOG = logging.getLogger(__name__)

WINDOW_SECONDS = 86_400  # 24 h

_STATE_PATH = os.path.join("logs", "state", "fallback_metrics.json")


@dataclass
class FallbackTelemetry:
    normal_count: int = 0
    fallback_count: int = 0
    conflict_count: int = 0

    normal_score_sum: float = 0.0
    fallback_score_sum: float = 0.0
    primary_score_sum: float = 0.0

    # Hydra quality counters
    hydra_exec_count: int = 0
    hydra_exec_score_sum: float = 0.0
    legacy_exec_count: int = 0
    legacy_exec_score_sum: float = 0.0
    hydra_primary_count: int = 0        # times hydra won merge
    hydra_primary_rejected: int = 0     # times hydra won merge but fallback fired
    hydra_rescue_count: int = 0         # times hydra executed via fallback

    # Conflict Edge Lift (CEL)
    cel_count: int = 0
    cel_sum: float = 0.0

    # Score Distribution Divergence (SDD) — paired conflict candidates
    sdd_hydra_sum: float = 0.0
    sdd_legacy_sum: float = 0.0
    sdd_count: int = 0

    # Regime score differential — per-regime, per-engine running sums
    _regime_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)

    last_reset_ts: float = field(default_factory=time.time)

    # ---- public API ---------------------------------------------------

    def maybe_reset(self) -> None:
        now = time.time()
        if now - self.last_reset_ts > WINDOW_SECONDS:
            self.normal_count = 0
            self.fallback_count = 0
            self.conflict_count = 0
            self.normal_score_sum = 0.0
            self.fallback_score_sum = 0.0
            self.primary_score_sum = 0.0
            self.hydra_exec_count = 0
            self.hydra_exec_score_sum = 0.0
            self.legacy_exec_count = 0
            self.legacy_exec_score_sum = 0.0
            self.hydra_primary_count = 0
            self.hydra_primary_rejected = 0
            self.hydra_rescue_count = 0
            self.cel_count = 0
            self.cel_sum = 0.0
            self.sdd_hydra_sum = 0.0
            self.sdd_legacy_sum = 0.0
            self.sdd_count = 0
            self._regime_scores = {}
            self.last_reset_ts = now

    def record_attempt(self, intent: Dict[str, Any]) -> None:
        """Record a single order-attempt intent.  ~O(1), no I/O."""
        self.maybe_reset()

        score = _safe_float(intent.get("score") or intent.get("hybrid_score"))
        fallback_used = bool(intent.get("fallback_used"))
        conflict = bool(intent.get("merge_conflict"))

        if conflict:
            self.conflict_count += 1

        if fallback_used:
            self.fallback_count += 1
            self.fallback_score_sum += score
            self.primary_score_sum += _safe_float(intent.get("merge_primary_score"))
        else:
            self.normal_count += 1
            self.normal_score_sum += score

        # Hydra quality tracking
        executed_engine = str(
            intent.get("source") or intent.get("strategy") or ""
        ).lower()
        # merge_primary_engine is only set on fallback swaps;
        # for normal conflict executions, the executed engine IS the primary
        if fallback_used:
            primary_engine = str(intent.get("merge_primary_engine") or "").lower()
        elif conflict:
            primary_engine = executed_engine
        else:
            primary_engine = ""

        if executed_engine == "hydra":
            self.hydra_exec_count += 1
            self.hydra_exec_score_sum += score
        elif executed_engine == "legacy":
            self.legacy_exec_count += 1
            self.legacy_exec_score_sum += score

        if primary_engine == "hydra":
            self.hydra_primary_count += 1
            if fallback_used:
                self.hydra_primary_rejected += 1

        if fallback_used and executed_engine == "hydra":
            self.hydra_rescue_count += 1

        # Conflict Edge Lift (CEL)
        if conflict:
            legacy_score = _safe_float(intent.get("merge_legacy_score"))
            if legacy_score or executed_engine != "legacy":
                self.cel_count += 1
                if executed_engine == "legacy":
                    self.cel_sum += 0.0
                else:
                    self.cel_sum += score - legacy_score

        # Score Distribution Divergence (SDD) — paired candidate scores
        if conflict:
            _hs = _safe_float(intent.get("merge_hydra_score"))
            _ls = _safe_float(intent.get("merge_legacy_score"))
            if _hs or _ls:  # at least one non-zero
                self.sdd_hydra_sum += _hs
                self.sdd_legacy_sum += _ls
                self.sdd_count += 1

        # Regime score differential
        regime = _extract_regime(intent)
        if regime and executed_engine in ("hydra", "legacy"):
            bucket = self._regime_scores.setdefault(regime, {
                "hydra_n": 0.0, "hydra_sum": 0.0,
                "legacy_n": 0.0, "legacy_sum": 0.0,
            })
            bucket[f"{executed_engine}_n"] += 1
            bucket[f"{executed_engine}_sum"] += score

    def snapshot(self) -> Dict[str, Any]:
        """Return current metric snapshot.  O(1), no I/O."""
        avg_normal = (
            self.normal_score_sum / self.normal_count
            if self.normal_count else 0.0
        )
        avg_fallback = (
            self.fallback_score_sum / self.fallback_count
            if self.fallback_count else 0.0
        )
        avg_primary = (
            self.primary_score_sum / self.fallback_count
            if self.fallback_count else 0.0
        )
        fallback_rate = (
            self.fallback_count / self.conflict_count
            if self.conflict_count else 0.0
        )
        # Hydra quality
        avg_hydra = (
            self.hydra_exec_score_sum / self.hydra_exec_count
            if self.hydra_exec_count else 0.0
        )
        avg_legacy = (
            self.legacy_exec_score_sum / self.legacy_exec_count
            if self.legacy_exec_count else 0.0
        )
        hydra_rescue_rate = (
            self.hydra_rescue_count / self.conflict_count
            if self.conflict_count else 0.0
        )
        hydra_overconfidence = (
            self.hydra_primary_rejected / self.hydra_primary_count
            if self.hydra_primary_count else 0.0
        )
        total_exec = self.hydra_exec_count + self.legacy_exec_count
        hydra_participation = (
            self.hydra_exec_count / total_exec
            if total_exec else 0.0
        )

        rsd = self._compute_regime_rsd()

        cel_raw = self.cel_sum / self.cel_count if self.cel_count else 0.0
        sdd_raw = (
            (self.sdd_hydra_sum / self.sdd_count if self.sdd_count else 0.0)
            - (self.sdd_legacy_sum / self.sdd_count if self.sdd_count else 0.0)
        )

        return {
            "fallback_rate": round(fallback_rate, 4),
            "fallback_edge_delta": round(avg_fallback - avg_normal, 4),
            "primary_rejection_gap": round(avg_primary - avg_fallback, 4),
            "normal_count": self.normal_count,
            "fallback_count": self.fallback_count,
            "conflict_count": self.conflict_count,
            "hydra_quality_diff": round(avg_hydra - avg_legacy, 4),
            "hydra_participation": round(hydra_participation, 4),
            "hydra_rescue_rate": round(hydra_rescue_rate, 4),
            "hydra_overconfidence": round(hydra_overconfidence, 4),
            "hydra_exec_count": self.hydra_exec_count,
            "legacy_exec_count": self.legacy_exec_count,
            "conflict_edge_lift": round(cel_raw, 4),
            "cel_count": self.cel_count,
            "conflict_rate": round(
                self.conflict_count / total_exec if total_exec else 0.0, 4
            ),
            "score_scale_delta": round(sdd_raw, 4),
            "sdd_hydra_mean": round(
                self.sdd_hydra_sum / self.sdd_count if self.sdd_count else 0.0, 4
            ),
            "sdd_legacy_mean": round(
                self.sdd_legacy_sum / self.sdd_count if self.sdd_count else 0.0, 4
            ),
            "sdd_count": self.sdd_count,
            "window_age_s": round(time.time() - self.last_reset_ts, 1),
            "regime_rsd": rsd,
            "regime_dependence_spread": round(
                max(e["rsd"] for e in rsd.values()) - min(e["rsd"] for e in rsd.values()), 4
            ) if len(rsd) >= 2 else 0.0,
            **self._compute_mri(fallback_rate, self.cel_count, cel_raw, sdd_raw),
        }

    def _compute_mri(
        self,
        fallback_rate: float,
        cel_count: int,
        cel: float,
        sdd: float,
    ) -> Dict[str, Any]:
        """Migration Readiness Index — derived readiness signal."""
        ready_trades = cel_count >= 300
        stable_recovery = fallback_rate < 0.05
        positive_edge = cel > 0
        score_calibrated = abs(sdd) <= 0.02

        ecs_ready = ready_trades and stable_recovery and positive_edge and score_calibrated

        score = (
            min(cel_count / 300, 1.0) * 0.4
            + max(0.0, (0.05 - fallback_rate) / 0.05) * 0.3
            + (min(cel / 0.03, 1.0) * 0.2 if cel > 0 else 0.0)
            + max(0.0, (0.02 - abs(sdd)) / 0.02) * 0.1
        )

        return {
            "ecs_ready": ecs_ready,
            "ecs_readiness_score": round(min(max(score, 0.0), 1.0), 4),
            "ecs_ready_trades": ready_trades,
            "ecs_stable_recovery": stable_recovery,
            "ecs_positive_edge": positive_edge,
            "ecs_score_calibrated": score_calibrated,
        }

    def _compute_regime_rsd(self) -> Dict[str, Any]:
        """RSD per regime: avg(hydra_score) - avg(legacy_score) plus counts."""
        out: Dict[str, Any] = {}
        for regime, b in self._regime_scores.items():
            h_n = int(b["hydra_n"])
            l_n = int(b["legacy_n"])
            avg_h = b["hydra_sum"] / h_n if h_n else 0.0
            avg_l = b["legacy_sum"] / l_n if l_n else 0.0
            if h_n and l_n:
                out[regime] = {
                    "rsd": round(avg_h - avg_l, 4),
                    "hydra_n": h_n,
                    "legacy_n": l_n,
                }
        return out

    def persist_snapshot(self, path: Optional[str] = None) -> None:
        """Write snapshot to state JSON for dashboard consumption."""
        dest = path or _STATE_PATH
        snap = self.snapshot()
        snap["ts"] = time.time()

        # Carry forward MRI score history (capped at 50 entries)
        history: list = []
        try:
            with open(dest) as fh:
                prev = json.load(fh)
            history = prev.get("ecs_score_history", [])
        except (OSError, json.JSONDecodeError, TypeError):
            pass
        history.append(round(snap.get("ecs_readiness_score", 0.0), 4))
        snap["ecs_score_history"] = history[-50:]

        try:
            tmp = dest + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(snap, fh, indent=2)
            os.replace(tmp, dest)
        except OSError as exc:
            _LOG.debug("[fallback_telemetry] persist failed: %s", exc)


def _safe_float(val: Any) -> float:
    if val is None:
        return 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _extract_regime(intent: Dict[str, Any]) -> str:
    """Pull regime from intent metadata or top-level field."""
    meta = intent.get("metadata")
    if isinstance(meta, dict):
        r = meta.get("entry_regime")
        if r:
            return str(r).upper()
    r = intent.get("entry_regime") or intent.get("regime")
    return str(r).upper() if r else ""
