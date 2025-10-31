from __future__ import annotations

import json
import logging
import math
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from execution.log_utils import safe_dump
from execution.risk_limits import RiskGate, RiskState

LOG = logging.getLogger("risk.autotune")

RISK_STATE_PATH = Path("logs/cache/risk_state.json")
DOCTOR_CACHE_PATH = Path("logs/cache/doctor.json")
SIGNAL_METRICS_PATH = Path("logs/execution/signal_metrics.jsonl")
ORDER_METRICS_PATH = Path("logs/execution/order_metrics.jsonl")


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        LOG.debug("[risk] cache_read_failed file=%s err=%s", path, exc)
        return None


def _tail_jsonl(path: Path, limit: int) -> Sequence[Mapping[str, Any]]:
    if limit <= 0 or not path.exists():
        return []
    rows = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, Mapping):
                    rows.append(payload)
    except Exception as exc:
        LOG.debug("[risk] jsonl_tail_failed path=%s err=%s", path, exc)
        return []
    if len(rows) > limit:
        rows = rows[-limit:]
    return rows


def _to_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


class RiskAutotuner:
    """Adaptive risk controls tied into executor heartbeats."""

    def __init__(
        self,
        gate: RiskGate,
        *,
        cache_path: Path | None = None,
        signal_history: int = 200,
        order_history: int = 200,
    ) -> None:
        self._gate = gate
        self._cache_path = cache_path or RISK_STATE_PATH
        self._signal_history = max(20, signal_history)
        self._order_history = max(20, order_history)
        self._lock = threading.RLock()
        self._base_risk = dict(gate.cfg.get("risk", {}) or {})
        self._last_adjust_ts = 0.0
        self._last_adjust_summary: Dict[str, Any] = {}

    # ------------------------------------------------------------------ state IO
    def restore_state(self, state: RiskState) -> None:
        payload = _read_json(self._cache_path)
        if isinstance(payload, Mapping):
            state.apply_snapshot(payload)
            LOG.info("[risk] restored state counters from %s", self._cache_path)

    def persist_state(self, state: RiskState) -> None:
        snapshot = safe_dump(state.snapshot())
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self._cache_path.open("w", encoding="utf-8") as handle:
                json.dump(snapshot, handle, ensure_ascii=False, sort_keys=True, indent=2)
        except Exception as exc:
            LOG.debug("[risk] persist_failed path=%s err=%s", self._cache_path, exc)

    # ---------------------------------------------------------------- heuristics
    def _doctor_confidence(self) -> float:
        cache = _read_json(DOCTOR_CACHE_PATH)
        if isinstance(cache, Mapping):
            # Allow cron cache to supply a direct confidence value
            nav_freshness = cache.get("nav_freshness") if isinstance(cache.get("nav_freshness"), Mapping) else {}
            conf = _to_float(nav_freshness.get("confidence"))
            if conf is not None:
                return max(0.0, min(1.0, conf))
        # Fallback: derive from recent doctor verdicts
        confidences: list[float] = []
        for row in _tail_jsonl(SIGNAL_METRICS_PATH, self._signal_history):
            doctor = row.get("doctor")
            if not isinstance(doctor, Mapping):
                continue
            conf = _to_float(doctor.get("confidence"))
            if conf is not None:
                confidences.append(conf)
        if confidences:
            return max(0.0, min(1.0, float(statistics.fmean(confidences))))
        return 0.6

    def _slippage_snapshot(self) -> Dict[str, float]:
        slips: list[float] = []
        for row in _tail_jsonl(ORDER_METRICS_PATH, self._order_history):
            slip = _to_float(row.get("slippage_bps"))
            if slip is None:
                continue
            slips.append(slip)
        if not slips:
            return {"median": 0.0, "p95": 0.0, "count": 0}
        slips_sorted = sorted(slips)
        median = float(statistics.median(slips_sorted))
        p95_idx = max(0, min(len(slips_sorted) - 1, int(round(0.95 * (len(slips_sorted) - 1)))))
        p95 = float(slips_sorted[p95_idx])
        return {"median": median, "p95": p95, "count": len(slips_sorted)}

    def _apply_adjustments(self, gate: RiskGate, adjustments: Mapping[str, Any]) -> None:
        if not adjustments:
            return
        gate_cfg = gate.cfg.setdefault("risk", {})
        for key, value in adjustments.items():
            if key == "burst_limit" and isinstance(value, Mapping):
                base = dict(gate_cfg.get("burst_limit") or {})
                base.update(value)
                gate_cfg["burst_limit"] = base
                gate.risk["burst_limit"] = dict(base)
                continue
            gate_cfg[key] = value
            gate.risk[key] = value

    def _compute_adjustments(self, conf: float, slip_stats: Mapping[str, Any]) -> Dict[str, Any]:
        adjustments: Dict[str, Any] = {}
        base_daily = _to_float(self._base_risk.get("daily_loss_limit_pct")) or 5.0
        scale = max(0.4, min(1.0, conf))
        tuned_daily = round(base_daily * scale, 3)
        current_daily = _to_float(self._gate.risk.get("daily_loss_limit_pct")) or base_daily
        if abs(tuned_daily - current_daily) >= 0.05:
            adjustments["daily_loss_limit_pct"] = tuned_daily

        base_burst = self._base_risk.get("burst_limit") or {}
        burst_max = int(float(base_burst.get("max_orders", 4) or 4))
        tuned_burst = burst_max
        p95 = _to_float(slip_stats.get("p95")) or 0.0
        if p95 > 3.0:
            tuned_burst = max(1, burst_max - 1)
        elif p95 < 1.0:
            tuned_burst = burst_max
        current_burst = self._gate.risk.get("burst_limit") or {}
        if int(float(current_burst.get("max_orders", tuned_burst) or tuned_burst)) != tuned_burst:
            adjustments["burst_limit"] = {"max_orders": tuned_burst, "window_sec": int(base_burst.get("window_sec", 60) or 60)}

        base_cooldown = _to_float(self._base_risk.get("cooldown_minutes_after_stop")) or 30.0
        tuned_cooldown = base_cooldown
        if p95 > 5.0:
            tuned_cooldown = base_cooldown + 10.0
        elif p95 < 1.5:
            tuned_cooldown = base_cooldown
        current_cd = _to_float(self._gate.risk.get("cooldown_minutes_after_stop")) or base_cooldown
        if abs(tuned_cooldown - current_cd) >= 1.0:
            adjustments["cooldown_minutes_after_stop"] = round(tuned_cooldown, 2)

        return adjustments

    # ------------------------------------------------------------------ heartbeat
    def on_heartbeat(self, state: RiskState) -> Dict[str, Any]:
        with self._lock:
            conf = self._doctor_confidence()
            slip_stats = self._slippage_snapshot()
            adjustments = self._compute_adjustments(conf, slip_stats)
            if adjustments:
                self._apply_adjustments(self._gate, adjustments)
                self._last_adjust_ts = time.time()
                summary = {
                    "confidence": conf,
                    "slippage": slip_stats,
                    "adjustments": adjustments,
                    "ts": self._last_adjust_ts,
                }
                self._last_adjust_summary = summary
                LOG.info("[risk] autotune adjustments=%s", summary)
            else:
                LOG.debug("[risk] autotune no adjustments (conf=%.3f slip=%s)", conf, slip_stats)
            self.persist_state(state)
            return {
                "confidence": conf,
                "slippage": slip_stats,
                "adjustments": adjustments,
                "state_path": str(self._cache_path),
            }

    @property
    def last_adjustment(self) -> Dict[str, Any]:
        return self._last_adjust_summary

