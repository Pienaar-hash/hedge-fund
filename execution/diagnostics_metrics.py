from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class VetoCounters:
    by_reason: Dict[str, int] = field(default_factory=dict)
    total_signals: int = 0
    total_orders: int = 0
    total_vetoes: int = 0
    last_veto_ts: Optional[str] = None
    last_order_ts: Optional[str] = None
    last_signal_ts: Optional[str] = None


@dataclass
class ExitPipelineStatus:
    last_exit_scan_ts: Optional[str] = None
    last_exit_trigger_ts: Optional[str] = None
    last_router_event_ts: Optional[str] = None
    open_positions_count: int = 0
    tp_sl_registered_count: int = 0
    tp_sl_missing_count: int = 0
    underwater_without_tp_sl_count: int = 0
    tp_sl_coverage_pct: float = 0.0
    ledger_registry_mismatch: bool = False
    mismatch_breakdown: Dict[str, int] = field(default_factory=dict)


@dataclass
class LivenessAlerts:
    idle_signals: bool = False
    idle_orders: bool = False
    idle_exits: bool = False
    idle_router: bool = False
    details: Dict[str, float] = field(default_factory=dict)
    missing: Dict[str, bool] = field(default_factory=dict)


@dataclass
class RuntimeDiagnosticsSnapshot:
    veto_counters: VetoCounters
    exit_pipeline_status: ExitPipelineStatus
    liveness_alerts: Optional[LivenessAlerts] = None


_veto_counters = VetoCounters()
_exit_status = ExitPipelineStatus()


def reset_diagnostics() -> None:
    """Reset diagnostics counters (used in tests)."""
    global _veto_counters, _exit_status
    _veto_counters = VetoCounters()
    _exit_status = ExitPipelineStatus()


def get_veto_counters() -> VetoCounters:
    return _veto_counters


def get_exit_status() -> ExitPipelineStatus:
    return _exit_status


def record_signal_emitted() -> None:
    vc = _veto_counters
    vc.total_signals += 1
    vc.last_signal_ts = _now_iso()


def record_order_placed() -> None:
    vc = _veto_counters
    vc.total_orders += 1
    vc.last_order_ts = _now_iso()


def record_veto(reason: str) -> None:
    vc = _veto_counters
    key = str(reason or "unknown")
    vc.by_reason[key] = vc.by_reason.get(key, 0) + 1
    vc.total_vetoes += 1
    vc.last_veto_ts = _now_iso()


def record_exit_scan_run() -> None:
    _exit_status.last_exit_scan_ts = _now_iso()


def record_exit_trigger() -> None:
    _exit_status.last_exit_trigger_ts = _now_iso()


def record_router_event() -> None:
    _exit_status.last_router_event_ts = _now_iso()


def update_exit_pipeline_status(
    open_positions_count: int,
    tp_sl_registered_count: int,
    tp_sl_missing_count: int,
    underwater_without_tp_sl_count: int,
    *,
    tp_sl_coverage_pct: Optional[float] = None,
    ledger_registry_mismatch: Optional[bool] = None,
    mismatch_breakdown: Optional[Mapping[str, int]] = None,
) -> None:
    es = _exit_status
    es.open_positions_count = int(open_positions_count)
    es.tp_sl_registered_count = int(tp_sl_registered_count)
    es.tp_sl_missing_count = int(tp_sl_missing_count)
    es.underwater_without_tp_sl_count = int(underwater_without_tp_sl_count)
    if tp_sl_coverage_pct is not None:
        es.tp_sl_coverage_pct = float(tp_sl_coverage_pct)
    if ledger_registry_mismatch is not None:
        es.ledger_registry_mismatch = bool(ledger_registry_mismatch)
    if mismatch_breakdown is not None:
        es.mismatch_breakdown = {k: int(v) for k, v in mismatch_breakdown.items()}


def build_runtime_diagnostics_snapshot() -> RuntimeDiagnosticsSnapshot:
    return RuntimeDiagnosticsSnapshot(
        veto_counters=_veto_counters,
        exit_pipeline_status=_exit_status,
        liveness_alerts=None,
    )


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _load_strategy_config(path: str = "config/strategy_config.json") -> Dict[str, Any]:
    try:
        import json

        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def compute_liveness_alerts(cfg: Optional[Mapping[str, Any]] = None) -> LivenessAlerts:
    """
    Compute idle flags for signals/orders/exits/router based on last activity timestamps.
    """
    alerts = LivenessAlerts()
    config = cfg or {}
    enabled = bool(config.get("enabled", False))
    if not enabled:
        return alerts

    def _threshold(key: str, default: float) -> float:
        try:
            val = float(config.get(key, default))
            return max(val, 0.0)
        except Exception:
            return default

    now = datetime.now(timezone.utc)

    def check(
        ts_str: Optional[str],
        threshold: float,
        key: str,
        attr_name: str,
        *,
        force_detail: bool = False,
    ) -> None:
        record_detail = force_detail or threshold > 0
        if not record_detail:
            return
        ts = _parse_iso(ts_str)
        if ts is None:
            alerts.missing[key] = True
            alerts.details[key] = float(threshold) + 1.0
            if threshold > 0:
                setattr(alerts, attr_name, True)
            return
        delta = (now - ts).total_seconds()
        if delta < 0:
            delta = 0.0
        alerts.details[key] = delta
        if threshold > 0 and delta > threshold:
            setattr(alerts, attr_name, True)

    vc = _veto_counters
    es = _exit_status

    check(vc.last_signal_ts, _threshold("max_idle_signals_seconds", 0), "signals_idle_seconds", "idle_signals")
    check(vc.last_order_ts, _threshold("max_idle_orders_seconds", 0), "orders_idle_seconds", "idle_orders")
    check(es.last_exit_trigger_ts, _threshold("max_idle_exits_seconds", 0), "exits_idle_seconds", "idle_exits")
    check(
        es.last_router_event_ts,
        _threshold("max_idle_router_events_seconds", 0),
        "router_idle_seconds",
        "idle_router",
        force_detail=True,
    )
    return alerts


def build_runtime_diagnostics_snapshot_with_liveness(
    liveness_cfg: Optional[Mapping[str, Any]] = None,
) -> RuntimeDiagnosticsSnapshot:
    alerts = compute_liveness_alerts(liveness_cfg)
    return RuntimeDiagnosticsSnapshot(
        veto_counters=_veto_counters,
        exit_pipeline_status=_exit_status,
        liveness_alerts=alerts,
    )


__all__ = [
    "VetoCounters",
    "ExitPipelineStatus",
    "LivenessAlerts",
    "RuntimeDiagnosticsSnapshot",
    "record_signal_emitted",
    "record_order_placed",
    "record_veto",
    "record_exit_scan_run",
    "record_exit_trigger",
    "record_router_event",
    "update_exit_pipeline_status",
    "build_runtime_diagnostics_snapshot",
    "build_runtime_diagnostics_snapshot_with_liveness",
    "compute_liveness_alerts",
    "_load_strategy_config",
    "get_veto_counters",
    "get_exit_status",
    "reset_diagnostics",
]
