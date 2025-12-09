#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import sys
import os

repo_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, repo_root)

import argparse
import json
import logging
import subprocess
import time
import socket
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, cast

from execution.log_utils import append_jsonl, get_logger, log_event, safe_dump
from execution.firestore_utils import _safe_load_json
from execution.events import now_utc, write_event
from execution.pnl_tracker import CloseResult as PnlCloseResult, Fill as PnlFill, PositionTracker
from execution.universe_resolver import (
    symbol_min_gross,
    symbol_target_leverage,
    symbol_tier,
    universe_by_symbol,
)
from execution.runtime_config import load_runtime_config
from execution.versioning import read_version
from execution import telegram_alerts_v7
from execution.utils.execution_health import record_execution_error, summarize_atr_regimes
from execution.utils.metrics import router_effectiveness_7d
from execution import router_metrics
from execution.intel import maker_offset, router_autotune_shared
from execution.exchange_utils import get_price

import requests
from execution.intel.router_policy import router_policy
from execution.v6_flags import get_flags, flags_to_dict, log_v6_flag_snapshot
# Optional .env so Supervisor doesn't need to export everything
try:
    from dotenv import load_dotenv

    load_dotenv(override=True)
    load_dotenv("/root/hedge-fund/.env", override=True)
except Exception:
    pass

LOG = logging.getLogger("exutil")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s [exutil] %(message)s"
)

HOSTNAME = socket.gethostname()
RUN_ID = os.getenv("EXECUTOR_RUN_ID") or str(uuid.uuid4())
LOG_ORDERS = get_logger("logs/execution/orders_executed.jsonl")
LOG_ATTEMPTS = get_logger("logs/execution/orders_attempted.jsonl")
LOG_VETOES = get_logger("logs/execution/risk_vetoes.jsonl")
LOG_POSITION = get_logger("logs/execution/position_state.jsonl")
LOG_HEART = get_logger("logs/execution/sync_heartbeats.jsonl")
ROUTER_HEALTH_LOG = get_logger("logs/router_health.jsonl")
EXEC_HEALTH_LOG = get_logger("logs/execution/execution_health.jsonl")
_HEARTBEAT_INTERVAL = 60.0
_LAST_HEARTBEAT = 0.0
_LAST_SIGNAL_PULL = 0.0
_LAST_QUEUE_DEPTH = 0
SIGNAL_METRICS_PATH = Path("logs/execution/signal_metrics.jsonl")
ORDER_METRICS_PATH = Path("logs/execution/order_metrics.jsonl")
ROUTER_METRICS_MIRROR_PATH = Path("logs/execution/router_metrics.jsonl")
SIGNAL_METRICS_LOG = get_logger(str(SIGNAL_METRICS_PATH))
ORDER_METRICS_LOG = get_logger(str(ORDER_METRICS_PATH))
ROUTER_METRICS_LOG = get_logger(str(ROUTER_METRICS_MIRROR_PATH))
LOGS_ROOT = Path(repo_root) / "logs"
POSITIONS_CACHE_PATH = LOGS_ROOT / "positions.json"
POSITIONS_STATE_PATH = LOGS_ROOT / "state" / "positions_state.json"
NAV_LOG_CACHE_PATH = LOGS_ROOT / "nav_log.json"
SPOT_STATE_CACHE_PATH = LOGS_ROOT / "spot_state.json"
NAV_LOG_MAX_POINTS = int(os.getenv("NAV_LOG_MAX_POINTS", "720") or 720)
_INTENT_REGISTRY: Dict[str, Dict[str, Any]] = {}
_POSITION_TRACKER = PositionTracker()
_FILL_POLL_INTERVAL = float(os.getenv("ORDER_FILL_POLL_INTERVAL", "0.5") or 0.5)
_FILL_POLL_TIMEOUT = float(os.getenv("ORDER_FILL_POLL_TIMEOUT", "8.0") or 8.0)
_FILL_FINAL_STATUSES = {"FILLED", "CANCELED", "REJECTED", "EXPIRED"}
_HEALTH_PUBLISH_INTERVAL_S = float(os.getenv("EXEC_HEALTH_PUBLISH_INTERVAL", "120") or 120)
# Backwards compatibility: some call sites referred to EXEC_HEALTH_PUBLISH_INTERVAL_S.
EXEC_HEALTH_PUBLISH_INTERVAL_S = _HEALTH_PUBLISH_INTERVAL_S
_LAST_HEALTH_PUBLISH: Dict[str, float] = {}
EXEC_ALERT_INTERVAL_S = float(os.getenv("EXEC_ALERT_INTERVAL_S", "60") or 60)
_LAST_EXEC_ALERT_EVAL: Dict[str, float] = {}
EXEC_INTEL_PUBLISH_INTERVAL_S = float(os.getenv("EXEC_INTEL_PUBLISH_INTERVAL_S", "300") or 300)
_LAST_INTEL_PUBLISH: Dict[str, float] = {}
ROUTER_HEALTH_REFRESH_INTERVAL_S = float(os.getenv("ROUTER_HEALTH_REFRESH_INTERVAL_S", "60") or 60)
_LAST_ROUTER_HEALTH_PUBLISH = 0.0
_LAST_RISK_PUBLISH = 0.0
_LAST_EXEC_HEALTH_PUBLISH = 0.0
_LAST_RISK_CACHE: Dict[str, Dict[str, Any]] = {}
_LAST_ALERT_TS: Dict[str, float] = {}
_V6_FLAGS = get_flags()
_RUNTIME_CFG = load_runtime_config()
_EXEC_CFG = _RUNTIME_CFG.get("execution") if isinstance(_RUNTIME_CFG, Mapping) else {}
_MAX_TRANSIENT_RETRIES = int(
    (_EXEC_CFG or {}).get("max_transient_retries")
    or os.getenv("EXEC_MAX_TRANSIENT_RETRIES", "1")
    or 1
)
_TRANSIENT_RETRY_BACKOFF_S = float(
    (_EXEC_CFG or {}).get("transient_retry_backoff_s")
    or os.getenv("EXEC_TRANSIENT_RETRY_BACKOFF_S", "1.5")
    or 1.5
)
_ERROR_COOLDOWN_SEC = float(
    (_EXEC_CFG or {}).get("error_cooldown_sec")
    or os.getenv("EXEC_ERROR_COOLDOWN_SEC", "60")
    or 60
)
INTEL_V6_ENABLED = _V6_FLAGS.intel_v6_enabled
INTEL_V6_REFRESH_INTERVAL_S = float(os.getenv("INTEL_V6_REFRESH_INTERVAL_S", "600") or 600)
ROUTER_AUTOTUNE_V6_ENABLED = _V6_FLAGS.router_autotune_v6_enabled
ROUTER_AUTOTUNE_V6_REFRESH_INTERVAL_S = float(os.getenv("ROUTER_AUTOTUNE_V6_REFRESH_INTERVAL_S", "900") or 900)
_LAST_ROUTER_AUTOTUNE_PUBLISH = 0.0
FEEDBACK_ALLOCATOR_V6_ENABLED = _V6_FLAGS.feedback_allocator_v6_enabled
FEEDBACK_ALLOCATOR_V6_REFRESH_INTERVAL_S = float(os.getenv("FEEDBACK_ALLOCATOR_V6_REFRESH_INTERVAL_S", "1800") or 1800)
_LAST_FEEDBACK_ALLOCATOR_PUBLISH = 0.0
RISK_ENGINE_V6_ENABLED = _V6_FLAGS.risk_engine_v6_enabled
_RISK_ENGINE_V6: Optional[RiskEngineV6] = None
_RISK_ENGINE_V6_CFG_DIGEST: Optional[str] = None
PIPELINE_V6_SHADOW_ENABLED = _V6_FLAGS.pipeline_v6_shadow_enabled
PIPELINE_V6_SHADOW_RECENT = int(os.getenv("PIPELINE_V6_SHADOW_RECENT", "50") or 50)
ROUTER_AUTOTUNE_V6_APPLY_ENABLED = _V6_FLAGS.router_autotune_v6_apply_enabled
_V6_RUNTIME_PROBE_INTERVAL = float(os.getenv("V6_RUNTIME_PROBE_INTERVAL", "300") or 300)
_LAST_V6_RUNTIME_PROBE = 0.0
_PIPELINE_V6_HEARTBEAT_INTERVAL_S = float(os.getenv("PIPELINE_V6_SHADOW_HEARTBEAT_INTERVAL_S", "600") or 600)
_LAST_PIPELINE_V6_HEARTBEAT = 0.0
_PIPELINE_V6_COMPARE_INTERVAL_S = float(os.getenv("PIPELINE_V6_COMPARE_INTERVAL_S", "900") or 900)
_LAST_PIPELINE_V6_COMPARE = 0.0
_LAST_NAV_STATE: Dict[str, Any] = {}
_LAST_POSITIONS_STATE: Dict[str, Any] = {}
_LAST_RISK_SNAPSHOT: Dict[str, Any] | None = None
_LAST_EXECUTION_HEALTH: Dict[str, Any] | None = None
_LAST_SYMBOL_SCORES_STATE: Dict[str, Any] | None = None
_LAST_KPIS_V7: Dict[str, Any] | None = None
_KPI_V7_PUBLISH_INTERVAL_S = float(os.getenv("KPI_V7_PUBLISH_INTERVAL_S", str(_HEALTH_PUBLISH_INTERVAL_S)) or _HEALTH_PUBLISH_INTERVAL_S)
_LAST_KPI_PUBLISH = 0.0
_SYMBOL_ERROR_COOLDOWN: Dict[str, float] = {}

_ENGINE_VERSION = read_version(default="v7.6")


def get_v6_flag_snapshot() -> Dict[str, bool]:
    return flags_to_dict(get_flags())


def mk_id(prefix: str) -> str:
    base = prefix.strip("_") or "id"
    return f"{base}_{uuid.uuid4().hex[:10]}"


def _append_signal_metrics(record: Mapping[str, Any]) -> None:
    try:
        SIGNAL_METRICS_LOG.write(record)
    except Exception as exc:
        LOG.debug("[metrics] signal_metrics_append_failed: %s", exc)


def _append_order_metrics(record: Mapping[str, Any]) -> None:
    try:
        ORDER_METRICS_LOG.write(record)
    except Exception as exc:
        LOG.debug("[metrics] order_metrics_append_failed: %s", exc)


def _write_router_metrics_local(record: Mapping[str, Any]) -> None:
    try:
        ROUTER_METRICS_LOG.write(record)
    except Exception as exc:
        LOG.debug("[metrics] router_metrics_append_failed: %s", exc)


def _build_router_health_snapshot() -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    now = time.time()
    symbols = sorted(universe_by_symbol().keys())
    quality_counts: Dict[str, int] = {"good": 0, "ok": 0, "degraded": 0, "broken": 0}
    maker_first_enabled = 0
    symbol_stats: Dict[str, Dict[str, Any]] = {}
    risk_snapshot = router_autotune_shared.load_risk_snapshot()
    global_health = router_metrics.build_router_health_snapshot()
    for sym in symbols:
        try:
            eff = router_effectiveness_7d(sym) or {}
        except Exception as exc:
            LOG.debug("[metrics] router_effectiveness_failed symbol=%s err=%s", sym, exc)
            eff = {}
        try:
            policy = router_policy(sym)
            policy_payload = {
                "maker_first": policy.maker_first,
                "taker_bias": policy.taker_bias,
                "quality": policy.quality,
                "reason": policy.reason,
                "offset_bps": policy.offset_bps,
            }
        except Exception:
            policy_payload = {
                "maker_first": None,
                "taker_bias": None,
                "quality": None,
                "reason": None,
                "offset_bps": None,
            }
        quality_key = str(policy_payload.get("quality") or "").lower()
        if quality_key in quality_counts:
            quality_counts[quality_key] += 1
        if policy_payload.get("maker_first"):
            maker_first_enabled += 1

        events = router_metrics.get_recent_router_events(symbol=sym)
        symbol_health = router_metrics.build_router_health_snapshot(router_events=events)
        stats: Dict[str, Any] = {
            "maker_count": int(symbol_health.get("maker_count") or 0),
            "taker_count": int(symbol_health.get("taker_count") or 0),
            "fallback_count": int(symbol_health.get("fallback_count") or 0),
            "reject_count": int(symbol_health.get("reject_count") or 0),
            "order_count": int(symbol_health.get("order_count") or 0),
            "avg_slippage_bps": float(symbol_health.get("avg_slippage_bps") or 0.0),
            "maker_reliability": float(symbol_health.get("maker_reliability") or 0.0),
            "last_ts": float(symbol_health.get("ts") or now),
        }
        autotune = router_autotune_shared.suggest_autotune_for_symbol(
            sym,
            float(policy_payload.get("offset_bps") or 0.0),
            router_health={"symbol_stats": {sym: stats}},
            risk_snapshot=risk_snapshot,
            min_offset_bps=getattr(maker_offset, "MIN_OFFSET_BPS", 0.5),
        )
        stats.update(
            {
                "adaptive_offset_bps": autotune["adaptive_offset_bps"],
                "maker_first": bool(autotune["maker_first"]),
                "risk_mode": autotune.get("risk_mode"),
                "effective_reliability": autotune.get("effective_reliability"),
            }
        )
        symbol_stats[sym.upper()] = stats
        entry = {
            "symbol": sym,
            "maker_fill_rate": eff.get("maker_fill_ratio"),
            "fallback_rate": eff.get("fallback_ratio"),
            "slippage_p50": eff.get("slip_q50"),
            "slippage_p95": eff.get("slip_q95"),
            "ack_latency_ms": eff.get("latency_p50_ms") or eff.get("ack_latency_ms"),
            "policy": policy_payload,
            "maker_first": stats.get("maker_first"),
            "taker_bias": policy_payload.get("taker_bias"),
            "bias": policy_payload.get("taker_bias"),
            "quality": policy_payload.get("quality"),
            "offset_bps": autotune["adaptive_offset_bps"],
            "updated_ts": now,
            "autotune": autotune,
        }
        entries.append(entry)
    summary = {
        "updated_ts": now,
        "count": len(entries),
        "quality_counts": quality_counts,
        "maker_first_enabled": maker_first_enabled,
    }
    global_block = {
        "router_health_score": float(global_health.get("router_health_score") or 0.0),
        "last_update_ts": now,
    }
    return {
        "updated_ts": now,
        "summary": summary,
        "symbols": entries,
        "per_symbol": entries,
        "symbol_stats": symbol_stats,
        "global": global_block,
    }


def _router_quality_label(stats: Mapping[str, Any]) -> str:
    maker = stats.get("maker_fill_ratio")
    fallback = stats.get("fallback_ratio")
    slip = stats.get("slip_q50")
    quality = "unknown"
    try:
        if maker is not None and float(maker) < 0.35:
            return "poor"
        if fallback is not None and float(fallback) > 0.6:
            return "poor"
        if slip is not None and float(slip) > 6.0:
            return "poor"
        if maker is not None and float(maker) >= 0.55 and (fallback is None or float(fallback) <= 0.25):
            quality = "good"
        elif fallback is not None and float(fallback) > 0.35:
            quality = "degraded"
        else:
            quality = "ok"
    except Exception:
        quality = "unknown"
    return quality


def _collect_execution_health() -> Dict[str, Any]:
    from execution.universe_resolver import universe_by_symbol
    from execution.utils.execution_health import compute_execution_health

    snapshot_entries: List[Dict[str, Any]] = []
    now = time.time()
    for sym in sorted(universe_by_symbol().keys()):
        try:
            entry = compute_execution_health(sym)
        except Exception as exc:
            LOG.debug("[metrics] execution_health_failed symbol=%s err=%s", sym, exc)
            continue
        entry["updated_ts"] = now
        snapshot_entries.append(entry)
    return {"updated_ts": now, "symbols": snapshot_entries}


def _collect_v7_kpis() -> Dict[str, Any]:
    from execution.universe_resolver import universe_by_symbol

    now = time.time()
    symbols = sorted(universe_by_symbol().keys())
    try:
        atr_summary = summarize_atr_regimes(symbols)
    except Exception:
        atr_summary = {"atr_regime": "unknown", "median_ratio": None, "symbols": []}
    g_cfg = (_RISK_CFG.get("global") or {}) if isinstance(_RISK_CFG, Mapping) else {}
    try:
        dd_snapshot = drawdown_snapshot(g_cfg)
    except Exception:
        dd_snapshot = {}
    dd_info = dd_snapshot.get("drawdown") if isinstance(dd_snapshot, Mapping) else {}
    dd_pct = dd_info.get("pct") if isinstance(dd_info, Mapping) else None
    if dd_pct is None and isinstance(dd_snapshot, Mapping):
        dd_pct = dd_snapshot.get("dd_pct")
    alert_pct = _to_float(g_cfg.get("drawdown_alert_pct"))
    kill_pct = _to_float(g_cfg.get("max_nav_drawdown_pct") or g_cfg.get("daily_loss_limit_pct"))
    dd_state = classify_drawdown_state(dd_pct=dd_pct or 0.0, alert_pct=alert_pct, kill_pct=kill_pct)
    try:
        fees_section = fee_pnl_ratio(symbol=None, window_days=3)
    except Exception:
        fees_section = {"fee_pnl_ratio": None, "fees": 0.0, "pnl": 0.0, "window_days": 3}
    try:
        router_stats = router_effectiveness_7d(None)
    except Exception:
        router_stats = {}
    router_quality = _router_quality_label(router_stats)
    return {
        "ts": now,
        "atr_regime": atr_summary.get("atr_regime"),
        "atr": atr_summary,
        "dd_state": dd_state,
        "drawdown": dd_snapshot,
        "fee_pnl_ratio": fees_section.get("fee_pnl_ratio"),
        "fees": fees_section.get("fees"),
        "pnl": fees_section.get("pnl"),
        "fee_pnl": fees_section,
        "router_quality": router_quality,
        "router_stats": router_stats,
    }


def _pairs_cfg_path() -> str:
    return os.getenv("PAIRS_UNIVERSE_CONFIG", "config/pairs_universe.json")


def _load_pairs_cfg() -> Dict[str, Any]:
    try:
        payload = json.loads(Path(_pairs_cfg_path()).read_text())
    except Exception:
        payload = {}
    return payload if isinstance(payload, dict) else {}


def _get_risk_engine_v6() -> Optional[RiskEngineV6]:
    global _RISK_ENGINE_V6, _RISK_ENGINE_V6_CFG_DIGEST
    try:
        digest = json.dumps(_RISK_CFG, sort_keys=True, default=str)
    except Exception:
        digest = None
    if _RISK_ENGINE_V6 is None or _RISK_ENGINE_V6_CFG_DIGEST != digest:
        pairs_cfg = _load_pairs_cfg()
        _RISK_ENGINE_V6 = RiskEngineV6.from_configs(_RISK_CFG, pairs_cfg)
        _RISK_ENGINE_V6_CFG_DIGEST = digest
    return _RISK_ENGINE_V6


def _maybe_write_v6_runtime_probe(force: bool = False) -> None:
    global _LAST_V6_RUNTIME_PROBE
    flag_snapshot = get_v6_flag_snapshot()
    if not any(flag_snapshot.values()):
        return
    now = time.time()
    if not force and (now - _LAST_V6_RUNTIME_PROBE) < _V6_RUNTIME_PROBE_INTERVAL:
        return
    payload = dict(flag_snapshot)
    payload["engine_version"] = _ENGINE_VERSION
    payload["ts"] = now
    try:
        write_v6_runtime_probe_state(payload)
        _LAST_V6_RUNTIME_PROBE = now
    except Exception as exc:
        LOG.debug("[v6] runtime_probe_write_failed err=%s", exc)


def _maybe_emit_router_health_snapshot(force: bool = False) -> None:
    global _LAST_ROUTER_HEALTH_PUBLISH
    now = time.time()
    if not force and (now - _LAST_ROUTER_HEALTH_PUBLISH) < ROUTER_HEALTH_REFRESH_INTERVAL_S:
        return
    try:
        router_stats_snapshot = get_router_stats_snapshot() if get_router_stats_snapshot else {}
    except Exception as exc:
        LOG.debug("[metrics] router_stats_snapshot_failed: %s", exc)
        router_stats_snapshot = {}
    try:
        snapshot = _build_router_health_snapshot()
    except Exception as exc:
        LOG.debug("[metrics] router_health_snapshot_failed: %s", exc)
        return
    try:
        ROUTER_HEALTH_LOG.write({"ts": now, "snapshot": snapshot, "router_stats": router_stats_snapshot})
    except Exception as exc:
        LOG.debug("[metrics] router_health_log_failed: %s", exc)
    try:
        write_router_health_state(snapshot, router_stats_snapshot=router_stats_snapshot)
        _LAST_ROUTER_HEALTH_PUBLISH = now
    except Exception as exc:
        LOG.debug("[metrics] router_health_state_write_failed: %s", exc)


def _mirror_router_metrics(event: Mapping[str, Any]) -> None:
    _write_router_metrics_local(event)
    _maybe_emit_router_health_snapshot(force=True)


def _maybe_emit_risk_snapshot(force: bool = False) -> None:
    global _LAST_RISK_PUBLISH, _LAST_RISK_CACHE, _LAST_RISK_SNAPSHOT
    now = time.time()
    if not force and (now - _LAST_RISK_PUBLISH) < _HEALTH_PUBLISH_INTERVAL_S:
        return
    try:
        engine = _get_risk_engine_v6()
    except Exception:
        engine = None
    try:
        if engine is not None:
            snapshot = engine.build_risk_snapshot()
        else:
            snapshot = _collect_execution_health()
    except Exception as exc:
        LOG.debug("[metrics] execution_health_collect_failed: %s", exc)
        return
    snapshot.setdefault("updated_ts", datetime.now(timezone.utc).isoformat())
    snapshot["risk_config_meta"] = {
        "testnet_overrides_active": bool((_RISK_CFG.get("_meta") or {}).get("testnet_overrides_active")),
        "max_nav_drawdown_pct": (_RISK_CFG.get("global") or {}).get("max_nav_drawdown_pct"),
        "daily_loss_limit_pct": (_RISK_CFG.get("global") or {}).get("daily_loss_limit_pct"),
    }
    try:
        _maybe_emit_v7_kpis()
        if _LAST_KPIS_V7:
            snapshot["kpis_v7"] = _LAST_KPIS_V7
    except Exception as exc:
        LOG.debug("[metrics] kpis_v7_attach_failed: %s", exc)
    try:
        write_risk_snapshot_state(snapshot)
    except Exception as exc:
        LOG.debug("[metrics] risk_snapshot_state_write_failed: %s", exc)
    _LAST_RISK_SNAPSHOT = snapshot
    cache: Dict[str, Dict[str, Any]] = {}
    for entry in snapshot.get("symbols", []):
        sym = entry.get("symbol")
        if sym:
            cache[str(sym)] = entry
    _LAST_RISK_CACHE = cache
    _LAST_RISK_PUBLISH = now


def _maybe_emit_execution_health_snapshot(force: bool = False) -> None:
    global _LAST_EXEC_HEALTH_PUBLISH, _LAST_EXECUTION_HEALTH
    now = time.time()
    if not force and (now - _LAST_EXEC_HEALTH_PUBLISH) < _HEALTH_PUBLISH_INTERVAL_S:
        return
    try:
        snapshot = _collect_execution_health()
    except Exception as exc:
        LOG.debug("[metrics] execution_health_collect_failed: %s", exc)
        return
    try:
        snapshot.setdefault("type", "execution_health")
        snapshot.setdefault("context", "executor")
        EXEC_HEALTH_LOG.write(snapshot)
    except Exception as exc:
        LOG.debug("[metrics] execution_health_log_failed: %s", exc)
    try:
        write_execution_health_state(snapshot)
    except Exception as exc:
        LOG.debug("[metrics] execution_health_state_write_failed: %s", exc)
    _LAST_EXECUTION_HEALTH = snapshot
    _LAST_EXEC_HEALTH_PUBLISH = now


def _maybe_emit_v7_kpis(force: bool = False) -> None:
    global _LAST_KPI_PUBLISH, _LAST_KPIS_V7
    now = time.time()
    if not force and (now - _LAST_KPI_PUBLISH) < _KPI_V7_PUBLISH_INTERVAL_S:
        return
    try:
        payload = _collect_v7_kpis()
    except Exception as exc:
        LOG.debug("[metrics] kpis_v7_collect_failed: %s", exc)
        return
    payload.setdefault("ts", now)
    try:
        write_kpis_v7_state(payload)
    except Exception as exc:
        LOG.debug("[metrics] kpis_v7_state_write_failed: %s", exc)
    _LAST_KPIS_V7 = payload
    _LAST_KPI_PUBLISH = now


def _maybe_run_telegram_alerts() -> None:
    try:
        context = {
            "risk_snapshot": _LAST_RISK_SNAPSHOT or {},
            "nav_snapshot": (_LAST_NAV_STATE.get("nav_detail") or _LAST_NAV_STATE),
            "kpis_snapshot": _LAST_KPIS_V7 or {},
            "now_ts": time.time(),
        }
        telegram_alerts_v7.run_alerts(context)
    except Exception as exc:
        LOG.debug("[telegram] alerts_v7_failed: %s", exc)


def _maybe_emit_execution_alerts(symbol: str) -> None:
    if not symbol:
        return
    entry = _LAST_RISK_CACHE.get(symbol)
    if not entry:
        return
    router_part = entry.get("router") or {}
    risk_part = entry.get("risk") or {}
    warnings = router_part.get("router_warnings") or []
    risk_flags = risk_part.get("risk_flags") or []
    if not warnings and not risk_flags:
        return
    now = time.time()
    key = f"{symbol}:{','.join(sorted(warnings + risk_flags))}"
    last_ts = _LAST_ALERT_TS.get(key, 0.0)
    if (now - last_ts) < EXEC_ALERT_INTERVAL_S:
        return
    payload = {
        "symbol": symbol,
        "router_warnings": warnings,
        "risk_flags": risk_flags,
        "ts": now,
    }
    try:
        append_jsonl(LOGS_ROOT / "execution" / "execution_alerts.jsonl", payload)
        _LAST_ALERT_TS[key] = now
    except Exception as exc:
        LOG.debug("[metrics] execution_alert_write_failed: %s", exc)


def _maybe_publish_execution_intel() -> None:
    global _LAST_SYMBOL_SCORES_STATE
    now = time.time()
    last = _LAST_INTEL_PUBLISH.get("universe", 0.0)
    if (now - last) >= EXEC_INTEL_PUBLISH_INTERVAL_S:
        try:
            from execution.universe_resolver import write_universe_snapshot

            write_universe_snapshot()
            _LAST_INTEL_PUBLISH["universe"] = now
        except Exception as exc:
            LOG.debug("[metrics] universe_snapshot_failed: %s", exc)
    if not INTEL_V6_ENABLED:
        return
    last_analysis = _LAST_INTEL_PUBLISH.get("analysis", 0.0)
    if (now - last_analysis) < INTEL_V6_REFRESH_INTERVAL_S:
        return
    try:
        from execution.intel import expectancy_v6, symbol_score_v6, router_autotune_v6, feedback_allocator_v6
        from execution.state_publish import (
            write_expectancy_state,
            write_symbol_scores_state,
            write_router_policy_suggestions_state,
            write_risk_allocation_suggestions_state,
            compute_and_write_rv_momentum_state,
            compute_and_write_factor_diagnostics_state,
        )

        trades = expectancy_v6.load_trade_records(lookback_hours=48.0)
        router_metrics = expectancy_v6.load_router_metrics(lookback_hours=48.0)
        trades = expectancy_v6.merge_trades_with_policy(trades, router_metrics)
        expectancy_snapshot = expectancy_v6.build_expectancy_snapshot(trades, 48.0)
        write_expectancy_state(expectancy_snapshot)
        router_health = _build_router_health_snapshot()
        scores_snapshot = symbol_score_v6.build_symbol_scores(expectancy_snapshot, router_health)
        scores_snapshot.setdefault("updated_ts", now)
        write_symbol_scores_state(scores_snapshot)
        # RV momentum state (v7.5_C1)
        compute_and_write_rv_momentum_state()
        # Factor diagnostics state (v7.5_C2)
        compute_and_write_factor_diagnostics_state(scores_snapshot)
        _LAST_SYMBOL_SCORES_STATE = dict(scores_snapshot)
        router_suggestions_payload = None
        if ROUTER_AUTOTUNE_V6_ENABLED:
            global _LAST_ROUTER_AUTOTUNE_PUBLISH
            if (now - _LAST_ROUTER_AUTOTUNE_PUBLISH) >= ROUTER_AUTOTUNE_V6_REFRESH_INTERVAL_S:
                lookback_hours = float(expectancy_snapshot.get("lookback_hours") or 0.0)
                lookback_days_val = lookback_hours / 24.0 if lookback_hours > 0 else 7.0
                suggestions = router_autotune_v6.build_suggestions(
                    expectancy_snapshot=expectancy_snapshot,
                    symbol_scores_snapshot=scores_snapshot,
                    router_health_snapshot=router_health,
                    risk_config=_RISK_CFG,
                    lookback_days=lookback_days_val,
                )
                write_router_policy_suggestions_state(suggestions)
                _LAST_ROUTER_AUTOTUNE_PUBLISH = now
                router_suggestions_payload = suggestions
        if FEEDBACK_ALLOCATOR_V6_ENABLED:
            global _LAST_FEEDBACK_ALLOCATOR_PUBLISH
            if (now - _LAST_FEEDBACK_ALLOCATOR_PUBLISH) >= FEEDBACK_ALLOCATOR_V6_REFRESH_INTERVAL_S:
                allocator_payload = feedback_allocator_v6.build_suggestions(
                    expectancy_snapshot=expectancy_snapshot,
                    symbol_scores_snapshot=scores_snapshot,
                    router_policy_snapshot=router_suggestions_payload,
                    risk_config=_RISK_CFG,
                )
                write_risk_allocation_suggestions_state(allocator_payload)
                _LAST_FEEDBACK_ALLOCATOR_PUBLISH = now
        _LAST_INTEL_PUBLISH["analysis"] = now
    except Exception as exc:
        LOG.debug("[intel] analysis_publish_failed: %s", exc)

# ---- Exchange utils (binance) ----
from execution.exchange_utils import (
    _req,
    _is_dual_side,
    build_order_payload,
    get_balances,
    get_positions,
    get_price,
    is_testnet,
    send_order,
    set_dry_run,
    should_use_close_position,
    get_um_client,
    um_client_error,
)
try:
    from execution.order_router import (
        route_order as _route_order,
        route_intent as _route_intent,
        submit_limit,
        effective_px,
        PlaceOrderResult,
        get_router_stats_snapshot,
    )
except Exception:
    _route_order = None  # type: ignore[assignment]
    _route_intent = None  # type: ignore[assignment]
    submit_limit = None  # type: ignore[assignment]
    effective_px = None  # type: ignore[assignment]
    PlaceOrderResult = None  # type: ignore[assignment]
    get_router_stats_snapshot = None  # type: ignore[assignment]
from execution.risk_limits import (
    RiskState,
    check_order,
    classify_drawdown_state,
    drawdown_snapshot,
)
from execution.risk_loader import load_risk_config
from execution.risk_engine_v6 import OrderIntent, RiskEngineV6
from execution.nav import compute_nav_pair, PortfolioSnapshot, nav_health_snapshot
from execution import pipeline_v6_shadow
from execution.utils import (
    load_json,
    write_nav_snapshots_pair,
    save_json,
    get_live_positions,
)
from execution.utils.metrics import fee_pnl_ratio, router_effectiveness_7d

from execution.signal_generator import (
    normalize_intent as generator_normalize_intent,
)
from execution.signal_screener import generate_intents
from execution import signal_doctor
from execution.state_publish import (
    build_synced_state_payload,
    write_nav_state,
    write_pipeline_v6_shadow_state,
    write_positions_state,
    write_positions_snapshot_state,
    write_positions_ledger_state,
    write_risk_snapshot_state,
    write_router_health_state,
    write_kpis_v7_state,
    write_symbol_scores_state,
    write_execution_health_state,
    write_synced_state,
    write_v6_runtime_probe_state,
    write_engine_metadata_state,
    compute_and_write_risk_advanced_state,
    compute_and_write_alpha_decay_state,
    write_runtime_diagnostics_state,
)
try:
    from execution.signal_screener import run_once as run_screener_once
except ImportError:  # pragma: no cover - optional dependency
    run_screener_once = None
# ---- Firestore publisher handle (revisions differ) ----
def _resolve_env(default: str = "dev") -> str:
    raw = (os.getenv("ENV") or os.getenv("ENVIRONMENT") or "").strip()
    if not raw:
        return default
    return raw


ENV = _resolve_env()
if ENV.lower() == "prod":
    allow_prod = os.getenv("ALLOW_PROD_WRITE", "0").strip().lower()
    if allow_prod not in {"1", "true", "yes"}:
        raise RuntimeError(
            "Refusing to run executor_live with ENV=prod. "
            "Set ALLOW_PROD_WRITE=1 to override explicitly."
        )
def publish_close_audit(*_args, **_kwargs) -> None:  # type: ignore[override]
    return None


def publish_intent_audit(*_args, **_kwargs) -> None:  # type: ignore[override]
    return None


def publish_order_audit(*_args, **_kwargs) -> None:  # type: ignore[override]
    return None


# Delegate nav publishing to state_publish to keep local/state files fresh.
try:
    from execution import state_publish as _state_publish

    def publish_nav_value(*args, **kwargs) -> None:  # type: ignore[override]
        return _state_publish.publish_nav_value(*args, **kwargs)
except Exception:
    def publish_nav_value(*_args, **_kwargs) -> None:  # type: ignore[override]
        return None

# ---- risk limits config ----
_RISK_CFG: Dict[str, Any] = {}
try:
    _RISK_CFG = load_risk_config()
except Exception as e:
    logging.getLogger("exutil").warning(
        "[risk] config load failed: %s", e
    )


def _load_registry_entries() -> Dict[str, Dict[str, Any]]:
    try:
        payload = load_json("config/strategy_registry.json") or {}
    except Exception:
        payload = {}
    raw = payload.get("strategies") if isinstance(payload, Mapping) else payload
    if not isinstance(raw, Mapping):
        return {}
    entries: Dict[str, Dict[str, Any]] = {}
    for key, value in raw.items():
        if isinstance(value, Mapping):
            entries[str(key)] = dict(value)
    return entries


def _strategy_concurrency_budget() -> int:
    registry = _load_registry_entries()
    total = 0
    if registry:
        for entry in registry.values():
            if not entry.get("enabled") or entry.get("sandbox"):
                continue
            try:
                val = int(float(entry.get("max_concurrent", 0) or 0))
            except Exception:
                val = 0
            if val <= 0:
                val = 1
            total += val
    if total <= 0:
        cfg = load_json("config/strategy_config.json") or {}
        strategies = cfg.get("strategies") or []
        for entry in strategies:
            if not isinstance(entry, Mapping) or not entry.get("enabled"):
                continue
            total += 1
    return max(total, 0)


_PAIRS_CFG_PATH = os.getenv("PAIRS_UNIVERSE_CONFIG", "config/pairs_universe.json")
_PAIRS_CFG: Dict[str, Any] = {}
try:
    with open(_PAIRS_CFG_PATH, "r") as fh:
        _PAIRS_CFG = json.load(fh) or {}
except Exception as e:
    logging.getLogger("exutil").warning(
        "[pairs] config load failed (%s): %s", _PAIRS_CFG_PATH, e
    )


def _apply_strategy_concurrency(risk_cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(risk_cfg, dict):
        return {}
    derived = _strategy_concurrency_budget()
    if derived <= 0:
        return risk_cfg
    global_cfg = risk_cfg.setdefault("global", {})
    try:
        existing = int(float(global_cfg.get("max_concurrent_positions") or 0))
    except Exception:
        existing = 0
    if existing <= 0 or derived < existing:
        global_cfg["max_concurrent_positions"] = derived
    return risk_cfg


def _refresh_risk_config() -> None:
    global _RISK_CFG, _RISK_ENGINE_V6, _RISK_ENGINE_V6_CFG_DIGEST
    try:
        if hasattr(load_risk_config, "cache_clear"):
            load_risk_config.cache_clear()  # type: ignore[attr-defined]
        cfg = load_risk_config()
    except Exception as exc:
        LOG.warning("[risk] config refresh failed: %s", exc)
        return
    cfg = _apply_strategy_concurrency(cfg)
    _RISK_CFG = cfg
    _RISK_ENGINE_V6 = None
    _RISK_ENGINE_V6_CFG_DIGEST = None


_RISK_CFG = _apply_strategy_concurrency(_RISK_CFG)
_RISK_STATE = RiskState()
_PORTFOLIO_SNAPSHOT = PortfolioSnapshot(load_json("config/strategy_config.json"))


def _nav_pct_fraction(value: Any) -> float:
    """Interpret numeric percent inputs as fractions; 10 -> 0.10, 0.02 -> 0.02."""
    try:
        pct = float(value)
    except (TypeError, ValueError):
        return 0.0
    if pct <= 0.0:
        return 0.0
    if pct > 1.0:
        return pct / 100.0
    return pct


def _normalize_pct_value(value: Any) -> float:
    try:
        pct = float(value)
    except (TypeError, ValueError):
        return 0.0
    if pct <= 0.0:
        return 0.0
    if pct > 1.0:
        return pct / 100.0
    return pct


def _size_from_nav(symbol: str, nav_usd: float, pct: float) -> float:
    """Compute gross notional from NAV * pct fraction."""
    try:
        return float(nav_usd) * float(pct)
    except Exception:
        return 0.0


def _clamp_intent_gross(symbol: str, gross: float, nav_usd: float, floor_gross: float) -> float:
    """Pass-through clamp: honor floor only (caps enforced in risk engine)."""
    return max(float(gross), float(floor_gross))

# ---- knobs ----
SLEEP = int(os.getenv("LOOP_SLEEP", "60"))
MAX_LOOPS = int(os.getenv("MAX_LOOPS", "0") or 0)
SCREENER_INTERVAL = int(os.getenv("SCREENER_INTERVAL", "300") or 300)
_LAST_SCREENER_RUN = 0.0


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "describe", "--tags", "--always"])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _startup_flags() -> Dict[str, Any]:
    testnet = is_testnet()
    dry_run = os.getenv("DRY_RUN", "0").lower() in ("1", "true", "yes")
    env = ENV
    base = "https://testnet.binancefuture.com" if testnet else "https://fapi.binance.com"
    fs_enabled = bool(int(os.getenv("FIRESTORE_ENABLED", "0") or 0))
    return {
        "testnet": testnet,
        "dry_run": dry_run,
        "env": env,
        "base": base,
        "fs_enabled": fs_enabled,
    }


def _log_startup_summary() -> Dict[str, Any]:
    flags = _startup_flags()
    prefix = "testnet" if flags["testnet"] else "live"
    LOG.info(
        "[%s] ENV=%s DRY_RUN=%s testnet=%s base=%s FIRESTORE=%s",
        prefix,
        flags["env"],
        int(flags["dry_run"]),
        flags["testnet"],
        flags["base"],
        "ON" if flags["fs_enabled"] else "OFF",
    )
    flags["prefix"] = prefix
    return flags


def _read_dry_run_flag() -> bool:
    return os.getenv("DRY_RUN", "1").lower() in ("1", "true", "yes")


def _truthy_env(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes")


def _publish_startup_heartbeat(flags: Dict[str, Any]) -> None:
    LOG.info("[%s] Firestore heartbeat skipped (disabled)", flags.get("prefix", "live"))


DRY_RUN = _read_dry_run_flag()
set_dry_run(DRY_RUN)
INTENT_TEST = _truthy_env("INTENT_TEST", "0")
EXTERNAL_SIGNAL = _truthy_env("EXTERNAL_SIGNAL", "0")

LOG.info(
    "[executor] starting loop ENV=%s DRY_RUN=%s commit=%s signal_source=generate_intents unified=True",
    ENV,
    DRY_RUN,
    _git_commit(),
)

try:
    _startup_flags_snapshot = _log_startup_summary()
except Exception as exc:
    LOG.exception("[live] startup summary logging failed: %s", exc)
    _startup_flags_snapshot = {
        "env": ENV,
        "fs_enabled": bool(int(os.getenv("FIRESTORE_ENABLED", "0") or 0)),
        "prefix": "live",
    }

_publish_startup_heartbeat(_startup_flags_snapshot)


def _sync_dry_run() -> None:
    global DRY_RUN
    current = _read_dry_run_flag()
    if current != DRY_RUN:
        LOG.info("[executor] DRY_RUN flag changed -> %s", current)
        DRY_RUN = current
    set_dry_run(DRY_RUN)


def _clean_testnet_caches() -> None:
    flag = str(os.getenv("BINANCE_TESTNET", "0")).strip().lower()
    if flag not in {"1", "true", "yes", "on"}:
        return
    cache_paths = [
        Path(repo_root) / "logs" / "cache" / "risk_state.json",
        Path(repo_root) / "logs" / "cache" / "nav_confirmed.json",
    ]
    for path in cache_paths:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            LOG.debug("[executor][testnet] cache cleanup skipped for %s", path)
    LOG.info("[executor][testnet] cleaned stale risk/nav cache for fresh start")


def _coerce_veto_reasons(raw: Any) -> List[str]:
    if not raw:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, Sequence):
        return [str(item) for item in raw if item]
    return [str(raw)]


def _normalize_intent(intent: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = generator_normalize_intent(intent)
    normalized.setdefault("tf", normalized.get("timeframe"))
    return normalized


def _symbol_on_cooldown(symbol: str, now: Optional[float] = None) -> bool:
    current = _SYMBOL_ERROR_COOLDOWN.get(str(symbol).upper())
    if current is None:
        return False
    ts = float(now) if now is not None else time.time()
    if ts >= current:
        _SYMBOL_ERROR_COOLDOWN.pop(str(symbol).upper(), None)
        return False
    return True


def _mark_symbol_cooldown(symbol: str, now: Optional[float] = None, seconds: float = _ERROR_COOLDOWN_SEC) -> float:
    ts = float(now) if now is not None else time.time()
    expiry = ts + float(seconds)
    _SYMBOL_ERROR_COOLDOWN[str(symbol).upper()] = expiry
    return expiry


def _publish_intent_audit(symbol: Optional[str], intent: Dict[str, Any]) -> None:
    LOG.info("[screener->executor] %s", intent)
    if not symbol:
        return
    payload = dict(intent)
    payload.setdefault("symbol", symbol)
    payload.setdefault("ts", time.time())
    return None


def _publish_veto_exec(symbol: Optional[str], reasons: Sequence[str], intent: Mapping[str, Any]) -> None:
    reasons_list = [str(r) for r in reasons if r]
    LOG.info("[screener] veto symbol=%s reasons=%s", symbol, reasons_list)
    payload = {
        "symbol": symbol,
        "reasons": reasons_list,
        "intent": dict(intent),
        "ts": time.time(),
    }
    try:
        save_json(f"logs/veto_exec_{(symbol or 'UNKNOWN').upper()}.json", payload)
    except Exception:
        pass
    return None


# ------------- helpers -------------
def _record_structured_event(logger_obj: Any, event_type: str, payload: Mapping[str, Any] | None) -> None:
    try:
        sanitized = safe_dump(payload or {})
        sanitized.setdefault("type", event_type)
        sanitized.setdefault("context", "executor")
        log_event(logger_obj, event_type, sanitized)
    except Exception as exc:
        LOG.debug("structured_log_failed event=%s err=%s", event_type, exc)


def _log_order_error(
    *,
    symbol: str,
    side: str,
    notional: float | None,
    reason: str,
    classification: Optional[Mapping[str, Any]] = None,
    retried: bool = False,
    exc: Optional[BaseException] = None,
    component: str = "executor",
    context: Optional[Mapping[str, Any]] = None,
) -> None:
    payload = {
        "symbol": symbol,
        "side": side,
        "notional": notional,
        "reason": reason,
        "retried": retried,
        "classification": dict(classification or {}),
        "last_exception_message": str(exc) if exc else None,
    }
    if context:
        payload["context"] = dict(context)
    _record_structured_event(LOG_ORDERS, "order_error", payload)
    try:
        record_execution_error(
            component,
            symbol=symbol,
            message=reason,
            classification=dict(classification or {}),
            context=dict(context or {}),
        )
    except Exception:
        LOG.debug("[health] record_execution_error_failed")


@dataclass
class OrderAckInfo:
    symbol: str
    side: str
    order_type: str
    request_qty: Optional[float]
    position_side: Optional[str]
    reduce_only: bool
    order_id: Optional[int]
    client_order_id: Optional[str]
    status: str
    latency_ms: Optional[float]
    attempt_id: Optional[str] = None
    intent_id: Optional[str] = None
    ts_ack: str = ""


@dataclass
class FillSummary:
    executed_qty: float
    avg_price: Optional[float]
    status: str
    fee_total: float
    fee_asset: Optional[str]
    trade_ids: List[Any]
    ts_fill_first: Optional[str]
    ts_fill_last: Optional[str]
    latency_ms: Optional[float] = None


def _normalize_status(status: Any) -> str:
    if not status:
        return "UNKNOWN"
    try:
        value = str(status).upper()
    except Exception:
        return "UNKNOWN"
    if value == "CANCELLED":
        return "CANCELED"
    return value


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _ms_to_iso(value: Any) -> Optional[str]:
    try:
        if value is None:
            return None
        val = float(value)
    except (TypeError, ValueError):
        return None
    if val <= 0:
        return None
    if val > 1e12:
        val /= 1000.0
    try:
        return datetime.fromtimestamp(val, tz=timezone.utc).isoformat()
    except Exception:
        return None


def _iso_to_ts(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _fetch_order_status(symbol: str, order_id: Optional[int], client_order_id: Optional[str]) -> Dict[str, Any]:
    params: Dict[str, Any] = {"symbol": symbol}
    if order_id:
        params["orderId"] = int(order_id)
    elif client_order_id:
        params["origClientOrderId"] = client_order_id
    else:
        return {}
    try:
        resp = _req("GET", "/fapi/v1/order", signed=True, params=params, timeout=6.0)
        return resp.json() or {}
    except Exception as exc:
        LOG.debug("[fills] order_status_fetch_failed symbol=%s order_id=%s err=%s", symbol, order_id, exc)
        return {}


def _fetch_order_trades(symbol: str, order_id: Optional[int]) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {"symbol": symbol}
    if order_id:
        params["orderId"] = int(order_id)
    try:
        resp = _req("GET", "/fapi/v1/userTrades", signed=True, params=params, timeout=6.0)
        data = resp.json() or []
        if not isinstance(data, list):
            return []
        return data
    except Exception as exc:
        LOG.debug("[fills] order_trades_fetch_failed symbol=%s order_id=%s err=%s", symbol, order_id, exc)
        return []


def _emit_order_ack(
    symbol: str,
    side: str,
    order_type: str,
    request_qty: Optional[float],
    position_side: Optional[str],
    reduce_only: bool,
    resp: Mapping[str, Any],
    *,
    latency_ms: Optional[float],
    attempt_id: Optional[str],
    intent_id: Optional[str],
) -> Optional[OrderAckInfo]:
    status = _normalize_status(resp.get("status"))
    order_id_raw = resp.get("orderId")
    try:
        order_id = int(order_id_raw) if order_id_raw is not None else None
    except (TypeError, ValueError):
        order_id = None
    client_order_id = resp.get("clientOrderId") or resp.get("orderId")
    if not order_id and not client_order_id:
        return None
    ts_ack = now_utc()

    ack = OrderAckInfo(
        symbol=symbol,
        side=str(side).upper(),
        order_type=str(order_type).upper(),
        request_qty=_to_float(request_qty),
        position_side=position_side,
        reduce_only=bool(reduce_only),
        order_id=order_id,
        client_order_id=str(client_order_id) if client_order_id is not None else None,
        status=status,
        latency_ms=_to_float(latency_ms),
        attempt_id=attempt_id,
        intent_id=intent_id,
        ts_ack=ts_ack,
    )
    payload: Dict[str, Any] = {
        "symbol": ack.symbol,
        "side": ack.side,
        "ts_ack": ts_ack,
        "orderId": ack.order_id,
        "clientOrderId": ack.client_order_id,
        "request_qty": ack.request_qty,
        "order_type": ack.order_type,
        "status": ack.status,
    }
    if ack.position_side:
        payload["positionSide"] = ack.position_side
    if ack.reduce_only:
        payload["reduceOnly"] = True
    if ack.latency_ms is not None:
        payload["latency_ms"] = ack.latency_ms
    if attempt_id:
        payload["attempt_id"] = attempt_id
    if intent_id:
        payload["intent_id"] = intent_id
    try:
        write_event("order_ack", payload)
    except Exception as exc:
        LOG.debug("[events] ack_write_failed %s %s", payload.get("orderId"), exc)
    return ack


def _should_emit_close(ack: OrderAckInfo, close_results: List[PnlCloseResult]) -> bool:
    if not close_results:
        return False
    if ack.reduce_only:
        return True
    pos_before = close_results[0].position_before
    pos_after = close_results[-1].position_after
    if abs(pos_after) < 1e-8:
        return True
    if pos_before == 0.0:
        return False
    return pos_before * pos_after <= 0.0


def _confirm_order_fill(
    ack: OrderAckInfo,
    metadata: Optional[Mapping[str, Any]] = None,
    strategy: Optional[str] = None,
) -> Optional[FillSummary]:
    if not ack.order_id and not ack.client_order_id:
        return None
    start = time.time()
    seen_trade_ids: set[str] = set()
    executed_qty = 0.0
    cum_quote = 0.0
    fee_total = 0.0
    fee_asset: Optional[str] = None
    ts_first: Optional[str] = None
    ts_last: Optional[str] = None
    status = ack.status
    last_summary: Optional[FillSummary] = None
    fill_latency_ms: Optional[float] = None

    metadata_payload: Optional[Dict[str, Any]] = None
    if isinstance(metadata, Mapping):
        try:
            metadata_payload = dict(metadata)
        except Exception:
            metadata_payload = None

    while (time.time() - start) <= _FILL_POLL_TIMEOUT:
        status_resp = _fetch_order_status(ack.symbol, ack.order_id, ack.client_order_id)
        if status_resp:
            status = _normalize_status(status_resp.get("status"))

        trades = _fetch_order_trades(ack.symbol, ack.order_id)
        new_trades: List[Dict[str, Any]] = []
        for trade in trades:
            trade_id = trade.get("id")
            if trade_id is None:
                continue
            trade_id_str = str(trade_id)
            if trade_id_str in seen_trade_ids:
                continue
            seen_trade_ids.add(trade_id_str)
            new_trades.append(trade)
            qty = _to_float(trade.get("qty")) or 0.0
            price = _to_float(trade.get("price")) or 0.0
            executed_qty += qty
            cum_quote += qty * price
            commission = _to_float(trade.get("commission")) or 0.0
            fee_total += commission
            fee_asset = fee_asset or trade.get("commissionAsset") or trade.get("marginAsset") or "USDT"
            trade_ts = _ms_to_iso(trade.get("time"))
            now_iso = now_utc()
            if trade_ts:
                if ts_first is None or trade_ts < ts_first:
                    ts_first = trade_ts
                if ts_last is None or trade_ts > ts_last:
                    ts_last = trade_ts
            else:
                if ts_first is None:
                    ts_first = now_iso
                ts_last = now_iso

        if new_trades:
            avg_price = (cum_quote / executed_qty) if executed_qty else None
            fill_payload: Dict[str, Any] = {
                "symbol": ack.symbol,
                "side": ack.side,
                "ts_fill_first": ts_first or now_utc(),
                "ts_fill_last": ts_last or now_utc(),
                "orderId": ack.order_id,
                "clientOrderId": ack.client_order_id,
                "executedQty": executed_qty,
                "avgPrice": avg_price,
                "fee_total": fee_total,
                "feeAsset": fee_asset or "USDT",
                "tradeIds": sorted(seen_trade_ids),
                "status": status,
            }
            if strategy:
                fill_payload["strategy"] = strategy
            if metadata_payload:
                fill_payload["metadata"] = metadata_payload
            if ack.position_side:
                fill_payload["positionSide"] = ack.position_side
            if ack.reduce_only:
                fill_payload["reduceOnly"] = True
            if ack.attempt_id:
                fill_payload["attempt_id"] = ack.attempt_id
            if ack.intent_id:
                fill_payload["intent_id"] = ack.intent_id
            try:
                write_event("order_fill", fill_payload)
            except Exception as exc:
                LOG.debug("[events] fill_write_failed %s %s", ack.order_id, exc)

            close_results: List[PnlCloseResult] = []
            for trade in new_trades:
                qty = _to_float(trade.get("qty")) or 0.0
                if qty <= 0:
                    continue
                price = _to_float(trade.get("price")) or 0.0
                commission = _to_float(trade.get("commission")) or 0.0
                fill_obj = PnlFill(
                    symbol=ack.symbol,
                    side=ack.side,
                    qty=qty,
                    price=price,
                    fee=commission,
                    position_side=ack.position_side,
                    reduce_only=ack.reduce_only,
                )
                close_res = _POSITION_TRACKER.apply_fill(fill_obj)
                if close_res:
                    close_results.append(close_res)

            if _should_emit_close(ack, close_results):
                total_realized = sum(r.realized_pnl for r in close_results)
                total_fees = sum(r.fees for r in close_results)
                pos_before = close_results[0].position_before if close_results else 0.0
                pos_after = close_results[-1].position_after if close_results else 0.0
                closed_qty = sum(r.closed_qty for r in close_results)
                close_payload: Dict[str, Any] = {
                    "symbol": ack.symbol,
                    "ts_close": ts_last or now_utc(),
                    "orderId": ack.order_id,
                    "clientOrderId": ack.client_order_id,
                    "realizedPnlUsd": total_realized,
                    "fees_total": total_fees,
                    "position_size_before": pos_before,
                    "position_size_after": pos_after,
                }
                if strategy:
                    close_payload["strategy"] = strategy
                if metadata_payload:
                    close_payload["metadata"] = metadata_payload
                if ack.position_side:
                    close_payload["positionSide"] = ack.position_side
                if closed_qty > 0:
                    close_payload["closed_qty"] = closed_qty
                if ack.attempt_id:
                    close_payload["attempt_id"] = ack.attempt_id
                if ack.intent_id:
                    close_payload["intent_id"] = ack.intent_id
                try:
                    write_event("order_close", close_payload)
                except Exception as exc:
                    LOG.debug("[events] close_write_failed %s %s", ack.order_id, exc)

            if ts_last:
                ack_ts = _iso_to_ts(ack.ts_ack)
                fill_ts = _iso_to_ts(ts_last)
                if ack_ts is not None and fill_ts is not None:
                    fill_latency_ms = max(0.0, (fill_ts - ack_ts) * 1000.0)

            last_summary = FillSummary(
                executed_qty=executed_qty,
                avg_price=avg_price,
                status=status,
                fee_total=fee_total,
                fee_asset=fee_asset,
                trade_ids=sorted(seen_trade_ids),
                ts_fill_first=ts_first,
                ts_fill_last=ts_last,
                latency_ms=fill_latency_ms,
            )

        if status in _FILL_FINAL_STATUSES:
            break
        if not new_trades:
            time.sleep(_FILL_POLL_INTERVAL)

    return last_summary


def _nav_snapshot() -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {}
    health = nav_health_snapshot()
    snapshot["nav_usd"] = float(health.get("nav_total") or 0.0)
    snapshot["nav_age_s"] = health.get("age_s")
    snapshot["nav_sources_ok"] = health.get("sources_ok")
    return snapshot


def _estimate_intent_qty(intent: Mapping[str, Any], gross_target: float, price_hint: float) -> float:
    for key in ("quantity", "qty", "order_qty", "orderQty", "size", "units"):
        if key in intent:
            try:
                return float(intent[key])
            except Exception:
                continue
    try:
        normalized = intent.get("normalized")
        if isinstance(normalized, Mapping) and "qty" in normalized:
            return float(normalized.get("qty") or 0.0)
    except Exception:
        pass
    try:
        if price_hint and price_hint > 0:
            return float(gross_target) / float(price_hint)
    except Exception:
        pass
    return float(intent.get("qty_estimate", 0.0) or 0.0)


def compute_final_gross_for_test(
    intent: Mapping[str, Any],
    nav_usd: float,
    size_risk_cfg: Mapping[str, Any],
) -> float:
    """Pure helper for tests to mirror the v6 sizing clamps without side effects."""
    try:
        symbol = str(intent.get("symbol") or intent.get("pair") or "").upper()
    except Exception:
        symbol = ""
    lev = float(intent.get("leverage", 1.0) or 1.0)
    per_trade_nav_pct = _nav_pct_fraction(intent.get("per_trade_nav_pct"))
    intent_min_notional = _to_float(intent.get("min_notional")) or 0.0
    try:
        screener_gross = float(intent.get("gross_usd") or 0.0)
    except Exception:
        screener_gross = 0.0
    gross_target = float(intent.get("gross_usd") or 0.0)
    if nav_usd is None:
        nav_usd = 0.0
    floor_gross = max(
        float(size_risk_cfg.get("min_notional_usd") or 0.0),
        intent_min_notional,
        symbol_min_gross(symbol),
    )
    if gross_target <= 0.0 and per_trade_nav_pct > 0.0:
        gross_target = _size_from_nav(symbol, nav_usd, per_trade_nav_pct)
    gross_target = max(gross_target, floor_gross)
    if screener_gross > 0.0:
        gross_target = min(gross_target, screener_gross)
    if gross_target < floor_gross:
        return 0.0
    return float(gross_target)


def _position_rows_for_symbol(symbol: str) -> List[Dict[str, Any]]:
    try:
        positions = list(get_positions() or [])
    except Exception:
        positions = []
    symbol_upper = str(symbol).upper()
    rows: List[Dict[str, Any]] = []
    for pos in positions:
        try:
            if str(pos.get("symbol", "")).upper() != symbol_upper:
                continue
            rows.append(dict(pos))
        except Exception:
            continue
    return rows


def _emit_position_snapshots(symbol: str) -> None:
    rows = _position_rows_for_symbol(symbol)
    ts = time.time()
    for row in rows:
        payload = {
            "symbol": symbol,
            "pos_qty": row.get("qty", row.get("positionAmt")),
            "entry_px": row.get("entryPrice"),
            "unrealized_pnl": row.get("unRealizedProfit"),
            "leverage": row.get("leverage"),
            "mode": row.get("positionSide", row.get("marginType")),
            "ts": ts,
            "run_id": RUN_ID,
            "hostname": HOSTNAME,
        }
        _record_structured_event(LOG_POSITION, "position_snapshot", payload)


def _maybe_emit_heartbeat() -> None:
    global _LAST_HEARTBEAT
    now = time.time()
    if (now - _LAST_HEARTBEAT) < _HEARTBEAT_INTERVAL:
        return
    _LAST_HEARTBEAT = now
    lag = None
    if _LAST_SIGNAL_PULL > 0:
        lag = max(0.0, now - _LAST_SIGNAL_PULL)
    payload: Dict[str, Any] = {
        "service": "executor_live",
        "run_id": RUN_ID,
        "hostname": HOSTNAME,
        "ts": now,
    }
    if lag is not None:
        payload["lag_secs"] = lag
    if _LAST_QUEUE_DEPTH is not None:
        payload["queue_depth"] = _LAST_QUEUE_DEPTH
    _record_structured_event(LOG_HEART, "heartbeat", payload)


def _sync_tp_sl_registry(positions: list) -> None:
    """
    Sync TP/SL registry with current positions at startup.
    
    V7.4_C3: Uses position_ledger.sync_ledger_with_positions() for unified
    ledger-based sync. Ensures all open positions have TP/SL registered,
    and removes entries for positions that no longer exist.
    """
    try:
        from execution.position_ledger import sync_ledger_with_positions
        from execution.utils import load_json
        
        # Load ATR multipliers from strategy config
        strategy_cfg = load_json("config/strategy_config.json") or {}
        sl_atr_mult = strategy_cfg.get("sl_atr_mult", 2.0)
        tp_atr_mult = strategy_cfg.get("tp_atr_mult", 3.0)
        
        ledger = sync_ledger_with_positions(
            seed_missing=True,
            remove_stale=True,
            sl_atr_mult=sl_atr_mult,
            tp_atr_mult=tp_atr_mult,
        )
        
        LOG.info("[startup-ledger] position ledger synced: %d entries", len(ledger))
    except Exception as exc:
        LOG.warning("[startup-sync] position ledger sync failed: %s", exc)
        # Fallback to legacy registry sync
        try:
            from execution.position_tp_sl_registry import sync_registry_with_positions
            from execution.utils import load_json
            
            strategy_cfg = load_json("config/strategy_config.json") or {}
            sl_atr_mult = strategy_cfg.get("sl_atr_mult", 2.0)
            tp_atr_mult = strategy_cfg.get("tp_atr_mult", 3.0)
            
            result = sync_registry_with_positions(positions, sl_atr_mult, tp_atr_mult)
            
            if result["stale_removed"] > 0 or result["new_seeded"] > 0:
                LOG.info(
                    "[startup-sync] TP/SL registry synced (fallback): %d stale removed, %d new seeded",
                    result["stale_removed"],
                    result["new_seeded"],
                )
        except Exception as exc2:
            LOG.warning("[startup-sync] fallback registry sync also failed: %s", exc2)


def _startup_position_check(client: Any) -> None:
    if client is None or getattr(client, "is_stub", False):
        LOG.info("[startup-sync] unable to check positions (client unavailable)")
        return
    
    # Allow skipping the blocking check if ALLOW_OPEN_POSITIONS=1
    allow_open = os.getenv("ALLOW_OPEN_POSITIONS", "0") == "1"
    
    LOG.info("[startup-sync] checking open positions …")
    retry_interval = 30
    first_warning = True

    while True:
        live = get_live_positions(client)
        if not live:
            if not first_warning:
                LOG.info("[startup-sync] all positions cleared -> resuming trading loop")
            else:
                LOG.info("[startup-sync] no open positions detected")
            # V7.4_C2: Sync registry even when no positions (cleanup stale entries)
            _sync_tp_sl_registry([])
            return

        LOG.warning(
            "[startup-sync] open positions detected (n=%d) -> trading init paused; will retry every %ss",
            len(live),
            retry_interval,
        )
        for pos in live:
            LOG.warning(
                "[startup-sync] %s side=%s amt=%.6f entry=%.4f upnl=%.2f",
                pos.get("symbol"),
                pos.get("positionSide"),
                pos.get("positionAmt"),
                pos.get("entryPrice"),
                pos.get("unRealizedProfit"),
            )
        
        # If ALLOW_OPEN_POSITIONS=1, proceed with existing positions
        if allow_open:
            LOG.info("[startup-sync] ALLOW_OPEN_POSITIONS=1 -> proceeding with %d open positions", len(live))
            # V7.4_C2: Sync TP/SL registry with current positions
            _sync_tp_sl_registry(live)
            return
        
        first_warning = False
        time.sleep(retry_interval)


def _maybe_run_internal_screener() -> None:
    global _LAST_SCREENER_RUN
    if EXTERNAL_SIGNAL or run_screener_once is None:
        return
    now = time.time()
    if (now - _LAST_SCREENER_RUN) < SCREENER_INTERVAL:
        return
    _LAST_SCREENER_RUN = now
    try:
        result = run_screener_once()
    except Exception as exc:
        LOG.error("[executor] internal screener failed: %s", exc)
        return

    attempted: Any = None
    emitted: Any = None

    if isinstance(result, dict):
        attempted = (
            result.get("attempted")
            or result.get("attempted_24h")
            or result.get("attempts")
        )
        emitted = (
            result.get("emitted")
            or result.get("emitted_24h")
            or result.get("count")
        )
    elif isinstance(result, tuple) and len(result) >= 2:
        attempted, emitted = result[0], result[1]
    elif isinstance(result, list):
        emitted = len(result)
    elif hasattr(result, "attempted") and hasattr(result, "emitted"):
        attempted = getattr(result, "attempted", None)
        emitted = getattr(result, "emitted", None)

    if attempted is None and emitted is not None:
        attempted = emitted

    intents = result.get("intents") if isinstance(result, Mapping) else []
    if not isinstance(intents, list):
        intents = []

    submitted = 0
    for entry in intents:
        symbol: Optional[str] = None
        now_ts = time.time()
        try:
            payload = entry.get("raw") if isinstance(entry, Mapping) else entry
            intent = _normalize_intent(payload)
            symbol = cast(Optional[str], intent.get("symbol"))
            if not symbol:
                continue
            if _symbol_on_cooldown(symbol, now_ts):
                continue
            _publish_intent_audit(symbol, intent)
            _send_order(intent)
            submitted += 1
        except Exception as exc:
            LOG.error("[executor] internal screener submit failed: %s", exc)
            if symbol:
                cooldown_until = _mark_symbol_cooldown(symbol, now_ts)
                LOG.warning(
                    "[screener][cooldown] symbol=%s entering temporary cooldown due to API failure",
                    symbol,
                )
                intent.setdefault("cooldown_until", cooldown_until)

    global _LAST_SIGNAL_PULL, _LAST_QUEUE_DEPTH
    _LAST_SIGNAL_PULL = time.time()
    _LAST_QUEUE_DEPTH = len(intents)

    LOG.info(
        "[screener] attempted=%s emitted=%s submitted=%d",
        attempted if attempted is not None else "n/a",
        emitted if emitted is not None else "n/a",
        submitted,
    )



def _account_snapshot() -> None:
    try:
        bals = get_balances() or []
        assets = sorted({b.get("asset", "?") for b in bals})
        pos = [
            p
            for p in get_positions()
            if float(p.get("qty", p.get("positionAmt", 0)) or 0) != 0
        ]
        LOG.info(
            "[executor] account OK — futures=%s testnet=%s dry_run=%s balances: %s positions: %d",
            True,
            is_testnet(),
            DRY_RUN,
            assets,
            len(pos),
        )
    except Exception as e:
        LOG.exception("[executor] preflight_error: %s", e)


def _compute_nav() -> float:
    cfg = load_json("config/strategy_config.json") or {}
    try:
        runtime_cfg = load_runtime_config()
        if isinstance(runtime_cfg, Mapping):
            rt_nav = runtime_cfg.get("nav")
            if isinstance(rt_nav, Mapping):
                cfg.setdefault("nav", {})
                cfg["nav"].update(rt_nav)
    except Exception:
        pass

    try:
        trading, reporting = compute_nav_pair(cfg)
        write_nav_snapshots_pair(trading, reporting)
        return float(trading[0])
    except Exception as exc:
        LOG.error("[executor] compute_nav not available: %s", exc)

    nav_val = 0.0
    try:
        from execution.exchange_utils import get_account

        acc = get_account()
        nav_val = float(
            acc.get("totalMarginBalance")
            or (
                float(acc.get("totalWalletBalance", 0) or 0)
                + float(acc.get("totalUnrealizedProfit", 0) or 0)
            )
        )
    except Exception as e:
        LOG.error("[executor] account NAV error: %s", e)
    # As a last resort try capital_base
    if not nav_val and cfg:
        return float(cfg.get("capital_base_usdt", 0.0) or 0.0)
    return float(nav_val or 0.0)


def _compute_nav_with_detail() -> tuple[float, Dict[str, Any]]:
    cfg = load_json("config/strategy_config.json") or {}
    try:
        runtime_cfg = load_runtime_config()
        if isinstance(runtime_cfg, Mapping):
            rt_nav = runtime_cfg.get("nav")
            if isinstance(rt_nav, Mapping):
                cfg.setdefault("nav", {})
                cfg["nav"].update(rt_nav)
    except Exception:
        pass

    try:
        trading, reporting = compute_nav_pair(cfg)
        write_nav_snapshots_pair(trading, reporting)
        nav_val = float(trading[0])
        detail = {}
        try:
            detail = dict(trading[1] or {})
        except Exception:
            detail = {}
        detail.setdefault("source", (trading[1] or {}).get("source") if isinstance(trading[1], Mapping) else None)
        detail.setdefault("nav", nav_val)
        detail.setdefault("nav_usd", nav_val)
        try:
            import logging

            logger = logging.getLogger(__name__)
            logger.info(
                "[nav-detail] nav_total=%s aum_total=%s future_only_nav=%s",
                detail.get("nav_total") or nav_val,
                (detail.get("aum") or {}).get("total") if isinstance(detail.get("aum"), Mapping) else None,
                detail.get("nav_total") or nav_val,
            )
        except Exception:
            pass
        return nav_val, detail
    except Exception as exc:
        LOG.error("[executor] compute_nav_with_detail not available: %s", exc)
        return _compute_nav(), {}


def _gross_and_open_qty(
    symbol: str, pos_side: str, positions: Iterable[Dict[str, Any]]
) -> tuple[float, float]:
    gross = 0.0
    open_qty = 0.0
    for p in positions or []:
        try:
            qty = float(p.get("qty", p.get("positionAmt", 0)) or 0.0)
            entry = float(p.get("entryPrice") or 0.0)
            gross += abs(qty) * abs(entry)
            if str(p.get("symbol")) == symbol:
                ps = p.get("positionSide", "BOTH")
                if ps == pos_side or (ps == "BOTH" and pos_side in ("LONG", "SHORT")):
                    open_qty = max(open_qty, abs(qty))
        except Exception:
            continue
    return gross, open_qty


def _symbol_gross_notional(symbol: str, pos_side: str, positions: Iterable[Dict[str, Any]]) -> float:
    sym_key = str(symbol).upper()
    total = 0.0
    for p in positions or []:
        try:
            if str(p.get("symbol", "")).upper() != sym_key:
                continue
            qty = float(p.get("qty", p.get("positionAmt", 0.0)) or 0.0)
            if qty == 0.0:
                continue
            ps = str(p.get("positionSide", "BOTH")).upper()
            if ps != "BOTH" and ps != str(pos_side).upper():
                continue
            entry = float(p.get("entryPrice") or p.get("markPrice") or 0.0)
            if entry <= 0.0:
                continue
            total += abs(qty) * abs(entry)
        except Exception:
            continue
    return total


def _opposite_position(
    symbol: str, desired_side: str, positions: Iterable[Dict[str, Any]]
) -> tuple[str | None, float, float]:
    sym = str(symbol).upper()
    desired = str(desired_side).upper()
    opposite = "SHORT" if desired == "LONG" else "LONG"
    for p in positions or []:
        try:
            if str(p.get("symbol", "")).upper() != sym:
                continue
            ps = str(p.get("positionSide", "BOTH")).upper()
            if ps != opposite:
                continue
            qty = float(p.get("qty", p.get("positionAmt", 0.0)) or 0.0)
            if qty == 0.0:
                continue
            mark = float(p.get("markPrice") or p.get("entryPrice") or 0.0)
            return opposite, abs(qty), abs(mark)
        except Exception:
            continue
    return None, 0.0, 0.0


def _update_risk_state_counters(
    positions: Iterable[Dict[str, Any]],
    portfolio_gross: float,
) -> None:
    open_positions = 0
    for p in positions or []:
        try:
            qty = float(p.get("qty", p.get("positionAmt", 0.0)) or 0.0)
        except Exception:
            qty = 0.0
        if qty != 0.0:
            open_positions += 1

    _RISK_STATE.open_notional = float(portfolio_gross)
    _RISK_STATE.open_positions = int(open_positions)
    _RISK_STATE.daily_pnl_pct = 0.0


def _current_bucket_gross(symbol_gross: Mapping[str, float], buckets: Mapping[str, str]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for sym, gross in symbol_gross.items():
        try:
            bucket = buckets.get(str(sym).upper())
        except Exception:
            bucket = None
        if not bucket:
            continue
        try:
            totals[bucket] = totals.get(bucket, 0.0) + float(gross)
        except Exception:
            continue
    return totals


def _build_order_intent_for_executor(
    symbol: str,
    side: str,
    *,
    gross_notional: float,
    nav: float,
    sym_open_qty: float,
    current_gross: float,
    open_positions_count: int,
    tier_name: Optional[str],
    current_tier_gross: float,
    lev: float,
    intent: Mapping[str, Any],
) -> OrderIntent:
    price = float(intent.get("price") or 0.0)
    qty = _to_float(intent.get("qty"))
    if (qty is None or qty <= 0.0) and price > 0.0:
        qty = (gross_notional / price)
    return OrderIntent(
        symbol=str(symbol).upper(),
        side=str(side).upper(),
        qty=qty,
        quote_notional=float(gross_notional),
        leverage=float(lev),
        price=price,
        tier_name=tier_name,
        tier_gross_notional=float(current_tier_gross),
        current_gross_notional=float(current_gross),
        symbol_open_qty=float(sym_open_qty),
        nav_usd=float(nav),
        open_positions_count=int(open_positions_count),
        strategy_id=str(intent.get("strategy_id") or intent.get("strategy") or "") or None,
        account_mode=str(intent.get("account_mode") or intent.get("marginMode") or "") or None,
        metadata={"intent": dict(intent)},
    )


def _evaluate_order_risk(
    symbol: str,
    side: str,
    *,
    gross_target: float,
    nav: float,
    sym_open_qty: float,
    current_gross: float,
    current_symbol_gross: float = 0.0,
    open_positions_count: int,
    tier_name: Optional[str],
    current_tier_gross: float,
    lev: float,
    reduce_only: bool,
    intent: Mapping[str, Any],
) -> tuple[bool, Mapping[str, Any]]:
    if reduce_only:
        return False, {}
    engine: Optional[RiskEngineV6] = None
    if RISK_ENGINE_V6_ENABLED:
        try:
            engine = _get_risk_engine_v6()
        except Exception:
            engine = None
    if engine is not None and RISK_ENGINE_V6_ENABLED:
        order_intent = _build_order_intent_for_executor(
            symbol,
            side,
            gross_notional=gross_target,
            nav=nav,
            sym_open_qty=sym_open_qty,
            current_gross=current_symbol_gross,
            open_positions_count=open_positions_count,
            tier_name=tier_name,
            current_tier_gross=current_tier_gross,
            lev=lev,
            intent=intent,
        )
        decision = engine.check_order(order_intent, _RISK_STATE)
        return (not decision.allowed), decision.diagnostics or {}
    risk_veto, details = check_order(
        symbol=symbol,
        side=side,
        requested_notional=gross_target,
        price=float(intent.get("price") or 0.0),
        nav=nav,
        open_qty=sym_open_qty,
        now=time.time(),
        cfg=_RISK_CFG,
        state=_RISK_STATE,
        current_gross_notional=current_symbol_gross,
        lev=lev,
        open_positions_count=open_positions_count,
        tier_name=tier_name,
        current_tier_gross_notional=current_tier_gross,
    )
    return risk_veto, details


def _record_shadow_decision(result: Mapping[str, Any]) -> None:
    pipeline_v6_shadow.append_shadow_decision(result)
    summary = pipeline_v6_shadow.build_shadow_summary(
        pipeline_v6_shadow.load_shadow_decisions(limit=PIPELINE_V6_SHADOW_RECENT)
    )
    summary["last_decision"] = result
    write_pipeline_v6_shadow_state(summary)


def _select_shadow_symbol(positions: Iterable[Mapping[str, Any]]) -> Optional[str]:
    for entry in positions or []:
        try:
            sym = str(entry.get("symbol") or "").upper()
        except Exception:
            sym = ""
        if sym:
            return sym
    try:
        universe = universe_by_symbol()
        for sym in sorted(universe.keys()):
            if sym:
                return sym
    except Exception:
        return None
    return None


def _maybe_run_pipeline_v6_shadow(
    symbol: str,
    side: str,
    *,
    gross_target: float,
    nav: float,
    sym_open_qty: float,
    current_gross: float,
    open_positions_count: int,
    tier_name: Optional[str],
    current_tier_gross: float,
    lev: float,
    intent: Mapping[str, Any],
    nav_snapshot: Mapping[str, Any],
    positions: Iterable[Mapping[str, Any]],
    sizing_cfg: Mapping[str, Any],
) -> None:
    if not PIPELINE_V6_SHADOW_ENABLED:
        return
    try:
        nav_state = dict(nav_snapshot or {})
        nav_state.setdefault("nav_usd", nav)
        nav_state.setdefault("portfolio_gross_usd", current_gross)
        nav_state.setdefault("symbol_open_qty", sym_open_qty)
        signal_payload = {
            "side": side,
            "notional": gross_target,
            "price": float(intent.get("price") or 0.0),
            "leverage": lev,
            "tier": tier_name,
            "open_positions_count": open_positions_count,
            "tier_gross_notional": current_tier_gross,
            "current_gross_notional": current_gross,
            "symbol_open_qty": sym_open_qty,
            "signal_strength": intent.get("signal_strength") or intent.get("confidence"),
        }
        positions_state = {"positions": list(positions or [])}
        result = pipeline_v6_shadow.run_pipeline_v6_shadow(
            symbol,
            signal_payload,
            nav_state,
            positions_state,
            _RISK_CFG,
            _PAIRS_CFG,
            sizing_cfg,
            risk_engine=_get_risk_engine_v6() if RISK_ENGINE_V6_ENABLED else None,
        )
        _record_shadow_decision(result)
    except Exception as exc:
        LOG.debug("[shadow] pipeline_v6_failed symbol=%s err=%s", symbol, exc)


def _maybe_run_pipeline_v6_shadow_heartbeat() -> None:
    global _LAST_PIPELINE_V6_HEARTBEAT
    if not PIPELINE_V6_SHADOW_ENABLED:
        return
    if not _LAST_NAV_STATE or not _LAST_POSITIONS_STATE:
        return
    now = time.time()
    if (now - _LAST_PIPELINE_V6_HEARTBEAT) < _PIPELINE_V6_HEARTBEAT_INTERVAL_S:
        return
    raw_positions = _LAST_POSITIONS_STATE.get("items") or _LAST_POSITIONS_STATE.get("positions") or []
    positions_rows = list(raw_positions)
    symbol = _select_shadow_symbol(positions_rows)
    if not symbol:
        return
    nav_state = dict(_LAST_NAV_STATE)
    nav_state.setdefault(
        "nav_usd",
        float(nav_state.get("nav_usd") or nav_state.get("nav") or 0.0),
    )
    nav_state.setdefault("portfolio_gross_usd", nav_state.get("portfolio_gross_usd") or 0.0)
    nav_state.setdefault("symbol_open_qty", 0.0)
    signal_payload = {
        "side": "BUY",
        "notional": 0.0,
        "price": 0.0,
        "leverage": 1.0,
        "tier": None,
        "open_positions_count": len(positions_rows),
        "tier_gross_notional": 0.0,
        "current_gross_notional": nav_state.get("portfolio_gross_usd") or 0.0,
        "symbol_open_qty": 0.0,
        "signal_strength": 0.0,
    }
    try:
        sizing_cfg = _RISK_CFG.get("sizing", {}) if isinstance(_RISK_CFG, Mapping) else {}
        result = pipeline_v6_shadow.run_pipeline_v6_shadow(
            symbol,
            signal_payload,
            nav_state,
            {"positions": positions_rows},
            _RISK_CFG,
            _PAIRS_CFG,
            sizing_cfg,
            risk_engine=_get_risk_engine_v6() if RISK_ENGINE_V6_ENABLED else None,
        )
        heartbeat_result = dict(result)
        heartbeat_result["heartbeat"] = True
        _record_shadow_decision(heartbeat_result)
        _LAST_PIPELINE_V6_HEARTBEAT = now
    except Exception as exc:
        LOG.debug("[shadow] pipeline_v6_heartbeat_failed symbol=%s err=%s", symbol, exc)
        try:
            record_execution_error(
                "pipeline_shadow",
                symbol=symbol,
                message="heartbeat_failed",
                context={"error": str(exc)},
            )
        except Exception:
            pass


def _maybe_run_pipeline_v6_compare(force: bool = False) -> None:
    global _LAST_PIPELINE_V6_COMPARE
    now = time.time()
    if not force and (now - _LAST_PIPELINE_V6_COMPARE) < _PIPELINE_V6_COMPARE_INTERVAL_S:
        return
    try:
        from execution.intel import pipeline_v6_compare

        pipeline_v6_compare.compare_pipeline_v6()
        _LAST_PIPELINE_V6_COMPARE = now
    except Exception as exc:
        LOG.debug("[shadow] pipeline_v6_compare_failed: %s", exc)
        try:
            record_execution_error(
                "pipeline_compare",
                symbol=None,
                message="compare_failed",
                context={"error": str(exc)},
            )
        except Exception:
            pass


# NOTE: _send_order must only pass canonical order fields into send_order.
def _send_order(intent: Dict[str, Any], *, skip_flip: bool = False) -> None:
    symbol = intent["symbol"]
    symbol_upper = str(symbol).upper()
    sig = str(intent.get("signal", "")).upper()
    side = "BUY" if sig == "BUY" else "SELL"
    reduce_only = bool(intent.get("reduceOnly", False))

    pos_side = intent.get("positionSide")
    if not pos_side:
        if reduce_only:
            # Reduce-only orders must target the opposite hedge leg.
            pos_side = "SHORT" if side == "BUY" else "LONG"
        else:
            pos_side = "LONG" if side == "BUY" else "SHORT"
        intent["positionSide"] = pos_side
        LOG.debug(
            "[executor] derived positionSide=%s side=%s reduce_only=%s symbol=%s",
            pos_side,
            side,
            reduce_only,
            symbol,
        )
        if reduce_only:
            LOG.info(
                "[send_order] reduceOnly=True inferred_side=%s symbol=%s",
                pos_side,
                symbol,
            )
    attempt_id = str(intent.get("attempt_id") or mk_id("sig"))
    intent_id = str(intent.get("intent_id") or mk_id("ord"))
    intent["attempt_id"] = attempt_id
    intent["intent_id"] = intent_id
    per_trade_nav_pct = _nav_pct_fraction(intent.get("per_trade_nav_pct"))
    try:
        screener_gross = float(intent.get("gross_usd") or 0.0)
    except Exception:
        screener_gross = 0.0
    cap = float(intent.get("capital_per_trade", 0) or 0)
    lev = float(intent.get("leverage", 1) or 1)
    if lev <= 0:
        lev = 1.0
    gross_from_intent = float(intent.get("gross_usd") or 0.0)
    gross_target = float(gross_from_intent or (cap * lev))
    intent_min_notional = _to_float(intent.get("min_notional")) or 0.0
    price_guess = 0.0
    try:
        price_guess = float(intent.get("price", 0.0) or 0.0)
    except Exception:
        price_guess = 0.0
    generated_at = intent.get("generated_at") or intent.get("signal_ts")
    decision_latency_ms: Optional[float] = None
    if generated_at is not None:
        try:
            gen_val = float(generated_at)
            if gen_val > 1e12:
                gen_val = gen_val / 1000.0
            decision_latency_ms = max(0.0, (time.time() - gen_val) * 1000.0)
        except Exception:
            decision_latency_ms = None
    try:
        _PORTFOLIO_SNAPSHOT.refresh()
    except Exception:
        pass
    nav_snapshot = _nav_snapshot()
    nav_usd = float(nav_snapshot.get("nav_usd", 0.0) or 0.0)
    using_nav_pct = False
    symbol_gross_map: Dict[str, float] = {}
    try:
        raw_map = nav_snapshot.get("symbol_gross_usd") or {}
        if isinstance(raw_map, Mapping):
            for key, value in raw_map.items():
                try:
                    symbol_gross_map[str(key).upper()] = float(value)
                except Exception:
                    continue
    except Exception:
        symbol_gross_map = {}
    symbol_buckets: Dict[str, str] = {}
    tier_name = symbol_tier(symbol)
    tier_gross_map: Dict[str, float] = {}
    current_tier_gross = 0.0
    per_symbol_limits: Dict[str, Dict[str, Any]] = {}
    sym_limits: Dict[str, Any] = {}
    sym_max_order = 0.0
    sym_max_nav_pct = 0.0

    def _persist_veto(reason: str, price_hint: float, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "positionSide": pos_side,
            "reason": reason,
            "gross_usd": gross_target,
            "price": price_hint,
            "ts": time.time(),
        }
        if extra:
            payload.update(extra)
        try:
            save_json(f"logs/veto_exec_{symbol}.json", payload)
        except Exception:
            try:
                import pathlib as _pl

                _pl.Path("logs").mkdir(exist_ok=True)
                _pl.Path(f"logs/veto_exec_{symbol}.json").write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )
            except Exception:
                pass
        thresholds: Dict[str, Any] = {}
        observations: Dict[str, Any] = {}
        diag_gate = None
        diagnostics: Dict[str, Any] = {}
        if extra:
            maybe_thresholds = extra.get("thresholds")
            if isinstance(maybe_thresholds, Mapping):
                thresholds = dict(maybe_thresholds)
            maybe_diag = extra.get("diagnostics")
            if isinstance(maybe_diag, Mapping):
                diagnostics = dict(maybe_diag)
                diag_gate = diagnostics.get("gate")
                diag_thresholds = diagnostics.get("thresholds")
                diag_obs = diagnostics.get("observations")
                if isinstance(diag_thresholds, Mapping):
                    thresholds.update(diag_thresholds)
                if isinstance(diag_obs, Mapping):
                    observations = dict(diag_obs)
        log_payload = {
            "symbol": symbol,
            "side": side,
            "position_side": pos_side,
            "run_id": RUN_ID,
            "hostname": HOSTNAME,
            "veto_reason": reason,
            "veto_detail": extra or {},
            "thresholds": thresholds,
            "observations": observations,
            "gate": diag_gate or (extra or {}).get("gate"),
            "ts": payload.get("ts"),
        }
        _record_structured_event(LOG_VETOES, "risk_veto", log_payload)
        return payload

    cfg = load_json("config/strategy_config.json") or {}
    sizing_cfg = (cfg.get("sizing") or {})
    floor_gross = max(
        symbol_min_gross(symbol.upper()),
        intent_min_notional,
        float((_RISK_CFG.get("global") or {}).get("min_notional_usdt", 0.0) or 0.0),
    )
    gross_target = float(intent.get("gross_usd") or gross_target or 0.0)
    margin_target = gross_target / max(lev, 1.0)
    attempt_start_monotonic = time.monotonic()
    attempt_payload = {
        "symbol": symbol,
        "side": side,
        "attempt_id": attempt_id,
        "intent_id": intent_id,
        "qty": _estimate_intent_qty(intent, gross_target, price_guess),
        "strategy": (
            intent.get("strategy")
            or intent.get("strategy_name")
            or intent.get("strategyId")
            or intent.get("source")
        ),
        "signal_ts": (
            intent.get("signal_ts")
            or intent.get("timestamp")
            or intent.get("ts")
            or intent.get("time")
        ),
        "local_ts": time.time(),
        "nav_snapshot": nav_snapshot,
        "run_id": RUN_ID,
        "hostname": HOSTNAME,
        "reduce_only": reduce_only,
        "price_hint": price_guess,
    }
    attempt_payload["decision_latency_ms"] = decision_latency_ms
    try:
        attempt_payload["confidence"] = float(intent.get("confidence", 1.0) or 1.0)
    except Exception:
        attempt_payload["confidence"] = 1.0
    attempt_payload["expected_edge"] = float(intent.get("expected_edge", 0.0) or 0.0)
    _record_structured_event(LOG_ATTEMPTS, "order_attempt", attempt_payload)

    if os.environ.get("KILL_SWITCH", "0").lower() in ("1", "true", "yes", "on"):
        price_hint = float(intent.get("price", 0.0) or 0.0)
        LOG.warning("[risk] kill switch active; veto %s %s", symbol, side)
        _persist_veto(
            "kill_switch_on",
            price_hint,
            {
                "intent": intent,
                "nav_snapshot": nav_snapshot,
                "thresholds": {"kill_switch": True},
            },
        )
        try:
            publish_order_audit(
                symbol,
                {
                    "phase": "blocked",
                    "side": side,
                    "positionSide": pos_side,
                    "reason": "kill_switch_on",
                    "notional": gross_target,
                },
            )
        except Exception:
            pass
        return

    LOG.info(
        "[executor] INTENT symbol=%s side=%s ps=%s gross=%.4f margin=%.4f lev=%.2f reduceOnly=%s screener_cap=%.4f",
        symbol,
        side,
        pos_side,
        gross_target,
        margin_target,
        lev,
        reduce_only,
        screener_gross or 0.0,
    )

    # Risk gating handled centrally in risk_engine_v6; executor does not re-evaluate caps.

    try:
        positions = list(get_positions() or [])
    except Exception:
        positions = []

    def _dispatch(payload: Dict[str, Any]) -> Dict[str, Any]:
        request_payload = dict(payload)
        order_type = str(request_payload.get("type") or "MARKET").upper()
        convert_close, _close_qty = should_use_close_position(
            request_payload.get("symbol"),
            request_payload.get("side"),
            request_payload.get("positionSide"),
            request_payload.get("reduceOnly"),
            order_type=order_type,
            positions=positions,
        )
        if convert_close and order_type != "MARKET":
            request_payload.pop("reduceOnly", None)
            request_payload.pop("quantity", None)
            request_payload["closePosition"] = True
        ro_val = request_payload.get("reduceOnly")
        if isinstance(ro_val, str):
            ro_val = ro_val.lower() in ("1", "true", "yes", "on")
        return send_order(
            symbol=request_payload["symbol"],
            side=request_payload["side"],
            type=request_payload.get("type", "MARKET"),
            quantity=request_payload.get("quantity"),
            positionSide=request_payload.get("positionSide"),
            reduceOnly=ro_val,
            price=request_payload.get("price"),
            closePosition=request_payload.get("closePosition"),
            timeInForce=request_payload.get("timeInForce"),
            newClientOrderId=request_payload.get("newClientOrderId"),
            positions=positions,
        )

    force_direct_send = False

    # Flip handling: flatten any opposing hedge-mode position via a dedicated
    # reduce-only order, then fall through to emit the new opening leg.
    if not reduce_only and not skip_flip:
        opp_side, opp_qty, opp_mark = _opposite_position(symbol, pos_side, positions)
        if opp_side and opp_qty > 0:
            reduce_price = opp_mark if opp_mark > 0 else float(intent.get("price", 0.0) or 0.0)
            if reduce_price <= 0:
                try:
                    reduce_price = float(get_price(symbol))
                except Exception:
                    reduce_price = 0.0
            if reduce_price <= 0:
                LOG.warning("Invalid reduce_price, skipping reduce-only intent")
                return

            reduce_notional = abs(opp_qty) * reduce_price
            reduce_signal = "BUY" if opp_side == "SHORT" else "SELL"
            try:
                reduce_payload, reduce_meta = build_order_payload(
                    symbol=symbol,
                    side=reduce_signal,
                    price=reduce_price,
                    desired_gross_usd=reduce_notional,
                    reduce_only=True,
                    position_side=opp_side,
                )
            except Exception as exc:
                LOG.error("[executor] reduce_only_build_failed %s %s", symbol, exc)
                return

            LOG.info(
                "[executor] flip flatten symbol=%s side=%s notional=%.4f",
                symbol,
                opp_side,
                reduce_notional,
            )
            reduce_resp: Dict[str, Any] = {}
            try:
                reduce_resp = _dispatch(reduce_payload)
            except Exception as exc:
                LOG.error("[executor] reduce_only_send_failed %s %s", symbol, exc)
                return
            try:
                publish_order_audit(
                    symbol,
                    {
                        "phase": "flip_reduce",
                        "side": reduce_signal,
                        "positionSide": opp_side,
                        "payload": {
                            k: reduce_payload[k]
                            for k in ("type", "quantity", "reduceOnly", "positionSide")
                            if k in reduce_payload
                        },
                        "normalized": {
                            "price": reduce_meta.get("normalized_price"),
                            "qty": reduce_meta.get("normalized_qty"),
                        },
                    },
                )
            except Exception:
                pass
            if reduce_resp:
                reduce_latency_ms = max(
                    0.0, (time.monotonic() - attempt_start_monotonic) * 1000.0
                )
                reduce_ack = _emit_order_ack(
                    symbol=symbol,
                    side=reduce_signal,
                    order_type=reduce_payload.get("type") or "MARKET",
                    request_qty=_to_float(reduce_payload.get("quantity")),
                    position_side=opp_side,
                    reduce_only=True,
                    resp=reduce_resp,
                    latency_ms=reduce_latency_ms,
                    attempt_id=f"{attempt_id}_reduce",
                    intent_id=intent_id,
                )
                reduce_fill: Optional[FillSummary] = None
                if reduce_ack and not reduce_resp.get("dryRun"):
                    reduce_fill = _confirm_order_fill(
                        reduce_ack,
                        intent.get("metadata"),
                        intent.get("strategy") or intent.get("strategy_id"),
                    )
                if reduce_ack:
                    LOG.info(
                        "[executor] FLIP_ACK id=%s status=%s qty=%s",
                        reduce_ack.order_id or reduce_ack.client_order_id,
                        reduce_ack.status,
                        reduce_ack.request_qty,
                    )
                if reduce_fill:
                    LOG.info(
                        "[executor] FLIP_FILL id=%s status=%s avgPrice=%s qty=%s",
                        reduce_ack.order_id if reduce_ack else reduce_resp.get("orderId"),
                        reduce_fill.status,
                        reduce_fill.avg_price,
                        reduce_fill.executed_qty,
                    )
                    if reduce_fill.executed_qty:
                        try:
                            _RISK_STATE.note_fill(symbol, time.time())
                        except Exception:
                            pass
                        _emit_position_snapshots(symbol)

            try:
                positions = list(get_positions() or [])
            except Exception:
                positions = []
            opp_after_side, opp_after_qty, _ = _opposite_position(symbol, pos_side, positions)
            if opp_after_qty > 0:
                LOG.warning(
                    "[executor] flip flatten incomplete symbol=%s side=%s qty=%.6f",
                    symbol,
                    opp_after_side,
                    opp_after_qty,
                )
                return
            force_direct_send = True
    nav = _compute_nav()
    current_gross, sym_open_qty = _gross_and_open_qty(symbol, pos_side, positions)
    current_symbol_gross = _symbol_gross_notional(symbol, pos_side, positions)
    _update_risk_state_counters(positions, current_gross)
    open_positions_count = getattr(_RISK_STATE, "open_positions", 0)
    try:
        nav_snapshot = {**nav_snapshot}
    except Exception:
        nav_snapshot = dict(nav_snapshot or {})
    nav_snapshot.setdefault("nav_usd", nav)
    nav_snapshot["portfolio_gross_usd"] = current_gross
    nav_snapshot["symbol_open_qty"] = sym_open_qty

    try:
        _RISK_STATE.note_attempt(time.time())
    except Exception:
        pass

    risk_veto, details = _evaluate_order_risk(
        symbol,
        side,
        gross_target=gross_target,
        nav=nav,
        sym_open_qty=sym_open_qty,
        current_gross=current_gross,
        current_symbol_gross=current_symbol_gross,
        open_positions_count=open_positions_count,
        tier_name=tier_name,
        current_tier_gross=current_tier_gross,
        lev=lev,
        reduce_only=reduce_only,
        intent=intent,
    )
    if PIPELINE_V6_SHADOW_ENABLED and not reduce_only:
        _maybe_run_pipeline_v6_shadow(
            symbol,
            side,
            gross_target=gross_target,
            nav=nav,
            sym_open_qty=sym_open_qty,
            current_gross=current_gross,
            open_positions_count=open_positions_count,
            tier_name=tier_name,
            current_tier_gross=current_tier_gross,
            lev=lev,
            intent=intent,
            nav_snapshot=nav_snapshot,
            positions=positions,
            sizing_cfg=cfg.get("sizing", {}) if isinstance(cfg, Mapping) else {},
        )
    reasons = details.get("reasons", []) if isinstance(details, dict) else []
    if risk_veto:
        reason = reasons[0] if reasons else "blocked"
        price_hint = float(intent.get("price", 0.0) or 0.0)
        block_info = {
            "symbol": symbol,
            "side": side,
            "reason": reason,
            "notional": gross_target,
            "price": price_hint,
        }
        LOG.warning("[risk] block %s", block_info)
        try:
            from execution.telegram_utils import send_telegram

            send_telegram(
                f"Risk‑block {symbol} {side}: {reason}\nnotional={gross_target:.2f} price={price_hint:.2f}",
                silent=True,
            )
        except Exception:
            pass
        try:
            audit = {
                "phase": "blocked",
                "side": side,
                "positionSide": pos_side,
                "reason": reason,
                "reasons": reasons,
                "notional": gross_target,
                "nav": nav,
                "open_qty": sym_open_qty,
                "gross": current_gross,
            }
            if isinstance(details, dict) and "cooldown_until" in details:
                audit["cooldown_until"] = details.get("cooldown_until")
            publish_order_audit(symbol, audit)
        except Exception:
            pass
        thresholds = {}
        if isinstance(details, Mapping):
            thresholds = dict(details.get("thresholds") or {})
        _persist_veto(
            reason,
            price_hint,
            {
                "reasons": reasons,
                "intent": intent,
                "nav_snapshot": nav_snapshot,
                "thresholds": thresholds,
                "diagnostics": details if isinstance(details, Mapping) else {},
            },
        )
        return

    price_hint = float(intent.get("price", 0.0) or 0.0)
    if price_hint <= 0:
        try:
            price_hint = float(get_price(symbol))
        except Exception as exc:
            LOG.error("[executor] price_fetch_err %s %s", symbol, exc)
            return
    try:
        payload, meta = build_order_payload(
            symbol=symbol,
            side=side,
            price=price_hint,
            desired_gross_usd=gross_target,
            reduce_only=reduce_only,
            position_side=pos_side,
        )
    except Exception as exc:
        LOG.error(
            "[executor] SIZE_ERR %s side=%s gross=%.4f err=%s",
            symbol,
            side,
            gross_target,
            exc,
        )
        try:
            publish_order_audit(
                symbol,
                {
                    "phase": "size_error",
                    "side": side,
                    "positionSide": pos_side,
                    "error": str(exc),
                    "requested_gross": gross_target,
                },
            )
        except Exception:
            pass
        return

    payload["positionSide"] = pos_side

    def _meta_float(val: Any, fallback: float) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return fallback

    def _attempt_maker_first(px: float, qty: float) -> Optional[PlaceOrderResult]:
        if submit_limit is None or effective_px is None or PlaceOrderResult is None:
            return None
        if px <= 0 or qty <= 0:
            return None
        try:
            post_px = effective_px(px, side, is_maker=True) or px
            return submit_limit(symbol, post_px, qty, side)
        except Exception as exc:
            LOG.warning("[executor] maker_first_failed symbol=%s err=%s", symbol, exc)
            return None

    def _maker_metrics(result: PlaceOrderResult) -> Dict[str, Any]:
        avg_fill = result.price if result.price is not None else None
        return {
            "attempt_id": attempt_id,
            "venue": "binance_futures",
            "route": "maker_first",
            "prices": {
                "mark": price_hint,
                "submitted": result.price,
                "avg_fill": avg_fill,
            },
            "qty": {
                "contracts": result.filled_qty or result.qty,
                "notional_usd": (avg_fill or price_hint) * (result.filled_qty or result.qty or 0.0)
                if (avg_fill or price_hint)
                else None,
            },
            "timing_ms": {
                "decision": decision_latency_ms,
                "submit": None,
                "ack": None,
                "fill": None,
            },
            "result": {
                "status": "FILLED" if (result.filled_qty or 0.0) > 0 else "NEW",
                "retries": result.rejections,
                "cancelled": False,
            },
            "fees_usd": None,
            "slippage_bps": result.slippage_bps,
        }

    if not reduce_only:
        # Ensure the opening order never carries the reduceOnly flag
        payload.pop("reduceOnly", None)

    norm_price = _meta_float(meta.get("normalized_price"), price_hint)
    norm_qty = _meta_float(meta.get("normalized_qty"), _meta_float(payload.get("quantity"), 0.0))
    maker_ctx_enabled = (
        not reduce_only
        and not force_direct_send
        and norm_price > 0
        and norm_qty > 0
        and submit_limit is not None
        and effective_px is not None
    )
    payload_view = {
        k: payload[k]
        for k in ("type", "quantity", "reduceOnly", "positionSide")
        if k in payload
    }

    # Final gate is delegated to risk_engine_v6; executor no-ops here.

    normalized_ctx = {"price": norm_price, "qty": norm_qty, **meta}

    try:
        publish_intent_audit(
            {
                **intent,
                "t": time.time(),
                "side": side,
                "positionSide": pos_side,
                "notional": gross_target,
                "reduceOnly": reduce_only,
                "normalized": normalized_ctx,
            }
        )
    except Exception:
        pass

    if DRY_RUN:
        LOG.info("[executor] DRY_RUN — skipping SEND_ORDER")
        try:
            publish_order_audit(
                symbol,
                {
                    "phase": "dry_run",
                    "side": side,
                    "positionSide": pos_side,
                    "notional": gross_target,
                    "normalized": normalized_ctx,
                    "payload": payload_view,
                },
            )
        except Exception:
            pass
        return

    LOG.info("[executor] SEND_ORDER %s %s payload=%s meta=%s", symbol, side, payload_view, meta)
    try:
        publish_order_audit(
            symbol,
            {
                "phase": "request",
                "side": side,
                "positionSide": pos_side,
                "notional": gross_target,
                "reduceOnly": reduce_only,
                "normalized": normalized_ctx,
                "payload": payload_view,
            },
        )
    except Exception:
        pass

    resp: Dict[str, Any] = {}
    router_error: Optional[str] = None
    router_metrics: Optional[Dict[str, Any]] = None

    if force_direct_send:
        try:
            resp = _dispatch(payload)
            router_metrics = {
                "attempt_id": attempt_id,
                "venue": "binance_futures",
                "route": intent.get("route", "market"),
                "prices": {
                    "mark": price_hint,
                    "submitted": _meta_float(payload.get("price"), price_hint),
                    "avg_fill": _meta_float(resp.get("avgPrice"), price_hint),
                },
                "qty": {
                    "contracts": _meta_float(resp.get("executedQty"), 0.0),
                    "notional_usd": None,
                },
                "timing_ms": {
                    "decision": decision_latency_ms,
                    "submit": None,
                    "ack": None,
                    "fill": None,
                },
                "result": {"status": resp.get("status"), "retries": 0, "cancelled": False},
                "fees_usd": None,
                "slippage_bps": None,
            }
        except Exception as exc:
            router_error = str(exc)
            LOG.error("[executor] flip open send_failed %s %s", symbol, exc)
            return
        try:
            publish_order_audit(
                symbol,
                {
                    "phase": "flip_open",
                    "side": side,
                    "positionSide": pos_side,
                    "payload": {
                        k: payload[k]
                        for k in ("type", "quantity", "positionSide")
                        if k in payload
                    },
                    "normalized": {"price": norm_price, "qty": norm_qty},
                },
            )
        except Exception:
            pass

    if _route_intent is not None and not force_direct_send:
        router_ctx = {
            "payload": payload,
            "price": price_hint,
            "positionSide": pos_side,
            "reduceOnly": reduce_only,
        }
        policy = router_policy(symbol)
        router_ctx["router_policy"] = {
            "quality": policy.quality,
            "taker_bias": policy.taker_bias,
            "maker_first": policy.maker_first,
            "reason": policy.reason,
        }
        if maker_ctx_enabled:
            router_ctx.update(
                {
                    "maker_first": True,
                    "maker_price": norm_price,
                    "maker_qty": norm_qty,
                }
            )
        router_payload = dict(intent)
        router_payload.update(
            {
                "symbol": symbol,
                "side": side,
                "positionSide": pos_side,
                "reduceOnly": reduce_only,
                "quantity": payload.get("quantity"),
                "type": payload.get("type"),
                "price": payload.get("price", price_hint),
                "router_ctx": router_ctx,
                "dry_run": DRY_RUN,
                "timing": {"decision": decision_latency_ms},
            }
        )
        if maker_ctx_enabled:
            router_payload["route"] = "maker_first"
        try:
            result, router_metrics = _route_intent(router_payload, attempt_id)
        except Exception as exc:
            router_error = str(exc)
        else:
            if result.get("accepted"):
                resp = result.get("raw") or {}
            else:
                router_error = str(result.get("reason") or "router_reject")
    elif _route_order is not None and not force_direct_send:
        router_intent = {
            **intent,
            "symbol": symbol,
            "side": side,
            "positionSide": pos_side,
            "reduceOnly": reduce_only,
            "quantity": payload.get("quantity"),
            "type": payload.get("type"),
            "price": payload.get("price", price_hint),
        }
        router_ctx = {
            "payload": payload,
            "price": price_hint,
            "positionSide": pos_side,
            "reduceOnly": reduce_only,
        }
        policy = router_policy(symbol)
        router_ctx["router_policy"] = {
            "quality": policy.quality,
            "taker_bias": policy.taker_bias,
            "maker_first": policy.maker_first,
            "reason": policy.reason,
        }
        if maker_ctx_enabled:
            router_ctx.update(
                {
                    "maker_first": True,
                    "maker_price": norm_price,
                    "maker_qty": norm_qty,
                }
            )
        try:
            result = _route_order(router_intent, router_ctx, DRY_RUN)
        except Exception as exc:
            router_error = str(exc)
        else:
            if result.get("accepted"):
                resp = result.get("raw") or {}
            else:
                router_error = str(result.get("reason") or "router_reject")

    if not resp:
        if router_error and not force_direct_send:
            LOG.error(
                "[executor] router_reject %s %s reason=%s", symbol, side, router_error
            )
            _log_order_error(
                symbol=symbol,
                side=side,
                notional=gross_target,
                reason=f"router_error:{router_error}",
                classification=None,
                retried=False,
                component="router",
            )
        if maker_ctx_enabled:
            maker_result = _attempt_maker_first(norm_price, norm_qty)
            if maker_result:
                resp = maker_result.raw or {}
                router_metrics = router_metrics or _maker_metrics(maker_result)
                LOG.info(
                    "[executor] maker_first_fallback symbol=%s side=%s is_maker=%s rejections=%s",
                    symbol,
                    side,
                    maker_result.is_maker,
                    maker_result.rejections,
                )
    if not resp:
        dispatch_attempt = 0
        while True:
            try:
                resp = _dispatch(payload)
                break
            except requests.HTTPError as exc:
                dispatch_attempt += 1
                try:
                    _RISK_STATE.note_error(time.time())
                except Exception:
                    pass
                err_code = None
                try:
                    if exc.response is not None:
                        err_code = exc.response.json().get("code")
                except Exception:
                    err_code = None
                classification = ex.classify_binance_error(exc, getattr(exc, "response", None))
                LOG.error(
                    "[executor] ORDER_ERR code=%s symbol=%s side=%s meta=%s payload=%s err=%s",
                    err_code,
                    symbol,
                    side,
                    meta,
                    payload_view,
                    exc,
                )
                retriable = bool(classification.get("retriable")) and dispatch_attempt <= _MAX_TRANSIENT_RETRIES
                _log_order_error(
                    symbol=symbol,
                    side=side,
                    notional=gross_target,
                    reason="http_error",
                    classification=classification,
                    retried=retriable,
                    exc=exc,
                    component="exchange",
                    context={"code": err_code, "payload": payload_view, "attempt": dispatch_attempt},
                )
                try:
                    publish_order_audit(
                        symbol,
                        {
                            "phase": "error",
                            "side": side,
                            "positionSide": pos_side,
                            "error": str(exc),
                            "code": err_code,
                            "normalized": normalized_ctx,
                            "payload": payload_view,
                        },
                    )
                except Exception:
                    pass
                if err_code == -1111:
                    LOG.error(
                        "[executor] ORDER_PRECISION ctx=%s payload=%s",
                        normalized_ctx,
                        payload_view,
                    )
                    return
                if retriable:
                    time.sleep(_TRANSIENT_RETRY_BACKOFF_S)
                    continue
                raise
            except Exception as exc:
                try:
                    _RISK_STATE.note_error(time.time())
                except Exception:
                    pass
                _log_order_error(
                    symbol=symbol,
                    side=side,
                    notional=gross_target,
                    reason="dispatch_error",
                    classification=None,
                    retried=False,
                    exc=exc,
                    component="executor",
                )
                try:
                    publish_order_audit(
                        symbol,
                        {
                            "phase": "error",
                            "side": side,
                            "positionSide": pos_side,
                            "error": str(exc),
                        },
                    )
                except Exception:
                    pass
                raise

    latency_ms = max(0.0, (time.monotonic() - attempt_start_monotonic) * 1000.0)

    ack_info: Optional[OrderAckInfo] = None
    fill_summary: Optional[FillSummary] = None
    request_qty_val = _to_float(payload.get("quantity")) or _to_float(norm_qty)
    if resp:
        ack_info = _emit_order_ack(
            symbol=symbol,
            side=side,
            order_type=payload.get("type") or intent.get("type") or "MARKET",
            request_qty=request_qty_val,
            position_side=pos_side,
            reduce_only=reduce_only,
            resp=resp,
            latency_ms=latency_ms,
            attempt_id=attempt_id,
            intent_id=intent_id,
        )
        if ack_info and not resp.get("dryRun"):
            fill_summary = _confirm_order_fill(
                ack_info,
                intent.get("metadata"),
                intent.get("strategy") or intent.get("strategy_id"),
            )

    avg_fill_price = (
        fill_summary.avg_price if fill_summary and fill_summary.avg_price is not None else _to_float(resp.get("avgPrice"))
    )
    executed_qty_val = (
        fill_summary.executed_qty
        if fill_summary and fill_summary.executed_qty is not None
        else _to_float(resp.get("executedQty"))
    )
    status_value = fill_summary.status if fill_summary else _normalize_status(resp.get("status"))
    fees_total = fill_summary.fee_total if fill_summary else _to_float(resp.get("commission"))
    fill_latency_ms = fill_summary.latency_ms if fill_summary else None

    if router_metrics is None:
        router_metrics = {
            "attempt_id": attempt_id,
            "venue": "binance_futures",
            "route": intent.get("route", "market"),
            "prices": {
                "mark": price_hint,
                "submitted": _meta_float(payload.get("price"), price_hint),
                "avg_fill": avg_fill_price,
            },
            "qty": {
                "contracts": executed_qty_val if executed_qty_val is not None else _meta_float(payload.get("quantity"), norm_qty),
                "notional_usd": None,
            },
            "timing_ms": {
                "decision": decision_latency_ms,
                "submit": None,
                "ack": latency_ms,
                "fill": fill_latency_ms,
            },
            "result": {
                "status": status_value,
                "retries": 0,
                "cancelled": status_value in {"CANCELED", "EXPIRED"},
            },
            "fees_usd": fees_total,
            "slippage_bps": None,
        }
    prices_section = router_metrics.setdefault(
        "prices",
        {"mark": price_hint, "submitted": _meta_float(payload.get("price"), price_hint), "avg_fill": None},
    )
    qty_section = router_metrics.setdefault(
        "qty", {"contracts": _meta_float(payload.get("quantity"), norm_qty), "notional_usd": None}
    )
    timing_section = router_metrics.setdefault(
        "timing_ms",
        {"decision": decision_latency_ms, "submit": None, "ack": latency_ms, "fill": fill_latency_ms},
    )
    result_section = router_metrics.setdefault(
        "result",
        {"status": status_value, "retries": 0, "cancelled": status_value in {"CANCELED", "EXPIRED"}},
    )
    if avg_fill_price is not None:
        prices_section["avg_fill"] = avg_fill_price
    prices_section.setdefault("mark", _meta_float(prices_section.get("mark"), price_hint))
    prices_section.setdefault("submitted", _meta_float(prices_section.get("submitted"), price_hint))
    if executed_qty_val is not None:
        qty_section["contracts"] = executed_qty_val
    if prices_section.get("avg_fill") is not None and qty_section.get("contracts") is not None:
        try:
            qty_section["notional_usd"] = float(prices_section["avg_fill"]) * float(qty_section["contracts"])
        except Exception:
            pass
    timing_section["ack"] = latency_ms
    if fill_latency_ms is not None:
        timing_section["fill"] = fill_latency_ms
    result_section["status"] = status_value
    result_section["cancelled"] = status_value in {"CANCELED", "EXPIRED"}
    if fees_total is not None:
        router_metrics["fees_usd"] = fees_total
    elif router_metrics.get("fees_usd") is None:
        router_metrics["fees_usd"] = None
    if router_metrics.get("slippage_bps") is None and prices_section.get("avg_fill") is not None:
        mark_val = _meta_float(prices_section.get("mark"), price_hint)
        fill_val = _meta_float(prices_section.get("avg_fill"), price_hint)
        if mark_val not in (None, 0):
            try:
                slip = ((fill_val - mark_val) / mark_val) * 10_000.0 if fill_val is not None else None
                if slip is not None and side == "SELL":
                    slip *= -1.0
                router_metrics["slippage_bps"] = slip
            except Exception:
                router_metrics["slippage_bps"] = None
    router_metrics["attempt_id"] = attempt_id
    router_metrics["intent_id"] = intent_id
    router_metrics["symbol"] = symbol
    router_metrics["side"] = side
    router_metrics["ts"] = time.time()
    _append_order_metrics(router_metrics)
    _mirror_router_metrics(router_metrics)
    _maybe_emit_risk_snapshot(force=True)
    _INTENT_REGISTRY[intent_id] = {
        "order_id": ack_info.order_id if ack_info else resp.get("orderId"),
        "symbol": symbol,
        "side": side,
        "attempt_id": attempt_id,
        "avg_fill": prices_section.get("avg_fill"),
        "qty": qty_section.get("contracts"),
    }

    if ack_info:
        LOG.info(
            "[executor] ORDER_ACK id=%s status=%s qty_req=%s",
            ack_info.order_id or ack_info.client_order_id,
            ack_info.status,
            ack_info.request_qty,
        )
    else:
        LOG.info("[executor] ORDER_ACK missing response symbol=%s", symbol)
    if fill_summary:
        LOG.info(
            "[executor] ORDER_FILL id=%s status=%s avgPrice=%s qty=%s",
            ack_info.order_id if ack_info else resp.get("orderId"),
            fill_summary.status,
            fill_summary.avg_price,
            fill_summary.executed_qty,
        )

    audit_payload = {
        "phase": "response",
        "side": side,
        "positionSide": pos_side,
        "status": status_value,
        "orderId": ack_info.order_id if ack_info else resp.get("orderId"),
        "avgPrice": avg_fill_price,
        "qty": executed_qty_val,
        "normalized": normalized_ctx,
        "payload": payload_view,
    }
    if fill_summary:
        audit_payload["fill"] = {
            "executedQty": fill_summary.executed_qty,
            "avgPrice": fill_summary.avg_price,
            "fee_total": fill_summary.fee_total,
            "feeAsset": fill_summary.fee_asset,
            "ts_fill_first": fill_summary.ts_fill_first,
            "ts_fill_last": fill_summary.ts_fill_last,
        }
    try:
        publish_order_audit(symbol, audit_payload)
    except Exception:
        pass

    executed_qty_float = executed_qty_val or 0.0
    if executed_qty_float > 0:
        try:
            _RISK_STATE.note_fill(symbol, time.time())
        except Exception:
            pass
        _emit_position_snapshots(symbol)
        is_position_close = (
            reduce_only
            or (side == "SELL" and pos_side == "LONG")
            or (side == "BUY" and pos_side == "SHORT")
        )
        if fill_summary and is_position_close:
            try:
                publish_close_audit(
                    symbol,
                    pos_side,
                    {
                        "orderId": ack_info.order_id if ack_info else resp.get("orderId"),
                        "avgPrice": fill_summary.avg_price,
                        "qty": fill_summary.executed_qty,
                        "status": fill_summary.status,
                    },
                )
            except Exception:
                pass
            close_record = {
                "ts": time.time(),
                "intent_id": intent_id,
                "attempt_id": attempt_id,
                "symbol": symbol,
                "side": side,
                "event": "position_close",
                "pnl_at_close_usd": (
                    intent.get("pnl_usd")
                    or intent.get("realized_pnl_usd")
                    or intent.get("pnl")
                ),
                "order_status": fill_summary.status,
            }
            _append_order_metrics(close_record)
            _INTENT_REGISTRY.pop(intent_id, None)
            # V7.3: Remove TP/SL registry entry on position close
            try:
                from execution.position_tp_sl_registry import unregister_position_tp_sl
                unregister_position_tp_sl(symbol, pos_side)
            except Exception:
                pass
        elif not is_position_close:
            # V7.3: Register TP/SL for new positions from vol_target strategy
            try:
                tp_price = intent.get("take_profit_price")
                sl_price = intent.get("stop_loss_price")
                if tp_price is not None or sl_price is not None:
                    metadata = intent.get("metadata", {}).get("vol_target", {}).get("tp_sl")
                    from execution.position_tp_sl_registry import register_position_tp_sl
                    register_position_tp_sl(
                        symbol=symbol,
                        position_side=pos_side,
                        take_profit_price=tp_price,
                        stop_loss_price=sl_price,
                        metadata=metadata,
                    )
            except Exception:
                pass



def _compute_nav_snapshot() -> Optional[float]:
    try:
        from execution.state_publish import compute_nav

        return float(compute_nav())
    except Exception as exc:
        LOG.error("[executor] compute_nav not available: %s", exc)
        try:
            from execution.exchange_utils import get_account

            acc = get_account()
            return float(
                acc.get("totalMarginBalance")
                or (
                    float(acc.get("totalWalletBalance", 0) or 0)
                    + float(acc.get("totalUnrealizedProfit", 0) or 0)
                )
            )
        except Exception as account_exc:
            LOG.error("[executor] account NAV error: %s", account_exc)
    return None


def _collect_rows() -> List[Dict[str, Any]]:
    try:
        from execution.exchange_utils import get_positions as _get_positions_snapshot

        raw_positions = list(_get_positions_snapshot() or [])
    except Exception as exc:
        LOG.error("[executor] positions fetch error: %s", exc)
        raw_positions = []
    rows: List[Dict[str, Any]] = []
    for payload in raw_positions:
        try:
            qty_val = float(payload.get("qty", payload.get("positionAmt", 0)) or 0.0)
            if abs(qty_val) <= 0.0:
                continue
            entry_price = _to_float(payload.get("entryPrice") or payload.get("entry_price"))
            if entry_price is None or entry_price <= 0.0:
                continue
            mark_val = _to_float(
                payload.get("markPrice")
                or payload.get("mark_price")
                or payload.get("mark")
            )
            if mark_val is None:
                try:
                    mark_val = float(get_price(str(payload.get("symbol") or "") or ""))
                except Exception:
                    mark_val = None
            if mark_val is None or mark_val <= 0.0:
                continue
            unrealized_val = _to_float(
                payload.get("unrealized", payload.get("unRealizedProfit", 0)) or 0.0
            )
            leverage_val = _to_float(payload.get("leverage"))
            rows.append(
                {
                    "symbol": payload.get("symbol"),
                    "positionSide": payload.get("positionSide", "BOTH"),
                    "qty": qty_val,
                    "entryPrice": entry_price,
                    "unrealized": float(unrealized_val or 0.0),
                    "leverage": float(leverage_val or 0.0),
                    "markPrice": mark_val,
                }
            )
        except Exception:
            continue
    return rows


def _build_positions_state_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize live positions for the canonical positions_state.json writer."""
    now_iso = datetime.now(timezone.utc).isoformat()
    normalized: List[Dict[str, Any]] = []
    for raw in rows:
        try:
            symbol = str(raw.get("symbol") or "").upper()
            if not symbol:
                continue
            qty = _to_float(raw.get("qty") or raw.get("positionAmt"))
            if qty is None:
                continue
            entry_price = _to_float(raw.get("entryPrice") or raw.get("entry_price"))
            mark_price = _to_float(
                raw.get("markPrice") or raw.get("mark_price") or raw.get("mark")
            )
            if mark_price is None and symbol:
                try:
                    mark_price = float(get_price(symbol))
                except Exception:
                    mark_price = None
            unrealized = _to_float(
                raw.get("unrealized")
                or raw.get("unRealizedProfit")
                or raw.get("pnl")
                or raw.get("pnl_usd")
                or raw.get("unrealized_pnl")
            )
            leverage = _to_float(raw.get("leverage"))
            if unrealized is None and mark_price is not None and entry_price is not None:
                unrealized = (mark_price - entry_price) * qty
            notional = abs(qty) * mark_price if mark_price is not None else None
            normalized.append(
                {
                    "symbol": symbol,
                    "side": str(raw.get("positionSide") or raw.get("side") or ("LONG" if qty > 0 else "SHORT")).upper(),
                    "qty": qty,
                    "entry_price": entry_price if entry_price is not None else 0.0,
                    "mark_price": mark_price if mark_price is not None else 0.0,
                    "unrealized_pnl": unrealized if unrealized is not None else 0.0,
                    "realized_pnl": _to_float(
                        raw.get("realized_pnl") or raw.get("realized") or raw.get("realizedProfit")
                    ),
                    "notional": notional if notional is not None else 0.0,
                    "leverage": leverage,
                    "ts": now_iso,
                }
            )
        except Exception:
            continue
    return normalized


def _write_positions_state(positions_rows: List[Dict[str, Any]], *, updated_ts: Optional[str] = None) -> None:
    """
    Canonical writer for logs/state/positions_state.json.

    Ensures non-zero positions carry non-zero entry/mark prices to catch bogus snapshots early.
    """
    ts_value = updated_ts or datetime.now(timezone.utc).isoformat()
    for row in positions_rows:
        qty = _to_float(row.get("qty"))
        if qty is None or abs(qty) <= 0:
            continue
        entry_price = _to_float(row.get("entry_price"))
        mark_price = _to_float(row.get("mark_price"))
        if entry_price is None or entry_price <= 0:
            LOG.warning(
                "[telemetry] positions_state_invalid_entry symbol=%s qty=%s entry=%s",
                row.get("symbol"),
                qty,
                entry_price,
            )
        if mark_price is None or mark_price <= 0:
            LOG.warning(
                "[telemetry] positions_state_invalid_mark symbol=%s qty=%s mark=%s",
                row.get("symbol"),
                qty,
                mark_price,
            )
        assert entry_price is not None and entry_price > 0, "positions_state entry_price must be >0 for open positions"
        assert mark_price is not None and mark_price > 0, "positions_state mark_price must be >0 for open positions"
    write_positions_state(
        positions_rows,
        path=POSITIONS_STATE_PATH,
        updated_at=ts_value,
    )


def _json_default(value: Any) -> str:
    try:
        if isinstance(value, (datetime,)):
            return value.isoformat()
    except Exception:
        pass
    return str(value)


def _write_json_cache(path: Path, payload: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
        path.write_text(serialized, encoding="utf-8")
    except Exception as exc:
        LOG.warning("[executor] cache_write_failed path=%s err=%s", path, exc)


def _persist_positions_cache(rows: List[Dict[str, Any]]) -> None:
    total_rows = len(rows)
    items: List[Dict[str, Any]] = []
    for row in rows:
        size = _to_float(row.get("qty")) or 0.0
        if abs(size) <= 0.0:
            continue
        symbol = str(row.get("symbol") or "").upper()
        if not symbol:
            continue
        mark_price = _to_float(
            row.get("markPrice") or row.get("mark_price") or row.get("mark")
        )
        if mark_price is None:
            try:
                mark_price = float(get_price(symbol))
            except Exception:
                mark_price = None
        pnl_val = _to_float(
            row.get("unrealized")
            or row.get("unRealizedProfit")
            or row.get("pnl")
            or row.get("pnl_usd")
        )
        side = str(row.get("positionSide") or ("LONG" if size > 0 else "SHORT")).upper()
        items.append(
            {
                "symbol": symbol,
                "side": side,
                "size": float(size),
                "entry": _to_float(row.get("entryPrice")),
                "mark": mark_price,
                "pnl_usd": pnl_val,
            }
        )
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "items": items,
    }
    _write_json_cache(POSITIONS_CACHE_PATH, payload)
    LOG.info(
        "[executor] positions.json updated n=%d len=%d",
        total_rows,
        len(items),
    )


def _persist_nav_log(nav_val: Optional[float], rows: List[Dict[str, Any]]) -> None:
    if nav_val is None:
        return
    entry: Dict[str, Any] = {
        "t": time.time(),
        "nav": float(nav_val),
    }
    try:
        unrealized_total = sum(float(row.get("unrealized") or 0.0) for row in rows)
        entry["unrealized_pnl"] = unrealized_total
    except Exception:
        pass
    try:
        existing = json.loads(NAV_LOG_CACHE_PATH.read_text(encoding="utf-8"))
        if not isinstance(existing, list):
            existing = []
    except FileNotFoundError:
        existing = []
    except Exception:
        existing = []
    existing.append(entry)
    if len(existing) > NAV_LOG_MAX_POINTS:
        existing = existing[-NAV_LOG_MAX_POINTS:]
    _write_json_cache(NAV_LOG_CACHE_PATH, existing)


def _build_spot_state_payload() -> Optional[Dict[str, Any]]:
    try:
        balances = get_balances() or []
    except Exception as exc:
        LOG.debug("[executor] spot balances fetch failed: %s", exc)
        return None

    assets: List[Dict[str, Any]] = []
    total_usd = 0.0
    stable_set = {"USDT", "USDC", "BUSD", "FDUSD"}

    def _parse_qty(value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    if isinstance(balances, Mapping):
        iterable = [
            {
                "asset": str(asset).upper(),
                "balance": _parse_qty(amount),
            }
            for asset, amount in balances.items()
        ]
    else:
        iterable = []
        if isinstance(balances, list):
            for entry in balances:
                if not isinstance(entry, Mapping):
                    continue
                asset = str(
                    entry.get("asset")
                    or entry.get("coin")
                    or entry.get("symbol")
                    or entry.get("currency")
                    or ""
                ).upper()
                if not asset:
                    continue
                qty = _parse_qty(
                    entry.get("balance")
                    or entry.get("availableBalance")
                    or entry.get("walletBalance")
                    or entry.get("free")
                    or entry.get("qty")
                    or 0.0
                )
                iterable.append({"asset": asset, "balance": qty})

    for item in iterable:
        asset = item.get("asset") or ""
        qty = _parse_qty(item.get("balance"))
        if not asset or abs(qty) < 1e-9:
            continue
        price: Optional[float]
        if asset in stable_set:
            price = 1.0
        else:
            try:
                price = float(get_price(f"{asset}USDT"))
            except Exception:
                price = None
        usd_val = qty * price if price is not None else None
        if usd_val is not None:
            total_usd += usd_val
        assets.append(
            {
                "asset": asset,
                "balance": qty,
                "price_usdt": price,
                "usd_value": usd_val,
            }
        )

    payload = {
        "source": "binance_testnet" if is_testnet() else "binance_futures",
        "assets": assets,
        "total_usd": float(total_usd),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    return payload


def _persist_spot_state() -> None:
    payload = _build_spot_state_payload()
    if payload is None:
        return
    _write_json_cache(SPOT_STATE_CACHE_PATH, payload)


def _pub_tick() -> None:
    global _LAST_NAV_STATE, _LAST_POSITIONS_STATE
    nav_val, nav_detail = _compute_nav_with_detail()
    rows = _collect_rows()
    _persist_positions_cache(rows)
    _persist_nav_log(nav_val, rows)
    _persist_spot_state()
    now = time.time()
    now_iso = datetime.now(timezone.utc).isoformat()
    nav_float = float(nav_val) if nav_val is not None else 0.0
    nav_payload = {"nav": nav_float, "nav_usd": nav_float, "updated_ts": now_iso}
    if isinstance(nav_detail, Mapping):
        nav_payload["nav_detail"] = dict(nav_detail)
        # Flatten a few helpful fields for dashboard/state readers
        for key in ("assets", "mark_prices", "nav_mode", "freshness"):
            if key in nav_detail:
                nav_payload[key] = nav_detail.get(key)
    nav_written = False
    positions_state_written = False
    positions_snapshot_written = False
    positions_ledger_written = False
    risk_written = False
    scores_written = False
    diagnostics_written = False
    synced_written = False
    try:
        write_nav_state(nav_payload)
        nav_written = True
    except Exception as exc:
        LOG.error("[telemetry] nav_state_write_failed: %s", exc)
    positions_state_rows = _build_positions_state_rows(rows)
    positions_state_ts = now_iso
    positions_state_payload = {
        "positions": positions_state_rows,
        "updated_at": positions_state_ts,
        "updated_ts": positions_state_ts,
    }
    try:
        _write_positions_state(positions_state_rows, updated_ts=positions_state_ts)
        positions_state_written = True
    except Exception as exc:
        LOG.error("[telemetry] positions_state_contract_write_failed: %s", exc)
    # Only persist non-zero positions; ignore exchange noise entries.
    filtered_rows = []
    for r in rows:
        try:
            qty = float(r.get("positionAmt") or r.get("qty") or 0.0)
        except Exception:
            qty = 0.0
        if abs(qty) < 1e-9:
            continue
        filtered_rows.append(r)
    # Persist non-zero positions with enriched fields for monitoring/ledger.
    items = []
    for r in filtered_rows:
        qty = r.get("qty") if "qty" in r else r.get("positionAmt")
        try:
            qty = float(qty or 0.0)
        except Exception:
            qty = 0.0
        entry = r.get("entryPrice") or r.get("entry_price") or 0.0
        mark = r.get("markPrice") or r.get("mark_price") or 0.0
        try:
            entry = float(entry)
        except Exception:
            entry = 0.0
        try:
            mark = float(mark)
        except Exception:
            mark = 0.0
        if not mark:
            try:
                sym = r.get("symbol")
                if sym:
                    mark = float(get_price(f"{sym}USDT"))
            except Exception:
                mark = 0.0
        try:
            pnl = float(r.get("unrealized") or r.get("pnl") or (qty * (mark - entry)))
        except Exception:
            pnl = 0.0
        try:
            notional = abs(qty * mark)
        except Exception:
            notional = 0.0
        items.append(
            {
                "symbol": r.get("symbol"),
                "side": r.get("positionSide") or r.get("side"),
                "qty": qty,
                "entry_price": entry,
                "mark_price": mark,
                "pnl": pnl,
                "leverage": r.get("leverage"),
                "notional": notional,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )
    positions_payload = {
        "rows": filtered_rows,
        "items": items,
        "positions": positions_state_rows,
        "updated": now,
    }
    try:
        write_positions_snapshot_state(positions_payload)
        positions_snapshot_written = True
    except Exception as exc:
        LOG.error("[telemetry] positions_snapshot_write_failed: %s", exc)
    try:
        from execution.position_ledger import build_position_ledger, build_positions_ledger_state

        ledger = build_position_ledger(POSITIONS_STATE_PATH.parent)
        ledger_snapshot = build_positions_ledger_state(
            ledger,
            updated_at=positions_state_ts,
            state_dir=POSITIONS_STATE_PATH.parent,
        )
        write_positions_ledger_state(ledger_snapshot)
        positions_ledger_written = True
    except Exception as exc:
        LOG.error("[telemetry] positions_ledger_write_failed: %s", exc)
    risk_payload = _LAST_RISK_SNAPSHOT or {"updated_ts": now_iso, "symbols": []}
    try:
        write_risk_snapshot_state(risk_payload)
        risk_written = True
    except Exception as exc:
        LOG.error("[telemetry] risk_snapshot_write_failed: %s", exc)
    # v7.5_A1: Publish VaR/CVaR and alpha decay state
    try:
        positions_for_var = []
        for r in rows:
            sym = r.get("symbol")
            if not sym:
                continue
            # Compute notional from qty * markPrice
            qty = abs(float(r.get("qty") or 0))
            mark = float(r.get("markPrice") or r.get("mark_price") or 0)
            notional = qty * mark if mark > 0 else abs(float(r.get("notional") or 0))
            if notional > 0:
                positions_for_var.append({"symbol": sym, "notional": notional})
        compute_and_write_risk_advanced_state(positions_for_var, nav_float)
    except Exception as exc:
        LOG.debug("[telemetry] risk_advanced_write_failed: %s", exc)
    try:
        symbols_list = [str(r.get("symbol")) for r in rows if r.get("symbol")]
        compute_and_write_alpha_decay_state(symbols_list)
    except Exception as exc:
        LOG.debug("[telemetry] alpha_decay_write_failed: %s", exc)
    scores_payload = (
        dict(_LAST_SYMBOL_SCORES_STATE)
        if isinstance(_LAST_SYMBOL_SCORES_STATE, Mapping)
        else {"symbols": [], "updated_ts": now_iso, "intel_enabled": bool(INTEL_V6_ENABLED)}
    )
    try:
        write_symbol_scores_state(scores_payload)
        scores_written = True
    except Exception as exc:
        LOG.error("[telemetry] symbol_scores_write_failed: %s", exc)
    try:
        write_runtime_diagnostics_state()
        diagnostics_written = True
    except Exception as exc:
        LOG.debug("[telemetry] diagnostics_state_write_failed: %s", exc)
    engine_meta_written = False
    try:
        write_engine_metadata_state(
            {
                "engine_version": _ENGINE_VERSION,
                "git_commit": _git_commit(),
                "run_id": RUN_ID,
                "hostname": HOSTNAME,
                "env": ENV,
                "status": "running",
            }
        )
        engine_meta_written = True
    except Exception as exc:
        LOG.debug("[telemetry] engine_metadata_write_failed: %s", exc)
    flag_snapshot = get_v6_flag_snapshot()
    try:
        synced_payload = build_synced_state_payload(
            items=rows,
            nav=nav_float,
            engine_version=_ENGINE_VERSION,
            flags=flag_snapshot,
            updated_at=now,
            nav_snapshot=nav_detail if isinstance(nav_detail, Mapping) else {},
        )
    except Exception as exc:
        LOG.debug("[telemetry] build_synced_state_payload_failed: %s", exc)
        synced_payload = {
            "items": [dict(row) for row in rows],
            "nav": nav_float,
            "engine_version": _ENGINE_VERSION,
            "v6_flags": flag_snapshot,
            "updated_at": now,
            "nav_snapshot": nav_detail if isinstance(nav_detail, Mapping) else {},
        }
    try:
        write_synced_state(synced_payload)
        synced_written = True
    except Exception as exc:
        LOG.error("[telemetry] synced_state_write_failed: %s", exc)
    LOG.info(
        "[v6-runtime] state write complete state_dir=logs/state nav=%s positions_state=%s positions_ledger=%s positions=%s risk=%s symbol_scores=%s diagnostics=%s engine_meta=%s synced=%s",
        nav_written,
        positions_state_written,
        positions_ledger_written,
        positions_snapshot_written,
        risk_written,
        scores_written,
        diagnostics_written,
        engine_meta_written,
        synced_written,
    )
    _LAST_NAV_STATE = nav_payload
    _LAST_POSITIONS_STATE = positions_state_payload
    return None


def _loop_once(i: int) -> None:
    # Signal path:
    #   runtime.yaml -> signal_screener.generate_intents() -> executor veto/doctor -> router -> exchange.
    #   generate_intents already applies local screener gates; veto=[] means risk+router should evaluate/send.
    global _LAST_SIGNAL_PULL, _LAST_QUEUE_DEPTH
    _sync_dry_run()
    _refresh_risk_config()
    _account_snapshot()

    try:
        baseline_positions = list(get_positions() or [])
    except Exception:
        baseline_positions = []
    gross_total, _ = _gross_and_open_qty("", "", baseline_positions)
    _update_risk_state_counters(baseline_positions, gross_total)
    active_symbols = {
        str(pos.get("symbol") or "").upper()
        for pos in baseline_positions
        if isinstance(pos, Mapping) and pos.get("symbol")
    }
    _maybe_emit_router_health_snapshot()
    _maybe_emit_risk_snapshot()
    for symbol in sorted(active_symbols):
        _maybe_emit_execution_alerts(symbol)
    _maybe_publish_execution_intel()

    # V7.3: Scan for TP/SL exits before generating new intents
    try:
        from execution.exit_scanner import scan_tp_sl_exits, build_exit_intent
        
        # Build price map from positions (markPrice) or fetch live
        # V7.6: Only build prices for positions with non-zero qty (active positions)
        price_map: Dict[str, float] = {}
        for pos in baseline_positions:
            sym = pos.get("symbol")
            if not sym:
                continue
            # Skip zero-qty positions (Binance returns all symbols from positionRisk)
            qty = float(pos.get("qty") or pos.get("positionAmt") or 0)
            if qty == 0:
                continue
            mark = pos.get("markPrice") or pos.get("mark_price")
            if mark:
                price_map[sym] = float(mark)
            else:
                try:
                    price_map[sym] = float(get_price(sym))
                except Exception:
                    pass
        
        exit_candidates = scan_tp_sl_exits(baseline_positions, price_map)
        for candidate in exit_candidates:
            exit_intent = build_exit_intent(candidate)
            LOG.info(
                "[exit_scanner] %s %s %s exit trigger at %.4f",
                candidate.symbol,
                candidate.position_side,
                candidate.exit_reason.value.upper(),
                candidate.trigger_price,
            )
            try:
                _send_order(exit_intent)
            except Exception as exc:
                LOG.error("[exit_scanner] failed to send exit intent %s: %s", candidate.symbol, exc)
    except ImportError:
        pass
    except Exception as exc:
        LOG.warning("[exit_scanner] error during TP/SL scan: %s", exc)

    if INTENT_TEST:
        intent = {
            "symbol": "BTCUSDT",
            "signal": "BUY",
            "capital_per_trade": 120.0,
            "leverage": 1,
            "positionSide": "LONG",
            "reduceOnly": False,
        }
        LOG.info("[screener->executor] %s", intent)
        _send_order(intent)
        _LAST_SIGNAL_PULL = time.time()
        _LAST_QUEUE_DEPTH = 0
    else:
        _LAST_SIGNAL_PULL = time.time()
        try:
            intents_raw = list(generate_intents(_LAST_SIGNAL_PULL))
        except Exception as e:
            LOG.error("[screener] error: %s", e)
            intents_raw = []
        _LAST_QUEUE_DEPTH = len(intents_raw)
        attempted = getattr(intents_raw, "attempted", len(intents_raw))
        screener_emitted = getattr(intents_raw, "emitted", len(intents_raw))
        submitted = 0
        for idx, raw_intent in enumerate(intents_raw):
            intent = _normalize_intent(raw_intent)
            symbol = cast(Optional[str], intent.get("symbol"))
            if not symbol:
                LOG.warning("[screener] missing symbol in intent %s", intent)
                continue
            now_ts = time.time()
            if _symbol_on_cooldown(symbol, now_ts):
                continue

            veto_reasons = _coerce_veto_reasons(intent.get("veto"))
            if veto_reasons:
                _publish_veto_exec(symbol, veto_reasons, intent)
                continue

            attempt_id = mk_id("sig")
            generated_at = intent.get("generated_at") or intent.get("signal_ts")
            latency_ms: Optional[float] = None
            if generated_at is not None:
                try:
                    gen = float(generated_at)
                    if gen > 1e12:
                        gen = gen / 1000.0
                    latency_ms = max(0.0, (time.time() - gen) * 1000.0)
                except Exception:
                    latency_ms = None
            doctor_verdict = signal_doctor.evaluate_signal(
                str(intent.get("signal", "")),
                symbol,
                intent,
            )
            signal_metrics_record: Dict[str, Any] = {
                "ts": time.time(),
                "attempt_id": attempt_id,
                "symbol": symbol,
                "signal": str(intent.get("signal")),
                "doctor": doctor_verdict,
                "queue_depth": max(0, len(intents_raw) - idx),
                "latency_ms": latency_ms,
            }
            _append_signal_metrics(signal_metrics_record)

            if not doctor_verdict.get("ok", False):
                LOG.info(
                    "[signal] veto attempt=%s symbol=%s reasons=%s",
                    attempt_id,
                    symbol,
                    ",".join(doctor_verdict.get("reasons", [])),
                )
                continue

            submitted += 1
            try:
                _publish_intent_audit(symbol, intent)
                intent_with_attempt = dict(intent)
                intent_with_attempt["attempt_id"] = attempt_id
                _send_order(intent_with_attempt)
            except Exception as exc:
                LOG.error("[executor] failed to send intent %s %s", symbol, exc)
                cooldown_until = _mark_symbol_cooldown(symbol, now_ts)
                LOG.warning(
                    "[screener][cooldown] symbol=%s entering temporary cooldown due to API failure",
                    symbol,
                )
                intent.setdefault("cooldown_until", cooldown_until)
        LOG.info(
            "[screener] attempted=%d emitted=%d submitted=%d",
            attempted,
            screener_emitted,
            submitted,
        )

    try:
        _pub_tick()
    except Exception as exc:
        LOG.exception("[loop] publish_tick_failed: %s", exc)
    _maybe_run_pipeline_v6_shadow_heartbeat()
    _maybe_emit_router_health_snapshot()
    _maybe_emit_risk_snapshot()
    _maybe_run_telegram_alerts()
    _maybe_emit_execution_health_snapshot()
    _maybe_run_pipeline_v6_compare()


def main(argv: Optional[Sequence[str]] | None = None) -> None:
    args_list = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description="Hedge live executor")
    parser.parse_args(args_list)

    _sync_dry_run()
    _clean_testnet_caches()
    
    # Refresh exchange precision cache at startup
    try:
        from execution.exchange_precision import refresh_precision_cache
        refresh_precision_cache()
        LOG.info("[executor] exchange precision cache refreshed")
    except Exception as exc:
        LOG.warning("[executor] precision cache refresh failed: %s", exc)
    
    LOG.debug("[exutil] ENV context testnet=%s dry_run=%s", is_testnet(), DRY_RUN)
    log_v6_flag_snapshot(LOG)
    _maybe_write_v6_runtime_probe(force=True)
    try:
        if not _is_dual_side():
            LOG.warning("[executor] WARNING — account not in hedge (dualSide) mode")
    except Exception as e:
        LOG.error("[executor] dualSide check failed: %s", e)

    client = get_um_client()
    if getattr(client, "is_stub", False):
        LOG.warning(
            "[startup-sync] UMFutures client unavailable (%s)",
            um_client_error() or "unknown",
        )
    _startup_position_check(client)

    i = 0
    while True:
        _loop_once(i)
        _maybe_emit_heartbeat()
        _maybe_run_internal_screener()
        _maybe_write_v6_runtime_probe()
        i += 1
        if MAX_LOOPS and i >= MAX_LOOPS:
            LOG.info("[executor] MAX_LOOPS reached — exiting.")
            break
        time.sleep(SLEEP)


if __name__ == "__main__":
    main()
