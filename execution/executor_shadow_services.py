from __future__ import annotations

import time
from typing import Any, Callable, Dict, Iterable, Mapping, Optional


def maybe_emit_execution_health_snapshot(
    *,
    force: bool,
    last_publish_ts: float,
    interval_s: float,
    collect_execution_health: Callable[[], Dict[str, Any]],
    exec_health_log: Any,
    write_execution_health_state: Callable[[Dict[str, Any]], None],
    logger: Any,
    doctrine_inputs: Optional[Dict[str, Any]] = None,
) -> tuple[float, Optional[Dict[str, Any]]]:
    now = time.time()
    if not force and (now - last_publish_ts) < interval_s:
        return last_publish_ts, None
    try:
        snapshot = collect_execution_health()
    except Exception as exc:
        logger.debug("[metrics] execution_health_collect_failed: %s", exc)
        return last_publish_ts, None
    if doctrine_inputs:
        snapshot["doctrine_inputs"] = doctrine_inputs
    try:
        snapshot.setdefault("type", "execution_health")
        snapshot.setdefault("context", "executor")
        exec_health_log.write(snapshot)
    except Exception as exc:
        logger.debug("[metrics] execution_health_log_failed: %s", exc)
    try:
        write_execution_health_state(snapshot)
    except Exception as exc:
        logger.debug("[metrics] execution_health_state_write_failed: %s", exc)
    return now, snapshot


def maybe_run_pipeline_v6_shadow(
    *,
    enabled: bool,
    symbol: str,
    side: str,
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
    risk_cfg: Mapping[str, Any],
    pairs_cfg: Mapping[str, Any],
    run_pipeline_v6_shadow: Callable[..., Dict[str, Any]],
    record_shadow_decision: Callable[[Dict[str, Any]], None],
    risk_engine: Any,
    logger: Any,
) -> None:
    if not enabled:
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
        result = run_pipeline_v6_shadow(
            symbol,
            signal_payload,
            nav_state,
            positions_state,
            risk_cfg,
            pairs_cfg,
            sizing_cfg,
            risk_engine=risk_engine,
        )
        record_shadow_decision(result)
    except Exception as exc:
        logger.debug("[shadow] pipeline_v6_failed symbol=%s err=%s", symbol, exc)


def maybe_run_pipeline_v6_shadow_heartbeat(
    *,
    enabled: bool,
    last_nav_state: Mapping[str, Any],
    last_positions_state: Mapping[str, Any],
    last_heartbeat_ts: float,
    interval_s: float,
    select_shadow_symbol: Callable[[Iterable[Mapping[str, Any]]], Optional[str]],
    sizing_cfg: Mapping[str, Any],
    risk_cfg: Mapping[str, Any],
    pairs_cfg: Mapping[str, Any],
    run_pipeline_v6_shadow: Callable[..., Dict[str, Any]],
    record_shadow_decision: Callable[[Dict[str, Any]], None],
    record_execution_error: Callable[..., None],
    risk_engine: Any,
    logger: Any,
) -> float:
    if not enabled:
        return last_heartbeat_ts
    if not last_nav_state or not last_positions_state:
        return last_heartbeat_ts
    now = time.time()
    if (now - last_heartbeat_ts) < interval_s:
        return last_heartbeat_ts
    raw_positions = last_positions_state.get("items") or last_positions_state.get("positions") or []
    positions_rows = list(raw_positions)
    symbol = select_shadow_symbol(positions_rows)
    if not symbol:
        return last_heartbeat_ts
    nav_state = dict(last_nav_state)
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
        result = run_pipeline_v6_shadow(
            symbol,
            signal_payload,
            nav_state,
            {"positions": positions_rows},
            risk_cfg,
            pairs_cfg,
            sizing_cfg,
            risk_engine=risk_engine,
        )
        heartbeat_result = dict(result)
        heartbeat_result["heartbeat"] = True
        record_shadow_decision(heartbeat_result)
        return now
    except Exception as exc:
        logger.debug("[shadow] pipeline_v6_heartbeat_failed symbol=%s err=%s", symbol, exc)
        try:
            record_execution_error(
                "pipeline_shadow",
                symbol=symbol,
                message="heartbeat_failed",
                context={"error": str(exc)},
            )
        except Exception:
            pass
        return last_heartbeat_ts


def maybe_run_pipeline_v6_compare(
    *,
    force: bool,
    last_compare_ts: float,
    interval_s: float,
    compare_pipeline_v6: Callable[[], Dict[str, Any]],
    record_execution_error: Callable[..., None],
    logger: Any,
) -> float:
    now = time.time()
    if not force and (now - last_compare_ts) < interval_s:
        return last_compare_ts
    try:
        compare_pipeline_v6()
        return now
    except Exception as exc:
        logger.debug("[shadow] pipeline_v6_compare_failed: %s", exc)
        try:
            record_execution_error(
                "pipeline_compare",
                symbol=None,
                message="compare_failed",
                context={"error": str(exc)},
            )
        except Exception:
            pass
        return last_compare_ts


__all__ = [
    "maybe_emit_execution_health_snapshot",
    "maybe_run_pipeline_v6_compare",
    "maybe_run_pipeline_v6_shadow",
    "maybe_run_pipeline_v6_shadow_heartbeat",
]
