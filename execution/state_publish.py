#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone

# Publishes read-only state to Firestore (positions + NAV).
# - Loads .env from repo root (override=True) so ad-hoc runs see keys.
# - Filters positions to non-zero qty and to symbols in pairs_universe.json (if present).
# - Debounces writes (executor can call this every loop safely).
# - Writes dashboard-facing state files under logs/state/* including kpis_v7.json
#   (aggregated NAV/risk/router KPIs for dashboard/investor views).
import logging
import os
import pathlib
import time
from typing import Any, Dict, List, Mapping, Optional

from utils.firestore_client import get_db
from execution.utils import get_usd_to_zar

# ----- robust .env load -----
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env", override=True)
except Exception:
    pass

def _resolve_env(default: str = "dev") -> str:
    value = (os.getenv("ENV") or os.getenv("ENVIRONMENT") or "").strip()
    return value or default


ENV = _resolve_env()
if ENV.lower() == "prod":
    allow_prod = os.getenv("ALLOW_PROD_WRITE", "0").strip().lower()
    if allow_prod not in {"1", "true", "yes"}:
        raise RuntimeError(
            "Refusing to publish state with ENV=prod. Set ALLOW_PROD_WRITE=1 to override explicitly."
        )
FS_ROOT = f"hedge/{ENV}/state"
LOG_DIR = ROOT_DIR / "logs"
EXEC_LOG_DIR = LOG_DIR / "execution"
STATE_DIR = LOG_DIR / "state"
_EXEC_STATS_CACHE: Dict[str, Any] = {"ts": 0.0, "data": None}
LOG = logging.getLogger("state_publish")


def _ensure_keys() -> None:
    """Backstop parse of .env in case python-dotenv wasn't available."""
    if os.getenv("BINANCE_API_KEY") and os.getenv("BINANCE_API_SECRET"):
        return
    env_path = ROOT_DIR / ".env"
    if not env_path.exists():
        return
    for ln in env_path.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#") or "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_ensure_keys()


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime,)):
        return value.isoformat()
    try:
        return float(value)
    except Exception:
        return str(value)


def _state_path(name: str, state_dir: pathlib.Path | None = None) -> pathlib.Path:
    target_dir = state_dir or STATE_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / name


def _write_state_file(name: str, payload: Dict[str, Any], state_dir: pathlib.Path | None = None) -> None:
    try:
        path = _state_path(name, state_dir)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, default=_json_default)
        tmp.replace(path)
    except Exception as exc:
        LOG.error("state_write_failed name=%s err=%s", name, exc)


def write_positions_state(payload: Dict[str, Any], state_dir: pathlib.Path | None = None) -> None:
    _write_state_file("positions.json", payload, state_dir)


def write_nav_state(payload: Dict[str, Any], state_dir: pathlib.Path | None = None) -> None:
    try:
        snapshot = dict(payload or {})
    except Exception:
        snapshot = payload or {}

    # Ensure AUM block exists for dashboard consumers
    if "aum" not in snapshot:
        nav_detail = snapshot.get("nav_detail") if isinstance(snapshot.get("nav_detail"), Mapping) else {}
        if isinstance(nav_detail, Mapping) and "aum" in nav_detail:
            snapshot["aum"] = nav_detail.get("aum")
        else:
            nav_total = snapshot.get("nav_total") or snapshot.get("nav") or snapshot.get("nav_usd")
            try:
                nav_total = float(nav_total) if nav_total is not None else None
            except Exception:
                nav_total = None
            snapshot["aum"] = {
                "futures": nav_total,
                "offexchange": {},
                "total": nav_total,
            }

    import logging  # noqa: WPS433 - local import to align with prompt

    logger = logging.getLogger(__name__)

    # --- v7 AUM Telemetry Guard ---
    aum = snapshot.get("aum")
    if not aum:
        logger.warning(
            "[nav-state] nav_snapshot missing AUM; injecting futures-only block. nav_total=%s",
            snapshot.get("nav_total"),
        )
    else:
        offx = aum.get("offexchange") or {}
        try:
            off_keys = sorted(offx.keys())
        except Exception:
            off_keys = []
        logger.info(
            "[nav-state] AUM present: futures=%s total=%s offexchange_keys=%s",
            aum.get("futures"),
            aum.get("total"),
            off_keys,
        )
        try:
            missing = [k for k, v in offx.items() if isinstance(v, Mapping) and v.get("usd_value") is None]
        except Exception:
            missing = []
        if missing:
            logger.warning("[nav-state] AUM warning: entries missing usd_value=%s", missing)

    _write_state_file("nav.json", snapshot, state_dir)


def write_router_health_state(payload: Dict[str, Any], state_dir: pathlib.Path | None = None) -> None:
    _write_state_file("router_health.json", payload, state_dir)


def write_risk_snapshot_state(payload: Dict[str, Any], state_dir: pathlib.Path | None = None) -> None:
    _write_state_file("risk_snapshot.json", payload, state_dir)


def write_execution_health_state(payload: Dict[str, Any], state_dir: pathlib.Path | None = None) -> None:
    _write_state_file("execution_health.json", payload, state_dir)


def write_universe_state(payload: Dict[str, Any], state_dir: pathlib.Path | None = None) -> None:
    _write_state_file("universe.json", payload, state_dir)


def write_expectancy_state(payload: Dict[str, Any], state_dir: pathlib.Path | None = None) -> None:
    _write_state_file("expectancy_v6.json", payload, state_dir)


def write_symbol_scores_state(payload: Dict[str, Any], state_dir: pathlib.Path | None = None) -> None:
    _write_state_file("symbol_scores_v6.json", payload, state_dir)


def write_router_policy_suggestions_state(payload: Dict[str, Any], state_dir: pathlib.Path | None = None) -> None:
    _write_state_file("router_policy_suggestions_v6.json", payload, state_dir)


def write_risk_allocation_suggestions_state(payload: Dict[str, Any], state_dir: pathlib.Path | None = None) -> None:
    _write_state_file("risk_allocation_suggestions_v6.json", payload, state_dir)


def write_pipeline_v6_shadow_state(payload: Dict[str, Any], state_dir: pathlib.Path | None = None) -> None:
    _write_state_file("pipeline_v6_shadow_head.json", payload, state_dir)


def write_pipeline_v6_compare_summary(payload: Dict[str, Any], state_dir: pathlib.Path | None = None) -> None:
    _write_state_file("pipeline_v6_compare_summary.json", payload, state_dir)


def write_v6_runtime_probe_state(payload: Dict[str, Any], state_dir: pathlib.Path | None = None) -> None:
    _write_state_file("v6_runtime_probe.json", payload, state_dir)


def _coerce_synced_items(items: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    if isinstance(items, Mapping):
        items = [items]
    for entry in items or []:
        if not isinstance(entry, Mapping):
            continue
        normalized.append(dict(entry))
    return normalized


def build_synced_state_payload(
    *,
    items: Any,
    nav: float,
    engine_version: str | None,
    flags: Mapping[str, Any] | None,
    updated_at: float,
    nav_snapshot: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    snapshot = {
        "items": _coerce_synced_items(items or []),
        "nav": float(nav),
        "engine_version": (engine_version or "v6.0-beta-preview"),
        "v6_flags": dict(flags or {}),
        "updated_at": float(updated_at),
    }
    if nav_snapshot:
        try:
            snapshot["nav_snapshot"] = dict(nav_snapshot)
        except Exception:
            snapshot["nav_snapshot"] = {}
    return snapshot


def write_synced_state(payload: Dict[str, Any], state_dir: pathlib.Path | None = None) -> None:
    try:
        snapshot = build_synced_state_payload(
            items=payload.get("items") or payload.get("positions") or [],
            nav=payload.get("nav", payload.get("nav_usd", 0.0)),
            engine_version=payload.get("engine_version"),
            flags=payload.get("v6_flags"),
            updated_at=payload.get("updated_at", time.time()),
            nav_snapshot=payload.get("nav_snapshot"),
        )
    except Exception as exc:
        LOG.debug("build_synced_state_payload_failed: %s", exc)
        snapshot = {
            "items": _coerce_synced_items(payload.get("items") or []),
            "nav": float(payload.get("nav", 0.0) or 0.0),
            "engine_version": str(payload.get("engine_version") or "v6.0-beta-preview"),
            "v6_flags": dict(payload.get("v6_flags") or {}),
            "updated_at": float(payload.get("updated_at", time.time())),
        }
    _write_state_file("synced_state.json", snapshot, state_dir)


def _normalize_status(value: Any) -> str:
    if not value:
        return "UNKNOWN"
    try:
        text = str(value).upper()
    except Exception:
        return "UNKNOWN"
    if text == "CANCELLED":
        return "CANCELED"
    return text


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_timestamp(record: Dict[str, Any]) -> Optional[float]:
    candidates = (
        record.get("ts"),
        record.get("timestamp"),
        record.get("time"),
        record.get("t"),
        record.get("local_ts"),
    )
    for value in candidates:
        if value is None:
            continue
        if isinstance(value, (int, float)):
            try:
                return float(value)
            except Exception:
                continue
        if isinstance(value, str):
            val = value.strip()
            if not val:
                continue
            try:
                return float(val)
            except ValueError:
                pass
            try:
                iso_val = val.replace("Z", "+00:00") if val.endswith("Z") else val
                dt = datetime.fromisoformat(iso_val)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.timestamp()
            except Exception:
                continue
    return None


def _to_iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except Exception:
        return None


def _iter_recent_records(path: pathlib.Path, cutoff: float):
    try:
        with path.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        return
    except Exception:
        return
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        if not isinstance(record, dict):
            continue
        ts = _extract_timestamp(record)
        if ts is None:
            continue
        if ts < cutoff:
            break
        yield record


def _avg(values: List[Optional[float]]) -> Optional[float]:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return float(sum(nums) / len(nums))


def build_kpis_v7(
    now_ts: float,
    nav_snapshot: Mapping[str, Any] | None,
    risk_snapshot: Mapping[str, Any] | None,
    router_state: Mapping[str, Any] | None,
    expectancy_state: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Build v7 KPI payload from existing snapshot structures (nav/risk/router).
    Consumers: dashboard panels, investor view.
    """
    ts_val = float(now_ts)
    try:
        ts_iso = datetime.fromtimestamp(ts_val, tz=timezone.utc).isoformat()
    except Exception:
        ts_iso = _to_iso(ts_val) or str(ts_val)

    nav_snap = dict(nav_snapshot or {})
    risk_snap = dict(risk_snapshot or {})
    router_snap = dict(router_state or {})
    expectancy_snap = dict(expectancy_state or {}) if expectancy_state else {}

    def _to_epoch_seconds(value: Any) -> Optional[float]:
        direct = _safe_float(value)
        if direct is not None:
            return direct
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                if text.endswith("Z"):
                    text = text[:-1] + "+00:00"
                return datetime.fromisoformat(text).timestamp()
            except Exception:
                return None
        return None

    def _first_float(*keys: str) -> Optional[float]:
        for key in keys:
            if key in nav_snap:
                val = _safe_float(nav_snap.get(key))
                if val is not None:
                    return val
        return None

    nav_total = _first_float("nav_total", "nav", "nav_usd", "total_equity")
    nav_age = _safe_float(nav_snap.get("nav_age_s") or nav_snap.get("age_s") or nav_snap.get("age"))
    sources_ok = nav_snap.get("sources_ok")
    aum_total = None
    try:
        aum_total = _safe_float((nav_snap.get("aum") or {}).get("total"))
    except Exception:
        aum_total = None
    nav_updated = _to_epoch_seconds(nav_snap.get("updated_at") or nav_snap.get("ts") or nav_snap.get("updated_ts"))
    if nav_age is None and nav_updated is not None:
        try:
            nav_age = max(0.0, time.time() - float(nav_updated))
        except Exception:
            nav_age = None

    dd_state = risk_snap.get("dd_state")
    if isinstance(dd_state, dict) and "dd_state" in dd_state:
        dd_state = dd_state.get("dd_state") or dd_state
    atr_regime = risk_snap.get("atr_regime")
    if atr_regime is None:
        try:
            atr_regime = (risk_snap.get("atr") or {}).get("atr_regime")
        except Exception:
            atr_regime = None
    fee_pnl_ratio = risk_snap.get("fee_pnl_ratio")
    if fee_pnl_ratio is None:
        try:
            fee_pnl_ratio = (risk_snap.get("fee_pnl") or {}).get("fee_pnl_ratio")
        except Exception:
            fee_pnl_ratio = None

    dd_pct = None
    try:
        dd_pct = _safe_float((risk_snap.get("drawdown") or {}).get("dd_pct"))
    except Exception:
        dd_pct = None
    atr_ratio = None
    try:
        atr_ratio = _safe_float((risk_snap.get("atr") or {}).get("median_ratio"))
    except Exception:
        atr_ratio = None

    router_entries = []
    per_symbol = router_snap.get("per_symbol") or router_snap.get("symbols")
    if isinstance(per_symbol, list):
        router_entries = [entry for entry in per_symbol if isinstance(entry, Mapping)]
    summary = router_snap.get("summary") if isinstance(router_snap.get("summary"), Mapping) else {}
    maker_fill_share = _avg(
        [
            _safe_float(entry.get("maker_fill_rate") or entry.get("maker_fill_ratio"))
            for entry in router_entries
        ]
    )
    avg_slippage_bps = _avg(
        [
            _safe_float(entry.get("slippage_p50") or entry.get("slip_q50"))
            for entry in router_entries
        ]
    )
    fallback_ratio = _avg(
        [
            _safe_float(entry.get("fallback_rate") or entry.get("fallback_ratio"))
            for entry in router_entries
        ]
    )
    slip_q25 = _avg([_safe_float(entry.get("slippage_p25") or entry.get("slip_q25")) for entry in router_entries])
    slip_q50 = _avg([_safe_float(entry.get("slippage_p50") or entry.get("slip_q50")) for entry in router_entries])
    slip_q75 = _avg([_safe_float(entry.get("slippage_p75") or entry.get("slip_q75")) for entry in router_entries])
    router_stats = risk_snap.get("router_stats") if isinstance(risk_snap.get("router_stats"), Mapping) else {}
    policy_quality = None
    if router_stats:
        fallback_ratio = (
            _safe_float(
                router_stats.get("fallback_ratio")
                or router_stats.get("fallback_rate")
                or router_stats.get("fallback")
            )
            or fallback_ratio
        )
        maker_fill_share = _safe_float(
            router_stats.get("maker_fill_ratio") or router_stats.get("maker_fill_rate") or maker_fill_share
        )
        slip_q25 = _safe_float(router_stats.get("slip_q25") or router_stats.get("slip_q25_bps") or slip_q25)
        slip_q50 = _safe_float(router_stats.get("slip_q50") or router_stats.get("slip_q50_bps") or slip_q50)
        slip_q75 = _safe_float(router_stats.get("slip_q75") or router_stats.get("slip_q75_bps") or slip_q75)
        avg_slippage_bps = _safe_float(router_stats.get("slip_q50_bps") or router_stats.get("slip_q50") or avg_slippage_bps)
        policy_quality = router_stats.get("quality") or router_stats.get("router_quality") or policy_quality
        if maker_first_flag is None:
            maker_first_flag = (
                router_stats.get("policy_maker_first")
                if isinstance(router_stats.get("policy_maker_first"), bool)
                else maker_first_flag
            )
    quality_counts = summary.get("quality_counts") if isinstance(summary, Mapping) else None
    if isinstance(quality_counts, Mapping) and quality_counts:
        try:
            policy_quality = max(quality_counts.items(), key=lambda kv: kv[1])[0]
        except Exception:
            policy_quality = None
    maker_first_flag: Optional[bool] = None
    try:
        total = int(summary.get("count") or len(router_entries) or 0)
        enabled = _safe_float(summary.get("maker_first_enabled"))
        if enabled is not None and total > 0:
            maker_first_flag = bool((enabled / float(total)) >= 0.5)
    except Exception:
        maker_first_flag = None
    reject_rate = None  # TODO: add when router health exposes reject counts

    expectancy_val = None
    for key in ("expectancy", "expectancy_usd", "expected_value"):
        if key in expectancy_snap:
            expectancy_val = _safe_float(expectancy_snap.get(key))
            if expectancy_val is not None:
                break
    sharpe_state = risk_snap.get("sharpe_state")
    if sharpe_state is None:
        try:
            symbols = risk_snap.get("symbols") or []
            for entry in symbols:
                risk_part = entry.get("risk") if isinstance(entry, Mapping) else None
                if isinstance(risk_part, Mapping) and risk_part.get("sharpe_state"):
                    sharpe_state = risk_part.get("sharpe_state")
                    break
        except Exception:
            sharpe_state = None

    try:
        usd_zar_rate = get_usd_to_zar()
    except Exception:
        usd_zar_rate = None

    symbol_risk: Dict[str, Dict[str, Any]] = {}
    try:
        for entry in risk_snap.get("symbols") or []:
            if not isinstance(entry, Mapping):
                continue
            sym = str(entry.get("symbol") or "").upper()
            if not sym:
                continue
            risk_part = entry.get("risk") if isinstance(entry.get("risk"), Mapping) else {}
            vol_part = entry.get("vol") if isinstance(entry.get("vol"), Mapping) else {}
            symbol_risk[sym] = {
                "symbol": sym,
                "dd_state": risk_part.get("dd_state"),
                "dd_today_pct": _safe_float(risk_part.get("dd_today_pct")),
                "atr_ratio": _safe_float(vol_part.get("atr_ratio")),
                "atr_regime": vol_part.get("atr_regime"),
            }
    except Exception:
        symbol_risk = symbol_risk or {}

    try:
        atr_block = risk_snap.get("atr") or {}
        for entry in atr_block.get("symbols") or []:
            if not isinstance(entry, Mapping):
                continue
            sym = str(entry.get("symbol") or "").upper()
            if not sym:
                continue
            target = symbol_risk.setdefault(sym, {})
            target.setdefault("symbol", sym)
            target.setdefault("atr_ratio", _safe_float(entry.get("atr_ratio")))
            target.setdefault("atr_regime", entry.get("atr_regime"))
    except Exception:
        pass

    payload = {
        "ts": ts_iso,
        "nav": {
            "nav_total": nav_total,
            "nav_age_s": nav_age,
            "age_s": nav_age,
            "updated_at": nav_updated,
            "sources_ok": sources_ok,
            "aum_total": aum_total,
        },
        "risk": {
            "dd_state": dd_state,
            "atr_regime": atr_regime,
            "fee_pnl_ratio": fee_pnl_ratio,
            "drawdown_pct": dd_pct,
            "atr_ratio": atr_ratio,
        },
        "router": {
            "policy_quality": policy_quality,
            "maker_first": maker_first_flag,
            "maker_fill_share": maker_fill_share,
            "avg_slippage_bps": avg_slippage_bps,
            "reject_rate": reject_rate,
            "maker_fill_rate": maker_fill_share,
            "fallback_ratio": fallback_ratio,
            "slip_q25_bps": slip_q25,
            "slip_q50_bps": slip_q50,
            "slip_q75_bps": slip_q75,
        },
        "performance": {"expectancy": expectancy_val, "sharpe_state": sharpe_state},
        # Convenience top-level mirrors for legacy/consumers
        "dd_state": dd_state,
        "atr_regime": atr_regime,
        "fee_pnl_ratio": fee_pnl_ratio,
        "router_quality": policy_quality,
        "aum_total": aum_total,
        "fx": {"usd_zar": usd_zar_rate, "ts": ts_iso},
        "router_stats": {
            "maker_fill_rate": maker_fill_share,
            "fallback_ratio": fallback_ratio,
            "slip_q25_bps": slip_q25,
            "slip_q50_bps": slip_q50,
            "slip_q75_bps": slip_q75,
            "quality": policy_quality,
        },
        "drawdown": {"dd_pct": dd_pct},
        "atr": {"atr_ratio": atr_ratio, "atr_regime": atr_regime, "symbols": list(symbol_risk.values())},
        "symbols": symbol_risk,
    }
    return payload


def write_kpis_v7_state(
    payload: Dict[str, Any] | None = None,
    state_dir: pathlib.Path | None = None,
    *,
    now_ts: float | None = None,
    nav_snapshot: Mapping[str, Any] | None = None,
    risk_snapshot: Mapping[str, Any] | None = None,
    router_state: Mapping[str, Any] | None = None,
    expectancy_state: Mapping[str, Any] | None = None,
) -> None:
    try:
        if payload is None or nav_snapshot is not None or risk_snapshot is not None or router_state is not None:
            payload = build_kpis_v7(
                now_ts if now_ts is not None else time.time(),
                nav_snapshot,
                risk_snapshot,
                router_state,
                expectancy_state=expectancy_state,
            )
    except Exception as exc:
        LOG.error("kpis_v7_build_failed: %s", exc)
        payload = payload or {}
    _write_state_file("kpis_v7.json", payload or {}, state_dir)


def _last_heartbeats() -> Dict[str, Optional[str]]:
    path = EXEC_LOG_DIR / "sync_heartbeats.jsonl"
    latest: Dict[str, float] = {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                if not isinstance(record, dict):
                    continue
                service = record.get("service")
                if not service:
                    continue
                ts = _extract_timestamp(record)
                if ts is None:
                    continue
                if ts >= latest.get(service, float("-inf")):
                    latest[service] = ts
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return {svc: _to_iso(ts) for svc, ts in latest.items()}


def _compute_exec_stats() -> Dict[str, Any]:
    now_ts = time.time()
    if (
        isinstance(_EXEC_STATS_CACHE.get("ts"), (int, float))
        and _EXEC_STATS_CACHE.get("data") is not None
        and (now_ts - float(_EXEC_STATS_CACHE["ts"])) < 30.0
    ):
        return _EXEC_STATS_CACHE["data"]

    try:
        cutoff = now_ts - 86400.0
        attempted = 0
        ack_ids: set[str] = set()
        fill_ids: set[str] = set()
        successful_ids: set[str] = set()
        veto_counter: Counter[str] = Counter()

        attempt_path = EXEC_LOG_DIR / "orders_attempted.jsonl"
        for _ in _iter_recent_records(attempt_path, cutoff) or []:
            attempted += 1

        executed_path = EXEC_LOG_DIR / "orders_executed.jsonl"
        legacy_counter = 0
        for record in _iter_recent_records(executed_path, cutoff) or []:
            if not isinstance(record, dict):
                continue
            event_type = str(record.get("event_type") or record.get("event") or "").lower()
            order_id = record.get("orderId") or record.get("order_id")
            client_id = record.get("clientOrderId") or record.get("client_order_id")
            identifier: str
            if order_id or client_id:
                identifier = str(order_id or client_id)
            else:
                legacy_counter += 1
                identifier = f"legacy_{legacy_counter}"
            if event_type == "order_ack":
                ack_ids.add(identifier)
                continue
            if event_type == "order_fill":
                fill_ids.add(identifier)
                status = _normalize_status(record.get("status"))
                executed_qty = (
                    _safe_float(record.get("executedQty"))
                    or _safe_float(record.get("qty"))
                )
                if executed_qty and executed_qty > 0 and status in {"FILLED", "PARTIALLY_FILLED"}:
                    successful_ids.add(identifier)
                continue
            if event_type == "order_close":
                continue
            # Legacy fallback: treat as executed fill (and ack) entry
            ack_ids.add(identifier)
            fill_ids.add(identifier)
            status = _normalize_status(record.get("status"))
            executed_qty = (
                _safe_float(record.get("executedQty"))
                or _safe_float(record.get("qty"))
            )
            if executed_qty and executed_qty > 0 and status in {"FILLED", "SUCCESS"}:
                successful_ids.add(identifier)

        veto_path = EXEC_LOG_DIR / "risk_vetoes.jsonl"
        for record in _iter_recent_records(veto_path, cutoff) or []:
            reason = record.get("veto_reason") or record.get("reason") or "unknown"
            veto_counter[str(reason)] += 1

        executed = len(fill_ids)
        successful = len(successful_ids)
        ack_count = len(ack_ids) if ack_ids else executed
        fill_rate = 0.0
        denominator = attempted if attempted > 0 else (ack_count if ack_count > 0 else executed)
        numerator = successful if attempted > 0 else successful
        if denominator > 0:
            fill_rate = float(numerator) / float(denominator)

        last_hb = _last_heartbeats()
        stats = {
            "attempted_24h": attempted,
            "executed_24h": executed,
            "vetoes_24h": sum(veto_counter.values()),
            "fill_rate": fill_rate,
            "top_vetoes": [
                {"reason": reason, "count": count}
                for reason, count in veto_counter.most_common(5)
            ],
            "last_heartbeats": {
                "executor_live": last_hb.get("executor_live"),
                "sync_daemon": last_hb.get("sync_daemon"),
            },
        }
    except Exception:
        stats = {
            "attempted_24h": 0,
            "executed_24h": 0,
            "vetoes_24h": 0,
            "fill_rate": 0.0,
            "top_vetoes": [],
            "last_heartbeats": {
                "executor_live": None,
                "sync_daemon": None,
            },
        }

    _EXEC_STATS_CACHE["ts"] = now_ts
    _EXEC_STATS_CACHE["data"] = stats
    return stats


def _firestore_enabled() -> bool:
    return False


# ----- Firestore client -----
def _fs_client():
    return None


# ----- Exchange helpers (import late so .env is loaded) -----
def get_positions(symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    from execution.exchange_utils import _req

    params = {}
    if symbol:
        params["symbol"] = symbol
    return _req("GET", "/fapi/v2/positionRisk", signed=True, params=params).json()


def get_account() -> Dict[str, Any]:
    from execution.exchange_utils import _req

    return _req("GET", "/fapi/v2/account", signed=True).json()


# ----- Normalizers / publishers -----
def normalize_positions(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Optional universe whitelist
    universe = None
    try:
        u = json.load(open(str(ROOT_DIR / "config/pairs_universe.json")))
        universe = set(u.get("symbols", []))
    except Exception:
        pass

    rows: List[Dict[str, Any]] = []
    for p in raw or []:
        try:
            sym = p.get("symbol")
            if universe and sym not in universe:
                continue
            qty = float(p.get("qty", p.get("positionAmt", 0)) or 0.0)
            if qty == 0.0:
                continue
            rows.append(
                {
                    "symbol": sym,
                    "positionSide": p.get("positionSide") or "BOTH",
                    "qty": qty,
                    "entryPrice": float(p.get("entryPrice") or 0),
                    "leverage": float(p.get("leverage") or 0),
                    "uPnl": float(
                        p.get("unRealizedProfit", p.get("unrealized", 0)) or 0
                    ),
                }
            )
        except Exception:
            pass
    return rows


def publish_positions(rows: List[Dict[str, Any]]) -> None:
    stats = _compute_exec_stats()
    payload = {"rows": rows, "updated": time.time(), "exec_stats": stats}
    if not _firestore_enabled():
        _append_local_jsonl("positions", payload)
        try:
            write_positions_state(payload)
        except Exception as exc:
            LOG.debug("write_positions_state_failed: %s", exc)
        return
    cli = _fs_client()
    cli.document(f"{FS_ROOT}/positions").set(payload, merge=True)


def compute_nav() -> float:
    acc = get_account()
    # Prefer totalMarginBalance (wallet + uPnL)
    tmb = acc.get("totalMarginBalance")
    if tmb is not None:
        return float(tmb)
    twb = float(acc.get("totalWalletBalance", 0) or 0)
    tup = float(acc.get("totalUnrealizedProfit", 0) or 0)
    return twb + tup


def publish_nav_value(
    nav: float, min_interval_s: int = 60, max_points: int = 20000
) -> None:
    now = time.time()
    stats = _compute_exec_stats()
    if not _firestore_enabled():
        payload = {"ts": now, "nav": float(nav), "exec_stats": stats}
        _append_local_jsonl("nav", payload)
        try:
            write_nav_state({"nav": float(nav), "updated_ts": now, "exec_stats": stats})
        except Exception as exc:
            LOG.debug("write_nav_state_failed: %s", exc)
        return
    cli = _fs_client()
    doc = cli.document(f"{FS_ROOT}/nav")
    snap = doc.get()
    data = snap.to_dict() if getattr(snap, "exists", False) else {}
    series = data.get("series") or []
    # Debounce
    if series:
        last = series[-1]
        try:
            last_t = float(last.get("t") or last.get("ts") or last.get("time") or 0)
        except Exception:
            last_t = 0.0
        if now - last_t < min_interval_s:
            return
    series.append({"t": now, "nav": float(nav)})
    if len(series) > max_points:
        series = series[-max_points:]
    doc.set({"series": series, "updated": now, "exec_stats": stats}, merge=True)


class StatePublisher:
    """Hash/debounce publisher for positions from the executor loop."""

    def __init__(self, interval_s: int = 60):
        self.interval_s = interval_s
        self._h: Optional[str] = None
        self._t: float = 0.0

    def maybe_publish_positions(self, rows: List[Dict[str, Any]]) -> None:
        body = json.dumps(rows, sort_keys=True, default=str).encode()
        h = hashlib.sha256(body).hexdigest()
        now = time.time()
        if h != self._h or (now - self._t) >= self.interval_s:
            publish_positions(rows)
            self._h = h
            self._t = now


# ----- Audit/event helpers -----
def _fs_client_safe():
    try:
        return _fs_client()
    except Exception:
        return None


def _append_local_jsonl(name: str, event: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"{name}.jsonl"
    with path.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(event) + '\n')


def _audit_append(doc_name: str, event: dict, max_len: int = 500) -> None:
    now = time.time()
    ev = dict(event)
    ev.setdefault("t", now)
    if not _firestore_enabled():
        _append_local_jsonl(doc_name, ev)
        return
    cli = _fs_client_safe()
    if not cli:
        _append_local_jsonl(doc_name, ev)
        return
    doc = cli.document(f"{FS_ROOT}/{doc_name}")
    try:
        snap = doc.get()
        data = snap.to_dict() if getattr(snap, "exists", False) else {}
    except Exception:
        data = {}
    hist = list(data.get("history", []))
    hist.append(ev)
    hist = hist[-max_len:]
    try:
        doc.set({"last": ev, "history": hist, "updated": now}, merge=True)
    except Exception:
        pass


def publish_intent_audit(intent: dict) -> None:
    ev = dict(intent)
    ev.setdefault("type", "intent")
    _audit_append("audit_intents", ev)


def publish_order_audit(symbol: str, event: dict) -> None:
    ev = dict(event)
    ev.setdefault("symbol", symbol)
    ev.setdefault("type", "order")
    _audit_append(f"audit_orders_{symbol}", ev)


def publish_close_audit(symbol: str, position_side: str, event: dict) -> None:
    ev = dict(event)
    ev.setdefault("symbol", symbol)
    ev.setdefault("positionSide", position_side)
    ev.setdefault("type", "close")
    _audit_append(f"audit_closes_{symbol}_{position_side}", ev)


if __name__ == "__main__":
    # Preflight: ensure keys & perms visible to THIS process
    from execution.exchange_utils import _req

    try:
        _ = _req("GET", "/fapi/v1/time")
        _ = _req("GET", "/fapi/v2/balance", signed=True)
    except Exception as e:
        print("preflight_error:", e)
        raise SystemExit(1)

    raw = get_positions()
    rows = normalize_positions(raw)
    publish_positions(rows)
    try:
        nav = compute_nav()
        publish_nav_value(nav)
        print(f"published positions: {len(rows)}, nav: {nav}")
    except Exception as e:
        print("nav_publish_warn:", e)


# ----- Exit plan publish/fetch -----
def publish_exit_plan(symbol: str, position_side: str, plan: dict) -> None:
    now = time.time()
    body = dict(plan)
    body["updated"] = now
    if not _firestore_enabled():
        _append_local_jsonl(f"exits_{symbol}_{position_side}", body)
        return
    cli = _fs_client()
    cli.document(f"{FS_ROOT}/exits_{symbol}_{position_side}").set(body, merge=True)


def get_exit_plan(symbol: str, position_side: str) -> dict | None:
    if not _firestore_enabled():
        return None
    cli = _fs_client()
    snap = cli.document(f"{FS_ROOT}/exits_{symbol}_{position_side}").get()
    return snap.to_dict() if getattr(snap, "exists", False) else None


def publish_exit_event(symbol: str, position_side: str, event: dict) -> None:
    now = time.time()
    ev = dict(event)
    ev.setdefault("t", now)
    if not _firestore_enabled():
        _append_local_jsonl(f"exit_event_{symbol}_{position_side}", ev)
        return
    cli = _fs_client()
    doc = cli.document(f"{FS_ROOT}/exit_event_{symbol}_{position_side}")
    snap = doc.get()
    data = snap.to_dict() if getattr(snap, "exists", False) else {}
    hist = list(data.get("history", []))
    hist.append(ev)
    doc.set({"last": ev, "history": hist[-200:], "updated": now}, merge=True)
