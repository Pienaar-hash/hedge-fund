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
import tempfile
import time
from typing import Any, Dict, List, Mapping, Optional

from execution.diagnostics_metrics import (
    build_runtime_diagnostics_snapshot_with_liveness,
    _load_strategy_config,
)
from execution.router_metrics import RouterQualityConfig, load_router_quality_config
from utils.firestore_client import get_db
from execution.pnl_tracker import export_pnl_attribution_state
from execution.versioning import read_version
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
POSITIONS_STATE_PATH = STATE_DIR / "positions_state.json"
POSITIONS_LEDGER_PATH = STATE_DIR / "positions_ledger.json"
KPIS_V7_PATH = STATE_DIR / "kpis_v7.json"
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


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_state(path: str | pathlib.Path, payload: Any) -> None:
    """Write JSON payload atomically by fsyncing a temp file then replacing."""
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, default=_json_default, separators=(",", ":"))
        os.replace(tmp_path, target)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _state_path(name: str, state_dir: pathlib.Path | None = None) -> pathlib.Path:
    target_dir = state_dir or STATE_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / name


def _write_state_file(name: str, payload: Dict[str, Any], state_dir: pathlib.Path | None = None) -> None:
    try:
        path = _state_path(name, state_dir)
        _atomic_write_state(path, payload)
    except Exception as exc:
        LOG.error("state_write_failed name=%s err=%s", name, exc)


def write_positions_snapshot_state(payload: Dict[str, Any], state_dir: pathlib.Path | None = None) -> None:
    """Legacy positions snapshot writer (positions.json)."""
    _write_state_file("positions.json", payload, state_dir)


def write_positions_state(
    positions_rows: List[Dict[str, Any]],
    state_dir: pathlib.Path | None = None,
    *,
    updated_at: Optional[str] = None,
    path: pathlib.Path | None = None,
) -> None:
    """
    Canonical atomic writer for logs/state/positions_state.json.

    positions_rows: list of normalized position dicts with qty/entry_price/mark_price.
    """
    ts = updated_at or utc_now_iso()
    payload = {
        "updated_at": ts,
        "updated_ts": ts,  # backwards-compatible alias for consumers expecting updated_ts
        "positions": list(positions_rows or []),
    }
    target_path = path or (state_dir or STATE_DIR) / "positions_state.json"
    _atomic_write_state(target_path, payload)


def write_nav_state(payload: Dict[str, Any], state_dir: pathlib.Path | None = None) -> None:
    try:
        snapshot = dict(payload or {})
    except Exception:
        snapshot = payload or {}
    # Ensure updated_ts present
    if "updated_ts" not in snapshot:
        snapshot["updated_ts"] = snapshot.get("updated_at") or utc_now_iso()

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
    # Mirror to nav_state.json for dashboard fallback
    try:
        target = (state_dir or STATE_DIR) / "nav_state.json"
        _atomic_write_state(target, snapshot)
    except Exception:
        logger.debug("nav_state_write_failed", exc_info=True)
    try:
        export_pnl_attribution_state()
    except Exception as exc:
        LOG.warning("pnl_attribution_export_failed: %s", exc)


def _safe_float_val(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_iso_ts(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        ts = float(value)
        if ts > 1e12:
            ts /= 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except Exception:
        if isinstance(value, str) and value:
            return value
        return None


def _router_bucket_for_symbol(symbol: str) -> str:
    try:
        from execution.liquidity_model import get_bucket_for_symbol

        bucket = get_bucket_for_symbol(symbol)
        if bucket:
            name = getattr(bucket, "name", None) or getattr(bucket, "bucket", None)
            if name:
                return str(name).upper()
    except Exception:
        pass
    return "B_MEDIUM"


def _slippage_bucket(value: float, cfg: RouterQualityConfig) -> str:
    if value <= cfg.slippage_drift_green_bps:
        return "GREEN"
    if value <= cfg.slippage_drift_yellow_bps:
        return "YELLOW"
    return "RED"


def _latency_bucket(value: float, cfg: RouterQualityConfig) -> str:
    if value <= cfg.latency_fast_ms:
        return "FAST"
    if value <= cfg.latency_normal_ms:
        return "NORMAL"
    return "SLOW"


def _bucket_penalty(bucket: str, cfg: RouterQualityConfig) -> float:
    penalties = {
        "A_HIGH": cfg.bucket_penalty_a_high,
        "B_MEDIUM": cfg.bucket_penalty_b_medium,
        "C_LOW": cfg.bucket_penalty_c_low,
    }
    return penalties.get(bucket.upper(), 0.0)


def _compute_quality_score_from_components(
    cfg: RouterQualityConfig,
    bucket_penalty: float,
    drift_bucket: str,
    latency_bucket: str,
    twap_usage_ratio: float,
) -> float:
    score = cfg.base_score + bucket_penalty

    if drift_bucket == "YELLOW":
        score -= 0.05
    elif drift_bucket == "RED":
        score -= 0.12

    if latency_bucket == "NORMAL":
        score -= 0.03
    elif latency_bucket == "SLOW":
        score -= 0.08

    twap_penalty = cfg.twap_skip_penalty * max(0.0, min(1.0, 1.0 - twap_usage_ratio))
    score -= twap_penalty
    return max(cfg.min_score, min(cfg.max_score, score))


def _build_router_health_from_stats(
    stats_map: Mapping[str, Any],
    cfg: RouterQualityConfig,
    *,
    meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    per_symbol: Dict[str, Any] = {}
    weights: Dict[str, float] = {}
    bucket_weights: Dict[str, float] = {}

    for sym_raw, raw_stats in stats_map.items():
        sym = str(sym_raw or "").upper()
        if not sym or not isinstance(raw_stats, Mapping):
            continue
        avg_slip = _safe_float_val(raw_stats.get("avg_slippage_bps"))
        drift_val = _safe_float_val(raw_stats.get("slippage_drift_bps"), avg_slip)
        latency_val = _safe_float_val(raw_stats.get("avg_latency_ms"))
        twap_usage = max(0.0, min(1.0, _safe_float_val(raw_stats.get("twap_usage_ratio"))))
        notional = max(0.0, _safe_float_val(raw_stats.get("total_notional")))
        twap_notional = max(0.0, _safe_float_val(raw_stats.get("twap_notional")))
        last_order_iso = _to_iso_ts(raw_stats.get("last_order_ts"))
        last_fill_iso = _to_iso_ts(raw_stats.get("last_fill_ts"))
        child_orders = raw_stats.get("child_orders") or {}
        child_count = int(child_orders.get("count") or 0)
        child_fill_ratio = _safe_float_val(child_orders.get("fill_ratio"), 0.0)

        router_bucket = str(raw_stats.get("router_bucket") or _router_bucket_for_symbol(sym)).upper()
        drift_bucket = _slippage_bucket(drift_val, cfg)
        latency_bucket = _latency_bucket(latency_val, cfg)
        bucket_penalty = _bucket_penalty(router_bucket, cfg)

        quality_score = _compute_quality_score_from_components(
            cfg,
            bucket_penalty,
            drift_bucket,
            latency_bucket,
            twap_usage,
        )

        per_symbol[sym] = {
            "symbol": sym,
            "quality_score": quality_score,
            "avg_slippage_bps": avg_slip,
            "slippage_drift_bps": drift_val,
            "slippage_drift_bucket": drift_bucket,
            "avg_latency_ms": latency_val,
            "latency_bucket": latency_bucket,
            "last_order_ts": last_order_iso,
            "last_fill_ts": last_fill_iso,
            "twap_usage_ratio": twap_usage,
            "twap_notional": twap_notional,
            "child_orders": {
                "count": child_count,
                "fill_ratio": child_fill_ratio,
            },
            "router_bucket": router_bucket,
            "total_notional": notional,
            "event_count": int(raw_stats.get("event_count") or 0),
        }
        weight = notional if notional > 0 else 1.0
        weights[sym] = weight
        bucket_weights[router_bucket] = bucket_weights.get(router_bucket, 0.0) + weight

    total_weight = sum(weights.values()) or 1.0
    total_notional = sum(v.get("total_notional", 0.0) for v in per_symbol.values())
    total_twap_notional = sum(v.get("twap_notional", 0.0) for v in per_symbol.values())

    def _wavg(key: str) -> float:
        return sum((per_symbol[sym].get(key, 0.0) or 0.0) * weights.get(sym, 1.0) for sym in per_symbol) / total_weight

    avg_slip_global = _wavg("avg_slippage_bps")
    drift_global = _wavg("slippage_drift_bps")
    latency_global = _wavg("avg_latency_ms")
    if total_notional > 0 and total_twap_notional > 0:
        twap_usage_global = total_twap_notional / total_notional
    else:
        twap_usage_global = _wavg("twap_usage_ratio")
    twap_usage_global = max(0.0, min(1.0, twap_usage_global))

    global_bucket = max(bucket_weights.items(), key=lambda kv: kv[1])[0] if bucket_weights else "B_MEDIUM"
    global_bucket_penalty = _bucket_penalty(global_bucket, cfg)
    global_slip_bucket = _slippage_bucket(drift_global, cfg)
    global_latency_bucket = _latency_bucket(latency_global, cfg)
    global_score = _compute_quality_score_from_components(
        cfg,
        global_bucket_penalty,
        global_slip_bucket,
        global_latency_bucket,
        twap_usage_global,
    )

    quality_counts = {"good": 0, "ok": 0, "degraded": 0, "broken": 0}
    for entry in per_symbol.values():
        score = entry["quality_score"]
        if score >= cfg.high_quality_threshold:
            quality_counts["good"] += 1
        elif score >= cfg.low_quality_threshold:
            quality_counts["ok"] += 1
        elif score >= cfg.min_score:
            quality_counts["degraded"] += 1
        else:
            quality_counts["broken"] += 1

    updated_ts = utc_now_iso()
    legacy_entries = [
        {
            "symbol": sym,
            "avg_slippage_bps": entry["avg_slippage_bps"],
            "slippage_drift_bps": entry["slippage_drift_bps"],
            "latency_ms": entry["avg_latency_ms"],
            "router_bucket": entry["router_bucket"],
            "quality_score": entry["quality_score"],
            "twap_usage_ratio": entry["twap_usage_ratio"],
            "updated_ts": entry.get("last_fill_ts") or updated_ts,
        }
        for sym, entry in per_symbol.items()
    ]

    router_health_block: Dict[str, Any] = {
        "global": {
            "quality_score": global_score,
            "avg_slippage_bps": avg_slip_global,
            "slippage_drift_bps": drift_global,
            "slippage_drift_bucket": global_slip_bucket,
            "avg_latency_ms": latency_global,
            "latency_bucket": global_latency_bucket,
            "twap_usage_ratio": twap_usage_global,
            "router_bucket": global_bucket,
            "updated_ts": updated_ts,
        },
        "per_symbol": per_symbol,
    }
    if meta:
        router_health_block["meta"] = dict(meta)

    return {
        "updated_ts": updated_ts,
        "router_health_score": global_score,
        "router_health": router_health_block,
        "global": router_health_block["global"],
        "summary": {
            "updated_ts": updated_ts,
            "count": len(per_symbol),
            "quality_counts": quality_counts,
        },
        "symbols": legacy_entries,
        "per_symbol": legacy_entries,
        "symbol_stats": {
            sym: {
                "maker_reliability": entry["quality_score"],
                "avg_slippage_bps": entry["avg_slippage_bps"],
                "last_ts": entry.get("last_fill_ts") or entry.get("last_order_ts"),
            }
            for sym, entry in per_symbol.items()
        },
    }


def write_router_health_state(
    payload: Optional[Dict[str, Any]] = None,
    state_dir: pathlib.Path | None = None,
    *,
    router_stats_snapshot: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Write router health state with computed quality scores and buckets.

    If router_stats_snapshot is provided, builds the new router_health structure.
    Falls back to legacy payload scoring for compatibility.
    """
    stats_map: Dict[str, Any] = {}
    meta: Dict[str, Any] = {}
    if isinstance(router_stats_snapshot, Mapping):
        maybe = router_stats_snapshot.get("per_symbol")
        if isinstance(maybe, Mapping):
            stats_map = {str(k).upper(): v for k, v in maybe.items() if isinstance(v, Mapping)}
        for key in ("window_seconds", "min_events", "updated_ts"):
            if key in router_stats_snapshot:
                meta[key] = router_stats_snapshot.get(key)

    if not stats_map and isinstance(payload, Mapping):
        rh = payload.get("router_health")
        if isinstance(rh, Mapping) and isinstance(rh.get("per_symbol"), Mapping):
            stats_map = {str(k).upper(): v for k, v in rh.get("per_symbol", {}).items() if isinstance(v, Mapping)}
        elif isinstance(payload.get("per_symbol"), Mapping):
            stats_map = {str(k).upper(): v for k, v in payload.get("per_symbol", {}).items() if isinstance(v, Mapping)}

    try:
        cfg = load_router_quality_config()
    except Exception:
        cfg = RouterQualityConfig()

    if stats_map:
        try:
            computed_payload = _build_router_health_from_stats(stats_map, cfg, meta=meta)
        except Exception as exc:
            LOG.warning("router_health_state_build_failed: %s", exc)
            computed_payload = payload or {}
        _write_state_file("router_health.json", computed_payload, state_dir)
        return

    legacy_payload = dict(payload or {})
    if "router_health_score" not in legacy_payload:
        try:
            from execution.router_metrics import compute_router_health_score

            maker_ratio = legacy_payload.get("maker_ratio") or legacy_payload.get("maker_fill_rate") or 0.0
            fallback_ratio = legacy_payload.get("fallback_ratio") or 0.0
            avg_slippage = legacy_payload.get("avg_slippage_bps") or legacy_payload.get("slip_q50_bps") or 0.0
            reject_ratio = legacy_payload.get("reject_ratio") or legacy_payload.get("reject_rate") or 0.0

            legacy_payload["router_health_score"] = compute_router_health_score(
                maker_ratio=float(maker_ratio) if maker_ratio else 0.0,
                fallback_ratio=float(fallback_ratio) if fallback_ratio else 0.0,
                avg_slippage_bps=float(avg_slippage) if avg_slippage else 0.0,
                reject_ratio=float(reject_ratio) if reject_ratio else 0.0,
            )
        except Exception:
            legacy_payload["router_health_score"] = 0.0

    _write_state_file("router_health.json", legacy_payload, state_dir)


def _to_frac(value: Any) -> Optional[float]:
    """
    Normalize a percentage-style value to a fraction (0-1).
    Values >1 are treated as percent-style (e.g., 1.14 -> 0.0114).
    Values <=1 are assumed already fractional and returned unchanged.
    Returns None if value is None or not numeric.
    """
    if value is None:
        return None
    try:
        v = float(value)
        if v > 1.0:
            return v / 100.0
        return v
    except (TypeError, ValueError):
        return None


def write_risk_snapshot_state(
    payload: Dict[str, Any],
    state_dir: pathlib.Path | None = None,
    *,
    router_health: Optional[Dict[str, Any]] = None,
) -> None:
    """Write risk_snapshot.json with added normalized fractional fields and risk mode."""
    from execution.risk_engine_v6 import compute_risk_mode_from_state
    from execution.drawdown_tracker import (
        get_portfolio_dd_state,
        load_nav_anomaly_config,
    )
    from execution.risk_loader import load_risk_config

    # Load previous snapshot for nav/dd continuity
    prev_snapshot: Dict[str, Any] = {}
    try:
        rs_path = (state_dir or STATE_DIR) / "risk_snapshot.json"
        if rs_path.exists():
            with rs_path.open("r", encoding="utf-8") as handle:
                prev_snapshot = json.load(handle) or {}
    except Exception:
        prev_snapshot = {}

    # Add normalized fractional fields for drawdown and daily loss
    enriched = dict(payload) if payload else {}
    anomalies: Dict[str, Any] = dict(enriched.get("anomalies") or {})
    dd_state_block = enriched.get("dd_state") or {}
    drawdown_block = dd_state_block.get("drawdown") or {} if isinstance(dd_state_block, dict) else {}
    has_input_dd_state = isinstance(dd_state_block, dict) and bool(dd_state_block)
    
    # Extract dd_pct from nested structure: dd_state.drawdown.dd_pct
    dd_pct_raw = drawdown_block.get("dd_pct")
    # Extract daily_loss.pct from nested structure: dd_state.drawdown.daily_loss.pct
    daily_loss_block = drawdown_block.get("daily_loss") or {}
    daily_loss_pct_raw = daily_loss_block.get("pct")
    
    # Compute normalized fractions
    dd_frac = _to_frac(dd_pct_raw)
    daily_loss_frac = _to_frac(daily_loss_pct_raw)
    
    # Add fractional fields to payload (preserve originals)
    enriched["dd_frac"] = dd_frac
    enriched["daily_loss_frac"] = daily_loss_frac
    enriched["updated_ts"] = enriched.get("updated_ts") or time.time()

    # Compute portfolio DD and circuit breaker status
    try:
        risk_cfg = load_risk_config()
        cb_cfg = risk_cfg.get("circuit_breakers") or {}
        max_portfolio_dd_nav_pct = cb_cfg.get("max_portfolio_dd_nav_pct")

        # Load NAV history for portfolio DD computation
        nav_log_path = LOG_DIR / "nav_log.json"
        nav_history: List[float] = []
        if nav_log_path.exists():
            try:
                import json as _json
                with nav_log_path.open("r", encoding="utf-8") as handle:
                    nav_log_data = _json.load(handle)
                if isinstance(nav_log_data, list):
                    for entry in nav_log_data[-200:]:
                        if isinstance(entry, dict):
                            nav_val = entry.get("nav") or entry.get("nav_usd")
                            if nav_val is not None:
                                try:
                                    nav_float = float(nav_val)
                                    if nav_float > 0:
                                        nav_history.append(nav_float)
                                except (TypeError, ValueError):
                                    pass
            except Exception:
                pass

        dd_state_obj = get_portfolio_dd_state(nav_history) if nav_history else None
        portfolio_dd_pct = dd_state_obj.current_dd_pct if dd_state_obj else None
        prev_dd_state = str(enriched.get("dd_state", {}).get("state") if isinstance(enriched.get("dd_state"), dict) else enriched.get("dd_state") or "NORMAL").upper()
        # Fallback: derive DD from nav delta if we don't have history-derived dd_state
        nav_curr_val = enriched.get("nav_total") or enriched.get("nav") or enriched.get("nav_usd")
        nav_prev_val = prev_snapshot.get("nav_total") or prev_snapshot.get("nav") or prev_snapshot.get("nav_usd")
        nav_prev = None
        try:
            nav_prev = float(nav_prev_val) if nav_prev_val is not None else None
        except Exception:
            nav_prev = None
        try:
            nav_curr = float(nav_curr_val) if nav_curr_val is not None else None
        except Exception:
            nav_curr = None
        dd_state = "NORMAL"
        if nav_prev is not None and nav_curr is not None:
            peak = max(nav_prev, nav_curr)
            dd_frac_est = (peak - nav_curr) / peak if peak > 0 else 0.0
            portfolio_dd_pct = max(portfolio_dd_pct or 0.0, dd_frac_est)
        if portfolio_dd_pct and portfolio_dd_pct > 0:
            threshold = max_portfolio_dd_nav_pct or 0.01
            if portfolio_dd_pct >= threshold:
                dd_state = "DRAWDOWN"
            elif prev_dd_state in {"DRAWDOWN", "RECOVERY"}:
                dd_state = "RECOVERY"
        circuit_breaker_active = False
        if max_portfolio_dd_nav_pct is not None and dd_state_obj is not None:
            circuit_breaker_active = dd_state_obj.current_dd_pct >= max_portfolio_dd_nav_pct

        enriched["portfolio_dd_pct"] = portfolio_dd_pct
        if not has_input_dd_state:
            enriched["dd_state"] = {"state": dd_state, "drawdown": {"dd_pct": portfolio_dd_pct}}
        enriched["circuit_breaker"] = {
            "max_portfolio_dd_nav_pct": max_portfolio_dd_nav_pct,
            "active": circuit_breaker_active,
        }
    except Exception as exc:
        LOG.warning("circuit_breaker_computation_failed: %s", exc)
        enriched["portfolio_dd_pct"] = None
        enriched.setdefault("dd_state", {"state": "UNKNOWN"})
        enriched["circuit_breaker"] = {
            "max_portfolio_dd_nav_pct": None,
            "active": False,
        }

    # Compute correlation group exposures
    try:
        from execution.risk_loader import load_correlation_groups_config
        from execution.correlation_groups import compute_group_exposure_nav_pct

        corr_cfg = load_correlation_groups_config()
        
        if corr_cfg.groups:
            # Get positions from payload or existing state
            positions_data = enriched.get("positions") or []
            if not positions_data:
                # Try to load from positions state file
                positions_path = (state_dir or STATE_DIR) / "positions.json"
                if positions_path.exists():
                    try:
                        with positions_path.open("r", encoding="utf-8") as f:
                            positions_payload = json.load(f)
                            positions_data = positions_payload.get("positions") or []
                    except Exception:
                        positions_data = []

            # Get NAV for exposure calculation
            nav_for_corr = enriched.get("nav_total") or enriched.get("nav") or 0
            try:
                nav_for_corr = float(nav_for_corr) if nav_for_corr else 0.0
            except (TypeError, ValueError):
                nav_for_corr = 0.0

            if nav_for_corr > 0:
                current_exposure = compute_group_exposure_nav_pct(
                    positions=positions_data,
                    nav_total_usd=nav_for_corr,
                    corr_cfg=corr_cfg,
                )
                
                correlation_groups_state = {}
                for group_name, group_cfg in corr_cfg.groups.items():
                    correlation_groups_state[group_name] = {
                        "gross_nav_pct": current_exposure.get(group_name, 0.0),
                        "max_group_nav_pct": group_cfg.max_group_nav_pct,
                    }
                enriched["correlation_groups"] = correlation_groups_state
            else:
                # NAV invalid - return empty but with caps for reference
                enriched["correlation_groups"] = {
                    group_name: {
                        "gross_nav_pct": 0.0,
                        "max_group_nav_pct": group_cfg.max_group_nav_pct,
                    }
                    for group_name, group_cfg in corr_cfg.groups.items()
                }
        else:
            enriched["correlation_groups"] = {}
    except Exception as exc:
        LOG.warning("correlation_groups_computation_failed: %s", exc)
        enriched["correlation_groups"] = {}
    
    # Compute risk mode
    try:
        nav_health = enriched.get("nav_health")
        # Create a snapshot dict with dd_frac/daily_loss_frac for risk mode computation
        risk_snapshot_for_mode = {
            "dd_frac": dd_frac,
            "daily_loss_frac": daily_loss_frac,
        }
        risk_mode_result = compute_risk_mode_from_state(
            nav_health=nav_health,
            risk_snapshot=risk_snapshot_for_mode,
            router_health=router_health,
        )
        enriched["risk_mode"] = risk_mode_result.mode.value
        enriched["risk_mode_reason"] = risk_mode_result.reason
        enriched["risk_mode_score"] = risk_mode_result.score
    except Exception as exc:
        LOG.warning("risk_mode_computation_failed: %s", exc)
        enriched["risk_mode"] = "HALTED"
        enriched["risk_mode_reason"] = f"computation_error: {exc}"
        enriched["risk_mode_score"] = 1.0
    
    # V7.4_C2: TP/SL registry canary check
    try:
        from execution.position_tp_sl_registry import get_all_tp_sl_positions
        
        # Get position count from positions state
        positions_path = (state_dir or STATE_DIR) / "positions.json"
        num_positions = 0
        if positions_path.exists():
            try:
                with positions_path.open("r", encoding="utf-8") as f:
                    positions_payload = json.load(f)
                    rows = positions_payload.get("rows") or []
                    # Count positions with non-zero qty
                    num_positions = sum(
                        1 for r in rows
                        if abs(float(r.get("qty") or r.get("positionAmt") or 0)) > 0
                    )
            except Exception:
                pass
        
        tp_sl_registry = get_all_tp_sl_positions()
        num_registry_entries = len(tp_sl_registry)
        registry_mismatch = bool(num_positions > 0 and num_registry_entries == 0)
        
        enriched["tp_sl_registry"] = {
            "num_positions": num_positions,
            "num_registry_entries": num_registry_entries,
            "registry_mismatch": registry_mismatch,
        }
        
        if registry_mismatch:
            LOG.warning(
                "[CANARY] TP/SL registry mismatch: %d positions, 0 registry entries — exit layer impaired!",
                num_positions,
            )
    except Exception as exc:
        LOG.debug("tp_sl_registry_canary_check_failed: %s", exc)
        enriched["tp_sl_registry"] = {
            "num_positions": 0,
            "num_registry_entries": 0,
            "registry_mismatch": False,
        }
    
    # v7.5_A1: Add VaR and CVaR to risk snapshot
    try:
        from execution.vol_risk import (
            load_var_config,
            load_cvar_config,
            compute_portfolio_var_from_positions,
            compute_all_position_cvars,
        )
        
        # Get NAV for VaR/CVaR calculation
        nav_for_var = enriched.get("nav_total") or enriched.get("nav") or 0
        try:
            nav_for_var = float(nav_for_var) if nav_for_var else 0.0
        except (TypeError, ValueError):
            nav_for_var = 0.0
        
        var_cfg = load_var_config()
        cvar_cfg = load_cvar_config()
        
        if nav_for_var > 0:
            # Compute Portfolio VaR
            if var_cfg.enabled:
                var_result = compute_portfolio_var_from_positions(
                    positions_data,
                    nav_for_var,
                    var_cfg,
                )
                enriched["var"] = {
                    "portfolio_var_usd": var_result.var_usd,
                    "portfolio_var_nav_pct": var_result.var_nav_pct,
                    "max_portfolio_var_nav_pct": var_cfg.max_portfolio_var_nav_pct,
                    "within_limit": var_result.within_limit,
                    "portfolio_volatility": var_result.portfolio_volatility,
                    "confidence": var_cfg.confidence,
                    "n_assets": var_result.n_assets,
                }
            
            # Compute Position CVaRs
            if cvar_cfg.enabled:
                cvar_results = compute_all_position_cvars(
                    positions_data,
                    nav_for_var,
                    cvar_cfg,
                )
                enriched["cvar"] = {
                    "per_symbol": {
                        symbol: {
                            "cvar_nav_pct": result.cvar_nav_pct,
                            "limit": cvar_cfg.max_position_cvar_nav_pct,
                            "within_limit": result.within_limit,
                        }
                        for symbol, result in cvar_results.items()
                    },
                    "max_position_cvar_nav_pct": cvar_cfg.max_position_cvar_nav_pct,
                    "confidence": cvar_cfg.confidence,
                }
                # Flag anomaly if any CVaR exceeds limits
                breach = any(
                    (isinstance(info, dict) and info.get("cvar_nav_pct") is not None and info.get("cvar_nav_pct") > cvar_cfg.max_position_cvar_nav_pct)
                    for info in enriched["cvar"]["per_symbol"].values()
                )
                if breach:
                    anomalies["cvar_limit_breach"] = True
    except ImportError:
        LOG.debug("vol_risk module not available for VaR/CVaR enrichment")
    except Exception as exc:
        LOG.debug("var_cvar_enrichment_failed: %s", exc)

    # Nav anomaly guard using previous snapshot
    try:
        prev_snapshot = {}
        rs_path = (state_dir or STATE_DIR) / "risk_snapshot.json"
        if rs_path.exists():
            with rs_path.open("r", encoding="utf-8") as handle:
                prev_snapshot = json.load(handle) or {}
        nav_cfg = load_nav_anomaly_config()
        nav_curr = enriched.get("nav_total") or enriched.get("nav") or enriched.get("nav_usd")
        nav_prev = prev_snapshot.get("nav_total") or prev_snapshot.get("nav") or prev_snapshot.get("nav_usd")
        nav_curr = float(nav_curr) if nav_curr is not None else None
        nav_prev = float(nav_prev) if nav_prev is not None else None
        if nav_cfg.enabled and nav_curr is not None and nav_prev is not None and nav_prev > 0:
            pct_jump = abs(nav_curr - nav_prev) / max(nav_prev, 1e-9)
            pct_threshold = max(nav_cfg.max_multiplier_intraday - 1.0, 0.0)
            gap_ok = abs(nav_curr - nav_prev) <= nav_cfg.max_gap_abs_usd
            if pct_jump > pct_threshold or not gap_ok:
                anomalies["nav_jump"] = {
                    "old": nav_prev,
                    "new": nav_curr,
                    "pct": pct_jump,
                }
    except Exception as exc:
        LOG.debug("nav_anomaly_guard_failed: %s", exc)

    # VaR breach anomaly
    try:
        var_block = enriched.get("var") or {}
        var_pct = var_block.get("portfolio_var_nav_pct")
        var_limit = var_block.get("max_portfolio_var_nav_pct")
        if var_pct is not None and var_limit is not None and var_pct > var_limit:
            anomalies["var_limit_breach"] = True
    except Exception:
        pass

    enriched["anomalies"] = anomalies

    _write_state_file("risk_snapshot.json", enriched, state_dir)


def write_execution_health_state(payload: Dict[str, Any], state_dir: pathlib.Path | None = None) -> None:
    _write_state_file("execution_health.json", payload, state_dir)


def write_universe_state(payload: Dict[str, Any], state_dir: pathlib.Path | None = None) -> None:
    _write_state_file("universe.json", payload, state_dir)


def write_expectancy_state(payload: Dict[str, Any], state_dir: pathlib.Path | None = None) -> None:
    _write_state_file("expectancy_v6.json", payload, state_dir)


def write_symbol_scores_state(payload: Dict[str, Any], state_dir: pathlib.Path | None = None) -> None:
    snapshot = dict(payload or {})
    snapshot.setdefault("updated_ts", utc_now_iso())
    _write_state_file("symbol_scores_v6.json", snapshot, state_dir)


def write_engine_metadata_state(
    payload: Mapping[str, Any] | None = None,
    state_dir: pathlib.Path | None = None,
) -> Dict[str, Any]:
    """Write engine metadata (version, git commit, run_id) for dashboards/preflight."""
    snapshot = {
        "engine_version": read_version(default="v7.6"),
        "updated_ts": utc_now_iso(),
    }
    if isinstance(payload, Mapping):
        snapshot.update(payload)
    snapshot.setdefault("engine_version", read_version(default="v7.6"))
    snapshot.setdefault("updated_ts", utc_now_iso())
    _write_state_file("engine_metadata.json", snapshot, state_dir)
    return snapshot


def _load_liveness_cfg() -> Dict[str, Any]:
    cfg = _load_strategy_config()
    diag_block = cfg.get("diagnostics") if isinstance(cfg, Mapping) else {}
    if isinstance(diag_block, Mapping):
        live = diag_block.get("liveness")
        if isinstance(live, Mapping):
            return dict(live)
    return {}


def write_runtime_diagnostics_state(
    state_dir: pathlib.Path | None = None,
    liveness_cfg: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Write runtime diagnostics (veto counters + exit pipeline health) to diagnostics.json.
    """
    cfg = dict(liveness_cfg) if isinstance(liveness_cfg, Mapping) else _load_liveness_cfg()
    snapshot = build_runtime_diagnostics_snapshot_with_liveness(cfg)
    vc = snapshot.veto_counters
    es = snapshot.exit_pipeline_status
    la = snapshot.liveness_alerts
    payload = {
        "runtime_diagnostics": {
            "veto_counters": {
                "by_reason": dict(vc.by_reason),
                "total_signals": vc.total_signals,
                "total_orders": vc.total_orders,
                "total_vetoes": vc.total_vetoes,
                "last_signal_ts": vc.last_signal_ts,
                "last_order_ts": vc.last_order_ts,
                "last_veto_ts": vc.last_veto_ts,
            },
            "exit_pipeline": {
                "last_exit_scan_ts": es.last_exit_scan_ts,
                "last_exit_trigger_ts": es.last_exit_trigger_ts,
                "last_router_event_ts": es.last_router_event_ts,
                "open_positions_count": es.open_positions_count,
                "tp_sl_registered_count": es.tp_sl_registered_count,
                "tp_sl_missing_count": es.tp_sl_missing_count,
                "underwater_without_tp_sl_count": es.underwater_without_tp_sl_count,
                "tp_sl_coverage_pct": getattr(es, "tp_sl_coverage_pct", 0.0),
                "ledger_registry_mismatch": getattr(es, "ledger_registry_mismatch", False),
                "mismatch_breakdown": getattr(es, "mismatch_breakdown", {}) or {},
            },
            "liveness": {
                "idle_signals": bool(la.idle_signals) if la else False,
                "idle_orders": bool(la.idle_orders) if la else False,
                "idle_exits": bool(la.idle_exits) if la else False,
                "idle_router": bool(la.idle_router) if la else False,
                "details": dict(la.details) if la and la.details is not None else {},
                "missing": dict(la.missing) if la and la.missing is not None else {},
            },
        }
    }
    payload["engine_version"] = read_version(default="v7.6")
    payload["updated_ts"] = utc_now_iso()
    _write_state_file("diagnostics.json", payload, state_dir)
    return payload


def write_router_policy_suggestions_state(payload: Dict[str, Any], state_dir: pathlib.Path | None = None) -> None:
    _write_state_file("router_policy_suggestions_v6.json", payload, state_dir)


def write_risk_allocation_suggestions_state(payload: Dict[str, Any], state_dir: pathlib.Path | None = None) -> None:
    _write_state_file("risk_allocation_suggestions_v6.json", payload, state_dir)


def write_pipeline_v6_shadow_state(payload: Dict[str, Any], state_dir: pathlib.Path | None = None) -> None:
    _write_state_file("pipeline_v6_shadow_head.json", payload, state_dir)


def write_pipeline_v6_compare_summary(payload: Dict[str, Any], state_dir: pathlib.Path | None = None) -> None:
    _write_state_file("pipeline_v6_compare_summary.json", payload, state_dir)

def _preserve_pipeline_intent(intent: Any) -> Any:
    if not isinstance(intent, Mapping):
        return intent
    metadata = intent.get("metadata")
    if isinstance(metadata, Mapping):
        sanitized = dict(intent)
        sanitized["metadata"] = dict(metadata)
        return sanitized
    return dict(intent)


def write_pipeline_snapshot_state(payload: Dict[str, Any], state_dir: pathlib.Path | None = None) -> None:
    if not isinstance(payload, Mapping):
        _write_state_file("pipeline_snapshot.json", {}, state_dir)
        return
    enriched = dict(payload)
    intents = enriched.get("intents")
    if isinstance(intents, list):
        enriched["intents"] = [_preserve_pipeline_intent(entry) for entry in intents]
    _write_state_file("pipeline_snapshot.json", enriched, state_dir)


def write_v6_runtime_probe_state(payload: Dict[str, Any], state_dir: pathlib.Path | None = None) -> None:
    _write_state_file("v6_runtime_probe.json", payload, state_dir)


def write_positions_ledger_state(
    payload: Dict[str, Any],
    state_dir: pathlib.Path | None = None,
    *,
    path: pathlib.Path | None = None,
) -> None:
    """
    Canonical atomic writer for logs/state/positions_ledger.json.
    
    payload is the unified ledger snapshot (entries + tp_sl_levels + metadata).
    """
    target_path = path or (state_dir or STATE_DIR) / "positions_ledger.json"
    _atomic_write_state(target_path, payload)


def write_risk_advanced_state(payload: Dict[str, Any], state_dir: pathlib.Path | None = None) -> None:
    """
    Write risk_advanced.json with VaR and CVaR metrics (v7.5_A1).
    
    This file contains:
    - Portfolio VaR (parametric EWMA)
    - Per-position CVaR (Expected Shortfall)
    """
    _write_state_file("risk_advanced.json", payload, state_dir)


def write_alpha_decay_state(payload: Dict[str, Any], state_dir: pathlib.Path | None = None) -> None:
    """
    Write alpha_decay.json with signal decay state (v7.5_A1).
    
    This file contains decay multipliers per symbol-direction pair.
    """
    _write_state_file("alpha_decay.json", payload, state_dir)


def compute_and_write_risk_advanced_state(
    positions: List[Dict[str, Any]],
    nav_usd: float,
    state_dir: pathlib.Path | None = None,
) -> Dict[str, Any]:
    """
    Compute VaR/CVaR and write to state file (v7.5_A1).
    
    Args:
        positions: List of position dicts with 'symbol' and 'notional'
        nav_usd: Total portfolio NAV
        state_dir: Optional state directory override
        
    Returns:
        The computed risk advanced snapshot
    """
    try:
        from execution.vol_risk import build_risk_advanced_snapshot
        
        snapshot = build_risk_advanced_snapshot(positions, nav_usd)
        write_risk_advanced_state(snapshot, state_dir)
        return snapshot
    except ImportError:
        LOG.debug("vol_risk module not available for risk_advanced state")
        return {}
    except Exception as exc:
        LOG.warning("risk_advanced_computation_failed: %s", exc)
        return {}


def compute_and_write_alpha_decay_state(
    symbols: List[str],
    state_dir: pathlib.Path | None = None,
) -> Dict[str, Any]:
    """
    Compute alpha decay and write to state file (v7.5_A1).
    
    Args:
        symbols: List of trading pair symbols
        state_dir: Optional state directory override
        
    Returns:
        The computed alpha decay snapshot
    """
    try:
        from execution.intel.symbol_score_v6 import build_alpha_decay_snapshot
        
        snapshot = build_alpha_decay_snapshot(symbols)
        write_alpha_decay_state(snapshot, state_dir)
        return snapshot
    except ImportError:
        LOG.debug("symbol_score_v6 alpha decay not available")
        return {}
    except Exception as exc:
        LOG.warning("alpha_decay_computation_failed: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# v7.5_B1: Slippage and Liquidity State Publishing
# ---------------------------------------------------------------------------

def write_slippage_state(payload: Dict[str, Any], state_dir: pathlib.Path | None = None) -> None:
    """
    Write slippage.json with per-symbol slippage metrics (v7.5_B1).
    
    This file contains:
    - EWMA expected slippage per symbol
    - EWMA realized slippage per symbol
    - Trade counts
    """
    _write_state_file("slippage.json", payload, state_dir)


def write_liquidity_buckets_state(payload: Dict[str, Any], state_dir: pathlib.Path | None = None) -> None:
    """
    Write liquidity_buckets.json with symbol-bucket assignments (v7.5_B1).
    
    This file contains:
    - Per-symbol bucket assignment
    - Bucket configuration (max_spread_bps, default_maker_bias)
    """
    _write_state_file("liquidity_buckets.json", payload, state_dir)


def write_router_quality_state(payload: Dict[str, Any], state_dir: pathlib.Path | None = None) -> None:
    """
    Write router_quality.json with per-symbol router quality scores (v7.5_B2).
    
    This file contains:
    - Per-symbol router quality scores
    - Slippage drift metrics
    - Liquidity bucket assignments
    - TWAP skip ratios
    - Aggregate quality summary
    """
    _write_state_file("router_quality.json", payload, state_dir)


def compute_and_write_slippage_state(
    state_dir: pathlib.Path | None = None,
) -> Dict[str, Any]:
    """
    Compute slippage metrics and write to state file (v7.5_B1).
    
    Returns:
        The computed slippage snapshot
    """
    try:
        from execution.router_metrics import build_slippage_metrics_snapshot
        
        snapshot = build_slippage_metrics_snapshot()
        write_slippage_state(snapshot, state_dir)
        return snapshot
    except ImportError:
        LOG.debug("slippage_model not available for slippage state")
        return {}
    except Exception as exc:
        LOG.warning("slippage_state_computation_failed: %s", exc)
        return {}


def compute_and_write_liquidity_buckets_state(
    state_dir: pathlib.Path | None = None,
) -> Dict[str, Any]:
    """
    Compute liquidity bucket assignments and write to state file (v7.5_B1).
    
    Returns:
        The computed liquidity buckets snapshot
    """
    try:
        from execution.router_metrics import build_liquidity_buckets_snapshot
        
        snapshot = build_liquidity_buckets_snapshot()
        write_liquidity_buckets_state(snapshot, state_dir)
        return snapshot
    except ImportError:
        LOG.debug("liquidity_model not available for bucket state")
        return {}
    except Exception as exc:
        LOG.warning("liquidity_buckets_state_computation_failed: %s", exc)
        return {}


def compute_and_write_router_quality_state(
    state_dir: pathlib.Path | None = None,
) -> Dict[str, Any]:
    """
    Compute router quality scores and write to state file (v7.5_B2).
    
    Returns:
        The computed router quality snapshot
    """
    try:
        from execution.router_metrics import build_router_quality_state_snapshot
        
        snapshot = build_router_quality_state_snapshot()
        write_router_quality_state(snapshot, state_dir)
        return snapshot
    except ImportError:
        LOG.debug("router_metrics not available for router quality state")
        return {}
    except Exception as exc:
        LOG.warning("router_quality_state_computation_failed: %s", exc)
        return {}


def write_rv_momentum_state(payload: Dict[str, Any], state_dir: pathlib.Path | None = None) -> None:
    """
    Write rv_momentum.json with per-symbol relative momentum scores (v7.5_C1).
    
    This file contains:
    - Per-symbol RV momentum scores
    - Basket spread values (BTC vs ETH, L1 vs ALT, Meme vs Rest)
    - Basket membership for each symbol
    """
    _write_state_file("rv_momentum.json", payload, state_dir)


def compute_and_write_rv_momentum_state(
    state_dir: pathlib.Path | None = None,
) -> Dict[str, Any]:
    """
    Compute RV momentum scores and write to state file (v7.5_C1).
    
    Returns:
        The computed RV momentum snapshot as dict
    """
    try:
        from execution.rv_momentum import load_rv_config, build_rv_snapshot
        
        cfg = load_rv_config()
        if not cfg.enabled:
            return {}
        
        snapshot = build_rv_snapshot(cfg)
        write_rv_momentum_state(snapshot.to_dict(), state_dir)
        return snapshot.to_dict()
    except ImportError:
        LOG.debug("rv_momentum not available for RV momentum state")
        return {}
    except Exception as exc:
        LOG.warning("rv_momentum_state_computation_failed: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# v7.5_C2: Factor Diagnostics & PnL Attribution State
# ---------------------------------------------------------------------------


def write_factor_diagnostics_state(payload: Dict[str, Any], state_dir: pathlib.Path | None = None) -> None:
    """
    Write factor_diagnostics.json with per-symbol normalized factor vectors and covariance (v7.5_C2).
    
    This file contains:
    - Per-symbol normalized factor vectors
    - Factor covariance and correlation matrices
    - Factor volatilities
    - Factor weights (v7.5_C3)
    - Orthogonalization status (v7.5_C3)
    """
    _write_state_file("factor_diagnostics.json", payload, state_dir)


def load_factor_diagnostics_state(state_dir: pathlib.Path | None = None) -> Dict[str, Any]:
    """
    Load factor_diagnostics.json state (v7.5_C3).
    
    Returns:
        Loaded state dict or empty dict if not found
    """
    sd = state_dir or STATE_DIR
    path = sd / "factor_diagnostics.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def write_factor_pnl_state(payload: Dict[str, Any], state_dir: pathlib.Path | None = None) -> None:
    """
    Write factor_pnl.json with factor PnL attribution (v7.5_C2).
    
    This file contains:
    - PnL attributed to each factor
    - Percentage contribution per factor
    - Total PnL and trade count
    """
    _write_state_file("factor_pnl.json", payload, state_dir)


def load_factor_pnl_state(state_dir: pathlib.Path | None = None) -> Dict[str, Any]:
    """
    Load factor_pnl.json state (v7.5_C3).
    
    Returns:
        Loaded state dict or empty dict if not found
    """
    sd = state_dir or STATE_DIR
    path = sd / "factor_pnl.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def compute_and_write_factor_diagnostics_state(
    hybrid_results: List[Dict[str, Any]] | None = None,
    state_dir: pathlib.Path | None = None,
) -> Dict[str, Any]:
    """
    Compute factor diagnostics and write to state file (v7.5_C2/C3).
    
    Args:
        hybrid_results: List of hybrid_score() result dicts OR scores_snapshot dict
        state_dir: Optional state directory override
        
    Returns:
        The computed factor diagnostics snapshot as dict
    """
    try:
        from execution.factor_diagnostics import (
            load_factor_diagnostics_config,
            extract_factor_vectors_from_hybrid_results,
            build_factor_diagnostics_snapshot,
            FactorVector,
        )
        from execution.intel.symbol_score_v6 import build_factor_vector
        
        cfg = load_factor_diagnostics_config()
        if not cfg.enabled:
            return {}
        
        factor_vectors: List[Any] = []
        
        # Handle different input formats
        if hybrid_results is not None:
            if isinstance(hybrid_results, dict) and "symbols" in hybrid_results:
                # scores_snapshot format from build_symbol_scores
                symbols_list = hybrid_results.get("symbols", [])
                for entry in symbols_list:
                    if not isinstance(entry, dict):
                        continue
                    symbol = entry.get("symbol", "")
                    if not symbol:
                        continue
                    components = entry.get("components", {})
                    # Build factor vector from scores_snapshot components
                    factors = {
                        "trend": 0.5,  # Default - not in scores_snapshot
                        "carry": 0.0,  # Default - not in scores_snapshot
                        "expectancy": float(components.get("expectancy", 0.5)),
                        "router": float(components.get("router", 0.5)),
                        "rv_momentum": 0.0,
                        "router_quality": float(components.get("router", 0.5)),
                        "vol_regime": 1.0,
                    }
                    fv = build_factor_vector(
                        symbol=symbol,
                        components=factors,
                        hybrid_score=float(entry.get("score", 0.5)),
                        direction="LONG",
                        regime="normal",
                    )
                    factor_vectors.append(fv)
            elif isinstance(hybrid_results, list):
                # hybrid_score() results format
                factor_vectors = extract_factor_vectors_from_hybrid_results(hybrid_results)
        
        if not factor_vectors:
            # Try to load from symbol_scores state file as fallback
            scores_path = STATE_DIR / "symbol_scores_v6.json"
            if scores_path.exists():
                try:
                    with scores_path.open("r", encoding="utf-8") as f:
                        scores_data = json.load(f)
                    for entry in scores_data.get("symbols", []):
                        if not isinstance(entry, dict):
                            continue
                        symbol = entry.get("symbol", "")
                        if not symbol:
                            continue
                        components = entry.get("components", {})
                        factors = {
                            "trend": 0.5,
                            "carry": 0.0,
                            "expectancy": float(components.get("expectancy", 0.5)),
                            "router": float(components.get("router", 0.5)),
                            "rv_momentum": 0.0,
                            "router_quality": float(components.get("router", 0.5)),
                            "vol_regime": 1.0,
                        }
                        fv = build_factor_vector(
                            symbol=symbol,
                            components=factors,
                            hybrid_score=float(entry.get("score", 0.5)),
                            direction="LONG",
                            regime="normal",
                        )
                        factor_vectors.append(fv)
                except Exception:
                    pass
        
        if not factor_vectors:
            return {}
        
        # v7.5_C3: Load previous weights for EWMA smoothing
        prev_weights = None
        try:
            from execution.factor_diagnostics import FactorWeights
            prev_state = load_factor_diagnostics_state(state_dir)
            prev_weights_dict = prev_state.get("factor_weights", {})
            prev_weights_raw = {}
            if isinstance(prev_weights_dict, dict):
                prev_weights_raw = prev_weights_dict.get("weights", {})
            if not prev_weights_raw:
                prev_weights_raw = prev_state.get("weights", {})
            if isinstance(prev_weights_raw, dict) and prev_weights_raw:
                prev_weights = FactorWeights(weights=prev_weights_raw)
        except Exception:
            pass
        
        # v7.5_C3: Load factor PnL for weight computation
        factor_pnl = None
        try:
            pnl_state = load_factor_pnl_state(state_dir)
            if pnl_state and "pnl_by_factor" in pnl_state:
                factor_pnl = pnl_state.get("pnl_by_factor", {})
        except Exception:
            pass
        
        snapshot = build_factor_diagnostics_snapshot(
            factor_vectors=factor_vectors,
            cfg=cfg,
            factor_pnl=factor_pnl,
            prev_weights=prev_weights,
        )
        
        snapshot_dict = snapshot.to_dict()
        snapshot_dict.setdefault("updated_ts", utc_now_iso())
        write_factor_diagnostics_state(snapshot_dict, state_dir)
        return snapshot_dict
    except ImportError:
        LOG.debug("factor_diagnostics not available")
        return {}
    except Exception as exc:
        LOG.warning("factor_diagnostics_state_computation_failed: %s", exc)
        return {}


def compute_and_write_factor_pnl_state(
    state_dir: pathlib.Path | None = None,
) -> Dict[str, Any]:
    """
    Compute factor PnL attribution and write to state file (v7.5_C2).
    
    Returns:
        The computed factor PnL snapshot as dict
    """
    try:
        from execution.factor_pnl_attribution import (
            build_factor_pnl_from_logs,
        )
        from execution.factor_diagnostics import load_factor_diagnostics_config
        
        cfg = load_factor_diagnostics_config()
        if not cfg.enabled:
            return {}
        
        snapshot = build_factor_pnl_from_logs(
            lookback_days=cfg.pnl_attribution_lookback_days,
            factor_names=cfg.factors,
        )
        
        write_factor_pnl_state(snapshot.to_dict(), state_dir)
        return snapshot.to_dict()
    except ImportError:
        LOG.debug("factor_pnl_attribution not available")
        return {}
    except Exception as exc:
        LOG.warning("factor_pnl_state_computation_failed: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# v7.7_P4: Edge Insights Surface (Research-Only)
# ---------------------------------------------------------------------------


def compute_and_write_edge_insights(
    state_dir: pathlib.Path | None = None,
) -> Dict[str, Any]:
    """
    Compute edge insights snapshot and write to state file (v7.7_P4).
    
    This is a RESEARCH-ONLY surface that aggregates:
    - Factor edges (IR, PnL contribution, weights)
    - Symbol edges (hybrid scores, conviction, recent PnL)
    - Category edges (momentum, IR, aggregate PnL)
    - Regime context (vol regime, DD state, risk mode)
    
    This function does NOT influence execution or trading decisions.
    
    Returns:
        The computed edge insights snapshot as dict
    """
    try:
        from execution.edge_scanner import (
            build_edge_insights_snapshot,
            write_edge_insights,
            EdgeScannerConfig,
        )
        
        sd = state_dir or STATE_DIR
        
        # Build snapshot from existing state surfaces
        snapshot = build_edge_insights_snapshot()
        
        # Write to state file
        write_edge_insights(snapshot, sd / "edge_insights.json")
        
        return snapshot.to_dict()
    except ImportError:
        LOG.debug("edge_scanner not available")
        return {}
    except Exception as exc:
        LOG.warning("edge_insights_computation_failed: %s", exc)
        return {}


def write_regimes_state(
    payload: Dict[str, Any] | None = None,
    state_dir: pathlib.Path | None = None,
    *,
    atr_value: Optional[float] = None,
    dd_frac: Optional[float] = None,
    atr_percentiles: Optional[Dict[str, float]] = None,
) -> None:
    """
    Write regimes.json with ATR and drawdown regime classification.

    Can either accept a pre-built payload or compute from raw values.

    Args:
        payload: Optional pre-built regime payload
        state_dir: Optional state directory override
        atr_value: ATR percentage value (used if payload is None)
        dd_frac: Drawdown fraction 0-1 (used if payload is None)
        atr_percentiles: Optional ATR percentile thresholds
    """
    from execution.utils.vol import build_regime_snapshot

    if payload is not None:
        enriched = dict(payload)
    else:
        enriched = build_regime_snapshot(
            atr_value=atr_value,
            dd_frac=dd_frac,
            atr_percentiles=atr_percentiles,
        )

    _write_state_file("regimes.json", enriched, state_dir)


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
    version_value = engine_version or read_version(default="v7.6")
    snapshot = {
        "items": _coerce_synced_items(items or []),
        "nav": float(nav),
        "engine_version": version_value,
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
            "engine_version": str(payload.get("engine_version") or read_version(default="v7.6")),
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
    path: pathlib.Path | None = None,
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
    payload = payload or {}
    payload.setdefault("updated_at", utc_now_iso())
    target_path = path or (state_dir or STATE_DIR) / "kpis_v7.json"
    _atomic_write_state(target_path, payload)


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
    
    # V7.4_C3: Add positions_ledger block
    try:
        from execution.position_ledger import build_position_ledger, ledger_to_dict
        ledger = build_position_ledger()
        payload["positions_ledger"] = ledger_to_dict(ledger)
    except Exception as exc:
        LOG.debug("positions_ledger_build_failed: %s", exc)
    
    if not _firestore_enabled():
        _append_local_jsonl("positions", payload)
        try:
            write_positions_snapshot_state(payload)
        except Exception as exc:
            LOG.debug("write_positions_snapshot_state_failed: %s", exc)
        return
    cli = _fs_client()
    cli.document(f"{FS_ROOT}/positions").set(payload, merge=True)


def publish_hybrid_scores(
    scores: List[Dict[str, Any]],
    state_dir: pathlib.Path | None = None,
) -> None:
    """
    Publish hybrid score rankings for dashboard display (v7.4 B1).
    
    Args:
        scores: List of hybrid score results from rank_intents_by_hybrid_score()
    """
    payload = {"symbols": scores, "updated_ts": utc_now_iso()}
    try:
        _write_state_file("hybrid_scores.json", payload, state_dir)
    except Exception as exc:
        LOG.warning("publish_hybrid_scores failed: %s", exc)


def publish_funding_snapshot(
    funding_data: Dict[str, Any],
    state_dir: pathlib.Path | None = None,
) -> None:
    """
    Publish funding rate snapshot for carry scoring (v7.4 B1).
    
    Args:
        funding_data: Dict with 'symbols' key containing symbol->rate mappings
    """
    payload = {"symbols": funding_data.get("symbols", funding_data), "updated_ts": utc_now_iso()}
    try:
        _write_state_file("funding_snapshot.json", payload, state_dir)
    except Exception as exc:
        LOG.warning("publish_funding_snapshot failed: %s", exc)


def publish_basis_snapshot(
    basis_data: Dict[str, Any],
    state_dir: pathlib.Path | None = None,
) -> None:
    """
    Publish basis (spot-perp spread) snapshot for carry scoring (v7.4 B1).
    
    Args:
        basis_data: Dict with 'symbols' key containing symbol->basis_pct mappings
    """
    payload = {"symbols": basis_data.get("symbols", basis_data), "updated_ts": utc_now_iso()}
    try:
        _write_state_file("basis_snapshot.json", payload, state_dir)
    except Exception as exc:
        LOG.warning("publish_basis_snapshot failed: %s", exc)


def publish_vol_regime_snapshot(symbols_data: List[Dict[str, Any]]) -> None:
    """
    Publish volatility regime snapshot for dashboard display (v7.4 B2).
    
    Args:
        symbols_data: List of dicts with 'symbol', 'vol_regime', 'vol' (short/long/ratio)
    """
    # Build summary counts
    summary = {"low": 0, "normal": 0, "high": 0, "crisis": 0}
    for entry in symbols_data:
        label = entry.get("vol_regime", "normal")
        if label in summary:
            summary[label] += 1
    
    payload = {
        "symbols": symbols_data,
        "vol_regime_summary": summary,
        "updated_ts": time.time(),
    }
    path = STATE_DIR / "vol_regimes.json"
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str))
    except Exception as exc:
        LOG.warning("publish_vol_regime_snapshot failed: %s", exc)


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
