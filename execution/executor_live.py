#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import sys
import os

repo_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, repo_root)

import json
import logging
import subprocess
import time
import socket
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, cast

try:
    from binance.um_futures import UMFutures
except Exception:  # pragma: no cover - optional dependency
    UMFutures = None

from execution.log_utils import get_logger, log_event, safe_dump

import requests
from utils.firestore_client import with_firestore
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
_HEARTBEAT_INTERVAL = 60.0
_LAST_HEARTBEAT = 0.0
_LAST_SIGNAL_PULL = 0.0
_LAST_QUEUE_DEPTH = 0

# ---- Exchange utils (binance) ----
from execution.exchange_utils import (
    _is_dual_side,
    build_order_payload,
    get_balances,
    get_positions,
    get_price,
    is_testnet,
    send_order,
    set_dry_run,
)
try:
    from execution.order_router import route_order as _route_order
except Exception:
    _route_order = None  # type: ignore[assignment]
from execution.risk_limits import RiskState, check_order, RiskGate
from execution.nav import compute_nav_pair, compute_treasury_only, PortfolioSnapshot
from execution.utils import (
    load_json,
    write_nav_snapshots_pair,
    write_treasury_snapshot,
    save_json,
    get_live_positions,
)

from execution.signal_generator import generate_intents, normalize_intent as generator_normalize_intent
try:
    from execution.signal_screener import run_once as run_screener_once
except ImportError:  # pragma: no cover - optional dependency
    run_screener_once = None

# ---- Firestore publisher handle (revisions differ) ----
ENV = os.getenv("ENV", "prod")
_FS_PUB_INTERVAL = int(os.getenv("FS_PUB_INTERVAL", "60") or 60)
_PUB: Any
try:
    from execution.state_publish import (
        StatePublisher,
        publish_close_audit,
        publish_intent_audit,
        publish_order_audit,
        publish_nav_value,
    )

    try:
        _PUB = StatePublisher(env=ENV, interval_s=_FS_PUB_INTERVAL)  # type: ignore[arg-type]
    except TypeError:
        _PUB = StatePublisher(interval_s=_FS_PUB_INTERVAL)
        setattr(_PUB, "env", ENV)
except Exception:

    class _Publisher:
        def __init__(self, interval_s: int = 60, env: str = ENV):
            self.interval_s = interval_s
            self.env = env

    def publish_intent_audit(intent: Dict[Any, Any]) -> None:
        pass

    def publish_order_audit(symbol: str, event: Dict[Any, Any]) -> None:
        pass

    def publish_close_audit(
        symbol: str, position_side: str, event: Dict[Any, Any]
    ) -> None:
        pass

    def publish_nav_value(nav: float, min_interval_s: int = 60, max_points: int = 20000) -> None:
        return None

    _PUB = _Publisher(interval_s=_FS_PUB_INTERVAL)

# ---- risk limits config ----
_RISK_CFG_PATH = os.getenv("RISK_LIMITS_CONFIG", "config/risk_limits.json")
_RISK_CFG: Dict[str, Any] = {}
try:
    with open(_RISK_CFG_PATH, "r") as fh:
        _RISK_CFG = json.load(fh) or {}
except Exception as e:
    logging.getLogger("exutil").warning(
        "[risk] config load failed (%s): %s", _RISK_CFG_PATH, e
    )

_RISK_STATE = RiskState()

# Build a minimal RiskGate config adapter from legacy risk_limits.json
def _mk_gate_cfg(raw: Dict[str, Any]) -> Dict[str, Any]:
    g = (raw.get("global") or {}) if isinstance(raw, dict) else {}
    # Legacy configs kept caps at top-level; prefer `global` section when present
    if not g and isinstance(raw, dict):
        for key in ("daily_loss_limit_pct", "max_gross_exposure_pct"):
            if key in raw:
                g[key] = raw[key]
    sizing = {
        "min_notional_usdt": g.get("min_notional_usdt", 5.0),
        # Accept either key for portfolio gross NAV cap
        "max_gross_exposure_pct": (
            g.get("max_gross_exposure_pct")
            or g.get("max_portfolio_gross_nav_pct")
            or g.get("max_gross_nav_pct")
            or 0.0
        ),
        # Fallback sensible default if not provided
        "max_symbol_exposure_pct": g.get("max_symbol_exposure_pct", 50.0),
        "max_trade_nav_pct": g.get("max_trade_nav_pct", 0.0),
    }
    risk = {
        "daily_loss_limit_pct": g.get("daily_loss_limit_pct", 5.0),
        # Derive cooldown (minutes) from error circuit cooldown (seconds) if present
        "cooldown_minutes_after_stop": (
            g.get("cooldown_minutes_after_stop")
            or max(
                0.0,
                float(((g.get("error_circuit") or {}).get("cooldown_sec") or 0))
                / 60.0,
            )
        ),
        "max_trades_per_symbol_per_hour": g.get("max_trades_per_symbol_per_hour", 6),
    }
    return {"sizing": sizing, "risk": risk}

_RISK_GATE = RiskGate(_mk_gate_cfg(_RISK_CFG))
_PORTFOLIO_SNAPSHOT = PortfolioSnapshot(load_json("config/strategy_config.json"))
_RISK_GATE.nav_provider = _PORTFOLIO_SNAPSHOT

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
    env = os.getenv("ENV", "prod")
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
    if not flags.get("fs_enabled"):
        return
    try:
        from execution.firestore_utils import publish_health  # local import to avoid circulars

        payload = {
            "process": "executor_live",
            "status": "ok",
            "ts": _now_iso(),
            "env": flags.get("env"),
        }
        publish_health(payload)
    except Exception as exc:
        LOG.exception("[%s] Firestore heartbeat failed: %s", flags.get("prefix", "live"), exc)
    else:
        LOG.info(
            "[firestore] heartbeat write ok path=hedge/%s/health/executor_live",
            flags.get("env"),
        )


DRY_RUN = _read_dry_run_flag()
set_dry_run(DRY_RUN)
INTENT_TEST = _truthy_env("INTENT_TEST", "0")
EXTERNAL_SIGNAL = _truthy_env("EXTERNAL_SIGNAL", "0")

LOG.info(
    "[executor] starting loop ENV=%s DRY_RUN=%s commit=%s signal_source=generate_intents unified=True",
    os.getenv("ENV", "dev"),
    DRY_RUN,
    _git_commit(),
)

try:
    _startup_flags_snapshot = _log_startup_summary()
except Exception as exc:
    LOG.exception("[live] startup summary logging failed: %s", exc)
    _startup_flags_snapshot = {
        "env": os.getenv("ENV", "prod"),
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


def _publish_intent_audit(symbol: Optional[str], intent: Dict[str, Any]) -> None:
    LOG.info("[screener->executor] %s", intent)
    if not symbol:
        return
    payload = dict(intent)
    payload.setdefault("symbol", symbol)
    payload.setdefault("ts", time.time())
    try:
        publish_intent_audit(payload)
    except Exception:
        pass


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
    try:
        publish_order_audit(
            (symbol or "UNKNOWN").upper(),
            {
                "phase": "veto",
                "reasons": reasons_list,
                "intent": dict(intent),
            },
        )
    except Exception:
        pass


# ------------- helpers -------------
def _record_structured_event(logger_obj: Any, event_type: str, payload: Mapping[str, Any] | None) -> None:
    try:
        sanitized = safe_dump(payload or {})
        log_event(logger_obj, event_type, sanitized)
    except Exception as exc:
        LOG.debug("structured_log_failed event=%s err=%s", event_type, exc)


def _nav_snapshot() -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {}
    try:
        snapshot["nav_usd"] = float(_PORTFOLIO_SNAPSHOT.current_nav_usd())
    except Exception:
        pass
    try:
        snapshot["gross_usd"] = float(_PORTFOLIO_SNAPSHOT.current_gross_usd())
    except Exception:
        pass
    try:
        gross_map = _PORTFOLIO_SNAPSHOT.symbol_gross_usd()
        if gross_map:
            snapshot["symbol_gross_usd"] = gross_map
    except Exception:
        pass
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


def _startup_position_check(client: Any) -> None:
    if client is None:
        LOG.info("[startup-sync] unable to check positions (client unavailable)")
        return
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
        try:
            payload = entry.get("raw") if isinstance(entry, Mapping) else entry
            intent = _normalize_intent(payload)
            symbol = cast(Optional[str], intent.get("symbol"))
            if not symbol:
                continue
            _publish_intent_audit(symbol, intent)
            _send_order(intent)
            submitted += 1
        except Exception as exc:
            LOG.error("[executor] internal screener submit failed: %s", exc)

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
        trading, reporting = compute_nav_pair(cfg)
        write_nav_snapshots_pair(trading, reporting)
        tre_val, tre_detail = compute_treasury_only()
        write_treasury_snapshot(tre_val, tre_detail)
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
    try:
        loss_pct = float(_RISK_GATE._daily_loss_pct())
    except Exception:
        loss_pct = 0.0
    _RISK_STATE.daily_pnl_pct = -loss_pct


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


def _send_order(intent: Dict[str, Any], *, skip_flip: bool = False) -> None:
    symbol = intent["symbol"]
    sig = str(intent.get("signal", "")).upper()
    side = "BUY" if sig == "BUY" else "SELL"
    pos_side = intent.get("positionSide") or ("LONG" if side == "BUY" else "SHORT")
    cap = float(intent.get("capital_per_trade", 0) or 0)
    lev = float(intent.get("leverage", 1) or 1)
    gross_target = float(intent.get("gross_usd") or (cap * lev))
    if lev <= 0:
        lev = 1.0
    price_guess = 0.0
    try:
        price_guess = float(intent.get("price", 0.0) or 0.0)
    except Exception:
        price_guess = 0.0
    reduce_only = bool(intent.get("reduceOnly", False))
    try:
        _PORTFOLIO_SNAPSHOT.refresh()
    except Exception:
        pass
    nav_snapshot = _nav_snapshot()
    nav_usd = float(nav_snapshot.get("nav_usd", 0.0) or 0.0)
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
        thresholds = {}
        if extra:
            maybe_thresholds = extra.get("thresholds")
            if isinstance(maybe_thresholds, Mapping):
                thresholds = dict(maybe_thresholds)
        log_payload = {
            "symbol": symbol,
            "side": side,
            "position_side": pos_side,
            "run_id": RUN_ID,
            "hostname": HOSTNAME,
            "veto_reason": reason,
            "veto_detail": extra or {},
            "thresholds": thresholds,
            "ts": payload.get("ts"),
        }
        _record_structured_event(LOG_VETOES, "risk_veto", log_payload)
        return payload

    floor_gross = 0.0
    try:
        cfg = load_json("config/strategy_config.json")
        sizing_cfg = (cfg.get("sizing") or {})
        floor_gross = float((sizing_cfg.get("min_gross_usd_per_order", 0.0)) or 0.0)
        per_symbol_cfg = sizing_cfg.get("per_symbol_min_gross_usd") or {}
        sym_floor = per_symbol_cfg.get(symbol.upper())
        if sym_floor is not None:
            floor_gross = max(floor_gross, float(sym_floor or 0.0))
        gross_target = max(gross_target, floor_gross)
        if not reduce_only:
            symbol_buckets: Dict[str, str] = {}
            for strat in cfg.get("strategies") or []:
                if not isinstance(strat, Mapping):
                    continue
                sym = str(strat.get("symbol") or "").upper()
                bucket = strat.get("bucket")
                if sym and bucket:
                    symbol_buckets[sym] = str(bucket)
            extra_map = sizing_cfg.get("symbol_buckets") or {}
            if isinstance(extra_map, Mapping):
                for key, value in extra_map.items():
                    if not value:
                        continue
                    symbol_buckets[str(key).upper()] = str(value)
            size_risk_cfg = {
                "min_notional_usd": floor_gross,
                "fallback_gross_usd": gross_target,
                "per_symbol_leverage": sizing_cfg.get("per_symbol_leverage") or {},
                "default_leverage": sizing_cfg.get("default_leverage", 1.0),
                "max_symbol_exposure_pct": sizing_cfg.get("max_symbol_exposure_pct"),
                "max_gross_exposure_pct": sizing_cfg.get("max_gross_exposure_pct"),
                "max_trade_nav_pct": sizing_cfg.get("max_trade_nav_pct"),
                "vol_target_bps": sizing_cfg.get("vol_target_bps"),
                "atr_interval": sizing_cfg.get("atr_interval"),
                "atr_lookback": sizing_cfg.get("atr_lookback"),
                "bucket_caps_pct": sizing_cfg.get("bucket_caps_pct"),
                "symbol_bucket": symbol_buckets,
                "current_symbol_gross": symbol_gross_map,
                "current_bucket_gross": _current_bucket_gross(symbol_gross_map, symbol_buckets),
                "current_portfolio_gross": nav_snapshot.get("gross_usd"),
                "price": price_guess,
            }
            signal_strength = float(
                intent.get("signal_strength")
                or intent.get("confidence")
                or intent.get("score")
                or 1.0
            )
            sizing_suggestion = size_model.suggest_gross_usd(
                symbol,
                nav_usd,
                signal_strength,
                size_risk_cfg,
            )
            sized_gross = float(sizing_suggestion.get("gross_usd", gross_target))
            if sized_gross <= 0.0:
                LOG.info(
                    "[sizer] sym=%s atr=%.4f blocked (reason=%s)",
                    symbol,
                    sizing_suggestion.get("atr", 0.0),
                    sizing_suggestion.get("reason", "sizer_cap"),
                )
                _persist_veto(
                    sizing_suggestion.get("reason", "sizer_cap"),
                    price_guess,
                    {
                        "intent": intent,
                        "nav_snapshot": nav_snapshot,
                        "thresholds": {
                            "bucket": sizing_suggestion.get("bucket"),
                            "bucket_cap": sizing_suggestion.get("bucket_cap"),
                        },
                    },
                )
                return
            gross_target = max(sized_gross, floor_gross)
            lev_cap = float(sizing_suggestion.get("leverage_cap") or lev)
            if lev_cap > 0.0 and lev > lev_cap:
                lev = lev_cap
            LOG.info(
                "[sizer] sym=%s atr=%.4f gross_usd=%.2f bucket_used=%.2f/%.2f",
                symbol,
                sizing_suggestion.get("atr", 0.0),
                gross_target,
                sizing_suggestion.get("bucket_used") or 0.0,
                sizing_suggestion.get("bucket_cap") or 0.0,
            )
    except Exception:
        gross_target = max(gross_target, floor_gross)
    margin_target = gross_target / max(lev, 1.0)
    attempt_start_monotonic = time.monotonic()
    attempt_payload = {
        "symbol": symbol,
        "side": side,
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
        "[executor] INTENT symbol=%s side=%s ps=%s margin=%.4f gross=%.4f lev=%.2f reduceOnly=%s",
        symbol,
        side,
        pos_side,
        margin_target,
        gross_target,
        lev,
        reduce_only,
    )

    if not reduce_only:
        # Shared gross-notional gate (canonical taxonomy)
        try:
            allowed, veto = _RISK_GATE.allowed_gross_notional(symbol, gross_target)
        except Exception:
            allowed, veto = True, ""
        if not allowed:
            price_hint = float(intent.get("price", 0.0) or 0.0)
            _persist_veto(
                veto or "blocked",
                price_hint,
                {
                    "intent": intent,
                    "nav_snapshot": nav_snapshot,
                    "thresholds": {
                        "max_gross_exposure_pct": _RISK_GATE.sizing.get("max_gross_exposure_pct"),
                        "max_trade_nav_pct": _RISK_GATE.sizing.get("max_trade_nav_pct"),
                    },
                },
            )
            return

    try:
        positions = list(get_positions() or [])
    except Exception:
        positions = []

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
                reduce_resp = send_order(**reduce_payload)
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
                try:
                    exchange_name = "binance_testnet" if is_testnet() else "binance"
                except Exception:
                    exchange_name = "binance"
                reduce_price = (
                    reduce_resp.get("avgPrice")
                    or reduce_resp.get("price")
                    or reduce_payload.get("price")
                )
                reduce_qty = (
                    reduce_resp.get("executedQty")
                    or reduce_resp.get("origQty")
                    or reduce_payload.get("quantity")
                )
                reduce_latency_ms = max(
                    0.0, (time.monotonic() - attempt_start_monotonic) * 1000.0
                )
                reduce_log_payload = {
                    "symbol": symbol,
                    "side": reduce_signal,
                    "client_order_id": reduce_resp.get("clientOrderId")
                    or reduce_resp.get("orderId"),
                    "exchange": exchange_name,
                    "price": reduce_price,
                    "qty": reduce_qty,
                    "order_type": reduce_payload.get("type"),
                    "reduce_only": True,
                    "latency_ms": reduce_latency_ms,
                    "status": reduce_resp.get("status"),
                    "context": "flip_reduce",
                    "run_id": RUN_ID,
                    "hostname": HOSTNAME,
                }
                _record_structured_event(LOG_ORDERS, "order_executed", reduce_log_payload)

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
    _update_risk_state_counters(positions, current_gross)
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

    if reduce_only:
        ok = True
        details: Dict[str, Any] = {"reasons": []}
    else:
        ok, details = check_order(
            symbol=symbol,
            side=side,
            requested_notional=gross_target,
            price=0.0,
            nav=nav,
            open_qty=sym_open_qty,
            now=time.time(),
            cfg=_RISK_CFG,
            state=_RISK_STATE,
            current_gross_notional=current_gross,
            lev=lev,
        )
    reasons = details.get("reasons", []) if isinstance(details, dict) else []
    if not ok:
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

    def _meta_float(val: Any, fallback: float) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return fallback

    if not reduce_only:
        # Ensure the opening order never carries the reduceOnly flag
        payload.pop("reduceOnly", None)

    norm_price = _meta_float(meta.get("normalized_price"), price_hint)
    norm_qty = _meta_float(meta.get("normalized_qty"), _meta_float(payload.get("quantity"), 0.0))
    payload_view = {
        k: payload[k]
        for k in ("type", "quantity", "reduceOnly", "positionSide")
        if k in payload
    }

    if not reduce_only:
        try:
            allowed_now, veto_now = _RISK_GATE.allowed_gross_notional(symbol, gross_target)
        except Exception:
            allowed_now, veto_now = True, ""
        if not allowed_now:
            LOG.warning(
                "[risk] final gate veto %s %s reason=%s", symbol, side, veto_now or "blocked"
            )
            _persist_veto(
                veto_now or "blocked",
                norm_price,
                {
                    "intent": intent,
                    "normalized_qty": norm_qty,
                    "payload": payload_view,
                    "meta": meta,
                    "nav_snapshot": nav_snapshot,
                    "thresholds": {
                        "max_gross_exposure_pct": _RISK_GATE.sizing.get("max_gross_exposure_pct"),
                        "max_trade_nav_pct": _RISK_GATE.sizing.get("max_trade_nav_pct"),
                    },
                },
            )
            try:
                publish_order_audit(
                    symbol,
                    {
                        "phase": "blocked",
                        "side": side,
                        "positionSide": pos_side,
                        "reason": veto_now or "blocked",
                        "notional": gross_target,
                    },
                )
            except Exception:
                pass
            return

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

    if force_direct_send:
        try:
            resp = send_order(**payload)
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

    if _route_order is not None and not force_direct_send:
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
        try:
            result = _route_order(router_intent, router_ctx, DRY_RUN)
        except Exception as exc:
            router_error = str(exc)
        else:
            if result.get("accepted"):
                resp = result.get("raw") or {}
                if result.get("price") is not None:
                    resp.setdefault("avgPrice", result.get("price"))
                if result.get("qty") is not None:
                    resp.setdefault("executedQty", str(result.get("qty")))
            else:
                router_error = str(result.get("reason") or "router_reject")

    if not resp:
        if router_error and not force_direct_send:
            LOG.error(
                "[executor] router_reject %s %s reason=%s", symbol, side, router_error
            )
        try:
            resp = send_order(**payload)
        except requests.HTTPError as exc:
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
            LOG.error(
                "[executor] ORDER_ERR code=%s symbol=%s side=%s meta=%s payload=%s err=%s",
                err_code,
                symbol,
                side,
                meta,
                payload_view,
                exc,
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
            raise
        except Exception as exc:
            try:
                _RISK_STATE.note_error(time.time())
            except Exception:
                pass
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

    try:
        exchange_name = "binance_testnet" if is_testnet() else "binance"
    except Exception:
        exchange_name = "binance"
    price_exec = (
        resp.get("avgPrice")
        or resp.get("price")
        or payload.get("price")
        or norm_price
    )
    qty_exec = (
        resp.get("executedQty")
        or resp.get("origQty")
        or payload.get("quantity")
        or norm_qty
    )
    latency_ms = max(0.0, (time.monotonic() - attempt_start_monotonic) * 1000.0)
    execution_payload = {
        "symbol": symbol,
        "side": side,
        "client_order_id": resp.get("clientOrderId") or resp.get("orderId"),
        "exchange": exchange_name,
        "price": price_exec,
        "qty": qty_exec,
        "order_type": resp.get("type") or payload.get("type"),
        "reduce_only": reduce_only,
        "latency_ms": latency_ms,
        "status": resp.get("status"),
        "run_id": RUN_ID,
        "hostname": HOSTNAME,
    }
    _record_structured_event(LOG_ORDERS, "order_executed", execution_payload)

    oid = resp.get("orderId")
    avg = resp.get("avgPrice", "0.00")
    qty = resp.get("executedQty", resp.get("origQty", "0"))
    st = resp.get("status")
    LOG.info("[executor] ORDER_REQ 200 id=%s avgPrice=%s qty=%s", oid, avg, qty)
    try:
        publish_order_audit(
            symbol,
            {
                "phase": "response",
                "side": side,
                "positionSide": pos_side,
                "status": st,
                "orderId": oid,
                "avgPrice": avg,
                "qty": qty,
                "normalized": normalized_ctx,
                "payload": payload_view,
            },
        )
    except Exception:
        pass

    if st == "FILLED":
        LOG.info(
            "[executor] ORDER_FILL id=%s status=%s avgPrice=%s qty=%s",
            oid,
            st,
            avg,
            qty,
        )
        try:
            _RISK_STATE.note_fill(symbol, time.time())
        except Exception:
            pass
        _emit_position_snapshots(symbol)
        if (
            reduce_only
            or (side == "SELL" and pos_side == "LONG")
            or (side == "BUY" and pos_side == "SHORT")
        ):
            try:
                publish_close_audit(
                    symbol,
                    pos_side,
                    {"orderId": oid, "avgPrice": avg, "qty": qty, "status": st},
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
        raw: Iterable[Dict[str, Any]] = get_positions()
    except Exception as exc:
        LOG.error("[executor] get_positions error: %s", exc)
        return []
    rows: List[Dict[str, Any]] = []
    for payload in raw:
        try:
            rows.append(
                {
                    "symbol": payload.get("symbol"),
                    "positionSide": payload.get("positionSide", "BOTH"),
                    "qty": float(payload.get("qty", payload.get("positionAmt", 0)) or 0.0),
                    "entryPrice": float(payload.get("entryPrice") or 0.0),
                    "unrealized": float(
                        payload.get("unRealizedProfit", payload.get("unrealized", 0)) or 0.0
                    ),
                    "leverage": float(payload.get("leverage") or 0.0),
                }
            )
        except Exception:
            continue
    return rows



def _pub_tick() -> None:
    """
    Try StatePublisher.{tick,run_once,step,publish,update}; if no such method,
    publish a minimal NAV+positions snapshot inline so the dashboard stays live.
    """
    # Try common method names on the object first
    for meth in ("tick", "run_once", "step", "publish", "update"):
        m = getattr(_PUB, meth, None)
        if callable(m):
            try:
                m()
                return
            except Exception as e:
                LOG.error("[executor] publisher.%s error: %s", meth, e)

    # Inline minimal publish (NAV + positions) — Firestore-first dashboard compatible
    try:
        nav_val = _compute_nav_snapshot()
        rows = _collect_rows()

        if hasattr(_PUB, "maybe_publish_positions"):
            try:
                _PUB.maybe_publish_positions(rows)
            except Exception as exc:
                LOG.error("[executor] StatePublisher.maybe_publish_positions error: %s", exc)

        if nav_val is not None:
            try:
                publish_nav_value(float(nav_val))
            except Exception as exc:
                LOG.error("[executor] publish_nav_value error: %s", exc)

        if os.environ.get("FIRESTORE_ENABLED", "1") == "0":
            return

        try:
            with with_firestore() as db:
                db.document(f"hedge/{ENV}/state/positions").set({"rows": rows}, merge=False)
            print("[executor] Firestore publish ok", flush=True)
        except Exception as exc:
            LOG.error("[executor] Firestore publish error: %s", exc)
    except Exception as exc:
        LOG.error("[executor] publisher fallback error: %s", exc)


def _loop_once(i: int) -> None:
    global _LAST_SIGNAL_PULL, _LAST_QUEUE_DEPTH
    _sync_dry_run()
    _account_snapshot()

    try:
        baseline_positions = list(get_positions() or [])
    except Exception:
        baseline_positions = []
    gross_total, _ = _gross_and_open_qty("", "", baseline_positions)
    _update_risk_state_counters(baseline_positions, gross_total)

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
        attempted = len(intents_raw)
        emitted = 0
        for raw_intent in intents_raw:
            intent = _normalize_intent(raw_intent)
            symbol = cast(Optional[str], intent.get("symbol"))
            if not symbol:
                LOG.warning("[screener] missing symbol in intent %s", intent)
                continue

            veto_reasons = _coerce_veto_reasons(intent.get("veto"))
            if veto_reasons:
                _publish_veto_exec(symbol, veto_reasons, intent)
                continue

            emitted += 1
            try:
                _publish_intent_audit(symbol, intent)
                _send_order(intent)
            except Exception as exc:
                LOG.error("[executor] failed to send intent %s %s", symbol, exc)
        LOG.info("[screener] attempted=%d emitted=%d", attempted, emitted)

    _pub_tick()


def main() -> None:
    _sync_dry_run()
    LOG.debug("[exutil] ENV context testnet=%s dry_run=%s", is_testnet(), DRY_RUN)
    try:
        if not _is_dual_side():
            LOG.warning("[executor] WARNING — account not in hedge (dualSide) mode")
    except Exception as e:
        LOG.error("[executor] dualSide check failed: %s", e)

    client = None
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    if UMFutures is not None and api_key and api_secret:
        try:
            client = UMFutures(key=api_key, secret=api_secret)
        except Exception as exc:
            LOG.error("[startup-sync] failed to initialise UMFutures client: %s", exc)
    _startup_position_check(client)

    i = 0
    while True:
        _loop_once(i)
        _maybe_emit_heartbeat()
        _maybe_run_internal_screener()
        i += 1
        if MAX_LOOPS and i >= MAX_LOOPS:
            LOG.info("[executor] MAX_LOOPS reached — exiting.")
            break
        time.sleep(SLEEP)


if __name__ == "__main__":
    main()
