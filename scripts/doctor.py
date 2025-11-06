#!/usr/bin/env python3
import os
import sys
import json
import time
import hmac
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from datetime import datetime, timezone

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from execution.firestore_utils import get_db as get_firestore_db
from dashboard.router_health import _load_order_events

ANSI_GREEN = "\033[92m"
ANSI_RED = "\033[91m"

# --- Load screener pieces
import execution.signal_screener as sc  # noqa: E402
from execution.nav import get_nav_age, is_nav_fresh  # noqa: E402
from execution.log_utils import AggWindow  # noqa: E402

try:
    from execution.utils import load_json as utils_load_json
except Exception:  # pragma: no cover
    utils_load_json = None
try:
    from execution.utils import get_live_positions as utils_get_live_positions
except Exception:  # pragma: no cover
    utils_get_live_positions = None
try:
    from execution.utils import get_coingecko_prices as utils_get_coingecko_prices
except Exception:  # pragma: no cover
    utils_get_coingecko_prices = None
try:
    from execution.utils import get_usd_to_zar as utils_get_usd_to_zar
except Exception:  # pragma: no cover
    utils_get_usd_to_zar = None

def _now_ms(): return int(time.time()*1000)

def _dual_side():
    if os.getenv("USE_FUTURES","") not in ("1","true","True"):
        return False, "spot-mode"
    key = os.environ.get("BINANCE_API_KEY","")
    sec = os.environ.get("BINANCE_API_SECRET","").encode()
    base = "https://testnet.binancefuture.com" if os.getenv("BINANCE_TESTNET","") in ("1","true","True") else "https://fapi.binance.com"
    q = f"timestamp={_now_ms()}"
    sig = hmac.new(sec, q.encode(), hashlib.sha256).hexdigest()
    r = requests.get(f"{base}/fapi/v1/positionSide/dual?{q}&signature={sig}", headers={"X-MBX-APIKEY":key}, timeout=10)
    j = r.json()
    return bool(j.get("dualSidePosition", False)), j

def _price(sym:str):
    try:
        from execution.exchange_utils import get_price
        p = get_price(sym)
        return None if (p is None or (isinstance(p,(int,float)) and p<=0)) else float(p)
    except Exception:
        return None

def _series(st:dict, sym:str, tf:str):
    arr = sc._series_for(st, sym, tf)
    return [p for _,p in arr]

def _meta(st:dict, sym:str, tf:str):
    return ((st.get(sym, {}) or {}).get(f"{tf}__meta", {}) or {})

def _parse_iso(ts: Any) -> Optional[float]:
    if not ts:
        return None
    if isinstance(ts, (int, float)):
        val = float(ts)
        return val / 1000.0 if val > 1e12 else val
    if isinstance(ts, str):
        txt = ts.strip()
        if not txt:
            return None
        try:
            # Handle Z suffix
            if txt.endswith("Z"):
                txt = txt[:-1] + "+00:00"
            dt = datetime.fromisoformat(txt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception:
            return None
    return None

def _mins_ago(ts: Optional[float]) -> Optional[int]:
    if ts is None:
        return None
    try:
        delta = max(0.0, time.time() - float(ts))
        return int(delta // 60)
    except Exception:
        return None

def _human_minutes(ts: Optional[float]) -> str:
    mins = _mins_ago(ts)
    if mins is None:
        return "n/a"
    if mins < 60:
        return f"{mins}m"
    hours = mins // 60
    return f"{hours}h"

def _colorize(text: str, color_code: str, enable: bool) -> str:
    if not enable:
        return text
    return f"{color_code}{text}\033[0m"


def _to_dt(value: Any) -> Optional[datetime]:
    """Normalize Firestore values (iso str, epoch, timestamp) into UTC datetimes."""
    if value is None:
        return None
    try:
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        if hasattr(value, "timestamp"):
            ts = value.timestamp() if callable(value.timestamp) else value.timestamp
            return datetime.fromtimestamp(float(ts), tz=timezone.utc)
        if hasattr(value, "seconds"):
            seconds = float(getattr(value, "seconds", 0))
            nanos = float(getattr(value, "nanos", 0))
            return datetime.fromtimestamp(seconds + nanos / 1_000_000_000, tz=timezone.utc)
        text = str(value).strip()
        if not text:
            return None
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _pick(doc: Mapping[str, Any], *paths: str) -> Any:
    """Return the first non-None value from dotted paths inside mapping."""
    for path in paths:
        current: Any = doc
        for segment in path.split("."):
            if not isinstance(current, Mapping) or segment not in current:
                current = None
                break
            current = current[segment]
        if current is not None:
            return current
    return None
SERVICE_ALIASES = {
    "executor_live": ("executor_live", "executor", "executorlive"),
    "sync_state": ("sync_state", "sync", "sync_daemon"),
}


def _extract_service_timestamp(doc: Mapping[str, Any], aliases: Sequence[str]) -> Optional[datetime]:
    if not isinstance(doc, Mapping):
        return None
    for alias in aliases:
        alias_norm = str(alias or "").strip()
        if not alias_norm:
            continue
        alias_key = alias_norm.replace("-", "_")
        candidates = [
            f"heartbeat.{alias_key}",
            f"heartbeats.{alias_key}",
            f"hb.{alias_key}",
            alias_key,
            f"{alias_key}_ts",
            f"{alias_key}.ts",
            f"{alias_key}.ts_iso",
            f"{alias_key}.timestamp",
            f"{alias_key}.updated_at",
        ]
        for path in candidates:
            target = _pick(doc, path) if "." in path else doc.get(path)
            if isinstance(target, Mapping):
                ts_nested = _extract_service_timestamp(target, aliases)
                if ts_nested:
                    return ts_nested
            else:
                ts_val = _to_dt(target)
                if ts_val:
                    return ts_val
    for fallback in ("ts_iso", "ts", "timestamp", "updated_at", "time", "last_seen"):
        target = doc.get(fallback)
        ts_val = _to_dt(target)
        if ts_val:
            return ts_val
    return None


def parse_heartbeats(doc: Mapping[str, Any]) -> Dict[str, Optional[datetime]]:
    """Normalize heartbeat schemas into canonical datetimes."""
    result = {"executor_live": None, "sync_state": None}
    nested = _pick(doc, "heartbeat", "heartbeats", "hb")
    if isinstance(nested, Mapping):
        for service, aliases in SERVICE_ALIASES.items():
            ts_val = _extract_service_timestamp(nested, aliases)
            if ts_val:
                result[service] = ts_val
    for service, aliases in SERVICE_ALIASES.items():
        if result.get(service):
            continue
        ts_val = _extract_service_timestamp(doc, aliases)
        if ts_val:
            result[service] = ts_val
    return result


def age_minutes(dt: Optional[datetime]) -> Optional[float]:
    if dt is None:
        return None
    try:
        delta = datetime.now(timezone.utc) - dt
        return max(0.0, delta.total_seconds() / 60.0)
    except Exception:
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default

def safe_load(path: str, default=None):
    default = {} if default is None else default
    if not path:
        print("[doctor] safe_load missing path, returning default")
        return default

    if utils_load_json is not None:
        try:
            data = utils_load_json(path)
            return data if data is not None else default
        except Exception as exc:
            print(f"[doctor] load_json failed for {path}: {exc}")

    loader = getattr(sc, "_safe_load_json", None) or getattr(sc, "safe_load_json", None)
    if callable(loader):
        try:
            data = loader(path, default)
            return data if data is not None else default
        except Exception as exc:
            print(f"[doctor] screener safe load failed for {path}: {exc}")

    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        print(f"[doctor] fallback load failed for {path}: {exc}")

    print(f"[doctor] returning default for {path}")
    return default

def _collect_nav() -> Dict[str, Any]:
    nav_value = None
    nav_source = "n/a"
    paths = [
        Path("logs/nav_log.json"),
        Path("logs/nav_snapshot.json"),
        Path("logs/nav_trading.json"),
    ]
    for path in paths:
        try:
            if not path.exists():
                continue
            data = safe_load(str(path), default=[])
            if isinstance(data, dict):
                candidates = [data]
            elif isinstance(data, list):
                candidates = data[-5:]
            else:
                continue
            for item in reversed(candidates):
                if not isinstance(item, dict):
                    continue
                nav_val = (
                    item.get("nav_usd")
                    or item.get("nav_usdt")
                    or item.get("nav")
                    or item.get("equity")
                    or item.get("total_equity")
                )
                numeric = None
                try:
                    numeric = float(nav_val)
                except Exception:
                    continue
                if numeric is not None:
                    nav_value = numeric
                    nav_source = path.name
                    break
            if nav_value is not None:
                break
        except Exception as exc:
            print(f"[doctor] NAV read failed for {path}: {exc}")
    return {"value": nav_value, "source": nav_source}

def _collect_positions() -> Dict[str, Any]:
    count = 0
    error = None
    if utils_get_live_positions is not None:
        try:
            positions = utils_get_live_positions(None)
            count = len(positions or [])
        except Exception as exc:
            error = str(exc)
    else:
        error = "helper unavailable"
    return {"count": count, "error": error}

def _collect_reserves() -> Dict[str, Any]:
    reserve_btc = 0.0
    reserve_xaut = 0.0
    try:
        reserve_btc = float(os.getenv("RESERVE_BTC") or os.getenv("DASHBOARD_RESERVE_BTC") or 0.0)
    except Exception:
        reserve_btc = 0.0
    try:
        reserve_xaut = float(os.getenv("RESERVE_XAUT") or 0.0)
    except Exception:
        reserve_xaut = 0.0

    prices: Dict[str, float] = {}
    if utils_get_coingecko_prices is not None:
        try:
            prices = utils_get_coingecko_prices()
        except Exception as exc:
            print(f"[doctor] reserves price fetch failed: {exc}")

    btc_px = float(prices.get("BTC", 0.0) or 0.0)
    xaut_px = float(prices.get("XAUT", 0.0) or 0.0)
    btc_val = reserve_btc * btc_px if btc_px else None
    xaut_val = reserve_xaut * xaut_px if xaut_px else None
    total = 0.0
    for val in (btc_val, xaut_val):
        if isinstance(val, (int, float)):
            total += float(val)
    return {
        "btc_qty": reserve_btc,
        "btc_px": btc_px,
        "btc_val": btc_val,
        "xaut_qty": reserve_xaut,
        "xaut_px": xaut_px,
        "xaut_val": xaut_val,
        "total": total if total > 0 else None,
    }


def _load_treasury_total(freshness_limit: float = 600.0) -> float:
    """Return treasury total from canonical source within freshness window."""
    now = time.time()
    sources: List[Dict[str, Any]] = []
    for label in ("logs/treasury.json",):
        path = Path(label)
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        updated_raw = payload.get("updated_at") or payload.get("ts")
        freshness = None
        if isinstance(updated_raw, str):
            try:
                ts = datetime.fromisoformat(updated_raw.replace("Z", "+00:00")).timestamp()
                freshness = now - ts
            except Exception:
                freshness = None
        sources.append({"label": label, "payload": payload, "freshness": freshness})

    for entry in sources:
        label = entry["label"]
        freshness = entry["freshness"]
        payload = entry["payload"]
        if freshness is not None and freshness <= freshness_limit:
            total = payload.get("total_usd")
            if isinstance(total, (int, float)):
                return float(total)
            breakdown = payload.get("breakdown")
            if isinstance(breakdown, dict):
                alt_total = breakdown.get("total_treasury_usdt")
                if isinstance(alt_total, (int, float)):
                    return float(alt_total)

    try:
        cfg = json.loads(Path("config/reserves.json").read_text())
    except Exception:
        cfg = None

    if isinstance(cfg, dict):
        stable = {"USDT", "USDC", "DAI", "FDUSD", "TUSD"}
        total = 0.0
        for symbol, qty in cfg.items():
            try:
                amount = float(qty)
            except Exception:
                continue
            asset = str(symbol).upper()
            if asset in stable:
                total += amount
                continue
            try:
                from execution.exchange_utils import get_price  # local import

                px = float(get_price(f"{asset}USDT") or 0.0)
            except Exception:
                px = 0.0
            if px <= 0 and asset.endswith("USDT"):
                px = 1.0
            if px > 0:
                total += amount * px
        return float(total)
    return 0.0

def _collect_firestore_status() -> Dict[str, Any]:
    enabled = str(os.getenv("FIRESTORE_ENABLED", "")).strip().lower() in ("1", "true", "yes", "on")
    status = "disabled"
    age_str = "n/a"
    last_ts = None
    if enabled:
        log_path = Path("logs/sync_state.log")
        if log_path.exists():
            try:
                lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                for line in reversed(lines):
                    if "Firestore publish ok" in line:
                        parts = line.strip().split()
                        ts_candidate = None
                        if parts:
                            ts_candidate = _parse_iso(parts[0])
                        if ts_candidate is None and len(parts) > 1:
                            ts_candidate = _parse_iso(parts[0] + " " + parts[1])
                        last_ts = ts_candidate
                        break
            except Exception as exc:
                print(f"[doctor] Firestore log parse failed: {exc}")
        status = "OK" if last_ts and (_mins_ago(last_ts) is not None and _mins_ago(last_ts) <= 5) else "stale"
        age_str = _human_minutes(last_ts)
    return {"enabled": enabled, "status": status, "age": age_str}

def _collect_heartbeats() -> Dict[str, Any]:
    path = Path("logs/execution/sync_heartbeats.jsonl")
    env = os.getenv("ENV", "prod")
    services: Dict[str, Optional[datetime]] = {"executor": None, "sync_state": None}
    service_source: Dict[str, Optional[str]] = {"executor": None, "sync_state": None}
    doc_source: Optional[str] = None

    def _update_service(name: str, ts: Optional[datetime], origin: Optional[str]) -> None:
        if ts is None:
            return
        current = services.get(name)
        if current is None or ts > current:
            services[name] = ts
            service_source[name] = origin

    try:
        db = get_firestore_db(strict=False)
        if db is not None and not getattr(db, "_is_noop", False):
            firestore_hits: List[str] = []
            telemetry_ref = (
                db.collection("hedge")
                .document(env)
                .collection("telemetry")
                .document("health")
            )
            try:
                telemetry_snap = telemetry_ref.get(timeout=5.0)
            except Exception as exc:
                print(f"[doctor] heartbeat telemetry read failed: {exc}")
            else:
                if getattr(telemetry_snap, "exists", False):
                    payload = telemetry_snap.to_dict() or {}
                    parsed = parse_heartbeats(payload)
                    _update_service("executor", parsed.get("executor_live"), "firestore:telemetry/health")
                    _update_service("sync_state", parsed.get("sync_state"), "firestore:telemetry/health")
                    firestore_hits.append("telemetry/health")
            executor_ref = (
                db.collection("hedge")
                .document(env)
                .collection("health")
                .document("executor_live")
            )
            try:
                exec_snap = executor_ref.get(timeout=5.0)
            except Exception as exc:
                print(f"[doctor] executor heartbeat read failed: {exc}")
            else:
                if getattr(exec_snap, "exists", False):
                    payload = exec_snap.to_dict() or {}
                    ts_val = _extract_service_timestamp(payload, SERVICE_ALIASES.get("executor_live", ("executor_live",)))
                    _update_service("executor", ts_val, "firestore:health/executor_live")
                    firestore_hits.append("health/executor_live")
            if firestore_hits:
                doc_source = "firestore"
    except Exception as exc:
        print(f"[doctor] heartbeat firestore read failed: {exc}")
    if path.exists():
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                lines = handle.readlines()[-500:]
            for line in lines:
                try:
                    payload = json.loads(line.strip())
                except Exception:
                    continue
                if not isinstance(payload, dict):
                    continue
                svc = str(payload.get("service") or "").lower()
                ts = _to_dt(
                    payload.get("timestamp")
                    or payload.get("ts_iso")
                    or payload.get("updated_at")
                    or payload.get("ts")
                )
                if ts is None:
                    continue
                if "executor" in svc:
                    _update_service("executor", ts, "logs")
                elif "sync" in svc:
                    _update_service("sync_state", ts, "logs")
        except Exception as exc:
            print(f"[doctor] heartbeat parse failed: {exc}")
    display_parts = []
    stale = False
    service_details: Dict[str, Dict[str, Any]] = {}
    for label in ("executor", "sync_state"):
        ts = services.get(label)
        mins = age_minutes(ts)
        if mins is None:
            age_str = "n/a"
            stale = True
        else:
            age_str = f"{mins:.0f}m"
            if mins >= 2:
                stale = True
        if mins is None:
            display_parts.append(f"{label}=n/a")
        else:
            display_parts.append(f"{label}≈{mins:.0f}m")
        service_details[label] = {
            "ts": ts,
            "age_minutes": mins,
            "age_display": age_str,
            "source": service_source.get(label),
        }
    status = "stale" if stale else "fresh"
    return {
        "display": " ".join(display_parts) if display_parts else "n/a",
        "status": status,
        "services": service_details,
        "source": doc_source or "fallback",
    }


def collect_router_health(limit: int = 200) -> Dict[str, Any]:
    order_path = Path("logs/execution/order_metrics.jsonl")
    signal_path = Path("logs/execution/signal_metrics.jsonl")
    slip_window = AggWindow(capacity=max(1, limit))
    latency_window = AggWindow(capacity=max(1, limit))

    ack_events, fill_events, _ = _load_order_events(limit)

    def _identifier(record: Mapping[str, Any]) -> str:
        order_id = record.get("orderId") or record.get("order_id")
        client_id = record.get("clientOrderId") or record.get("client_order_id")
        if order_id:
            return str(order_id)
        if client_id:
            return str(client_id)
        return f"anon_{id(record)}"

    ack_ids = {_identifier(ack) for ack in ack_events}
    fill_ids: set[str] = set()

    metrics_map: Dict[str, Dict[str, Any]] = {}
    if order_path.exists():
        try:
            with order_path.open("r", encoding="utf-8", errors="ignore") as handle:
                lines = handle.readlines()[-limit:]
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if not isinstance(row, dict):
                    continue
                attempt = row.get("attempt_id") or row.get("attemptId")
                if attempt:
                    metrics_map[str(attempt)] = row
        except Exception as exc:
            print(f"[doctor] router_metrics_read_failed: {exc}")

    for fill in fill_events:
        identifier = fill.get("identifier") or _identifier(fill)
        fill_ids.add(identifier)
        latency_val = fill.get("latency_ms")
        if latency_val is not None:
            latency_window.add(latency_val)
        attempt_id = str(fill.get("attempt_id") or "")
        metrics = metrics_map.get(attempt_id)
        if metrics:
            prices = metrics.get("prices") or {}
            mark = _safe_float(prices.get("mark"))
            fill_price = _safe_float(fill.get("avg_price"))
            if mark not in (None, 0) and fill_price not in (None, 0):
                slip = ((fill_price - mark) / mark) * 10_000.0
                if str(fill.get("side") or "").upper() == "SELL":
                    slip *= -1.0
                slip_window.add(slip)

    ack_count = len(ack_ids)
    fill_count = len(fill_ids)
    fill_rate_pct = (fill_count / ack_count * 100.0) if ack_count else 0.0

    attempted = emitted = vetoed = 0
    if signal_path.exists():
        try:
            with signal_path.open("r", encoding="utf-8", errors="ignore") as handle:
                lines = handle.readlines()[-limit:]
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if not isinstance(row, dict):
                    continue
                attempted += 1
                doctor = row.get("doctor") or {}
                ok = bool(doctor.get("ok"))
                if ok:
                    emitted += 1
                else:
                    vetoed += 1
        except Exception as exc:
            print(f"[doctor] signal_metrics_read_failed: {exc}")

    slip_stats = slip_window.snapshot()
    latency_stats = latency_window.snapshot()
    summary = {
        "slippage_p50": slip_stats.get("p50", 0.0),
        "slippage_p95": slip_stats.get("p95", 0.0),
        "latency_p50_ms": latency_stats.get("p50", 0.0),
        "latency_p95_ms": latency_stats.get("p95", 0.0),
        "attempted": attempted,
        "emitted": emitted,
        "vetoed": vetoed,
        "ack_count": ack_count,
        "fill_count": fill_count,
        "fill_rate_pct": fill_rate_pct,
    }
    print(
        "[doctor] Router: "
        f"slip_p50={summary['slippage_p50']:.2f}bps slip_p95={summary['slippage_p95']:.2f}bps "
        f"lat_p50={summary['latency_p50_ms']:.0f}ms lat_p95={summary['latency_p95_ms']:.0f}ms "
        f"fill_rate={summary['fill_rate_pct']:.1f}% attempted={attempted} emitted={emitted} vetoed={vetoed}"
    )
    return summary


def _drawdown_state() -> Dict[str, Any]:
    payload = safe_load("logs/cache/peak_state.json", default={}) or {}
    dd_pct = _safe_float(payload.get("dd_pct"))
    dd_abs = _safe_float(payload.get("dd_abs"))
    peak = _safe_float(payload.get("peak") or payload.get("peak_equity"))
    nav = _safe_float(payload.get("nav") or payload.get("nav_usd"))
    updated_at = payload.get("updated_at") or payload.get("ts")
    return {
        "dd_pct": dd_pct,
        "dd_abs": dd_abs,
        "peak": peak,
        "nav": nav,
        "updated_at": updated_at,
    }


def collect_doctor_snapshot() -> Dict[str, Any]:
    nav_freshness = {
        "fresh": is_nav_fresh(),
        "age": get_nav_age(),
        "drawdown": _drawdown_state(),
        "zar_rate": None,
        "zar_source": "none",
    }
    snapshot = {
        "nav": _collect_nav(),
        "positions": _collect_positions(),
        "firestore": _collect_firestore_status(),
        "heartbeats": _collect_heartbeats(),
        "reserves": _collect_reserves(),
        "nav_freshness": nav_freshness,
    }
    if utils_get_usd_to_zar is not None:
        rate = None
        source = "none"
        try:
            rate = utils_get_usd_to_zar(force=True)
            if rate:
                source = "fresh"
        except Exception as exc:
            print(f"[doctor] ZAR conversion fetch failed: {exc}")
        if not rate:
            try:
                cached = utils_get_usd_to_zar()
            except Exception:
                cached = None
            if cached:
                rate = cached
                source = "cached"
                print("[doctor] ZAR conversion stale, using cached value")
            else:
                print("[doctor] ZAR conversion unavailable (no rate)")
        nav_freshness["zar_rate"] = rate
        nav_freshness["zar_source"] = source
    else:
        print("[doctor] ZAR conversion helper unavailable")
    snapshot["drawdown"] = nav_freshness["drawdown"]
    snapshot["zar_rate"] = nav_freshness["zar_rate"]
    snapshot["zar_source"] = nav_freshness["zar_source"]
    print("[doctor] nav_freshness integrated (is_nav_fresh + get_nav_age)")
    return snapshot


def main():
    cfg_loader = getattr(sc, "_load_cfg", None) or getattr(sc, "load_cfg", None)
    try:
        if callable(cfg_loader):
            cfg = cfg_loader() or {}
            print("[doctor] config loaded ok")
        else:
            cfg = {}
            print("[doctor] fallback load ok (no loader)")
    except Exception as exc:
        cfg = {}
        print(f"[doctor] fallback load ok: {exc}")

    state_path = getattr(sc, "STATE_PATH", "")
    st = safe_load(state_path)
    if st:
        print("[doctor] state load ok")
    else:
        print("[doctor] state fallback (empty)")
    out = {"env":{}, "strategies":[]}

    # env snapshot
    out["env"]["dualSide"], out["env"]["dualSide_raw"] = _dual_side()
    out["env"]["telegram_enabled"] = os.getenv("TELEGRAM_ENABLED","") in ("1","true","True")
    out["env"]["heartbeat_minutes"] = os.getenv("HEARTBEAT_MINUTES","")
    out["env"]["dry_run"] = os.getenv("DRY_RUN","")
    out["env"]["testnet"] = os.getenv("BINANCE_TESTNET","")
    out["env"]["use_futures"] = os.getenv("USE_FUTURES","")

    for name, scfg in (cfg.get("strategies") or {}).items():
        sym = scfg["symbol"]
        tf  = scfg.get("timeframe","30m")
        pxs = _series(st, sym, tf)
        meta= _meta(st, sym, tf)
        prev_z = meta.get("prev_z", None)
        in_trade = bool(meta.get("in_trade", False))
        side = meta.get("side")
        px_live = _price(sym)

        inds: Dict[str, Any] = {}
        if pxs:
            inds = sc._fetch_indicators_from_your_stack(sym, tf, scfg) or sc._compute_indicators_from_series(pxs, scfg)
        z   = float(inds.get("z", 0.0))
        rsi = float(inds.get("rsi", 50.0))
        atr = float(inds.get("atr_proxy", 0.0))

        entry = scfg.get("entry",{})
        zmin  = float(entry.get("zscore_min", 0.8))
        rband = entry.get("rsi_band", [0,100])
        rlo, rhi = float(rband[0]), float(rband[1])
        atr_min = float(entry.get("atr_min", 0.0))
        rsi_ok  = (rlo <= rsi <= rhi)
        atr_ok  = atr >= atr_min

        if prev_z is None:
            cross_up = cross_down = False
            cross_reason = "prev_z not seeded"
        else:
            cross_up   = (prev_z <  zmin) and (z >=  zmin)
            cross_down = (prev_z > -zmin) and (z <= -zmin)
            cross_reason = "cross ok" if (cross_up or cross_down) else "no cross"

        blocked_by = []
        if px_live is None:
            blocked_by.append("price_unavailable")
        if not (cross_up or cross_down):
            blocked_by.append("no_cross")
        if not rsi_ok:
            blocked_by.append("rsi_veto")
        if not atr_ok:
            blocked_by.append("atr_floor")
        if in_trade:
            blocked_by.append("already_in_trade")

        out["strategies"].append({
            "name": name, "symbol": sym, "tf": tf,
            "series_len": len(pxs),
            "live_px": px_live,
            "z": z, "prev_z": prev_z,
            "cross_up": cross_up, "cross_down": cross_down, "cross_reason": cross_reason,
            "rsi": rsi, "rsi_band": [rlo, rhi], "rsi_ok": rsi_ok,
            "atr": atr, "atr_min": atr_min, "atr_ok": atr_ok,
            "in_trade": in_trade, "side": side,
            "blocked_by": blocked_by
        })

    snapshot = collect_doctor_snapshot()
    nav_info = snapshot["nav"]
    positions_info = snapshot["positions"]
    heartbeat_info = snapshot["heartbeats"]
    reserves_info = snapshot["reserves"]
    nav_freshness = snapshot["nav_freshness"]
    drawdown_info = nav_freshness.get("drawdown", {})
    zar_rate = nav_freshness.get("zar_rate")
    zar_source = nav_freshness.get("zar_source")

    env = os.getenv("ENV", "dev")
    fs_status = "STALE"
    fs_age = None
    try:
        db = get_firestore_db(strict=False)
        if db is None or getattr(db, "_is_noop", False):
            raise RuntimeError("Firestore offline (noop client)")
        doc_ref = db.collection("hedge").document(env).collection("health").document("sync_state")
        doc = doc_ref.get(timeout=5.0)
        if doc.exists:
            update_time = getattr(doc, "update_time", None)
            if update_time is not None:
                fs_age = (datetime.now(timezone.utc) - update_time).total_seconds()
                if fs_age < 120:
                    fs_status = "OK"
                elif fs_age < 600:
                    fs_status = "STALE"
                else:
                    fs_status = "DOWN"
            else:
                fs_status = "UNKNOWN"
        else:
            fs_status = "MISSING"
    except Exception as exc:
        print(json.dumps({"status": "offline", "reason": str(exc)}))
        return

    payload = json.dumps(out, indent=2, sort_keys=False)
    print(payload)

    is_tty = sys.stdout.isatty()
    nav_value = nav_info["value"] if isinstance(nav_info.get("value"), (int, float)) else None
    nav_display = f"{nav_value:,.2f} USD" if nav_value is not None else "n/a"
    nav_usd_val = float(nav_value) if nav_value is not None else None
    nav_line = f"[doctor] NAV check: {nav_display}"
    if nav_info["source"] and nav_info["source"] != "n/a":
        nav_line += f" (source={nav_info['source']})"
    print(nav_line)

    nav_age = nav_freshness.get("age")
    age_display = f"{nav_age:.1f}s" if isinstance(nav_age, (int, float)) else "n/a"
    nav_status = "FRESH" if nav_freshness.get("fresh") else "STALE"
    print(f"[doctor] NAV freshness: {nav_status} (age={age_display})")

    dd_pct = drawdown_info.get("dd_pct") or 0.0
    dd_abs = drawdown_info.get("dd_abs") or 0.0
    peak_equity = drawdown_info.get("peak") or 0.0
    nav_equity = drawdown_info.get("nav") or 0.0
    print(
        "[doctor] Drawdown: "
        f"dd={dd_pct:.2f}% (abs=${dd_abs:,.2f}, peak=${peak_equity:,.2f}, nav=${nav_equity:,.2f})"
    )

    if positions_info["error"]:
        print(f"[doctor] Positions: {positions_info['count']} open (fallback: {positions_info['error']})")
    else:
        print(f"[doctor] Positions: {positions_info['count']} open")

    btc_val_display = (
        f"${reserves_info['btc_val']:,.2f}" if isinstance(reserves_info["btc_val"], (int, float)) else "n/a"
    )
    xaut_val_display = (
        f"${reserves_info['xaut_val']:,.2f}" if isinstance(reserves_info["xaut_val"], (int, float)) else "n/a"
    )
    raw_reserves_total = reserves_info.get("total")
    treasury_total: Optional[float] = (
        float(raw_reserves_total) if isinstance(raw_reserves_total, (int, float)) and raw_reserves_total > 0 else None
    )
    if treasury_total is None:
        fallback_total = _load_treasury_total()
        if fallback_total > 0:
            treasury_total = fallback_total
    reserves_usd_val = float(treasury_total) if isinstance(treasury_total, (int, float)) else None
    total_display = f"${reserves_usd_val:,.2f}" if reserves_usd_val is not None else "n/a"
    total_equity_val = None
    if nav_usd_val is not None or reserves_usd_val is not None:
        total_equity_val = (nav_usd_val or 0.0) + (reserves_usd_val or 0.0)
    total_equity_display = f"${total_equity_val:,.2f}" if total_equity_val is not None else "n/a"
    nav_zar = None
    reserves_zar = None
    total_zar = None
    if zar_rate and nav_usd_val is not None:
        nav_zar = nav_usd_val * float(zar_rate)
    if zar_rate and reserves_usd_val is not None:
        reserves_zar = reserves_usd_val * float(zar_rate)
    if zar_rate and total_equity_val is not None:
        total_zar = total_equity_val * float(zar_rate)
    print(
        "[doctor] reserves "
        f"BTC: {reserves_info['btc_qty']:.4f} → {btc_val_display} | "
        f"XAUT: {reserves_info['xaut_qty']:.4f} → {xaut_val_display} | "
        f"Total: {total_display}"
    )

    print(f"[doctor] NAV: {f'${nav_usd_val:,.2f}' if nav_usd_val is not None else 'n/a'}")
    print(f"[doctor] Reserves: {total_display}")
    print(f"[doctor] Total Equity: {total_equity_display}")

    if zar_rate:
        nav_zar_display = f"R{nav_zar:,.0f}" if isinstance(nav_zar, (int, float)) else "n/a"
        reserves_zar_display = f"R{reserves_zar:,.0f}" if isinstance(reserves_zar, (int, float)) else "n/a"
        total_zar_display = f"R{total_zar:,.0f}" if isinstance(total_zar, (int, float)) else "n/a"
        print(
            f"[doctor] ZAR conversion @ {zar_rate:.2f} ({zar_source}) → "
            f"NAV≈{nav_zar_display} | Reserves≈{reserves_zar_display} | Total≈{total_zar_display}"
        )
    else:
        print("[doctor] ZAR conversion data unavailable")

    print(f"[doctor] Firestore: {fs_status} (age={fs_age or 'n/a'}s)")

    hb_status = heartbeat_info["status"]
    print(f"[doctor] Heartbeat: {heartbeat_info['display']} ({hb_status})")

    router_stats = collect_router_health()

    nav_summary = f"NAV: {f'${nav_usd_val:,.2f}' if nav_usd_val is not None else 'n/a'}"
    reserves_summary = f"Reserves: {total_display}"
    total_summary = f"Total Equity: {total_equity_display}"
    summary_parts = [
        nav_summary,
        f"NAV Freshness: {nav_status}",
        f"Drawdown: {dd_pct:.2f}%",
        f"Positions: {positions_info['count']}",
        f"Firestore: {fs_status}",
        reserves_summary,
        total_summary,
        f"Heartbeat: {hb_status}",
        (
            f"Router slip50={router_stats['slippage_p50']:.2f}bps "
            f"slip95={router_stats['slippage_p95']:.2f}bps"
        ),
    ]
    if zar_rate and isinstance(total_zar, (int, float)):
        zar_summary = f"Total ZAR≈R{total_zar:,.0f}"
    else:
        zar_summary = "ZAR≈n/a"
    summary_parts.append(zar_summary)
    colored_summary = " | ".join(summary_parts)
    if is_tty:
        if nav_usd_val is not None:
            colored_summary = colored_summary.replace(
                nav_summary,
                f"NAV: {_colorize(nav_summary.split(': ', 1)[1], ANSI_GREEN, True)}",
            )
        nav_color = ANSI_GREEN if nav_status == "FRESH" else ANSI_RED
        colored_summary = colored_summary.replace(
            f"NAV Freshness: {nav_status}",
            f"NAV Freshness: {_colorize(nav_status, nav_color, True)}",
        )
        dd_color = ANSI_RED if dd_pct and dd_pct > 0 else ANSI_GREEN
        colored_summary = colored_summary.replace(
            f"Drawdown: {dd_pct:.2f}%",
            f"Drawdown: {_colorize(f'{dd_pct:.2f}%', dd_color, True)}",
        )
        fs_color = ANSI_GREEN if fs_status == "OK" else ANSI_RED
        colored_summary = colored_summary.replace(
            f"Firestore: {fs_status}",
            f"Firestore: {_colorize(fs_status, fs_color, True)}",
        )
        if reserves_usd_val is not None:
            colored_summary = colored_summary.replace(
                reserves_summary,
                f"Reserves: {_colorize(reserves_summary.split(': ', 1)[1], ANSI_GREEN, True)}",
            )
        if total_equity_val is not None:
            colored_summary = colored_summary.replace(
                total_summary,
                f"Total Equity: {_colorize(total_summary.split(': ', 1)[1], ANSI_GREEN, True)}",
            )
        hb_color = ANSI_GREEN if hb_status == "fresh" else ANSI_RED
        colored_summary = colored_summary.replace(
            f"Heartbeat: {hb_status}",
            f"Heartbeat: {_colorize(hb_status, hb_color, True)}",
        )
        if zar_rate and isinstance(total_zar, (int, float)):
            colored_summary = colored_summary.replace(
                zar_summary,
                f"{zar_summary.split('≈', 1)[0]}≈{_colorize(f'R{total_zar:,.0f}', ANSI_GREEN, True)}",
            )

    print(f"[doctor] summary {colored_summary}")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - defensive CLI guard
        print(f"[doctor] warning: {exc}", file=sys.stderr)
        sys.exit(0)


__all__ = ["collect_doctor_snapshot", "collect_router_health"]
