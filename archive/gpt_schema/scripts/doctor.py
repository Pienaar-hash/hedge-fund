# mypy: ignore-errors
#!/usr/bin/env python3
import os
import sys
import json
import time
import hmac
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from datetime import datetime, timezone

import requests
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import streamlit as st  # type: ignore

    _HAVE_ST = True
except Exception:  # pragma: no cover
    st = None  # type: ignore
    _HAVE_ST = False

from dashboard.router_health import _load_order_events  # noqa: E402

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
    from execution.utils import get_usd_to_zar as utils_get_usd_to_zar
except Exception:  # pragma: no cover
    utils_get_usd_to_zar = None
try:
    from binance.um_futures import UMFutures  # type: ignore
except Exception:  # pragma: no cover
    UMFutures = None

try:
    from execution.exchange_utils import (
        get_um_client as _get_um_client_helper,
        um_client_error as _get_um_client_error,
        _is_dual_side as _exchange_dual_side,
    )
except Exception:  # pragma: no cover - optional dependency
    _get_um_client_helper = None  # type: ignore[assignment]

    def _get_um_client_error() -> Optional[str]:
        return None

    _exchange_dual_side = None  # type: ignore[assignment]

def _now_ms(): return int(time.time()*1000)

def _dual_side():
    last_error: Optional[str] = None
    if _exchange_dual_side is not None:
        try:
            return bool(_exchange_dual_side()), "exchange_utils._is_dual_side"
        except Exception as exc:
            last_error = f"exchange_utils_failed:{exc}"
    client = _get_um_client_helper() if _get_um_client_helper is not None else None
    if client is not None:
        try:
            payload = client.get_position_mode()
            return bool(payload.get("dualSidePosition", False)), payload
        except Exception as exc:  # pragma: no cover - network dependency
            last_error = f"um_client_error:{exc}"
    elif _get_um_client_error is not None:
        last_error = _get_um_client_error() or last_error
    key = os.environ.get("BINANCE_API_KEY", "")
    sec_txt = os.environ.get("BINANCE_API_SECRET", "")
    if not key or not sec_txt:
        return False, last_error or "missing_credentials"
    base = (
        "https://testnet.binancefuture.com"
        if os.getenv("BINANCE_TESTNET", "").strip().lower() in ("1", "true", "yes", "on")
        else "https://fapi.binance.com"
    )
    query = f"timestamp={_now_ms()}"
    sig = hmac.new(sec_txt.encode(), query.encode(), hashlib.sha256).hexdigest()
    try:
        resp = requests.get(
            f"{base}/fapi/v1/positionSide/dual?{query}&signature={sig}",
            headers={"X-MBX-APIKEY": key},
            timeout=10,
        )
        resp.raise_for_status()
    except Exception as exc:
        return False, last_error or f"http_error:{exc}"
    payload = resp.json()
    return bool(payload.get("dualSidePosition", False)), payload

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


_FUTURES_CLIENT: Optional[Any] = None
_FUTURES_CLIENT_ERROR: Optional[str] = None


def _ensure_futures_client() -> Tuple[Optional[Any], Optional[str]]:
    global _FUTURES_CLIENT, _FUTURES_CLIENT_ERROR
    if _FUTURES_CLIENT is not None or _FUTURES_CLIENT_ERROR is not None:
        return _FUTURES_CLIENT, _FUTURES_CLIENT_ERROR
    if _get_um_client_helper is not None:
        _FUTURES_CLIENT = _get_um_client_helper()
        if _FUTURES_CLIENT is None:
            _FUTURES_CLIENT_ERROR = _get_um_client_error() or "um_client_unavailable"
        return _FUTURES_CLIENT, _FUTURES_CLIENT_ERROR
    if UMFutures is None:
        _FUTURES_CLIENT_ERROR = "UMFutures unavailable"
        return None, _FUTURES_CLIENT_ERROR
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        _FUTURES_CLIENT_ERROR = "missing credentials"
        return None, _FUTURES_CLIENT_ERROR
    kwargs: Dict[str, Any] = {"key": api_key, "secret": api_secret}
    testnet = str(os.getenv("BINANCE_TESTNET", "")).strip().lower() in ("1", "true", "yes", "on")
    if testnet:
        kwargs["base_url"] = "https://testnet.binancefuture.com"
    try:
        _FUTURES_CLIENT = UMFutures(**kwargs)  # type: ignore[arg-type]
    except Exception as exc:  # pragma: no cover - network dependency
        _FUTURES_CLIENT = None
        _FUTURES_CLIENT_ERROR = str(exc)
    return _FUTURES_CLIENT, _FUTURES_CLIENT_ERROR


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


def _load_cached_positions(path: Path = Path("logs/execution/position_state.jsonl")) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            lines = handle.readlines()
    except Exception as exc:
        print(f"[doctor] cached positions read failed: {exc}")
        return []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            entries = payload.get("items") or payload.get("positions") or payload.get("rows")
            if isinstance(entries, list):
                return entries
    return []


def _collect_positions(client: Optional[Any], init_error: Optional[str]) -> Dict[str, Any]:
    count = 0
    error = init_error
    source = "none"
    positions: List[Dict[str, Any]] = []
    if utils_get_live_positions is not None and client is not None and not init_error:
        try:
            positions = utils_get_live_positions(client) or []
        except Exception as exc:
            error = str(exc)
    elif utils_get_live_positions is None:
        error = "helper unavailable"
    elif client is None and not init_error:
        error = "client unavailable"
    if positions:
        count = len(positions)
        source = "live"
    else:
        cached_positions = _load_cached_positions()
        if cached_positions:
            positions = cached_positions
            count = len(positions)
            source = "cache"
    sample = positions[:5] if positions else []
    return {"count": count, "error": error, "source": source, "sample": sample}

def _load_treasury_payload(env: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    return None, None


def _collect_reserves() -> Dict[str, Any]:
    return {"total": None, "assets": [], "updated_at": None, "source": "disabled"}


def _load_treasury_total(freshness_limit: float = 600.0) -> float:
    return 0.0

def _collect_firestore_status() -> Dict[str, Any]:
    return {"enabled": False, "status": "disabled", "age": "n/a"}

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


def collect_doctor_snapshot(
    futures_client: Optional[Any] = None,
    *,
    client_error: Optional[str] = None,
) -> Dict[str, Any]:
    if futures_client is None and client_error is None:
        futures_client, client_error = _ensure_futures_client()

    nav_freshness = {
        "fresh": is_nav_fresh(),
        "age": get_nav_age(),
        "drawdown": _drawdown_state(),
        "zar_rate": None,
        "zar_source": "none",
        "zar_age_seconds": None,
        "zar_status": "MISSING",
    }

    snapshot = {
        "nav": _collect_nav(),
        "positions": _collect_positions(futures_client, client_error),
        "firestore": _collect_firestore_status(),
        "heartbeats": _collect_heartbeats(),
        "reserves": _collect_reserves(),
        "nav_freshness": nav_freshness,
    }

    if client_error:
        print(f"[doctor] futures client unavailable: {client_error}")

    if utils_get_usd_to_zar is not None:
        rate: Optional[float] = None
        source = "none"
        age_seconds: Optional[float] = None
        try:
            rate, meta = utils_get_usd_to_zar(force=True, with_meta=True)  # type: ignore[assignment]
            source = (meta or {}).get("source") or "api"
            age_seconds = (meta or {}).get("age")  # type: ignore[assignment]
        except Exception as exc:
            print(f"[doctor] ZAR conversion fetch failed: {exc}")
            rate = None
            source = "error"
            age_seconds = None
        if rate is None:
            try:
                cached_rate, cached_meta = utils_get_usd_to_zar(with_meta=True)  # type: ignore[assignment]
            except Exception as exc:
                print(f"[doctor] ZAR conversion cached fetch failed: {exc}")
                cached_rate = None
                cached_meta = None
            if cached_rate is not None:
                rate = cached_rate
                source = (cached_meta or {}).get("source") or "cache"
                age_seconds = (cached_meta or {}).get("age")  # type: ignore[assignment]
                print("[doctor] ZAR conversion stale, using cached value")
            else:
                print("[doctor] ZAR conversion unavailable (no rate)")
        status = "MISSING"
        if rate is not None:
            if source == "cache" or (age_seconds is not None and age_seconds > 6 * 3600):
                status = "STALE"
            else:
                status = "FRESH"
        nav_freshness["zar_rate"] = rate
        nav_freshness["zar_source"] = source or "none"
        nav_freshness["zar_age_seconds"] = age_seconds
        nav_freshness["zar_status"] = status
    else:
        print("[doctor] ZAR conversion helper unavailable")

    snapshot["drawdown"] = nav_freshness["drawdown"]
    snapshot["zar_rate"] = nav_freshness["zar_rate"]
    snapshot["zar_source"] = nav_freshness["zar_source"]
    snapshot["zar_status"] = nav_freshness.get("zar_status")
    snapshot["zar_age_seconds"] = nav_freshness.get("zar_age_seconds")
    print("[doctor] nav_freshness integrated (is_nav_fresh + get_nav_age)")
    return snapshot


def run_doctor_subprocess(timeout: int = 30, placeholder: Any = None) -> Tuple[str, Optional[str]]:
    cmd = [sys.executable, str(Path(__file__).resolve())]
    if placeholder is None and _HAVE_ST:
        placeholder = st.empty()
    process = subprocess.Popen(  # noqa: S603
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: List[str] = []
    start = time.time()
    stream_target = placeholder
    try:
        if process.stdout is None:
            raise RuntimeError("doctor subprocess missing stdout pipe")
        for line in process.stdout:
            lines.append(line)
            if stream_target is not None:
                stream_target.code("".join(lines[-120:]))
            if timeout and timeout > 0 and (time.time() - start) > timeout:
                process.kill()
                raise TimeoutError("doctor subprocess timed out")
        process.wait()
    finally:
        if process.stdout is not None:
            process.stdout.close()
    output = "".join(lines)
    if process.returncode == 1:
        message = "doctor subprocess exited with status 1"
        if stream_target is not None:
            stream_target.warning(message)
        return output, message
    if process.returncode not in (0, None):
        message = f"doctor subprocess exited with status {process.returncode}"
        if stream_target is not None:
            stream_target.error(message)
        raise RuntimeError(message)
    if stream_target is not None and output:
        stream_target.code(output)
    return output, None


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
    zar_age_seconds = nav_freshness.get("zar_age_seconds")
    zar_status_label = nav_freshness.get("zar_status")

    env = os.getenv("ENV", "dev")
    fs_status = "DISABLED"
    fs_age = None

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

    pos_source = positions_info.get("source") or "unknown"
    pos_error = positions_info.get("error")
    if pos_error:
        print(
            f"[doctor] Positions [{pos_source}]: {positions_info['count']} open "
            f"(note: {pos_error})"
        )
    else:
        print(f"[doctor] Positions [{pos_source}]: {positions_info['count']} open")

    total_equity_val = nav_usd_val
    total_equity_display = f"${total_equity_val:,.2f}" if total_equity_val is not None else "n/a"
    nav_zar = None
    total_zar = None
    if zar_rate and nav_usd_val is not None:
        nav_zar = nav_usd_val * float(zar_rate)
    if zar_rate and total_equity_val is not None:
        total_zar = total_equity_val * float(zar_rate)

    assets_raw: List[Dict[str, Any]] = []

    print(f"[doctor] NAV: {f'${nav_usd_val:,.2f}' if nav_usd_val is not None else 'n/a'}")
    print(f"[doctor] Total Equity: {total_equity_display}")

    if zar_rate:
        nav_zar_display = f"R{nav_zar:,.0f}" if isinstance(nav_zar, (int, float)) else "n/a"
        total_zar_display = f"R{total_zar:,.0f}" if isinstance(total_zar, (int, float)) else "n/a"
        zar_age_display = (
            f"{zar_age_seconds/3600:.1f}h" if isinstance(zar_age_seconds, (int, float)) else "n/a"
        )
        zar_status_display = zar_status_label or "n/a"
        print(
            f"[doctor] ZAR conversion @ {zar_rate:.2f} ({zar_source}) "
            f"status={zar_status_display} age={zar_age_display} → "
            f"NAV≈{nav_zar_display} | Total≈{total_zar_display}"
        )
    else:
        print("[doctor] ZAR conversion data unavailable")

    print(f"[doctor] Firestore: {fs_status} (age={fs_age or 'n/a'}s)")

    hb_status = heartbeat_info["status"]
    print(f"[doctor] Heartbeat: {heartbeat_info['display']} ({hb_status})")

    router_stats = collect_router_health()

    nav_summary = f"NAV: {f'${nav_usd_val:,.2f}' if nav_usd_val is not None else 'n/a'}"
    total_summary = f"Total Equity: {total_equity_display}"
    summary_parts = [
        nav_summary,
        f"NAV Freshness: {nav_status}",
        f"Drawdown: {dd_pct:.2f}%",
        f"Positions: {positions_info['count']}",
        f"Firestore: {fs_status}",
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


__all__ = ["collect_doctor_snapshot", "collect_router_health", "run_doctor_subprocess"]
