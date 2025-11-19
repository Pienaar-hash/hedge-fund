from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from copy import deepcopy

from execution.firestore_utils import (
    get_db as _firestore_get_db,
    publish_heartbeat as _firestore_publish_heartbeat,
)
from execution.v6_flags import log_v6_flag_snapshot

# --- Ensure repo root is importable & files are read from repo root ---
# /root/hedge-fund/execution/sync_state.py -> repo_root=/root/hedge-fund
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
print(f"[sync] PYTHONPATH bootstrapped: {REPO_ROOT}", flush=True)

LOGGER = logging.getLogger("sync_state")
if not LOGGER.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)
_ENV_DEFAULT = "prod"
_ENV = (os.environ.get("HEDGE_ENV") or os.environ.get("ENV") or _ENV_DEFAULT).strip() or _ENV_DEFAULT
os.environ["ENV"] = _ENV
print(f"[sync] ENV resolved: {_ENV}", flush=True)


def _dry_run_mode() -> str:
    """Return DRY_RUN flag as string (default 0) for consistent logging."""
    return os.getenv("DRY_RUN", "0")


def _firestore_enabled() -> bool:
    flag = os.getenv("FIRESTORE_ENABLED")
    if flag is not None:
        return flag.strip().lower() in {"1", "true", "yes", "on"}
    env = _ENV.lower()
    if env == "prod":
        allow = os.getenv("ALLOW_PROD_SYNC") or os.getenv("ALLOW_PROD_WRITE")
        if allow is None:
            return False
        return allow.strip().lower() in {"1", "true", "yes", "on"}
    return True


def _repo_root_log() -> None:
    """Log repo root resolution for clarity in supervisor output."""
    print(f"[sync] repo_root resolved: {REPO_ROOT}", flush=True)


def _resolve_firestore_creds() -> str | None:
    return None


def safe_publish_health(service: str = "sync_state", status: str = "ok", err: str = "") -> None:
    return None


def _log_startup_summary() -> Dict[str, Any]:
    testnet = os.getenv("BINANCE_TESTNET", "0").lower() in ("1", "true", "yes")
    env = _get_env()
    dry_run = _dry_run_mode()
    base = "https://testnet.binancefuture.com" if testnet else "https://fapi.binance.com"
    fs_enabled = _firestore_enabled()
    prefix = "testnet" if testnet else "live"
    print(
        f"[{prefix}] ENV={env} DRY_RUN={dry_run} testnet={testnet} base={base} FIRESTORE={'ON' if fs_enabled else 'OFF'}",
        flush=True,
    )
    return {
        "testnet": testnet,
        "env": env,
        "dry_run": dry_run,
        "base": base,
        "fs_enabled": fs_enabled,
        "prefix": prefix,
    }


def _publish_startup_heartbeat(flags: Dict[str, Any]) -> None:
    print(f"[{flags.get('prefix', 'live')}] Firestore heartbeat skipped (disabled)", flush=True)

# --- Force package import path when launched via Supervisor ---
import importlib.util  # noqa: E402

class _NoopDoc:
    exists = False

    def to_dict(self):
        return {}


class _NoopFirestore:
    _is_noop = True

    def collection(self, *_args, **_kwargs):
        return self

    def document(self, *_args, **_kwargs):
        return self

    def set(self, *_args, **_kwargs):
        return None

    def get(self, *_args, **_kwargs):
        return _NoopDoc()


def _noop_db() -> _NoopFirestore:
    return _NoopFirestore()

# Make relative file reads (nav_log.json, etc.) deterministic under Supervisor
try:
    os.chdir(REPO_ROOT)
except Exception:
    pass

# execution/sync_state.py — Phase‑4.1 “Stability & Signals” (hardened sync)
#
# What this does
#  - Reads local files (nav_log.json, peak_state.json, synced_state.json)
#  - Applies cutoff filtering to NAV history if configured
#  - Guards against zero-equity rows and empty tails
#  - Computes exposure KPIs from positions
#  - Derives peak from best available source (file, rows, cached doc)
#  - Persists compact state locally for dashboard/cache consumers (no Firestore writes)
#
# Env knobs
#  ENV=prod|dev
#  NAV_CUTOFF_ISO="2025-08-01T00:00:00+00:00"   # preferred explicit cutoff
#  NAV_CUTOFF_SECAGO=86400                       # or relative cutoff in seconds
#  SYNC_INTERVAL_SEC=20
#
from datetime import datetime, timedelta, timezone  # noqa: E402
try:  # Python 3.9+
    from zoneinfo import ZoneInfo  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ZoneInfo = None  # type: ignore
from typing import Any, Callable, Dict, List, Optional, Tuple  # noqa: E402

from execution.drawdown_tracker import (  # noqa: E402
    compute_intraday_drawdown,
    load_peak_state,
    mirror_peak_state_to_firestore,
    save_peak_state,
)
from execution.exchange_utils import get_income_history  # noqa: E402

# ---------------- Firestore helpers (imported via guarded loader above) -----

# ------------------------------- Files ---------------------------------------
LOGS_DIR = os.path.join(REPO_ROOT, "logs")
NAV_LOG: str = os.path.join(LOGS_DIR, "nav_log.json")
PEAK_STATE: str = os.path.join(LOGS_DIR, "cache", "peak_state.json")
SYNCED_STATE: str = os.path.join(LOGS_DIR, "state", "synced_state.json")
SPOT_CACHE_PATH: str = os.path.join(LOGS_DIR, "spot_state.json")
TREASURY_CACHE_PATH: str = os.path.join(LOGS_DIR, "cache", "treasury_sync.json")
POLYMARKET_CACHE_PATH: str = os.path.join(LOGS_DIR, "polymarket.json")
print(f"[sync] file paths => NAV_LOG={NAV_LOG}", flush=True)

# ------------------------------ Settings ------------------------------------
MAX_POINTS: int = 500  # dashboard series cap

_FIRESTORE_FAIL_COUNT = 0
_FAILURE_THRESHOLD = 5
_LAST_SUCCESS_TS: Optional[float] = None

_STABLE_ASSETS = {"USDT", "USDC", "BUSD", "FDUSD", "TUSD", "USDP"}


# ------------------------------- Utilities ----------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_float(value: Any) -> float:
    try:
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return 0.0
        return float(value)
    except Exception:
        return 0.0


def _load_existing_cache(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list):
            return {"items": payload}
    except Exception:
        pass
    return {}


def _cache_asset_lookup(cache: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    items = cache.get("assets")
    if isinstance(items, list):
        for entry in items:
            if not isinstance(entry, dict):
                continue
            sym = str(
                entry.get("asset")
                or entry.get("symbol")
                or entry.get("Asset")
                or entry.get("code")
                or ""
            ).upper()
            if sym:
                lookup[sym] = entry
    for key, value in cache.items():
        if key in {"assets", "total_usd", "updated_at", "source"}:
            continue
        if isinstance(value, dict):
            lookup[str(key).upper()] = value
    return lookup


def _safe_usd_price(asset: str) -> float:
    sym = str(asset or "").upper()
    if not sym:
        return 0.0
    if sym in _STABLE_ASSETS:
        return 1.0
    try:
        from execution.exchange_utils import get_price  # type: ignore

        price_symbol = f"{sym}USDT"
        px = get_price(price_symbol)
        return float(px or 0.0)
    except Exception:
        return 0.0


def _price_from_nav_log(symbol: str) -> float:
    sym = str(symbol or "").upper()
    if not sym or not os.path.exists(NAV_LOG):
        return 0.0
    try:
        with open(NAV_LOG, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return 0.0

    candidates: List[Any]
    if isinstance(data, list):
        candidates = list(reversed(data[-500:]))  # tail-first, cap for perf
    else:
        candidates = [data]

    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        for key in (
            "prices",
            "price_map",
            "spot_prices",
            "spot",
            "px",
            "marks",
        ):
            bucket = entry.get(key)
            if isinstance(bucket, dict):
                for search in (sym, f"{sym}USDT"):
                    if search in bucket:
                        return _to_float(bucket.get(search))
        for key, value in entry.items():
            key_u = str(key).upper()
            if key_u in (sym, f"{sym}USDT"):
                return _to_float(value)
    return 0.0


def _convert_to_usd(symbol: str, qty: float) -> float:
    sym = str(symbol or "").upper()
    amount = _to_float(qty)
    if not sym or amount <= 0:
        return 0.0

    price = _safe_usd_price(sym)
    if price > 0:
        print(f"[sync] price resolve {sym}: {price}", flush=True)
        return amount * price

    price = _price_from_nav_log(sym)
    if price > 0:
        print(f"[sync] price resolve {sym}: {price}", flush=True)
        return amount * price

    return 0.0


def _infer_price(asset: str, quantity: float, previous: Optional[Dict[str, Any]]) -> float:
    sym = str(asset or "").upper()
    if quantity <= 0:
        return 0.0
    price = _safe_usd_price(sym)
    if price > 0:
        return price
    if previous:
        prev_val = _to_float(
            previous.get("value_usd")
            or previous.get("USD Value")
            or previous.get("usd")
        )
        prev_qty = _to_float(
            previous.get("balance")
            or previous.get("qty")
            or previous.get("Units")
            or previous.get("free")
        )
        if prev_qty > 0 and prev_val > 0:
            try:
                return float(prev_val / prev_qty)
            except Exception:
                return 0.0
    return 0.0


def _write_json_cache(
    path: str,
    data: Dict[str, Any],
    *,
    label: Optional[str] = None,
    total_note: Optional[float] = None,
) -> None:
    directory = os.path.dirname(path)
    if directory:
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception:
            pass
    tmp_path = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(
                data,
                handle,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        os.replace(tmp_path, path)
        if label:
            suffix = ""
            if total_note is not None:
                suffix = f": total_usd={total_note:.2f}"
            print(f"[sync] cache written ({label}){suffix}", flush=True)
        else:
            print(f"[sync] cache written: {os.path.basename(path)}", flush=True)
    except Exception as exc:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        print(
            f"[sync] WARN cache_write_failed: {os.path.basename(path)} err={exc}",
            flush=True,
        )


def _collect_spot_cache(previous: Dict[str, Any]) -> Dict[str, Any]:
    updated_at = _now_iso()
    assets: List[Dict[str, Any]] = []
    total_usd = 0.0
    prev_lookup = _cache_asset_lookup(previous)

    raw: Any = None
    try:
        import execution.exchange_utils as exu  # type: ignore

        for attr in (
            "get_spot_balances",
            "get_spot_snapshot",
            "get_spot_account",
            "get_spot_account_balances",
        ):
            getter = getattr(exu, attr, None)
            if not callable(getter):
                continue
            candidate = getter()
            if candidate:
                raw = candidate
                break
    except Exception as exc:
        print(f"[sync] WARN spot_cache_fetch_failed: {exc}", flush=True)

    if raw is None:
        alt_path = os.path.join(LOGS_DIR, "spot_snapshot.json")
        if os.path.exists(alt_path):
            raw = _load_existing_cache(alt_path)

    entries: List[Dict[str, Any]] = []
    if isinstance(raw, dict):
        balances = raw.get("balances")
        if isinstance(balances, list):
            entries = [entry for entry in balances if isinstance(entry, dict)]
        elif isinstance(balances, dict):
            entries = [
                {"asset": key, "free": balances.get(key)}
                for key in balances
                if balances.get(key) is not None
            ]
        elif isinstance(raw.get("assets"), list):
            entries = [entry for entry in raw["assets"] if isinstance(entry, dict)]
        else:
            entries = [
                {"asset": key, "free": raw[key]}
                for key in raw
                if isinstance(raw.get(key), (int, float, str))
            ]
    elif isinstance(raw, list):
        entries = [entry for entry in raw if isinstance(entry, dict)]

    for entry in entries:
        asset_code = str(
            entry.get("asset")
            or entry.get("symbol")
            or entry.get("coin")
            or ""
        ).upper()
        if not asset_code:
            continue
        free_amt = _to_float(
            entry.get("free")
            or entry.get("available")
            or entry.get("availableBalance")
            or entry.get("balance")
            or entry.get("qty")
            or entry.get("units")
        )
        if free_amt <= 0:
            continue
        locked_amt = _to_float(entry.get("locked") or entry.get("freeze"))
        prev_entry = prev_lookup.get(asset_code)
        usd_val = _to_float(
            entry.get("usdValue")
            or entry.get("usd")
            or entry.get("value_usd")
        )
        price = 0.0
        if usd_val > 0 and free_amt > 0:
            try:
                price = float(usd_val / free_amt)
            except Exception:
                price = 0.0
        if price <= 0:
            price = _infer_price(asset_code, free_amt, prev_entry)
        if usd_val <= 0 and price > 0:
            usd_val = free_amt * price
        if usd_val <= 0 and prev_entry:
            usd_val = _to_float(
                prev_entry.get("value_usd")
                or prev_entry.get("USD Value")
                or prev_entry.get("usd")
            )
        assets.append(
            {
                "asset": asset_code,
                "free": float(free_amt),
                "locked": float(max(locked_amt, 0.0)),
                "price": float(max(price, 0.0)),
                "value_usd": float(max(usd_val, 0.0)),
            }
        )
        total_usd += max(usd_val, 0.0)

    if not assets and previous:
        assets = deepcopy(previous.get("assets") or [])
        total_usd = _to_float(previous.get("total_usd"))

    if total_usd <= 0 and assets:
        total_usd = sum(_to_float(item.get("value_usd")) for item in assets)

    payload = {
        "source": "binance_spot",
        "assets": assets,
        "total_usd": float(total_usd),
        "updated_at": updated_at,
    }
    return payload


def _collect_treasury_cache(previous: Dict[str, Any]) -> Dict[str, Any]:
    updated_at = _now_iso()
    prev_lookup = _cache_asset_lookup(previous)

    base_qty: Dict[str, float] = {}
    asset_order: List[str] = []

    def _register_qty(sym: str, qty: float) -> None:
        sym_up = str(sym or "").upper()
        if not sym_up:
            return
        quantity = _to_float(qty)
        if quantity < 0:
            return
        base_qty[sym_up] = quantity
        if sym_up not in asset_order:
            asset_order.append(sym_up)

    for sym, entry in prev_lookup.items():
        qty = _to_float(
            entry.get("qty")
            or entry.get("balance")
            or entry.get("Units")
            or entry.get("units")
            or entry.get("free")
        )
        if qty > 0:
            _register_qty(sym, qty)

    for cfg_name in ("treasury.json", "reserves.json"):
        cfg_path = os.path.join(REPO_ROOT, "config", cfg_name)
        try:
            with open(cfg_path, "r", encoding="utf-8") as handle:
                cfg_payload = json.load(handle) or {}
        except Exception:
            cfg_payload = {}
        if isinstance(cfg_payload, dict):
            for raw_asset, raw_qty in cfg_payload.items():
                _register_qty(raw_asset, _to_float(raw_qty))

    if not asset_order:
        return {
            "assets": [],
            "total_usd": 0.0,
            "updated_at": updated_at,
        }

    assets_payload: List[Dict[str, Any]] = []
    total_usd = 0.0

    for sym in asset_order:
        qty = base_qty.get(sym, 0.0)
        prev_entry = prev_lookup.get(sym, {})

        if sym in _STABLE_ASSETS:
            price = 1.0
            usd_val = qty * price
        else:
            usd_val = _convert_to_usd(sym, qty)
            if usd_val <= 0 and prev_entry:
                usd_val = _to_float(
                    prev_entry.get("value_usd")
                    or prev_entry.get("usd_value")
                    or prev_entry.get("USD Value")
                    or prev_entry.get("usd")
                )
            price = 0.0
            if qty > 0 and usd_val > 0:
                try:
                    price = float(usd_val / qty)
                except Exception:
                    price = 0.0
            if price <= 0:
                price = _infer_price(sym, qty, prev_entry)
                if price > 0 and usd_val <= 0:
                    usd_val = qty * price

        if price <= 0:
            price = 0.0
            usd_val = 0.0 if sym not in _STABLE_ASSETS else qty * price

        usd_val = max(usd_val, 0.0)
        total_usd += usd_val

        assets_payload.append(
            {
                "asset": sym,
                "balance": float(qty),
                "price_usdt": float(price),
                "usd_value": float(usd_val),
            }
        )

    return {
        "assets": assets_payload,
        "total_usd": float(total_usd),
        "updated_at": updated_at,
    }


def _collect_polymarket_cache(previous: Dict[str, Any]) -> Dict[str, Any]:
    updated_at = _now_iso()
    positions: List[Dict[str, Any]] = []
    total_usd = 0.0

    try:
        from scripts.polymarket_insiders import get_polymarket_snapshot  # type: ignore

        snapshot = get_polymarket_snapshot()
        if isinstance(snapshot, list) and snapshot:
            for entry in snapshot:
                if not isinstance(entry, dict):
                    continue
                positions.append(entry)
                total_usd += _to_float(
                    entry.get("total_usd")
                    or entry.get("totalUsd")
                    or entry.get("usd")
                    or entry.get("sizeUsd")
                    or entry.get("notional_usd")
                )
    except Exception as exc:
        print(f"[sync] WARN polymarket_snapshot_unavailable: {exc}", flush=True)

    alt_path = os.path.join(LOGS_DIR, "polymarket_snapshot.json")
    if not positions and os.path.exists(alt_path):
        existing = _load_existing_cache(alt_path)
        if isinstance(existing, dict):
            maybe_positions = existing.get("positions") or existing.get("entries")
            if isinstance(maybe_positions, list):
                positions = [
                    entry for entry in maybe_positions if isinstance(entry, dict)
                ]
            total_usd = _to_float(
                existing.get("total_usd")
                or existing.get("usd")
                or existing.get("notional")
            )
        elif isinstance(existing, list):
            positions = [entry for entry in existing if isinstance(entry, dict)]

    if total_usd <= 0 and previous:
        if not positions:
            maybe_prev_positions = previous.get("positions") or previous.get("entries")
            if isinstance(maybe_prev_positions, list):
                positions = deepcopy(
                    [entry for entry in maybe_prev_positions if isinstance(entry, dict)]
                )
        total_usd = _to_float(
            previous.get("total_usd")
            or previous.get("USD")
            or previous.get("usd")
        )

    payload = {
        "source": "polymarket",
        "positions": positions,
        "total_usd": float(total_usd),
        "updated_at": updated_at,
    }
    return payload


def _export_dashboard_caches() -> None:
    previous_spot = _load_existing_cache(SPOT_CACHE_PATH)
    previous_treasury = _load_existing_cache(TREASURY_CACHE_PATH)
    previous_poly = _load_existing_cache(POLYMARKET_CACHE_PATH)

    tasks: List[
        Tuple[str, Dict[str, Any], Callable[[Dict[str, Any]], Dict[str, Any]]]
    ] = [
        (SPOT_CACHE_PATH, previous_spot, _collect_spot_cache),
        (TREASURY_CACHE_PATH, previous_treasury, _collect_treasury_cache),
        (POLYMARKET_CACHE_PATH, previous_poly, _collect_polymarket_cache),
    ]

    for path, previous, builder in tasks:
        try:
            payload = builder(previous)
        except Exception as exc:
            print(
                f"[sync] WARN cache_build_failed: {os.path.basename(path)} err={exc}",
                flush=True,
            )
            payload = {
                "total_usd": 0.0,
                "assets": [],
                "updated_at": _now_iso(),
            }
        if "updated_at" not in payload:
            payload["updated_at"] = _now_iso()
        if "total_usd" not in payload:
            payload["total_usd"] = 0.0
        if os.path.basename(path) == "spot_state.json" and "assets" not in payload:
            payload["assets"] = []
        if os.path.basename(path) == "treasury.json" and "assets" not in payload:
            payload["assets"] = []
        if os.path.abspath(path) == os.path.abspath(TREASURY_CACHE_PATH):
            total_note = _to_float(payload.get("total_usd"))
            _write_json_cache(path, payload, label="treasury", total_note=total_note)
        else:
            _write_json_cache(path, payload)


def _parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _cutoff_dt() -> Optional[datetime]:
    """Cutoff datetime from env: NAV_CUTOFF_ISO (preferred) or NAV_CUTOFF_SECAGO."""
    iso = os.getenv("NAV_CUTOFF_ISO")
    if iso:
        dt = _parse_iso(iso)
        if dt:
            return dt
    sec = os.getenv("NAV_CUTOFF_SECAGO")
    if sec:
        try:
            s = int(sec)
            return datetime.now(timezone.utc) - timedelta(seconds=s)
        except Exception:
            pass
    return None


def _resolve_equity_value(payload: Dict[str, Any]) -> float:
    """Best-effort float conversion for equity-like keys."""
    for key in ("equity", "total_equity", "nav"):
        if key in payload:
            try:
                return float(payload[key])
            except Exception:
                continue
    return 0.0


# ----------------------------- File readers ---------------------------------


def _read_nav_rows(path: str) -> List[Dict[str, Any]]:
    """Read nav_log.json list and normalize timestamps to key 't'."""
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            cut = _cutoff_dt()
            for p in data:
                item = dict(p)
                # normalize timestamp key for series
                ts = item.get("timestamp")
                if ts is None:
                    ts = item.get("t")
                if isinstance(ts, (int, float)):
                    item["t"] = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                    print("[sync] normalized numeric timestamp -> ISO", flush=True)
                elif ts and "t" not in item:
                    item["t"] = ts
                equity = _resolve_equity_value(item)
                if equity > 0.0:
                    item.setdefault("equity", equity)
                    item.setdefault("total_equity", equity)
                # enforce cutoff if configured
                if cut:
                    dt = _parse_iso(str(item.get("t") or ""))
                    if dt and dt < cut:
                        continue
                rows.append(item)
    except Exception:
        pass
    return rows


def _read_peak_file(_path: str) -> float:
    state = load_peak_state()
    try:
        return float(state.get("peak_equity") or state.get("peak") or 0.0)
    except Exception:
        return 0.0


def _read_positions_snapshot(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"items": [], "updated_at": _now_iso()}
    try:
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f) or {}
        items = j.get("items") or []
        return {"items": items, "updated_at": j.get("updated_at") or _now_iso()}
    except Exception:
        return {"items": [], "updated_at": _now_iso()}


# --- KPI tail reader for nav card metrics
def _read_nav_tail_metrics(path: str) -> Dict[str, float]:
    """Return last point's metrics for nav KPIs (safe defaults)."""
    out = {
        "total_equity": 0.0,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "drawdown": 0.0,
    }
    if not os.path.exists(path):
        print(f"[sync] tail equity resolved: {out['total_equity']}", flush=True)
        return out
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            last = data[-1]
            candidate_maps: List[Dict[str, Any]] = []
            top_level_map: Optional[Dict[str, Any]] = None
            if isinstance(last, dict):
                last_map = dict(last)
                top_level_map = last_map
                data_block = last_map.get("data")
                if isinstance(data_block, dict):
                    candidate_maps.append(dict(data_block))
                candidate_maps.append(last_map)

            equity_value = 0.0
            for mp in candidate_maps:
                val = _resolve_equity_value(mp)
                if val > 0.0:
                    equity_value = val
                    if top_level_map is not None and mp is top_level_map:
                        print(
                            f"[sync] tail equity from top-level map: {equity_value}",
                            flush=True,
                        )
                    break
            if equity_value <= 0.0 and top_level_map is not None:
                equity_value = _resolve_equity_value(top_level_map)
                if equity_value > 0.0:
                    print(
                        f"[sync] tail equity from top-level map: {equity_value}",
                        flush=True,
                    )
            out["total_equity"] = equity_value

            def _resolve_metric(keys: Tuple[str, ...]) -> float:
                for mp in candidate_maps:
                    for key in keys:
                        if key in mp:
                            try:
                                return float(mp[key] or 0.0)
                            except Exception:
                                continue
                return 0.0

            out["realized_pnl"] = _resolve_metric(("realized", "realized_pnl"))
            out["unrealized_pnl"] = _resolve_metric(("unrealized", "unrealized_pnl"))
            out["drawdown"] = _resolve_metric(("drawdown_pct", "drawdown"))
    except Exception:
        pass
    print(f"[sync] tail equity resolved: {out['total_equity']}", flush=True)
    return out


def _load_risk_limits_config() -> Dict[str, Any]:
    path = os.path.join(REPO_ROOT, "config", "risk_limits.json")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _reset_timezone_name() -> str:
    cfg = _load_risk_limits_config()
    candidates: List[Any] = []
    global_cfg = cfg.get("global")
    if isinstance(global_cfg, dict):
        candidates.append(global_cfg.get("reset_timezone"))
    candidates.append(cfg.get("reset_timezone"))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return "UTC"


def _resolve_zone(name: str) -> timezone | ZoneInfo:
    if not name or str(name).upper() == "UTC":
        return timezone.utc
    if ZoneInfo is None:
        return timezone.utc
    try:
        return ZoneInfo(str(name))
    except Exception:
        return timezone.utc


def _fetch_realized_pnl_today(reset_tz: str) -> float:
    zone = _resolve_zone(reset_tz)
    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(zone)
    start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    start_utc = start_local.astimezone(timezone.utc)
    start_ms = int(start_utc.timestamp() * 1000)
    end_ms = int(now_utc.timestamp() * 1000)

    total = 0.0
    next_start = start_ms
    try:
        while True:
            rows = get_income_history(
                next_start,
                end_time_ms=end_ms,
                income_type="REALIZED_PNL",
                limit=1000,
            )
            if not rows:
                break
            for row in rows:
                total += _to_float(row.get("income"))
            if len(rows) < 1000:
                break
            last_time = max(_to_float(row.get("time") or row.get("T") or 0.0) for row in rows)
            if last_time <= 0.0:
                break
            next_start = int(last_time) + 1
            if next_start > end_ms:
                break
    except Exception as exc:
        print(f"[sync] WARN realized_pnl_fetch_failed: {exc}", flush=True)
        cached = load_peak_state()
        return _to_float(cached.get("realized_pnl_today"))
    return total


# --------------------------- Filtering / metrics -----------------------------


def _is_good_nav_row(p: Dict[str, Any]) -> bool:
    if p.get("heartbeat_reason") == "exchange_unhealthy":
        return False
    return _resolve_equity_value(p) > 0.0


def _compute_peak_from_rows(rows: List[Dict[str, Any]]) -> float:
    try:
        return max((float(p.get("equity") or 0.0) for p in rows), default=0.0)
    except Exception:
        return 0.0


# --------------------------- Position normalization -------------------------


def _normalize_positions_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    now = _now_iso()
    for it in items or []:
        try:
            qty = float(it.get("qty", 0.0))
            entry = float(it.get("entry", it.get("entry_price", 0.0)))
            mark = float(
                it.get("mark_price", it.get("mark", it.get("latest_price", 0.0)))
            )
            pnl = (mark - entry) * qty  # recompute; don't trust inbound 'pnl'
            side = it.get("side") or ("LONG" if qty >= 0 else "SHORT")
            out.append(
                {
                    "symbol": it.get("symbol"),
                    "side": side,
                    "qty": qty,
                    "entry_price": entry,
                    "mark_price": mark,
                    "pnl": pnl,
                    "leverage": int(float(it.get("leverage", 1))),
                    "notional": abs(qty) * mark,
                    "ts": it.get("updated_at", now),
                }
            )
        except Exception:
            continue
    return out


# ------------------------------ Exposure KPIs --------------------------------


def _exposure_from_positions(items: List[Dict[str, Any]]) -> Dict[str, float]:
    gross = net = 0.0
    largest = 0.0
    for it in items or []:
        try:
            qty = float(it.get("qty", 0.0))
            price = float(
                it.get("mark_price", it.get("mark", it.get("latest_price", 0.0)))
            )
            pv = abs(qty) * price
            gross += pv
            net += qty * price
            largest = max(largest, pv)
        except Exception:
            continue
    return {
        "gross_exposure": gross,
        "net_exposure": net,
        "largest_position_value": largest,
    }


# ----------------------------- Firestore helpers ----------------------------


def _get_env() -> str:
    global _ENV
    env = (
        os.environ.get("HEDGE_ENV")
        or os.environ.get("ENV")
        or _ENV_DEFAULT
    )
    env = (env or _ENV_DEFAULT).strip() or _ENV_DEFAULT
    if env != _ENV:
        _ENV = env
        os.environ["ENV"] = _ENV
    return _ENV


def _nav_doc_ref(db):
    return (
        db.collection("hedge").document(_get_env()).collection("state").document("nav")
    )


def _pos_doc_ref(db):
    return (
        db.collection("hedge")
        .document(_get_env())
        .collection("state")
        .document("positions")
    )


def _lb_doc_ref(db):
    return (
        db.collection("hedge")
        .document(_get_env())
        .collection("state")
        .document("leaderboard")
    )


def _sync_state_doc_ref(db):
    return (
        db.collection("hedge")
        .document(_get_env())
        .collection("state")
        .document("sync_state")
    )


def _maybe_fetch_nav_doc(db) -> Dict[str, Any]:
    try:
        snap = _nav_doc_ref(db).get()
        if snap and snap.exists:
            return snap.to_dict() or {}
    except Exception:
        pass
    return {}


def _commit_nav(db, rows: List[Dict[str, Any]], peak: float) -> Dict[str, Any]:
    tail = _read_nav_tail_metrics(NAV_LOG)  # include KPIs for dashboard cards
    if not rows:
        payload = {
            "series": [],
            "peak_equity": float(peak),
            "updated_at": _now_iso(),
            **tail,
        }
    else:
        slim = rows[-MAX_POINTS:]
        payload = {
            "series": slim,
            "peak_equity": float(peak),
            "updated_at": slim[-1].get("t", _now_iso()),
            **tail,
        }
    return payload


def _commit_positions(db, positions: Dict[str, Any]) -> Dict[str, Any]:
    items = positions.get("items") or []
    norm = _normalize_positions_items(items)  # normalize to dashboard schema
    exp = _exposure_from_positions(norm)
    payload = {"items": norm, "updated_at": _now_iso(), **exp}
    return payload


def _commit_leaderboard(db, positions: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal leaderboard: aggregate notional and pnl by symbol."""
    items = positions.get("items") or []
    agg: Dict[str, Dict[str, float]] = {}
    for it in items:
        sym = it.get("symbol") or "UNKNOWN"
        agg.setdefault(sym, {"notional": 0.0, "pnl": 0.0})
        try:
            agg[sym]["notional"] += float(it.get("notional", 0.0))
            agg[sym]["pnl"] += float(it.get("pnl", 0.0))
        except Exception:
            pass
    leaderboard = sorted(
        [
            {"symbol": k, "notional": v["notional"], "pnl": v["pnl"]}
            for k, v in agg.items()
        ],
        key=lambda r: r["notional"],
        reverse=True,
    )
    payload = {"items": leaderboard, "updated_at": _now_iso()}
    return payload


def _publish_health(db, ok: bool, last_error: str) -> None:
    return None


# ------------------------------ Public API ----------------------------------


def _sync_once_with_db(db) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    print("[sync] executing _sync_once_with_db", flush=True)
    # NAV rows with cutoff + zero-equity filter
    rows = [p for p in _read_nav_rows(NAV_LOG) if _is_good_nav_row(p)]

    # Tail equity guard: avoid writing garbage when executor/sync is cold
    tail_kpis = _read_nav_tail_metrics(NAV_LOG)
    nav_total_equity = float(tail_kpis.get("total_equity", 0.0) or 0.0)
    if nav_total_equity <= 0.0:
        print(
            f"[sync] skipping write because nav tail total_equity <= 0 "
            f"(value={tail_kpis.get('total_equity')})",
            flush=True,
        )
        # still log positions exposure to nav doc (merge), but skip full write
        return {}, {}, {}

    # Peak: choose the best available source in this order:
    #  1) existing Firestore nav doc (peak_equity)
    #  2) file peak_state.json
    #  3) computed from filtered rows
    nav_existing = _maybe_fetch_nav_doc(db)
    peak_doc = 0.0
    try:
        peak_doc = float(nav_existing.get("peak_equity") or 0.0)
    except Exception:
        pass
    peak_file = _read_peak_file(PEAK_STATE)
    peak_rows = _compute_peak_from_rows(rows)

    # If we're filtering history (cutoff active), prefer rows-based peak (local regime)
    cutoff_active = bool(os.getenv("NAV_CUTOFF_ISO") or os.getenv("NAV_CUTOFF_SECAGO"))
    if cutoff_active:
        peak = (
            max(peak_rows, peak_doc, peak_file)
            if peak_rows > 0
            else max(peak_doc, peak_file)
        )
    else:
        peak = max(peak_doc, peak_file, peak_rows)

    # Positions snapshot (executor populates synced_state.json)
    pos_snap = _read_positions_snapshot(SYNCED_STATE)

    # Firestore upserts
    nav_payload = _commit_nav(db, rows, peak)

    # attach exposure locally for prints and persist to the same doc
    exposure = _exposure_from_positions(pos_snap.get("items") or [])
    if exposure:
        nav_payload.update(exposure)

    nav_for_drawdown = nav_total_equity
    if nav_for_drawdown <= 0.0:
        nav_for_drawdown = _to_float(nav_payload.get("total_equity"))
    reset_tz = _reset_timezone_name()
    realized_today = _fetch_realized_pnl_today(reset_tz)
    drawdown_state = compute_intraday_drawdown(
        nav_for_drawdown,
        realized_today,
        reset_timezone=reset_tz,
    )
    save_peak_state(drawdown_state)
    if db is not None and not getattr(db, "_is_noop", False):
        mirror_peak_state_to_firestore(drawdown_state, db, env=_get_env())
    dd_payload = {
        "drawdown": drawdown_state.get("dd_pct", 0.0),
        "drawdown_pct": drawdown_state.get("dd_pct", 0.0),
        "drawdown_abs": drawdown_state.get("dd_abs", 0.0),
        "peak_equity": drawdown_state.get("peak", nav_payload.get("peak_equity")),
        "realized_pnl": realized_today,
        "realized_pnl_today": realized_today,
        "drawdown_snapshot_at": drawdown_state.get("updated_at"),
    }
    nav_payload.update(dd_payload)
    print(
        "[sync] drawdown tracker "
        f"peak={drawdown_state.get('peak', 0.0):.2f} "
        f"dd_pct={drawdown_state.get('dd_pct', 0.0):.2f} "
        f"realized={realized_today:.2f}",
        flush=True,
    )

    pos_payload = _commit_positions(db, pos_snap)
    lb_payload = _commit_leaderboard(db, pos_snap)

    # Console log
    try:
        updated_at = nav_payload.get("updated_at")
        print(
            f"[sync] upsert ok: points={len(nav_payload.get('series') or [])} "
            f"peak={nav_payload.get('peak_equity')} at={updated_at}",
            flush=True,
        )
    except Exception:
        pass

    _export_dashboard_caches()

    return nav_payload, pos_payload, lb_payload


def sync_once() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Read files -> filter/compute using local data only.
    Returns (nav_payload, positions_payload, leaderboard_payload).
    """
    global _FIRESTORE_FAIL_COUNT, _LAST_SUCCESS_TS
    try:
        if _firestore_enabled():
            db = _firestore_get_db(strict=False)
        else:
            db = _noop_db()
        result = _sync_once_with_db(db)
        _LAST_SUCCESS_TS = time.time()
        _FIRESTORE_FAIL_COUNT = 0
        return result
    except Exception as exc:
        _FIRESTORE_FAIL_COUNT += 1
        LOGGER.error("[sync] local sync failed count=%s err=%s", _FIRESTORE_FAIL_COUNT, exc)
        raise


# --------------------------------- Runner -----------------------------------


_HEARTBEAT_LOCK = threading.Lock()
_HEARTBEAT_THREAD: Optional[threading.Thread] = None
_HEARTBEAT_STOP: Optional[threading.Event] = None


def _interval_seconds() -> int:
    try:
        return int(os.getenv("SYNC_INTERVAL_SEC", "20"))
    except Exception:
        return 20


def _heartbeat_worker(env: str, stop_event: threading.Event) -> None:
    LOGGER.info("[heartbeat] thread starting (ENV=%s)", env)
    while not stop_event.is_set():
        try:
            _firestore_publish_heartbeat(service="sync_state", status="ok", env=env)
            LOGGER.info("[heartbeat] sync_state published ok")
        except Exception as exc:
            LOGGER.warning("[heartbeat] sync_state failed: %s", exc, exc_info=True)
        stop_event.wait(60)
    LOGGER.info("[heartbeat] thread stopping (ENV=%s)", env)


def _ensure_heartbeat_thread(env: str) -> None:
    global _HEARTBEAT_THREAD, _HEARTBEAT_STOP
    with _HEARTBEAT_LOCK:
        if not _firestore_enabled():
            return
        if _HEARTBEAT_STOP is not None and _HEARTBEAT_STOP.is_set():
            return
        if _HEARTBEAT_THREAD is not None and _HEARTBEAT_THREAD.is_alive():
            return
        stop_event = threading.Event()
        _HEARTBEAT_STOP = stop_event
        thread = threading.Thread(
            target=_heartbeat_worker,
            args=(env, stop_event),
            name="sync_state-heartbeat",
            daemon=True,
        )
        _HEARTBEAT_THREAD = thread
        thread.start()


def _current_heartbeat_stop_event() -> Optional[threading.Event]:
    with _HEARTBEAT_LOCK:
        return _HEARTBEAT_STOP


def _stop_heartbeat_thread() -> None:
    global _HEARTBEAT_THREAD, _HEARTBEAT_STOP
    with _HEARTBEAT_LOCK:
        stop_event = _HEARTBEAT_STOP
        thread = _HEARTBEAT_THREAD
    if stop_event is not None:
        stop_event.set()
    if thread is not None and thread.is_alive():
        thread.join(timeout=5.0)
    with _HEARTBEAT_LOCK:
        _HEARTBEAT_THREAD = None
        _HEARTBEAT_STOP = None


def main_loop() -> None:
    env = _get_env()
    dry_run = _dry_run_mode()
    interval_seconds = _interval_seconds()
    interval = max(float(interval_seconds), 1.0)
    print(
        f"[sync] entering main_loop (ENV={env}, DRY_RUN={dry_run}, interval={interval_seconds}s, repo_root={REPO_ROOT})",
        flush=True,
    )
    _ensure_heartbeat_thread(env)

    try:
        while True:
            start_ts = time.monotonic()
            try:
                sync_once()
            except Exception as exc:
                LOGGER.error("[sync_state] sync_once failed: %s", exc)
            _ensure_heartbeat_thread(env)
            elapsed = time.monotonic() - start_ts
            sleep_for = max(0.0, interval - elapsed)
            stop_event = _current_heartbeat_stop_event()
            if stop_event is not None:
                if stop_event.wait(timeout=sleep_for):
                    break
            elif sleep_for > 0.0:
                time.sleep(sleep_for)
    except KeyboardInterrupt:
        raise
    finally:
        _stop_heartbeat_thread()


def run() -> None:
    env = _get_env()
    os.environ["ENV"] = env
    _repo_root_log()
    dry_run = _dry_run_mode()
    if not _firestore_enabled():
        print("[sync] Firestore disabled -> running in local-only mode", flush=True)
    try:
        flags = _log_startup_summary()
    except Exception as exc:
        print(f"[live] startup summary logging failed: {exc}", flush=True)
        flags = {
            "env": env,
            "fs_enabled": _firestore_enabled(),
            "prefix": "live",
            "testnet": False,
            "dry_run": dry_run,
        }
    LOGGER.info(
        "[exutil] ENV context testnet=%s dry_run=%s",
        flags.get("testnet"),
        flags.get("dry_run"),
    )
    _ensure_heartbeat_thread(flags.get("env", "prod"))
    _publish_startup_heartbeat(flags)
    try:
        log_v6_flag_snapshot(LOGGER)
    except Exception:
        LOGGER.debug("v6 flag snapshot logging failed", exc_info=True)
    print(
        f"[sync] startup context: ENV={env}, DRY_RUN={dry_run}, repo_root={REPO_ROOT}",
        flush=True,
    )
    try:
        main_loop()
    except KeyboardInterrupt:
        print("[sync] shutdown requested via KeyboardInterrupt", flush=True)


if __name__ == "__main__":
    run()
