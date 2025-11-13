# execution/firestore_utils.py
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from google.cloud import firestore as _firestore_client  # type: ignore
except Exception:  # pragma: no cover
    _firestore_client = None  # type: ignore

from utils.firestore_client import get_db, write_doc

LOGGER = logging.getLogger("firestore")

_STABLE_ASSETS = {"USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDE"}
_DIRECT_CLIENT = None


def _env() -> str:
    return os.environ.get("ENV", os.environ.get("ENVIRONMENT", "prod"))


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> str:
    return os.environ.get("REPO_ROOT") or os.getcwd()


def _safe_load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _firestore_available(db: Any) -> bool:
    return db is not None and not getattr(db, "_is_noop", False)


def _direct_client() -> Any:
    global _DIRECT_CLIENT
    if _DIRECT_CLIENT is not None:
        return _DIRECT_CLIENT
    if _firestore_client is None:
        raise RuntimeError("google.cloud.firestore unavailable")
    _DIRECT_CLIENT = _firestore_client.Client()
    return _DIRECT_CLIENT


def publish_router_metrics(doc_id: str, payload: Dict[str, Any], *, env: Optional[str] = None) -> None:
    """
    Mirror router metrics into Firestore under hedge/{env}/router_metrics/{doc_id}.
    """
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")
    client = _direct_client()
    env_name = env or _env()
    data = dict(payload)
    data.setdefault("env", env_name)
    data.setdefault("updated_at", time.time())
    client.collection("hedge").document(env_name).collection("router_metrics").document(doc_id).set(data, merge=True)


def publish_symbol_toggle(symbol: str, meta: Dict[str, Any], *, env: Optional[str] = None) -> None:
    """
    Mirror symbol toggle state into Firestore under hedge/{env}/symbol_toggles/{symbol}.
    """
    if not symbol:
        raise ValueError("symbol is required")
    if not isinstance(meta, dict):
        raise ValueError("meta must be a dict")
    client = _direct_client()
    env_name = env or _env()
    doc_id = symbol.upper()
    data = dict(meta)
    data.setdefault("symbol", doc_id)
    data.setdefault("env", env_name)
    data.setdefault("updated_at", time.time())
    client.collection("hedge").document(env_name).collection("symbol_toggles").document(doc_id).set(data, merge=True)


def publish_execution_health(symbol: Optional[str], payload: Dict[str, Any], *, env: Optional[str] = None) -> None:
    """
    Mirror execution health snapshots for remote dashboards.
    """
    try:
        client = _direct_client()
    except Exception:  # pragma: no cover - optional dependency
        return

    env_name = env or os.getenv("HEDGE_ENV") or _env()
    ts = int(time.time())
    doc_id = f"{(symbol or 'ALL').upper()}_{ts}"
    data = dict(payload)
    data.setdefault("symbol", symbol)
    data.setdefault("env", env_name)
    data.setdefault("updated_at", ts)
    try:
        client.collection("hedge").document(env_name).collection("execution_health").document(doc_id).set(data, merge=True)
    except Exception as exc:  # pragma: no cover - telemetry best effort
        LOGGER.debug("[firestore] execution_health_publish_failed symbol=%s err=%s", symbol, exc)


def publish_execution_alert(alert: Dict[str, Any], *, env: Optional[str] = None) -> None:
    """
    Mirror execution alerts for dashboard consumption.
    """
    if not isinstance(alert, dict):
        raise ValueError("alert must be a dict")
    try:
        client = _direct_client()
    except Exception:  # pragma: no cover - optional dependency
        return

    env_name = env or os.getenv("HEDGE_ENV") or _env()
    ts = int(time.time())
    symbol = str(alert.get("symbol") or "ALL").upper()
    a_type = str(alert.get("type") or "generic")
    doc_id = f"{symbol}_{a_type}_{ts}"
    data = dict(alert)
    data.setdefault("symbol", symbol)
    data.setdefault("type", a_type)
    data.setdefault("env", env_name)
    data.setdefault("updated_at", ts)
    try:
        client.collection("hedge").document(env_name).collection("execution_alerts").document(doc_id).set(data, merge=True)
    except Exception as exc:  # pragma: no cover - telemetry best effort
        LOGGER.debug("[firestore] execution_alert_publish_failed symbol=%s err=%s", symbol, exc)


def fetch_symbol_toggles(*, env: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetch all symbol toggle docs for the requested environment.
    """
    client = _direct_client()
    env_name = env or _env()
    toggles: List[Dict[str, Any]] = []
    try:
        collection = client.collection("hedge").document(env_name).collection("symbol_toggles")
        for doc in collection.stream():
            data = doc.to_dict() or {}
            data.setdefault("symbol", str(doc.id).upper())
            toggles.append(data)
    except Exception as exc:  # pragma: no cover - network failures
        LOGGER.warning("[firestore] fetch_symbol_toggles_failed env=%s err=%s", env_name, exc)
    return toggles


def publish_health_if_needed(
    payload: Dict[str, Any],
    *,
    env: Optional[str] = None,
    service: Optional[str] = None,
    db: Optional[Any] = None,
) -> None:
    """
    Write a health payload to the new telemetry/health document while mirroring legacy paths.
    """
    body = dict(payload)
    env_name = env or body.get("env") or _env()
    service_name = service or body.get("service") or body.get("process") or "unknown"
    now_ts = time.time()
    now_iso = _utcnow_iso()
    body.setdefault("env", env_name)
    body.setdefault("service", service_name)
    body.setdefault("process", service_name)
    ts_val = body.get("ts")
    if not isinstance(ts_val, (int, float)):
        body["ts"] = now_ts
    if not body.get("ts_iso"):
        body["ts_iso"] = now_iso
    body.setdefault("updated_at", body.get("ts_iso", now_iso))
    body.setdefault("status", body.get("status") or "ok")

    db_handle = db or get_db(strict=False)
    if not _firestore_available(db_handle):
        raise RuntimeError("Firestore unavailable")

    telemetry_collection = db_handle.collection("hedge").document(env_name).collection("telemetry")

    try:
        telemetry_collection.document("health").set(body, merge=True)
    except Exception as exc:
        LOGGER.warning(
            "[firestore] telemetry health write failed env=%s service=%s err=%s",
            env_name,
            service_name,
            exc,
        )

    try:
        telemetry_collection.document("heartbeats").set(body, merge=True)
    except Exception as exc:
        LOGGER.debug(
            "[firestore] legacy heartbeat mirror failed env=%s service=%s err=%s",
            env_name,
            service_name,
            exc,
        )

    try:
        write_doc(db_handle, f"hedge/{env_name}/health/{service_name}", body, require=False)
    except Exception as exc:
        LOGGER.debug(
            "[firestore] legacy health doc mirror failed env=%s service=%s err=%s",
            env_name,
            service_name,
            exc,
        )


def _last_price_from_logs(asset: str) -> Optional[float]:
    asset_code = str(asset or "").upper()
    if not asset_code:
        return None
    try:
        from execution.exchange_utils import get_last_known_price

        price = get_last_known_price(asset_code)
        if price is not None and price > 0:
            return float(price)
    except Exception:
        return None
    return None


def _to_float(value: Any) -> float:
    try:
        if value in (None, "", "null"):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _to_int(value: Any) -> int:
    try:
        if value in (None, "", "null"):
            return 0
        return int(float(value))
    except Exception:
        return 0


def _extract_router_metrics(payload: Optional[Dict[str, Any]]) -> Dict[str, float]:
    metrics = {
        "trades": 0.0,
        "win_rate": 0.0,
        "avg_pnl": 0.0,
        "cum_pnl": 0.0,
        "fill_rate": 0.0,
        "lat_p50_ms": 0.0,
        "lat_p95_ms": 0.0,
        "slip_p50_bps": 0.0,
        "slip_p95_bps": 0.0,
    }
    if not isinstance(payload, dict):
        return metrics

    source = payload.get("metrics")
    if not isinstance(source, dict):
        source = payload

    intents = payload.get("intents")
    if isinstance(intents, dict):
        emitted = intents.get("emitted")
        if emitted is None:
            emitted = intents.get("executed")
        metrics["trades"] = float(_to_int(emitted))
        attempted = _to_int(intents.get("attempted"))
        executed = _to_int(intents.get("executed") or intents.get("emitted"))
        if attempted > 0:
            metrics["fill_rate"] = min(100.0, max(0.0, (executed / attempted) * 100.0))

    for key in ("trades", "win_rate", "avg_pnl", "avg_pnl_usd", "avg_pnl_usdt"):
        value = source.get(key)
        if value is not None:
            if key == "trades":
                metrics["trades"] = float(_to_int(value))
            elif key.startswith("avg_pnl"):
                metrics["avg_pnl"] = _to_float(value)
            elif key == "win_rate":
                metrics["win_rate"] = _to_float(value)

    cum_pnl = source.get("cum_pnl") or source.get("cumulative_pnl") or source.get("cum_pnl_usd")
    metrics["cum_pnl"] = _to_float(cum_pnl)

    fill_rate = source.get("fill_rate") or source.get("fill_rate_pct")
    if fill_rate is not None:
        metrics["fill_rate"] = _to_float(fill_rate)

    latency = source.get("latency_ms") or source.get("latency")
    if isinstance(latency, dict):
        p50 = latency.get("decision_p50") or latency.get("p50") or latency.get("latency_p50_ms")
        p95 = latency.get("decision_p95") or latency.get("p95") or latency.get("latency_p95_ms")
        metrics["lat_p50_ms"] = _to_float(p50)
        metrics["lat_p95_ms"] = _to_float(p95)
    else:
        metrics["lat_p50_ms"] = _to_float(source.get("lat_p50_ms"))
        metrics["lat_p95_ms"] = _to_float(source.get("lat_p95_ms"))

    slippage = source.get("slippage_bps") or source.get("slippage")
    if isinstance(slippage, dict):
        metrics["slip_p50_bps"] = _to_float(slippage.get("p50"))
        metrics["slip_p95_bps"] = _to_float(slippage.get("p95"))
    else:
        metrics["slip_p50_bps"] = _to_float(source.get("slip_p50_bps"))
        metrics["slip_p95_bps"] = _to_float(source.get("slip_p95_bps"))

    return metrics


def _format_treasury_asset(asset: str, balance: float, price: Optional[float], usd_value: float) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "asset": asset,
        "balance": float(balance),
        "usd_value": float(usd_value),
    }
    payload["price_usdt"] = float(price) if price is not None else None
    return payload


def _normalize_treasury_payload(payload: Optional[Dict[str, Any]], source: str) -> Dict[str, Any]:
    assets_acc: Dict[str, Dict[str, float]] = {}

    def ingest_node(node: Any, hint: Optional[str] = None) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                key_lower = str(key).lower()
                if key_lower in {"total_usd", "total_treasury_usdt", "updated_at", "ts"}:
                    continue
                if key_lower in {"assets", "treasury"}:
                    ingest_node(value)
                    continue
                ingest_entry(str(key), value)
        elif isinstance(node, list):
            for entry in node:
                ingest_entry(hint, entry)

    def ingest_entry(hint: Optional[str], entry: Any) -> None:
        asset_name = ""
        qty = 0.0
        price = 0.0
        usd_value = 0.0

        if isinstance(entry, dict):
            candidate = entry.get("asset") or entry.get("Asset") or entry.get("symbol") or entry.get("code")
            asset_name = str(hint or candidate or "").upper()
            qty = _to_float(entry.get("qty") or entry.get("Units") or entry.get("units") or entry.get("balance") or entry.get("amount"))
            price = _to_float(entry.get("px") or entry.get("price") or entry.get("price_usdt"))
            usd_value = _to_float(
                entry.get("val_usdt")
                or entry.get("USD Value")
                or entry.get("usd_value")
                or entry.get("value_usd")
                or entry.get("usd")
            )
        else:
            asset_name = str(hint or "").upper()
            qty = _to_float(entry)

        if not asset_name or asset_name in {"TOTAL_USD", "TOTAL_TREASURY_USDT"}:
            return

        acc = assets_acc.setdefault(asset_name, {"balance": 0.0, "price": 0.0, "usd_value": 0.0})
        if qty:
            acc["balance"] = float(qty)
        if price:
            acc["price"] = float(price)
        if usd_value:
            acc["usd_value"] = float(usd_value)

    if isinstance(payload, dict):
        candidates: List[Any] = []
        if isinstance(payload.get("assets"), (list, dict)):
            candidates.append(payload["assets"])
        if isinstance(payload.get("treasury"), dict):
            treas = payload["treasury"]
            candidates.append(treas)
            if isinstance(treas.get("assets"), (list, dict)):
                candidates.append(treas["assets"])
        if isinstance(payload.get("breakdown"), dict):
            breakdown = payload["breakdown"]
            tre = breakdown.get("treasury")
            if isinstance(tre, (list, dict)):
                candidates.append(tre)
                if isinstance(tre, dict) and isinstance(tre.get("assets"), (list, dict)):
                    candidates.append(tre["assets"])

        for node in candidates:
            ingest_node(node)

    assets: List[Dict[str, Any]] = []
    total_payload = 0.0
    if isinstance(payload, dict):
        total_payload = _to_float(
            payload.get("total_usd")
            or payload.get("treasury_usdt")
            or payload.get("total_treasury_usdt")
            or payload.get("treasury_total")
        )

    total_usd = 0.0
    for asset, info in assets_acc.items():
        balance = float(info.get("balance") or 0.0)
        raw_price = info.get("price")
        price_hint = float(raw_price) if raw_price not in (None, "", "null") else 0.0
        raw_usd = info.get("usd_value")
        usd_hint = float(raw_usd) if raw_usd not in (None, "", "null") else 0.0

        price: Optional[float] = None
        if price_hint > 0:
            price = price_hint
        elif balance > 0 and usd_hint > 0:
            derived = usd_hint / balance if balance else 0.0
            if derived > 0:
                price = derived

        if price is None and asset in _STABLE_ASSETS:
            price = 1.0

        if price is None:
            fallback = _last_price_from_logs(asset)
            if fallback is not None:
                price = fallback

        if price is None and (balance > 0 or usd_hint > 0):
            LOGGER.warning("[treasury] missing_price asset=%s source=%s balance=%.8f", asset, source, balance)

        usd_value = usd_hint if usd_hint > 0 else (balance * price if price is not None else 0.0)
        if price is None:
            usd_value = 0.0

        assets.append(_format_treasury_asset(asset, balance, price, usd_value))
        total_usd += usd_value

    if not assets and isinstance(payload, dict):
        for key, value in payload.items():
            if key in {"total_usd", "treasury_usdt", "treasury"}:
                continue
            try:
                balance = _to_float(value)
            except Exception:
                continue
            if balance:
                asset_code = str(key).upper()
                price = 1.0 if asset_code in _STABLE_ASSETS else _last_price_from_logs(asset_code)
                if price is None and balance > 0:
                    LOGGER.warning("[treasury] missing_price asset=%s source=%s balance=%.8f", asset_code, source, balance)
                usd_value = balance * price if price is not None else 0.0
                assets.append(_format_treasury_asset(asset_code, balance, price, usd_value))
                total_usd += usd_value

    total = total_payload if total_payload > 0 else total_usd
    return {
        "assets": assets,
        "total_usd": float(total),
        "source": source,
    }


def _read_treasury_sources(root: str) -> List[Dict[str, Any]]:
    """Return a list of treasury source candidates with freshness metadata."""
    candidates = [
        ("logs/treasury.json", os.path.join(root, "logs", "treasury.json")),
    ]
    sources: List[Dict[str, Any]] = []
    now = datetime.now(timezone.utc)
    for label, path in candidates:
        payload = _safe_load_json(path)
        if not isinstance(payload, dict):
            continue
        updated_raw = payload.get("updated_at") or payload.get("ts")
        updated_at = None
        freshness = None
        if isinstance(updated_raw, str):
            try:
                updated_at = datetime.fromisoformat(updated_raw.replace("Z", "+00:00"))
            except Exception:
                updated_at = None
        if updated_at:
            freshness = (now - updated_at).total_seconds()
        sources.append(
            {
                "label": label,
                "payload": payload,
                "updated_at": updated_at,
                "freshness_seconds": freshness,
            }
        )
    config_payload = _safe_load_json(os.path.join(root, "config", "reserves.json"))
    if isinstance(config_payload, dict):
        sources.append(
            {
                "label": "config/reserves.json",
                "payload": config_payload,
                "updated_at": None,
                "freshness_seconds": None,
            }
        )
    return sources


def _select_canonical_treasury(
    sources: List[Dict[str, Any]], freshness_limit: float = 600.0
) -> tuple[Dict[str, Dict[str, Any]], float, str, List[str]]:
    """Select a single treasury source using freshness priority rules."""
    assets: Dict[str, Dict[str, Any]] = {}
    total_usd = 0.0
    source_used = "config/reserves.json"
    sources_seen: List[str] = []

    best_candidate: Optional[Dict[str, Any]] = None
    for label in ("logs/treasury.json",):
        for entry in sources:
            if entry.get("label") != label:
                continue
            sources_seen.append(label)
            freshness = entry.get("freshness_seconds")
            if freshness is not None and freshness <= freshness_limit:
                best_candidate = entry
                break
        if best_candidate:
            break

    if best_candidate is None:
        for entry in sources:
            if entry.get("label") == "config/reserves.json":
                best_candidate = entry
                sources_seen.append("config/reserves.json")
                break

    if best_candidate is None:
        return assets, total_usd, source_used, sources_seen

    label = str(best_candidate.get("label") or "unknown")
    payload = best_candidate.get("payload") if isinstance(best_candidate.get("payload"), dict) else {}
    source_used = label

    def _resolve_price(
        asset: str,
        balance: float,
        price_hint: Optional[float],
        usd_hint: float,
        *,
        allow_live: bool = False,
    ) -> tuple[Optional[float], float]:
        price = price_hint if price_hint and price_hint > 0 else None

        if price is None and balance > 0 and usd_hint > 0:
            derived = usd_hint / balance if balance else 0.0
            if derived > 0:
                price = derived

        if price is None and asset in _STABLE_ASSETS:
            price = 1.0

        if price is None and allow_live:
            try:
                from execution.exchange_utils import get_price

                fetched = float(get_price(f"{asset}USDT") or 0.0)
                if fetched > 0:
                    price = fetched
            except Exception as exc:
                LOGGER.debug("[treasury] live_price_failed asset=%s source=%s error=%s", asset, source_used, exc)

        if price is None:
            fallback = _last_price_from_logs(asset)
            if fallback is not None:
                price = fallback

        usd_value = usd_hint if usd_hint > 0 else (balance * price if price is not None else 0.0)

        if price is None:
            if balance > 0 or usd_hint > 0:
                LOGGER.warning("[treasury] price_unavailable asset=%s source=%s balance=%.8f", asset, source_used, balance)
            usd_value = 0.0

        return price, usd_value

    if label == "config/reserves.json":
        config = payload
        for symbol, qty in config.items():
            try:
                amount = float(qty)
            except Exception:
                continue
            asset = str(symbol).upper()
            if not asset or amount == 0:
                continue
            price, usd_value = _resolve_price(asset, amount, None, 0.0, allow_live=True)
            assets[asset] = {
                "balance": amount,
                "price": price,
                "usd_value": usd_value,
            }
            total_usd += usd_value
    else:
        normalized = _normalize_treasury_payload(payload, label)
        for entry in normalized.get("assets", []):
            if not isinstance(entry, dict):
                continue
            asset = str(entry.get("asset") or "").upper()
            if not asset:
                continue
            balance = _to_float(entry.get("balance"))
            price_hint_raw = entry.get("price_usdt") or entry.get("price")
            try:
                price_hint = float(price_hint_raw) if price_hint_raw is not None else None
            except Exception:
                price_hint = None
            if price_hint is not None and price_hint <= 0:
                price_hint = None
            usd_hint = _to_float(entry.get("usd_value") or entry.get("usd"))
            price, usd_value = _resolve_price(asset, balance, price_hint, usd_hint)
            assets[asset] = {
                "balance": balance,
                "price": price,
                "usd_value": usd_value,
            }
            total_usd += usd_value
    return assets, float(total_usd), source_used, sources_seen


def fetch_leaderboard(limit: int = 10) -> list[Dict[str, Any]]:
    """Fetch leaderboard entries ordered by pnl descending."""
    db = get_db(strict=False)
    if not _firestore_available(db):
        return []
    env = _env()
    try:
        query = (
            db.collection("hedge")
            .document(env)
            .collection("leaderboard")
            .order_by("pnl", direction="DESCENDING")
            .limit(limit)
        )
        docs = query.stream()
        return [doc.to_dict() for doc in docs]
    except Exception as exc:
        LOGGER.warning("[firestore] fetch_leaderboard failed: %s", exc)
        return []


def publish_health(payload: Dict[str, Any]) -> None:
    """Publish health heartbeat to hedge/{ENV}/health."""
    env = payload.get("env") or _env()
    process = payload.get("process") or "unknown"
    try:
        publish_health_if_needed(payload, env=env, service=process)
        LOGGER.info("[firestore] heartbeat write ok env=%s service=%s", env, process)
    except Exception as exc:
        LOGGER.warning(
            "[firestore] heartbeat write failed env=%s service=%s error=%s",
            env,
            process,
            exc,
        )


def publish_state(snapshot: Dict[str, Any]) -> None:
    """Publish NAV/state snapshot to hedge/{ENV}/state/snapshot."""
    env = snapshot.get("env") or _env()
    path = f"hedge/{env}/state/snapshot"
    try:
        payload = dict(snapshot)
        payload["nav_ts"] = time.time()
        db = get_db(strict=False)
        if not _firestore_available(db):
            raise RuntimeError("Firestore unavailable")
        write_doc(db, path, payload, require=False)
        LOGGER.info("[firestore] state publish ok path=%s", path)
    except Exception as exc:
        LOGGER.warning("[firestore] state publish failed path=%s error=%s", path, exc)


def publish_heartbeat(
    *,
    service: str,
    status: str = "ok",
    env: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Publish lightweight heartbeat under hedge/{env}/telemetry/health."""
    try:
        now = datetime.now(timezone.utc)
        payload: Dict[str, Any] = {
            "service": service,
            "status": status,
            "ts": now.timestamp(),
            "ts_iso": now.isoformat(),
            "updated_at": now.isoformat(),
        }
        if extra:
            payload.update(extra)
        publish_health_if_needed(payload, env=env, service=service)
        LOGGER.info(
            "[firestore] heartbeat publish ok env=%s service=%s",
            env or _env(),
            service,
        )
    except Exception as exc:
        LOGGER.warning(
            "[firestore] heartbeat publish failed env=%s service=%s err=%s",
            env or _env(),
            service,
            exc,
        )


def safe_publish_health(payload: Dict[str, Any]) -> None:
    """Publish telemetry heartbeat safely with credential fallback."""
    try:
        env = payload.get("env") or _env()
        service = payload.get("service") or payload.get("process") or "unknown"
        publish_health_if_needed(payload, env=env, service=service)
        LOGGER.info(
            "[firestore] telemetry publish ok path=hedge/%s/telemetry/health service=%s",
            env,
            service,
        )
    except Exception as exc:
        LOGGER.exception("[firestore] telemetry publish failed: %s", exc)


def publish_router_health(payload: Optional[Dict[str, Any]] = None) -> None:
    """Publish router health snapshot to Firestore, falling back to local cache."""
    env = _env()
    if payload is None:
        root = _repo_root()
        payload = (
            _safe_load_json(os.path.join(root, "logs", "router_health.json"))
            or _safe_load_json(os.path.join(root, "logs", "router.json"))
            or {}
        )
    metrics = _extract_router_metrics(payload)
    body: Dict[str, Any] = {
        "env": env,
        "updated_at": _utcnow_iso(),
        "metrics": metrics,
    }
    if isinstance(payload, dict):
        body.update(payload)
    body["metrics"] = metrics
    db = get_db(strict=False)
    if not _firestore_available(db):
        raise RuntimeError("Firestore unavailable")
    write_doc(db, f"hedge/{env}/router/health", body, require=False)
    LOGGER.info("[firestore] router health publish ok env=%s", env)


def publish_positions(payload: Optional[Dict[str, Any]] = None) -> None:
    """Publish positions snapshot to Firestore, defaulting to local cache."""
    env = _env()
    root = _repo_root()
    if payload is None:
        payload = (
            _safe_load_json(os.path.join(root, "logs", "spot_state.json"))
            or _safe_load_json(os.path.join(root, "logs", "positions.json"))
            or {}
        )
    positions = None
    snapshot = None
    if isinstance(payload, dict):
        for key in ("positions", "items", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                positions = value
                break
        if positions is None:
            snapshot = payload
    else:
        positions = payload
    if not isinstance(positions, list):
        positions = []
    now_iso = _utcnow_iso()
    doc_items: Dict[str, Any] = {
        "env": env,
        "updated_at": now_iso,
        "items": positions,
    }
    combined_snapshot: Dict[str, Any] = {
        "source": "combined_spot_futures",
        "sources": {},
    }
    if snapshot is not None and isinstance(snapshot, dict):
        combined_snapshot["base"] = snapshot
    spot_info: Dict[str, Any] = {}
    try:
        from execution.exchange_utils import get_spot_balances

        spot_info = get_spot_balances() or {}
    except Exception as exc:
        LOGGER.warning("[firestore] get_spot_balances failed: %s", exc)
        spot_info = {}
    spot_balances = {
        str(asset).upper(): _to_float(val)
        for asset, val in (spot_info.get("balances") or {}).items()
        if val is not None
    }
    spot_total = _to_float(spot_info.get("total_usd"))
    if spot_total <= 0 and spot_balances:
        spot_total = sum(_to_float(val) for val in spot_balances.values())
    spot_source = str(spot_info.get("source") or "treasury_file")
    spot_updated_at = spot_info.get("updated_at")
    if spot_balances or spot_total:
        combined_snapshot["sources"]["spot"] = {
            "source": spot_source,
            "balances": spot_balances,
            "total_usd": float(spot_total),
            "updated_at": spot_updated_at,
        }
        raw_payload = spot_info.get("raw")
        if isinstance(raw_payload, dict) and raw_payload:
            combined_snapshot["sources"]["spot"]["raw"] = raw_payload
    futures_balances: Dict[str, float] = {}
    nav_trading: Optional[Dict[str, Any]] = None
    futures_source = "futures_api"
    futures_updated_at = None
    try:
        from execution.exchange_utils import get_futures_balances

        futures_balances = get_futures_balances() or {}
    except Exception as exc:
        LOGGER.warning("[firestore] get_futures_balances failed: %s", exc)
        futures_balances = {}
    if futures_balances:
        futures_updated_at = _utcnow_iso()
    if not futures_balances:
        nav_trading = _safe_load_json(os.path.join(root, "logs", "nav_trading.json")) or {}
        breakdown = nav_trading.get("breakdown") if isinstance(nav_trading, dict) else {}
        futures_usd = _to_float((breakdown or {}).get("futures_wallet_usdt"))
        if futures_usd > 0:
            futures_balances["USDT"] = futures_usd
            futures_source = "nav_trading.json"
            futures_updated_at = (nav_trading or {}).get("ts") if isinstance(nav_trading, dict) else None
    futures_total = sum(_to_float(val) for val in futures_balances.values())
    if futures_balances or futures_total:
        combined_snapshot["sources"]["futures_wallet"] = {
            "source": "futures_api" if futures_source == "futures_api" else futures_source,
            "balances": futures_balances,
            "total_usd": float(futures_total),
            "updated_at": futures_updated_at,
        }
        if nav_trading:
            combined_snapshot["sources"]["futures_wallet"]["raw"] = nav_trading
    if combined_snapshot["sources"]:
        doc_items["snapshot"] = combined_snapshot
        doc_items["source"] = combined_snapshot["source"]
        doc_items["sources"] = combined_snapshot["sources"]
    elif snapshot is not None:
        doc_items["snapshot"] = snapshot
    db = get_db(strict=False)
    if not _firestore_available(db):
        raise RuntimeError("Firestore unavailable")
    write_doc(db, f"hedge/{env}/state/positions", doc_items, require=False)
    LOGGER.info("[firestore] positions publish ok env=%s count=%d", env, len(positions))

    try:
        legacy_doc = dict(doc_items)
        legacy_doc["positions"] = positions
        write_doc(db, f"hedge/{env}/positions/latest", legacy_doc, require=False)
        LOGGER.debug("[firestore] positions legacy mirror ok env=%s", env)
    except Exception as exc:
        LOGGER.debug("[firestore] positions legacy mirror failed env=%s err=%s", env, exc)


def publish_treasury(payload: Optional[Dict[str, Any]] = None) -> None:
    """Publish treasury holdings snapshot to Firestore using the on-disk cache."""
    env = _env()
    root = _repo_root()
    treasury_path = os.path.join(root, "logs", "treasury.json")

    try:
        with open(treasury_path, "r", encoding="utf-8") as handle:
            treasury_data = json.load(handle)
    except FileNotFoundError:
        LOGGER.warning("[firestore] publish_treasury skipped (missing %s)", treasury_path)
        return
    except Exception as exc:
        LOGGER.warning("[firestore] publish_treasury read failed: %s", exc)
        return

    if not isinstance(treasury_data, dict):
        LOGGER.warning("[firestore] publish_treasury invalid payload type=%s", type(treasury_data).__name__)
        return

    assets = treasury_data.get("assets")
    if not isinstance(assets, list):
        LOGGER.warning("[firestore] publish_treasury assets missing or invalid")
        assets = []

    total_usd = treasury_data.get("total_usd")
    try:
        total_usd = float(total_usd)
    except Exception:
        total_usd = float(
            sum(
                _to_float(entry.get("usd_value") or entry.get("USD Value"))
                for entry in assets
                if isinstance(entry, dict)
            )
        )

    treasury_payload: Dict[str, Any] = {
        "assets": assets,
        "total_usd": float(total_usd),
        "source": "logs/treasury.json",
        "sources_seen": ["logs/treasury.json"],
    }

    doc = {
        "env": env,
        "updated_at": _utcnow_iso(),
        "treasury": treasury_payload,
    }

    db = get_db(strict=False)
    if not _firestore_available(db):
        raise RuntimeError("Firestore unavailable")
    write_doc(db, f"hedge/{env}/treasury/latest", doc, require=False)
    LOGGER.info("[firestore] treasury publish ok env=%s source=%s total=%.2f", env, "logs/treasury.json", total_usd)
