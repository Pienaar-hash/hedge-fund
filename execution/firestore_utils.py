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

LOGGER = logging.getLogger("firestore")

_DIRECT_CLIENT = None
_FIRESTORE_WRITES_DISABLED = True  # local-only mode disables all remote publishes
_TRUTHY = {"1", "true", "yes", "on"}


class _NoopDoc:
    _is_noop = True

    def __init__(self, path: str = "noop") -> None:
        self.path = path

    def set(self, *_args, **_kwargs):
        return None

    def update(self, *_args, **_kwargs):
        return None

    def get(self):
        class _Empty:
            exists = False

            def to_dict(self_inner):
                return {}

        return _Empty()


class _NoopFirestore:
    _is_noop = True

    def __init__(self, root: str = "noop") -> None:
        self.root = root

    def collection(self, name: str):
        return self

    def document(self, name: str):
        return self

    def set(self, *_args, **_kwargs):
        return None

    def get(self):
        return _NoopDoc().get()

    def batch(self):
        return self

    def commit(self):
        return None


_NOOP_DB = _NoopFirestore()


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
        return None
    try:
        _DIRECT_CLIENT = _firestore_client.Client()
    except Exception as exc:  # pragma: no cover - optional dependency
        LOGGER.warning("[firestore] client_init_failed: %s", exc)
        _DIRECT_CLIENT = None
    return _DIRECT_CLIENT


def get_db(strict: bool = True) -> Any:
    client = _direct_client()
    if client is None:
        if strict:
            raise RuntimeError("Firestore client unavailable")
        return _NOOP_DB
    return client


def publish_router_metrics(doc_id: str, payload: Dict[str, Any], *, env: Optional[str] = None) -> None:
    """
    Mirror router metrics into Firestore under hedge/{env}/router_metrics/{doc_id}.
    """
    return None


def publish_symbol_toggle(symbol: str, meta: Dict[str, Any], *, env: Optional[str] = None) -> None:
    """
    Mirror symbol toggle state into Firestore under hedge/{env}/symbol_toggles/{symbol}.
    """
    return None


def publish_execution_health(symbol: Optional[str], payload: Dict[str, Any], *, env: Optional[str] = None) -> None:
    """
    Mirror execution health snapshots for remote dashboards.
    """
    return None


def publish_execution_alert(alert: Dict[str, Any], *, env: Optional[str] = None) -> None:
    """
    Mirror execution alerts for dashboard consumption.
    """
    return None


def publish_execution_intel(symbol: str, payload: Dict[str, Any], *, env: Optional[str] = None) -> None:
    """
    Mirror execution intelligence snapshots into Firestore.
    """
    return None


def fetch_symbol_toggles(*, env: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetch all symbol toggle docs for the requested environment.
    """
    client = _direct_client()
    if client is None:
        return []
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
    return None


def publish_state(snapshot: Dict[str, Any]) -> None:
    """Publish NAV/state snapshot to hedge/{ENV}/state/snapshot."""
    return None


def publish_heartbeat(
    *,
    service: str,
    status: str = "ok",
    env: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Publish lightweight heartbeat under hedge/{env}/telemetry/health."""
    db = get_db(strict=False)
    if not _firestore_available(db):
        return
    env_name = env or _env()
    payload = {
        "status": status,
        "ts": _utcnow_iso(),
    }
    if extra:
        payload.update(extra)
    try:
        doc = (
            db.collection("hedge")
            .document(env_name)
            .collection("telemetry")
            .document("health")
        )
        doc.set({service: payload, "updated_at": _utcnow_iso()}, merge=True)
    except Exception as exc:
        LOGGER.debug("[firestore] heartbeat_publish_failed env=%s service=%s err=%s", env_name, service, exc)


def safe_publish_health(payload: Dict[str, Any]) -> None:
    """Publish telemetry heartbeat safely with credential fallback."""
    return None


def publish_router_health(payload: Optional[Dict[str, Any]] = None) -> None:
    """Publish router health snapshot to Firestore, falling back to local cache."""
    return None


def publish_positions(payload: Optional[Dict[str, Any]] = None) -> None:
    """Publish positions snapshot to Firestore, defaulting to local cache."""
    return None
