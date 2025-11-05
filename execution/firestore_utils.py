# execution/firestore_utils.py
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from utils.firestore_client import get_db, write_doc

LOGGER = logging.getLogger("firestore")


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
    body = dict(payload)
    now_ts = time.time()
    body.setdefault("ts", now_ts)
    body.setdefault("ts_iso", _utcnow_iso())
    path = f"hedge/{env}/health/{process}"
    try:
        db = get_db(strict=False)
        if not _firestore_available(db):
            raise RuntimeError("Firestore unavailable")
        write_doc(db, path, body, require=False)
        LOGGER.info("[firestore] heartbeat write ok path=%s", path)
    except Exception as exc:
        LOGGER.warning("[firestore] heartbeat write failed path=%s error=%s", path, exc)


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
    """Publish lightweight heartbeat under hedge/{env}/telemetry/heartbeats."""
    env_name = env or os.getenv("ENV", os.getenv("ENVIRONMENT", "prod"))
    try:
        db = get_db(strict=False)
        if not _firestore_available(db):
            raise RuntimeError("Firestore client unavailable")
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
        doc_ref = (
            db.collection("hedge")
            .document(env_name)
            .collection("telemetry")
            .document("heartbeats")
        )
        doc_ref.set(payload, merge=True)
        LOGGER.info("[firestore] heartbeat publish ok env=%s service=%s", env_name, service)
    except Exception as exc:
        LOGGER.warning(
            "[firestore] heartbeat publish failed env=%s service=%s err=%s",
            env_name,
            service,
            exc,
        )


def safe_publish_health(payload: Dict[str, Any]) -> None:
    """Publish telemetry heartbeat safely with credential fallback."""
    try:
        env = payload.get("env") or _env()
        service = payload.get("service") or payload.get("process") or "unknown"
        doc = dict(payload)
        doc.setdefault("ts", time.time())
        doc.setdefault("ts_iso", _utcnow_iso())
        db = get_db(strict=False)
        if not _firestore_available(db):
            raise RuntimeError("Firestore unavailable")
        write_doc(db, f"hedge/{env}/health/{service}", doc, require=False)
        LOGGER.info("[firestore] telemetry publish ok path=hedge/%s/health/%s", env, service)
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
    body = {
        "env": env,
        "updated_at": _utcnow_iso(),
    }
    if isinstance(payload, dict):
        body.update(payload)
    db = get_db(strict=False)
    if not _firestore_available(db):
        raise RuntimeError("Firestore unavailable")
    write_doc(db, f"hedge/{env}/router/health", body, require=False)
    LOGGER.info("[firestore] router health publish ok env=%s", env)


def publish_positions(payload: Optional[Dict[str, Any]] = None) -> None:
    """Publish positions snapshot to Firestore, defaulting to local cache."""
    env = _env()
    if payload is None:
        root = _repo_root()
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
    doc = {
        "env": env,
        "updated_at": _utcnow_iso(),
        "positions": positions,
    }
    if snapshot is not None:
        doc["snapshot"] = snapshot
    db = get_db(strict=False)
    if not _firestore_available(db):
        raise RuntimeError("Firestore unavailable")
    write_doc(db, f"hedge/{env}/positions/latest", doc, require=False)
    LOGGER.info("[firestore] positions publish ok env=%s count=%d", env, len(doc["positions"]))
