# execution/firestore_utils.py
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict

import firebase_admin
from firebase_admin import credentials, firestore

_firestore_client = None
LOGGER = logging.getLogger("firestore")


def get_firestore():
    global _firestore_client
    if _firestore_client is None:
        creds_path = os.environ.get("FIREBASE_CREDS_PATH")
        if not creds_path or not os.path.exists(creds_path):
            raise FileNotFoundError(f"Firebase credentials not found at {creds_path}")
        LOGGER.debug("[firestore] initializing client creds=%s", creds_path)
        cred = credentials.Certificate(creds_path)
        firebase_admin.initialize_app(cred)
        _firestore_client = firestore.client()
    return _firestore_client


def fetch_leaderboard(limit=10):
    """Fetch leaderboard from Firestore."""
    db = get_firestore()
    docs = (
        db.collection("leaderboard")
        .order_by("pnl", direction=firestore.Query.DESCENDING)
        .limit(limit)
        .stream()
    )
    return [doc.to_dict() for doc in docs]


def publish_health(payload: Dict[str, Any]) -> None:
    """Publish health heartbeat to hedge/{ENV}/health."""
    env = payload.get("env") or os.getenv("ENV", os.getenv("ENVIRONMENT", "prod"))
    process = payload.get("process") or "unknown"
    body = dict(payload)
    now_ts = time.time()
    body["ts"] = now_ts
    body.setdefault("ts_iso", datetime.now(timezone.utc).isoformat())
    path = f"hedge/{env}/health/{process}"
    creds = os.environ.get("FIREBASE_CREDS_PATH") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds:
        LOGGER.info("[firestore] skipped: no credentials set path=%s", path)
        return
    try:
        db = get_firestore()
        LOGGER.debug("[firestore] client ready path=%s", path)
        LOGGER.debug("[firestore] heartbeat payload=%s", body)
        doc_ref = db.collection("hedge").document(env).collection("health").document(process)
        doc_ref.set(body, merge=True)
        LOGGER.info("[firestore] heartbeat write ok path=%s ts=%.3f", path, now_ts)
    except Exception as exc:
        LOGGER.warning("[firestore] heartbeat write failed path=%s error=%s", path, exc)


def publish_state(snapshot: Dict[str, Any]) -> None:
    """Publish NAV/state snapshot to hedge/{ENV}/state/snapshot."""
    env = snapshot.get("env") or os.getenv("ENV", os.getenv("ENVIRONMENT", "prod"))
    path = f"hedge/{env}/state/snapshot"
    creds = os.environ.get("FIREBASE_CREDS_PATH") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds:
        LOGGER.info("[firestore] skipped: no credentials set path=%s", path)
        return
    try:
        db = get_firestore()
        doc_ref = db.collection("hedge").document(env).collection("state").document("snapshot")
        payload = dict(snapshot)
        payload["nav_ts"] = time.time()
        doc_ref.set(payload, merge=True)
        LOGGER.info(
            "[firestore] state publish ok path=%s nav_usd=%s",
            path,
            payload.get("nav_usd"),
        )
    except Exception as exc:
        LOGGER.warning("[firestore] state publish failed path=%s error=%s", path, exc)
