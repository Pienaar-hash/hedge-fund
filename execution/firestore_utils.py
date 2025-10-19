# execution/firestore_utils.py
import logging
import os
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
    body.setdefault("ts", datetime.now(timezone.utc).isoformat())
    path = f"hedge/{env}/health/{process}"
    creds = os.environ.get("FIREBASE_CREDS_PATH") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds:
        LOGGER.info("[firestore] skipped: no credentials set path=%s", path)
        return
    try:
        db = get_firestore()
        LOGGER.debug("[firestore] client ready path=%s", path)
        LOGGER.debug("[firestore] heartbeat payload=%s", body)
        db.collection("hedge").document(env).collection("health").document(process).set(body, merge=True)
        LOGGER.info("[firestore] heartbeat write ok path=%s", path)
    except Exception as exc:
        LOGGER.warning("[firestore] heartbeat write failed path=%s error=%s", path, exc)
