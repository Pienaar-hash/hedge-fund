from __future__ import annotations

import os
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

try:
    from google.cloud import firestore
except Exception:  # pragma: no cover - optional dependency
    firestore = None  # type: ignore


ENV = os.getenv("ENV", "prod")
DB = firestore.Client() if firestore is not None else None  # type: ignore[assignment]
LOG = logging.getLogger("firestore_mirror")
MAX_DOC_ITEMS = 200


def _payload(items: List[Any]) -> Dict[str, Any]:
    return {
        "ts_iso": datetime.now(timezone.utc).isoformat(),
        "count": len(items),
        "items": list(items)[:MAX_DOC_ITEMS],
    }


def publish_exec_snapshot(kind: str, items: List[Any]) -> None:
    """Mirror local router/trade summaries to Firestore for dashboard consumption."""
    if DB is None:
        LOG.info("[mirror] skipped %s: Firestore client unavailable", kind)
        return
    try:
        payload = _payload(items)
        doc_ref = (
            DB.collection("hedge")
            .document(ENV)
            .collection("executions")
            .document(kind)
        )
        doc_ref.set(payload)
        LOG.info("[mirror] published %s n=%d", kind, len(items))
    except Exception as exc:
        LOG.warning("[mirror] error %s: %s", kind, exc)


def publish_signals_snapshot(signals: List[Any]) -> None:
    """Publish latest signals (for dashboard Signals tab)."""
    if DB is None:
        LOG.info("[mirror] skipped signals: Firestore client unavailable")
        return
    try:
        payload = _payload(signals)
        doc_ref = (
            DB.collection("hedge")
            .document(ENV)
            .collection("signals")
            .document("latest")
        )
        doc_ref.set(payload)
        LOG.info("[mirror] published signals n=%d", len(signals))
    except Exception as exc:
        LOG.warning("[mirror] error signals: %s", exc)
