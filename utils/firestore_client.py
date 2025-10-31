from __future__ import annotations

import base64
import json
import logging
import traceback
import os
import time
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Dict, Optional, TYPE_CHECKING, Callable

service_account: Any
firestore: Any
try:
    from google.oauth2 import service_account as _service_account
    from google.cloud import firestore as _firestore
except Exception:  # pragma: no cover - optional dependency
    service_account = None
    firestore = None
else:
    service_account = _service_account
    firestore = _firestore

if TYPE_CHECKING:  # pragma: no cover - type hint only
    from google.cloud.firestore import Client as FirestoreClient
else:
    FirestoreClient = Any

_LOG = logging.getLogger("firestore")
REQUIRED_FIELDS = ("project_id", "client_email", "private_key")
_FORCE_STRICT_INIT = True


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


class _NoopDoc:
    _is_noop = True

    def __init__(self, path: str) -> None:
        self.path = path

    def set(self, *_a, **_k):
        return None

    def update(self, *_a, **_k):
        return None

    def add(self, *_a, **_k):
        return None

    def commit(self, *_a, **_k):
        return None

    def collection(self, name: str) -> "_NoopCollection":
        return _NoopCollection(f"{self.path}/{name}")

    def get(self):
        class _Empty:
            exists = False

            def to_dict(self_inner):
                return {}

        return _Empty()


class _NoopCollection:
    _is_noop = True

    def __init__(self, path: str) -> None:
        self.path = path

    def document(self, name: str) -> _NoopDoc:
        return _NoopDoc(f"{self.path}/{name}")

    def add(self, *_a, **_k):
        return (None, None)


class _NoopBatch:
    _is_noop = True

    def commit(self):
        return None


class _NoopDB:
    _is_noop = True

    def collection(self, name: str) -> _NoopCollection:
        return _NoopCollection(name)

    def batch(self) -> _NoopBatch:
        return _NoopBatch()


def _noop_db() -> _NoopDB:
    return _NoopDB()


def _call_with_retry(op: Callable[[], Any], action: str) -> Any:
    delay = 0.25
    for attempt in range(1, 6):
        try:
            return op()
        except Exception as exc:
            if attempt == 5:
                _LOG.error("Firestore action failed (%s): %s", action, exc)
                raise
            _LOG.warning("Firestore retry %s/5 for %s: %s", attempt, action, exc)
            _sleep(delay)
            delay = min(delay * 2.0, 4.0)
    return None


class _DocWrapper:
    _is_noop = False

    def __init__(self, doc: Any) -> None:
        self._doc = doc
        self._path = getattr(doc, "path", "doc")

    def set(self, *args, **kwargs):
        return _call_with_retry(
            lambda: self._doc.set(*args, **kwargs), f"doc.set {self._path}"
        )

    def collection(self, *args, **kwargs):
        return _CollectionWrapper(self._doc.collection(*args, **kwargs))

    def __getattr__(self, item: str) -> Any:
        return getattr(self._doc, item)


class _CollectionWrapper:
    _is_noop = False

    def __init__(self, col: Any) -> None:
        self._col = col
        self._path = getattr(col, "path", "collection")

    def document(self, *args, **kwargs):
        return _DocWrapper(self._col.document(*args, **kwargs))

    def add(self, *args, **kwargs):
        result = _call_with_retry(
            lambda: self._col.add(*args, **kwargs), f"collection.add {self._path}"
        )
        if isinstance(result, tuple) and result and hasattr(result[0], "path"):
            wrapped = _DocWrapper(result[0])
            if len(result) == 1:
                return (wrapped,)
            return (wrapped,) + tuple(result[1:])
        return result

    def __getattr__(self, item: str) -> Any:
        return getattr(self._col, item)


class _BatchWrapper:
    _is_noop = False

    def __init__(self, batch: Any) -> None:
        self._batch = batch

    def commit(self):
        return _call_with_retry(self._batch.commit, "batch.commit")

    def set(self, doc_ref, *args, **kwargs):
        base_ref = getattr(doc_ref, "_doc", doc_ref)
        return self._batch.set(base_ref, *args, **kwargs)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._batch, item)


class _ClientWrapper:
    _is_noop = False

    def __init__(self, client: FirestoreClient) -> None:
        self._client = client

    def collection(self, *args, **kwargs):
        return _CollectionWrapper(self._client.collection(*args, **kwargs))

    def collection_group(self, *args, **kwargs):
        return self._client.collection_group(*args, **kwargs)

    def document(self, *args, **kwargs):
        return _DocWrapper(self._client.document(*args, **kwargs))

    def batch(self):
        return _BatchWrapper(self._client.batch())

    def __getattr__(self, item: str) -> Any:
        return getattr(self._client, item)


def _is_noop_client(db: Any) -> bool:
    return getattr(db, "_is_noop", False)


def _load_creds_dict() -> Dict[str, Any]:
    """Load Firestore service-account JSON via multiple fallbacks:
    1) FIREBASE_CREDS_PATH -> file path
    2) FIREBASE_CREDS_JSON -> raw JSON or base64-encoded JSON
    3) GOOGLE_APPLICATION_CREDENTIALS -> file path (GCP standard)
    4) ./config/firebase_creds.json -> repo default
    """
    path = os.environ.get("FIREBASE_CREDS_PATH")
    if path and os.path.exists(path):
        os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", path)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    inline = os.environ.get("FIREBASE_CREDS_JSON")
    if inline:
        try:
            return json.loads(inline)
        except Exception:
            try:
                decoded = base64.b64decode(inline).decode("utf-8")
                return json.loads(decoded)
            except Exception as e:
                raise RuntimeError(
                    f"FIREBASE_CREDS_JSON invalid (not JSON or base64): {e}"
                )

    gpath = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if gpath and os.path.exists(gpath):
        with open(gpath, "r", encoding="utf-8") as f:
            return json.load(f)

    default_path = os.path.join(os.getcwd(), "config", "firebase_creds.json")
    if os.path.exists(default_path):
        with open(default_path, "r", encoding="utf-8") as f:
            return json.load(f)

    raise RuntimeError(
        "Firebase credentials not found. Set FIREBASE_CREDS_PATH (file), "
        "FIREBASE_CREDS_JSON (json/base64), or place config/firebase_creds.json."
    )


def _validate_creds(info: Dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_FIELDS if not info.get(k)]
    if missing:
        raise RuntimeError(f"Firebase creds missing fields: {', '.join(missing)}")


def _firestore_env_enabled() -> bool:
    flag = os.environ.get("FIRESTORE_ENABLED", "1").strip().lower()
    return flag not in {"0", "false", "no"}


@lru_cache(maxsize=2)
def _get_db_cached(strict_flag: bool) -> FirestoreClient:
    env_enabled = _firestore_env_enabled()
    if not env_enabled:
        if strict_flag:
            raise RuntimeError("Firestore disabled via FIRESTORE_ENABLED flag")
        _LOG.info("[firestore] disabled via FIRESTORE_ENABLED=0 -> using noop client")
        return _noop_db()
    if firestore is None or service_account is None:
        if strict_flag:
            raise RuntimeError("Firestore client unavailable; install google-cloud-firestore")
        _LOG.warning("[firestore] client libraries unavailable -> noop client")
        return _noop_db()
    try:
        info = _load_creds_dict()
        _validate_creds(info)
        _LOG.info("[firestore] initializing strict Firestore client")
        creds = service_account.Credentials.from_service_account_info(info)
        client = firestore.Client(credentials=creds, project=creds.project_id)
        _LOG.info(f"[firestore] Firestore client initialized for project {creds.project_id}")
        return _ClientWrapper(client)
    except Exception as exc:
        _LOG.error(f"[firestore] Firestore init failed: {exc}\n{traceback.format_exc()}")
        if strict_flag:
            raise
        return _noop_db()


def get_db(strict: Optional[bool] = None) -> FirestoreClient:
    """
    Return a Firestore client, requiring availability unless explicitly disabled.
    - strict=None (default): strict follows FIRESTORE_ENABLED (defaults to True).
    - strict=True: raise if dependencies or credentials are missing.
    - strict=False: return a no-op client when unavailable.
    """
    env_enabled = _firestore_env_enabled()
    if _FORCE_STRICT_INIT and strict is False:
        strict_flag = False
    elif _FORCE_STRICT_INIT:
        strict_flag = True
    else:
        strict_flag = env_enabled if strict is None else bool(strict)
    db = _get_db_cached(strict_flag)
    if strict_flag and _is_noop_client(db):
        raise RuntimeError("Firestore unavailable: client initialization returned no-op")
    return db


@contextmanager
def with_firestore():
    db = get_db()
    if _is_noop_client(db):
        if os.environ.get("FIRESTORE_ENABLED", "1") == "0":
            raise RuntimeError("Firestore disabled (FIRESTORE_ENABLED=0)")
        raise RuntimeError(
            "Firestore unavailable: credentials missing or client not installed"
        )
    yield db


def write_doc(
    db: Any,
    path: str,
    data: Dict[str, Any],
    *,
    require: bool = True,
) -> bool:
    """
    Write a Firestore document. Returns True on success.
    When require=True and the write cannot be completed, raise an error.
    """
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Firestore path must be a non-empty string")
    parts = [segment for segment in path.split("/") if segment]
    if len(parts) % 2 != 0:
        raise ValueError(f"Invalid Firestore document path: {path}")
    if db is None or _is_noop_client(db):
        if require:
            raise RuntimeError(f"Firestore unavailable for required write: {path}")
        return False
    ref = db
    try:
        while parts:
            collection_name, doc_name = parts[:2]
            parts = parts[2:]
            ref = ref.collection(collection_name).document(doc_name)
        ref.set(data)
        return True
    except Exception as exc:
        if require:
            raise
        _LOG.warning("Firestore write failed path=%s err=%s", path, exc)
        return False


def publish_heartbeat(
    db: Any,
    env: str,
    service: str,
    status: str,
    *,
    error_count: int = 0,
    last_success_ts: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Publish a structured heartbeat document under hedge/{env}/health/{service}.
    Returns True when the write succeeds, False otherwise (never raises).
    """
    payload: Dict[str, Any] = {
        "ts": time.time(),
        "env": env,
        "service": service,
        "status": status,
        "error_count": int(error_count),
        "last_success_ts": last_success_ts,
    }
    if extra:
        payload.update(extra)
    path = f"hedge/{env}/health/{service}"
    return write_doc(db, path, payload, require=False)
