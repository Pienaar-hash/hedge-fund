from functools import lru_cache
import os
import sys
import json
import base64
from typing import Any, Dict, TYPE_CHECKING

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

_ADC_WARNED = False

REQUIRED_FIELDS = ("project_id", "client_email", "private_key")


def _load_creds_dict() -> Dict[str, Any]:
    """Load Firestore service-account JSON via multiple fallbacks:
    1) FIREBASE_CREDS_PATH -> file path
    2) FIREBASE_CREDS_JSON -> raw JSON or base64-encoded JSON
    3) GOOGLE_APPLICATION_CREDENTIALS -> file path (GCP standard)
    4) ./config/firebase_creds.json -> repo default
    """
    # 1) Explicit path
    path = os.environ.get("FIREBASE_CREDS_PATH")
    if path and os.path.exists(path):
        os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", path)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # 2) Inline env JSON (raw or base64)
    inline = os.environ.get("FIREBASE_CREDS_JSON")
    if inline:
        try:
            # Try raw JSON first
            return json.loads(inline)
        except Exception:
            try:
                decoded = base64.b64decode(inline).decode("utf-8")
                return json.loads(decoded)
            except Exception as e:
                raise RuntimeError(f"FIREBASE_CREDS_JSON invalid (not JSON or base64): {e}")

    # 3) Standard GCP path
    gpath = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if gpath and os.path.exists(gpath):
        with open(gpath, "r", encoding="utf-8") as f:
            return json.load(f)

    # 4) Repo default
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


def _noop_db():
    class _NoopDoc:
        def set(self, *_a, **_k):
            return None

        def update(self, *_a, **_k):
            return None

        def get(self):
            class _Empty:
                def to_dict(self_inner):
                    return {}

            return _Empty()

    class _NoopCollection:
        def document(self, *_a, **_k):
            return _NoopDoc()

        def add(self, *_a, **_k):
            return (None, None)

        def where(self, *_a, **_k):
            return self

    class _NoopDB:
        def collection(self, *_a, **_k):
            return _NoopCollection()

    return _NoopDB()


@lru_cache(maxsize=1)
def get_db() -> FirestoreClient:
    global _ADC_WARNED
    if os.environ.get("FIRESTORE_ENABLED", "1") == "0":
        return _noop_db()
    if firestore is None or service_account is None:
        if not _ADC_WARNED:
            _ADC_WARNED = True
            print(
                "[firestore] WARN client unavailable; install google-cloud-firestore",
                file=sys.stderr,
            )
        return _noop_db()
    try:
        info = _load_creds_dict()
        _validate_creds(info)

        project_id = os.environ.get("FIREBASE_PROJECT_ID") or info.get("project_id")
        creds = service_account.Credentials.from_service_account_info(info)
        return firestore.Client(credentials=creds, project=project_id)
    except Exception as exc:
        fallback_exc = exc
        # Fallback: use GOOGLE_APPLICATION_CREDENTIALS directly if valid
        gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if gac and os.path.exists(gac):
            try:
                return firestore.Client()
            except Exception as inner:
                fallback_exc = inner
        if not _ADC_WARNED:
            _ADC_WARNED = True
            print(
                f"[firestore] WARN ADC unavailable; running in no-op mode: {fallback_exc}",
                file=sys.stderr,
            )
        return _noop_db()
