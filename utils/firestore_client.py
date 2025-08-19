from functools import lru_cache
import os
import json
import base64
from typing import Any, Dict
from google.oauth2 import service_account
from google.cloud import firestore

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


@lru_cache(maxsize=1)
def get_db() -> firestore.Client:
    info = _load_creds_dict()
    _validate_creds(info)

    # Allow project override via env if needed
    project_id = os.environ.get("FIREBASE_PROJECT_ID") or info.get("project_id")

    creds = service_account.Credentials.from_service_account_info(info)
    return firestore.Client(credentials=creds, project=project_id)
