from functools import lru_cache
import os
import json
import google.auth
from google.oauth2 import service_account
from google.cloud import firestore

@lru_cache(maxsize=1)
def get_db():
    creds_path = os.environ.get("FIREBASE_CREDS_PATH")
    if not creds_path or not os.path.exists(creds_path):
        raise RuntimeError("FIREBASE_CREDS_PATH missing or invalid")
    with open(creds_path, "r") as f:
        info = json.load(f)
    creds = service_account.Credentials.from_service_account_info(info)
    return firestore.Client(credentials=creds, project=info["project_id"])
