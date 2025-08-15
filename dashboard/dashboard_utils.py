import sys
import os
from pathlib import Path

# Ensure absolute project root in sys.path for imports
ROOT = "/root/hedge-fund"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.firestore_client import get_db

# --------------------
# Dashboard utility functions
# --------------------

def get_firestore_connection():
    """Return a Firestore DB client using the configured credentials."""
    return get_db()


def format_currency(value: float) -> str:
    return f"{value:,.2f}"


def fetch_state_document(doc_name: str, env: str = None):
    """Fetch a document from the Firestore state collection for the given ENV."""
    db = get_db()
    env = env or os.getenv("ENV", "prod")
    doc_ref = db.collection("hedge").document(env).collection("state").document(doc_name)
    snapshot = doc_ref.get()
    return snapshot.to_dict() if snapshot.exists else {}
