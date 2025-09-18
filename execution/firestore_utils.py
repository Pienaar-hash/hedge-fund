# execution/firestore_utils.py
import os

import firebase_admin
from firebase_admin import credentials, firestore

_firestore_client = None


def get_firestore():
    global _firestore_client
    if _firestore_client is None:
        creds_path = os.environ.get("FIREBASE_CREDS_PATH")
        if not creds_path or not os.path.exists(creds_path):
            raise FileNotFoundError(f"Firebase credentials not found at {creds_path}")
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
