# scripts/fetch_synced_state.py

import os
import json
from pathlib import Path
from firebase_admin import credentials, firestore, initialize_app

FIREBASE_CREDS = os.getenv("FIREBASE_CREDS_PATH", "config/firebase_creds.json")
LOCAL_STATE_PATH = "synced_state.json"
FIRESTORE_COLLECTION = "hedge_fund"
FIRESTORE_DOCUMENT = "synced_state"

def init_firebase():
    if not Path(FIREBASE_CREDS).exists():
        raise FileNotFoundError(f"Firebase credentials not found at {FIREBASE_CREDS}")
    cred = credentials.Certificate(FIREBASE_CREDS)
    initialize_app(cred)
    return firestore.client()

def fetch_synced_state():
    db = init_firebase()
    doc_ref = db.collection(FIRESTORE_COLLECTION).document(FIRESTORE_DOCUMENT)
    doc = doc_ref.get()

    if not doc.exists:
        raise ValueError("❌ Firestore document does not exist.")
    
    state = doc.to_dict()
    with open(LOCAL_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)

    print(f"✅ Synced Firestore → {LOCAL_STATE_PATH}")
    return state

if __name__ == "__main__":
    try:
        fetch_synced_state()
    except Exception as e:
        print(f"❌ Error during sync: {e}")
