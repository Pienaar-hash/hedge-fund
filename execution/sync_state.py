# execution/sync_state.py

import os
import json
import time
from pathlib import Path
from firebase_admin import credentials, firestore, initialize_app
from execution.telegram_utils import send_telegram

STATE_FILE = "synced_state.json"

# Use relative path to avoid nesting issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIREBASE_CREDS = os.getenv("FIREBASE_CREDS_PATH", os.path.join(BASE_DIR, "../config/firebase_creds.json"))
print(f"📂 Trying to load Firebase credentials from: {FIREBASE_CREDS}")

def init_firebase():
    if not Path(FIREBASE_CREDS).exists():
        print(f"❌ Firebase credentials not found at {FIREBASE_CREDS}")
        return None

    try:
        cred = credentials.Certificate(FIREBASE_CREDS)
        initialize_app(cred)
        return firestore.client()
    except Exception as e:
        print(f"❌ Failed to initialize Firebase: {e}")
        return None

def load_local_state() -> dict:
    if Path(STATE_FILE).exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_local_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def sync_portfolio_state():
    db = init_firebase()
    if db is None:
        send_telegram("❌ Firebase not initialized — skipping sync.")
        return

    local_state = load_local_state()
    print(f"📥 Local state loaded: {local_state}")

    try:
        doc_ref = db.collection("hedge_fund").document("synced_state")
        remote_doc = doc_ref.get()
        remote_state = remote_doc.to_dict() if remote_doc.exists else {}

        merged = {**remote_state, **local_state}
        doc_ref.set(merged)
        save_local_state(merged)

        print("✅ Synced portfolio state with Firestore.")
        send_telegram("✅ Portfolio state synced with Firestore.", silent=True)

    except Exception as e:
        print(f"❌ Sync error: {e}")
        send_telegram(f"❌ Sync error: {e}", silent=True)

if __name__ == "__main__":
    while True:
        sync_portfolio_state()
        time.sleep(600)  # Sync every 10 minutes
