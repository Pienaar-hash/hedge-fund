import os
import json
import firebase_admin
from firebase_admin import credentials, firestore

# Paths
FIREBASE_CREDS_PATH = os.path.join(os.path.dirname(__file__), "../config/firebase_creds.json")
SYNCED_STATE_PATH = os.path.join(os.path.dirname(__file__), "../synced_state.json")
TRADE_LOG_PATH = os.path.join(os.path.dirname(__file__), "../logs/trade_log.json")

# Firebase initialization guard
def init_firebase():
    """Initialize Firebase app only once."""
    if not firebase_admin._apps:
        if os.path.exists(FIREBASE_CREDS_PATH):
            try:
                cred = credentials.Certificate(FIREBASE_CREDS_PATH)
                firebase_admin.initialize_app(cred)
                print("✅ Firebase initialized")
            except Exception as e:
                print(f"❌ Failed to initialize Firebase: {e}")
        else:
            print(f"⚠️ Firebase credentials not found at {FIREBASE_CREDS_PATH}")

# Sync portfolio state
def sync_portfolio_state(state_map):
    """Push synced_state.json to Firestore."""
    try:
        init_firebase()
        with open(SYNCED_STATE_PATH, "w") as f:
            json.dump(state_map, f, indent=2)

        db = firestore.client()
        db.collection("hedge_portfolio").document("latest").set({
            "portfolio": state_map,
            "updated_at": firestore.SERVER_TIMESTAMP
        })
        print("✅ Portfolio state synced with Firestore.")
    except Exception as e:
        print(f"❌ Failed to sync portfolio state: {e}")

# Sync trade log
def sync_trade_log():
    """Push trade_log.json to Firestore."""
    try:
        init_firebase()
        if not os.path.exists(TRADE_LOG_PATH):
            print("⚠️ Trade log file not found, skipping sync.")
            return

        with open(TRADE_LOG_PATH, "r") as f:
            trades = json.load(f)

        db = firestore.client()
        db.collection("hedge_trade_log").document("latest").set({
            "trades": trades,
            "updated_at": firestore.SERVER_TIMESTAMP
        })
        print("✅ Trade log synced with Firestore.")
    except Exception as e:
        print(f"❌ Failed to sync trade log: {e}")

# Combined sync helper
def sync_all(state_map):
    """Sync both portfolio state and trade log in one call."""
    sync_portfolio_state(state_map)
    sync_trade_log()