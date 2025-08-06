import os
import json
import time
import firebase_admin
from firebase_admin import credentials, firestore

# Firebase setup
if not firebase_admin._apps:
    cred = credentials.Certificate("/root/hedge-fund/firebase_creds.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()
signal_dir = "data/signals"

def sync_signals():
    for filename in os.listdir(signal_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(signal_dir, filename)

            try:
                with open(filepath, "r") as f:
                    signal = json.load(f)

                symbol = signal.get("symbol") or filename.replace(".json", "")
                doc_ref = db.collection("signals").document(symbol)
                doc_ref.set(signal)

                print(f"✅ Synced to Firebase: {symbol}")

            except Exception as e:
                print(f"❌ Failed to sync {filename}:", e)

while True:
    sync_signals()
    time.sleep(30)
