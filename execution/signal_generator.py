import os
import json
import time
import logging
from datetime import datetime

from execution.utils import fetch_candles, calculate_indicators, generate_signal
from firebase_admin import db, credentials, initialize_app
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Firebase setup
FIREBASE_SIGNALS = os.getenv("FIREBASE_SIGNALS")
DB_URL = os.getenv("FIREBASE_DB_URL")
cred_path = os.getenv("FIREBASE_CREDS")

if cred_path and DB_URL:
    cred = credentials.Certificate(cred_path)
    if not db._apps:
        initialize_app(cred, {"databaseURL": DB_URL})
else:
    logging.warning("Firebase not configured properly. Skipping sync.")

# Symbols to screen
symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
interval = "4h"

# Output folder
os.makedirs("data/signals", exist_ok=True)

while True:
    now = datetime.utcnow()
    if now.hour % 4 == 0 and now.minute < 5:
        logging.info("⏳ Running signal screening...")
        top_signals = []

        for symbol in symbols:
            try:
                df = fetch_candles(symbol=symbol, interval=interval)
                df = calculate_indicators(df)
                signal_data = generate_signal(df)

                if signal_data["signal"]:
                    payload = {
                        "symbol": symbol,
                        **signal_data,
                        "timestamp": now.isoformat()
                    }
                    top_signals.append(payload)

                    # Save locally
                    path = f"data/signals/{symbol.lower()}.json"
                    with open(path, "w") as f:
                        json.dump(payload, f, indent=2)

                    # Push to Firebase
                    if FIREBASE_SIGNALS:
                        db.reference(FIREBASE_SIGNALS).child(symbol).set(payload)

            except Exception as e:
                logging.error(f"Error processing {symbol}: {e}")

        if top_signals:
            logging.info(f"✅ {len(top_signals)} top signals generated:")
            for s in top_signals:
                logging.info(f"{s['symbol']} | {s['signal']} | z: {s['z_score']} | rsi: {s['rsi']} | momentum: {s['momentum']}")
        else:
            logging.info("No qualifying signals this round.")

        time.sleep(300)  # Avoid duplicate runs
    else:
        logging.info("Waiting for next 4h close...")
        time.sleep(60)
