import os
import json
import time
import logging
from datetime import datetime
import traceback

from execution.utils import fetch_candles, calculate_indicators, generate_signal
from firebase_admin import db, credentials, initialize_app
from dotenv import load_dotenv
from execution.exchange_utils import send_telegram

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

send_telegram("🚦 Signal generator started. Waiting for 4h close...")

while True:
    now = datetime.utcnow()
    if True:
        logging.info("⏳ Running signal screening...")
        send_telegram("⏳ Running signal screening on 4h close...")

        signal_payloads = []

        for symbol in symbols:
            try:
                df = fetch_candles(symbol=symbol, interval=interval)
                if df is None or df.empty:
                    logging.warning(f"⚠️ No data fetched for {symbol}. Skipping.")
                    continue
                logging.info(f"✅ {symbol} data fetched:\n{df.tail(1).to_string()}")

                df = calculate_indicators(df)
                signal_data = generate_signal(df)

                payload = {
                    "symbol": symbol,
                    **signal_data,
                    "timestamp": now.isoformat()
                }

                # Save locally
                path = f"data/signals/{symbol.lower()}.json"
                with open(path, "w") as f:
                    json.dump(payload, f, indent=2)

                # Push to Firebase
                if FIREBASE_SIGNALS:
                    db.reference(FIREBASE_SIGNALS).child(symbol).set(payload)

                payload['score'] = abs(payload.get('z_score', 0))  # use z-score as strength proxy
                signal_payloads.append(payload)

            except Exception as e:
                logging.error(f"Error processing {symbol}: {e}\n{traceback.format_exc()}")

        # 📊 Telegram Summary Logic
        filtered = [p for p in signal_payloads if p['signal'] and abs(p['z_score']) > 1.0]
        filtered.sort(key=lambda x: abs(x['z_score']), reverse=True)

        summary_lines = []
        for p in filtered[:3]:
            emoji = "✅" if p['signal'] == 'buy' else "❌"
            trend = "UP" if p['momentum'] > 0 else "DOWN"
            line = (
                f"{emoji} {p['signal'].upper()} | {p['symbol']} | Z: {p['z_score']:+.2f} | RSI: {p['rsi']:.1f} | Trend: {trend}"
            )
            summary_lines.append(line)

        if not summary_lines:
            summary_lines.append("⚠️ No strong signals this cycle.")

        summary = "\n".join(summary_lines)
        logging.info(f"📊 4h Signal Summary:\n{summary}")
        send_telegram(f"📊 4h Signal Summary:\n{summary}")

        time.sleep(300)  # Restore default 5-min delay after 4h close
    else:
        logging.info("Waiting for next 4h close...")
        time.sleep(60)
