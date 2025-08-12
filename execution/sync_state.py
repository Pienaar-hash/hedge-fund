import os
import time
from datetime import datetime, timezone
import firebase_admin
from firebase_admin import credentials, firestore
from execution.utils import load_json
from execution.telegram_utils import send_telegram

# === Config ===
INTERVAL = int(os.getenv("SYNC_INTERVAL", 60))
TELEGRAM_COOLDOWN = int(os.getenv("TELEGRAM_COOLDOWN", 3600))  # 1 hour default
STATE_PATH = "synced_state.json"
LEADERBOARD_PATH = "logs/leaderboard.json"
NAV_LOG_PATH = "nav_log.json"

# === Firebase Init ===
FIREBASE_CREDS_PATH = os.getenv("FIREBASE_CREDS_PATH", "config/firebase_creds.json")
if not os.path.exists(FIREBASE_CREDS_PATH):
    raise FileNotFoundError(f"❌ Firebase credentials not found at {FIREBASE_CREDS_PATH}")

cred = credentials.Certificate(FIREBASE_CREDS_PATH)
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()

# === Helpers ===
def compute_leaderboard(portfolio_state):
    leaderboard = []
    for symbol, info in portfolio_state.items():
        pnl = info.get("pnl", 0)
        entry_price = info.get("entry", None)
        pct_return = 0
        if entry_price and entry_price != 0:
            pct_return = (pnl / (entry_price * info.get("qty", 1))) * 100
        leaderboard.append({
            "symbol": symbol,
            "pnl": round(pnl, 2),
            "pct_return": round(pct_return, 2)
        })
    leaderboard.sort(key=lambda x: x["pnl"], reverse=True)
    return leaderboard

def save_and_push_leaderboard(leaderboard):
    from execution.utils import save_json
    save_json(LEADERBOARD_PATH, leaderboard)
    db.collection("leaderboard").document("latest").set({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "leaderboard": leaderboard
    })

def push_nav_log():
    nav_data = load_json(NAV_LOG_PATH)
    db.collection("nav_log").document("latest").set({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "nav": nav_data
    })

def send_telegram_snapshot(leaderboard):
    if os.getenv("TELEGRAM_ENABLED") != "1":
        return
    top3 = leaderboard[:3]
    lines = ["📊 Leaderboard:"]
    for i, entry in enumerate(top3, start=1):
        lines.append(f"{i}️⃣ {entry['symbol']} {entry['pnl']} USDT")
    send_telegram("\n".join(lines), silent=True)

# === Main Loop ===
if __name__ == "__main__":
    last_telegram_push = 0
    while True:
        try:
            print(f"🔄 hedge-sync starting (interval={INTERVAL}s) — {datetime.now(timezone.utc)}")
            state = load_json(STATE_PATH)
            print(f"📥 Local state loaded: {state}")

            # Push portfolio state
            db.collection("portfolio_state").document("latest").set({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "state": state
            })
            print(f"✅ Synced portfolio state with Firestore — {datetime.now(timezone.utc)}")

            # Compute & push leaderboard
            leaderboard = compute_leaderboard(state)
            save_and_push_leaderboard(leaderboard)
            print(f"🏆 Leaderboard computed & pushed — {leaderboard}")

            # Push NAV log
            push_nav_log()
            print(f"📈 NAV log pushed — {datetime.now(timezone.utc)}")

            # Telegram snapshot on cooldown
            if time.time() - last_telegram_push >= TELEGRAM_COOLDOWN:
                send_telegram_snapshot(leaderboard)
                last_telegram_push = time.time()
                print(f"📨 Telegram leaderboard snapshot sent — {datetime.now(timezone.utc)}")

        except Exception as e:
            print(f"❌ Sync loop error: {e}")

        time.sleep(INTERVAL)
