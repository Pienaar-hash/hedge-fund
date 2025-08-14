import os, time, traceback, sys
from utils.firestore_client import get_db  # quick import smoke
from execution.sync_state import sync_leaderboard, sync_nav, sync_positions

INTERVAL = int(os.getenv("SYNC_INTERVAL_SEC", "15"))

def run_once():
    # idempotent one-shot syncs
    sync_leaderboard()
    sync_nav()
    sync_positions()

if __name__ == "__main__":
    while True:
        try:
            run_once()
            sys.stdout.write("✔ sync ok\n"); sys.stdout.flush()
        except Exception:
            traceback.print_exc()
            time.sleep(5)  # backoff on error
        time.sleep(INTERVAL)
