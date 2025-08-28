import os
import time
from utils.firestore_client import get_db
from execution.sync_state import sync_positions, sync_nav

def main():
    ENV = os.environ.get("ENV", "dev")
    db = get_db()
    while True:
        # TODO: replace with real readers for balances/positions from exchange or local state
        positions_payload = {"items": []}
        nav_payload = {"series": [], "total_equity": 0, "realized_pnl": 0, "unrealized_pnl": 0, "peak_equity": 0, "drawdown": 0}
        try:
            sync_positions(db, positions_payload, ENV)
            sync_nav(db, nav_payload, ENV)
        except Exception as e:
            print(f"[hedge-sync] error: {e}")
        time.sleep(60)

if __name__ == "__main__":
    main()
