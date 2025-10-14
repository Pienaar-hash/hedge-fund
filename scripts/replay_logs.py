"""
Replay dry-run audit logs into Firestore for dashboard continuity.
Usage:
    python -m scripts.replay_logs --accelerated --seed-nav --seed-positions
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------
# Ensure project root on path
# ---------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.firestore_client import get_db  # noqa: E402

# Firestore client
db = get_db()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def normalize_order(event):
    """Transform dry-run audit order into Firestore document."""
    n = event.get("normalized", {}) or {}
    return {
        "timestamp": datetime.utcfromtimestamp(event.get("t", time.time())),
        "symbol": event.get("symbol"),
        "side": event.get("side"),
        "positionSide": event.get("positionSide"),
        "price": n.get("price"),
        "qty": n.get("qty"),
        "notional": event.get("notional"),
        "finalNotional": n.get("finalNotional"),
        "reduceOnly": event.get("reduceOnly", False),
        "phase": event.get("phase"),
        "mode": "replay",
    }


def inject_orders(log_dir, accelerated=False, delay=0.25):
    """Read audit_orders_*.jsonl logs and push to Firestore."""
    order_files = sorted(Path(log_dir).glob("audit_orders_*.jsonl"))
    total_files = len(order_files)
    print(
        f"Scanning for audit logs in {log_dir} … found {total_files} file"
        f"{'s' if total_files != 1 else ''}"
    )

    if not order_files:
        print("⚠️  No audit order logs found")
        return 0

    injected = 0
    for index, file in enumerate(order_files, start=1):
        print(f"[{index}/{total_files}] Replaying orders from {file}")
        symbol = file.stem.replace("audit_orders_", "")
        with open(file, "r") as f:
            for line_number, line in enumerate(f, start=1):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(
                        "⚠️  JSON decode error in "
                        f"{file}:{line_number} - {exc}"
                    )
                    continue

                doc = normalize_order(event)
                ts = str(event.get("t", time.time()))

                try:
                    db.collection("orders_replay").document(f"{symbol}_{ts}").set(doc)
                except Exception as exc:  # noqa: BLE001
                    print(
                        "❌  Failed to write order from "
                        f"{file}:{line_number} - {exc}"
                    )
                    continue

                print(f"[Firestore] Injected {symbol} {doc['side']} @ {doc['price']}")
                injected += 1

                if not accelerated:
                    time.sleep(delay)

    print(f"Injected {injected} order entries")
    return injected


def inject_nav(nav_file="logs/nav.jsonl"):
    """Optional: seed NAV history for dashboard continuity."""
    path = Path(nav_file)
    if not path.exists():
        print(f"⚠️  NAV file not found: {nav_file}")
        return 0

    print(f"Seeding NAV from {nav_file} …")
    nav_collection = db.collection("nav_replay")
    batch = db.batch()
    batch_size = 0
    seeded = 0
    last_line = 0

    def commit_batch():
        nonlocal batch, batch_size, seeded
        if batch_size == 0:
            return
        try:
            batch.commit()
            seeded += batch_size
        except Exception as exc:  # noqa: BLE001
            print(
                "❌  Failed to commit NAV batch ending at "
                f"{path}:{last_line} - {exc}"
            )
        finally:
            batch = db.batch()
            batch_size = 0

    with open(path, "r") as f:
        for line_number, line in enumerate(f, start=1):
            last_line = line_number
            try:
                nav = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    "⚠️  JSON decode error in "
                    f"{path}:{line_number} - {exc}"
                )
                continue

            doc = {
                "timestamp": datetime.utcfromtimestamp(nav.get("t", time.time())),
                "nav": nav.get("nav"),
                "balance": nav.get("balance"),
                "equity": nav.get("equity"),
                "mode": "replay",
            }

            try:
                batch.set(nav_collection.document(), doc)
                batch_size += 1
            except Exception as exc:  # noqa: BLE001
                print(
                    "❌  Failed to queue NAV entry from "
                    f"{path}:{line_number} - {exc}"
                )
                continue

            if batch_size >= 200:
                commit_batch()

    commit_batch()
    print(f"Seeded {seeded} NAV entries")
    return seeded


def inject_positions(positions_file="logs/positions.jsonl"):
    """Optional: seed live positions for dashboard continuity."""
    path = Path(positions_file)
    if not path.exists():
        print(f"⚠️  Positions file not found: {positions_file}")
        return 0

    print(f"Seeding positions from {positions_file} …")
    positions_collection = db.collection("positions_replay")
    batch = db.batch()
    batch_size = 0
    seeded = 0
    last_line = 0
    next_progress = 500

    def commit_batch():
        nonlocal batch, batch_size, seeded, next_progress
        if batch_size == 0:
            return
        try:
            batch.commit()
            seeded += batch_size
            while seeded >= next_progress:
                print(f"[Firestore] Injected {next_progress} positions so far …")
                next_progress += 500
        except Exception as exc:  # noqa: BLE001
            print(
                "❌  Failed to commit positions batch ending at "
                f"{path}:{last_line} - {exc}"
            )
        finally:
            batch = db.batch()
            batch_size = 0

    with open(path, "r") as f:
        for line_number, line in enumerate(f, start=1):
            last_line = line_number
            try:
                position = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    "⚠️  JSON decode error in "
                    f"{path}:{line_number} - {exc}"
                )
                continue

            doc = {
                "timestamp": datetime.utcfromtimestamp(position.get("t", time.time())),
                "symbol": position.get("symbol"),
                "positionSide": position.get("positionSide"),
                "entryPrice": position.get("entryPrice"),
                "unrealizedPnl": position.get("unrealizedPnl"),
                "mode": "replay",
            }

            if "size" in position:
                doc["size"] = position.get("size")
            elif "positionAmt" in position:
                doc["positionAmt"] = position.get("positionAmt")

            try:
                batch.set(positions_collection.document(), doc)
                batch_size += 1
            except Exception as exc:  # noqa: BLE001
                print(
                    "❌  Failed to queue position entry from "
                    f"{path}:{line_number} - {exc}"
                )
                continue

            if batch_size >= 200:
                commit_batch()

    commit_batch()
    print(f"Seeded {seeded} position entries")
    return seeded


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs", help="Directory with dry-run JSONL logs")
    parser.add_argument("--accelerated", action="store_true", help="Fast playback without sleep")
    parser.add_argument("--delay", type=float, default=0.25, help="Delay between events (sec)")
    parser.add_argument("--seed-nav", action="store_true", help="Seed NAV history from nav.jsonl")
    parser.add_argument(
        "--seed-positions",
        action="store_true",
        help="Seed position history from positions.jsonl",
    )
    args = parser.parse_args()

    nav_count = 0
    position_count = 0

    if args.seed_nav:
        nav_path = Path(args.log_dir) / "nav.jsonl"
        try:
            nav_count = inject_nav(nav_path)
            print(f"NAV seeding complete (entries: {nav_count}). Continuing …")
        except Exception as exc:  # noqa: BLE001
            print(f"❌  NAV seeding failed - {exc}. Continuing to order replay …")

    if args.seed_positions:
        positions_path = Path(args.log_dir) / "positions.jsonl"
        try:
            position_count = inject_positions(positions_path)
            print(
                "Position seeding complete "
                f"(entries: {position_count}). Continuing to order replay …"
            )
        except Exception as exc:  # noqa: BLE001
            print(
                "❌  Position seeding failed - "
                f"{exc}. Continuing to order replay …"
            )

    print("Replaying dry orders …")
    order_count = inject_orders(args.log_dir, accelerated=args.accelerated, delay=args.delay)

    summary = [f"orders: {order_count}"]
    if args.seed_positions:
        summary.append(f"positions: {position_count}")
    if args.seed_nav:
        summary.append(f"nav: {nav_count}")

    print(f"✅ Replay complete ({', '.join(summary)})")
