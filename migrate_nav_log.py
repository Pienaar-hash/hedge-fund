import json
from collections import defaultdict
from datetime import datetime

# Load original nav_log.json
with open("nav_log.json", "r") as f:
    raw_logs = json.load(f)

# Skip if already in new format
if all("equity" in entry and isinstance(entry["equity"], dict) for entry in raw_logs):
    print("✅ nav_log.json already in consolidated format. No migration needed.")
    exit()

# Group by timestamp (rounded to seconds)
grouped = defaultdict(dict)
for entry in raw_logs:
    if "symbol" not in entry or "equity" not in entry:
        continue  # Skip malformed or already-converted entries

    ts = entry["timestamp"]
    ts_rounded = datetime.fromisoformat(ts).replace(microsecond=0).isoformat()
    symbol = entry["symbol"]
    equity = entry["equity"]
    grouped[ts_rounded][symbol] = equity

# Build new log format
new_logs = []
for ts, equities in grouped.items():
    new_logs.append({
        "timestamp": ts,
        "equity": equities
    })

# Sort and save
new_logs.sort(key=lambda x: x["timestamp"])

with open("nav_log.json", "w") as f:
    json.dump(new_logs, f, indent=2)

print(f"✅ Migration complete: {len(new_logs)} consolidated entries saved.")
