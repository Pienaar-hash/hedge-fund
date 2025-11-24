import json, os

# canonical path (matches doctor + dashboard conventions)
PATH = "config/reserves.json"
OUT  = "config/reserves_non_futures.json"

# what counts as non-futures holdings
ALLOW = {"wallet", "spot", "off_exchange", "bybit", "treasury", "custody"}

def main():
    if not os.path.exists(PATH):
        print(f"❌ missing {PATH}")
        return
    with open(PATH) as f:
        raw = json.load(f)

    keep, excl = {}, 0.0
    total_keep = 0.0
    for k, v in raw.items():
        usd = float(v.get("usd", 0) or 0)
        src = str(v.get("source", "wallet")).lower()
        if src == "futures":
            excl += usd
            continue
        keep[k] = v
        total_keep += usd

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(keep, f, indent=2)

    print(f"✅ wrote {OUT}")
    print(f"Counted (non-futures): ${total_keep:,.2f} | Excluded (futures): ${excl:,.2f}")

if __name__ == "__main__":
    main()
