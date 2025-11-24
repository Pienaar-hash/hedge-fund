"""
Diagnostic utility: summarize current live trading state.
- Reads latest NAV snapshot
- Loads risk caps from config/risk_limits.json
- Aggregates veto counts from logs
- Displays current emitted intents and veto patterns
Usage:
    python -m scripts.diagnose_live_activity
"""

import os
from datetime import datetime

from dashboard.live_helpers import get_caps, get_nav_snapshot, get_veto_counts
from execution.signal_generator import generate_intents

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def sample_intents():
    now = datetime.utcnow().timestamp()
    try:
        intents = generate_intents(now)
    except Exception as e:
        print(f"[warn] could not generate intents: {e}")
        intents = []
    return intents[:10]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    caps = get_caps()
    nav_snapshot = get_nav_snapshot()
    veto_counts = get_veto_counts()

    print("\n=== LIVE SYSTEM DIAGNOSTIC ===")
    print(f"ENV: {os.getenv('ENV', 'unknown')} | DRY_RUN={os.getenv('DRY_RUN')}")
    print(f"Time: {datetime.utcnow().isoformat()} UTC\n")

    if nav_snapshot:
        nav = float(nav_snapshot.get("nav", 0.0) or 0.0)
        eq = float(nav_snapshot.get("equity", nav))
        ts = nav_snapshot.get("ts")
        print(f"NAV snapshot: {nav:.2f} | Equity: {eq:.2f} | ts={ts}")
    else:
        print("NAV snapshot: not found")

    print("\n--- Risk caps ---")
    print(f"max_trade_nav_pct: {caps.get('max_trade_nav_pct')}")
    print(f"max_gross_exposure_pct: {caps.get('max_gross_exposure_pct')}")
    print(f"max_symbol_exposure_pct: {caps.get('max_symbol_exposure_pct')}")
    print(f"min_notional: {caps.get('min_notional')}")

    print("\n--- Veto summary ---")
    if veto_counts:
        for veto, count in sorted(veto_counts.items(), key=lambda item: item[1], reverse=True)[:10]:
            print(f"{veto:30s} {int(count):5d}")
    else:
        print("No veto logs found.")

    print("\n--- Current emitted intents ---")
    intents = sample_intents()
    if not intents:
        print("No new intents generated.")
    else:
        for i in intents:
            print(f"{i['symbol']:8s} {i.get('signal')} @ {i.get('price')} "
                  f"gross={i.get('gross_usd')} lev={i.get('leverage')}")

    print("\n✅ Diagnostic complete.")


if __name__ == "__main__":
    main()
