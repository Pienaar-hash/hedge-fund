"""
Diagnostic utility: summarize current live trading state.
- Reads latest NAV snapshot
- Loads risk caps from config/risk_limits.json
- Aggregates veto counts from logs
- Displays current emitted intents and veto patterns
Usage:
    python -m scripts.diagnose_live_activity
"""

import json
import os
from pathlib import Path
from datetime import datetime
from collections import Counter

from execution.risk_limits import RiskGate
from execution.signal_generator import generate_intents

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_json(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with p.open() as f:
        return json.load(f)


def load_risk():
    cfg = load_json("config/risk_limits.json")
    gate = RiskGate(cfg)
    return gate, cfg


def get_nav_snapshot():
    nav_path = Path("logs/nav.jsonl")
    if not nav_path.exists():
        return None
    *_, last = nav_path.read_text().splitlines()
    try:
        nav = json.loads(last)
        return nav.get("nav", 0), nav.get("equity", 0), nav.get("t", 0)
    except json.JSONDecodeError:
        return None


def summarize_vetoes(log_dir="logs"):
    veto_counts = Counter()
    veto_samples = {}

    for file in Path(log_dir).glob("veto_exec_*.json"):
        for line in file.read_text().splitlines():
            try:
                obj = json.loads(line)
                if "veto" in obj:
                    vetoes = obj["veto"]
                    if isinstance(vetoes, list):
                        for v in vetoes:
                            veto_counts[v] += 1
                            veto_samples.setdefault(v, obj.get("symbol"))
                    elif isinstance(vetoes, str):
                        veto_counts[vetoes] += 1
                        veto_samples.setdefault(vetoes, obj.get("symbol"))
            except json.JSONDecodeError:
                continue
    return veto_counts, veto_samples


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
    gate, cfg = load_risk()
    nav_tuple = get_nav_snapshot()
    veto_counts, veto_samples = summarize_vetoes()

    print("\n=== LIVE SYSTEM DIAGNOSTIC ===")
    print(f"ENV: {os.getenv('ENV', 'unknown')} | DRY_RUN={os.getenv('DRY_RUN')}")
    print(f"Time: {datetime.utcnow().isoformat()} UTC\n")

    if nav_tuple:
        nav, eq, ts = nav_tuple
        print(f"NAV snapshot: {nav:.2f} | Equity: {eq:.2f} | ts={ts}")
    else:
        print("NAV snapshot: not found")

    print("\n--- Risk caps ---")
    print(f"max_trade_nav_pct: {gate.sizing.get('max_trade_nav_pct')}")
    print(f"max_gross_exposure_pct: {gate.sizing.get('max_gross_exposure_pct')}")
    print(f"max_symbol_exposure_pct: {gate.sizing.get('max_symbol_exposure_pct')}")
    print(f"min_notional: {gate.min_notional}")

    print("\n--- Veto summary ---")
    if veto_counts:
        for veto, count in veto_counts.most_common(10):
            print(f"{veto:30s} {count:5d}  (example: {veto_samples[veto]})")
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
