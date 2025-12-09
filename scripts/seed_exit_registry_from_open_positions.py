#!/usr/bin/env python3
"""
Seed or repair TP/SL registry entries using ledger reconciliation (report-only by default).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

from execution.position_ledger import (
    PositionTP_SL,
    build_position_ledger,
    load_positions_state,
    load_tp_sl_registry,
    reconcile_ledger_and_registry,
    save_tp_sl_registry,
    _compute_tp_sl_for_position,  # type: ignore  # internal helper for seeding
)

STATE_DIR = Path("logs/state")
STRATEGY_CONFIG_PATH = Path("config/strategy_config.json")


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def main() -> None:
    print("=" * 60)
    print("TP/SL Registry Seed Script (reconciliation-aware)")
    print("=" * 60)

    strategy_config = _load_json(STRATEGY_CONFIG_PATH)
    sl_atr_mult = float(strategy_config.get("sl_atr_mult", 2.0))
    tp_atr_mult = float(strategy_config.get("tp_atr_mult", 3.0))
    atr_pct = 0.007

    positions_state = load_positions_state(STATE_DIR)
    ledger = build_position_ledger(STATE_DIR)
    registry = load_tp_sl_registry(STATE_DIR)

    report = reconcile_ledger_and_registry(positions_state, ledger, registry)

    print(f"Open positions: {len(report.position_keys)}")
    print(f"Missing ledger entries: {len(report.missing_ledger_positions)}")
    print(f"Missing TP/SL entries: {len(report.missing_tp_sl_entries)}")
    print(f"Stale registry entries: {len(report.stale_tp_sl_entries)}")

    if not report.missing_tp_sl_entries and not report.missing_ledger_positions:
        print("\n✅ Nothing to seed; registry/ledger already aligned.")
        return

    seeded = 0
    for key in report.missing_tp_sl_entries:
        entry = ledger.get(key)
        if not entry:
            print(f"⚠️  Skipping {key}: missing ledger entry")
            continue
        tp, sl = _compute_tp_sl_for_position(entry.side, entry.entry_price, sl_atr_mult, tp_atr_mult, atr_pct)
        registry[key] = {
            "symbol": entry.symbol,
            "position_side": entry.side,
            "entry_price": float(entry.entry_price),
            "take_profit_price": float(tp),
            "stop_loss_price": float(sl),
            "qty": float(entry.qty),
            "enable_tp_sl": True,
            "created_at": time.time(),
            "source": "registry_seed_script",
            "metadata": {
                "sl_atr_mult": sl_atr_mult,
                "tp_atr_mult": tp_atr_mult,
                "atr_pct": atr_pct,
            },
        }
        entry.tp_sl = PositionTP_SL(tp=tp, sl=sl)
        seeded += 1
        print(f"✚  Seeded {key}: tp={tp:.4f} sl={sl:.4f}")

    # Persist registry
    save_tp_sl_registry(registry, STATE_DIR)
    print(f"\n✅ Seed complete. Added {seeded} entries. Total registry entries: {len(registry)}")
    if report.stale_tp_sl_entries:
        print("ℹ️  Stale registry entries present; run cleanup separately if desired.")


if __name__ == "__main__":
    main()
