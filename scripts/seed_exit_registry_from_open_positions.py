#!/usr/bin/env python3
"""
One-shot script to repopulate the TP/SL registry from current open positions.

This fixes the scenario where the registry becomes empty (e.g., after restart,
concurrency lock issues, or registry corruption) and the exit scanner cannot
fire TP/SL triggers.

Usage:
    python scripts/seed_exit_registry_from_open_positions.py

This will:
    1. Read current positions from logs/state/positions.json
    2. Compute TP/SL levels using ATR-based config from strategy_config.json
    3. Write entries to logs/state/position_tp_sl.json
    4. Restore exit scanner functionality immediately
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Paths
STATE_DIR = Path("logs/state")
POSITIONS_PATH = STATE_DIR / "positions.json"
REGISTRY_PATH = STATE_DIR / "position_tp_sl.json"
STRATEGY_CONFIG_PATH = Path("config/strategy_config.json")


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file, return empty dict if not found."""
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    """Save data as JSON to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def compute_tp_sl_from_atr(
    entry_price: float,
    direction: str,
    atr_value: Optional[float],
    sl_atr_mult: float = 2.0,
    tp_atr_mult: float = 3.0,
) -> tuple[float, float]:
    """
    Compute TP/SL prices using ATR multipliers.
    
    If ATR is not available, use a fallback percentage-based approach.
    """
    if atr_value is None or atr_value <= 0:
        # Fallback: use 2% SL, 3% TP
        sl_pct = 0.02
        tp_pct = 0.03
        if direction == "LONG":
            sl = entry_price * (1 - sl_pct)
            tp = entry_price * (1 + tp_pct)
        else:
            sl = entry_price * (1 + sl_pct)
            tp = entry_price * (1 - tp_pct)
        return sl, tp
    
    # ATR-based calculation
    sl_distance = atr_value * sl_atr_mult
    tp_distance = atr_value * tp_atr_mult
    
    if direction == "LONG":
        sl = entry_price - sl_distance
        tp = entry_price + tp_distance
    else:
        sl = entry_price + sl_distance
        tp = entry_price - tp_distance
    
    return sl, tp


def get_estimated_atr(symbol: str, entry_price: float) -> Optional[float]:
    """
    Get estimated ATR for a symbol.
    
    This is a rough approximation based on typical crypto volatility.
    For production, this should fetch real ATR from market data.
    """
    # Rough ATR estimates as percentage of price (typical 14-period ATR)
    atr_pct_estimates = {
        "BTCUSDT": 0.003,   # ~0.3% daily
        "ETHUSDT": 0.005,   # ~0.5% daily
        "SOLUSDT": 0.007,   # ~0.7% daily
        "DOGEUSDT": 0.008,  # ~0.8% daily
        "LTCUSDT": 0.005,   # ~0.5% daily
        "SUIUSDT": 0.008,   # ~0.8% daily
        "WIFUSDT": 0.010,   # ~1.0% daily
        "LINKUSDT": 0.006,  # ~0.6% daily
    }
    
    pct = atr_pct_estimates.get(symbol, 0.005)  # default 0.5%
    return entry_price * pct


def main():
    print("=" * 60)
    print("TP/SL Registry Seed Script")
    print("=" * 60)
    
    # Load positions
    positions_data = load_json(POSITIONS_PATH)
    rows = positions_data.get("rows", [])
    
    if not rows:
        print("❌ No positions found in positions.json")
        return
    
    print(f"📊 Found {len(rows)} open positions")
    
    # Load strategy config
    strategy_config = load_json(STRATEGY_CONFIG_PATH)
    sl_atr_mult = strategy_config.get("sl_atr_mult", 2.0)
    tp_atr_mult = strategy_config.get("tp_atr_mult", 3.0)
    
    print(f"📐 Using SL ATR mult: {sl_atr_mult}, TP ATR mult: {tp_atr_mult}")
    
    # Load existing registry
    existing_registry = load_json(REGISTRY_PATH)
    existing_entries = existing_registry.get("entries", {})
    
    # Build new entries
    new_entries = {}
    
    for pos in rows:
        symbol = pos.get("symbol")
        position_side = pos.get("positionSide", "LONG")
        entry_price = pos.get("entryPrice", 0.0)
        qty = pos.get("qty", 0.0)
        
        if not symbol or entry_price <= 0 or qty <= 0:
            print(f"⚠️  Skipping {symbol}: invalid entry_price={entry_price} or qty={qty}")
            continue
        
        # Create registry key
        key = f"{symbol}:{position_side}"
        
        # Skip if already registered
        if key in existing_entries:
            print(f"✓  {key} already registered")
            continue
        
        # Estimate ATR
        atr_value = get_estimated_atr(symbol, entry_price)
        
        # Compute TP/SL
        sl_price, tp_price = compute_tp_sl_from_atr(
            entry_price=entry_price,
            direction=position_side,
            atr_value=atr_value,
            sl_atr_mult=sl_atr_mult,
            tp_atr_mult=tp_atr_mult,
        )
        
        # Create entry
        new_entries[key] = {
            "symbol": symbol,
            "position_side": position_side,
            "entry_price": entry_price,
            "take_profit_price": tp_price,
            "stop_loss_price": sl_price,
            "qty": qty,
            "enable_tp_sl": True,
            "created_at": time.time(),
            "source": "registry_seed_script",
            "metadata": {
                "atr_estimate": atr_value,
                "sl_atr_mult": sl_atr_mult,
                "tp_atr_mult": tp_atr_mult,
            }
        }
        
        print(f"✚  {key}: entry={entry_price:.4f} SL={sl_price:.4f} TP={tp_price:.4f}")
    
    if not new_entries:
        print("\n✅ No new entries to add (all positions already registered)")
        return
    
    # Merge with existing entries
    merged_entries = {**existing_entries, **new_entries}
    
    # Write registry
    registry_data = {
        "entries": merged_entries,
        "updated_at": time.time(),
    }
    
    save_json(REGISTRY_PATH, registry_data)
    
    print(f"\n✅ Registry updated with {len(new_entries)} new entries")
    print(f"📁 Total entries: {len(merged_entries)}")
    print(f"💾 Saved to: {REGISTRY_PATH}")
    
    # Summary
    print("\n" + "=" * 60)
    print("REGISTRY CONTENTS:")
    print("=" * 60)
    for key, entry in merged_entries.items():
        symbol = entry.get("symbol", key.split(":")[0])
        sl = entry.get("stop_loss_price", 0)
        tp = entry.get("take_profit_price", 0)
        print(f"  {symbol}: SL={sl:.4f} TP={tp:.4f}")
    
    print("\n🚀 Exit scanner should now detect TP/SL triggers!")
    print("   Restart executor: sudo supervisorctl restart hedge:hedge-executor")


if __name__ == "__main__":
    main()
