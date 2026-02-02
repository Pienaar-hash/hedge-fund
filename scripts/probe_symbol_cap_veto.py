#!/usr/bin/env python3
"""
Phase A.3: Probe script to validate symbol_cap veto enrichment.

Forces a single symbol_cap veto without placing any orders.
Validates that:
  - source_head appears at top level
  - veto_detail.constraint_geometry contains distance-to-wall metrics

Usage:
    PYTHONPATH=. python3 scripts/probe_symbol_cap_veto.py
"""

import json
import time
from pathlib import Path

from execution.risk_limits import check_order, RiskState, load_risk_config

VETO_LOG = Path("logs/execution/risk_vetoes.jsonl")


def count_lines() -> int:
    """Count current lines in veto log."""
    if not VETO_LOG.exists():
        return 0
    return sum(1 for _ in open(VETO_LOG))


def get_last_veto() -> dict | None:
    """Get the last veto record."""
    if not VETO_LOG.exists():
        return None
    lines = VETO_LOG.read_text().strip().split("\n")
    if not lines or not lines[-1]:
        return None
    return json.loads(lines[-1])


def main():
    print("=" * 60)
    print("Phase A.3: Symbol Cap Veto Probe")
    print("=" * 60)
    
    # Record baseline
    lines_before = count_lines()
    print(f"Veto log lines before: {lines_before}")
    
    # Load config
    cfg = load_risk_config()
    print(f"Loaded risk config")
    
    # Create minimal risk state
    state = RiskState()
    state.open_notional = 5000.0  # Simulate existing exposure
    state.open_positions = 1
    
    # Probe parameters designed to trigger symbol_cap:
    # - Request notional much larger than cap
    # - current_gross_notional already at 80% of NAV
    symbol = "BTCUSDT"
    nav = 10000.0
    current_gross = 8000.0  # Already 80% exposed
    requested_notional = 5000.0  # Request another 50% - will exceed 20% symbol cap
    
    print(f"\nProbe parameters:")
    print(f"  symbol: {symbol}")
    print(f"  nav: ${nav:,.2f}")
    print(f"  current_gross: ${current_gross:,.2f}")
    print(f"  requested_notional: ${requested_notional:,.2f}")
    print(f"  source_head: 'probe_test'")
    
    # Call check_order with source_head
    print(f"\nCalling check_order...")
    veto, details = check_order(
        symbol=symbol,
        side="BUY",
        requested_notional=requested_notional,
        price=50000.0,
        nav=nav,
        open_qty=0.1,
        now=time.time(),
        cfg=cfg,
        state=state,
        current_gross_notional=current_gross,
        lev=3.0,
        open_positions_count=1,
        tier_name="CORE",
        current_tier_gross_notional=0.0,
        source_head="probe_test",  # <-- Phase A.3 enrichment
    )
    
    print(f"\nResult:")
    print(f"  veto: {veto}")
    print(f"  reasons: {details.get('reasons', [])}")
    
    # Check if veto was logged
    lines_after = count_lines()
    print(f"\nVeto log lines after: {lines_after}")
    
    if lines_after > lines_before:
        print(f"  ✅ New veto logged (+{lines_after - lines_before} lines)")
        
        # Examine the last veto
        last_veto = get_last_veto()
        if last_veto:
            print(f"\nLast veto record validation:")
            
            # Check source_head
            source_head = last_veto.get("source_head")
            print(f"  source_head: {source_head}")
            if source_head == "probe_test":
                print(f"    ✅ source_head correctly logged")
            else:
                print(f"    ❌ source_head mismatch (expected 'probe_test')")
            
            # Check constraint_geometry
            geometry = last_veto.get("veto_detail", {}).get("constraint_geometry")
            print(f"  constraint_geometry present: {geometry is not None}")
            
            if geometry:
                print(f"    ✅ constraint_geometry logged")
                print(f"    Fields:")
                for k, v in geometry.items():
                    print(f"      {k}: {v}")
            else:
                print(f"    ❌ constraint_geometry missing")
                print(f"    veto_detail keys: {list(last_veto.get('veto_detail', {}).keys())}")
    else:
        print(f"  ⚠️ No new veto logged")
        print(f"  This may happen if the order wasn't vetoed or logging failed")
    
    print("\n" + "=" * 60)
    print("Probe complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
