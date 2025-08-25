#!/usr/bin/env python3
"""Flatten all hedge legs safely (MARKET closes with correct positionSide)."""
import argparse
from typing import List


from execution.exchange_utils import get_positions, place_market_order




def flatten(symbols: List[str], confirm: bool = False):
positions = get_positions()
to_close = []
for p in positions:
sym = p.get("symbol")
if symbols and sym not in symbols:
continue
qty = float(p.get("positionAmt", 0))
side = p.get("positionSide") # LONG or SHORT
if abs(qty) < 1e-12:
continue
close_side = "SELL" if side == "LONG" else "BUY"
to_close.append((sym, side, abs(qty), close_side))


if not to_close:
print("No hedge legs to close.")
return


print("Will close:")
for sym, side, qty, close_side in to_close:
print(f" {sym} {side} qty={qty} via {close_side} MARKET")


if not confirm:
print("Dry run. Use --confirm to send orders.")
return


for sym, side, qty, close_side in to_close:
place_market_order(symbol=sym, side=close_side, quantity=qty, position_side=side)
print(f"Closed {sym} {side} qty={qty}")




if __name__ == "__main__":
ap = argparse.ArgumentParser()
ap.add_argument("symbols", nargs="*", help="optional symbols to flatten (default: all)")
ap.add_argument("--confirm", action="store_true", help="send orders")
args = ap.parse_args()
flatten(args.symbols, confirm=args.confirm)