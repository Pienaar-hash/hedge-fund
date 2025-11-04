from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

LotKey = Tuple[str, str]


@dataclass
class Lot:
    side: str  # "BUY" or "SELL"
    qty: float
    price: float


@dataclass
class Fill:
    symbol: str
    side: str
    qty: float
    price: float
    fee: float = 0.0
    position_side: Optional[str] = None
    reduce_only: bool = False


@dataclass
class CloseResult:
    symbol: str
    position_side: str
    closed_qty: float
    realized_pnl: float
    fees: float
    position_before: float
    position_after: float


class PositionTracker:
    """FIFO position tracker that yields realized PnL for reducing fills."""

    _EPS = 1e-9

    def __init__(self) -> None:
        self._lots: Dict[LotKey, Deque[Lot]] = defaultdict(deque)

    @staticmethod
    def _normalize_side(side: str) -> str:
        value = (side or "").upper()
        if value not in {"BUY", "SELL"}:
            raise ValueError(f"invalid side: {side}")
        return value

    @staticmethod
    def _normalize_position_side(position_side: Optional[str]) -> str:
        if not position_side:
            return "BOTH"
        value = position_side.upper()
        if value not in {"LONG", "SHORT", "BOTH"}:
            return "BOTH"
        return value

    def _key(self, symbol: str, position_side: Optional[str]) -> LotKey:
        return (symbol.upper(), self._normalize_position_side(position_side))

    @staticmethod
    def _net_position(lots: Deque[Lot]) -> float:
        net = 0.0
        for lot in lots:
            sign = 1.0 if lot.side == "BUY" else -1.0
            net += sign * lot.qty
        return net

    def apply_fill(self, fill: Fill) -> Optional[CloseResult]:
        """Apply a fill and return realized PnL details when the fill reduces exposure."""
        qty = float(fill.qty or 0.0)
        if qty <= 0:
            return None
        side = self._normalize_side(fill.side)
        key = self._key(fill.symbol, fill.position_side)
        lots = self._lots[key]
        position_before = self._net_position(lots)

        opposite = "SELL" if side == "BUY" else "BUY"
        remaining = qty
        realized = 0.0
        closed_qty = 0.0
        fees_closed = 0.0

        while remaining > self._EPS and lots and lots[0].side == opposite:
            lot = lots[0]
            matched = min(remaining, lot.qty)
            if side == "SELL":
                realized += (fill.price - lot.price) * matched
            else:  # side == "BUY"
                realized += (lot.price - fill.price) * matched
            lot.qty -= matched
            remaining -= matched
            closed_qty += matched
            if fill.qty:
                fees_closed += fill.fee * (matched / fill.qty)
            if lot.qty <= self._EPS:
                lots.popleft()
            else:
                lots[0] = lot

        if remaining > self._EPS:
            lots.append(Lot(side=side, qty=remaining, price=fill.price))

        position_after = self._net_position(lots)
        if closed_qty <= self._EPS:
            return None

        return CloseResult(
            symbol=fill.symbol,
            position_side=self._normalize_position_side(fill.position_side),
            closed_qty=closed_qty,
            realized_pnl=realized,
            fees=fees_closed,
            position_before=position_before,
            position_after=position_after,
        )
