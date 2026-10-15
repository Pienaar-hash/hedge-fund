"""Crossing, classification, sweep/reclaim, accepted-breakout, entry-proxy and
outcome logic for H-LIQ-001.

All functions are pure and operate on a single symbol's full chronologically
sorted trade tape (t: int64 ms epoch, p: float64 price, q: float64 quantity).
Every timestamp comparison follows the frozen ordering from the
pre-registration:

    level_timestamp < t0 <= tC <= entry_timestamp <= exit_timestamp

No function here ever looks at a trade whose timestamp is needed only to
decide something that must be known strictly earlier.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from research.liquidity.structural_levels import Level

TC_OFFSET_MS = 5 * 60 * 1000
VWAP_WINDOW_MS = 30 * 1000
ENTRY_DELAY_MS = 1 * 1000
ENTRY_TOLERANCE_MS = 5 * 1000

PENETRATION_CAP = 1.0025
RECLAIM_UPPER = 0.9998
ACCEPT_UPPER = 1.0005
PENETRATION_CAP_LOWER = 0.9975
RECLAIM_LOWER = 1.0002
ACCEPT_LOWER = 0.9995


@dataclass(frozen=True)
class Crossing:
    level: Level
    t0_ms: int
    t0_index: int


@dataclass(frozen=True)
class Classification:
    crossing: Crossing
    tC_ms: int
    extreme_price: float
    vwap30: Optional[float]
    state: str  # "SWEEP_RECLAIM" | "ACCEPTED_BREAKOUT" | "NEITHER" | "INVALID_NO_VWAP_DATA"


@dataclass(frozen=True)
class ExecutionProxy:
    entry_t_ms: int
    entry_price: float


def find_first_crossing(
    t: np.ndarray, p: np.ndarray, day_start_ms: int, day_end_ms: int, level: Level
) -> Optional[Crossing]:
    """First outside crossing of `level` whose trade timestamp falls in
    [day_start_ms, day_end_ms). The previous trade may belong to the prior day
    (it is simply the immediately preceding tape entry)."""
    start_idx = int(np.searchsorted(t, day_start_ms, side="left"))
    end_idx = int(np.searchsorted(t, day_end_ms, side="left"))
    if start_idx == 0:
        start_idx = 1  # no previous trade exists to compare against
    for i in range(max(start_idx, 1), end_idx):
        prev_price = p[i - 1]
        cur_price = p[i]
        if level.side == "PDH":
            if prev_price <= level.price < cur_price:
                return Crossing(level=level, t0_ms=int(t[i]), t0_index=i)
        else:  # PDL
            if prev_price >= level.price > cur_price:
                return Crossing(level=level, t0_ms=int(t[i]), t0_index=i)
    return None


def compute_vwap(t: np.ndarray, p: np.ndarray, q: np.ndarray, window_end_ms: int, window_ms: int) -> Optional[float]:
    lo = window_end_ms - window_ms
    left = int(np.searchsorted(t, lo, side="left"))
    right = int(np.searchsorted(t, window_end_ms, side="right"))
    if right <= left:
        return None
    notional = float(np.dot(p[left:right], q[left:right]))
    volume = float(np.sum(q[left:right]))
    if volume <= 0:
        return None
    return notional / volume


def classify(t: np.ndarray, p: np.ndarray, q: np.ndarray, crossing: Crossing) -> Classification:
    tC = crossing.t0_ms + TC_OFFSET_MS
    end_idx = int(np.searchsorted(t, tC, side="right"))
    start_idx = crossing.t0_index
    window_prices = p[start_idx:end_idx]

    vwap30 = compute_vwap(t, p, q, tC, VWAP_WINDOW_MS)
    if vwap30 is None:
        return Classification(
            crossing=crossing, tC_ms=tC, extreme_price=float("nan"), vwap30=None, state="INVALID_NO_VWAP_DATA"
        )

    level_price = crossing.level.price
    if crossing.level.side == "PDH":
        extreme = float(np.max(window_prices))
        penetration_ok = extreme <= level_price * PENETRATION_CAP
        reclaim_ok = vwap30 <= level_price * RECLAIM_UPPER
        accepted_ok = vwap30 >= level_price * ACCEPT_UPPER
    else:
        extreme = float(np.min(window_prices))
        penetration_ok = extreme >= level_price * PENETRATION_CAP_LOWER
        reclaim_ok = vwap30 >= level_price * RECLAIM_LOWER
        accepted_ok = vwap30 <= level_price * ACCEPT_LOWER

    if penetration_ok and reclaim_ok:
        state = "SWEEP_RECLAIM"
    elif accepted_ok:
        state = "ACCEPTED_BREAKOUT"
    else:
        state = "NEITHER"

    return Classification(crossing=crossing, tC_ms=tC, extreme_price=extreme, vwap30=vwap30, state=state)


def penetration_bps(classification: Classification) -> float:
    level_price = classification.crossing.level.price
    if classification.crossing.level.side == "PDH":
        return (classification.extreme_price / level_price - 1.0) * 10_000.0
    return (1.0 - classification.extreme_price / level_price) * 10_000.0


def find_trade_at_or_after(
    t: np.ndarray, p: np.ndarray, target_ms: int, tolerance_ms: int
) -> Optional[ExecutionProxy]:
    idx = int(np.searchsorted(t, target_ms, side="left"))
    if idx >= len(t):
        return None
    if t[idx] > target_ms + tolerance_ms:
        return None
    return ExecutionProxy(entry_t_ms=int(t[idx]), entry_price=float(p[idx]))


def find_entry(t: np.ndarray, p: np.ndarray, tC_ms: int) -> Optional[ExecutionProxy]:
    target = tC_ms + ENTRY_DELAY_MS
    return find_trade_at_or_after(t, p, target, ENTRY_TOLERANCE_MS)


def find_outcome_price(t: np.ndarray, p: np.ndarray, tC_ms: int, horizon_ms: int) -> Optional[ExecutionProxy]:
    target = tC_ms + horizon_ms
    return find_trade_at_or_after(t, p, target, ENTRY_TOLERANCE_MS)


def signed_return(side: str, entry_price: float, exit_price: float) -> float:
    log_ratio = math.log(exit_price / entry_price)
    return -log_ratio if side == "PDH" else log_ratio
