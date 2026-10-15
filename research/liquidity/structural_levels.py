"""Ex-ante previous-day high/low structural levels.

PDH(D) = max(high) over completed UTC day D-1
PDL(D) = min(low)  over completed UTC day D-1

Levels become observable at 00:00:00 UTC on day D. No information from day D
may contribute to constructing its own PDH/PDL — this module enforces that by
grouping klines strictly by their own UTC calendar day and only ever reading
day D-1's aggregate to produce day D's level.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MS_PER_DAY = 24 * 60 * 60 * 1000


@dataclass(frozen=True)
class Level:
    symbol: str
    side: str  # "PDH" | "PDL"
    day_epoch_ms: int  # 00:00:00 UTC of day D, the day the level is observable/usable
    price: float
    source_day_epoch_ms: int  # 00:00:00 UTC of day D-1, the day the level was built from


def _day_start_ms(t_ms: int) -> int:
    return (t_ms // MS_PER_DAY) * MS_PER_DAY


def compute_daily_levels(symbol: str, open_time_ms: np.ndarray, high: np.ndarray, low: np.ndarray) -> list[Level]:
    """Build one PDH and one PDL per UTC day boundary present in the kline series.

    A day D's level uses only klines whose open_time falls in [D-1, D). The
    resulting Level.day_epoch_ms is D — the day the level becomes usable — while
    source_day_epoch_ms records D-1 for auditability.
    """
    if open_time_ms.shape[0] == 0:
        return []

    order = np.argsort(open_time_ms, kind="stable")
    ot = open_time_ms[order]
    hi = high[order]
    lo = low[order]

    day_index = ot // MS_PER_DAY
    unique_days = np.unique(day_index)

    levels: list[Level] = []
    for d in unique_days:
        mask = day_index == d
        source_day_start = int(d) * MS_PER_DAY
        next_day_start = source_day_start + MS_PER_DAY
        levels.append(
            Level(
                symbol=symbol,
                side="PDH",
                day_epoch_ms=next_day_start,
                price=float(np.max(hi[mask])),
                source_day_epoch_ms=source_day_start,
            )
        )
        levels.append(
            Level(
                symbol=symbol,
                side="PDL",
                day_epoch_ms=next_day_start,
                price=float(np.min(lo[mask])),
                source_day_epoch_ms=source_day_start,
            )
        )
    return levels


def levels_by_usable_day(levels: list[Level]) -> dict[int, dict[str, Level]]:
    """Index levels by (day_epoch_ms) -> {"PDH": Level, "PDL": Level}."""
    out: dict[int, dict[str, Level]] = {}
    for lvl in levels:
        out.setdefault(lvl.day_epoch_ms, {})[lvl.side] = lvl
    return out
