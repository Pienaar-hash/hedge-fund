"""Orchestrates H-LIQ-001 event construction for one symbol's full trade tape:

    structural levels -> first crossings -> 60min overlap embargo
    -> classification -> matching features -> entry/outcome proxies
    -> primary event records + matched sweep/acceptance control pairs

Every function here is pure given its inputs (trades, klines, levels); no
network or filesystem access happens in this module.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from research.liquidity.structural_levels import Level, levels_by_usable_day
from research.liquidity.sweep_reclaim import (
    Crossing,
    classify,
    find_entry,
    find_first_crossing,
    find_outcome_price,
    penetration_bps,
    signed_return,
)

MS_PER_DAY = 24 * 60 * 60 * 1000
EMBARGO_MS = 60 * 60 * 1000
MATCH_LOOKBACK_MS = 30 * 60 * 1000
HORIZONS_MS = {"5m": 5 * 60 * 1000, "15m": 15 * 60 * 1000, "60m": 60 * 60 * 1000}
PRIMARY_HORIZON = "15m"


@dataclass(frozen=True)
class MatchFeatures:
    realized_vol_30m: Optional[float]
    quote_notional_30m: Optional[float]
    utc_4h_block: int
    quarter: str
    symbol: str
    side: str


@dataclass
class Event:
    event_id: str
    symbol: str
    side: str
    level_price: float
    t0_ms: int
    tC_ms: int
    state: str
    penetration_bps: Optional[float]
    vwap30: Optional[float]
    features: Optional[MatchFeatures]
    entry_t_ms: Optional[int] = None
    entry_price: Optional[float] = None
    outcomes: dict = field(default_factory=dict)
    reject_reason: Optional[str] = None


@dataclass(frozen=True)
class MatchedPair:
    sweep_event_id: str
    control_event_id: str
    match_distance: float


def _utc_4h_block(t_ms: int) -> int:
    hour = (t_ms // (60 * 60 * 1000)) % 24
    return int(hour // 4)


def _quarter(t_ms: int) -> str:
    import datetime as dt

    d = dt.datetime.fromtimestamp(t_ms / 1000, tz=dt.timezone.utc)
    q = (d.month - 1) // 3 + 1
    return f"{d.year}Q{q}"


def _event_id(symbol: str, side: str, t0_ms: int) -> str:
    return hashlib.sha256(f"{symbol}|{side}|{t0_ms}".encode()).hexdigest()[:16]


def _realized_vol_30m(kt: np.ndarray, kc: np.ndarray, t0_ms: int) -> Optional[float]:
    """Realized volatility from completed 1m bar log-returns strictly before t0."""
    closed_before = kt + 60_000 <= t0_ms
    idx = np.nonzero(closed_before)[0]
    if idx.shape[0] < 2:
        return None
    window_start = t0_ms - MATCH_LOOKBACK_MS
    sel = idx[kt[idx] >= window_start]
    if sel.shape[0] < 2:
        return None
    closes = kc[sel]
    rets = np.diff(np.log(closes))
    if rets.shape[0] == 0:
        return None
    return float(np.sqrt(np.sum(rets ** 2)))


def _quote_notional_30m(t: np.ndarray, p: np.ndarray, q: np.ndarray, t0_ms: int) -> Optional[float]:
    window_start = t0_ms - MATCH_LOOKBACK_MS
    left = int(np.searchsorted(t, window_start, side="left"))
    right = int(np.searchsorted(t, t0_ms, side="left"))
    if right <= left:
        return None
    return float(np.dot(p[left:right], q[left:right]))


def build_matching_features(
    symbol: str, side: str, t0_ms: int, t: np.ndarray, p: np.ndarray, q: np.ndarray, kt: np.ndarray, kc: np.ndarray
) -> MatchFeatures:
    return MatchFeatures(
        realized_vol_30m=_realized_vol_30m(kt, kc, t0_ms),
        quote_notional_30m=_quote_notional_30m(t, p, q, t0_ms),
        utc_4h_block=_utc_4h_block(t0_ms),
        quarter=_quarter(t0_ms),
        symbol=symbol,
        side=side,
    )


def _collect_candidate_crossings(
    symbol: str, t: np.ndarray, p: np.ndarray, levels: list[Level]
) -> list[Crossing]:
    by_day = levels_by_usable_day(levels)
    candidates: list[Crossing] = []
    for day_start, sides in sorted(by_day.items()):
        day_end = day_start + MS_PER_DAY
        for side_name in ("PDH", "PDL"):
            level = sides.get(side_name)
            if level is None:
                continue
            crossing = find_first_crossing(t, p, day_start, day_end, level)
            if crossing is not None:
                candidates.append(crossing)
    candidates.sort(key=lambda c: c.t0_ms)
    return candidates


def build_events_for_symbol(
    symbol: str,
    t: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    kt: np.ndarray,
    kh: np.ndarray,
    kl: np.ndarray,
    kc: np.ndarray,
    levels: list[Level],
) -> list[Event]:
    """Build the full per-symbol event list (qualified + rejected), applying the
    60-minute per-symbol overlap embargo chronologically across both level sides."""
    candidates = _collect_candidate_crossings(symbol, t, p, levels)

    events: list[Event] = []
    prior_accepted_tC: Optional[int] = None

    for crossing in candidates:
        if prior_accepted_tC is not None and crossing.t0_ms < prior_accepted_tC + EMBARGO_MS:
            events.append(
                Event(
                    event_id=_event_id(symbol, crossing.level.side, crossing.t0_ms), symbol=symbol,
                    side=crossing.level.side, level_price=crossing.level.price, t0_ms=crossing.t0_ms,
                    tC_ms=crossing.t0_ms + 5 * 60 * 1000, state="REJECTED", penetration_bps=None, vwap30=None,
                    features=None, reject_reason="overlap_embargo",
                )
            )
            continue  # not accepted; does not update prior_accepted_tC

        classification = classify(t, p, q, crossing)
        prior_accepted_tC = classification.tC_ms  # accepted for embargo purposes regardless of state

        event_id = _event_id(symbol, crossing.level.side, crossing.t0_ms)

        if classification.state == "INVALID_NO_VWAP_DATA":
            events.append(
                Event(
                    event_id=event_id, symbol=symbol, side=crossing.level.side, level_price=crossing.level.price,
                    t0_ms=crossing.t0_ms, tC_ms=classification.tC_ms, state="REJECTED", penetration_bps=None,
                    vwap30=None, features=None, reject_reason="no_vwap_data",
                )
            )
            continue

        if classification.state == "NEITHER":
            events.append(
                Event(
                    event_id=event_id, symbol=symbol, side=crossing.level.side, level_price=crossing.level.price,
                    t0_ms=crossing.t0_ms, tC_ms=classification.tC_ms, state="REJECTED",
                    penetration_bps=penetration_bps(classification), vwap30=classification.vwap30, features=None,
                    reject_reason="deadband_neither",
                )
            )
            continue

        features = build_matching_features(symbol, crossing.level.side, crossing.t0_ms, t, p, q, kt, kc)
        entry = find_entry(t, p, classification.tC_ms)
        if entry is None:
            events.append(
                Event(
                    event_id=event_id, symbol=symbol, side=crossing.level.side, level_price=crossing.level.price,
                    t0_ms=crossing.t0_ms, tC_ms=classification.tC_ms, state="REJECTED",
                    penetration_bps=penetration_bps(classification), vwap30=classification.vwap30,
                    features=features, reject_reason="no_execution_proxy",
                )
            )
            continue

        outcomes: dict[str, Optional[float]] = {}
        for horizon_name, horizon_ms in HORIZONS_MS.items():
            exit_proxy = find_outcome_price(t, p, classification.tC_ms, horizon_ms)
            if exit_proxy is None:
                outcomes[horizon_name] = None
            else:
                outcomes[horizon_name] = signed_return(crossing.level.side, entry.entry_price, exit_proxy.entry_price)

        events.append(
            Event(
                event_id=event_id, symbol=symbol, side=crossing.level.side, level_price=crossing.level.price,
                t0_ms=crossing.t0_ms, tC_ms=classification.tC_ms, state=classification.state,
                penetration_bps=penetration_bps(classification), vwap30=classification.vwap30, features=features,
                entry_t_ms=entry.entry_t_ms, entry_price=entry.entry_price, outcomes=outcomes,
            )
        )

    return events


def _feature_distance(a: MatchFeatures, b: MatchFeatures) -> Optional[float]:
    if a.realized_vol_30m is None or b.realized_vol_30m is None:
        return None
    if a.quote_notional_30m is None or b.quote_notional_30m is None:
        return None
    if a.realized_vol_30m <= 0 or b.realized_vol_30m <= 0:
        return None
    if a.quote_notional_30m <= 0 or b.quote_notional_30m <= 0:
        return None
    rv_ratio = a.realized_vol_30m / b.realized_vol_30m
    vol_ratio = a.quote_notional_30m / b.quote_notional_30m
    if not (0.5 <= rv_ratio <= 2.0) or not (0.5 <= vol_ratio <= 2.0):
        return None
    return abs(math.log(rv_ratio)) + abs(math.log(vol_ratio))


def match_events(events: list[Event]) -> list[MatchedPair]:
    """Match each qualified SWEEP_RECLAIM event to its nearest unused
    ACCEPTED_BREAKOUT control sharing symbol/side/quarter/4h-block, per the
    frozen matching algorithm. Deterministic tie-break: earliest control t0."""
    sweeps = sorted(
        [e for e in events if e.state == "SWEEP_RECLAIM" and e.features is not None],
        key=lambda e: e.t0_ms,
    )
    controls = [e for e in events if e.state == "ACCEPTED_BREAKOUT" and e.features is not None]

    used_control_ids: set[str] = set()
    pairs: list[MatchedPair] = []

    for sweep in sweeps:
        eligible = [
            c
            for c in controls
            if c.event_id not in used_control_ids
            and c.symbol == sweep.symbol
            and c.side == sweep.side
            and c.features.quarter == sweep.features.quarter
            and c.features.utc_4h_block == sweep.features.utc_4h_block
        ]
        best: Optional[tuple[float, int, str]] = None
        for c in eligible:
            dist = _feature_distance(sweep.features, c.features)
            if dist is None:
                continue
            key = (dist, c.t0_ms, c.event_id)
            if best is None or key < best:
                best = key
        if best is not None:
            _, _, control_id = best
            used_control_ids.add(control_id)
            pairs.append(MatchedPair(sweep_event_id=sweep.event_id, control_event_id=control_id, match_distance=best[0]))

    return pairs
