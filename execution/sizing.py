from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Dict, Sequence

_MINUTES_PER_YEAR = 365 * 24 * 60


@dataclass
class SizingResult:
    ok: bool
    gross: float
    reasons: list[str]
    meta: Dict[str, float | str]


def _timeframe_to_minutes(tf: str) -> float:
    tf = (tf or "").strip().lower()
    if not tf:
        return 0.0
    try:
        if tf.endswith("m"):
            return float(tf[:-1])
        if tf.endswith("h"):
            return float(tf[:-1]) * 60.0
        if tf.endswith("d"):
            return float(tf[:-1]) * 60.0 * 24.0
        if tf.endswith("w"):
            return float(tf[:-1]) * 60.0 * 24.0 * 7.0
    except Exception:
        return 0.0
    # default fall-back for unsupported intervals (e.g., "1H")
    mapping = {
        "1h": 60.0,
        "4h": 240.0,
        "1d": 1440.0,
        "1w": 10080.0,
    }
    return float(mapping.get(tf, 0.0))


def estimate_annualized_volatility(
    closes: Sequence[float],
    timeframe: str,
    window: int,
) -> float:
    if len(closes) < max(window, 2):
        return 0.0
    tf_minutes = _timeframe_to_minutes(timeframe)
    if tf_minutes <= 0:
        return 0.0
    returns = []
    for i in range(1, len(closes)):
        try:
            prev = float(closes[i - 1])
            curr = float(closes[i])
        except (TypeError, ValueError):
            continue
        if prev <= 0 or curr <= 0:
            continue
        returns.append(math.log(curr / prev))
    if len(returns) < 2:
        return 0.0
    sample = returns[-window:] if window > 0 else returns
    if len(sample) < 2:
        return 0.0
    std = statistics.pstdev(sample)
    annualization = math.sqrt(_MINUTES_PER_YEAR / tf_minutes)
    return float(std * annualization)


def determine_position_size(
    *,
    symbol: str,
    strategy_cfg: Dict[str, Any],
    sizing_cfg: Dict[str, Any],
    risk_global_cfg: Dict[str, Any],
    nav: float,
    timeframe: str,
    closes: Sequence[float],
    price: float,
    exchange_min_notional: float,
    min_qty_notional: float,
    size_floor_usd: float,
) -> SizingResult:
    reasons: list[str] = []
    meta: Dict[str, float | str] = {
        "symbol": symbol,
        "nav": float(nav),
        "price": float(price),
    }

    nav = float(nav or 0.0)
    if nav <= 0.0:
        reasons.append("nav_unavailable")
        return SizingResult(False, 0.0, reasons, meta)

    lev = float(strategy_cfg.get("leverage", sizing_cfg.get("default_leverage", 1.0)) or 1.0)
    cap_base = strategy_cfg.get("capital_per_trade")
    if cap_base is None:
        cap_base = sizing_cfg.get("capital_per_trade_usdt", 0.0)
    cap_base = float(cap_base or 0.0)
    base_gross = max(0.0, cap_base * max(lev, 1.0))
    meta["base_gross"] = base_gross

    nav_cap_pct = None
    for key_space in (risk_global_cfg, sizing_cfg):
        if nav_cap_pct:
            break
        try:
            nav_cap_pct = float((key_space or {}).get("max_trade_nav_pct", 0.0) or 0.0)
        except Exception:
            nav_cap_pct = None
    if nav_cap_pct is None:
        nav_cap_pct = 0.0
    nav_cap_usd = nav * (nav_cap_pct / 100.0) if nav_cap_pct > 0 else float("inf")
    meta["nav_cap_pct"] = nav_cap_pct
    meta["nav_cap_usd"] = nav_cap_usd if math.isfinite(nav_cap_usd) else float("inf")

    vol_window = int(float(sizing_cfg.get("vol_window_bars", 96) or 96))
    target_vol = float(sizing_cfg.get("vol_target_annual_pct", 0.0) or 0.0) / 100.0
    vol_floor = float(sizing_cfg.get("vol_floor_annual_pct", 5.0) or 5.0) / 100.0
    kelly_fraction = float(sizing_cfg.get("kelly_fraction", 1.0) or 1.0)
    est_vol = 0.0
    vol_cap_usd = float("inf")
    if target_vol > 0.0:
        est_vol = estimate_annualized_volatility(closes, timeframe, vol_window)
        meta["vol_estimate"] = est_vol
        if est_vol > 0.0:
            ratio = target_vol / max(est_vol, vol_floor)
            ratio = min(max(ratio, 0.0), 1.0)
            vol_cap_usd = nav * ratio * max(kelly_fraction, 0.0)
            meta["vol_ratio"] = ratio
            meta["vol_cap_usd"] = vol_cap_usd
        else:
            reasons.append("vol_insufficient_history")
    else:
        meta["vol_estimate"] = 0.0

    candidates = [c for c in (base_gross, nav_cap_usd, vol_cap_usd) if c and math.isfinite(c) and c > 0]
    if not candidates:
        reasons.append("no_sizing_candidate")
        return SizingResult(False, 0.0, reasons, meta)

    recommended = min(candidates)
    floors = max(size_floor_usd, exchange_min_notional, min_qty_notional)
    meta["size_floor_usd"] = floors
    if recommended < floors:
        reasons.append("below_size_floor")
        meta["recommended_gross"] = recommended
        return SizingResult(False, floors, reasons, meta)

    meta["recommended_gross"] = recommended
    return SizingResult(True, recommended, reasons, meta)


__all__ = [
    "SizingResult",
    "determine_position_size",
    "estimate_annualized_volatility",
]
