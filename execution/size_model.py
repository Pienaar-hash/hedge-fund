from __future__ import annotations

import json
import math
import os
from statistics import pstdev
from typing import Any, Dict, Iterable, List, Optional

from execution.exchange_utils import get_klines, get_price

KLINE_CACHE_DIR = os.path.join("logs", "cache", "klines")


def _ensure_cache_dir() -> None:
    try:
        os.makedirs(KLINE_CACHE_DIR, exist_ok=True)
    except Exception:
        pass


def _load_cached_klines(symbol: str, interval: str) -> List[List[float]]:
    path = os.path.join(KLINE_CACHE_DIR, f"{symbol.upper()}_{interval}.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            # Expect list of [openTime, open, high, low, close, volume, ...]
            return [
                row for row in payload if isinstance(row, (list, tuple)) and len(row) >= 5
            ]
    except Exception:
        return []
    return []


def _cache_klines(symbol: str, interval: str, rows: List[List[float]]) -> None:
    if not rows:
        return
    _ensure_cache_dir()
    path = os.path.join(KLINE_CACHE_DIR, f"{symbol.upper()}_{interval}.json")
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(rows, handle)
    except Exception:
        pass


def _true_range(high: float, low: float, prev_close: float) -> float:
    return max(
        high - low,
        abs(high - prev_close),
        abs(prev_close - low),
    )


def atr(
    symbol: str,
    *,
    lookback: int = 14,
    interval: str = "1h",
    min_rows: Optional[int] = None,
) -> float:
    """Compute a simple ATR (in price units) from cached klines, falling back to std-dev returns."""
    min_rows = min_rows or (lookback + 2)
    rows = _load_cached_klines(symbol, interval)
    if len(rows) < min_rows:
        try:
            rows = get_klines(symbol, interval, limit=max(lookback * 3, 100))
            if rows:
                _cache_klines(symbol, interval, rows)
        except Exception:
            rows = rows or []
    if len(rows) < min_rows:
        closes = _closing_prices(rows)
        return _fallback_volatility(closes, lookback)
    true_ranges: List[float] = []
    prev_close: Optional[float] = None
    for row in rows[-(lookback + 1) :]:
        try:
            high = float(row[2])
            low = float(row[3])
            close = float(row[4])
        except Exception:
            continue
        if prev_close is None:
            prev_close = close
            continue
        tr = _true_range(high, low, prev_close)
        true_ranges.append(tr)
        prev_close = close
    if not true_ranges:
        closes = _closing_prices(rows)
        return _fallback_volatility(closes, lookback)
    return sum(true_ranges[-lookback:]) / float(len(true_ranges[-lookback:]))


def _closing_prices(rows: Iterable[Iterable[Any]]) -> List[float]:
    closes: List[float] = []
    for row in rows:
        try:
            closes.append(float(row[4]))
        except Exception:
            continue
    return closes


def _fallback_volatility(closes: List[float], lookback: int) -> float:
    if len(closes) < lookback + 1:
        return 0.0
    returns: List[float] = []
    for prev, curr in zip(closes[-(lookback + 1) : -1], closes[-lookback:]):
        try:
            if prev <= 0:
                continue
            returns.append((float(curr) - float(prev)) / float(prev))
        except Exception:
            continue
    if not returns:
        return 0.0
    sigma = pstdev(returns) if len(returns) > 1 else abs(returns[0])
    last_close = closes[-1]
    if not math.isfinite(sigma) or not math.isfinite(last_close):
        return 0.0
    return abs(last_close) * abs(sigma)


def _symbol_bucket_map(risk_cfg: Dict[str, Any]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    raw = risk_cfg.get("symbol_bucket")
    if isinstance(raw, dict):
        for key, value in raw.items():
            if not value:
                continue
            mapping[str(key).upper()] = str(value)
    return mapping


def _bucket_caps(risk_cfg: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    raw = risk_cfg.get("bucket_caps_pct") or {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            try:
                out[str(key)] = float(value)
            except Exception:
                continue
    return out


def _current_symbol_gross(risk_cfg: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    raw = risk_cfg.get("current_symbol_gross") or {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            try:
                out[str(key).upper()] = float(value)
            except Exception:
                continue
    return out


def _current_bucket_gross(risk_cfg: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    raw = risk_cfg.get("current_bucket_gross") or {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            try:
                out[str(key)] = float(value)
            except Exception:
                continue
    return out


def _per_symbol_leverage(risk_cfg: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    raw = risk_cfg.get("per_symbol_leverage") or {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            try:
                out[str(key).upper()] = float(value)
            except Exception:
                continue
    return out


def suggest_gross_usd(
    symbol: str,
    nav_usd: float,
    signal_strength: float,
    risk_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Suggest a gross notional sized to a volatility budget and correlation caps."""
    out: Dict[str, Any] = {}
    symbol_upper = str(symbol).upper()
    nav = float(nav_usd or 0.0)
    if nav <= 0.0:
        fallback = float(risk_cfg.get("fallback_gross_usd") or 0.0)
        return {
            "gross_usd": max(fallback, 0.0),
            "reason": "nav_unavailable",
        }
    signal_factor = float(signal_strength or 0.0)
    if not math.isfinite(signal_factor) or signal_factor <= 0.0:
        signal_factor = 1.0
    interval = str(risk_cfg.get("atr_interval") or "1h")
    lookback = int(risk_cfg.get("atr_lookback") or 14)
    atr_value = atr(symbol_upper, lookback=lookback, interval=interval)
    price_hint = risk_cfg.get("price") or risk_cfg.get("price_hint")
    try:
        price = float(price_hint or 0.0)
    except Exception:
        price = 0.0
    if price <= 0.0:
        try:
            price = float(get_price(symbol_upper) or 0.0)
        except Exception:
            price = 0.0
    min_notional = float(risk_cfg.get("min_notional_usd") or 0.0)
    fallback_gross = float(risk_cfg.get("fallback_gross_usd") or min_notional or 0.0)
    vol_target_bps = float(risk_cfg.get("vol_target_bps") or 25.0)
    vol_target = abs(vol_target_bps) / 10000.0
    gross_from_vol = 0.0
    atr_pct = 0.0
    if price > 0.0 and atr_value > 0.0:
        atr_pct = atr_value / price
        if atr_pct > 0.0:
            gross_from_vol = (nav * vol_target) / max(atr_pct, 1e-8)
    if gross_from_vol <= 0.0:
        gross_from_vol = max(fallback_gross, min_notional)
    gross_base = gross_from_vol * max(signal_factor, 0.25)
    gross = max(gross_base, min_notional)
    bucket_mapping = _symbol_bucket_map(risk_cfg)
    bucket_name = bucket_mapping.get(symbol_upper)
    bucket_caps = _bucket_caps(risk_cfg)
    bucket_used = 0.0
    bucket_cap_abs = None
    current_bucket = _current_bucket_gross(risk_cfg)
    if bucket_name and bucket_name in bucket_caps:
        bucket_cap_pct = float(bucket_caps[bucket_name])
        bucket_cap_abs = nav * (bucket_cap_pct / 100.0)
        bucket_used = current_bucket.get(bucket_name, 0.0)
        remaining_bucket = max(0.0, bucket_cap_abs - bucket_used)
        gross = min(gross, remaining_bucket)
    symbol_caps_pct = float(risk_cfg.get("max_symbol_exposure_pct") or 0.0)
    current_symbol_gross = _current_symbol_gross(risk_cfg)
    symbol_used = current_symbol_gross.get(symbol_upper, 0.0)
    if symbol_caps_pct > 0.0:
        symbol_cap_abs = nav * (symbol_caps_pct / 100.0)
        remaining_symbol = max(0.0, symbol_cap_abs - symbol_used)
        gross = min(gross, remaining_symbol)
    max_trade_pct = float(risk_cfg.get("max_trade_nav_pct") or 0.0)
    if max_trade_pct > 0.0:
        trade_cap_abs = nav * (max_trade_pct / 100.0)
        gross = min(gross, trade_cap_abs)
    portfolio_cap_pct = float(risk_cfg.get("max_gross_exposure_pct") or 0.0)
    if portfolio_cap_pct > 0.0:
        portfolio_cap_abs = nav * (portfolio_cap_pct / 100.0)
        current_portfolio_gross = float(risk_cfg.get("current_portfolio_gross") or 0.0)
        remaining_portfolio = max(0.0, portfolio_cap_abs - current_portfolio_gross)
        gross = min(gross, remaining_portfolio)
    leverage_caps = _per_symbol_leverage(risk_cfg)
    default_leverage = float(risk_cfg.get("default_leverage") or 1.0)
    leverage_cap = leverage_caps.get(symbol_upper, default_leverage)
    leverage_cap = max(leverage_cap, 1.0)
    gross = min(gross, nav * leverage_cap)
    gross = max(gross, 0.0)
    result: Dict[str, Any] = {
        "gross_usd": float(gross),
        "atr": float(atr_value),
        "atr_pct": float(atr_pct),
        "bucket": bucket_name,
        "bucket_used": float(bucket_used),
        "bucket_cap": float(bucket_cap_abs) if bucket_cap_abs is not None else None,
        "symbol_used": float(symbol_used),
        "signal_strength": float(signal_factor),
        "vol_target_bps": float(vol_target_bps),
        "leverage_cap": float(leverage_cap),
    }
    if gross <= 0.0:
        reason = "bucket_cap" if bucket_name else "sizer_cap"
        result["reason"] = reason
    return result


__all__ = ["atr", "suggest_gross_usd"]
