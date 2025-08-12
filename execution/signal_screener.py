# execution/signal_screener.py — Sprint Phase 2
# Parametrized thresholds + strategy_name plumbing, multi-asset helpers + per-asset overrides

import os
import json
from typing import Dict, Any, List, Optional

import requests
import pandas as pd

BINANCE_BASE = "https://api.binance.com"

# -------------------------- Data fetch --------------------------

def fetch_candles(symbol: str = "BTCUSDT", interval: str = "4h", limit: int = 200) -> pd.DataFrame:
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=15)
    data = r.json()

    df = pd.DataFrame(
        data,
        columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

# -------------------------- Indicators --------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss.replace(0, pd.NA))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method="bfill")


def calculate_indicators(
    df: pd.DataFrame,
    rsi_period: int = 14,
    z_lookback: int = 20,
    mom_lookback: int = 4,
) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = compute_rsi(out["close"], period=rsi_period)
    mean = out["close"].rolling(z_lookback).mean()
    std = out["close"].rolling(z_lookback).std()
    out["z_score"] = (out["close"] - mean) / std
    out["momentum"] = out["close"] - out["close"].shift(mom_lookback)
    return out

# -------------------------- Screening --------------------------

def _default_thresholds(name: str) -> Dict[str, Any]:
    # Sensible defaults per family
    if name == "momentum":
        return {"zscore_threshold": 1.5, "rsi_buy": 60, "rsi_sell": 40, "min_volume": 0.0}
    if name == "volatility_target":
        return {"zscore_threshold": 1.2, "rsi_buy": 55, "rsi_sell": 45, "min_volume": 0.0}
    if name == "relative_value":
        return {"zscore_threshold": 1.2, "rsi_buy": 55, "rsi_sell": 45, "min_volume": 0.0}
    return {"zscore_threshold": 1.5, "rsi_buy": 60, "rsi_sell": 40, "min_volume": 0.0}


def build_thresholds_from_strategy(name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    thr = _default_thresholds(name)
    # pull known keys if present; otherwise keep defaults
    for k in ["zscore_threshold", "rsi_buy", "rsi_sell", "min_volume"]:
        if k in params:
            thr[k] = params[k]
    # allow custom RSI / Z / MOM windows
    thr["rsi_period"] = int(params.get("rsi_period", 14))
    thr["z_lookback"] = int(params.get("z_lookback", params.get("lookback", 20)))
    thr["mom_lookback"] = int(params.get("mom_lookback", 4))
    # timeframe override
    thr["timeframe"] = str(params.get("timeframe", "4h"))
    thr["limit"] = int(params.get("limit", 200))
    return thr


def evaluate_latest_row(latest: pd.Series, thr: Dict[str, Any]) -> Optional[str]:
    z = float(latest.get("z_score", 0))
    rsi = float(latest.get("rsi", 50))
    mom = float(latest.get("momentum", 0))
    vol = float(latest.get("volume", 0))

    if vol < float(thr.get("min_volume", 0.0)):
        return None

    zthr = float(thr.get("zscore_threshold", 1.5))
    rsi_up = float(thr.get("rsi_buy", 60))
    rsi_dn = float(thr.get("rsi_sell", 40))

    if z > zthr and rsi > rsi_up and mom > 0:
        return "BUY"
    if z < -zthr and rsi < rsi_dn and mom < 0:
        return "SELL"
    return None


def screen_symbol(
    symbol: str,
    strategy_name: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Screen one symbol using strategy params. Returns signal payload including strategy_name.
    params can include timeframe, zscore_threshold, rsi_buy/rsi_sell, *_periods, min_volume, limit.
    """
    thr = build_thresholds_from_strategy(strategy_name, params)

    df = fetch_candles(symbol=symbol, interval=thr["timeframe"], limit=thr["limit"])
    df = calculate_indicators(
        df,
        rsi_period=thr["rsi_period"],
        z_lookback=thr["z_lookback"],
        mom_lookback=thr["mom_lookback"],
    )

    latest = df.iloc[-1]
    side = evaluate_latest_row(latest, thr)

    return {
        "strategy": strategy_name,
        "symbol": symbol,
        "signal": side,
        "price": round(float(latest["close"]), 2),
        "z_score": round(float(latest["z_score"]), 2),
        "rsi": round(float(latest["rsi"]), 2),
        "momentum": round(float(latest["momentum"]), 2),
        "timeframe": thr["timeframe"],
    }

# -------------------------- Batch helpers --------------------------

def symbols_from_strategy(strategy_cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    """Return a list of {symbol, tf} dicts from strategy config."""
    name = strategy_cfg.get("name", "")
    p = strategy_cfg.get("params", {})
    if name == "momentum":
        return [{"symbol": s, "tf": p.get("timeframe", "1h")} for s in p.get("symbols", ["BTCUSDT"])]
    if name == "volatility_target":
        assets = p.get("assets", [{"symbol": "BTCUSDT", "tf": p.get("timeframe", "4h")}])
        return [{"symbol": a.get("symbol", "BTCUSDT"), "tf": a.get("tf", "4h")} for a in assets]
    if name == "relative_value":
        # base/pairs handled elsewhere; for screening we still return pair legs for diagnostics
        base = p.get("base", "ETHUSDT")
        pairs = p.get("pairs", [])
        return ([{"symbol": base, "tf": p.get("timeframe", "1d")}] +
                [{"symbol": s, "tf": p.get("timeframe", "1d")} for s in pairs])
    return [{"symbol": "BTCUSDT", "tf": p.get("timeframe", "4h")}]


def _merge_params_for_symbol(base_params: Dict[str, Any], item_tf: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Merge strategy-level params with per-asset overrides.
    Precedence: overrides.timeframe/tf > item_tf > base_params.timeframe > default '4h'.
    """
    p_local = dict(base_params)
    tf = (
        overrides.get("timeframe")
        or overrides.get("tf")
        or item_tf
        or base_params.get("timeframe")
        or "4h"
    )
    p_local["timeframe"] = tf
    # thresholds/windows and any other keys can be overridden directly
    for k, v in overrides.items():
        p_local[k] = v
    return p_local


def screen_strategy(strategy_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Screen all symbols declared under a strategy config and return signal dicts.
    Supports optional per-asset overrides via:
    {
      "name": "momentum",
      "params": {
         "symbols": ["BTCUSDT", "ETHUSDT"],
         "timeframe": "1h",
         "per_asset_overrides": {
            "BTCUSDT": {"timeframe": "30m", "zscore_threshold": 0.7},
            "ETHUSDT": {"rsi_buy": 58}
         }
      }
    }
    """
    name = strategy_cfg.get("name", "?")
    params = strategy_cfg.get("params", {})
    overrides_map: Dict[str, Any] = params.get("per_asset_overrides", {}) or {}

    results: List[Dict[str, Any]] = []
    for item in symbols_from_strategy(strategy_cfg):
        sym = item["symbol"]
        ovr = overrides_map.get(sym, {}) if isinstance(overrides_map, dict) else {}
        p_local = _merge_params_for_symbol(params, item.get("tf", params.get("timeframe", "4h")), ovr)
        results.append(screen_symbol(sym, name, p_local))
    return results

# -------------------------- Shared JSON utils --------------------------

def load_json(path: str):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, data: Dict[str, Any]):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
