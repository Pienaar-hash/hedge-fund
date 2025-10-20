import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

ASSET_DECIMALS = {"BTC": 8, "ETH": 8, "USDT": 6, "USDC": 6}


def load_env_var(key, default=None):
    val = os.getenv(key)
    if val is None:
        print(f"⚠️ Environment variable {key} not set.")
    return val or default


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def log_trade(entry, path="logs/trade_log.json"):
    log = load_json(path)
    timestamp = datetime.now(timezone.utc).isoformat()
    log[timestamp] = entry
    save_json(path, log)


def load_local_state(path="synced_state.json"):
    """Loads local synced state from JSON."""
    return load_json(path)


def write_nav_snapshot(nav_usdt: float, breakdown: dict, path: str = "logs/nav_snapshot.json") -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "nav_usdt": float(nav_usdt),
            "breakdown": breakdown,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        save_json(path, payload)
    except Exception:
        pass


def write_nav_snapshots_pair(
    trading: tuple[float, dict], reporting: tuple[float, dict]
) -> None:
    try:
        nav_t, det_t = trading
        nav_r, det_r = reporting
        write_nav_snapshot(nav_t, det_t, "logs/nav_trading.json")
        write_nav_snapshot(nav_r, det_r, "logs/nav_reporting.json")
        # Legacy single snapshot (trading NAV)
        write_nav_snapshot(nav_t, det_t, "logs/nav_snapshot.json")
    except Exception:
        pass


def write_treasury_snapshot(
    val_usdt: float, breakdown: dict, path: str = "logs/nav_treasury.json"
) -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "treasury_usdt": float(val_usdt),
            "breakdown": breakdown,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        save_json(path, payload)
    except Exception:
        pass


def get_live_positions(client) -> list:
    """Return a list of open positions from the Binance futures account."""
    try:
        positions = client.get_position_risk()
    except Exception as exc:
        print(f"[get_live_positions] error: {exc}")
        return []

    live: list = []
    for raw in positions or []:
        try:
            amt = float(raw.get("positionAmt", 0.0) or 0.0)
        except Exception:
            amt = 0.0
        if abs(amt) <= 0.0:
            continue
        try:
            entry_price = float(raw.get("entryPrice", 0.0) or 0.0)
        except Exception:
            entry_price = 0.0
        try:
            upnl = float(raw.get("unRealizedProfit", 0.0) or 0.0)
        except Exception:
            upnl = 0.0
        live.append(
            {
                "symbol": raw.get("symbol"),
                "positionSide": raw.get("positionSide", "BOTH"),
                "positionAmt": amt,
                "entryPrice": entry_price,
                "unRealizedProfit": upnl,
            }
        )
    return live


COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"
COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "USDT": "tether",
    "XAUT": "tether-gold",
}
_COINGECKO_CACHE: dict = {}
_COINGECKO_TS: float = 0.0
_USD_ZAR_CACHE: dict = {}
_USD_ZAR_TS: float = 0.0
CACHE_PATH = Path("logs/cache/coingecko_cache.json")


def _load_cache_from_disk() -> None:
    global _COINGECKO_CACHE, _COINGECKO_TS, _USD_ZAR_CACHE, _USD_ZAR_TS
    try:
        if CACHE_PATH.exists():
            data = json.loads(CACHE_PATH.read_text())
            _COINGECKO_CACHE = data.get("prices") or {}
            _COINGECKO_TS = float(data.get("prices_ts") or 0.0)
            rate = data.get("usd_zar")
            if rate:
                _USD_ZAR_CACHE = {"rate": float(rate)}
                _USD_ZAR_TS = float(data.get("usd_zar_ts") or 0.0)
    except Exception as exc:  # pragma: no cover - cache load best effort
        logging.debug("[coingecko] cache load failed: %s", exc)


def _persist_cache(prices: dict, usd_zar: Optional[float]) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "prices": prices,
            "prices_ts": _COINGECKO_TS,
            "usd_zar": usd_zar if usd_zar is not None else _USD_ZAR_CACHE.get("rate"),
            "usd_zar_ts": _USD_ZAR_TS,
        }
        CACHE_PATH.write_text(json.dumps(payload))
    except Exception as exc:  # pragma: no cover
        logging.debug("[coingecko] cache persist failed: %s", exc)


_load_cache_from_disk()


def get_coingecko_prices(vs: str = "usd", force: bool = False) -> dict:
    """Return cached CoinGecko prices for configured assets."""
    global _COINGECKO_CACHE, _COINGECKO_TS
    now = time.time()
    if not force and _COINGECKO_CACHE and (now - _COINGECKO_TS) < 120:
        return _COINGECKO_CACHE

    try:
        ids = ",".join(COINGECKO_IDS.values())
        resp = requests.get(
            COINGECKO_URL,
            params={"ids": ids, "vs_currencies": vs},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json() or {}
        prices = {}
        for ticker, coingecko_id in COINGECKO_IDS.items():
            try:
                prices[ticker] = float(data[coingecko_id][vs])
            except Exception:
                continue
        usd_zar = get_usd_to_zar(force=force)  # ensure cache alignment
        if prices:
            _COINGECKO_CACHE = prices
            _COINGECKO_TS = now
            logging.info("[coingecko] updated: %s usd→zar=%.4f", prices, usd_zar or -1.0)
            _persist_cache(_COINGECKO_CACHE, usd_zar)
        return _COINGECKO_CACHE or prices
    except Exception as exc:
        logging.error("[coingecko] fetch error: %s", exc)
        if _COINGECKO_CACHE:
            logging.info("[coingecko] using cached prices")
            return _COINGECKO_CACHE
        return {}


def get_usd_to_zar(force: bool = False) -> Optional[float]:
    """Return USD→ZAR rate using CoinGecko."""
    global _USD_ZAR_CACHE, _USD_ZAR_TS
    now = time.time()
    if not force and _USD_ZAR_CACHE and (now - _USD_ZAR_TS) < 120:
        return _USD_ZAR_CACHE.get("rate")
    try:
        resp = requests.get(
            COINGECKO_URL,
            params={"ids": "usd", "vs_currencies": "zar"},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json() or {}
        rate = float(((data or {}).get("usd") or {}).get("zar") or 0.0)
        if rate > 0:
            _USD_ZAR_CACHE = {"rate": rate}
            _USD_ZAR_TS = now
            logging.info("[coingecko] usd→zar updated: %.4f", rate)
            _persist_cache(_COINGECKO_CACHE, rate)
        return _USD_ZAR_CACHE.get("rate")
    except Exception as exc:
        logging.error("[coingecko] usd→zar fetch error: %s", exc)
        cached = _USD_ZAR_CACHE.get("rate")
        if cached:
            logging.info("[coingecko] using cached usd→zar rate")
        return cached
