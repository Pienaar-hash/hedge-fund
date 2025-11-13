import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Mapping

import requests

ASSET_DECIMALS = {"BTC": 8, "ETH": 8, "USDC": 6, "USDC": 6}


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


def atr_pct(symbol: str, lookback: int = 14, median_only: bool = False):
    """
    Placeholder ATR percent fetcher.
    Real implementation is expected to pull from cached indicators.
    """
    _ = (symbol, lookback, median_only)
    return None


_TREASURY_RESERVED_KEYS = {"assets", "total_usd", "treasury_usdt", "breakdown", "updated_at", "ts"}
_TREASURY_STABLES = {"USDC", "USDC", "DAI", "FDUSD", "TUSD", "USDE"}


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except Exception:
        try:
            return float(str(value))
        except Exception:
            return None


def _normalize_asset_entry(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    asset = (
        entry.get("asset")
        or entry.get("Asset")
        or entry.get("symbol")
        or entry.get("code")
        or entry.get("name")
    )
    asset_code = str(asset or "").upper()
    if not asset_code:
        return None
    balance = (
        _optional_float(entry.get("balance"))
        or _optional_float(entry.get("qty"))
        or _optional_float(entry.get("Units"))
        or _optional_float(entry.get("amount"))
        or 0.0
    )
    price = (
        _optional_float(entry.get("price_usdt"))
        or _optional_float(entry.get("price"))
        or _optional_float(entry.get("px"))
        or None
    )
    usd_value = (
        _optional_float(entry.get("usd_value"))
        or _optional_float(entry.get("USD Value"))
        or _optional_float(entry.get("value_usd"))
        or _optional_float(entry.get("val_usdt"))
        or None
    )
    return {
        "asset": asset_code,
        "balance": float(balance or 0.0),
        "price_usdt": float(price) if price is not None else None,
        "usd_value": float(usd_value) if usd_value is not None else None,
    }


def _read_existing_treasury(path: str) -> Dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        return {}

    assets_list: List[Dict[str, Any]] = []

    assets_raw = payload.get("assets") or payload.get("Assets")
    if isinstance(assets_raw, list):
        for entry in assets_raw:
            if isinstance(entry, dict):
                normalized = _normalize_asset_entry(entry)
                if normalized:
                    assets_list.append(normalized)

    for key, value in payload.items():
        if key in _TREASURY_RESERVED_KEYS:
            continue
        if not isinstance(value, dict):
            continue
        normalized = _normalize_asset_entry({"asset": key, **value})
        if normalized:
            assets_list.append(normalized)

    total_usd = _optional_float(payload.get("total_usd"))
    if total_usd is None and assets_list:
        total_candidates = [entry.get("usd_value") for entry in assets_list if isinstance(entry.get("usd_value"), (int, float))]
        if total_candidates:
            total_usd = float(sum(float(x) for x in total_candidates if x is not None))

    updated_at = payload.get("updated_at") or payload.get("ts")
    return {
        "assets": assets_list,
        "total_usd": float(total_usd) if total_usd is not None else None,
        "updated_at": updated_at,
    }


def _extract_holdings_from_breakdown(
    breakdown: Any,
) -> Tuple[Dict[str, Dict[str, Optional[float]]], List[str]]:
    holdings: Dict[str, Dict[str, Optional[float]]] = {}
    order: List[str] = []

    def ensure(asset_code: str) -> Dict[str, Optional[float]]:
        if asset_code not in holdings:
            holdings[asset_code] = {"balance": 0.0, "price": None, "usd_value": None}
            order.append(asset_code)
        return holdings[asset_code]

    def register(asset: str, payload: Any) -> None:
        asset_code = str(asset or "").upper()
        if not asset_code:
            return
        slot = ensure(asset_code)
        if isinstance(payload, dict):
            bal = (
                _optional_float(payload.get("balance"))
                or _optional_float(payload.get("qty"))
                or _optional_float(payload.get("Units"))
                or _optional_float(payload.get("amount"))
            )
            if bal is not None:
                slot["balance"] = bal
            price = (
                _optional_float(payload.get("price_usdt"))
                or _optional_float(payload.get("price"))
                or _optional_float(payload.get("px"))
            )
            if price is not None and price > 0:
                slot["price"] = price
            usd = (
                _optional_float(payload.get("usd_value"))
                or _optional_float(payload.get("USD Value"))
                or _optional_float(payload.get("value_usd"))
                or _optional_float(payload.get("val_usdt"))
            )
            if usd is not None and usd > 0:
                slot["usd_value"] = usd
        else:
            bal = _optional_float(payload)
            if bal is not None:
                slot["balance"] = bal

    if isinstance(breakdown, dict):
        tre_obj = breakdown.get("treasury")
        if isinstance(tre_obj, dict):
            for asset, info in tre_obj.items():
                register(asset, info)
        assets_list = breakdown.get("assets")
        if isinstance(assets_list, list):
            for entry in assets_list:
                if isinstance(entry, dict):
                    asset_name = entry.get("asset") or entry.get("Asset") or entry.get("symbol") or entry.get("code")
                    register(asset_name, entry)
        for key, value in breakdown.items():
            key_lower = str(key).lower()
            if key_lower in {"total_usd", "total_treasury_usdt", "updated_at", "ts"}:
                continue
            if key_lower in {"treasury", "assets"}:
                continue
            register(str(key), value)

    return holdings, order


def _extract_price_map(snapshot: Dict[str, Any]) -> Dict[str, float]:
    prices: Dict[str, float] = {}
    for entry in snapshot.get("assets", []) or []:
        if not isinstance(entry, dict):
            continue
        asset = str(entry.get("asset") or "").upper()
        price = entry.get("price_usdt")
        if asset and isinstance(price, (int, float)) and price > 0:
            prices[asset] = float(price)
    return prices


def _resolve_asset_price(
    asset: str,
    balance: float,
    price_hint: Optional[float],
    usd_hint: Optional[float],
    last_prices: Dict[str, float],
    get_price_fn: Optional[Any],
    get_last_known_price_fn: Optional[Any],
    logger: logging.Logger,
) -> Optional[float]:
    asset_code = asset.upper()
    if asset_code in _TREASURY_STABLES:
        return 1.0
    if price_hint is not None and price_hint > 0:
        return float(price_hint)
    if balance > 0 and usd_hint is not None and usd_hint > 0:
        derived = usd_hint / balance if balance else 0.0
        if derived > 0:
            return float(derived)
    if callable(get_price_fn):
        try:
            fetched = float(get_price_fn(f"{asset_code}USDC") or 0.0)
            if fetched > 0:
                return fetched
        except Exception as exc:
            logger.debug("[treasury] live_price_failed asset=%s error=%s", asset_code, exc)
    last_price = last_prices.get(asset_code)
    if last_price is not None and last_price > 0:
        return float(last_price)
    if callable(get_last_known_price_fn):
        try:
            fallback = get_last_known_price_fn(asset_code)
            if fallback is not None and fallback > 0:
                return float(fallback)
        except Exception as exc:
            logger.debug("[treasury] last_known_lookup_failed asset=%s error=%s", asset_code, exc)
    return None


def write_treasury_snapshot(
    val_usdt: float, breakdown: dict, path: str = "logs/treasury.json"
) -> None:
    logger = logging.getLogger("treasury_snapshot")
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        previous_snapshot = _read_existing_treasury(path)
        last_price_map = _extract_price_map(previous_snapshot)

        holdings, order = _extract_holdings_from_breakdown(breakdown)
        if not holdings:
            try:
                from execution.nav import compute_treasury_only

                _, fallback_breakdown = compute_treasury_only()
                holdings, order = _extract_holdings_from_breakdown(fallback_breakdown)
            except Exception as exc:
                logger.debug("[treasury] compute_treasury_only fallback failed: %s", exc)

        reserves_cfg = load_json("config/reserves.json")
        if isinstance(reserves_cfg, dict):
            for asset, qty in reserves_cfg.items():
                asset_code = str(asset).upper()
                qty_val = _optional_float(qty)
                if qty_val is None:
                    continue
                slot = holdings.setdefault(asset_code, {"balance": 0.0, "price": None, "usd_value": None})
                slot["balance"] = qty_val
                if asset_code not in order:
                    order.append(asset_code)

        treasury_cfg = load_json("config/treasury.json")
        if isinstance(treasury_cfg, dict):
            for asset, qty in treasury_cfg.items():
                asset_code = str(asset).upper()
                qty_val = _optional_float(qty)
                if qty_val is None:
                    continue
                slot = holdings.setdefault(asset_code, {"balance": 0.0, "price": None, "usd_value": None})
                slot["balance"] = qty_val if qty_val is not None else slot.get("balance", 0.0)
                if asset_code not in order:
                    order.append(asset_code)

        try:
            from execution.exchange_utils import get_price, get_last_known_price
        except Exception:
            get_price = None  # type: ignore
            get_last_known_price = None  # type: ignore

        seen: set[str] = set()
        assets_payload: List[Dict[str, Any]] = []
        for asset in order:
            asset_code = str(asset).upper()
            if not asset_code or asset_code in seen:
                continue
            seen.add(asset_code)
            info = holdings.get(asset_code) or {}
            balance = _optional_float(info.get("balance")) or 0.0
            price_hint = _optional_float(info.get("price"))
            usd_hint = _optional_float(info.get("usd_value"))
            price = _resolve_asset_price(
                asset_code,
                balance,
                price_hint,
                usd_hint,
                last_price_map,
                get_price,
                get_last_known_price,
                logger,
            )
            if price is None:
                if balance > 0:
                    logger.warning("[treasury] price_unavailable asset=%s balance=%.8f", asset_code, balance)
                price = 0.0
            usd_value = balance * price if price > 0 else 0.0
            assets_payload.append(
                {
                    "asset": asset_code,
                    "balance": float(balance),
                    "price_usdt": float(price),
                    "usd_value": float(usd_value),
                }
            )

        # Guarantee deterministic ordering for downstream readers
        total_usd = float(sum(entry["usd_value"] for entry in assets_payload))

        snapshot = {
            "assets": assets_payload,
            "total_usd": total_usd,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "treasury_usdt": total_usd,
        }
        save_json(path, snapshot)
    except Exception as exc:
        logger.error("[treasury] snapshot_write_failed: %s", exc)


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
    "USDC": "tether",
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
                _USD_ZAR_CACHE = {
                    "rate": float(rate),
                    "source": data.get("usd_zar_source") or "cache",
                }
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
            "usd_zar_source": _USD_ZAR_CACHE.get("source"),
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
        usd_zar_rate = get_usd_to_zar(force=force)  # ensure cache alignment
        if prices:
            _COINGECKO_CACHE = prices
            _COINGECKO_TS = now
            logging.info("[coingecko] updated: %s usd→zar=%.4f", prices, usd_zar_rate or -1.0)
            _persist_cache(_COINGECKO_CACHE, usd_zar_rate)
        return _COINGECKO_CACHE or prices
    except Exception as exc:
        logging.error("[coingecko] fetch error: %s", exc)
        if _COINGECKO_CACHE:
            logging.info("[coingecko] using cached prices")
            return _COINGECKO_CACHE
        return {}


def get_treasury_snapshot(path: str = "logs/treasury.json") -> Dict[str, Any]:
    """Return parsed treasury snapshot from disk; {} when unavailable."""
    try:
        payload = load_json(path)
    except Exception as exc:  # pragma: no cover - best effort
        logging.debug("[treasury] snapshot load failed: %s", exc)
        return {}
    return payload if isinstance(payload, dict) else {}


def get_usd_to_zar(
    force: bool = False,
    with_meta: bool = False,
) -> Union[Optional[float], Tuple[Optional[float], Dict[str, Optional[float]]]]:
    """Return USD→ZAR rate; optionally include freshness metadata."""
    global _USD_ZAR_CACHE, _USD_ZAR_TS
    now = time.time()
    cache_rate = _USD_ZAR_CACHE.get("rate")
    cache_source = _USD_ZAR_CACHE.get("source") or "cache"
    cache_age = (now - _USD_ZAR_TS) if _USD_ZAR_TS else None
    if not force and cache_rate is not None and (cache_age is None or cache_age < 120):
        meta = {"source": cache_source, "age": cache_age}
        return (cache_rate, meta) if with_meta else cache_rate
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
            _USD_ZAR_CACHE = {"rate": rate, "source": "api"}
            _USD_ZAR_TS = now
            logging.info("[coingecko] usd→zar updated: %.4f", rate)
            _persist_cache(_COINGECKO_CACHE, rate)
            meta = {"source": "api", "age": 0.0}
            return (rate, meta) if with_meta else rate
        # Fallback to cached values if API returns unexpected payload
        if cache_rate is not None:
            meta = {"source": cache_source, "age": cache_age}
            return (cache_rate, meta) if with_meta else cache_rate
        meta = {"source": "unavailable", "age": None}
        return (None, meta) if with_meta else None
    except Exception as exc:
        logging.error("[coingecko] usd→zar fetch error: %s", exc)
        if cache_rate is not None:
            logging.info("[coingecko] using cached usd→zar rate")
            meta = {"source": cache_source, "age": cache_age}
            return (cache_rate, meta) if with_meta else cache_rate
        meta = {"source": "error", "age": None}
        return (None, meta) if with_meta else None


def compute_treasury_pnl(snapshot: Mapping[str, Any]) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Compute per-asset PnL metrics from a treasury snapshot.

    Returns a dict keyed by symbol containing value_usd, avg_entry_price, pnl_pct.
    """
    if not isinstance(snapshot, Mapping):
        return {}

    treasury: Mapping[str, Any]
    if isinstance(snapshot.get("treasury"), Mapping):
        treasury = snapshot["treasury"]  # type: ignore[assignment]
    else:
        treasury = snapshot

    assets = treasury.get("assets")
    if not isinstance(assets, list):
        return {}

    results: Dict[str, Dict[str, Optional[float]]] = {}
    for entry in assets:
        if not isinstance(entry, Mapping):
            continue
        symbol = str(
            entry.get("asset")
            or entry.get("Asset")
            or entry.get("symbol")
            or entry.get("code")
            or ""
        ).upper()
        if not symbol:
            continue

        balance = (
            _optional_float(entry.get("balance"))
            or _optional_float(entry.get("qty"))
            or _optional_float(entry.get("Units"))
            or _optional_float(entry.get("units"))
            or _optional_float(entry.get("amount"))
        )
        value_usd = (
            _optional_float(entry.get("usd_value"))
            or _optional_float(entry.get("USD Value"))
            or _optional_float(entry.get("value_usd"))
            or _optional_float(entry.get("usd"))
        )
        price_usd = (
            _optional_float(entry.get("price_usdt"))
            or _optional_float(entry.get("price"))
            or _optional_float(entry.get("px"))
        )
        avg_entry_price = (
            _optional_float(entry.get("avg_entry_price"))
            or _optional_float(entry.get("avg_price"))
            or _optional_float(entry.get("avg_entry"))
        )
        cost_basis_usd = (
            _optional_float(entry.get("cost_basis_usd"))
            or _optional_float(entry.get("usd_cost"))
            or _optional_float(entry.get("cost_basis"))
        )

        if avg_entry_price is None and cost_basis_usd is not None and balance:
            if balance != 0:
                avg_entry_price = cost_basis_usd / balance
        if cost_basis_usd is None and avg_entry_price is not None and balance:
            cost_basis_usd = avg_entry_price * balance
        if price_usd is None and value_usd is not None and balance:
            if balance != 0:
                price_usd = value_usd / balance

        pnl_pct: Optional[float] = None
        if avg_entry_price and price_usd and avg_entry_price != 0:
            pnl_pct = ((price_usd - avg_entry_price) / avg_entry_price) * 100.0
        elif cost_basis_usd and value_usd and cost_basis_usd != 0:
            pnl_pct = ((value_usd - cost_basis_usd) / cost_basis_usd) * 100.0

        results[symbol] = {
            "value_usd": value_usd if value_usd is not None else None,
            "avg_entry_price": avg_entry_price if avg_entry_price is not None else None,
            "pnl_pct": float(pnl_pct) if pnl_pct is not None else None,
        }

    return results
