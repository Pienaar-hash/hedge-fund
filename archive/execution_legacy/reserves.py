import json
import logging
import os
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)

RESERVES_PATH = os.environ.get("RESERVES_PATH", "config/reserves.json")
_STABLE_PAR = {"USDC", "USDCUSDT", "USDT", "DAI", "TUSD", "FDUSD", "USDE"}


def load_reserves(path: str = RESERVES_PATH) -> Dict[str, Any]:
    """Load off-exchange reserves (e.g., XAUT, USDC, BTC on Luno) as a dict {symbol: amount}."""
    if not os.path.exists(path):
        logger.warning(f"[reserves] file missing: {path} → treating reserves=0")
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("reserves file must be a JSON object of {symbol: amount}")
        return data
    except Exception as e:
        logger.error(f"[reserves] load_failed: {e}")
        return {}


def value_reserves_usd(reserves: Dict[str, float]) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """Value reserves in USD and return (total_usd, detail_by_symbol).

    - Common stablecoins ≈ 1.0
    - XAUT via CoinGecko tether-gold
    - Other coins via exchange_utils.get_price('XXXUSDT', venue='spot')
    """
    from execution.exchange_utils import get_price
    from execution.utils import get_coingecko_prices

    total = 0.0
    detail: Dict[str, Dict[str, float]] = {}
    coingecko_cache: Dict[str, float] | None = None

    for sym, amt in reserves.items():
        sym_u = sym.upper()
        try:
            qty = float(amt)
        except Exception:
            logger.warning(f"[reserves] non_numeric_amt symbol={sym_u} amt={amt}")
            continue

        if abs(qty) < 1e-12:
            continue

        try:
            if sym_u in _STABLE_PAR:
                px = 1.0
            elif sym_u in ("XAUT", "XAUTUSDT"):
                if coingecko_cache is None:
                    coingecko_cache = get_coingecko_prices() or {}
                px = float(coingecko_cache.get("XAUT") or 0.0)
                if px <= 0:
                    raise ValueError("xaut_price_missing")
            elif sym_u.endswith("USDT"):
                px = float(get_price(sym_u, venue="spot", signed=False))
            else:
                # Try symbolUSDT on spot
                px = float(get_price(f"{sym_u}USDT", venue="spot", signed=False))
            value = qty * px
            total += value
            detail[sym_u] = {"amount": qty, "price_usd": px, "value_usd": value}
        except Exception as e:
            logger.warning(f"[reserves] price_fail symbol={sym_u} amt={amt}: {e}")
    logger.info(f"[reserves] total_usd={total:.2f}")
    return total, detail
