from __future__ import annotations

import json
import math
import os
import time
import logging
from typing import Any, Dict, List, Tuple

from execution.exchange_utils import get_balances, get_positions, get_price
from execution.reserves import load_reserves, value_reserves_usd

_NAV_CACHE_PATH = "logs/cache/nav_confirmed.json"
_NAV_LOG_PATH = "logs/nav_log.json"

LOGGER = logging.getLogger("nav")


def _load_json(path: str) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _futures_nav_usdt() -> Tuple[float, Dict]:
    """Compute USD-M futures wallet NAV and provide detail."""
    balances_ok = False
    positions_ok = False
    try:
        bal = get_balances() or {}
        balances_ok = True
    except Exception as exc:
        LOGGER.warning("[nav] balances_fetch_failed: %s", exc)
        bal = {}
    # Support both dict-of-balances and list-of-dicts returns
    wallet = 0.0
    if isinstance(bal, dict):
        wallet = float(bal.get("USDT", bal.get("walletBalance", 0.0)) or 0.0)
    elif isinstance(bal, list):
        for entry in bal:
            try:
                if entry.get("asset") == "USDT":
                    wallet = float(entry.get("balance") or entry.get("walletBalance") or 0.0)
                    break
            except Exception:
                continue
    detail = {"futures_wallet_usdt": wallet}
    # Include unrealized PnL if present via positions
    try:
        positions = get_positions() or []
        positions_ok = True
        unreal = 0.0
        for pos in positions:
            try:
                unreal += float(pos.get("unrealized", 0.0))
            except Exception:
                continue
        detail["unrealized_pnl"] = unreal
        wallet += unreal
    except Exception as exc:
        LOGGER.warning("[nav] positions_fetch_failed: %s", exc)
    detail["balances_ok"] = bool(balances_ok)
    detail["positions_ok"] = bool(positions_ok)
    source_health = {
        "balances_ok": balances_ok,
        "positions_ok": positions_ok,
    }
    _persist_confirmed_nav(wallet, detail=detail, source_health=source_health)
    return wallet, detail


_STABLES = {"USDT", "USDC", "BUSD"}


def _treasury_nav_usdt(treasury_path: str = "config/treasury.json") -> Tuple[float, Dict]:
    treasury = _load_json(treasury_path)
    if not treasury:
        return 0.0, {"treasury": {}}

    total = 0.0
    holdings: Dict[str, Dict[str, float]] = {}
    missing: Dict[str, str] = {}

    for asset, qty in treasury.items():
        if qty in (None, "", []):
            continue
        try:
            qty_float = float(qty)
        except Exception:
            continue
        if abs(qty_float) < 1e-12:
            continue

        asset_code = str(asset).upper()
        try:
            if asset_code in _STABLES:
                price = 1.0
            else:
                price = float(get_price(f"{asset_code}USDT"))
        except Exception as exc:
            missing[asset_code] = str(exc)
            continue

        value = qty_float * price
        total += value
        holdings[asset_code] = {
            "qty": qty_float,
            "px": price,
            "val_usdt": value,
        }

    breakdown: Dict[str, Any] = {"treasury": holdings, "total_treasury_usdt": total}
    if missing:
        breakdown["missing_prices"] = missing
    return total, breakdown


def _reserves_nav_usd() -> Tuple[float, Dict]:
    """Load off-exchange reserves and value them in USD."""
    try:
        reserves = load_reserves()
    except Exception as exc:
        LOGGER.warning("[nav] reserves_load_failed: %s", exc)
        return 0.0, {"reserves": {}, "error": str(exc)}

    if not reserves:
        return 0.0, {"reserves": {}}

    try:
        total, detail = value_reserves_usd(reserves)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("[nav] reserves_value_failed: %s", exc)
        return 0.0, {"reserves": {}, "error": str(exc), "raw": reserves}

    reserves_detail: Dict[str, Any] = {
        "reserves": detail,
        "total_reserves_usd": float(total),
        "raw": reserves,
    }
    return float(total), reserves_detail


def _nav_sources(cfg: Dict) -> Tuple[str, str, bool, Any]:
    nav_cfg = cfg.get("nav") or {}
    trading_source = nav_cfg.get("trading_source") or nav_cfg.get("source") or "exchange"
    reporting_source = nav_cfg.get("reporting_source") or trading_source
    include_treasury = bool(nav_cfg.get("include_spot_treasury", False))
    manual = nav_cfg.get("manual_nav_usdt")
    return str(trading_source), str(reporting_source), include_treasury, manual


def _fallback_capital(cfg: Dict) -> Tuple[float, Dict]:
    fallback = float(cfg.get("capital_base_usdt", 0.0) or 0.0)
    return fallback, {"source": "capital_base"}


def compute_trading_nav(cfg: Dict) -> Tuple[float, Dict]:
    trading_source, _, _, manual = _nav_sources(cfg)
    if trading_source == "manual":
        if manual is not None:
            return float(manual), {"source": "manual"}
        return _fallback_capital(cfg)

    fut_nav, fut_detail = _futures_nav_usdt()
    if fut_nav > 0:
        return float(fut_nav), {"source": "exchange", **fut_detail}
    return _fallback_capital(cfg)


def compute_nav_summary(cfg: Dict | None = None) -> Dict[str, Any]:
    cfg = cfg or _load_json("config/strategy_config.json")

    futures_nav, futures_detail = _futures_nav_usdt()
    treasury_nav, treasury_detail = _treasury_nav_usdt()
    reserves_nav, reserves_detail = _reserves_nav_usd()

    summary = {
        "futures_nav": float(futures_nav),
        "treasury_nav": float(treasury_nav),
        "reserves_nav": float(reserves_nav),
        "total_nav": float(futures_nav + treasury_nav + reserves_nav),
        "details": {
            "futures": futures_detail,
            "treasury": treasury_detail,
            "reserves": reserves_detail,
        },
    }
    return summary


def compute_reporting_nav(cfg: Dict) -> Tuple[float, Dict]:
    _, reporting_source, include_treasury, manual = _nav_sources(cfg)
    if reporting_source == "manual":
        if manual is not None:
            return float(manual), {"source": "manual"}
        return _fallback_capital(cfg)

    summary = compute_nav_summary(cfg)
    fut_detail = summary["details"].get("futures", {})
    detail: Dict[str, Any] = {"source": "exchange", **fut_detail}
    total_nav = float(summary["futures_nav"])

    if include_treasury:
        treasury_detail = summary["details"].get("treasury", {})
        holdings = treasury_detail.get("treasury", {}) if isinstance(treasury_detail, dict) else {}
        if holdings:
            detail["treasury"] = holdings
        missing = treasury_detail.get("missing_prices") if isinstance(treasury_detail, dict) else None
        if missing:
            detail["treasury_missing_prices"] = missing
        detail["treasury_total_usdt"] = float(summary["treasury_nav"])

        reserves_detail = summary["details"].get("reserves", {})
        reserves_map = reserves_detail.get("reserves") if isinstance(reserves_detail, dict) else {}
        if reserves_map:
            detail["reserves"] = reserves_map
        if isinstance(reserves_detail, dict) and reserves_detail.get("error"):
            detail["reserves_error"] = reserves_detail.get("error")
        reserves_total = float(summary.get("reserves_nav", 0.0) or 0.0)
        if reserves_total > 0:
            detail["reserves_total_usd"] = reserves_total

        if reserves_total > 0:
            detail["source"] = "exchange+treasury+reserves"
        else:
            detail["source"] = "exchange+treasury"

        total_nav = float(summary["futures_nav"] + summary["treasury_nav"] + reserves_total)

    if total_nav > 0:
        return total_nav, detail
    return _fallback_capital(cfg)


def compute_nav_pair(cfg: Dict) -> Tuple[Tuple[float, Dict], Tuple[float, Dict]]:
    trading = compute_trading_nav(cfg)
    reporting = compute_reporting_nav(cfg)
    return trading, reporting


def compute_treasury_only() -> Tuple[float, Dict]:
    try:
        total, breakdown = _treasury_nav_usdt()
        return float(total), breakdown
    except Exception as exc:
        return 0.0, {"treasury": {}, "error": str(exc)}


def compute_nav(cfg: Dict) -> Tuple[float, Dict]:
    # Backwards-compatible proxy for callers expecting single NAV
    return compute_trading_nav(cfg)


def compute_symbol_gross_usd() -> Dict[str, float]:
    """Return per-symbol absolute gross exposure in USD."""
    try:
        positions = get_positions() or []
    except Exception:
        return {}
    gross: Dict[str, float] = {}
    for pos in positions:
        try:
            qty = float(pos.get("qty", pos.get("positionAmt", 0.0)) or 0.0)
            if qty == 0.0:
                continue
            mark = float(pos.get("markPrice") or pos.get("entryPrice") or 0.0)
            if mark <= 0:
                symbol = pos.get("symbol")
                if symbol:
                    try:
                        mark = float(get_price(str(symbol)))
                    except Exception:
                        mark = 0.0
            if mark <= 0:
                continue
            symbol = str(pos.get("symbol", "")).upper()
            gross[symbol] = gross.get(symbol, 0.0) + abs(qty) * abs(mark)
        except Exception:
            continue
    return gross


def compute_gross_exposure_usd() -> float:
    """Aggregate absolute notional exposure across all open futures positions."""
    gross_map = compute_symbol_gross_usd()
    return float(sum(gross_map.values()))


def _load_nav_series() -> List[Dict[str, Any]]:
    if not os.path.exists(_NAV_LOG_PATH):
        return []
    try:
        with open(_NAV_LOG_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def write_nav(nav_value: float) -> None:
    """Append a NAV point with timestamp to the nav log."""
    try:
        nav_float = float(nav_value)
    except Exception as exc:
        LOGGER.error("[nav] write_failed: invalid nav value (%s)", exc)
        return
    if not math.isfinite(nav_float):
        LOGGER.error("[nav] write_failed: non-finite nav value %s", nav_value)
        return

    log_dir = os.path.dirname(_NAV_LOG_PATH) or "."
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception as exc:
        LOGGER.error("[nav] write_failed: mkdir %s (%s)", log_dir, exc)
        return

    if not os.path.exists(_NAV_LOG_PATH):
        try:
            with open(_NAV_LOG_PATH, "w", encoding="utf-8") as handle:
                json.dump([], handle)
                handle.write("\n")
        except Exception as exc:
            LOGGER.error("[nav] write_failed: init log %s (%s)", _NAV_LOG_PATH, exc)
            return

    ts = time.time()
    entry = {"t": ts, "nav": nav_float}
    try:
        series = _load_nav_series()
        series.append(entry)
        with open(_NAV_LOG_PATH, "w", encoding="utf-8") as handle:
            json.dump(series, handle, indent=2)
            handle.write("\n")
    except Exception as exc:
        LOGGER.error("[nav] write_failed: %s", exc)
        return
    LOGGER.info("[nav] write nav=%.2f ts=%.3f path=%s", nav_float, ts, _NAV_LOG_PATH)


def _persist_confirmed_nav(
    nav_value: float,
    detail: Dict[str, Any] | None = None,
    source_health: Dict[str, Any] | None = None,
) -> None:
    try:
        nav_float = float(nav_value)
    except Exception:
        return
    if not math.isfinite(nav_float):
        return
    record: Dict[str, Any] = {
        "ts": time.time(),
        "nav": nav_float,
    }
    record["nav_usd"] = nav_float
    health: Dict[str, bool] = {}
    if isinstance(source_health, dict):
        health = {str(key): bool(val) for key, val in source_health.items()}
    sources_ok = all(health.values()) if health else True
    record["sources_ok"] = sources_ok
    if health:
        record["source_health"] = health
        record["stale_flags"] = {key: not val for key, val in health.items()}
    if isinstance(detail, dict) and detail:
        record["detail"] = detail
    try:
        os.makedirs(os.path.dirname(_NAV_CACHE_PATH), exist_ok=True)
    except Exception as exc:
        LOGGER.error("[nav] snapshot_mkdir_failed: %s", exc)
        return
    try:
        with open(_NAV_CACHE_PATH, "w", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True)
        LOGGER.info(
            "[nav] snapshot ts=%.0f nav_usd=%.2f sources_ok=%s path=%s",
            record["ts"],
            nav_float,
            sources_ok,
            _NAV_CACHE_PATH,
        )
    except Exception as exc:
        LOGGER.error("[nav] snapshot_write_failed: %s", exc)


def get_confirmed_nav() -> Dict[str, Any]:
    cached = _load_json(_NAV_CACHE_PATH)
    if not isinstance(cached, dict):
        return {}
    out: Dict[str, Any] = {}
    ts_val = cached.get("ts")
    try:
        ts_float = float(ts_val)
        if math.isfinite(ts_float) and ts_float > 0:
            out["ts"] = ts_float
    except Exception:
        pass
    nav_val = cached.get("nav")
    try:
        nav_candidate = float(nav_val)
        if math.isfinite(nav_candidate):
            out["nav"] = nav_candidate
    except Exception:
        pass
    nav_usd_val = cached.get("nav_usd", nav_val)
    try:
        nav_usd_candidate = float(nav_usd_val)
        if math.isfinite(nav_usd_candidate):
            out["nav_usd"] = nav_usd_candidate
            if "nav" not in out:
                out["nav"] = nav_usd_candidate
    except Exception:
        pass
    detail = cached.get("detail")
    if isinstance(detail, dict):
        out["detail"] = detail
    health = cached.get("source_health")
    if isinstance(health, dict):
        sanitized_health = {str(key): bool(val) for key, val in health.items()}
        out["source_health"] = sanitized_health
        if "sources_ok" not in out:
            out["sources_ok"] = all(sanitized_health.values())
    sources_ok_val = cached.get("sources_ok")
    if "sources_ok" not in out:
        if isinstance(sources_ok_val, bool):
            out["sources_ok"] = sources_ok_val
        elif sources_ok_val is not None:
            try:
                out["sources_ok"] = bool(sources_ok_val)
            except Exception:
                pass
    stale_flags = cached.get("stale_flags")
    if isinstance(stale_flags, dict):
        out["stale_flags"] = {str(key): bool(val) for key, val in stale_flags.items()}
    return out


def get_nav_age(default: float | None = None) -> float | None:
    """Return age in seconds for the last confirmed NAV, or default if unknown."""
    record = get_confirmed_nav()
    ts_val = record.get("ts")
    if not isinstance(ts_val, (int, float)):
        return default
    try:
        age = max(0.0, time.time() - float(ts_val))
    except Exception:
        return default
    return age


def is_nav_fresh(threshold_s: float | int | None = None) -> bool:
    try:
        threshold = float(threshold_s or 0.0)
    except Exception:
        threshold = 0.0
    if threshold <= 0.0:
        return True
    age = get_nav_age()
    if age is None:
        return False
    return age <= threshold


class PortfolioSnapshot:
    """Single-call helper to expose current NAV and gross exposure."""

    def __init__(self, cfg: Dict | None = None) -> None:
        self.cfg = cfg or _load_json("config/strategy_config.json")
        self._nav: float | None = None
        self._gross: float | None = None
        self._symbol_gross: Dict[str, float] = {}
        self._stale = True

    def refresh(self) -> None:
        try:
            nav_val, _ = compute_trading_nav(self.cfg)
            self._nav = float(nav_val or 0.0)
        except Exception:
            self._nav = float(self.cfg.get("capital_base_usdt", 0.0) or 0.0)
        self._symbol_gross = compute_symbol_gross_usd()
        self._gross = float(sum(self._symbol_gross.values()))
        self._stale = False

    def current_nav_usd(self) -> float:
        if self._stale or self._nav is None:
            self.refresh()
        return float(self._nav or 0.0)

    def current_gross_usd(self) -> float:
        if self._stale or self._gross is None:
            self.refresh()
        return float(self._gross or 0.0)

    def symbol_gross_usd(self) -> Dict[str, float]:
        if self._stale:
            self.refresh()
        return dict(self._symbol_gross)


__all__ = [
    "compute_nav",
    "compute_trading_nav",
    "compute_reporting_nav",
    "compute_nav_pair",
    "compute_treasury_only",
    "compute_gross_exposure_usd",
    "compute_nav_summary",
    "get_confirmed_nav",
    "is_nav_fresh",
    "PortfolioSnapshot",
    "write_nav",
]
