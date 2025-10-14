from __future__ import annotations

import json
from typing import Any, Dict, Tuple

from execution.exchange_utils import get_balances, get_positions, get_price


def _load_json(path: str) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _futures_nav_usdt() -> Tuple[float, Dict]:
    """Compute USD-M futures wallet NAV and provide detail."""
    bal = get_balances() or {}
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
        unreal = 0.0
        for pos in positions:
            try:
                unreal += float(pos.get("unrealized", 0.0))
            except Exception:
                continue
        detail["unrealized_pnl"] = unreal
        wallet += unreal
    except Exception:
        pass
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

    summary = {
        "futures_nav": float(futures_nav),
        "treasury_nav": float(treasury_nav),
        "total_nav": float(futures_nav + treasury_nav),
        "details": {
            "futures": futures_detail,
            "treasury": treasury_detail,
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
        detail["source"] = "exchange+treasury"
        total_nav = float(summary["total_nav"])

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
    "PortfolioSnapshot",
]
