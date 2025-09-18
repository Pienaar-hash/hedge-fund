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
        symbol = f"{asset}USDT"
        try:
            price = float(get_price(symbol))
        except Exception as exc:
            missing[asset] = str(exc)
            continue
        value = qty_float * price
        total += value
        holdings[asset] = {
            "qty": qty_float,
            "px": price,
            "val_usdt": value,
        }
    breakdown: Dict[str, Any] = {"treasury": holdings}
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


def compute_reporting_nav(cfg: Dict) -> Tuple[float, Dict]:
    _, reporting_source, include_treasury, manual = _nav_sources(cfg)
    if reporting_source == "manual":
        if manual is not None:
            return float(manual), {"source": "manual"}
        return _fallback_capital(cfg)

    fut_nav, fut_detail = _futures_nav_usdt()
    detail: Dict[str, Any] = {"source": "exchange", **fut_detail}
    if include_treasury:
        detail["treasury_note"] = "treasury_excluded"
    if fut_nav > 0:
        return float(fut_nav), detail
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


def compute_gross_exposure_usd() -> float:
    """Aggregate absolute notional exposure across all open futures positions."""
    try:
        positions = get_positions() or []
    except Exception:
        return 0.0
    gross = 0.0
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
            gross += abs(qty) * abs(mark)
        except Exception:
            continue
    return float(gross)


class PortfolioSnapshot:
    """Single-call helper to expose current NAV and gross exposure."""

    def __init__(self, cfg: Dict | None = None) -> None:
        self.cfg = cfg or _load_json("config/strategy_config.json")
        self._nav: float | None = None
        self._gross: float | None = None
        self._stale = True

    def refresh(self) -> None:
        try:
            nav_val, _ = compute_trading_nav(self.cfg)
            self._nav = float(nav_val or 0.0)
        except Exception:
            self._nav = float(self.cfg.get("capital_base_usdt", 0.0) or 0.0)
        self._gross = compute_gross_exposure_usd()
        self._stale = False

    def current_nav_usd(self) -> float:
        if self._stale or self._nav is None:
            self.refresh()
        return float(self._nav or 0.0)

    def current_gross_usd(self) -> float:
        if self._stale or self._gross is None:
            self.refresh()
        return float(self._gross or 0.0)


__all__ = [
    "compute_nav",
    "compute_trading_nav",
    "compute_reporting_nav",
    "compute_nav_pair",
    "compute_treasury_only",
    "compute_gross_exposure_usd",
    "PortfolioSnapshot",
]
