from __future__ import annotations

"""
v7 AUM source: AUM = futures NAV (pure) + off-exchange holdings.
Legacy reserves/treasury handling removed.
"""

import json
import math
import os
import threading
import time
import logging
from typing import Any, Dict, List, Tuple, Optional, cast

from execution.exchange_utils import (
    get_balances,
    get_positions,
    get_price,
    get_futures_balances,
    get_um_client,
)
from execution.risk_loader import load_risk_config

_NAV_CACHE_PATH = "logs/cache/nav_confirmed.json"
_NAV_LOG_PATH = "logs/nav_log.json"
_NAV_HEALTH_PATH = "logs/nav_health.json"
_NAV_WRITER_DEFAULT_INTERVAL = float(os.environ.get("NAV_WRITER_INTERVAL_SEC", "60"))
_NAV_FRESHNESS_SECONDS = float(os.environ.get("NAV_FRESHNESS_SECONDS", "90"))
_NAV_CACHE_MAX_AGE_SECONDS = float(os.environ.get("NAV_CACHE_MAX_AGE_S", "900"))
_OFFEXCHANGE_PATH = os.getenv("OFFEXCHANGE_HOLDINGS_PATH", "config/offexchange_holdings.json")

LOGGER = logging.getLogger("nav")

JSONDict = Dict[str, Any]
JSONList = List[JSONDict]


def _normalize_balances(raw: Any) -> Dict[str, float]:
    """Coerce exchange balance payloads (dict or list) into an asset->balance map."""
    out: Dict[str, float] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            try:
                out[str(key).upper()] = float(value)
            except Exception:
                continue
        # Futures balance payloads sometimes expose walletBalance without asset key
        if "USDT" not in out and "WALLETBALANCE" in out:
            try:
                out["USDT"] = float(out["WALLETBALANCE"])
            except Exception:
                pass
        return out

    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            asset = str(entry.get("asset") or entry.get("code") or "").upper()
            if not asset:
                continue
            value = (
                entry.get("balance")
                or entry.get("walletBalance")
                or entry.get("crossWalletBalance")
                or entry.get("availableBalance")
                or entry.get("available")
            )
            try:
                out[asset] = float(value or 0.0)
            except Exception:
                continue
    return out


def _mark_price_usdt(asset: str) -> float:
    """Best-effort mark/ticker price resolver for non-stable assets."""
    sym = str(asset or "").upper()
    if not sym or sym in {"USDT", "USDC"}:
        return 1.0
    try:
        px = get_price(f"{sym}USDT")
        return float(px or 0.0)
    except Exception as exc:
        LOGGER.warning("[nav] mark_price_fetch_failed asset=%s err=%s", sym, exc)
        return 0.0


def get_mark_price_for_symbol(symbol: str) -> float:
    """Alias used by off-exchange holdings valuation."""
    return _mark_price_usdt(symbol)


def _get_usd_fallback_price(symbol: str, default: float | None = None):
    """
    Fallback USD spot price for off-exchange holdings when UM mark price is unavailable.
    Order:
        1) coingecko price (if available)
        2) default (avg_cost from config)
    Returns None if nothing available.
    """
    try:
        from execution.exchange_utils import coingecko_price_usd
    except Exception:
        coingecko_price_usd = None

    if coingecko_price_usd:
        try:
            px = coingecko_price_usd(symbol)
            if px and px > 0:
                return float(px)
        except Exception:
            pass
    return default


def load_offexchange_holdings(path: str | None = None) -> Dict[str, Dict[str, float]]:
    """Load off-exchange holdings from config; returns empty dict on failure."""
    target = path or _OFFEXCHANGE_PATH
    try:
        with open(target, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            holdings: Dict[str, Dict[str, float]] = {}
            for sym, data in payload.items():
                if not isinstance(data, dict):
                    continue
                try:
                    qty = float(data.get("qty", 0.0) or 0.0)
                except Exception:
                    qty = 0.0
                avg_cost = data.get("avg_cost")
                try:
                    avg_cost = float(avg_cost) if avg_cost is not None else None
                except Exception:
                    avg_cost = None
                holdings[str(sym).upper()] = {"qty": qty, "avg_cost": avg_cost}
            return holdings
    except FileNotFoundError:
        return {}
    except Exception:
        LOGGER.debug("[nav] offexchange_holdings_load_failed path=%s", target, exc_info=True)
    return {}


def _load_json(path: str) -> JSONDict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
            if isinstance(loaded, dict):
                return cast(JSONDict, dict(loaded))
    except Exception:
        pass
    return {}


def _live_nav_snapshot(quote_symbols: set[str]) -> Tuple[float, Dict[str, Any]]:
    balances_ok = False
    conversions: Dict[str, Any] = {}
    quote_breakdown: Dict[str, float] = {}
    breakdown: Dict[str, float] = {}
    try:
        balances = get_futures_balances() or {}
        balances_ok = bool(balances)
    except Exception as exc:
        LOGGER.warning("[nav] futures_balances_failed: %s", exc)
        balances = {}
    total_nav = 0.0
    for asset, raw_amt in balances.items():
        try:
            amt = float(raw_amt or 0.0)
        except Exception:
            continue
        asset_key = str(asset).upper()
        if asset_key in quote_symbols:
            quote_breakdown[asset_key] = amt
            breakdown[asset_key] = amt
            total_nav += amt
            continue
        if asset_key not in {"BTC", "ETH"}:
            continue
        px = _ticker_last_price(f"{asset_key}USDT")
        usd_val = amt * px if px > 0 else 0.0
        conversions[asset_key] = {"amount": amt, "price": px, "value_usd": usd_val}
        breakdown[asset_key] = usd_val
        total_nav += usd_val
    detail = {
        "breakdown": breakdown,
        "quote_breakdown": quote_breakdown,
        "conversions": conversions,
        "futures_balances": balances,
        "fresh": balances_ok,
        "source": "live" if balances_ok else "cache",
    }
    return total_nav, detail


def _ticker_last_price(symbol: str) -> float:
    try:
        client = get_um_client()
        resp = client.ticker_price(symbol=symbol)
        if isinstance(resp, dict):
            price_val = resp.get("price") or resp.get("lastPrice")
        else:
            price_val = resp
        return float(price_val or 0.0)
    except Exception as exc:
        LOGGER.warning("[nav] ticker_price_failed symbol=%s err=%s", symbol, exc)
        return 0.0


def _attach_aum(nav_total: float, snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Augment NAV snapshot with off-exchange holdings and total AUM."""
    holdings = load_offexchange_holdings()
    offexchange_usd: Dict[str, Dict[str, Any]] = {}
    for sym, data in holdings.items():
        qty = float(data.get("qty", 0.0) or 0.0)
        avg_cost = data.get("avg_cost")
        mark = get_mark_price_for_symbol(sym)

        # Try futures mark price first; fallback to coingecko/avg_cost if unavailable
        spot_usd = None
        if not mark or mark <= 0.0:
            spot_usd = _get_usd_fallback_price(sym, default=avg_cost)
            if spot_usd and spot_usd > 0:
                usd_value = qty * spot_usd
            else:
                usd_value = None
        else:
            usd_value = qty * mark

        try:
            import logging

            logger = logging.getLogger(__name__)
            logger.info(
                "[aum-v7-detail] symbol=%s qty=%s avg_cost=%s mark=%s fallback=%s usd_value=%s",
                sym,
                qty,
                avg_cost,
                mark if (mark and mark > 0) else None,
                spot_usd,
                usd_value,
            )
        except Exception:
            pass

        offexchange_usd[sym] = {
            "qty": qty,
            "avg_cost": avg_cost,
            "mark": mark if mark and mark > 0 else spot_usd,
            "usd_value": usd_value,
        }
    off_total = sum(v.get("usd_value", 0.0) or 0.0 for v in offexchange_usd.values())
    aum_total = float(nav_total) + float(off_total)
    try:
        import logging

        logger = logging.getLogger(__name__)
        logger.info(
            "[aum-v7-summary] offexchange_keys=%s offexchange_usd_total=%s",
            sorted(offexchange_usd.keys()),
            off_total,
        )
    except Exception:
        pass
    snapshot["aum"] = {
        "futures": float(nav_total),
        "offexchange": offexchange_usd,
        "total": aum_total,
    }
    return snapshot


def _write_nav_health(detail: Dict[str, Any]) -> None:
    payload = {
        "nav_total": detail.get("total_nav"),
        "breakdown": detail.get("breakdown") or {},
        "quote_breakdown": detail.get("quote_breakdown") or {},
        "conversions": detail.get("conversions") or {},
        "fresh": detail.get("fresh"),
        "source": detail.get("source"),
        "ts": time.time(),
    }
    payload["sources"] = {
        "balances": bool(detail.get("fresh")),
        "cache_used": detail.get("source") == "cache",
    }
    try:
        os.makedirs(os.path.dirname(_NAV_HEALTH_PATH), exist_ok=True)
        with open(_NAV_HEALTH_PATH, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
    except Exception as exc:
        LOGGER.debug("[nav] nav_health_write_failed: %s", exc)


def _futures_nav_usdt(nav_cfg: Optional[Dict[str, Any]] = None) -> Tuple[float, JSONDict]:
    """Compute USD-M futures wallet NAV and provide detail."""
    balances_ok = False
    positions_ok = False
    try:
        bal = get_balances() or {}
        balances_ok = True
    except Exception as exc:
        LOGGER.warning("[nav] balances_fetch_failed: %s", exc)
        bal = {}

    include_cfg = None
    try:
        include_cfg = (nav_cfg or {}).get("include_assets")
    except Exception:
        include_cfg = None
    # Support both dict-of-balances and list-of-dicts returns
    asset_breakdown: Dict[str, float] = _normalize_balances(bal)
    wallet = float(asset_breakdown.get("USDT", 0.0))
    configured_assets = include_cfg if isinstance(include_cfg, list) else None
    include_assets = {str(a).upper() for a in configured_assets or [] if str(a).strip()}
    if not include_assets:
        include_assets = {"USDT"}
    use_mark_price = bool((nav_cfg or {}).get("use_mark_price"))
    nav_mode = "enhanced" if configured_assets is not None or use_mark_price else "legacy"

    assets_nav: Dict[str, float] = {}
    mark_prices: Dict[str, float] = {}
    mark_price_failures = 0
    for asset, amount in asset_breakdown.items():
        if asset not in include_assets:
            continue
        if asset in {"USDT", "USDC"}:
            assets_nav[asset] = float(amount)
            continue
        if not use_mark_price:
            assets_nav[asset] = 0.0
            mark_price_failures += 1
            continue
        price = _mark_price_usdt(asset)
        if price > 0.0:
            mark_prices[asset] = price
            assets_nav[asset] = float(amount) * price
        else:
            assets_nav[asset] = 0.0
            mark_price_failures += 1

    nav_total = float(sum(assets_nav.values()))
    detail: JSONDict = {
        "futures_wallet_usdt": wallet,
        "asset_breakdown": asset_breakdown,
        "assets": assets_nav,
        "mark_prices": mark_prices,
        "nav_mode": nav_mode,
        "source": nav_mode,
        "ts": time.time(),
    }
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
        nav_total += unreal
    except Exception as exc:
        LOGGER.warning("[nav] positions_fetch_failed: %s", exc)
    detail["balances_ok"] = bool(balances_ok)
    detail["positions_ok"] = bool(positions_ok)
    source_health = {
        "balances_ok": balances_ok,
        "positions_ok": positions_ok,
    }
    sources_ok = all(source_health.values())
    detail["sources_ok"] = sources_ok

    try:
        cached_age = get_nav_age()
        cached_age_int = int(cached_age) if cached_age is not None else None
    except Exception:
        cached_age_int = None
    mark_prices_fresh = not use_mark_price or mark_price_failures == 0
    detail["freshness"] = {
        "exchange_balances_fresh": bool(balances_ok),
        "mark_prices_fresh": bool(mark_prices_fresh),
        "cached_nav_age_s": cached_age_int,
    }
    detail["nav"] = nav_total
    try:
        _attach_aum(nav_total, detail)
    except Exception:
        LOGGER.debug("[nav] attach_aum_failed", exc_info=True)

    if sources_ok:
        _persist_confirmed_nav(nav_total, detail=detail, source_health=source_health)
    else:
        LOGGER.warning(
            "[nav] snapshot_mark_unhealthy balances_ok=%s positions_ok=%s",
            balances_ok,
            positions_ok,
        )
        _mark_nav_unhealthy(detail=detail, source_health=source_health)
    return nav_total, detail


def _treasury_nav_usdt(_: str = "config/treasury.json") -> Tuple[float, JSONDict]:
    """Treasury valuations are no longer part of NAV (v5.10+)."""
    return 0.0, {"treasury": {}}


def _reserves_nav_usd() -> Tuple[float, JSONDict]:
    """Reserve valuations are excluded from NAV (v5.10+)."""
    return 0.0, {"reserves": {}}


def _nav_sources(cfg: JSONDict) -> Tuple[str, str, bool, Any]:
    nav_cfg = cfg.get("nav") or {}
    trading_source = nav_cfg.get("trading_source") or nav_cfg.get("source") or "exchange"
    reporting_source = nav_cfg.get("reporting_source") or trading_source
    include_treasury = bool(nav_cfg.get("include_spot_treasury", False))
    manual = nav_cfg.get("manual_nav_usdt")
    return str(trading_source), str(reporting_source), include_treasury, manual


def compute_trading_nav(cfg: JSONDict) -> Tuple[float, JSONDict]:
    nav_cfg = (cfg or {}).get("nav") or {}
    trading_source, _, _, manual = _nav_sources(cfg)
    if trading_source == "manual":
        if manual is not None:
            detail = {
                "source": "manual",
                "total_nav": float(manual),
                "nav": float(manual),
                "nav_usd": float(manual),
                "fresh": True,
            }
            try:
                _attach_aum(float(manual), detail)
            except Exception:
                LOGGER.debug("[nav] attach_aum_manual_failed", exc_info=True)
            _write_nav_health(detail)
            return float(manual), detail
        capital, detail = _fallback_capital(cfg)
        detail["source"] = "capital_base"
        detail["total_nav"] = float(capital)
        detail["nav"] = float(capital)
        detail["nav_usd"] = float(capital)
        _write_nav_health(detail)
        return capital, detail

    risk_cfg = load_risk_config()
    quote_symbols = {str(q).upper() for q in (risk_cfg.get("quote_symbols") or [])}
    fut_nav, fut_detail = _live_nav_snapshot(quote_symbols)
    fut_detail["nav_mode"] = "live_wallet"
    fut_detail["total_nav"] = fut_nav
    fut_detail["nav"] = fut_nav
    fut_detail["nav_usd"] = fut_nav
    fut_detail.setdefault("assets", fut_detail.get("breakdown", {}))
    fut_detail.setdefault("asset_breakdown", fut_detail.get("breakdown", {}))
    if fut_detail.get("fresh") and fut_nav > 0:
        fut_detail["source"] = "live"
        _persist_confirmed_nav(
            fut_nav,
            detail=fut_detail,
            source_health={"balances_ok": True},
        )
        _write_nav_health(fut_detail)
        return float(fut_nav), fut_detail

    nav_health = nav_health_snapshot()
    confirmed = get_confirmed_nav()
    confirmed_nav = float(confirmed.get("nav") or confirmed.get("nav_usd") or 0.0) if confirmed else 0.0
    cache_detail = confirmed.get("detail") if isinstance(confirmed, dict) else {}
    cache_breakdown = {}
    if isinstance(cache_detail, dict):
        cache_breakdown = cache_detail.get("breakdown") or cache_detail.get("assets") or {}
    fallback_detail: Dict[str, Any] = {
        "source": "cache" if confirmed_nav > 0 else "unavailable",
        "breakdown": cache_breakdown,
        "asset_breakdown": cache_breakdown,
        "total_nav": confirmed_nav,
        "nav": confirmed_nav,
        "nav_usd": confirmed_nav,
        "fresh": bool(nav_health.get("fresh")),
        "nav_health": nav_health,
    }
    try:
        _attach_aum(confirmed_nav, fallback_detail)
    except Exception:
        LOGGER.debug("[nav] attach_aum_cache_failed", exc_info=True)
    _write_nav_health(fallback_detail)
    return confirmed_nav, fallback_detail


def compute_nav_summary(cfg: JSONDict | None = None) -> JSONDict:
    cfg = _load_strategy_cfg(cfg)
    futures_nav, futures_detail = compute_trading_nav(cfg)
    return {
        "futures_nav": float(futures_nav),
        "total_nav": float(futures_nav),
        "details": {"futures": futures_detail},
    }


def _fallback_capital(cfg: JSONDict) -> Tuple[float, JSONDict]:
    """Fallback NAV when no positions or manual override exist."""
    base = float(cfg.get("capital_base_usdt", 0.0))
    detail = {"source": "fallback_capital", "nav": base, "nav_usd": base, "total_nav": base}
    try:
        _attach_aum(base, detail)
    except Exception:
        LOGGER.debug("[nav] attach_aum_fallback_failed", exc_info=True)
    return base, detail


def compute_reporting_nav(cfg: JSONDict) -> Tuple[float, JSONDict]:
    _, reporting_source, _, manual = _nav_sources(cfg)
    if reporting_source == "manual":
        if manual is not None:
            nav_val = float(manual)
            detail = {"source": "manual", "nav": nav_val, "nav_usd": nav_val, "total_nav": nav_val}
            try:
                _attach_aum(nav_val, detail)
            except Exception:
                LOGGER.debug("[nav] attach_aum_reporting_manual_failed", exc_info=True)
            return nav_val, detail
        return _fallback_capital(cfg)

    nav_val, detail = compute_trading_nav(cfg)
    if nav_val > 0:
        enriched = {"source": (detail or {}).get("source", "live")}
        if isinstance(detail, dict):
            enriched.update(detail)
        return nav_val, enriched

    confirmed = get_confirmed_nav()
    confirmed_nav = float(confirmed.get("nav") or 0.0) if confirmed else 0.0
    if confirmed_nav > 0:
        cache_detail = {
            "source": "exchange_cache",
            "confirmed_ts": confirmed.get("ts"),
            "sources_ok": confirmed.get("sources_ok"),
        }
        if confirmed.get("stale_flags"):
            cache_detail["stale_flags"] = confirmed["stale_flags"]
        return confirmed_nav, cache_detail
    return 0.0, {"source": "unavailable"}


def compute_nav_pair(cfg: JSONDict) -> Tuple[Tuple[float, JSONDict], Tuple[float, JSONDict]]:
    trading = compute_trading_nav(cfg)
    reporting = compute_reporting_nav(cfg)
    return trading, reporting


def compute_treasury_only() -> Tuple[float, JSONDict]:
    return 0.0, {"treasury": {}}


def compute_aum_breakdown(cfg: JSONDict | None = None) -> JSONDict:
    """
    Return the total AUM split without mutating NAV consumers.
    """
    summary = compute_nav_summary(cfg)
    return {
        "trading_nav": float(summary["futures_nav"]),
        "treasury_nav": 0.0,
        "reserves_nav": 0.0,
        "total_aum": float(summary["futures_nav"]),
    }


def compute_nav(cfg: JSONDict) -> Tuple[float, JSONDict]:
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


def _load_strategy_cfg(existing: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if isinstance(existing, dict) and existing:
        return dict(existing)
    cfg = _load_json("config/strategy_config.json")
    if not isinstance(cfg, dict):
        cfg = {}
    # Optionally merge runtime nav overrides (runtime.yaml)
    try:
        from execution.runtime_config import load_runtime_config

        runtime_cfg = load_runtime_config()
        if isinstance(runtime_cfg, dict):
            rt_nav = runtime_cfg.get("nav")
            if isinstance(rt_nav, dict):
                nav_cfg = cfg.get("nav") if isinstance(cfg.get("nav"), dict) else {}
                merged = dict(nav_cfg)
                merged.update(rt_nav)
                cfg["nav"] = merged
    except Exception:
        pass
    return cfg


def run_nav_writer(
    interval_s: float | int = _NAV_WRITER_DEFAULT_INTERVAL,
    cfg: Optional[Dict[str, Any]] = None,
    stop_event: Optional["threading.Event"] = None,
) -> None:
    """Continuously record NAV to the log on a timer."""

    try:
        interval = max(float(interval_s), 5.0)
    except Exception:
        interval = _NAV_WRITER_DEFAULT_INTERVAL
    LOGGER.info("[nav] nav_writer_start interval=%.1fs", interval)
    cfg_data = _load_strategy_cfg(cfg)

    while True:
        start_ts = time.time()
        try:
            nav_val, detail = compute_trading_nav(cfg_data)
            if not math.isfinite(float(nav_val)) or float(nav_val) <= 0.0:
                fallback = get_confirmed_nav()
                nav_val = float(fallback.get("nav") or fallback.get("nav_usd") or 0.0)
            if float(nav_val) > 0.0:
                write_nav(float(nav_val))
            else:
                LOGGER.warning(
                    "[nav] nav_writer_skip nav<=0 source=%s",
                    (detail or {}).get("source"),
                )
        except Exception as exc:
            LOGGER.warning("[nav] nav_writer_iteration_failed: %s", exc)

        if stop_event and stop_event.is_set():
            LOGGER.info("[nav] nav_writer_stop signal_received")
            break

        remaining = max(0.0, interval - (time.time() - start_ts))
        if stop_event:
            if stop_event.wait(remaining):
                LOGGER.info("[nav] nav_writer_stop signal_received")
                break
        else:
            time.sleep(remaining)


def start_nav_writer(
    interval_s: float | int = _NAV_WRITER_DEFAULT_INTERVAL,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[threading.Thread, "threading.Event"]:
    """Spawn a daemon thread that runs the NAV writer loop."""

    stop_event: "threading.Event" = threading.Event()
    thread = threading.Thread(
        target=run_nav_writer,
        args=(interval_s, cfg, stop_event),
        name="nav-writer",
        daemon=True,
    )
    thread.start()
    return thread, stop_event


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


def _mark_nav_unhealthy(
    *,
    detail: Optional[Dict[str, Any]] = None,
    source_health: Optional[Dict[str, Any]] = None,
) -> None:
    """Mark the confirmed NAV snapshot as stale without overwriting last good value."""
    try:
        existing = _load_json(_NAV_CACHE_PATH)
        record: Dict[str, Any] = dict(existing if isinstance(existing, dict) else {})
    except Exception:
        record = {}

    stale_ts = time.time()
    nav_val: float
    try:
        nav_val = float(record.get("nav") or record.get("nav_usd") or 0.0)
    except Exception:
        nav_val = 0.0
    record["nav"] = nav_val
    record["nav_usd"] = nav_val
    record["sources_ok"] = False
    sanitized: Dict[str, bool] = {}
    if isinstance(source_health, dict):
        sanitized = {str(key): bool(val) for key, val in source_health.items()}
    record["source_health"] = sanitized
    record["stale_flags"] = {key: not val for key, val in sanitized.items()}
    if isinstance(detail, dict) and detail:
        detail_payload = dict(detail)
        detail_payload["sources_ok"] = False
        record["detail"] = detail_payload
    record["ts"] = stale_ts
    record["stale_ts"] = stale_ts
    try:
        os.makedirs(os.path.dirname(_NAV_CACHE_PATH), exist_ok=True)
    except Exception as exc:
        LOGGER.error("[nav] snapshot_mkdir_failed: %s", exc)
        return
    try:
        with open(_NAV_CACHE_PATH, "w", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True)
        LOGGER.warning(
            "[nav] snapshot_marked_stale ts=%.0f path=%s", stale_ts, _NAV_CACHE_PATH
        )
    except Exception as exc:
        LOGGER.error("[nav] snapshot_mark_stale_failed: %s", exc)


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
    stale_ts = cached.get("stale_ts") or cached.get("last_failure_ts")
    if stale_ts is not None:
        try:
            out["stale_ts"] = float(stale_ts)
        except Exception:
            pass
    return out


def get_nav_age(default: float | None = None) -> float | None:
    """Return age in seconds for the last confirmed NAV, or default if unknown."""
    health = nav_health_snapshot()
    age_val = health.get("age_s")
    if age_val is None:
        return default
    try:
        return float(age_val)
    except Exception:
        return default


def nav_health_snapshot(threshold_s: float | int | None = None) -> Dict[str, Any]:
    """Return a consistent NAV health snapshot (age, freshness, sources)."""
    try:
        threshold = float(threshold_s) if threshold_s is not None else _NAV_FRESHNESS_SECONDS
    except Exception:
        threshold = _NAV_FRESHNESS_SECONDS
    if threshold <= 0.0:
        threshold = _NAV_FRESHNESS_SECONDS
    record = get_confirmed_nav()
    ts_val = record.get("ts")
    try:
        age = max(0.0, time.time() - float(ts_val)) if ts_val is not None else None
    except Exception:
        age = None
    stale_flags = record.get("stale_flags") if isinstance(record, dict) else {}
    stale_flags = {str(k): bool(v) for k, v in (stale_flags or {}).items()}
    sources_ok = bool(record.get("sources_ok", True))
    if stale_flags and any(stale_flags.values()):
        sources_ok = False
    fresh = sources_ok and age is not None and age <= threshold
    return {
        "age_s": age,
        "sources_ok": sources_ok,
        "stale_flags": stale_flags,
        "fresh": fresh,
        "threshold_s": threshold,
        "cache_ts": record.get("ts"),
        "cache_path": _NAV_CACHE_PATH,
        "nav_total": float(record.get("nav") or record.get("nav_usd") or 0.0),
    }


def is_nav_fresh(threshold_s: float | int | None = None) -> bool:
    return bool(nav_health_snapshot(threshold_s).get("fresh"))


class PortfolioSnapshot:
    """Single-call helper to expose current NAV and gross exposure."""

    def __init__(self, cfg: Dict | None = None) -> None:
        self.cfg = _load_strategy_cfg(cfg)
        self._nav: float | None = None
        self._gross: float | None = None
        self._symbol_gross: Dict[str, float] = {}
        self._stale = True

    def refresh(self) -> None:
        try:
            nav_val, _ = compute_trading_nav(self.cfg)
            self._nav = float(nav_val or 0.0)
        except Exception:
            confirmed = get_confirmed_nav()
            age = get_nav_age()
            sources_ok = bool(confirmed.get("sources_ok", True))
            stale_flags = confirmed.get("stale_flags")
            if isinstance(stale_flags, dict) and any(stale_flags.values()):
                sources_ok = False
            confirmed_nav = float((confirmed or {}).get("nav") or confirmed.get("nav_usd") or 0.0) if confirmed else 0.0
            if (
                sources_ok
                and confirmed_nav > 0.0
                and age is not None
                and age <= max(_NAV_CACHE_MAX_AGE_SECONDS, _NAV_FRESHNESS_SECONDS)
            ):
                self._nav = confirmed_nav
            else:
                self._nav = 0.0
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
    "compute_aum_breakdown",
    "compute_gross_exposure_usd",
    "compute_nav_summary",
    "get_confirmed_nav",
    "nav_health_snapshot",
    "is_nav_fresh",
    "PortfolioSnapshot",
    "write_nav",
    "run_nav_writer",
    "start_nav_writer",
]
