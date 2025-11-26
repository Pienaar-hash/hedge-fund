"""Unified v7 dashboard state loader."""
from __future__ import annotations

import copy
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

STATE_DIR = Path(os.getenv("STATE_DIR") or "logs/state")
NAV_STATE_PATH = Path(os.getenv("NAV_STATE_PATH") or (STATE_DIR / "nav_state.json"))
NAV_DETAIL_PATH = Path(os.getenv("NAV_DETAIL_PATH") or (STATE_DIR / "nav.json"))
KPI_V7_STATE_PATH = Path(os.getenv("KPI_V7_STATE_PATH") or (STATE_DIR / "kpis_v7.json"))
POSITIONS_STATE_PATH = Path(os.getenv("POSITIONS_STATE_PATH") or (STATE_DIR / "positions_state.json"))
POSITIONS_PATH = Path(os.getenv("POSITIONS_PATH") or (STATE_DIR / "positions.json"))
ROUTER_STATE_PATH = Path(os.getenv("ROUTER_STATE_PATH") or (STATE_DIR / "router.json"))
ROUTER_HEALTH_STATE_PATH = Path(os.getenv("ROUTER_HEALTH_STATE_PATH") or (STATE_DIR / "router_health.json"))

# Config paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFEXCHANGE_HOLDINGS_PATH = Path(os.getenv("OFFEXCHANGE_HOLDINGS_PATH") or (PROJECT_ROOT / "config" / "offexchange_holdings.json"))
ORDERS_EXECUTED_PATH = Path(os.getenv("ORDERS_EXECUTED_PATH") or (PROJECT_ROOT / "logs" / "execution" / "orders_executed.jsonl"))


def _load_state_json(path: Path, default: Any | None = None) -> Any:
    default_obj = {} if default is None else copy.deepcopy(default)
    try:
        if path.exists() and path.stat().st_size > 0:
            return json.loads(path.read_text())
    except Exception:
        pass
    return default_obj


def _safe_load_json(path: Path, default: Any) -> Any:
    try:
        if path.exists() and path.stat().st_size > 0:
            return json.loads(path.read_text())
    except Exception:
        pass
    return default


def _safe_float(val: Any) -> Optional[float]:
    try:
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            txt = val.replace(",", "").strip()
            if not txt:
                return None
            return float(txt)
        return None
    except Exception:
        return None


def _fmt_usd(val: Any, nd: int = 2) -> str:
    num = _safe_float(val)
    if num is None:
        return "0.00" if nd == 2 else "0"
    return f"{num:,.{nd}f}"


def _to_epoch_seconds(value: Any) -> Optional[float]:
    if value in (None, "", "null"):
        return None
    try:
        if isinstance(value, (int, float)):
            val = float(value)
            if val > 1e12:
                val /= 1000.0
            return val
        if isinstance(value, str):
            txt = value.strip()
            if not txt:
                return None
            if txt.isdigit():
                return _to_epoch_seconds(float(txt))
            if txt.endswith("Z"):
                txt = txt[:-1] + "+00:00"
            try:
                return datetime.fromisoformat(txt).astimezone(timezone.utc).timestamp()
            except Exception:
                return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def _snapshot_age(payload: Dict[str, Any]) -> Optional[float]:
    if not isinstance(payload, dict) or not payload:
        return None
    candidates: List[Any] = []
    for key in ("updated_at", "updated_ts", "ts", "timestamp", "time"):
        if key in payload:
            candidates.append(payload.get(key))
    nested_nav = payload.get("nav")
    if isinstance(nested_nav, dict):
        for key in ("updated_at", "updated_ts", "ts"):
            if key in nested_nav:
                candidates.append(nested_nav.get(key))
    now = time.time()
    for raw in candidates:
        ts_val = _to_epoch_seconds(raw)
        if ts_val is not None:
            return max(0.0, now - float(ts_val))
    return None


def _calculate_alltime_futures_stats() -> Dict[str, Any]:
    """Calculate all-time futures trading stats from order log and expectancy."""
    result = {
        "total_fees": 0.0,
        "total_notional": 0.0,
        "fill_count": 0,
        "trade_count": 0,
        "realized_pnl": 0.0,
        "first_trade": None,
        "last_trade": None,
    }
    # Get fees and volume from order log
    try:
        if ORDERS_EXECUTED_PATH.exists():
            with open(ORDERS_EXECUTED_PATH) as f:
                for line in f:
                    try:
                        order = json.loads(line)
                        if order.get("event_type") != "order_fill":
                            continue
                        result["fill_count"] += 1
                        result["total_fees"] += float(order.get("fee_total") or 0)
                        qty = float(order.get("executedQty") or 0)
                        price = float(order.get("avgPrice") or 0)
                        result["total_notional"] += qty * price
                        ts = order.get("ts")
                        if ts:
                            if result["first_trade"] is None:
                                result["first_trade"] = ts
                            result["last_trade"] = ts
                    except Exception:
                        continue
    except Exception:
        pass
    
    # Get realized PnL from expectancy (completed round-trip trades)
    try:
        expectancy_path = PROJECT_ROOT / "logs" / "state" / "expectancy_v6.json"
        if expectancy_path.exists():
            exp_data = json.loads(expectancy_path.read_text())
            for sym_data in exp_data.get("symbols", {}).values():
                count = int(sym_data.get("count", 0))
                avg_return = float(sym_data.get("avg_return", 0))
                result["trade_count"] += count
                result["realized_pnl"] += count * avg_return
    except Exception:
        pass
    
    return result


def load_nav_state(default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        return _load_state_json(NAV_STATE_PATH, default or {}) or (default or {})
    except Exception:
        return default or {}


def load_nav_detail(default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        return _load_state_json(NAV_DETAIL_PATH, default or {}) or (default or {})
    except Exception:
        return default or {}


def load_kpis_v7(default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        return _load_state_json(KPI_V7_STATE_PATH, default or {}) or (default or {})
    except Exception:
        return default or {}


def _symbol_meta(symbol: str, kpis_symbols: Dict[str, Any]) -> Dict[str, Any]:
    if not symbol:
        return {}
    sym_upper = symbol.upper()
    return kpis_symbols.get(sym_upper, {}) if isinstance(kpis_symbols, dict) else {}


def _normalize_positions(raw_positions: Any, kpis: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[float]]:
    positions_raw = []
    age_s: Optional[float] = None
    if isinstance(raw_positions, dict):
        if isinstance(raw_positions.get("items"), list):
            positions_raw = raw_positions.get("items") or []
        elif isinstance(raw_positions.get("rows"), list):
            positions_raw = raw_positions.get("rows") or []
        else:
            positions_raw = []
        age_s = _snapshot_age(raw_positions)
    elif isinstance(raw_positions, list):
        positions_raw = raw_positions
    kpis_symbols = kpis.get("symbols") if isinstance(kpis.get("symbols"), dict) else {}
    normalized: List[Dict[str, Any]] = []
    for pos in positions_raw or []:
        if not isinstance(pos, dict):
            continue
        symbol = str(
            pos.get("symbol") or pos.get("pair") or pos.get("asset") or pos.get("underlying") or ""
        ).upper()
        side = pos.get("side") or pos.get("positionSide") or pos.get("direction")
        qty = _safe_float(pos.get("qty") or pos.get("positionAmt") or pos.get("size"))
        mark_price = _safe_float(pos.get("mark_price") or pos.get("markPrice") or pos.get("price"))
        entry_price = _safe_float(pos.get("entry_price") or pos.get("entryPrice"))
        notional = _safe_float(pos.get("notional") or pos.get("positionValue"))
        if notional is None and qty is not None:
            price = mark_price or entry_price
            if price is not None:
                notional = qty * price
        pnl_val = _safe_float(pos.get("pnl") or pos.get("unrealized") or pos.get("unrealized_pnl"))
        meta = _symbol_meta(symbol, kpis_symbols)
        row = {
            "symbol": symbol,
            "side": side,
            "qty": float(qty) if qty is not None else 0.0,
            "notional": float(notional) if notional is not None else 0.0,
            "pnl": float(pnl_val) if pnl_val is not None else 0.0,
            "notional_fmt": _fmt_usd(notional),
            "pnl_fmt": _fmt_usd(pnl_val),
            "dd_state": meta.get("dd_state"),
            "dd_today_pct": meta.get("dd_today_pct"),
            "atr_ratio": meta.get("atr_ratio"),
            "atr_regime": meta.get("atr_regime"),
        }
        normalized.append(row)
    return normalized, age_s


def _aggregate(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _normalize_router(router_payload: Dict[str, Any], kpis_router: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[float]]:
    per_symbol = []
    if isinstance(router_payload.get("per_symbol"), list):
        per_symbol = router_payload.get("per_symbol") or []
    elif isinstance(router_payload.get("symbols"), list):
        per_symbol = router_payload.get("symbols") or []
    
    # Prefer quality from KPIs (which has policy_quality), then from router health
    quality = kpis_router.get("quality") or kpis_router.get("policy_quality")
    if quality is None:
        quality_counts = (router_payload.get("summary") or {}).get("quality_counts") if isinstance(router_payload.get("summary"), dict) else {}
        if isinstance(quality_counts, dict):
            if quality_counts.get("broken"):
                quality = "broken"
            elif quality_counts.get("degraded"):
                quality = "degraded"
            elif quality_counts.get("ok"):
                quality = "ok"
    if quality is None:
        quality = router_payload.get("quality") or (router_payload.get("summary") or {}).get("quality")
    
    maker_fill_rate = _aggregate([_safe_float(item.get("maker_fill_rate")) for item in per_symbol])
    fallback_ratio = _aggregate(
        [
            _safe_float(item.get("fallback_ratio") or item.get("fallback_rate"))
            for item in per_symbol
        ]
    )
    slip_q50 = _aggregate(
        [
            _safe_float(item.get("slippage_p50") or item.get("slip_q50") or item.get("slip_q50_bps"))
            for item in per_symbol
        ]
    )
    router_block = {
        "quality": quality,
        "maker_fill_rate": maker_fill_rate if maker_fill_rate is not None else _safe_float(kpis_router.get("maker_fill_rate")),
        "fallback_ratio": fallback_ratio if fallback_ratio is not None else _safe_float(kpis_router.get("fallback_ratio")),
        "slip_q50_bps": slip_q50 if slip_q50 is not None else _safe_float(kpis_router.get("slip_q50_bps")),
    }
    return router_block, _snapshot_age(router_payload)


def _load_positions_payload() -> Dict[str, Any]:
    for path in (POSITIONS_STATE_PATH, POSITIONS_PATH):
        payload = _safe_load_json(path, {})
        if isinstance(payload, dict) and payload:
            return payload
    return {}


def _load_router_payload() -> Dict[str, Any]:
    for path in (ROUTER_STATE_PATH, ROUTER_HEALTH_STATE_PATH):
        payload = _safe_load_json(path, {})
        if isinstance(payload, dict) and payload:
            return payload
    return {}


def load_all_state() -> Dict[str, Any]:
    """
    Load all v7 state surfaces (nav, aum, kpis, router, positions, symbols, meta)
    and return a single normalized dictionary.

    Never raises; returns sane defaults when files are missing.
    """
    base: Dict[str, Any] = {
        "nav": {},
        "aum": {"slices": []},
        "kpis": {},
        "positions": [],
        "router": {},
        "meta": {},
    }
    try:
        nav_state = load_nav_state({})
        nav_detail = load_nav_detail({})
        kpis_raw = load_kpis_v7({})
        positions_payload = _load_positions_payload()
        router_payload = _load_router_payload()

        kpis_norm: Dict[str, Any] = {
            "nav": kpis_raw.get("nav", {}),
            "risk": kpis_raw.get("risk", {}),
            "router": kpis_raw.get("router", kpis_raw.get("router_stats", {})),
            "symbols": kpis_raw.get("symbols", {}),
        }
        nav_age = _safe_float(nav_state.get("age_s"))
        # Always prefer calculating age from updated_at for accuracy
        nav_updated_at = _to_epoch_seconds(nav_state.get("updated_at") or nav_state.get("ts") or nav_state.get("time")) or 0.0
        if nav_updated_at and nav_updated_at > 0:
            nav_age = max(0.0, time.time() - float(nav_updated_at))
        elif nav_age is None or nav_age == 0:
            nav_age = _snapshot_age(nav_state) or 0.0
        nav_usd = _safe_float(nav_state.get("total_equity") or nav_state.get("nav")) or 0.0
        nav_block: Dict[str, Any] = {
            "nav_usd": float(nav_usd),
            "peak_equity": float(_safe_float(nav_state.get("peak_equity")) or 0.0),
            "drawdown_pct": float(_safe_float(nav_state.get("drawdown_pct")) or 0.0),
            "drawdown_abs": float(_safe_float(nav_state.get("drawdown_abs")) or 0.0),
            "gross_exposure": float(_safe_float(nav_state.get("gross_exposure")) or 0.0),
            "net_exposure": float(_safe_float(nav_state.get("net_exposure")) or 0.0),
            "realized_pnl_today": float(_safe_float(nav_state.get("realized_pnl_today")) or 0.0),
            "unrealized_pnl": float(_safe_float(nav_state.get("unrealized_pnl")) or 0.0),
            "age_s": float(nav_age),
            "updated_at": float(nav_updated_at),
            "source": NAV_STATE_PATH.name,
        }

        # AUM = Futures NAV + Off-exchange holdings
        aum_slices: List[Dict[str, Any]] = []
        futures_nav = _safe_float(nav_state.get("total_equity") or nav_state.get("nav")) or 0.0
        
        # Calculate futures PnL (today)
        futures_realized = float(_safe_float(nav_state.get("realized_pnl_today")) or 0.0)
        futures_unrealized = float(_safe_float(nav_state.get("unrealized_pnl")) or 0.0)
        futures_pnl = futures_realized + futures_unrealized
        
        # Get all-time trading stats
        alltime_stats = _calculate_alltime_futures_stats()
        
        aum_slices.append({
            "label": "Futures",
            "usd": float(futures_nav),
            "pnl_usd": float(futures_pnl),
            "pnl_pct": (futures_pnl / futures_nav * 100.0) if futures_nav > 0 else 0.0,
            "alltime_realized_pnl": alltime_stats.get("realized_pnl", 0.0),
            "alltime_fees": alltime_stats.get("total_fees", 0.0),
            "alltime_notional": alltime_stats.get("total_notional", 0.0),
            "alltime_trades": alltime_stats.get("trade_count", 0),
            "alltime_fills": alltime_stats.get("fill_count", 0),
            "first_trade": alltime_stats.get("first_trade"),
            "last_trade": alltime_stats.get("last_trade"),
        })

        # Load off-exchange holdings from config
        offexchange_holdings = _load_state_json(OFFEXCHANGE_HOLDINGS_PATH, {})
        
        # Load coingecko prices for off-exchange assets
        coingecko_cache_path = Path("logs/cache/coingecko_cache.json")
        coingecko_prices = {}
        try:
            if coingecko_cache_path.exists():
                cg_data = json.loads(coingecko_cache_path.read_text())
                coingecko_prices = cg_data.get("prices", {})
        except Exception:
            pass
        
        offexchange_details = []
        if isinstance(offexchange_holdings, dict):
            # Get current market prices from nav_detail conversions (Binance) or coingecko
            conversions = {}
            if isinstance(nav_detail.get("nav_detail"), dict):
                conversions = nav_detail["nav_detail"].get("conversions", {})
            
            for symbol, holding in offexchange_holdings.items():
                if not isinstance(holding, dict):
                    continue
                qty = _safe_float(holding.get("qty"))
                avg_cost = _safe_float(holding.get("avg_cost"))
                if qty is None or qty <= 0:
                    continue
                
                # Get price from conversions (Binance), coingecko, or avg_cost fallback
                current_price = None
                if symbol in conversions and isinstance(conversions[symbol], dict):
                    current_price = _safe_float(conversions[symbol].get("price"))
                if current_price is None and symbol in coingecko_prices:
                    current_price = _safe_float(coingecko_prices.get(symbol))
                if current_price is None:
                    current_price = avg_cost
                
                if current_price is not None and current_price > 0:
                    usd_value = float(qty) * float(current_price)
                    cost_basis = float(qty) * float(avg_cost) if avg_cost else usd_value
                    pnl_usd = usd_value - cost_basis
                    pnl_pct = (pnl_usd / cost_basis * 100.0) if cost_basis > 0 else 0.0
                    
                    aum_slices.append({
                        "label": str(symbol).upper(), 
                        "usd": usd_value,
                        "pnl_usd": pnl_usd,
                        "pnl_pct": pnl_pct
                    })
                    offexchange_details.append({
                        "symbol": str(symbol).upper(),
                        "qty": float(qty),
                        "avg_cost": float(avg_cost) if avg_cost else 0.0,
                        "current_price": float(current_price),
                        "cost_basis": float(cost_basis),
                        "market_value": float(usd_value),
                        "pnl_usd": float(pnl_usd),
                        "pnl_pct": float(pnl_pct),
                    })
                else:
                    # If no price available, skip this asset
                    pass

        usd_zar = (
            _safe_float((nav_detail.get("fx") or {}).get("usd_zar"))
            or _safe_float(nav_state.get("fx_usd_zar") or nav_state.get("usd_zar"))
            or 18.0
        )
        # Try coingecko cache for ZAR rate too
        if coingecko_prices:
            try:
                cg_data = json.loads(coingecko_cache_path.read_text())
                cg_zar = _safe_float(cg_data.get("usd_zar"))
                if cg_zar and cg_zar > 0:
                    usd_zar = float(cg_zar)
            except Exception:
                pass
        
        for slice_row in aum_slices:
            slice_row["zar"] = round(float(slice_row.get("usd") or 0.0) * float(usd_zar), 2)
        
        # Calculate total AUM PnL (futures + off-exchange)
        total_offexchange_pnl = sum(d.get("pnl_usd") or 0.0 for d in offexchange_details)
        futures_pnl = float(_safe_float(nav_state.get("realized_pnl_today")) or 0.0) + float(_safe_float(nav_state.get("unrealized_pnl")) or 0.0)
        total_aum_pnl = futures_pnl + total_offexchange_pnl
        
        aum_block_norm = {
            "slices": aum_slices,
            "total_usd": sum(s.get("usd") or 0.0 for s in aum_slices),
            "total_zar": sum(s.get("zar") or 0.0 for s in aum_slices),
            "offexchange_details": offexchange_details,
            "offexchange_pnl_usd": float(total_offexchange_pnl),
            "futures_pnl_usd": float(futures_pnl),
            "total_pnl_usd": float(total_aum_pnl),
        }

        nav_block["fx_usd_zar"] = float(usd_zar)
        nav_block["nav_zar"] = float(round(nav_block["nav_usd"] * float(usd_zar), 2))

        positions, pos_age = _normalize_positions(positions_payload, kpis_norm)
        router_block, router_age = _normalize_router(router_payload, kpis_norm.get("router") or {})
        kpis_age = _snapshot_age(kpis_raw)
        data_age_candidates = [nav_block.get("age_s"), pos_age, router_age, kpis_age]
        data_age = max([age for age in data_age_candidates if isinstance(age, (int, float))], default=None)
        
        # Check testnet status from both ENV and BINANCE_TESTNET
        env_label = (os.getenv("ENV") or os.getenv("HEDGE_ENV") or "").lower()
        binance_testnet = os.getenv("BINANCE_TESTNET", "0").strip() in ("1", "true", "True", "yes")
        is_testnet = "test" in env_label or binance_testnet
        
        meta = {"data_age_s": data_age, "testnet": is_testnet}

        base.update({"nav": nav_block, "aum": aum_block_norm, "kpis": kpis_norm, "positions": positions, "router": router_block, "meta": meta})
    except Exception:
        pass
    return base


__all__ = ["load_all_state"]
