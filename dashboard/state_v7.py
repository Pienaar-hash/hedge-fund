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
POSITIONS_LEDGER_PATH = Path(os.getenv("POSITIONS_LEDGER_PATH") or (STATE_DIR / "positions_ledger.json"))
TP_SL_REGISTRY_PATH = Path(os.getenv("TP_SL_REGISTRY_PATH") or (STATE_DIR / "position_tp_sl.json"))
ROUTER_STATE_PATH = Path(os.getenv("ROUTER_STATE_PATH") or (STATE_DIR / "router.json"))
ROUTER_HEALTH_STATE_PATH = Path(os.getenv("ROUTER_HEALTH_STATE_PATH") or (STATE_DIR / "router_health.json"))
DIAGNOSTICS_STATE_PATH = Path(os.getenv("DIAGNOSTICS_STATE_PATH") or (STATE_DIR / "diagnostics.json"))
RISK_STATE_PATH = Path(os.getenv("RISK_STATE_PATH") or (STATE_DIR / "risk_snapshot.json"))
ENGINE_METADATA_PATH = Path(os.getenv("ENGINE_METADATA_PATH") or (STATE_DIR / "engine_metadata.json"))
STATE_FILE_SPECS: Dict[str, Dict[str, Any]] = {
    "nav_state": {
        "path": str(NAV_STATE_PATH),
        "required_keys": [],
        "any_of_keys": [["series", "total_equity", "nav", "nav_usd"]],
    },
    "positions_state": {
        "path": str(POSITIONS_STATE_PATH),
        "required_keys": [],
        "any_of_keys": [["positions", "rows", "items", "updated_at"]],
    },
    "positions_ledger": {
        "path": str(POSITIONS_LEDGER_PATH),
        "required_keys": [],
        "any_of_keys": [["entries", "updated_at"]],
    },
    "kpis_v7": {
        "path": str(KPI_V7_STATE_PATH),
        "required_keys": [],
        "any_of_keys": [["updated_at", "ts"], ["atr_regime", "dd_state", "drawdown"]],
    },
    "risk_snapshot": {
        "path": str(RISK_STATE_PATH),
        "required_keys": [],
        "any_of_keys": [["updated_ts", "risk_mode", "dd_state"]],
    },
    "runtime_diagnostics": {
        "path": str(DIAGNOSTICS_STATE_PATH),
        "required_keys": [],
        "any_of_keys": [["runtime_diagnostics"]],
    },
    "router_health": {
        "path": str(ROUTER_HEALTH_STATE_PATH),
        "required_keys": [],
        "any_of_keys": [["updated_ts"], ["global", "summary", "symbols", "per_symbol", "router_health"]],
    },
    "engine_metadata": {
        "path": str(ENGINE_METADATA_PATH),
        "required_keys": [],
        "any_of_keys": [["engine_version", "updated_ts"]],
    },
}
FUNDING_SNAPSHOT_PATH = Path(os.getenv("FUNDING_SNAPSHOT_PATH") or (STATE_DIR / "funding_snapshot.json"))
BASIS_SNAPSHOT_PATH = Path(os.getenv("BASIS_SNAPSHOT_PATH") or (STATE_DIR / "basis_snapshot.json"))
HYBRID_SCORES_PATH = Path(os.getenv("HYBRID_SCORES_PATH") or (STATE_DIR / "hybrid_scores.json"))
VOL_REGIMES_PATH = Path(os.getenv("VOL_REGIMES_PATH") or (STATE_DIR / "vol_regimes.json"))
ROUTER_QUALITY_PATH = Path(os.getenv("ROUTER_QUALITY_PATH") or (STATE_DIR / "router_quality.json"))
RV_MOMENTUM_PATH = Path(os.getenv("RV_MOMENTUM_PATH") or (STATE_DIR / "rv_momentum.json"))
FACTOR_DIAGNOSTICS_PATH = Path(os.getenv("FACTOR_DIAGNOSTICS_PATH") or (STATE_DIR / "factor_diagnostics.json"))
FACTOR_PNL_PATH = Path(os.getenv("FACTOR_PNL_PATH") or (STATE_DIR / "factor_pnl.json"))
EDGE_INSIGHTS_PATH = Path(os.getenv("EDGE_INSIGHTS_PATH") or (STATE_DIR / "edge_insights.json"))

# Offchain/Treasury state paths (v7.6)
OFFCHAIN_ASSETS_PATH = Path(os.getenv("OFFCHAIN_ASSETS_PATH") or (STATE_DIR / "offchain_assets.json"))
OFFCHAIN_YIELD_PATH = Path(os.getenv("OFFCHAIN_YIELD_PATH") or (STATE_DIR / "offchain_yield.json"))

# Config paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFEXCHANGE_HOLDINGS_PATH = Path(os.getenv("OFFEXCHANGE_HOLDINGS_PATH") or (PROJECT_ROOT / "config" / "offexchange_holdings.json"))
COINTRACKER_CSV_PATH = Path(os.getenv("COINTRACKER_CSV_PATH") or (PROJECT_ROOT / "treasury" / "cointracker" / "cointracker_holdings.csv"))
ORDERS_EXECUTED_PATH = Path(os.getenv("ORDERS_EXECUTED_PATH") or (PROJECT_ROOT / "logs" / "execution" / "orders_executed.jsonl"))


def _load_state_json(path: Path, default: Any | None = None) -> Any:
    default_obj = {} if default is None else copy.deepcopy(default)
    try:
        if path.exists() and path.stat().st_size > 0:
            return json.loads(path.read_text())
    except Exception:
        pass
    return default_obj


def load_risk_snapshot(default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load risk_snapshot.json with safe fallback."""
    return _load_state_json(RISK_STATE_PATH, default or {}) or (default or {})


def load_engine_metadata(default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load engine_metadata.json if present."""
    return _load_state_json(ENGINE_METADATA_PATH, default or {}) or (default or {})


def _safe_load_json(path: Path, default: Any) -> Any:
    try:
        if path.exists() and path.stat().st_size > 0:
            return json.loads(path.read_text())
    except Exception:
        pass
    return default


def _load_strategy_config() -> Dict[str, Any]:
    try:
        cfg_path = PROJECT_ROOT / "config" / "strategy_config.json"
        if cfg_path.exists():
            return json.loads(cfg_path.read_text())
    except Exception:
        return {}
    return {}


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


def _get_live_prices_for_aum(nav_state: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    Get live prices for AUM calculation from multiple sources.
    
    Priority:
    1. NAV state conversions (from Binance via executor)
    2. Coingecko cache
    3. Hardcoded stablecoin prices
    
    Returns:
        Dict mapping symbol (uppercase) to USD price
    """
    prices: Dict[str, float] = {}
    
    # Stablecoins
    for stable in ("USDT", "USDC", "BUSD", "DAI", "TUSD", "FDUSD"):
        prices[stable] = 1.0
    
    # Try to get prices from NAV state conversions (Binance prices)
    if nav_state and isinstance(nav_state, dict):
        conversions = nav_state.get("conversions", {})
        if isinstance(conversions, dict):
            for symbol, conv_data in conversions.items():
                if isinstance(conv_data, dict):
                    price = _safe_float(conv_data.get("price"))
                    if price and price > 0:
                        prices[symbol.upper()] = price
    
    # Try coingecko cache for additional assets (XAUT, etc.)
    try:
        coingecko_cache_path = Path("logs/cache/coingecko_cache.json")
        if coingecko_cache_path.exists():
            cg_data = json.loads(coingecko_cache_path.read_text())
            cg_prices = cg_data.get("prices", {})
            for symbol, price in cg_prices.items():
                if symbol.upper() not in prices:
                    p = _safe_float(price)
                    if p and p > 0:
                        prices[symbol.upper()] = p
    except Exception:
        pass
    
    return prices


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


def load_router_quality(default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load router quality state from file (v7.5_B2).
    
    Returns:
        Dict with:
        - enabled: bool
        - summary: aggregate metrics
        - symbols: per-symbol router quality snapshots
    """
    try:
        return _load_state_json(ROUTER_QUALITY_PATH, default or {}) or (default or {})
    except Exception:
        return default or {}


def get_symbol_router_quality_score(
    symbol: str,
    router_quality_state: Optional[Dict[str, Any]] = None,
    default: float = 0.8,
) -> float:
    """
    Get router quality score for a symbol (v7.5_B2).
    
    Args:
        symbol: Trading pair symbol
        router_quality_state: Pre-loaded state or None to load
        default: Default score if not found
        
    Returns:
        Router quality score in [0, 1]
    """
    if router_quality_state is None:
        router_quality_state = load_router_quality()
    
    symbols = router_quality_state.get("symbols", {})
    symbol_data = symbols.get(symbol.upper(), {})
    
    if isinstance(symbol_data, dict):
        return float(symbol_data.get("score", default))
    
    return default


def _normalize_router_health_per_symbol(per_symbol: Any) -> Dict[str, Dict[str, Any]]:
    """Normalize per-symbol router health into a dict keyed by symbol."""
    if isinstance(per_symbol, dict):
        return {str(k).upper(): v for k, v in per_symbol.items() if isinstance(v, dict)}
    if isinstance(per_symbol, list):
        result: Dict[str, Dict[str, Any]] = {}
        for entry in per_symbol:
            if not isinstance(entry, dict):
                continue
            sym = entry.get("symbol")
            if sym:
                result[str(sym).upper()] = entry
        return result
    return {}


def load_router_health_state(
    default: Optional[Dict[str, Any]] = None,
    snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load router_health.json with tolerance for legacy layouts."""
    base_default = default or {}
    try:
        if snapshot is not None:
            raw = snapshot
        else:
            raw = _safe_load_json(ROUTER_HEALTH_STATE_PATH, base_default) or base_default
    except Exception:
        return base_default
    if not isinstance(raw, dict):
        return base_default
    block = raw.get("router_health") if isinstance(raw.get("router_health"), dict) else raw
    if not isinstance(block, dict):
        return base_default
    global_block = block.get("global") if isinstance(block.get("global"), dict) else {}
    per_symbol_block = (
        block.get("per_symbol")
        or raw.get("per_symbol")
        or raw.get("symbols")
        or {}
    )
    per_symbol_norm = _normalize_router_health_per_symbol(per_symbol_block)
    updated_ts = raw.get("updated_ts") or global_block.get("updated_ts")
    return {
        "updated_ts": updated_ts,
        "router_health": {
            "global": global_block or {},
            "per_symbol": per_symbol_norm,
        },
    }


def get_router_global_quality(router_health_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Extract global router quality metrics with safe defaults."""
    state = router_health_state or load_router_health_state({})
    rh_block = state.get("router_health") if isinstance(state, dict) else {}
    global_block = rh_block.get("global") if isinstance(rh_block, dict) else {}
    return {
        "quality_score": _safe_float(global_block.get("quality_score")),
        "slippage_drift_bucket": (global_block.get("slippage_drift_bucket") or "").upper() if global_block else "",
        "latency_bucket": (global_block.get("latency_bucket") or "").upper() if global_block else "",
        "router_bucket": (global_block.get("router_bucket") or "").upper() if global_block else "",
    }


def get_router_symbol_quality(
    symbol: str,
    router_health_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Get per-symbol router quality stats from router_health_state."""
    state = router_health_state or load_router_health_state({})
    rh_block = state.get("router_health") if isinstance(state, dict) else {}
    per_symbol = rh_block.get("per_symbol") if isinstance(rh_block, dict) else {}
    per_symbol_map = _normalize_router_health_per_symbol(per_symbol)
    entry = per_symbol_map.get(symbol.upper()) if symbol else None
    return {
        "quality_score": _safe_float(entry.get("quality_score")) if isinstance(entry, dict) else None,
        "avg_slippage_bps": _safe_float(entry.get("avg_slippage_bps")) if isinstance(entry, dict) else None,
        "avg_latency_ms": _safe_float(entry.get("avg_latency_ms")) if isinstance(entry, dict) else None,
        "slippage_drift_bucket": (entry.get("slippage_drift_bucket") or "").upper() if isinstance(entry, dict) else "",
        "latency_bucket": (entry.get("latency_bucket") or "").upper() if isinstance(entry, dict) else "",
        "twap_usage_ratio": _safe_float(entry.get("twap_usage_ratio")) if isinstance(entry, dict) else None,
    }


def load_rv_momentum(default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load RV momentum state from file (v7.5_C1).
    
    Returns:
        Dict with:
        - updated_ts: timestamp
        - per_symbol: per-symbol RV scores and basket membership
        - spreads: basket spreads (btc_vs_eth, l1_vs_alt, meme_vs_rest)
    """
    try:
        return _load_state_json(RV_MOMENTUM_PATH, default or {}) or (default or {})
    except Exception:
        return default or {}


def get_symbol_rv_momentum_score(
    symbol: str,
    rv_momentum_state: Optional[Dict[str, Any]] = None,
    default: float = 0.0,
) -> float:
    """
    Get RV momentum score for a symbol (v7.5_C1).
    
    Args:
        symbol: Trading pair symbol
        rv_momentum_state: Pre-loaded state or None to load
        default: Default score if not found
        
    Returns:
        RV momentum score in [-1, 1]
    """
    if rv_momentum_state is None:
        rv_momentum_state = load_rv_momentum()
    
    per_symbol = rv_momentum_state.get("per_symbol", {})
    symbol_data = per_symbol.get(symbol.upper(), {})
    
    if isinstance(symbol_data, dict):
        return float(symbol_data.get("score", default))
    
    return default


def get_rv_momentum_spreads(
    rv_momentum_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Get RV momentum basket spreads (v7.5_C1).
    
    Args:
        rv_momentum_state: Pre-loaded state or None to load
        
    Returns:
        Dict with spread values for btc_vs_eth, l1_vs_alt, meme_vs_rest
    """
    if rv_momentum_state is None:
        rv_momentum_state = load_rv_momentum()
    
    spreads = rv_momentum_state.get("spreads", {})
    return {
        "btc_vs_eth": float(spreads.get("btc_vs_eth", 0.0)),
        "l1_vs_alt": float(spreads.get("l1_vs_alt", 0.0)),
        "meme_vs_rest": float(spreads.get("meme_vs_rest", 0.0)),
    }


# ---------------------------------------------------------------------------
# v7.5_C2: Factor Diagnostics & PnL Attribution Loaders
# ---------------------------------------------------------------------------


def load_factor_diagnostics_state(default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load factor diagnostics state from file (v7.5_C2).
    
    Returns:
        Dict with:
        - updated_ts: timestamp
        - per_symbol: per-symbol normalized factor vectors
        - covariance: factor covariance and correlation matrices
        - config: factor diagnostics configuration
    """
    try:
        return _load_state_json(FACTOR_DIAGNOSTICS_PATH, default or {}) or (default or {})
    except Exception:
        return default or {}


def load_factor_pnl_state(default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load factor PnL attribution state from file (v7.5_C2).
    
    Returns:
        Dict with:
        - by_factor: PnL attributed to each factor
        - pct_by_factor: percentage contribution per factor
        - total_pnl_usd: total PnL over window
        - window_days: lookback window
        - trade_count: number of trades included
    """
    try:
        return _load_state_json(FACTOR_PNL_PATH, default or {}) or (default or {})
    except Exception:
        return default or {}


def load_edge_insights_state(default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load edge insights from EdgeScanner state file (v7.7_P5).
    
    Returns:
        Dict with:
        - updated_ts: timestamp
        - vol_regime: current volatility regime
        - dd_state: drawdown state from NAV
        - risk_mode: current risk mode (normal/cautious/defensive)
        - router_quality: router execution quality bucket
        - top_factors: list of highest-scoring factors
        - weak_factors: list of lowest-scoring factors
        - top_symbols: list of highest-scoring symbols
        - weak_symbols: list of lowest-scoring symbols
        - top_categories: list of highest-scoring categories
        - weak_categories: list of lowest-scoring categories
        - edge_by_factor: dict mapping factor -> edge magnitude
    """
    try:
        return _load_state_json(EDGE_INSIGHTS_PATH, default or {}) or (default or {})
    except Exception:
        return default or {}


def get_factor_correlation_matrix(
    factor_diagnostics_state: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], List[List[float]]]:
    """
    Get factor correlation matrix from diagnostics state (v7.5_C2).
    
    Args:
        factor_diagnostics_state: Pre-loaded state or None to load
        
    Returns:
        Tuple of (factor_names, correlation_matrix as 2D list)
    """
    if factor_diagnostics_state is None:
        factor_diagnostics_state = load_factor_diagnostics_state()
    
    covariance = factor_diagnostics_state.get("covariance", {})
    factors = covariance.get("factors", [])
    corr_matrix = covariance.get("correlation_matrix", [])
    
    return factors, corr_matrix


def get_factor_volatilities(
    factor_diagnostics_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Get per-factor volatilities from diagnostics state (v7.5_C2).
    
    Args:
        factor_diagnostics_state: Pre-loaded state or None to load
        
    Returns:
        Dict mapping factor name to volatility
    """
    if factor_diagnostics_state is None:
        factor_diagnostics_state = load_factor_diagnostics_state()
    
    covariance = factor_diagnostics_state.get("covariance", {})
    return covariance.get("factor_vols", {})


def get_symbol_factor_fingerprint(
    symbol: str,
    direction: str = "LONG",
    factor_diagnostics_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Get normalized factor fingerprint for a symbol (v7.5_C2).
    
    Args:
        symbol: Trading pair symbol
        direction: LONG or SHORT
        factor_diagnostics_state: Pre-loaded state or None to load
        
    Returns:
        Dict mapping factor name to normalized value
    """
    if factor_diagnostics_state is None:
        factor_diagnostics_state = load_factor_diagnostics_state()
    
    per_symbol = factor_diagnostics_state.get("per_symbol", {})
    key = f"{symbol.upper()}:{direction.upper()}"
    symbol_data = per_symbol.get(key, {})
    
    if isinstance(symbol_data, dict):
        return symbol_data.get("factors", {})
    
    return {}


def get_factor_pnl_summary(
    factor_pnl_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Get factor PnL attribution summary (v7.5_C2).
    
    Args:
        factor_pnl_state: Pre-loaded state or None to load
        
    Returns:
        Dict with by_factor, pct_by_factor, total_pnl_usd, trade_count
    """
    if factor_pnl_state is None:
        factor_pnl_state = load_factor_pnl_state()
    
    return {
        "by_factor": factor_pnl_state.get("by_factor", {}),
        "pct_by_factor": factor_pnl_state.get("pct_by_factor", {}),
        "total_pnl_usd": float(factor_pnl_state.get("total_pnl_usd", 0.0)),
        "trade_count": int(factor_pnl_state.get("trade_count", 0)),
        "window_days": int(factor_pnl_state.get("window_days", 14)),
    }


# ---------------------------------------------------------------------------
# v7.5_C3: Factor Weights & Orthogonalization Loaders
# ---------------------------------------------------------------------------


def get_factor_weights(
    factor_diagnostics_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Get auto-computed factor weights from diagnostics state (v7.6).
    
    Args:
        factor_diagnostics_state: Pre-loaded state or None to load
        
    Returns:
        Dict mapping factor name to weight
    """
    if factor_diagnostics_state is None:
        factor_diagnostics_state = load_factor_diagnostics_state()
    
    # Prefer flat weights surface, fall back to legacy block
    weights = factor_diagnostics_state.get("weights", {})
    if not isinstance(weights, dict) or not weights:
        factor_weights_block = factor_diagnostics_state.get("factor_weights", {}) or {}
        weights = factor_weights_block.get("weights", {}) if isinstance(factor_weights_block, dict) else {}
    return weights if isinstance(weights, dict) else {}


def get_factor_ir(
    factor_diagnostics_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Get per-factor information ratios from diagnostics state (v7.6).
    
    Args:
        factor_diagnostics_state: Pre-loaded state or None to load
        
    Returns:
        Dict mapping factor name to IR value
    """
    if factor_diagnostics_state is None:
        factor_diagnostics_state = load_factor_diagnostics_state()
    
    ir = factor_diagnostics_state.get("factor_ir", {})
    if not isinstance(ir, dict) or not ir:
        factor_weights = factor_diagnostics_state.get("factor_weights", {}) or {}
        if isinstance(factor_weights, dict):
            ir = factor_weights.get("factor_ir", {})
    return ir if isinstance(ir, dict) else {}


def get_orthogonalization_status(
    factor_diagnostics_state: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, bool]:
    """
    Get orthogonalization and auto-weighting status (v7.5_C3).
    
    Args:
        factor_diagnostics_state: Pre-loaded state or None to load
        
    Returns:
        Tuple of (orthogonalization_enabled, auto_weighting_enabled)
    """
    if factor_diagnostics_state is None:
        factor_diagnostics_state = load_factor_diagnostics_state()
    
    ortho_enabled = bool(factor_diagnostics_state.get("orthogonalization_enabled", False))
    auto_weight_enabled = bool(factor_diagnostics_state.get("auto_weighting_enabled", False))
    
    return ortho_enabled, auto_weight_enabled


def get_orthogonalized_factors(
    symbol: str,
    factor_diagnostics_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Get orthogonalized factor values for a symbol (v7.5_C3).
    
    Args:
        symbol: Trading pair symbol
        factor_diagnostics_state: Pre-loaded state or None to load
        
    Returns:
        Dict mapping factor name to orthogonalized value
    """
    if factor_diagnostics_state is None:
        factor_diagnostics_state = load_factor_diagnostics_state()
    
    orthogonalized = factor_diagnostics_state.get("orthogonalized", {})
    if isinstance(orthogonalized, dict):
        per_symbol = orthogonalized.get("per_symbol", {})
        return per_symbol.get(symbol.upper(), {})
    return {}


def get_covariance_matrix(
    factor_diagnostics_state: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], List[List[float]]]:
    """
    Get covariance matrix from diagnostics state (v7.6 surface).
    """
    if factor_diagnostics_state is None:
        factor_diagnostics_state = load_factor_diagnostics_state()
    covariance = factor_diagnostics_state.get("covariance", {}) or {}
    factors = covariance.get("factors", []) if isinstance(covariance, dict) else []
    cov = covariance.get("covariance_matrix", []) if isinstance(covariance, dict) else []
    return factors, cov


def get_orthogonalization_summary(
    factor_diagnostics_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Retrieve norms, degenerate flags, and dot products for orthogonalized factors.
    """
    if factor_diagnostics_state is None:
        factor_diagnostics_state = load_factor_diagnostics_state()
    ortho = factor_diagnostics_state.get("orthogonalized", {}) or {}
    if not isinstance(ortho, dict):
        return {}
    return {
        "norms": ortho.get("norms", {}),
        "degenerate": ortho.get("degenerate", []),
        "dot_products": ortho.get("dot_products", {}),
    }


def _symbol_meta(symbol: str, kpis_symbols: Dict[str, Any]) -> Dict[str, Any]:
    if not symbol:
        return {}
    sym_upper = symbol.upper()
    return kpis_symbols.get(sym_upper, {}) if isinstance(kpis_symbols, dict) else {}


def _normalize_positions(raw_positions: Any, kpis: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[float]]:
    positions_raw = []
    age_s: Optional[float] = None
    if isinstance(raw_positions, dict):
        if isinstance(raw_positions.get("positions"), list):
            positions_raw = raw_positions.get("positions") or []
        elif isinstance(raw_positions.get("items"), list):
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
            "entryPrice": float(entry_price) if entry_price is not None else 0.0,
            "markPrice": float(mark_price) if mark_price is not None else 0.0,
            "notional": float(notional) if notional is not None else 0.0,
            "pnl": float(pnl_val) if pnl_val is not None else 0.0,
            "unrealized": float(pnl_val) if pnl_val is not None else 0.0,
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


def load_funding_snapshot() -> Dict[str, Any]:
    """Load funding rate snapshot for carry scoring."""
    return _safe_load_json(FUNDING_SNAPSHOT_PATH, {})


# ---------------------------------------------------------------------------
# v7.6: Offchain Treasury & AUM Loaders
# ---------------------------------------------------------------------------


def load_offchain_assets(default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load offchain_assets.json with safe fallback (v7.6).
    
    Returns:
        Dict with:
        - updated_ts: timestamp
        - source: data source (cointracker, manual, etc.)
        - assets: per-asset holdings with qty, avg_cost, current_price, usd_value
        - totals: aggregated values
        - metadata: import info
    """
    try:
        return _load_state_json(OFFCHAIN_ASSETS_PATH, default or {}) or (default or {})
    except Exception:
        return default or {}


def load_offchain_yield(default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load offchain_yield.json with safe fallback (v7.6).
    
    Returns:
        Dict with:
        - updated_ts: timestamp
        - source: data source
        - yields: per-asset yield info with apr_pct, strategy, platform
        - totals: aggregated yield values
    """
    try:
        return _load_state_json(OFFCHAIN_YIELD_PATH, default or {}) or (default or {})
    except Exception:
        return default or {}


def compute_unified_aum(
    nav_state: Optional[Dict[str, Any]] = None,
    offchain_assets: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute unified AUM from trading NAV + offchain assets (v7.6).
    
    AUM = Futures NAV + Spot Treasury + External Account Assets
    
    Args:
        nav_state: Pre-loaded nav state or None to load
        offchain_assets: Pre-loaded offchain assets or None to load
        
    Returns:
        Dict with:
        - total_aum_usd: Total AUM
        - futures_nav_usd: Trading NAV from executor
        - offchain_usd: External account asset value
        - slices: breakdown by category
        - stale_flags: staleness indicators
    """
    if nav_state is None:
        nav_state = load_nav_state({})
    if offchain_assets is None:
        offchain_assets = load_offchain_assets({})
    
    # Futures NAV from trading engine
    futures_nav = _safe_float(
        nav_state.get("total_equity") or
        nav_state.get("nav") or
        nav_state.get("nav_usd") or
        0.0
    ) or 0.0
    
    # Load live prices for external account assets
    live_prices = _get_live_prices_for_aum(nav_state)
    
    # External account assets - compute with live prices
    offchain_usd = 0.0
    assets = offchain_assets.get("assets", {}) if isinstance(offchain_assets, dict) else {}
    for symbol, asset_data in assets.items():
        if isinstance(asset_data, dict):
            qty = _safe_float(asset_data.get("qty")) or 0.0
            if qty > 0:
                # Use live price if available, else stored price, else avg_cost
                price = live_prices.get(symbol.upper())
                if not price:
                    price = _safe_float(asset_data.get("current_price_usd"))
                if not price:
                    price = _safe_float(asset_data.get("avg_cost_usd"))
                if price and price > 0:
                    offchain_usd += qty * price
    
    # Spot treasury (from nav_state aum block if present)
    nav_aum = nav_state.get("aum", {}) if isinstance(nav_state, dict) else {}
    spot_treasury_usd = 0.0
    if isinstance(nav_aum, dict):
        offex = nav_aum.get("offexchange", {})
        if isinstance(offex, dict):
            for asset_data in offex.values():
                if isinstance(asset_data, dict):
                    spot_treasury_usd += _safe_float(asset_data.get("usd_value")) or 0.0
    
    total_aum = futures_nav + offchain_usd + spot_treasury_usd
    
    # Build slices
    slices = {
        "futures_nav": {
            "value_usd": futures_nav,
            "pct": (futures_nav / total_aum * 100) if total_aum > 0 else 0.0,
            "label": "Futures NAV",
        },
        "offchain": {
            "value_usd": offchain_usd,
            "pct": (offchain_usd / total_aum * 100) if total_aum > 0 else 0.0,
            "label": "External Accounts",
        },
        "spot_treasury": {
            "value_usd": spot_treasury_usd,
            "pct": (spot_treasury_usd / total_aum * 100) if total_aum > 0 else 0.0,
            "label": "Spot Treasury",
        },
    }
    
    # Staleness checks
    nav_age = _snapshot_age(nav_state)
    offchain_age = _snapshot_age(offchain_assets)
    stale_flags = {
        "nav_stale": nav_age is not None and nav_age > 300,
        "offchain_stale": offchain_age is not None and offchain_age > 86400,  # 24h
    }
    
    return {
        "total_aum_usd": total_aum,
        "futures_nav_usd": futures_nav,
        "offchain_usd": offchain_usd,
        "spot_treasury_usd": spot_treasury_usd,
        "slices": slices,
        "stale_flags": stale_flags,
        "nav_age_s": nav_age,
        "offchain_age_s": offchain_age,
    }


def compute_treasury_yield_summary(
    offchain_assets: Optional[Dict[str, Any]] = None,
    offchain_yield: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute treasury yield summary from assets and yield rates (v7.6).
    
    Args:
        offchain_assets: Pre-loaded offchain assets or None to load
        offchain_yield: Pre-loaded offchain yield or None to load
        
    Returns:
        Dict with:
        - per_asset: yield info per asset
        - daily_yield_usd: total daily yield
        - monthly_yield_usd: total monthly yield
        - annual_yield_usd: total annual yield
        - weighted_avg_apr_pct: weighted average APR
    """
    if offchain_assets is None:
        offchain_assets = load_offchain_assets({})
    if offchain_yield is None:
        offchain_yield = load_offchain_yield({})
    
    assets = offchain_assets.get("assets", {}) if isinstance(offchain_assets, dict) else {}
    yields = offchain_yield.get("yields", {}) if isinstance(offchain_yield, dict) else {}
    
    per_asset = {}
    total_value = 0.0
    total_annual_yield = 0.0
    
    for asset_name, asset_data in assets.items():
        if not isinstance(asset_data, dict):
            continue
        usd_value = _safe_float(asset_data.get("usd_value")) or 0.0
        yield_data = yields.get(asset_name, {}) if isinstance(yields, dict) else {}
        apr_pct = _safe_float(yield_data.get("apr_pct")) or 0.0
        
        annual_yield = usd_value * (apr_pct / 100.0)
        daily_yield = annual_yield / 365.0
        monthly_yield = annual_yield / 12.0
        
        per_asset[asset_name] = {
            "usd_value": usd_value,
            "apr_pct": apr_pct,
            "annual_yield_usd": annual_yield,
            "monthly_yield_usd": monthly_yield,
            "daily_yield_usd": daily_yield,
            "strategy": yield_data.get("strategy", "unknown"),
        }
        
        total_value += usd_value
        total_annual_yield += annual_yield
    
    weighted_avg_apr = (total_annual_yield / total_value * 100) if total_value > 0 else 0.0
    
    return {
        "per_asset": per_asset,
        "daily_yield_usd": total_annual_yield / 365.0,
        "monthly_yield_usd": total_annual_yield / 12.0,
        "annual_yield_usd": total_annual_yield,
        "weighted_avg_apr_pct": weighted_avg_apr,
        "total_treasury_value_usd": total_value,
    }


def get_treasury_health(
    offchain_assets: Optional[Dict[str, Any]] = None,
    offchain_yield: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Get treasury state health diagnostics (v7.6).
    
    Returns:
        Dict with:
        - assets_surface_age_s: age of assets surface
        - yield_surface_age_s: age of yield surface
        - missing_fields: list of missing required fields
        - warnings: list of warning messages
        - healthy: overall health status
    """
    if offchain_assets is None:
        offchain_assets = load_offchain_assets({})
    if offchain_yield is None:
        offchain_yield = load_offchain_yield({})
    
    assets_age = _snapshot_age(offchain_assets)
    yield_age = _snapshot_age(offchain_yield)
    
    warnings = []
    missing_fields = []
    
    # Check assets surface
    if not offchain_assets or not offchain_assets.get("assets"):
        missing_fields.append("offchain_assets.assets")
    if assets_age is not None and assets_age > 86400:
        warnings.append(f"Assets surface stale ({assets_age/3600:.1f}h)")
    
    # Check yield surface
    if not offchain_yield or not offchain_yield.get("yields"):
        missing_fields.append("offchain_yield.yields")
    if yield_age is not None and yield_age > 86400:
        warnings.append(f"Yield surface stale ({yield_age/3600:.1f}h)")
    
    # Check for zero values
    assets = offchain_assets.get("assets", {}) if isinstance(offchain_assets, dict) else {}
    for asset_name, asset_data in assets.items():
        if isinstance(asset_data, dict):
            if asset_data.get("usd_value") is None:
                warnings.append(f"{asset_name} missing usd_value")
    
    healthy = len(missing_fields) == 0 and len(warnings) == 0
    
    return {
        "assets_surface_age_s": assets_age,
        "yield_surface_age_s": yield_age,
        "missing_fields": missing_fields,
        "warnings": warnings,
        "healthy": healthy,
    }


def load_basis_snapshot() -> Dict[str, Any]:
    """Load basis (spot-perp spread) snapshot for carry scoring."""
    return _safe_load_json(BASIS_SNAPSHOT_PATH, {})


def load_hybrid_scores() -> Dict[str, Any]:
    """Load hybrid score rankings for intel display."""
    return _safe_load_json(HYBRID_SCORES_PATH, {})


def load_vol_regimes() -> Dict[str, Any]:
    """Load volatility regime snapshot for dashboard display (v7.4 B2)."""
    return _safe_load_json(VOL_REGIMES_PATH, {})


def load_positions_ledger() -> Dict[str, Any]:
    """
    Load positions_ledger from the state file or positions.json (v7.4_C3).
    
    The ledger contains unified position + TP/SL data.
    Falls back to extracting from positions.json if dedicated file doesn't exist.
    """
    # Try dedicated ledger file first
    ledger_data = _safe_load_json(POSITIONS_LEDGER_PATH, {})
    if ledger_data:
        return ledger_data
    
    # Fallback: check if positions_ledger is embedded in positions.json
    positions_payload = _load_positions_payload()
    if "positions_ledger" in positions_payload:
        return {"entries": positions_payload["positions_ledger"], "updated_ts": positions_payload.get("updated")}
    
    return {}


def load_positions_state() -> List[Dict[str, Any]]:
    """
    Load positions from state file (v7.4_C3).
    
    Returns list of position dicts.
    """
    payload = _load_positions_payload()
    rows = payload.get("rows") or payload.get("positions") or []
    if isinstance(rows, list):
        return rows
    return []


def load_runtime_diagnostics_state() -> Dict[str, Any]:
    """
    Load runtime diagnostics (veto counters, exit pipeline) from diagnostics.json.
    """
    try:
        data = _safe_load_json(DIAGNOSTICS_STATE_PATH, {})
        if isinstance(data, dict):
            return data.get("runtime_diagnostics", {}) or {}
    except Exception:
        return {}
    return {}


def get_exit_coverage_metrics(runtime_diag: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract exit pipeline coverage metrics with safe defaults.
    """
    diag = runtime_diag or load_runtime_diagnostics_state()
    exit_block = diag.get("exit_pipeline", {}) if isinstance(diag, dict) else {}
    return {
        "open_positions_count": int(exit_block.get("open_positions_count", 0) or 0),
        "tp_sl_registered_count": int(exit_block.get("tp_sl_registered_count", 0) or 0),
        "tp_sl_missing_count": int(exit_block.get("tp_sl_missing_count", 0) or 0),
        "underwater_without_tp_sl_count": int(exit_block.get("underwater_without_tp_sl_count", 0) or 0),
        "tp_sl_coverage_pct": float(exit_block.get("tp_sl_coverage_pct", 0.0) or 0.0),
        "ledger_registry_mismatch": bool(exit_block.get("ledger_registry_mismatch", False)),
    }


def get_exit_mismatch_breakdown(runtime_diag: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    """
    Get mismatch breakdown counts for ledger/registry reconciliation.
    """
    diag = runtime_diag or load_runtime_diagnostics_state()
    exit_block = diag.get("exit_pipeline", {}) if isinstance(diag, dict) else {}
    breakdown = exit_block.get("mismatch_breakdown", {}) if isinstance(exit_block, dict) else {}
    defaults = {
        "missing_ledger_positions": 0,
        "ghost_ledger_entries": 0,
        "missing_tp_sl_entries": 0,
        "stale_tp_sl_entries": 0,
    }
    if isinstance(breakdown, dict):
        defaults.update({k: int(v) for k, v in breakdown.items() if v is not None})
    return defaults


def iter_state_file_specs():
    """Yield (name, spec) for known state files and required top-level keys."""
    return STATE_FILE_SPECS.items()


def get_ledger_consistency_status() -> Dict[str, Any]:
    """
    Check consistency between positions and ledger (v7.4_C3).
    
    Returns:
        Dict with:
        - status: "ok" | "partial" | "error"
        - num_positions: count of raw positions
        - num_ledger: count of ledger entries
        - num_with_tp_sl: count of entries with TP/SL
        - message: human-readable status
    """
    positions = load_positions_state()
    ledger_data = load_positions_ledger()
    
    # Count non-zero positions
    num_positions = 0
    for pos in positions:
        qty = _safe_float(pos.get("qty") or pos.get("positionAmt") or 0)
        if qty is not None and abs(qty) > 0:
            num_positions += 1
    
    # Count ledger entries
    ledger_entries = ledger_data.get("entries", ledger_data)
    if isinstance(ledger_entries, dict):
        num_ledger = len(ledger_entries)
        num_with_tp_sl = sum(
            1 for e in ledger_entries.values()
            if isinstance(e, dict) and (e.get("tp") is not None or e.get("sl") is not None)
        )
    elif isinstance(ledger_entries, list):
        num_ledger = len(ledger_entries)
        num_with_tp_sl = sum(
            1 for e in ledger_entries
            if isinstance(e, dict) and (e.get("tp") is not None or e.get("sl") is not None)
        )
    else:
        num_ledger = 0
        num_with_tp_sl = 0
    
    # Determine status
    if num_positions == 0:
        status = "ok"
        message = "No open positions"
    elif num_ledger == 0:
        status = "error"
        message = f"⚠️ {num_positions} positions but ledger is empty"
    elif num_with_tp_sl < num_ledger:
        status = "partial"
        message = f"🟡 {num_with_tp_sl}/{num_ledger} positions have TP/SL"
    elif num_positions == num_ledger:
        status = "ok"
        message = f"🟢 {num_ledger} positions, all with TP/SL"
    else:
        status = "partial"
        message = f"🟡 {num_positions} positions, {num_ledger} ledger entries"
    
    return {
        "status": status,
        "num_positions": num_positions,
        "num_ledger": num_ledger,
        "num_with_tp_sl": num_with_tp_sl,
        "message": message,
    }


def get_dd_state(risk_snapshot: Optional[Dict[str, Any]] = None) -> str:
    snap = risk_snapshot or load_risk_snapshot()
    if isinstance(snap, dict):
        dd = snap.get("dd_state")
        if isinstance(dd, dict):
            return str(dd.get("state") or dd.get("dd_state") or "UNKNOWN").upper()
        if dd:
            return str(dd).upper()
    return "UNKNOWN"


def get_nav_anomaly(risk_snapshot: Optional[Dict[str, Any]] = None) -> bool:
    snap = risk_snapshot or load_risk_snapshot()
    if not isinstance(snap, dict):
        return False
    anomalies = snap.get("anomalies", {})
    if not isinstance(anomalies, dict):
        return False
    return bool(anomalies.get("nav_jump"))


def get_var_metrics(risk_snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    snap = risk_snapshot or load_risk_snapshot()
    var_block = snap.get("var", {}) if isinstance(snap, dict) else {}
    return {
        "portfolio_var_nav_pct": var_block.get("portfolio_var_nav_pct"),
        "max_portfolio_var_nav_pct": var_block.get("max_portfolio_var_nav_pct"),
        "within_limit": var_block.get("within_limit", True),
    }


def get_cvar_metrics(risk_snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    snap = risk_snapshot or load_risk_snapshot()
    cvar_block = snap.get("cvar", {}) if isinstance(snap, dict) else {}
    return {
        "max_position_cvar_nav_pct": cvar_block.get("max_position_cvar_nav_pct"),
        "per_symbol": cvar_block.get("per_symbol", {}),
    }


# ---------------------------------------------------------------------------
# Regime badges
# ---------------------------------------------------------------------------


def compute_vol_regime_badge(vol_state: Optional[Dict[str, Any]] = None) -> str:
    """
    Compute volatility regime badge from vol_regimes surface or defaults.
    """
    if not isinstance(vol_state, dict):
        return "VOL_NORMAL"
    bucket = (
        vol_state.get("current_regime")
        or vol_state.get("regime")
        or vol_state.get("vol_regime")
        or vol_state.get("bucket")
        or "normal"
    )
    bucket = str(bucket).upper()
    if bucket in {"LOW", "NORMAL", "HIGH", "CRISIS"}:
        return f"VOL_{bucket}"
    return "VOL_NORMAL"


def compute_dd_regime_badge(risk_snapshot: Optional[Dict[str, Any]] = None) -> str:
    snap = risk_snapshot or {}
    dd_state = snap.get("dd_state") if isinstance(snap, dict) else None
    state_val = ""
    if isinstance(dd_state, dict):
        state_val = dd_state.get("state") or dd_state.get("dd_state") or ""
    elif isinstance(dd_state, str):
        state_val = dd_state
    state_val = str(state_val).upper()
    mapping = {
        "NORMAL": "DD_NORMAL",
        "DRAWDOWN": "DD_DRAWDOWN",
        "RECOVERY": "DD_RECOVERY",
    }
    return mapping.get(state_val, "DD_NORMAL")


def compute_router_quality_badge(
    router_health: Optional[Dict[str, Any]] = None,
    *,
    low_threshold: float = 0.5,
    high_threshold: float = 0.9,
) -> str:
    if not isinstance(router_health, dict):
        return "ROUTER_NEUTRAL"
    rh_block = router_health.get("router_health") if isinstance(router_health, dict) else {}
    global_block = router_health.get("global") or (rh_block.get("global") if isinstance(rh_block, dict) else {})
    score = None
    try:
        score = float(global_block.get("quality_score"))
    except Exception:
        score = None
    if score is None:
        return "ROUTER_NEUTRAL"
    if score >= high_threshold:
        return "ROUTER_STRONG"
    if score <= low_threshold:
        return "ROUTER_WEAK"
    return "ROUTER_NEUTRAL"


def compute_risk_mode_badge(risk_snapshot: Optional[Dict[str, Any]] = None) -> str:
    snap = risk_snapshot or {}
    mode = str(snap.get("risk_mode", "NORMAL")).upper()
    anomalies = snap.get("anomalies", {}) if isinstance(snap, dict) else {}
    breach = False
    if isinstance(anomalies, dict):
        breach = bool(anomalies.get("var_limit_breach") or anomalies.get("cvar_limit_breach"))
    if breach:
        return "RISK_BREACH"
    mapping = {
        "OK": "RISK_NORMAL",
        "NORMAL": "RISK_NORMAL",
        "DEFENSIVE": "RISK_DEFENSIVE",
        "CRISIS": "RISK_CRITICAL",
        "HALTED": "RISK_CRITICAL",
    }
    return mapping.get(mode, "RISK_NORMAL")


def get_regime_badges(
    *,
    vol_state: Optional[Dict[str, Any]] = None,
    risk_snapshot: Optional[Dict[str, Any]] = None,
    router_health: Optional[Dict[str, Any]] = None,
    strategy_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    cfg = strategy_config or _load_strategy_config()
    router_cfg = (cfg.get("router_quality") or {}) if isinstance(cfg, dict) else {}
    low_th = float(router_cfg.get("low_quality_threshold", 0.5) or 0.5)
    high_th = float(router_cfg.get("high_quality_threshold", 0.9) or 0.9)
    return {
        "vol": compute_vol_regime_badge(vol_state),
        "dd": compute_dd_regime_badge(risk_snapshot),
        "router": compute_router_quality_badge(router_health, low_threshold=low_th, high_threshold=high_th),
        "risk": compute_risk_mode_badge(risk_snapshot),
    }


# ---------------------------------------------------------------------------
# Surface Health Validation
# ---------------------------------------------------------------------------


def _surface_updated_ts(payload: Any) -> Optional[float]:
    if not isinstance(payload, dict):
        return None
    for key in ("updated_ts", "updated_at", "ts"):
        val = payload.get(key)
        ts = _to_epoch_seconds(val)
        if ts is not None:
            return ts
    return None


def validate_surface_health(
    *,
    state_dir: Optional[Path] = None,
    allowable_lag_seconds: float = 900.0,
) -> Dict[str, List[str]]:
    base = state_dir or STATE_DIR
    missing: List[str] = []
    stale: List[str] = []
    schema_violations: List[str] = []
    cross_surface: List[str] = []

    now_ts = time.time()
    # Validate required surfaces
    for name, spec in STATE_FILE_SPECS.items():
        path = Path(spec["path"])
        if state_dir:
            path = base / path.name
        elif not path.is_absolute():
            path = base / path
        if not path.exists():
            missing.append(name)
            continue
        try:
            payload = json.loads(path.read_text()) if path.stat().st_size > 0 else {}
        except Exception:
            schema_violations.append(f"{name}:invalid_json")
            continue
        # Required keys
        for req in spec.get("required_keys", []):
            if req not in payload:
                schema_violations.append(f"{name}:missing_{req}")
        # Any-of key groups (at least one key from each group must be present)
        for group in spec.get("any_of_keys", []):
            if not any(k in payload for k in group):
                schema_violations.append(f"{name}:missing_one_of_{','.join(group)}")
        ts_val = _surface_updated_ts(payload)
        if ts_val is None:
            schema_violations.append(f"{name}:missing_updated_ts")
        elif now_ts - ts_val > allowable_lag_seconds:
            stale.append(name)

    # Cross-surface: positions vs ledger
    try:
        # Load from provided state_dir to avoid global paths
        pos_path = base / "positions_state.json"
        ledger_path = base / "positions_ledger.json"
        positions_payload = json.loads(pos_path.read_text()) if pos_path.exists() else {}
        ledger_payload = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
        positions = positions_payload.get("positions", []) if isinstance(positions_payload, dict) else []
        ledger_entries = ledger_payload.get("entries", []) if isinstance(ledger_payload, dict) else []
        keys = {f"{pos.get('symbol','').upper()}:{(pos.get('positionSide') or pos.get('side') or '').upper()}"
                for pos in positions if float(pos.get('qty') or pos.get('positionAmt') or 0) != 0}
        ledger_keys = set()
        for entry in ledger_entries:
            key = entry.get("position_key") or f"{entry.get('symbol','').upper()}:{entry.get('side','').upper()}"
            if key:
                ledger_keys.add(key)
        missing_keys = [k for k in keys if k not in ledger_keys]
        if missing_keys:
            cross_surface.append(f"positions_without_ledger:{len(missing_keys)}")
    except Exception:
        pass

    return {
        "missing_files": missing,
        "stale_files": stale,
        "schema_violations": schema_violations,
        "cross_surface_violations": cross_surface,
    }


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
                # Only use a price source if the value is actually > 0
                current_price = None
                if symbol in conversions and isinstance(conversions[symbol], dict):
                    conv_price = _safe_float(conversions[symbol].get("price"))
                    if conv_price and conv_price > 0:
                        current_price = conv_price
                if current_price is None and symbol in coingecko_prices:
                    cg_price = _safe_float(coingecko_prices.get(symbol))
                    if cg_price and cg_price > 0:
                        current_price = cg_price
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
        engine_meta = load_engine_metadata({})
        if engine_meta:
            meta["engine_version"] = engine_meta.get("engine_version")
            meta["engine_updated_ts"] = engine_meta.get("updated_ts") or engine_meta.get("ts")

        base.update({"nav": nav_block, "aum": aum_block_norm, "kpis": kpis_norm, "positions": positions, "router": router_block, "meta": meta})
    except Exception:
        pass
    return base


__all__ = [
    "load_all_state",
    "iter_state_file_specs",
    "load_router_health_state",
    "get_router_global_quality",
    "get_router_symbol_quality",
]
