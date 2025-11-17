#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

try:
    from .exchange_utils import get_symbol_filters  # online listing check
except Exception:  # pragma: no cover
    def get_symbol_filters(_s: str) -> Dict[str, Any]:
        raise RuntimeError("exchange access unavailable")


def _path_tiers() -> str:
    return os.getenv("SYMBOL_TIERS_CONFIG", "config/symbol_tiers.json")


def _path_discovery() -> str:
    return os.getenv("DISCOVERY_PATH", "config/discovery.yml")


def _path_settings() -> str:
    return os.getenv("SETTINGS_PATH", "config/settings.json")


def _path_pairs() -> str:
    return os.getenv("PAIRS_UNIVERSE_PATH", "config/pairs_universe.json")


def _load_json(path: str, default):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default


def _load_settings() -> Dict[str, Any]:
    return _load_json(_path_settings(), {})


def load_tiers() -> Dict[str, List[str]]:
    tiers = _load_json(_path_tiers(), {})
    out: Dict[str, List[str]] = {}
    for k, v in (tiers.items() if isinstance(tiers, dict) else []):
        if not isinstance(v, list):
            continue
        out[str(k)] = [str(s).upper() for s in v]
    return out


def symbol_tier(symbol: str) -> Optional[str]:
    s = str(symbol).upper()
    entry = universe_by_symbol().get(s)
    if entry:
        tier = entry.get("tier")
        if tier:
            return str(tier)
    tiers = load_tiers()
    for t, arr in tiers.items():
        if s in arr:
            return t
    return None


def is_listed_on_futures(symbol: str) -> bool:
    try:
        f = get_symbol_filters(str(symbol).upper())
        return bool(f)
    except Exception:
        return False


def _load_discovery() -> List[Dict[str, Any]]:
    # Try YAML first; fall back to a minimal ad-hoc parser to avoid dependency
    try:
        import yaml

        with open(_path_discovery(), "r") as f:
            data = yaml.safe_load(f) or []
        return data if isinstance(data, list) else []
    except Exception:
        pass
    # Minimal parser: expect lines like '- key: value' grouped per item
    rows: List[Dict[str, Any]] = []
    cur: Dict[str, Any] = {}
    try:
        with open(_path_discovery(), "r") as f:
            for raw in f:
                ln = raw.strip()
                if not ln:
                    continue
                if ln.startswith("- "):
                    # New item starts
                    if cur:
                        rows.append(cur)
                        cur = {}
                    ln = ln[2:].strip()
                if ":" in ln:
                    k, v = ln.split(":", 1)
                    k = k.strip()
                    v = v.strip()
                    # cast booleans
                    if v.lower() in ("true", "false"):
                        val: Any = v.lower() == "true"
                    else:
                        # strip quotes if any
                        val = v.strip("\"'")
                    cur[k] = val
            if cur:
                rows.append(cur)
    except Exception:
        return []
    return rows


def _raw_pairs_payload() -> Any:
    return _load_json(_path_pairs(), {})


def _iter_legacy_pairs(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    symbols = payload.get("symbols") or []
    overrides = payload.get("overrides") or {}
    universe: List[Dict[str, Any]] = []
    for raw in symbols:
        sym = str(raw or "").upper()
        if not sym:
            continue
        entry = dict(overrides.get(sym) or {})
        entry["symbol"] = sym
        universe.append(entry)
    return universe


def _normalize_pair_entry(raw: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    symbol = str(raw.get("symbol") or "").upper()
    if not symbol:
        return None
    tier = str(raw.get("tier") or "").upper() or None
    bucket = str(raw.get("bucket") or "").upper() or tier
    caps = raw.get("caps") or {}
    norm_caps: Dict[str, Any] = {}
    if isinstance(caps, Mapping):
        for key in ("max_nav_pct", "max_order_notional", "max_concurrent_positions"):
            if key in caps:
                norm_caps[key] = caps[key]
    target_lev = raw.get("target_leverage")
    try:
        target_lev = float(target_lev)
    except Exception:
        target_lev = None
    max_lev = raw.get("max_leverage")
    try:
        max_lev = float(max_lev)
    except Exception:
        max_lev = None
    min_notional = raw.get("min_notional")
    try:
        min_notional = float(min_notional)
    except Exception:
        min_notional = None
    min_gross = raw.get("min_gross_usd")
    try:
        min_gross = float(min_gross)
    except Exception:
        min_gross = None
    entry = {
        "symbol": symbol,
        "tier": tier,
        "bucket": bucket or tier,
        "enabled": bool(raw.get("enabled", True)),
        "status": raw.get("status") or ("enabled" if raw.get("enabled", True) else "disabled"),
        "target_leverage": target_lev,
        "max_leverage": max_lev,
        "min_notional": min_notional,
        "min_gross_usd": min_gross,
        "caps": norm_caps,
    }
    return entry


@lru_cache(maxsize=1)
def load_pairs_universe() -> List[Dict[str, Any]]:
    payload = _raw_pairs_payload()
    if isinstance(payload, dict):
        raw_universe = payload.get("universe")
        if raw_universe is None:
            raw_universe = _iter_legacy_pairs(payload)
    elif isinstance(payload, list):
        raw_universe = payload
    else:
        raw_universe = []
    normalized: List[Dict[str, Any]] = []
    for entry in raw_universe or []:
        if not isinstance(entry, Mapping):
            continue
        norm = _normalize_pair_entry(entry)
        if norm:
            normalized.append(norm)
    return normalized


@lru_cache(maxsize=1)
def universe_by_symbol() -> Dict[str, Dict[str, Any]]:
    return {entry["symbol"]: entry for entry in load_pairs_universe() if entry.get("symbol")}


def symbol_universe_info(symbol: str) -> Dict[str, Any]:
    return dict(universe_by_symbol().get(str(symbol).upper()) or {})


def symbol_min_gross(symbol: str) -> float:
    info = symbol_universe_info(symbol)
    try:
        val = float(info.get("min_gross_usd", 0.0) or 0.0)
    except Exception:
        val = 0.0
    return max(val, 0.0)


def symbol_min_notional(symbol: str) -> float:
    info = symbol_universe_info(symbol)
    try:
        val = float(info.get("min_notional", 0.0) or 0.0)
    except Exception:
        val = 0.0
    return max(val, 0.0)


def symbol_target_leverage(symbol: str) -> float:
    info = symbol_universe_info(symbol)
    try:
        val = float(info.get("target_leverage", 0.0) or 0.0)
    except Exception:
        val = 0.0
    return val


def resolve_allowed_symbols() -> Tuple[List[str], Dict[str, str]]:
    """Return (allowed_symbols, tier_by_symbol).

    allowed = (tiers_whitelist ∩ exchange_listed ∩ not_throttled)
    Optionally merge discovery symbols if settings.automerge_discovery is true and records have
    liquidity_ok and trend_ok.
    """
    pairs = load_pairs_universe()
    ordered: List[str] = [entry["symbol"] for entry in pairs if entry.get("enabled", True)]
    tier_by: Dict[str, str] = {entry["symbol"]: str(entry.get("tier")) for entry in pairs if entry.get("symbol")}

    tiers_cfg = load_tiers()
    for t, arr in tiers_cfg.items():
        for s in arr:
            tier_by.setdefault(s, t)
        if t not in tier_by:
            continue
        for sym in arr:
            if sym not in ordered:
                ordered.append(sym)

    # Optional throttle blocklist
    throttle_block: Set[str] = set()
    try:
        cfg = _load_json("config/risk_limits.json", {})
        bl = ((cfg.get("global") or {}).get("throttle") or {}).get("blocked") or []
        throttle_block = {str(x).upper() for x in bl}
    except Exception:
        pass

    # Discovery merge (operator gated)
    settings = _load_settings()
    do_merge = bool((settings.get("settings") or {}).get("automerge_discovery", False)) or bool(
        settings.get("automerge_discovery", False)
    )
    if do_merge:
        for rec in _load_discovery():
            try:
                s = str(rec.get("symbol")).upper()
                if not s:
                    continue
                if not rec.get("liquidity_ok") or not rec.get("trend_ok"):
                    continue
                if s not in ordered:
                    ordered.append(s)
                    tier_by.setdefault(s, "DISCOVERY")
            except Exception:
                continue

    # Apply listing + throttle filters
    out: List[str] = []
    for s in ordered:
        if s in throttle_block:
            continue
        if is_listed_on_futures(s):
            out.append(s)
    return out, tier_by


__all__ = [
    "load_tiers",
    "symbol_tier",
    "resolve_allowed_symbols",
    "is_listed_on_futures",
    "load_pairs_universe",
    "universe_by_symbol",
    "symbol_universe_info",
    "symbol_min_gross",
    "symbol_min_notional",
    "symbol_target_leverage",
    "get_symbol_price",
]


def get_symbol_price(symbol: str) -> float:
    """
    Return a price for the supplied futures symbol, guarding off-exchange assets.
    Routes XAUT/USDC style symbols to spot to avoid unsupported futures lookups.
    """
    from execution.exchange_utils import get_price

    sym = str(symbol or "").upper()
    if not sym:
        raise ValueError("symbol_required")

    if sym in {"XAUT", "XAUTUSDT"}:
        return float(get_price("XAUTUSDT", venue="spot", signed=False))
    if sym in {"USDC", "USDCUSDT"}:
        return float(get_price("USDC", venue="spot", signed=False))
    return float(get_price(sym, venue="fapi", signed=False))
