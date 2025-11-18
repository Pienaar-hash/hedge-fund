#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple

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


def resolve_allowed_symbols() -> Tuple[List[str], Dict[str, str]]:
    """Return (allowed_symbols, tier_by_symbol).

    allowed = (tiers_whitelist ∩ exchange_listed ∩ not_throttled)
    Optionally merge discovery symbols if settings.automerge_discovery is true and records have
    liquidity_ok and trend_ok.
    """
    tiers = load_tiers()
    ordered: List[str] = []
    for key in ("CORE", "SATELLITE", "TACTICAL", "ALT-EXT"):
        ordered.extend([s for s in tiers.get(key, [])])
    tier_by: Dict[str, str] = {}
    for t, arr in tiers.items():
        for s in arr:
            tier_by[s] = t

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
