from __future__ import annotations

import json
import os
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

RISK_LIMITS_PATH = Path("config/risk_limits.json")
PAIRS_UNIVERSE_PATH = Path("config/pairs_universe.json")
CORRELATION_GROUPS_PATH = Path("config/correlation_groups.json")
LOGGER = logging.getLogger("risk_loader")


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def normalize_percentage(value: Any) -> float:
    """
    Normalize percentage-like values to a fraction in [0, 1].

    - If 0 < value <= 1, treat as already expressed as a fraction.
    - If value > 1, treat as percent and divide by 100.
    - Clamp absurd values (>100%) to 1.0 and log an error.
    """
    try:
        pct = float(value)
    except Exception:
        return 0.0
    if pct <= 0:
        return 0.0
    if pct <= 1.0:
        return pct
    if pct > 100.0:
        LOGGER.error("[risk_loader] normalize_percentage clamp for value>100: %s", pct)
        pct = 100.0
    return pct / 100.0


def _merge_pair_caps(risk_cfg: Dict[str, Any], pairs_cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    per_symbol = risk_cfg.setdefault("per_symbol", {})
    merged_caps: Dict[str, Dict[str, Any]] = {}
    universe = pairs_cfg.get("universe") if isinstance(pairs_cfg, Mapping) else []
    for entry in universe or []:
        if not isinstance(entry, Mapping):
            continue
        sym = str(entry.get("symbol") or "").upper()
        if not sym:
            continue
        caps = entry.get("caps") or {}
        base = per_symbol.get(sym) if isinstance(per_symbol, Mapping) else {}
        merged = dict(caps) if isinstance(caps, Mapping) else {}
        if isinstance(base, Mapping):
            merged.update(base)
        if merged:
            per_symbol[sym] = merged
            merged_caps[sym] = merged
    return risk_cfg, merged_caps


def apply_testnet_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    flag = os.getenv("BINANCE_TESTNET", "0").lower()
    enabled = flag in {"1", "true", "yes"}
    overrides = (cfg or {}).get("testnet_overrides") or {}
    if not enabled or not overrides.get("enabled"):
        return cfg
    cfg = dict(cfg or {})
    global_cfg = cfg.setdefault("global", {})
    for key, value in overrides.items():
        if key in {"enabled", "log_notice"}:
            continue
        global_cfg[key] = value
        cfg[key] = value
    meta = cfg.setdefault("_meta", {})
    meta["testnet_overrides_active"] = True
    if overrides.get("log_notice"):
        try:
            import logging

            logging.getLogger("risk_loader").info(
                "[risk_loader][testnet] overrides applied %s", {k: v for k, v in overrides.items() if k != "enabled"}
            )
        except Exception:
            pass
    return cfg


@lru_cache(maxsize=1)
def load_risk_config() -> Dict[str, Any]:
    """
    Canonical risk configuration loader.

    - Reads risk_limits.json and pairs_universe.json.
    - Merges per-symbol caps from pairs into risk config.
    - Applies BINANCE_TESTNET overrides.
    - Ensures quote_symbols includes all supported quote assets.
    """
    risk_cfg = _load_json(RISK_LIMITS_PATH)
    pairs_cfg = _load_json(PAIRS_UNIVERSE_PATH)
    if not isinstance(risk_cfg.get("global"), dict):
        risk_cfg["global"] = {}
    risk_cfg, _ = _merge_pair_caps(risk_cfg, pairs_cfg)
    g = risk_cfg.setdefault("global", {})
    for key in ("nav_freshness_seconds", "fail_closed_on_nav_stale"):
        if key in risk_cfg and key not in g:
            g[key] = risk_cfg[key]
    # Normalize percent-like caps to guard against mis-scaled configs
    for key in (
        "trade_equity_nav_pct",
        "max_trade_nav_pct",
        "symbol_notional_share_cap_pct",
        "max_symbol_exposure_pct",
        "max_gross_exposure_pct",
    ):
        if key in g:
            g[key] = normalize_percentage(g[key])
    per_symbol = risk_cfg.get("per_symbol") or {}
    if isinstance(per_symbol, dict):
        for sym_key, entry in per_symbol.items():
            if not isinstance(entry, dict):
                continue
            if "max_nav_pct" in entry:
                entry["max_nav_pct"] = normalize_percentage(entry["max_nav_pct"])
                if entry["max_nav_pct"] <= 0:
                    entry.pop("max_nav_pct", None)
            if "symbol_notional_share_cap_pct" in entry:
                entry["symbol_notional_share_cap_pct"] = normalize_percentage(entry["symbol_notional_share_cap_pct"])
            if "symbol_drawdown_cap_pct" in entry:
                entry["symbol_drawdown_cap_pct"] = normalize_percentage(entry["symbol_drawdown_cap_pct"])
    risk_cfg = apply_testnet_overrides(risk_cfg)

    # Normalize circuit_breakers config
    cb_cfg = risk_cfg.get("circuit_breakers") or {}
    if isinstance(cb_cfg, dict):
        max_dd = cb_cfg.get("max_portfolio_dd_nav_pct")
        if max_dd is not None:
            cb_cfg["max_portfolio_dd_nav_pct"] = normalize_percentage(max_dd)
        risk_cfg["circuit_breakers"] = cb_cfg

    quotes = set()
    try:
        for q in risk_cfg.get("quote_symbols") or []:
            quotes.add(str(q).upper())
    except Exception:
        quotes = set()
    quotes.update({"USDT", "USDC", "FDUSD"})
    risk_cfg["quote_symbols"] = sorted(quotes)
    return risk_cfg


def load_symbol_caps() -> Dict[str, Dict[str, Any]]:
    """
    Return per-symbol cap configuration normalized for exposure checks.

    Prefers per-symbol max_nav_pct, falling back to global symbol_notional_share_cap_pct.
    """
    cfg = load_risk_config()
    global_cfg = cfg.get("global") or {}
    default_cap = normalize_percentage(global_cfg.get("symbol_notional_share_cap_pct"))
    per_symbol_raw = cfg.get("per_symbol") or {}
    caps: Dict[str, Dict[str, Any]] = {}
    for sym, entry in per_symbol_raw.items():
        try:
            sym_key = str(sym).upper()
        except Exception:
            continue
        cap_cfg = entry.get("max_nav_pct", entry.get("symbol_notional_share_cap_pct"))
        cap_frac = normalize_percentage(cap_cfg if cap_cfg is not None else default_cap)
        caps[sym_key] = {
            "cap_cfg_raw": cap_cfg if cap_cfg is not None else default_cap,
            "cap_cfg_normalized": cap_frac,
        }
    return caps


__all__ = [
    "apply_testnet_overrides",
    "load_risk_config",
    "load_symbol_caps",
    "normalize_percentage",
    "CorrelationGroupConfig",
    "CorrelationGroupsConfig",
    "load_correlation_groups_config",
]


# --- Correlation Groups Config ---

@dataclass
class CorrelationGroupConfig:
    """Configuration for a single correlation group."""
    max_group_nav_pct: float
    symbols: List[str] = field(default_factory=list)


@dataclass
class CorrelationGroupsConfig:
    """Configuration for all correlation groups."""
    groups: Dict[str, CorrelationGroupConfig] = field(default_factory=dict)


@lru_cache(maxsize=1)
def load_correlation_groups_config(
    config_path: Path = CORRELATION_GROUPS_PATH,
) -> CorrelationGroupsConfig:
    """
    Load and normalize correlation groups configuration.

    Returns CorrelationGroupsConfig with empty groups if file is missing,
    malformed, or empty.
    """
    try:
        if not config_path.exists():
            return CorrelationGroupsConfig(groups={})

        with open(config_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        if not isinstance(data, dict):
            return CorrelationGroupsConfig(groups={})

        raw_groups = data.get("groups")
        if not isinstance(raw_groups, dict):
            return CorrelationGroupsConfig(groups={})

        groups: Dict[str, CorrelationGroupConfig] = {}
        for group_name, group_data in raw_groups.items():
            if not isinstance(group_data, dict):
                continue

            symbols_raw = group_data.get("symbols")
            if not isinstance(symbols_raw, list):
                continue

            symbols = [str(s).upper() for s in symbols_raw if s]
            if not symbols:
                continue

            max_pct_raw = group_data.get("max_group_nav_pct")
            if max_pct_raw is None:
                continue

            max_pct = normalize_percentage(max_pct_raw)
            if max_pct <= 0:
                continue

            groups[str(group_name)] = CorrelationGroupConfig(
                max_group_nav_pct=max_pct,
                symbols=symbols,
            )

        return CorrelationGroupsConfig(groups=groups)

    except Exception as exc:
        LOGGER.error("[risk_loader] load_correlation_groups_config failed: %s", exc)
        return CorrelationGroupsConfig(groups={})
