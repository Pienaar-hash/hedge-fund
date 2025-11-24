"""Feedback-based risk allocator (intel-only, v6)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from execution.utils import load_json
from execution.risk_loader import load_risk_config


DEFAULT_STATE_DIR = Path(os.getenv("HEDGE_STATE_DIR") or "logs/state")
DEFAULT_EXPECTANCY_PATH = DEFAULT_STATE_DIR / "expectancy_v6.json"
DEFAULT_SCORES_PATH = DEFAULT_STATE_DIR / "symbol_scores_v6.json"
DEFAULT_ROUTER_POLICY_PATH = DEFAULT_STATE_DIR / "router_policy_suggestions_v6.json"
DEFAULT_NAV_PATH = DEFAULT_STATE_DIR / "nav.json"
DEFAULT_RISK_SNAPSHOT_PATH = DEFAULT_STATE_DIR / "risk_snapshot.json"
DEFAULT_RISK_CONFIG_PATH = Path(os.getenv("RISK_LIMITS_CONFIG") or "config/risk_limits.json")
DEFAULT_PAIRS_PATH = Path(os.getenv("PAIRS_UNIVERSE_PATH") or "config/pairs_universe.json")

RISK_MODE_THRESHOLDS = [5.0, 10.0]  # drawdown %
RISK_MODES = ("normal", "cautious", "defensive")
TOTAL_WEIGHT_TARGET = {
    "normal": 0.9,
    "cautious": 0.65,
    "defensive": 0.45,
}
NAV_CAP_MODE_SCALE = {
    "normal": 1.05,
    "cautious": 0.85,
    "defensive": 0.65,
}
TRADE_CAP_MODE_SCALE = {
    "normal": 1.0,
    "cautious": 0.85,
    "defensive": 0.7,
}


@dataclass
class SymbolCaps:
    max_nav_pct: Optional[float]
    max_trade_nav_pct: Optional[float]
    max_concurrent: Optional[int]


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        return {}
    return {}


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_pct(value: Any) -> Optional[float]:
    num = _safe_float(value)
    if num is None:
        return None
    if num > 1.0:
        num /= 100.0
    if num < 0:
        num = 0.0
    if num > 1.0:
        num = 1.0
    return num


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _symbol_expectancy_map(snapshot: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    symbols = snapshot.get("symbols")
    if isinstance(symbols, Mapping):
        return {str(sym).upper(): data for sym, data in symbols.items() if isinstance(data, Mapping)}
    return {}


def _symbol_scores_map(snapshot: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    rows = snapshot.get("symbols")
    scores: Dict[str, Mapping[str, Any]] = {}
    if isinstance(rows, Iterable):
        for entry in rows:
            if not isinstance(entry, Mapping):
                continue
            symbol = entry.get("symbol")
            if not symbol:
                continue
            scores[str(symbol).upper()] = entry
    return scores


def _router_policy_map(snapshot: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    rows = snapshot.get("symbols")
    policies: Dict[str, Mapping[str, Any]] = {}
    if isinstance(rows, Iterable):
        for entry in rows:
            if not isinstance(entry, Mapping):
                continue
            symbol = entry.get("symbol")
            if not symbol:
                continue
            policies[str(symbol).upper()] = entry
    return policies


def _universe_caps(pairs_payload: Mapping[str, Any]) -> Dict[str, SymbolCaps]:
    universe = pairs_payload.get("universe")
    results: Dict[str, SymbolCaps] = {}
    if isinstance(universe, Iterable):
        for entry in universe:
            if not isinstance(entry, Mapping):
                continue
            symbol = str(entry.get("symbol") or "").upper()
            if not symbol:
                continue
            caps = entry.get("caps") if isinstance(entry.get("caps"), Mapping) else {}
            max_nav = _normalize_pct(caps.get("max_nav_pct"))
            max_trade = _normalize_pct(caps.get("max_trade_nav_pct"))
            max_conc = caps.get("max_concurrent_positions")
            results[symbol] = SymbolCaps(
                max_nav_pct=max_nav,
                max_trade_nav_pct=max_trade,
                max_concurrent=int(max_conc) if isinstance(max_conc, (int, float)) else None,
            )
    return results


def _risk_config_caps(risk_cfg: Mapping[str, Any]) -> Tuple[Dict[str, SymbolCaps], Optional[float], Optional[float], Optional[int]]:
    per_symbol = risk_cfg.get("per_symbol")
    per_symbol_caps: Dict[str, SymbolCaps] = {}
    if isinstance(per_symbol, Mapping):
        for symbol, entry in per_symbol.items():
            if not isinstance(entry, Mapping):
                continue
            nav_pct = _normalize_pct(entry.get("max_nav_pct"))
            max_trade = _normalize_pct(entry.get("max_trade_nav_pct"))
            max_conc = entry.get("max_concurrent_positions")
            per_symbol_caps[str(symbol).upper()] = SymbolCaps(
                max_nav_pct=nav_pct,
                max_trade_nav_pct=max_trade,
                max_concurrent=int(max_conc) if isinstance(max_conc, (int, float)) else None,
            )
    global_cfg = risk_cfg.get("global") if isinstance(risk_cfg.get("global"), Mapping) else {}
    global_symbol_cap = _normalize_pct(global_cfg.get("symbol_notional_share_cap_pct"))
    global_trade_cap = _normalize_pct(global_cfg.get("max_trade_nav_pct"))
    max_positions = global_cfg.get("max_concurrent_positions")
    max_positions_int = int(max_positions) if isinstance(max_positions, (int, float)) else None
    return per_symbol_caps, global_symbol_cap, global_trade_cap, max_positions_int


def _merge_caps(
    symbol: str,
    pair_caps: Dict[str, SymbolCaps],
    risk_caps: Dict[str, SymbolCaps],
    global_trade_cap: Optional[float],
) -> SymbolCaps:
    from_pairs = pair_caps.get(symbol)
    from_risk = risk_caps.get(symbol)
    max_nav = from_risk.max_nav_pct if from_risk and from_risk.max_nav_pct is not None else (
        from_pairs.max_nav_pct if from_pairs and from_pairs.max_nav_pct is not None else None
    )
    max_trade = from_risk.max_trade_nav_pct if from_risk and from_risk.max_trade_nav_pct is not None else (
        from_pairs.max_trade_nav_pct if from_pairs and from_pairs.max_trade_nav_pct is not None else global_trade_cap
    )
    max_conc = (
        from_risk.max_concurrent if from_risk and from_risk.max_concurrent is not None else (
            from_pairs.max_concurrent if from_pairs and from_pairs.max_concurrent is not None else None
        )
    )
    return SymbolCaps(max_nav_pct=max_nav, max_trade_nav_pct=max_trade, max_concurrent=max_conc)


def _risk_mode(drawdown_pct: float) -> str:
    if drawdown_pct >= RISK_MODE_THRESHOLDS[1]:
        return RISK_MODES[2]
    if drawdown_pct >= RISK_MODE_THRESHOLDS[0]:
        return RISK_MODES[1]
    return RISK_MODES[0]


def _drawdown_from_risk_snapshot(snapshot: Mapping[str, Any]) -> float:
    symbols = snapshot.get("symbols")
    worst = 0.0
    if isinstance(symbols, Iterable):
        for entry in symbols:
            if not isinstance(entry, Mapping):
                continue
            risk_section = entry.get("risk")
            if not isinstance(risk_section, Mapping):
                continue
            dd = _safe_float(risk_section.get("dd_today_pct"))
            if dd is None:
                continue
            worst = max(worst, abs(min(dd, 0.0)))
    return worst


def _equity_from_nav(snapshot: Mapping[str, Any]) -> Optional[float]:
    if not isinstance(snapshot, Mapping):
        return None
    nav = snapshot.get("equity")
    if nav is None:
        nav = snapshot.get("nav")
    return _safe_float(nav)


def _weight_signal(score: Optional[float], expectancy: Optional[float], router_quality: Optional[str]) -> float:
    base = max(0.0, score or 0.0)
    if expectancy is not None and expectancy > 0:
        base += min(expectancy / 10.0, 0.3)
    if router_quality == "broken":
        base *= 0.2
    elif router_quality == "degraded":
        base *= 0.6
    return base


def _router_policy_bundle(router_entry: Mapping[str, Any]) -> Dict[str, Any]:
    current = router_entry.get("current_policy") if isinstance(router_entry.get("current_policy"), Mapping) else None
    proposed = router_entry.get("proposed_policy") if isinstance(router_entry.get("proposed_policy"), Mapping) else None
    return {
        "current": current,
        "proposed": proposed,
    }


def build_suggestions(
    expectancy_snapshot: Optional[Mapping[str, Any]] = None,
    symbol_scores_snapshot: Optional[Mapping[str, Any]] = None,
    router_policy_snapshot: Optional[Mapping[str, Any]] = None,
    nav_snapshot: Optional[Mapping[str, Any]] = None,
    risk_snapshot: Optional[Mapping[str, Any]] = None,
    risk_config: Optional[Mapping[str, Any]] = None,
    pairs_universe: Optional[Mapping[str, Any]] = None,
    *,
    state_dir: Path | str | None = None,
    risk_config_path: Path | str | None = None,
    pairs_universe_path: Path | str | None = None,
    lookback_days: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Build risk allocation suggestions. Pure intel; never mutates live state.
    """
    state_dir_path = Path(state_dir) if state_dir else DEFAULT_STATE_DIR
    exp_snapshot = expectancy_snapshot or _read_json(Path(state_dir_path) / DEFAULT_EXPECTANCY_PATH.name)
    scores_snapshot = symbol_scores_snapshot or _read_json(Path(state_dir_path) / DEFAULT_SCORES_PATH.name)
    router_snapshot = router_policy_snapshot or _read_json(Path(state_dir_path) / DEFAULT_ROUTER_POLICY_PATH.name)
    nav_payload = nav_snapshot or _read_json(Path(state_dir_path) / DEFAULT_NAV_PATH.name)
    risk_payload = risk_snapshot or _read_json(Path(state_dir_path) / DEFAULT_RISK_SNAPSHOT_PATH.name)
    if risk_config is not None:
        risk_cfg = risk_config
    elif risk_config_path:
        risk_cfg = load_json(str(Path(risk_config_path)))
    else:
        risk_cfg = load_risk_config()
    pairs_payload = pairs_universe or _read_json(Path(pairs_universe_path) if pairs_universe_path else DEFAULT_PAIRS_PATH)

    exp_map = _symbol_expectancy_map(exp_snapshot or {})
    scores_map = _symbol_scores_map(scores_snapshot or {})
    router_map = _router_policy_map(router_snapshot or {})
    pair_caps = _universe_caps(pairs_payload or {})
    risk_caps, global_symbol_cap, global_trade_cap, _ = _risk_config_caps(risk_cfg if isinstance(risk_cfg, Mapping) else {})

    candidates = sorted(set(exp_map.keys()) | set(scores_map.keys()))
    enabled_candidates = [sym for sym in candidates if sym in pair_caps]
    drawdown = _drawdown_from_risk_snapshot(risk_payload or {})
    risk_mode = _risk_mode(drawdown)
    equity = _equity_from_nav(nav_payload or {})

    target_total_weight = TOTAL_WEIGHT_TARGET.get(risk_mode, 0.5)
    weight_signals: Dict[str, float] = {}
    router_quality_cache: Dict[str, Optional[str]] = {}

    for symbol in enabled_candidates:
        score = _safe_float((scores_map.get(symbol) or {}).get("score"))
        expectancy_val = _safe_float((exp_map.get(symbol) or {}).get("expectancy"))
        router_quality = (router_map.get(symbol) or {}).get("quality")
        router_quality_cache[symbol] = router_quality if isinstance(router_quality, str) else None
        weight_signals[symbol] = _weight_signal(score, expectancy_val, router_quality_cache[symbol])

    total_signal = sum(val for val in weight_signals.values() if val > 0)
    if total_signal <= 0:
        return {
            "generated_ts": _iso_now(),
            "lookback_days": float(lookback_days or _safe_float(exp_snapshot.get("lookback_hours") or 0) or 0) / 24.0 if exp_snapshot else 7.0,
            "global": {
                "current_equity_usd": equity,
                "current_drawdown_pct": drawdown,
                "risk_mode": risk_mode,
            },
            "symbols": [],
        }

    suggestions: List[Dict[str, Any]] = []
    lookback = lookback_days
    if lookback is None:
        hours = _safe_float((exp_snapshot or {}).get("lookback_hours"))
        lookback = (hours / 24.0) if hours else 7.0

    for symbol in enabled_candidates:
        signal = weight_signals.get(symbol, 0.0)
        weight_share = signal / total_signal if total_signal > 0 else 0.0
        suggested_weight = round(weight_share * target_total_weight, 6)

        pair_cap = pair_caps.get(symbol)
        caps = _merge_caps(symbol, pair_caps, risk_caps, global_trade_cap)
        current_nav_cap = caps.max_nav_pct
        if current_nav_cap is None and pair_cap:
            current_nav_cap = pair_cap.max_nav_pct
        if current_nav_cap is None:
            current_nav_cap = global_symbol_cap
        cap_upper_bound = global_symbol_cap if global_symbol_cap is not None else current_nav_cap
        if cap_upper_bound is None:
            cap_upper_bound = 0.25
        nav_cap_scale = NAV_CAP_MODE_SCALE.get(risk_mode, 1.0)
        nav_cap_factor = nav_cap_scale * (0.8 + 0.4 * min(weight_share, 1.0))
        base_nav_cap = current_nav_cap if current_nav_cap is not None else cap_upper_bound
        proposed_nav_cap = min(cap_upper_bound, base_nav_cap * nav_cap_factor) if base_nav_cap is not None else cap_upper_bound
        proposed_nav_cap = max(0.0, proposed_nav_cap)

        trade_cap_scale = TRADE_CAP_MODE_SCALE.get(risk_mode, 1.0)
        current_trade_cap = caps.max_trade_nav_pct
        if current_trade_cap is None and pair_cap:
            current_trade_cap = pair_cap.max_trade_nav_pct
        if current_trade_cap is None:
            current_trade_cap = global_trade_cap
        if current_trade_cap is None:
            current_trade_cap = 0.2
        proposed_trade_cap = min(current_trade_cap, current_trade_cap * trade_cap_scale * (0.85 + 0.3 * min(weight_share, 1.0)))

        current_concurrent = caps.max_concurrent if caps.max_concurrent is not None else pair_cap.max_concurrent if pair_cap else None
        proposed_concurrent = current_concurrent
        if current_concurrent is not None and risk_mode != "normal" and weight_share < 0.5:
            drop = 1 if risk_mode == "defensive" or weight_share < 0.2 else 0
            if drop:
                proposed_concurrent = max(0, current_concurrent - drop)

        expectancy_entry = exp_map.get(symbol, {})
        router_entry = router_map.get(symbol, {})
        score_entry = scores_map.get(symbol, {})

        rationale = []
        score_val = _safe_float(score_entry.get("score"))
        if score_val is not None:
            rationale.append(f"Symbol score={score_val:.2f} drove weight share {weight_share:.2%}.")
        exp_val = _safe_float(expectancy_entry.get("expectancy"))
        if exp_val is not None:
            rationale.append(f"Expectancy snapshot={exp_val:.2f} (hit_rate={expectancy_entry.get('hit_rate')}).")
        rationale.append(f"Risk mode {risk_mode} scaled caps/weights.")

        suggestions.append(
            {
                "symbol": symbol,
                "score": score_val,
                "expectancy": {
                    "expectancy": exp_val,
                    "avg_return": _safe_float(expectancy_entry.get("avg_return")),
                    "hit_rate": _safe_float(expectancy_entry.get("hit_rate")),
                    "regime": router_entry.get("regime"),
                },
                "router_policy": _router_policy_bundle(router_entry),
                "caps": {
                    "current_max_nav_pct": base_nav_cap,
                    "current_max_trade_nav_pct": current_trade_cap,
                    "current_max_concurrent_positions": current_concurrent,
                },
                "suggested_caps": {
                    "max_nav_pct": round(proposed_nav_cap, 6),
                    "max_trade_nav_pct": round(proposed_trade_cap, 6),
                    "max_concurrent_positions": proposed_concurrent,
                },
                "suggested_weight": suggested_weight,
                "rationale": rationale,
            }
        )

    suggestions.sort(key=lambda item: item["symbol"])
    total_weight = sum(item["suggested_weight"] for item in suggestions)
    if total_weight > 1.0 and total_weight > 0:
        scale = 1.0 / total_weight
        for item in suggestions:
            item["suggested_weight"] = round(item["suggested_weight"] * scale, 6)

    return {
        "generated_ts": _iso_now(),
        "lookback_days": float(lookback or 7.0),
        "global": {
            "current_equity_usd": equity,
            "current_drawdown_pct": drawdown,
            "risk_mode": risk_mode,
        },
        "symbols": suggestions,
    }


__all__ = ["build_suggestions"]
