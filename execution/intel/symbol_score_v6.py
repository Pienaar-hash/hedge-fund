"""Symbol scoring helpers for v6.0 telemetry (analysis-only)."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


DEFAULT_STATE_DIR = Path(os.getenv("HEDGE_STATE_DIR") or "logs/state")
DEFAULT_EXPECTANCY_PATH = DEFAULT_STATE_DIR / "expectancy_v6.json"
DEFAULT_ROUTER_HEALTH_PATH = DEFAULT_STATE_DIR / "router_health.json"


def load_expectancy_snapshot(path: Path | str = DEFAULT_EXPECTANCY_PATH) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_router_health_snapshot(path: Path | str = DEFAULT_ROUTER_HEALTH_PATH) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _get_symbol_stats(expectancy_snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    data = expectancy_snapshot.get("symbols") if isinstance(expectancy_snapshot, Mapping) else None
    return data if isinstance(data, Mapping) else {}


def _unpack_router_health(router_health: Mapping[str, Any]) -> Mapping[str, Any]:
    symbols = router_health.get("symbols") if isinstance(router_health, Mapping) else None
    if isinstance(symbols, list):
        return {str(entry.get("symbol")).upper(): entry for entry in symbols if entry.get("symbol")}
    return {}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _scale_expectancy(value: float) -> float:
    return 0.5 + 0.5 * math.tanh(value / 10.0)


def _scale_router_quality(router: Mapping[str, Any]) -> float:
    maker = float(router.get("maker_fill_rate") or 0.0)
    fallback = float(router.get("fallback_rate") or 0.0)
    raw = maker - fallback
    return 0.5 + 0.5 * math.tanh(raw)


def _slippage_penalty(router: Mapping[str, Any]) -> float:
    slip = float(router.get("slippage_p50") or 0.0)
    if slip <= 0:
        return 0.0
    return min(0.4, slip / 20.0)


def _fee_drag_penalty(router: Mapping[str, Any]) -> float:
    fees = router.get("fees_total")
    pnl = router.get("realized_pnl") or router.get("cum_pnl")
    if fees is None or pnl in (None, 0):
        return 0.0
    try:
        ratio = abs(float(fees)) / max(1e-6, abs(float(pnl)))
    except Exception:
        return 0.0
    return min(0.4, ratio * 0.2)


def _volatility_penalty(router: Mapping[str, Any]) -> float:
    scale = router.get("volatility_scale")
    if scale is None:
        return 0.0
    try:
        delta = abs(float(scale) - 1.0)
    except Exception:
        delta = 0.0
    return min(0.5, delta)


def score_symbol(symbol: str, metrics: Mapping[str, Any]) -> Dict[str, Any]:
    expect = metrics.get("expectancy") if isinstance(metrics, Mapping) else None
    router = metrics.get("router") if isinstance(metrics, Mapping) else None
    expect = expect if isinstance(expect, Mapping) else {}
    router = router if isinstance(router, Mapping) else {}
    expectancy_score = _scale_expectancy(float(expect.get("expectancy") or 0.0))
    hit_rate = expect.get("hit_rate")
    if isinstance(hit_rate, (int, float)):
        expectancy_score = 0.7 * expectancy_score + 0.3 * _clamp01(float(hit_rate))
    router_score = _scale_router_quality(router)
    slippage_pen = _slippage_penalty(router)
    fee_pen = _fee_drag_penalty(router)
    vol_pen = _volatility_penalty(router)
    raw = expectancy_score * 0.55 + router_score * 0.35 - slippage_pen - fee_pen - vol_pen * 0.3
    score = _clamp01(raw)
    return {
        "symbol": symbol,
        "score": score,
        "components": {
            "expectancy": expectancy_score,
            "router": router_score,
            "slippage_penalty": slippage_pen,
            "fee_drag_penalty": fee_pen,
            "volatility_penalty": vol_pen,
        },
        "inputs": {
            "expectancy": expect,
            "router": router,
        },
    }


def score_universe(expectancy_snapshot: Mapping[str, Any], router_health_snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    exp_data = _get_symbol_stats(expectancy_snapshot)
    router_map = _unpack_router_health(router_health_snapshot)
    symbols = sorted({sym for sym in exp_data.keys() | router_map.keys() if sym})
    rows = []
    for symbol in symbols:
        entry = score_symbol(
            symbol,
            {
                "expectancy": exp_data.get(symbol, {}),
                "router": router_map.get(symbol, {}),
            },
        )
        rows.append(entry)
    rows.sort(key=lambda item: item["score"], reverse=True)
    return {"updated_ts": time.time(), "symbols": rows}


__all__ = [
    "load_expectancy_snapshot",
    "load_router_health_snapshot",
    "score_symbol",
    "score_universe",
]
