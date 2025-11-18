"""Router policy auto-tune suggestions (analysis-only, v6)."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from execution.intel import expectancy_v6
from execution.intel import maker_offset
from execution.intel.router_policy import classify_router_quality, classify_router_regime
from execution.utils import load_json


DEFAULT_STATE_DIR = Path(os.getenv("HEDGE_STATE_DIR") or "logs/state")
DEFAULT_RISK_LIMITS_PATH = Path(os.getenv("RISK_LIMITS_CONFIG") or "config/risk_limits.json")
DEFAULT_LOOKBACK_DAYS = 7.0
BIAS_LABEL_TO_VALUE = {
    "prefer_maker": 0.25,
    "maker": 0.25,
    "balanced": 0.5,
    "neutral": 0.5,
    "prefer_taker": 0.75,
    "taker": 0.8,
}


@dataclass(frozen=True)
class AutotuneBounds:
    min_bias: float = 0.0
    max_bias: float = 1.0
    max_bias_step: float = 0.05
    min_offset_bps: float = max(0.0, float(getattr(maker_offset, "MIN_OFFSET_BPS", 0.5)))
    max_offset_bps: float = float(getattr(maker_offset, "MAX_OFFSET_BPS", 8.0))
    max_offset_step_bps: float = 1.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _to_float(value: Any) -> Optional[float]:
    try:
        if value in (None, "", "null"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        return {}
    return {}


def _resolve_bounds(risk_config: Mapping[str, Any] | None) -> AutotuneBounds:
    if isinstance(risk_config, Mapping):
        raw = risk_config.get("router_autotune_v6") or risk_config.get("router_autotune")
        if isinstance(raw, Mapping):
            min_bias = _to_float(raw.get("min_bias")) or AutotuneBounds.min_bias
            max_bias = _to_float(raw.get("max_bias")) or AutotuneBounds.max_bias
            max_bias_step = abs(_to_float(raw.get("max_bias_step")) or AutotuneBounds.max_bias_step)
            min_offset_bps = _to_float(raw.get("min_offset_bps")) or AutotuneBounds.min_offset_bps
            max_offset_bps = _to_float(raw.get("max_offset_bps")) or AutotuneBounds.max_offset_bps
            max_offset_step = abs(_to_float(raw.get("max_offset_step_bps")) or AutotuneBounds.max_offset_step_bps)
            return AutotuneBounds(
                min_bias=_clamp(min_bias, 0.0, 1.0),
                max_bias=_clamp(max_bias, 0.0, 1.0),
                max_bias_step=max(0.01, min(0.25, max_bias_step)),
                min_offset_bps=max(0.0, min_offset_bps),
                max_offset_bps=max(0.5, max_offset_bps),
                max_offset_step_bps=max(0.25, min(3.0, max_offset_step)),
            )
    return AutotuneBounds()


def _bias_label_from_value(value: float) -> str:
    if value <= 0.35:
        return "prefer_maker"
    if value >= 0.65:
        return "prefer_taker"
    return "balanced"


def _current_policy(symbol: str, policy: Mapping[str, Any] | None, bounds: AutotuneBounds) -> Dict[str, Any]:
    policy = policy or {}
    maker_first = bool(policy.get("maker_first", True))
    label = str(policy.get("taker_bias") or policy.get("bias_label") or "").lower()
    bias_value = BIAS_LABEL_TO_VALUE.get(label)
    if bias_value is None:
        bias_value = _to_float(policy.get("taker_bias"))
    if bias_value is None:
        bias_value = 0.5
    bias_value = _clamp(float(bias_value), bounds.min_bias, bounds.max_bias)

    offset = maker_offset.BASELINE_BPS if hasattr(maker_offset, "BASELINE_BPS") else 2.0
    try:
        offset = float(maker_offset.suggest_maker_offset_bps(symbol))
    except Exception:
        pass
    offset = _clamp(offset, bounds.min_offset_bps, bounds.max_offset_bps)
    return {
        "maker_first": maker_first,
        "taker_bias": bias_value,
        "bias_label": label or _bias_label_from_value(bias_value),
        "offset_bps": offset,
    }


def _normalize_router_metrics(entry: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    return {
        "symbol": entry.get("symbol"),
        "maker_fill_rate": _to_float(entry.get("maker_fill_rate") or entry.get("maker_fill_ratio")),
        "fallback_rate": _to_float(entry.get("fallback_rate") or entry.get("fallback_ratio")),
        "slippage_p50": _to_float(entry.get("slippage_p50") or entry.get("slip_q50")),
        "slippage_p95": _to_float(entry.get("slippage_p95") or entry.get("slip_q95")),
        "ack_latency_p50_ms": _to_float(entry.get("latency_p50_ms") or entry.get("ack_latency_ms")),
    }


def _expectancy_strength(
    expectancy_entry: Mapping[str, Any] | None,
    score_entry: Mapping[str, Any] | None,
) -> Tuple[float, Optional[float], Optional[float]]:
    strength = 0.0
    expect_val = _to_float((expectancy_entry or {}).get("expectancy"))
    hit_rate = _to_float((expectancy_entry or {}).get("hit_rate"))
    score_val = _to_float((score_entry or {}).get("score"))

    if expect_val is not None:
        if expect_val >= 5.0:
            strength += 1.5
        elif expect_val >= 1.0:
            strength += 0.75
        elif expect_val <= -3.0:
            strength -= 1.5
        elif expect_val <= -1.0:
            strength -= 0.75
    if hit_rate is not None:
        if hit_rate >= 0.65:
            strength += 0.25
        elif hit_rate <= 0.45:
            strength -= 0.25
    if score_val is not None:
        if score_val >= 0.7:
            strength += 0.75
        elif score_val <= 0.3:
            strength -= 0.75
    return strength, expect_val, score_val


def _rationalize_expectancy(strength: float, expect_val: Optional[float], score_val: Optional[float]) -> Optional[str]:
    if strength >= 0.5 and (expect_val is not None or score_val is not None):
        parts = []
        if expect_val is not None:
            parts.append(f"expectancy={expect_val:.2f}")
        if score_val is not None:
            parts.append(f"score={score_val:.2f}")
        return f"Positive expectancy ({', '.join(parts)}) supports leaning maker."
    if strength <= -0.5 and (expect_val is not None or score_val is not None):
        parts = []
        if expect_val is not None:
            parts.append(f"expectancy={expect_val:.2f}")
        if score_val is not None:
            parts.append(f"score={score_val:.2f}")
        return f"Weak expectancy ({', '.join(parts)}) warrants taker caution."
    return None


def _propose_policy(
    symbol: str,
    router_metrics: Mapping[str, Any],
    current_policy: Mapping[str, Any],
    regime: str,
    quality: str,
    expectancy_entry: Mapping[str, Any],
    score_entry: Mapping[str, Any],
    bounds: AutotuneBounds,
) -> Tuple[Dict[str, Any], List[str]]:
    rationale: List[str] = [f"Router regime={regime}, quality={quality}."]
    bias_signal = 0.0
    offset_signal = 0.0
    maker_first = bool(current_policy.get("maker_first", True))

    fallback_rate = _to_float(router_metrics.get("fallback_rate")) or 0.0
    slip_p95 = _to_float(router_metrics.get("slippage_p95"))

    if quality == "broken" or regime == "broken":
        maker_first = False
        bias_signal += 1.0
        offset_signal += 1.0
        rationale.append("Router marked broken: disabling maker-first and widening offsets.")
    elif quality == "degraded":
        bias_signal += 0.5
        offset_signal += 0.5
        rationale.append("Router degraded: easing maker appetite.")

    if regime == "fallback_heavy":
        bias_signal += 0.75
        offset_signal += 0.5
        if fallback_rate >= 0.6 and not expectancy_entry:
            maker_first = False
            rationale.append("Fallback rate heavy with no positive expectancy: suggesting taker-first.")
        else:
            rationale.append(f"Fallback rate {fallback_rate:.0%} indicates taker-friendly routing.")
    elif regime == "slippage_hot":
        bias_signal += 0.5
        offset_signal += 0.75
        rationale.append(f"Slippage p95 at {slip_p95 or 0:.1f} bps: widen offsets.")
    elif regime == "maker_strong":
        bias_signal -= 0.75
        offset_signal -= 0.5
        rationale.append("Maker fills strong: can lean maker and tighten offset.")

    expectancy_strength, expect_val, score_val = _expectancy_strength(expectancy_entry, score_entry)
    if expectancy_strength >= 1.5:
        bias_signal -= 1.0
        offset_signal -= 0.5
    elif expectancy_strength >= 0.5:
        bias_signal -= 0.5
    elif expectancy_strength <= -1.5:
        bias_signal += 1.0
        offset_signal += 0.5
    elif expectancy_strength <= -0.5:
        bias_signal += 0.5

    exp_reason = _rationalize_expectancy(expectancy_strength, expect_val, score_val)
    if exp_reason:
        rationale.append(exp_reason)

    if not maker_first and regime == "maker_strong" and expectancy_strength >= 0.5:
        maker_first = True
        rationale.append("Re-enabling maker-first with strong expectancy and maker regime.")

    bias_delta = _clamp(bias_signal, -1.0, 1.0) * bounds.max_bias_step
    offset_delta = _clamp(offset_signal, -1.0, 1.0) * bounds.max_offset_step_bps
    base_bias = float(current_policy.get("taker_bias") or 0.5)
    base_offset = float(current_policy.get("offset_bps") or maker_offset.BASELINE_BPS)
    new_bias = _clamp(base_bias + bias_delta, bounds.min_bias, bounds.max_bias)
    new_offset = _clamp(base_offset + offset_delta, bounds.min_offset_bps, bounds.max_offset_bps)

    proposed_policy = {
        "maker_first": maker_first,
        "taker_bias": new_bias,
        "bias_label": _bias_label_from_value(new_bias),
        "offset_bps": new_offset,
    }
    return proposed_policy, rationale


def _build_symbol_maps(
    expectancy_snapshot: Mapping[str, Any],
    scores_snapshot: Mapping[str, Any],
) -> Tuple[Dict[str, Mapping[str, Any]], Dict[str, Mapping[str, Any]]]:
    exp_data: Dict[str, Mapping[str, Any]] = {}
    raw_symbols = expectancy_snapshot.get("symbols")
    if isinstance(raw_symbols, Mapping):
        for sym, entry in raw_symbols.items():
            if not sym:
                continue
            exp_data[str(sym).upper()] = entry if isinstance(entry, Mapping) else {}

    scores_map: Dict[str, Mapping[str, Any]] = {}
    rows = scores_snapshot.get("symbols")
    if isinstance(rows, Iterable) and not isinstance(rows, (str, bytes)):
        for entry in rows:
            if not isinstance(entry, Mapping):
                continue
            symbol = entry.get("symbol")
            if not symbol:
                continue
            scores_map[str(symbol).upper()] = entry
    return exp_data, scores_map


def build_suggestions(
    expectancy_snapshot: Mapping[str, Any] | None = None,
    symbol_scores_snapshot: Mapping[str, Any] | None = None,
    router_health_snapshot: Mapping[str, Any] | None = None,
    risk_config: Mapping[str, Any] | None = None,
    *,
    state_dir: Path | str | None = None,
    risk_config_path: Path | str | None = None,
    lookback_days: float = DEFAULT_LOOKBACK_DAYS,
) -> Dict[str, Any]:
    """
    Build router policy suggestions from pre-computed intel snapshots.
    """
    state_dir_path = Path(state_dir or DEFAULT_STATE_DIR)
    rx_path = Path(risk_config_path or DEFAULT_RISK_LIMITS_PATH)

    if expectancy_snapshot is None:
        expectancy_snapshot = expectancy_v6.load_expectancy(state_dir_path / "expectancy_v6.json")
    if symbol_scores_snapshot is None:
        symbol_scores_snapshot = _load_json_if_exists(state_dir_path / "symbol_scores_v6.json")
    if router_health_snapshot is None:
        router_health_snapshot = _load_json_if_exists(state_dir_path / "router_health.json")
    if risk_config is None:
        risk_config = load_json(str(rx_path))

    exp_map, score_map = _build_symbol_maps(expectancy_snapshot or {}, symbol_scores_snapshot or {})
    bounds = _resolve_bounds(risk_config if isinstance(risk_config, Mapping) else {})
    router_symbols = router_health_snapshot.get("symbols") if isinstance(router_health_snapshot, Mapping) else None
    if not isinstance(router_symbols, list):
        return {"generated_ts": _iso_now(), "lookback_days": float(lookback_days), "symbols": []}

    suggestions: List[Dict[str, Any]] = []
    for entry in router_symbols:
        if not isinstance(entry, Mapping):
            continue
        symbol = str(entry.get("symbol") or "").upper()
        if not symbol:
            continue
        router_metrics = _normalize_router_metrics(entry)
        regime = classify_router_regime(router_metrics)
        quality = classify_router_quality(
            router_metrics,
            slip_p95_bps=router_metrics.get("slippage_p95"),
            fallback_rate=router_metrics.get("fallback_rate"),
            latency_ms=router_metrics.get("ack_latency_p50_ms"),
        )
        current_policy = _current_policy(symbol, entry.get("policy") if isinstance(entry.get("policy"), Mapping) else None, bounds)
        expectancy_entry = exp_map.get(symbol, {})
        score_entry = score_map.get(symbol, {})
        proposed_policy, rationale = _propose_policy(
            symbol,
            router_metrics,
            current_policy,
            regime,
            quality,
            expectancy_entry,
            score_entry,
            bounds,
        )
        metrics_payload = {
            "fill_rate": router_metrics.get("maker_fill_rate"),
            "fallback_rate": router_metrics.get("fallback_rate"),
            "slip_p50_bps": router_metrics.get("slippage_p50"),
            "slip_p95_bps": router_metrics.get("slippage_p95"),
            "latency_p50_ms": router_metrics.get("ack_latency_p50_ms"),
        }
        suggestion = {
            "symbol": symbol,
            "regime": regime,
            "quality": quality,
            "metrics": metrics_payload,
            "current_policy": current_policy,
            "proposed_policy": proposed_policy,
            "rationale": rationale or ["No substantial intel change; keeping stance."],
        }
        suggestions.append(suggestion)

    suggestions.sort(key=lambda item: item["symbol"])
    return {
        "generated_ts": _iso_now(),
        "lookback_days": float(lookback_days),
        "symbols": suggestions,
    }


__all__ = ["build_suggestions"]
