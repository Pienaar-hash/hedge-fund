from __future__ import annotations

from typing import Any, Mapping

from execution.strategy_adaptation import adaptive_sizing as _adaptive_sizing
from execution.utils.vol import atr_pct, rolling_sigma

MAX_VOL_SCALE = 100.0  # cap inverse-vol multiplier to avoid blow-ups when sigma is tiny
MIN_SIGMA = 1e-3

ATR_MED_LOOKBACK = 500


def inverse_vol_size(symbol: str, base_size: float, lookback: int = 50) -> float:
    sigma = rolling_sigma(symbol, lookback=lookback) or 0.0
    sigma = max(float(sigma), MIN_SIGMA)
    inv_scale = 1.0 / sigma
    inv_scale = min(inv_scale, MAX_VOL_SCALE)
    return float(base_size) * inv_scale


def volatility_regime_scale(symbol: str) -> float:
    """
    Simple ATR regime classifier to taper sizing during volatile regimes.
    """
    atr_now = atr_pct(symbol, lookback_bars=50)
    atr_med = atr_pct(symbol, lookback_bars=ATR_MED_LOOKBACK)
    if atr_med <= 0:
        return 1.0
    ratio = atr_now / atr_med
    if ratio < 0.7:
        return 0.75
    if ratio < 1.5:
        return 1.0
    if ratio < 2.5:
        return 0.5
    return 0.25


def adaptive_sizing(
    symbol: str,
    gross_usd: float,
    atr_regime: Any,
    dd_regime: Any,
    risk_mode: Any,
    overrides: Mapping[str, Any] | None = None,
) -> tuple[float, float]:
    """Proxy to the shared strategy adaptation helper."""
    return _adaptive_sizing(symbol, gross_usd, atr_regime, dd_regime, risk_mode, overrides)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        text = str(value)
    except Exception:
        return default
    return text


def _safe_number(value: Any, default: float = 0.0) -> float:
    try:
        num = float(value)
        if num != num:
            return default
        return num
    except Exception:
        return default


def compute_strategy_performance_factor(
    strategy_id: str,
    pnl_attr: Mapping[str, Any],
    performance_rules: Mapping[str, Any] | None = None,
) -> float:
    """
    Compute performance factor from per-strategy win rate.
    """
    perf = 1.0
    per_strategy = pnl_attr.get("per_strategy") if isinstance(pnl_attr, Mapping) else {}
    entry = per_strategy.get(strategy_id) if isinstance(per_strategy, Mapping) else {}
    if not isinstance(entry, Mapping):
        return perf
    wins = entry.get("wins")
    trade_count = entry.get("trade_count")
    win_rate = entry.get("win_rate") or entry.get("winrate") or entry.get("win_rate_pct")
    try:
        if win_rate is None and wins is not None and trade_count:
            win_rate = float(wins) / float(trade_count)
        elif win_rate is not None and win_rate > 1:
            win_rate = float(win_rate) / 100.0
        elif win_rate is not None:
            win_rate = float(win_rate)
    except Exception:
        win_rate = None
    if win_rate is None:
        return perf
    perf_cfg = performance_rules if isinstance(performance_rules, Mapping) else {}
    low_thresh = _safe_number(perf_cfg.get("winrate_low_threshold"), 0.40)
    high_thresh = _safe_number(perf_cfg.get("winrate_high_threshold"), 0.60)
    low_factor = _safe_number(perf_cfg.get("low_factor"), 0.7)
    high_factor = _safe_number(perf_cfg.get("high_factor"), 1.1)
    if win_rate < low_thresh:
        return low_factor
    if win_rate > high_thresh:
        return high_factor
    return perf


def compute_adaptive_weight(
    strategy_id: str,
    regimes: Mapping[str, Any] | None,
    risk_snapshot: Mapping[str, Any] | None,
    pnl_attr: Mapping[str, Any] | None,
    strategy_cfg: Mapping[str, Any] | None,
) -> float:
    cfg = strategy_cfg or {}
    base_weight = float(cfg.get("base_weight", 1.0))
    if cfg.get("adaptive_enabled") is False:
        return base_weight
    min_weight = float(cfg.get("min_weight", 0.0))
    max_weight = float(cfg.get("max_weight", 2.0))
    risk_mode_rules = cfg.get("risk_mode_rules") if isinstance(cfg.get("risk_mode_rules"), Mapping) else {}
    atr_rules = cfg.get("atr_rules") if isinstance(cfg.get("atr_rules"), Mapping) else {}
    dd_rules = cfg.get("dd_rules") if isinstance(cfg.get("dd_rules"), Mapping) else {}
    perf_rules = cfg.get("performance_rules") if isinstance(cfg.get("performance_rules"), Mapping) else {}

    def _rule_lookup(rule_map: Mapping[str, Any], key: str, default: float = 1.0) -> float:
        try:
            return float(rule_map.get(key, default))
        except Exception:
            return default

    risk_mode = _safe_str((risk_snapshot or {}).get("risk_mode"), "OK").upper()
    risk_factor = _rule_lookup(
        risk_mode_rules,
        risk_mode,
        default={"OK": 1.0, "WARN": 0.8, "DEFENSIVE": 0.5, "HALTED": 0.0}.get(risk_mode, 1.0),
    )

    atr_idx = _safe_int((regimes or {}).get("atr_regime"), 0)
    dd_idx = _safe_int((regimes or {}).get("dd_regime"), 0)
    atr_factor = _rule_lookup(
        atr_rules,
        str(atr_idx),
        default={0: 1.0, 1: 1.0, 2: 0.7, 3: 0.5}.get(atr_idx, 1.0),
    )
    dd_factor = _rule_lookup(
        dd_rules,
        str(dd_idx),
        default={0: 1.0, 1: 0.8, 2: 0.5, 3: 0.0}.get(dd_idx, 1.0),
    )

    perf_factor = compute_strategy_performance_factor(strategy_id, pnl_attr or {}, performance_rules=perf_rules)

    final = base_weight * risk_factor * atr_factor * dd_factor * perf_factor
    return _clamp(final, min_weight, max_weight)
