"""Expectancy analytics for v6.0 using canonical telemetry."""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional


DEFAULT_LOG_DIR = Path(os.getenv("HEDGE_LOG_DIR") or "logs")
DEFAULT_FILLS_PATH = DEFAULT_LOG_DIR / "execution" / "orders_executed.jsonl"
DEFAULT_ROUTER_METRICS_PATH = DEFAULT_LOG_DIR / "execution" / "order_metrics.jsonl"
DEFAULT_NAV_STATE_PATH = DEFAULT_LOG_DIR / "state" / "nav.json"


def _parse_timestamp(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 1e12:
            ts /= 1000.0
        return ts
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            from datetime import datetime

            try:
                if text.endswith("Z"):
                    text = text[:-1] + "+00:00"
                dt = datetime.fromisoformat(text)
                if dt.tzinfo is None:
                    from datetime import timezone

                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.timestamp()
            except Exception:  # pragma: no cover - defensive
                return None
    return None


def _read_jsonl(path: Path, limit: int = 5000) -> List[Dict[str, Any]]:
    if not path.exists() or limit <= 0:
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, MutableMapping):
                    rows.append(dict(payload))
    except Exception:  # pragma: no cover - filesystem noise
        return []
    if len(rows) > limit:
        rows = rows[-limit:]
    return rows


def load_trade_records(
    path: Path | str = DEFAULT_FILLS_PATH,
    lookback_hours: float = 24.0,
) -> List[Dict[str, Any]]:
    rows = _read_jsonl(Path(path))
    if not rows:
        return []
    cutoff = time.time() - max(lookback_hours, 0.0) * 3600.0
    trades: List[Dict[str, Any]] = []
    for row in rows:
        event = str(row.get("event_type") or row.get("event") or "").lower()
        if event not in {"order_close", "trade_close", "position_close"}:
            continue
        pnl = row.get("realizedPnlUsd") or row.get("pnl_usd")
        if pnl is None:
            continue
        ts = _parse_timestamp(row.get("ts_close") or row.get("ts"))
        if ts is not None and ts < cutoff:
            continue
        trade = {
            "symbol": str(row.get("symbol") or "").upper(),
            "pnl_usd": float(pnl),
            "ts": ts,
            "strategy": row.get("strategy") or row.get("strategy_id"),
            "attempt_id": row.get("attempt_id") or row.get("intent_id"),
        }
        trades.append(trade)
    return trades


def load_router_metrics(
    path: Path | str = DEFAULT_ROUTER_METRICS_PATH,
    lookback_hours: float = 24.0,
) -> Dict[str, Dict[str, Any]]:
    path = Path(path)
    rows = _read_jsonl(path)
    if not rows:
        return {}
    cutoff = time.time() - max(lookback_hours, 0.0) * 3600.0
    metrics: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        ts = _parse_timestamp(row.get("ts"))
        if ts is not None and ts < cutoff:
            continue
        attempt = row.get("attempt_id") or row.get("attemptId")
        if not attempt:
            continue
        payload = {
            "policy": row.get("policy"),
            "slippage_bps": row.get("slippage_bps"),
            "maker_start": row.get("maker_start") or row.get("started_maker"),
            "is_maker_final": row.get("is_maker_final"),
            "used_fallback": row.get("used_fallback"),
        }
        metrics[str(attempt)] = payload
    return metrics


def merge_trades_with_policy(
    trades: List[Dict[str, Any]], metrics: Mapping[str, Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    if not trades or not metrics:
        return trades
    for trade in trades:
        attempt = trade.get("attempt_id")
        if attempt and attempt in metrics:
            trade["router_policy"] = metrics[attempt].get("policy")
            trade["router_metric"] = metrics[attempt]
    return trades


def _max_drawdown(pnls: Iterable[float]) -> float:
    peak = 0.0
    drawdown = 0.0
    cumulative = 0.0
    for pnl in pnls:
        cumulative += pnl
        peak = max(peak, cumulative)
        drawdown = max(drawdown, peak - cumulative)
    return drawdown


# Minimum trades before expectancy statistics are considered meaningful.
# Below this threshold, return neutral values to prevent sparse-data
# distortion in intel scoring and operator dashboards.
MIN_EXPECTANCY_TRADES: int = 30


def _expectancy_stats(
    records: List[Dict[str, Any]],
    min_trades: int = MIN_EXPECTANCY_TRADES,
) -> Dict[str, Any]:
    pnls = [float(r.get("pnl_usd") or 0.0) for r in records]
    n = len(pnls)
    if n == 0:
        return {
            "count": 0,
            "hit_rate": None,
            "avg_return": None,
            "expectancy": None,
            "expectancy_per_risk": None,
            "drawdown_penalty": None,
            "is_mature": False,
        }
    # Presence ≠ validity: refuse to compute meaningful expectancy
    # from fewer than min_trades observations.
    if n < min_trades:
        return {
            "count": n,
            "hit_rate": None,
            "avg_return": None,
            "avg_win": None,
            "avg_loss": None,
            "expectancy": 0.0,
            "expectancy_per_risk": 0.0,
            "max_drawdown": 0.0,
            "drawdown_adjusted": 0.0,
            "is_mature": False,
        }
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    avg_return = sum(pnls) / n
    hit_rate = len(wins) / n if n else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    expectancy = hit_rate * avg_win + (1 - hit_rate) * avg_loss
    risk_unit = abs(avg_loss) if avg_loss else 1.0
    expectancy_per_risk = expectancy / risk_unit
    drawdown = _max_drawdown(pnls)
    dd_adjusted = expectancy - (drawdown / max(n, 1))
    return {
        "count": n,
        "hit_rate": hit_rate,
        "avg_return": avg_return,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "expectancy_per_risk": expectancy_per_risk,
        "max_drawdown": drawdown,
        "drawdown_adjusted": dd_adjusted,
        "is_mature": True,
    }


def _group_records(trades: Iterable[Dict[str, Any]], key: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not key:
        return {}
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for trade in trades:
        value = trade.get(key)
        if not value:
            continue
        grouped.setdefault(str(value), []).append(trade)
    return {k: _expectancy_stats(v) for k, v in grouped.items()}


def compute_symbol_expectancy(trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return _group_records(trades, "symbol")


def compute_hourly_expectancy(trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for trade in trades:
        ts = trade.get("ts")
        if ts is None:
            continue
        hour = datetime.utcfromtimestamp(ts).hour
        groups.setdefault(str(hour), []).append(trade)
    return {k: _expectancy_stats(v) for k, v in groups.items()}


def compute_regime_expectancy(trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for trade in trades:
        for tag in _regime_tags(trade.get("router_metric")):
            groups.setdefault(tag, []).append(trade)
    return {k: _expectancy_stats(v) for k, v in groups.items()}


def _regime_tags(metric: Optional[Mapping[str, Any]]) -> List[str]:
    if not isinstance(metric, Mapping):
        return []
    tags: List[str] = []
    if metric.get("is_maker_final"):
        tags.append("maker_success")
    elif metric.get("maker_start"):
        tags.append("maker_attempt")
    if metric.get("used_fallback"):
        tags.append("fallback")
    slip = metric.get("slippage_bps")
    if slip is not None:
        try:
            slip_val = float(slip)
        except Exception:
            slip_val = 0.0
        if slip_val <= 1.0:
            tags.append("slip_low")
        elif slip_val <= 5.0:
            tags.append("slip_med")
        else:
            tags.append("slip_high")
    policy = metric.get("policy")
    if isinstance(policy, Mapping):
        if policy.get("quality"):
            tags.append(f"policy_{policy['quality']}")
        if policy.get("taker_bias"):
            tags.append(f"bias_{policy['taker_bias']}")
    return tags


def load_inputs(
    log_dir: Path | str = DEFAULT_LOG_DIR,
    lookback_days: float = 2.0,
) -> Dict[str, Any]:
    log_dir = Path(log_dir)
    lookback_hours = float(lookback_days) * 24.0
    trades = load_trade_records(log_dir / "execution" / "orders_executed.jsonl", lookback_hours)
    router_metrics = load_router_metrics(log_dir / "execution" / "order_metrics.jsonl", lookback_hours)
    trades = merge_trades_with_policy(trades, router_metrics)
    nav_snapshot = _load_nav_snapshot(log_dir / "state" / "nav.json")
    return {
        "trades": trades,
        "lookback_hours": lookback_hours,
        "metadata": {"nav_snapshot": nav_snapshot},
    }


def _load_nav_snapshot(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        return {}
    return {}


def build_expectancy(
    table_inputs: Mapping[str, Any],
    prior_inputs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build expectancy snapshot, blending priors for symbols below MIN_EXPECTANCY_TRADES.

    Args:
        table_inputs: Standard inputs dict with ``trades``, ``lookback_hours``, ``metadata``.
        prior_inputs: Optional observable inputs for the Bayesian prior.
            Expected keys: ``regime``, ``regime_confidence``, ``vol_state``,
            ``router_health_score``, ``trend_score``, ``carry_score``.
            When ``None`` or empty, symbols with insufficient episodes
            get neutral expectancy (backward-compatible).
    """
    trades = list(table_inputs.get("trades") or [])
    lookback_hours = float(table_inputs.get("lookback_hours") or 0.0)
    symbol_stats = compute_symbol_expectancy(trades)

    # If prior_inputs are provided, enrich immature symbols
    if prior_inputs and isinstance(prior_inputs, Mapping):
        prior = compute_expectancy_prior(
            regime=prior_inputs.get("regime"),
            regime_confidence=float(prior_inputs.get("regime_confidence") or 0.5),
            vol_state=prior_inputs.get("vol_state"),
            router_health_score=float(prior_inputs.get("router_health_score") or 0.5),
            trend_score=float(prior_inputs.get("trend_score") or 0.5),
            carry_score=float(prior_inputs.get("carry_score") or 0.5),
        )
        for sym, stats in symbol_stats.items():
            count = int(stats.get("count") or 0)
            if count < MIN_EXPECTANCY_TRADES:
                symbol_stats[sym] = blend_expectancy(prior, stats, episode_count=count)
        # Ensure prior metadata is surfaced for diagnostics
        for sym in (prior_inputs.get("symbols") or []):
            sym = str(sym).upper()
            if sym and sym not in symbol_stats:
                symbol_stats[sym] = dict(prior)

    agora = {
        "symbols": symbol_stats,
        "hours": compute_hourly_expectancy(trades),
        "regimes": compute_regime_expectancy(trades),
        "sample_count": len(trades),
        "lookback_hours": lookback_hours,
        "updated_ts": time.time(),
    }
    meta = table_inputs.get("metadata")
    if isinstance(meta, Mapping):
        agora["metadata"] = dict(meta)
    if prior_inputs:
        agora["has_prior"] = True
    return agora


def build_expectancy_snapshot(
    trades: List[Dict[str, Any]],
    lookback_hours: float,
    prior_inputs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return build_expectancy(
        {"trades": trades, "lookback_hours": lookback_hours},
        prior_inputs=prior_inputs,
    )


def save_expectancy(path: Path | str, snapshot: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(snapshot, handle)
    tmp.replace(path)


def load_expectancy(path: Path | str) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Bayesian Expectancy Prior (v7.9)
# ---------------------------------------------------------------------------
# Produces non-degenerate expectancy from observable inputs only.
# This eliminates the cold-start gap: expectancy_score exists with
# zero trade episodes and varies with regime / vol / router health.

# Regime → base prior (before vol/router adjustment).
# TREND_UP has positive tilt; CRISIS has negative tilt.
_REGIME_PRIOR: Dict[str, float] = {
    "TREND_UP": 0.08,
    "TREND_DOWN": -0.02,
    "MEAN_REVERT": 0.03,
    "BREAKOUT": 0.06,
    "CHOPPY": -0.04,
    "CRISIS": -0.12,
}

# Vol state → additive adjustment.
_VOL_PRIOR_ADJ: Dict[str, float] = {
    "low": 0.02,
    "normal": 0.0,
    "high": -0.03,
    "crisis": -0.06,
}

# Minimum shrinkage weight on the prior even with mature observations.
_MIN_PRIOR_WEIGHT: float = 0.10
# Number of episodes at which posterior is fully mature.
_MATURITY_EPISODES: int = 60


def compute_expectancy_prior(
    regime: Optional[str] = None,
    regime_confidence: float = 0.5,
    vol_state: Optional[str] = None,
    router_health_score: float = 0.5,
    trend_score: float = 0.5,
    carry_score: float = 0.5,
) -> Dict[str, Any]:
    """
    Produce expectancy from observable inputs only (no trade records).

    Args:
        regime: Sentinel-X primary regime label.
        regime_confidence: Confidence ∈ [0, 1] of the regime classification.
        vol_state: Volatility state label ("low", "normal", "high", "crisis").
        router_health_score: Router health ∈ [0, 1] (0 = poor, 1 = excellent).
        trend_score: Trend score ∈ [0, 1] (0.5 = neutral).
        carry_score: Carry score ∈ [0, 1] (0.5 = neutral).

    Returns:
        Dict with ``expectancy``, ``expectancy_per_risk``, ``is_prior``, and components.
    """
    # 1. Regime base prior (scaled by confidence)
    conf = max(0.0, min(1.0, regime_confidence))
    base = _REGIME_PRIOR.get((regime or "").upper(), 0.0) * conf

    # 2. Vol-state adjustment
    vol_adj = _VOL_PRIOR_ADJ.get((vol_state or "normal").lower(), 0.0)

    # 3. Router health contribution: good router → boosts prior
    #    Map [0, 1] to [-0.04, +0.04]
    router_adj = (router_health_score - 0.5) * 0.08

    # 4. Alignment signals: trend + carry as directional confirmation
    #    Each is [0, 1]; remap to [-0.03, +0.03]
    trend_adj = (trend_score - 0.5) * 0.06
    carry_adj = (carry_score - 0.5) * 0.04

    expectancy = base + vol_adj + router_adj + trend_adj + carry_adj

    # Simple risk-unit mapping (assume unit risk = $1):
    # expectancy_per_risk ≈ expectancy clamped
    expectancy_per_risk = max(-1.0, min(1.0, expectancy * 10.0))

    return {
        "count": 0,
        "expectancy": round(expectancy, 6),
        "expectancy_per_risk": round(expectancy_per_risk, 6),
        "hit_rate": None,
        "avg_return": None,
        "avg_win": None,
        "avg_loss": None,
        "max_drawdown": 0.0,
        "drawdown_adjusted": round(expectancy, 6),
        "is_mature": False,
        "is_prior": True,
        "prior_components": {
            "regime_base": round(base, 6),
            "vol_adj": round(vol_adj, 6),
            "router_adj": round(router_adj, 6),
            "trend_adj": round(trend_adj, 6),
            "carry_adj": round(carry_adj, 6),
        },
        "prior_inputs": {
            "regime": regime,
            "regime_confidence": round(conf, 4),
            "vol_state": vol_state,
            "router_health_score": round(router_health_score, 4),
            "trend_score": round(trend_score, 4),
            "carry_score": round(carry_score, 4),
        },
    }


def blend_expectancy(
    prior: Mapping[str, Any],
    posterior: Mapping[str, Any],
    episode_count: int = 0,
    maturity_n: int = _MATURITY_EPISODES,
) -> Dict[str, Any]:
    """
    Bayesian shrinkage between prior and posterior expectancy.

    When episode_count == 0 the prior is returned as-is.
    As episodes grow toward *maturity_n*, posterior weight increases.
    Even at full maturity a minimum prior weight is retained.

    Returns a merged expectancy dict.
    """
    if episode_count <= 0 or posterior.get("expectancy") is None:
        return dict(prior)

    # Shrinkage weight on prior (starts at 1, decays toward _MIN_PRIOR_WEIGHT)
    t = min(episode_count / max(maturity_n, 1), 1.0)
    prior_w = max(_MIN_PRIOR_WEIGHT, 1.0 - t * (1.0 - _MIN_PRIOR_WEIGHT))
    post_w = 1.0 - prior_w

    prior_exp = float(prior.get("expectancy") or 0.0)
    post_exp = float(posterior.get("expectancy") or 0.0)
    blended_exp = prior_w * prior_exp + post_w * post_exp

    prior_epr = float(prior.get("expectancy_per_risk") or 0.0)
    post_epr = float(posterior.get("expectancy_per_risk") or 0.0)
    blended_epr = prior_w * prior_epr + post_w * post_epr

    return {
        "count": episode_count,
        "expectancy": round(blended_exp, 6),
        "expectancy_per_risk": round(blended_epr, 6),
        "hit_rate": posterior.get("hit_rate"),
        "avg_return": posterior.get("avg_return"),
        "avg_win": posterior.get("avg_win"),
        "avg_loss": posterior.get("avg_loss"),
        "max_drawdown": posterior.get("max_drawdown", 0.0),
        "drawdown_adjusted": posterior.get("drawdown_adjusted", 0.0),
        "is_mature": posterior.get("is_mature", False),
        "is_prior": False,
        "blend_weights": {
            "prior": round(prior_w, 4),
            "posterior": round(post_w, 4),
            "episode_count": episode_count,
            "maturity_n": maturity_n,
        },
    }


__all__ = [
    "load_trade_records",
    "load_router_metrics",
    "merge_trades_with_policy",
    "load_inputs",
    "compute_symbol_expectancy",
    "compute_hourly_expectancy",
    "compute_regime_expectancy",
    "build_expectancy",
    "build_expectancy_snapshot",
    "save_expectancy",
    "load_expectancy",
    "compute_expectancy_prior",
    "blend_expectancy",
    "MIN_EXPECTANCY_TRADES",
]
