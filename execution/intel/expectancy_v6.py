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


def _expectancy_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
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


def build_expectancy(table_inputs: Mapping[str, Any]) -> Dict[str, Any]:
    trades = list(table_inputs.get("trades") or [])
    lookback_hours = float(table_inputs.get("lookback_hours") or 0.0)
    agora = {
        "symbols": compute_symbol_expectancy(trades),
        "hours": compute_hourly_expectancy(trades),
        "regimes": compute_regime_expectancy(trades),
        "sample_count": len(trades),
        "lookback_hours": lookback_hours,
        "updated_ts": time.time(),
    }
    meta = table_inputs.get("metadata")
    if isinstance(meta, Mapping):
        agora["metadata"] = dict(meta)
    return agora


def build_expectancy_snapshot(trades: List[Dict[str, Any]], lookback_hours: float) -> Dict[str, Any]:
    return build_expectancy({"trades": trades, "lookback_hours": lookback_hours})


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
]
