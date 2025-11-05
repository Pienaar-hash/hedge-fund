from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

DEFAULT_OUTPUT_PATH = Path("logs/cache/capital_allocation.json")

__all__ = [
    "DEFAULT_OUTPUT_PATH",
    "StrategyStats",
    "AllocationConfig",
    "CapitalAllocation",
    "compute_strategy_stats",
    "compute_allocation",
    "persist_allocation",
]


@dataclass(slots=True)
class StrategyStats:
    """Container for summary statistics derived from a strategy's return stream."""

    sharpe: float
    volatility: float
    drawdown: Optional[float] = None
    sample_size: int = 0


@dataclass(slots=True)
class AllocationConfig:
    """Configuration toggles for the dynamic allocator."""

    volatility_floor: float = 1e-3
    min_positive_sharpe: float = 0.25
    neutral_sharpe: float = 1.0
    correlation_floor: float = 0.05
    floor_weight: float = 0.0
    cap_weight: float = 0.6
    decay: float = 0.2
    min_strategies: int = 1


@dataclass(slots=True)
class CapitalAllocation:
    """Final allocation package written to disk and consumed by the router."""

    weights: Dict[str, float]
    scores: Dict[str, float]
    metadata: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Mapping[str, object]:
        return {
            "weights": {k: float(v) for k, v in self.weights.items()},
            "scores": {k: float(v) for k, v in self.scores.items()},
            "metadata": {k: float(v) for k, v in self.metadata.items()},
        }


def compute_strategy_stats(
    returns: Sequence[float] | pd.Series,
    *,
    annualization: int = 252,
    min_periods: int = 20,
) -> StrategyStats:
    series = pd.Series(list(returns), dtype=float).dropna()
    sample_size = int(series.size)
    if sample_size < max(2, min_periods):
        return StrategyStats(sharpe=0.0, volatility=0.0, drawdown=None, sample_size=sample_size)
    mean = float(series.mean())
    std = float(series.std(ddof=1))
    sharpe = 0.0 if std == 0.0 else (mean / std) * math.sqrt(float(annualization))
    cumulative = (1.0 + series).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = float((cumulative / rolling_max - 1.0).min())
    return StrategyStats(sharpe=sharpe, volatility=std, drawdown=drawdown, sample_size=sample_size)


def _normalize_correlation_matrix(matrix: pd.DataFrame, strategies: Iterable[str]) -> pd.DataFrame:
    strategies = list(dict.fromkeys(strategies))
    if matrix.empty:
        return pd.DataFrame(np.eye(len(strategies)), index=strategies, columns=strategies)
    matrix = matrix.reindex(index=strategies, columns=strategies).fillna(0.0)
    matrix = matrix.clip(lower=-1.0, upper=1.0)
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix.values, 1.0)
    return matrix


def _correlation_penalty(matrix: pd.DataFrame, strategy: str) -> float:
    row = matrix.loc[strategy]
    if len(row) <= 1:
        return 1.0
    off_diag = row.drop(labels=[strategy], errors="ignore").to_numpy(dtype=float)
    if off_diag.size == 0:
        return 1.0
    positives = np.maximum(off_diag, 0.0)
    penalty = 1.0 - float(np.mean(positives))
    return max(0.0, penalty)


def compute_allocation(
    stats: Mapping[str, StrategyStats],
    correlations: pd.DataFrame | Mapping[str, Mapping[str, float]],
    *,
    config: Optional[AllocationConfig] = None,
) -> CapitalAllocation:
    if not stats:
        raise ValueError("no strategy statistics provided")
    config = config or AllocationConfig()
    strategies = list(stats.keys())
    matrix = pd.DataFrame(correlations) if isinstance(correlations, dict) else correlations
    matrix = _normalize_correlation_matrix(matrix, strategies)

    scores: Dict[str, float] = {}
    for name, metric in stats.items():
        sharpe = float(metric.sharpe)
        if sharpe <= 0.0:
            # dampen strategies with negative Sharpe depending on how negative they are
            sharpe = max(0.0, sharpe + config.decay)
        positive_sharpe = max(0.0, sharpe - config.min_positive_sharpe)
        if config.neutral_sharpe > 0:
            normalized_sharpe = positive_sharpe / config.neutral_sharpe
        else:
            normalized_sharpe = positive_sharpe

        volatility = max(config.volatility_floor, float(metric.volatility))
        inverse_vol = 1.0 / volatility
        corr_penalty = max(config.correlation_floor, _correlation_penalty(matrix, name))
        score = max(0.0, normalized_sharpe * inverse_vol * corr_penalty)
        scores[name] = score

    total_score = sum(scores.values())
    if total_score <= 0.0:
        equal_weight = 1.0 / max(len(strategies), 1)
        weights = {name: equal_weight for name in strategies}
    else:
        weights = {name: max(0.0, score) / total_score for name, score in scores.items()}

    min_active = max(1, config.min_strategies)
    epsilon = max(1e-6, config.floor_weight if config.floor_weight > 0 else 1e-4)
    active_names = [name for name, value in weights.items() if value > 0.0]
    if len(active_names) < min_active:
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        for name, _score in ranked[:min_active]:
            weights[name] = max(weights.get(name, 0.0), epsilon)

    def _apply_floor(current: Dict[str, float]) -> Dict[str, float]:
        floor = config.floor_weight
        if floor <= 0.0:
            return current
        adjusted = dict(current)
        shortages: Dict[str, float] = {}
        surplus: Dict[str, float] = {}
        for name, value in adjusted.items():
            if value <= 0.0:
                continue
            if value < floor:
                shortages[name] = floor - value
                adjusted[name] = floor
            elif value > floor:
                surplus[name] = value - floor
        total_shortage = sum(shortages.values())
        total_surplus = sum(surplus.values())
        if total_shortage > 0.0 and total_surplus > 0.0:
            for name in surplus:
                share = surplus[name] / total_surplus
                reduction = total_shortage * share
                adjusted[name] = max(floor, adjusted[name] - reduction)
        return adjusted

    def _apply_cap(current: Dict[str, float]) -> Dict[str, float]:
        cap = config.cap_weight
        if not (0.0 < cap < 1.0):
            return current
        adjusted = dict(current)
        excess: Dict[str, float] = {}
        headroom: Dict[str, float] = {}
        for name, value in adjusted.items():
            if value > cap:
                excess[name] = value - cap
                adjusted[name] = cap
            else:
                headroom[name] = cap - value
        total_excess = sum(excess.values())
        total_headroom = sum(headroom.values())
        if total_excess > 0.0 and total_headroom > 0.0:
            for name in headroom:
                share = headroom[name] / total_headroom
                adjusted[name] += total_excess * share
        return adjusted

    weights = _apply_floor(weights)
    weights = _apply_cap(weights)

    weight_sum = sum(weights.values())
    if weight_sum <= 0.0:
        weights = {name: 1.0 / len(weights) for name in weights} if weights else {}
    else:
        weights = {name: max(0.0, value) / weight_sum for name, value in weights.items()}

    metadata = {
        "average_abs_correlation": float(np.mean(np.abs(matrix.to_numpy(dtype=float) - np.eye(len(matrix))))),
        "strategies": float(len(weights)),
    }
    return CapitalAllocation(weights=weights, scores=scores, metadata=metadata)


def persist_allocation(allocation: CapitalAllocation, *, path: Path = DEFAULT_OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(allocation.to_dict(), indent=2, sort_keys=True))


def from_returns_frame(
    returns: pd.DataFrame,
    *,
    config: Optional[AllocationConfig] = None,
    method: str = "pearson",
    window: int = 60,
    min_periods: Optional[int] = None,
) -> CapitalAllocation:
    if returns.empty:
        raise ValueError("returns frame is empty")
    stats = {col: compute_strategy_stats(returns[col]) for col in returns.columns}
    lookback = max(5, min(window, len(returns)))
    min_periods = min_periods or lookback
    method_lower = method.lower()
    if method_lower == "pearson":
        base = returns
    elif method_lower == "spearman":
        base = returns.rank(axis=0, method="average", na_option="keep")
    else:
        raise ValueError(f"unsupported correlation method: {method}")
    corr = base.rolling(window=window, min_periods=min_periods).corr()
    if corr.empty:
        raise ValueError("insufficient observations to compute correlation matrix")
    latest_ts = corr.index.get_level_values(0).max()
    matrix = corr.loc[latest_ts].reindex(index=returns.columns, columns=returns.columns).fillna(0.0)
    return compute_allocation(stats, matrix, config=config)


def main() -> int:  # pragma: no cover - CLI helper
    import argparse

    parser = argparse.ArgumentParser(description="Compute capital allocations from strategy statistics.")
    parser.add_argument("source", type=str, help="CSV file or directory with strategy return series.")
    parser.add_argument("--window", type=int, default=60, help="Lookback window for volatility/correlation.")
    parser.add_argument("--method", choices=("pearson", "spearman"), default="pearson")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()

    from research.correlation_matrix import load_returns

    returns = load_returns(Path(args.source))
    allocation = from_returns_frame(returns, window=args.window, method=args.method)
    persist_allocation(allocation, path=Path(args.output))
    print(json.dumps(allocation.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
