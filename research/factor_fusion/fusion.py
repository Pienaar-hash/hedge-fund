from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "FactorFusionConfig",
    "FactorFusionResult",
    "FactorFusion",
]


@dataclass(slots=True)
class FactorFusionConfig:
    """Configuration governing how factor signals are blended."""

    regularization: float = 1e-3
    positive_weights: bool = True
    weight_floor: float = 0.0
    clip_output: Optional[float] = 5.0
    neutralize_mean: bool = True
    annualization: int = 252


@dataclass(slots=True)
class FactorFusionResult:
    """Diagnostic bundle returned after fitting the fusion layer."""

    fused_signal: pd.Series
    weights: pd.Series
    ic: float
    r_squared: float
    residual_vol: float
    signal_sharpe: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "ic": float(self.ic),
            "r_squared": float(self.r_squared),
            "residual_vol": float(self.residual_vol),
            "signal_sharpe": float(self.signal_sharpe),
            "weights": {str(k): float(v) for k, v in self.weights.items()},
            "signal": self.fused_signal.to_list(),
        }


class FactorFusion:
    """Blend heterogeneous factor inputs into a single tradable alpha."""

    def __init__(self, config: Optional[FactorFusionConfig] = None) -> None:
        self.config = config or FactorFusionConfig()
        self._weights: pd.Series = pd.Series(dtype=float)
        self._feature_mean: pd.Series = pd.Series(dtype=float)
        self._feature_std: pd.Series = pd.Series(dtype=float)

    @property
    def weights(self) -> pd.Series:
        return self._weights.copy()

    def fit(self, factors: pd.DataFrame, target: Sequence[float] | pd.Series) -> FactorFusionResult:
        feature_frame = self._prepare_features(factors)
        target_series = pd.Series(target, index=feature_frame.index, dtype=float).fillna(0.0)
        X = feature_frame.to_numpy(dtype=float)
        y = target_series.to_numpy(dtype=float)

        weights = self._solve_ridge(X, y, feature_frame.columns)
        fused = self._fused_signal(X, weights, target_series.index)
        ic = _safe_corr(fused.to_numpy(dtype=float), y)
        residual = y - fused.to_numpy(dtype=float)
        y_var = float(np.var(y))
        residual_var = float(np.var(residual))
        r_squared = 0.0 if y_var == 0 else max(0.0, 1.0 - residual_var / y_var)
        residual_vol = math.sqrt(residual_var)
        signal_sharpe = _sharpe_ratio(fused, annualization=self.config.annualization)

        self._weights = weights
        return FactorFusionResult(
            fused_signal=fused,
            weights=weights,
            ic=ic,
            r_squared=r_squared,
            residual_vol=residual_vol,
            signal_sharpe=signal_sharpe,
        )

    def transform(self, factors: pd.DataFrame) -> pd.Series:
        if self._weights.empty:
            raise RuntimeError("FactorFusion.transform called before fit")
        feature_frame = self._prepare_features(factors, refit=False)
        X = feature_frame.to_numpy(dtype=float)
        return self._fused_signal(X, self._weights, feature_frame.index)

    def explain(self) -> pd.Series:
        if self._weights.empty:
            raise RuntimeError("FactorFusion.explain called before fit")
        return self._weights.sort_values(ascending=False)

    def _prepare_features(self, factors: pd.DataFrame, *, refit: bool = True) -> pd.DataFrame:
        if factors.empty:
            raise ValueError("no factor data supplied to FactorFusion")
        numeric = factors.select_dtypes(include=[np.number]).copy()
        numeric = numeric.ffill().bfill().fillna(0.0)
        if refit:
            self._feature_mean = numeric.mean()
            self._feature_std = numeric.std(ddof=0).replace(0.0, 1.0)
        else:
            if self._feature_mean.empty or self._feature_std.empty:
                raise RuntimeError("FactorFusion not fitted; mean/std statistics missing")
        normalized = (numeric - self._feature_mean) / self._feature_std
        normalized = normalized.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        if self.config.neutralize_mean:
            normalized = normalized - normalized.mean(axis=0)
        return normalized

    def _solve_ridge(self, X: np.ndarray, y: np.ndarray, columns: pd.Index) -> pd.Series:
        if X.size == 0:
            raise ValueError("empty feature matrix provided to FactorFusion")
        lam = float(max(self.config.regularization, 1e-8))
        XtX = X.T @ X
        ridge = XtX + lam * np.eye(X.shape[1])
        Xty = X.T @ y
        try:
            raw_weights = np.linalg.solve(ridge, Xty)
        except np.linalg.LinAlgError:
            raw_weights = np.linalg.pinv(ridge) @ Xty
        weights = pd.Series(raw_weights, index=columns)
        if self.config.positive_weights:
            weights = weights.clip(lower=0.0)
        if self.config.weight_floor > 0.0:
            weights = weights.where(weights >= self.config.weight_floor, self.config.weight_floor)
        weight_sum = float(weights.sum())
        if weight_sum <= 0.0:
            weights = pd.Series(np.full(len(weights), 1.0 / len(weights)), index=columns)
        else:
            weights = weights / weight_sum
        return weights

    def _fused_signal(self, X: np.ndarray, weights: pd.Series, index: pd.Index) -> pd.Series:
        fused = X @ weights.to_numpy(dtype=float)
        if self.config.clip_output is not None:
            fused = np.clip(fused, -self.config.clip_output, self.config.clip_output)
        return pd.Series(fused, index=index, name="fused_alpha")


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_std = np.std(x)
    y_std = np.std(y)
    if x_std == 0.0 or y_std == 0.0:
        return 0.0
    corr = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(corr):
        return 0.0
    return corr


def _sharpe_ratio(series: pd.Series, *, annualization: int) -> float:
    mean = float(series.mean())
    std = float(series.std(ddof=1))
    if std == 0.0:
        return 0.0
    return (mean / std) * math.sqrt(float(annualization))
