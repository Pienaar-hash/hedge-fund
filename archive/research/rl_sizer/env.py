from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from execution.metrics_normalizer import NormalizedMetrics, compute_normalized_metrics


def _ensure_min_length(data: Sequence[float], window: int) -> None:
    if len(data) < window + 2:
        raise ValueError(f"returns history must be at least window+2 long (got {len(data)} < {window + 2})")


@dataclass
class EpisodeStats:
    rewards: List[float]
    pnl: List[float]
    equity_curve: List[float]
    normalized_sharpe: float


class SizingEnv:
    """Lightweight RL environment for sizing decisions (Gymnasium-compatible)."""

    metadata = {"render_modes": ["ansi"], "name": "SizingEnv-v0"}

    def __init__(
        self,
        returns: Sequence[float],
        *,
        target_vol: float = 0.02,
        fee: float = 0.0005,
        window: int = 20,
        max_position: float = 3.0,
        seed: Optional[int] = None,
    ) -> None:
        if window < 3:
            raise ValueError("window must be >= 3")
        _ensure_min_length(returns, window)
        self._rng = random.Random(seed)
        self._returns = np.asarray(list(returns), dtype=float)
        self.window = window
        self.target_vol = target_vol
        self.fee = fee
        self.max_position = max_position

        self.position = 0.0
        self._step_index = window
        self.equity = 1.0
        self._peak_equity = 1.0
        self._min_equity = 1.0

    # ------------------------------------------------------------------ API
    def reset(self, *, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng.seed(seed)
        self._step_index = self.window
        self.position = 0.0
        self.equity = 1.0
        self._peak_equity = 1.0
        self._min_equity = 1.0
        obs = self._build_observation()
        return obs, {"index": self._step_index}

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = float(np.clip(action, -self.max_position, self.max_position))
        previous_position = self.position
        pnl = action * float(self._returns[self._step_index])
        fee_cost = abs(action - previous_position) * self.fee
        reward = pnl - fee_cost

        self.position = action
        self.equity *= (1.0 + reward)
        self._peak_equity = max(self._peak_equity, self.equity)
        self._min_equity = min(self._min_equity, self.equity)

        self._step_index += 1
        terminated = self._step_index >= len(self._returns) - 1
        truncated = False
        obs = self._build_observation()
        metrics = self._metrics_snapshot()
        info = {
            "pnl": pnl,
            "fee_cost": fee_cost,
            "reward": reward,
            "equity": self.equity,
            "position": self.position,
            "normalized_sharpe": metrics.normalized_sharpe,
            "volatility_scale": metrics.volatility_scale,
            "drawdown": (self._peak_equity - self.equity) / max(self._peak_equity, 1e-9),
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> str:
        return (
            f"step={self._step_index} position={self.position:.2f} equity={self.equity:.3f} "
            f"peak={self._peak_equity:.3f}"
        )

    # ------------------------------------------------------------------ helpers
    def _build_observation(self) -> np.ndarray:
        start = self._step_index - self.window
        end = self._step_index
        window_returns = self._returns[start:end]
        metrics = self._metrics_snapshot()
        realized_vol = float(np.std(window_returns, ddof=1)) if window_returns.size > 1 else 0.0
        obs = np.concatenate(
            [
                window_returns,
                np.asarray([self.position, realized_vol, metrics.normalized_sharpe], dtype=float),
            ]
        )
        return obs.astype(float)

    def _metrics_snapshot(self) -> NormalizedMetrics:
        start = self._step_index - self.window
        end = self._step_index
        window_returns = self._returns[start:end]
        return compute_normalized_metrics(
            window_returns,
            target_vol=self.target_vol,
            annualization=252,
            min_observations=2,
            window=len(window_returns),
        )

    # ------------------------------------------------------------------ utils
    def episode_stats(self, rewards: Sequence[float]) -> EpisodeStats:
        pnl = list(rewards)
        equity = []
        nav = 1.0
        for reward in rewards:
            nav *= (1.0 + reward)
            equity.append(nav)
        metrics = compute_normalized_metrics(pnl, target_vol=self.target_vol, annualization=252)
        return EpisodeStats(
            rewards=list(rewards),
            pnl=pnl,
            equity_curve=equity,
            normalized_sharpe=metrics.normalized_sharpe,
        )
