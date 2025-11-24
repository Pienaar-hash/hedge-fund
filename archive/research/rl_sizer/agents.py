from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np


class BaseAgent:
    """Minimal agent interface used by the RL sizing pilot."""

    def act(self, observation: np.ndarray) -> float:  # pragma: no cover - interface definition
        raise NotImplementedError

    def update(self, reward: float, observation: np.ndarray) -> None:  # pragma: no cover - interface definition
        pass


@dataclass
class PPOConfig:
    learning_rate: float = 0.05
    discount: float = 0.95
    action_scale: float = 1.5
    entropy_coef: float = 0.01


class PPOAgent(BaseAgent):
    """Tiny PPO-style policy with a linear head; adequate for dry-run simulations."""

    def __init__(self, obs_size: int, config: Optional[PPOConfig] = None, seed: Optional[int] = None) -> None:
        cfg = config or PPOConfig()
        self.cfg = cfg
        self._rng = random.Random(seed)
        self.weights = np.zeros(obs_size, dtype=float)
        # favour recency slightly
        if obs_size >= 3:
            self.weights[-3] = 0.25  # current position
            self.weights[-2] = 0.4  # realized vol
            self.weights[-1] = 0.6  # normalized sharpe
        self._baseline = 0.0

    def _policy(self, features: np.ndarray) -> float:
        logits = float(np.dot(self.weights, features))
        action = math.tanh(logits) * self.cfg.action_scale
        return action

    def act(self, observation: np.ndarray) -> float:
        return self._policy(observation)

    def update(self, reward: float, observation: np.ndarray) -> None:
        advantage = reward - self._baseline
        self._baseline = self.cfg.discount * self._baseline + (1 - self.cfg.discount) * reward
        gradient = advantage * observation
        self.weights += self.cfg.learning_rate * gradient
        entropy_noise = self._rng.uniform(-1.0, 1.0) * self.cfg.entropy_coef
        self.weights += entropy_noise


@dataclass
class DQNConfig:
    action_bins: int = 5
    epsilon: float = 0.1
    gamma: float = 0.9
    learning_rate: float = 0.1
    max_position: float = 3.0


class DQNAgent(BaseAgent):
    """Discrete-action Q-learning agent; keeps state small for dry-runs."""

    def __init__(self, config: Optional[DQNConfig] = None, seed: Optional[int] = None) -> None:
        self.cfg = config or DQNConfig()
        self._rng = random.Random(seed)
        self.actions = np.linspace(-self.cfg.max_position, self.cfg.max_position, self.cfg.action_bins, dtype=float)
        self.q_values = np.zeros(self.cfg.action_bins, dtype=float)
        self._last_action_index: Optional[int] = None

    def act(self, observation: np.ndarray) -> float:
        if self._rng.random() < self.cfg.epsilon:
            idx = self._rng.randrange(len(self.actions))
        else:
            idx = int(np.argmax(self.q_values))
        self._last_action_index = idx
        return float(self.actions[idx])

    def update(self, reward: float, observation: np.ndarray) -> None:
        if self._last_action_index is None:
            return
        best_future = float(np.max(self.q_values))
        target = reward + self.cfg.gamma * best_future
        delta = target - self.q_values[self._last_action_index]
        self.q_values[self._last_action_index] += self.cfg.learning_rate * delta
