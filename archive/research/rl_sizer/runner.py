from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .agents import BaseAgent, PPOAgent
from .env import SizingEnv

LOG_DIR = Path("logs/research/rl_runs")


def _default_episode_limit(env: SizingEnv) -> int:
    return max(1, len(env._returns) - env.window - 1)


def run_episode(
    env: SizingEnv,
    agent: BaseAgent,
    *,
    max_steps: Optional[int] = None,
    log_path: Optional[Path] = LOG_DIR / "episodes.jsonl",
    dry_run: bool = True,
) -> Dict[str, float]:
    observation, _info = env.reset()
    rewards = []
    positions = []
    limit = max_steps or _default_episode_limit(env)

    for _ in range(limit):
        action = agent.act(observation)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.update(reward, next_obs)
        rewards.append(float(reward))
        positions.append(float(info.get("position", 0.0)))
        observation = next_obs
        if terminated or truncated:
            break

    stats = env.episode_stats(rewards)
    result = {
        "steps": len(rewards),
        "total_reward": float(np.sum(rewards) if rewards else 0.0),
        "normalized_sharpe": float(stats.normalized_sharpe),
        "avg_position": float(np.mean(positions) if positions else 0.0),
        "equity_final": float(stats.equity_curve[-1] if stats.equity_curve else 1.0),
    }
    if dry_run and log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")
    return result


def _cli() -> None:
    parser = argparse.ArgumentParser(description="RL Sizer Pilot Dry-Run")
    parser.add_argument("--episodes", type=int, default=3, help="Number of dry-run episodes")
    parser.add_argument("--window", type=int, default=20, help="Observation window length")
    parser.add_argument("--dry-run", action="store_true", help="Skip logging to disk")
    parser.add_argument("--log-path", type=str, default=None, help="Custom log file (JSONL)")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic returns",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    returns = rng.normal(loc=0.001, scale=0.01, size=500)
    env = SizingEnv(returns.tolist(), window=args.window, target_vol=0.02)
    obs, _ = env.reset()
    agent = PPOAgent(obs_size=obs.shape[0], seed=args.seed)
    log_path = Path(args.log_path) if args.log_path else LOG_DIR / "episodes.jsonl"

    for episode in range(args.episodes):
        result = run_episode(env, agent, dry_run=args.dry_run, log_path=log_path)
        print(json.dumps({"episode": episode, **result}))


__all__ = ["run_episode", "LOG_DIR"]


if __name__ == "__main__":  # pragma: no cover
    _cli()
