from __future__ import annotations

import numpy as np

from research.rl_sizer import DQNAgent, DQNConfig, PPOAgent, SizingEnv, run_episode


def _sample_returns(length: int = 80) -> np.ndarray:
    rng = np.random.default_rng(seed=42)
    return rng.normal(loc=0.001, scale=0.01, size=length)


def test_sizing_env_observation_shape() -> None:
    returns = _sample_returns()
    env = SizingEnv(returns, window=12, target_vol=0.02)
    obs, info = env.reset()
    assert obs.shape[0] == 12 + 3  # window + [position, vol, sharpe]
    assert info["index"] == 12


def test_run_episode_with_ppo_agent() -> None:
    returns = _sample_returns()
    env = SizingEnv(returns, window=15, target_vol=0.02)
    obs, _ = env.reset()
    agent = PPOAgent(obs_size=obs.shape[0], seed=7)
    result = run_episode(env, agent, max_steps=30, dry_run=False, log_path=None)
    assert "normalized_sharpe" in result
    assert "total_reward" in result


def test_dqn_agent_learning_step() -> None:
    returns = _sample_returns()
    env = SizingEnv(returns, window=10, target_vol=0.02)
    obs, _ = env.reset()
    agent = DQNAgent(DQNConfig(action_bins=3, max_position=1.0), seed=3)
    action = agent.act(obs)
    assert abs(action) <= 1.0
    new_obs, reward, *_ = env.step(action)
    agent.update(reward, new_obs)
    # ensure Q-values updated for last action
    assert np.any(agent.q_values != 0.0)
