"""Validates the PortfolioEnv using stable-baselines3 environment checker.

Run this script to confirm the environment is correctly implemented
before starting RL agent training in Week 4.

Expected output:
  - "Environment check passed!" from SB3 checker
  - 5-episode sanity test showing consistent weights sum = 1.0
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from loguru import logger
from stable_baselines3.common.env_checker import check_env

from models.rl_agent.portfolio_env import PortfolioEnv


def run_env_checker() -> None:
    logger.info("Initialising PortfolioEnv...")
    env = PortfolioEnv(seed=42)

    print(f"State shape : {env.observation_space.shape}")
    print(f"Action shape: {env.action_space.shape}")
    print(f"N assets    : {env.n_assets}")
    print(f"Tickers     : {env.tickers[:5]} ... (showing first 5)")
    print()

    # SB3 environment checker — validates all Gym interface requirements
    logger.info("Running stable-baselines3 environment checker...")
    check_env(env, warn=True)
    print("Environment check passed!")
    print()

    # 5-episode sanity test
    print("--- 5-episode sanity test ---")
    for ep in range(1, 6):
        obs, info = env.reset(seed=ep)
        assert obs.shape == env.observation_space.shape, "Obs shape mismatch"

        total_reward = 0.0
        final_weights = None

        for _ in range(env.episode_length):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            final_weights = info["weights"]

            assert not np.isnan(reward),       "Reward is NaN"
            assert not np.isnan(obs).any(),    "Obs contains NaN"
            assert abs(info["weights"].sum() - 1.0) < 1e-5,                 f"Weights don't sum to 1: {info['weights'].sum()}"

            if terminated or truncated:
                break

        sharpe = env.episode_sharpe
        print(
            f"Ep {ep} | steps={env._current_step} | "
            f"total_reward={total_reward:+.3f} | "
            f"sharpe={sharpe:+.3f} | "
            f"weights_sum={final_weights.sum():.6f}"
        )

    print()
    print("All checks passed. Ready for Week 4 RL training.")
    env.close()


if __name__ == "__main__":
    run_env_checker()