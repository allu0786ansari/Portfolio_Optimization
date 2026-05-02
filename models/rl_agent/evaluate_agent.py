"""Evaluation utilities for trained RL agents.

Runs N episodes on a given environment and computes:
  - Mean Sharpe ratio across episodes
  - Mean Sortino ratio
  - Mean CAGR (compounded annual growth rate)
  - Mean maximum drawdown
  - Mean episode reward
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from loguru import logger


def compute_metrics(episode_returns: list[float]) -> dict:
    """Compute risk/return metrics from a list of daily log returns."""
    r = np.array(episode_returns)
    if len(r) == 0:
        return {"sharpe": 0.0, "sortino": 0.0, "cagr": 0.0, "max_dd": 0.0}

    # Annualised metrics
    mean_r = r.mean()
    std_r  = r.std() + 1e-8

    # Sharpe
    sharpe = float(mean_r / std_r * np.sqrt(252))

    # Sortino
    downside = r[r < 0]
    down_std = float(np.sqrt(np.mean(downside ** 2))) if len(downside) > 0 else 1e-8
    sortino  = float(mean_r / down_std * np.sqrt(252))

    # CAGR from cumulative log returns
    total_log_return = float(r.sum())
    n_years = len(r) / 252
    cagr = float(np.exp(total_log_return / max(n_years, 1e-8)) - 1)

    # Maximum drawdown on equity curve
    equity = np.exp(np.cumsum(r))
    peak   = np.maximum.accumulate(equity)
    dd     = (equity - peak) / peak
    max_dd = float(dd.min())

    return {
        "sharpe":  sharpe,
        "sortino": sortino,
        "cagr":    cagr,
        "max_dd":  max_dd,
    }


def evaluate_agent(
    model,
    env,
    n_episodes: int = 20,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Run n_episodes and return aggregated metrics.

    Args:
        model:      trained SB3 model (PPO or SAC)
        env:        PortfolioEnv instance
        n_episodes: number of evaluation episodes
        seed:       base seed (each episode uses seed + i)

    Returns:
        dict with mean_ and std_ of all metrics across episodes
    """
    all_sharpes  = []
    all_sortinos = []
    all_cagrs    = []
    all_max_dds  = []
    all_rewards  = []

    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        episode_returns = []
        total_reward    = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_returns.append(info["portfolio_return"])
            total_reward += reward
            done = terminated or truncated

        metrics = compute_metrics(episode_returns)
        all_sharpes.append(metrics["sharpe"])
        all_sortinos.append(metrics["sortino"])
        all_cagrs.append(metrics["cagr"])
        all_max_dds.append(metrics["max_dd"])
        all_rewards.append(total_reward)

    results = {
        "mean_sharpe":  float(np.mean(all_sharpes)),
        "std_sharpe":   float(np.std(all_sharpes)),
        "mean_sortino": float(np.mean(all_sortinos)),
        "mean_cagr":    float(np.mean(all_cagrs)),
        "mean_max_dd":  float(np.mean(all_max_dds)),
        "mean_reward":  float(np.mean(all_rewards)),
        "n_episodes":   n_episodes,
    }

    if verbose:
        logger.info(
            f"Eval ({n_episodes} eps) — "
            f"Sharpe={results['mean_sharpe']:.3f}±{results['std_sharpe']:.3f} | "
            f"Sortino={results['mean_sortino']:.3f} | "
            f"CAGR={results['mean_cagr']:.1%} | "
            f"MaxDD={results['mean_max_dd']:.1%}"
        )
    return results