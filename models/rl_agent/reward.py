"""Reward functions for the portfolio RL environment.

We use Sortino ratio as the primary reward signal:
- Rewards positive returns
- Penalises downside deviation only (not all volatility like Sharpe)
- Subtracts transaction cost proportional to portfolio turnover

Why Sortino over Sharpe?
  Sharpe penalises both upside and downside volatility equally.
  Sortino only penalises downside — the agent learns to cut losses
  without being punished for large positive returns.
"""
import numpy as np

TRANSACTION_COST_RATE = 0.001   # 0.1% per unit of turnover
RISK_FREE_DAILY       = 0.0     # simplification: 0% daily risk-free rate
MIN_DOWNSIDE_STD      = 1e-8    # avoid division by zero


def sortino_reward(
    returns: np.ndarray,
    weights_prev: np.ndarray,
    weights_curr: np.ndarray,
    annualise: bool = False,
) -> tuple[float, dict]:
    """Compute Sortino-based reward for one episode window.

    Args:
        returns:      array of daily portfolio returns during the episode
        weights_prev: portfolio weights at start of episode
        weights_curr: portfolio weights at end of episode
        annualise:    if True, scale to annual Sortino

    Returns:
        reward: scalar reward value
        info:   dict with breakdown components
    """
    if len(returns) == 0:
        return 0.0, {}

    # Portfolio return components
    mean_return   = float(np.mean(returns))
    total_return  = float(np.sum(returns))

    # Downside deviation (only negative returns)
    excess = returns - RISK_FREE_DAILY
    downside = excess[excess < 0]
    if len(downside) > 0:
        downside_std = float(np.sqrt(np.mean(downside ** 2)))
    else:
        downside_std = MIN_DOWNSIDE_STD   # no negative days = great episode

    # Sortino ratio
    scale = np.sqrt(252) if annualise else 1.0
    sortino = scale * (mean_return - RISK_FREE_DAILY) / max(downside_std, MIN_DOWNSIDE_STD)

    # Transaction cost penalty
    turnover = float(np.sum(np.abs(weights_curr - weights_prev)))
    tc_penalty = TRANSACTION_COST_RATE * turnover

    reward = float(sortino - tc_penalty)

    info = {
        "sortino":       sortino,
        "mean_return":   mean_return,
        "total_return":  total_return,
        "downside_std":  downside_std,
        "turnover":      turnover,
        "tc_penalty":    tc_penalty,
        "reward":        reward,
    }
    return reward, info


def step_reward(
    portfolio_return: float,
    weights_prev: np.ndarray,
    weights_curr: np.ndarray,
) -> tuple[float, dict]:
    """Per-step reward: log return minus transaction cost.

    Used at each individual time step (as opposed to episode-level Sortino).
    Simpler signal that works well with PPO's per-step advantage estimation.
    """
    tc_penalty = TRANSACTION_COST_RATE * float(np.sum(np.abs(weights_curr - weights_prev)))
    reward = portfolio_return - tc_penalty

    info = {
        "portfolio_return": portfolio_return,
        "turnover":         float(np.sum(np.abs(weights_curr - weights_prev))),
        "tc_penalty":       tc_penalty,
        "reward":           reward,
    }
    return reward, info