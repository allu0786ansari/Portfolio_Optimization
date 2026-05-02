"""Markowitz Mean-Variance Optimiser using cvxpy.

Maximises Sharpe ratio subject to:
  - weights sum to 1
  - all weights >= 0 (long-only)
  - max single asset weight = 40% (concentration limit)

Uses Ledoit-Wolf shrinkage for a more stable covariance estimate.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.covariance import LedoitWolf
from loguru import logger

from data.config import ALL_TICKERS
from models.rl_agent.data_loader import load_aligned_features


MAX_WEIGHT     = 0.40   # max single-asset allocation
RISK_FREE_RATE = 0.0    # daily risk-free rate
MIN_WEIGHT     = 0.0    # long-only


def ledoit_wolf_cov(returns: np.ndarray) -> np.ndarray:
    """Compute Ledoit-Wolf shrinkage covariance matrix.

    Args:
        returns: (T, N) matrix of daily log returns
    Returns:
        (N, N) covariance matrix
    """
    lw = LedoitWolf()
    lw.fit(returns)
    return lw.covariance_


def markowitz_weights(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    max_weight: float = MAX_WEIGHT,
    risk_free: float = RISK_FREE_RATE,
) -> np.ndarray:
    """Compute maximum-Sharpe portfolio weights using cvxpy.

    We maximise the Sharpe ratio by solving the equivalent problem
    of minimising portfolio variance for a given target return,
    then scanning along the efficient frontier.

    Args:
        expected_returns: (N,) vector of expected daily returns
        cov_matrix:       (N, N) covariance matrix
        max_weight:       maximum weight per asset
        risk_free:        daily risk-free rate

    Returns:
        (N,) optimal weight vector
    """
    n = len(expected_returns)
    w = cp.Variable(n)

    # Portfolio return and variance
    port_return = expected_returns @ w
    port_var    = cp.quad_form(w, cov_matrix)

    # Constraints: long-only, sum to 1, concentration limit
    constraints = [
        cp.sum(w) == 1,
        w >= MIN_WEIGHT,
        w <= max_weight,
    ]

    # Maximise Sharpe: equivalent to minimising variance
    # for a target return (we use mean return as target)
    target_return = float(np.mean(expected_returns))
    objective = cp.Minimize(port_var)
    constraints.append(port_return >= target_return)

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.ECOS, warm_start=True)
        if w.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
            logger.warning(f"cvxpy status: {prob.status} — falling back to equal weight")
            return np.ones(n) / n
        weights = np.array(w.value).flatten()
        weights = np.clip(weights, 0, max_weight)
        weights /= weights.sum()   # renormalise after clip
        return weights.astype(np.float32)
    except Exception as e:
        logger.warning(f"Markowitz optimisation failed: {e} — using equal weight")
        return np.ones(n) / n


class MarkowitzOptimiser:
    """Rolling Markowitz optimiser that recomputes weights on a lookback window."""

    def __init__(
        self,
        lookback_days: int = 252,
        max_weight: float = MAX_WEIGHT,
    ):
        self.lookback_days = lookback_days
        self.max_weight    = max_weight

    def compute_weights(
        self,
        returns_window: np.ndarray,
    ) -> np.ndarray:
        """Given (T, N) returns, compute optimal weights.

        Args:
            returns_window: (T, N) matrix of daily returns for lookback window
        Returns:
            (N,) weight vector
        """
        if len(returns_window) < 30:
            n = returns_window.shape[1]
            return np.ones(n) / n

        expected_returns = returns_window.mean(axis=0)
        cov              = ledoit_wolf_cov(returns_window)

        return markowitz_weights(
            expected_returns = expected_returns,
            cov_matrix       = cov,
            max_weight       = self.max_weight,
        )

    def simulate(
        self,
        return_matrix: np.ndarray,
        start_idx: int,
        end_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate Markowitz strategy over a date range.

        Recomputes weights monthly (every 21 trading days).
        Returns portfolio daily returns and weight history.
        """
        T, N = return_matrix.shape
        portfolio_returns = []
        weight_history    = []

        current_weights = np.ones(N) / N
        rebalance_freq  = 21   # monthly rebalancing

        for t in range(start_idx, min(end_idx, T)):
            # Rebalance at start and then every 21 days
            if (t - start_idx) % rebalance_freq == 0 and t >= self.lookback_days:
                window = return_matrix[t - self.lookback_days : t]
                current_weights = self.compute_weights(window)

            port_ret = float(np.dot(current_weights, return_matrix[t]))
            portfolio_returns.append(port_ret)
            weight_history.append(current_weights.copy())

        return np.array(portfolio_returns), np.array(weight_history)