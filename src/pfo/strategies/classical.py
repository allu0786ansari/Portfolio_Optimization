"""Classical baselines used to re-evaluate v1: equal weight, min-variance, Markowitz."""

from __future__ import annotations

import cvxpy as cp
import numpy as np
from loguru import logger
from sklearn.covariance import LedoitWolf

from pfo.strategies.base import History, Strategy, normalize_long_only


class EqualWeight(Strategy):
    name = "equal_weight"

    def decide(self, hist: History) -> np.ndarray:
        n = hist.returns.shape[1]
        return np.full(n, 1.0 / n)


def _lw_cov(window: np.ndarray) -> np.ndarray:
    window = window[np.isfinite(window).all(axis=1)]
    return LedoitWolf().fit(window).covariance_


def _solve(problem: cp.Problem, w: cp.Variable, n: int, label: str) -> np.ndarray:
    try:
        problem.solve()  # cvxpy picks an installed solver (Clarabel/OSQP); no hard-coded ECOS
    except cp.error.SolverError as exc:
        logger.warning(f"{label}: solver error {exc}; falling back to equal weight")
        return np.full(n, 1.0 / n)
    if w.value is None or problem.status not in ("optimal", "optimal_inaccurate"):
        logger.warning(f"{label}: status {problem.status}; falling back to equal weight")
        return np.full(n, 1.0 / n)
    return normalize_long_only(np.asarray(w.value).ravel())


class MinVariance(Strategy):
    """Long-only minimum variance with Ledoit-Wolf covariance."""

    name = "min_variance"

    def __init__(self, lookback: int = 252, max_weight: float = 0.4) -> None:
        self.lookback = lookback
        self.max_weight = max_weight

    def decide(self, hist: History) -> np.ndarray:
        window = hist.returns[-self.lookback :]
        window = window[np.isfinite(window).all(axis=1)]
        n = hist.returns.shape[1]
        if len(window) < 60:
            return np.full(n, 1.0 / n)
        sigma = _lw_cov(window)
        w = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(sigma))),
                          [cp.sum(w) == 1, w >= 0, w <= self.max_weight])
        return _solve(prob, w, n, self.name)


class MarkowitzV1(Strategy):
    """The v1 Markowitz rule, unchanged except for the solver and timing.

    Minimise variance subject to expected return >= the cross-sectional mean
    of historical mean returns, long-only, 40% cap. (v1's docstring called this
    "max Sharpe"; it is not.)
    """

    name = "markowitz_v1"

    def __init__(self, lookback: int = 252, max_weight: float = 0.4) -> None:
        self.lookback = lookback
        self.max_weight = max_weight

    def decide(self, hist: History) -> np.ndarray:
        window = hist.returns[-self.lookback :]
        window = window[np.isfinite(window).all(axis=1)]
        n = hist.returns.shape[1]
        if len(window) < 60:
            return np.full(n, 1.0 / n)
        mu = window.mean(axis=0)
        sigma = _lw_cov(window)
        w = cp.Variable(n)
        cons = [cp.sum(w) == 1, w >= 0, w <= self.max_weight, mu @ w >= float(mu.mean())]
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(sigma))), cons)
        return _solve(prob, w, n, self.name)
