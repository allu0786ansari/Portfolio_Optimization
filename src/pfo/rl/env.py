"""Corrected portfolio environment (replaces the v1 ``PortfolioEnv`` logic).

Fixes relative to v1:
1. Timing: the observation at step ``t`` contains features of row ``t``; the
   reward for the action taken at ``t`` is the portfolio return of row
   ``t + execution_lag + 1``. v1 rewarded the return of row ``t`` itself,
   which was also part of the observation (look-ahead).
2. Action projection: ``softmax(action_scale * a)`` with ``a`` in [-1, 1].
   v1 used scale 1, which bounds every weight to roughly 2%-16% for 40 assets
   and forces the policy to stay near equal weight.
3. Costs are charged on turnover against *drifted* weights.
4. Observations are pre-standardized by a fold-fitted scaler (no raw clipping).
5. Episodes are confined to ``[start, end)`` so a training env cannot sample
   validation/test rows.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pfo.timing import DEFAULT_EXECUTION_LAG


def project_action(action: np.ndarray, scale: float) -> np.ndarray:
    """Map a raw action in [-1, 1]^N onto the simplex."""
    a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0) * scale
    a = a - np.max(a)
    e = np.exp(a)
    return e / e.sum()


def max_weight_bound(n_assets: int, scale: float) -> float:
    """Largest weight reachable by ``project_action`` (all others at -1)."""
    hi = np.exp(scale)
    lo = np.exp(-scale)
    return float(hi / (hi + (n_assets - 1) * lo))


class PortfolioEnvV2(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,  # (T, N, F), already standardized
        returns: np.ndarray,  # (T, N) simple returns, row s = close s-1 -> close s
        start: int,
        end: int,  # exclusive: no reward may use a return at row >= end
        episode_length: int = 252,
        cost_bps: float = 10.0,
        action_scale: float = 5.0,
        execution_lag: int = DEFAULT_EXECUTION_LAG,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if features.shape[:2] != returns.shape:
            raise ValueError("features and returns must share (T, N)")
        if execution_lag < 0:
            raise ValueError("execution_lag must be >= 0")
        self.features = features.astype(np.float32)
        self.returns = np.nan_to_num(returns.astype(np.float64), nan=0.0)
        self.start, self.end = int(start), int(end)
        self.lag = int(execution_lag)
        self.n_assets = returns.shape[1]
        self.cost = cost_bps / 1e4
        self.scale = action_scale
        max_len = self.end - self.start - self.lag - 1
        if max_len < 2:
            raise ValueError("window too short for one decision")
        self.episode_length = min(episode_length, max_len)
        obs_dim = self.n_assets * features.shape[2] + self.n_assets
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n_assets,), dtype=np.float32)
        self._t = self.start
        self._steps = 0
        self._w = np.full(self.n_assets, 1.0 / self.n_assets)  # drifted holdings (for costs)
        self._last_target = self._w.copy()  # last decision (what the agent observes)
        self.reset(seed=seed)

    # -- helpers ---------------------------------------------------------
    def _obs(self) -> np.ndarray:
        # The agent sees its previous *target* weights, not the drifted holdings:
        # with execution_lag >= 1 the drift uses returns after row t (unknown at t).
        return np.concatenate([self.features[self._t].ravel(), self._last_target.astype(np.float32)]).astype(
            np.float32
        )

    def reward_row(self, t: int | None = None) -> int:
        """Row whose return is earned by the action taken at row ``t``."""
        t = self._t if t is None else t
        return t + self.lag + 1

    # -- gym API -----------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        last_start = self.end - self.lag - 1 - self.episode_length
        if options and "start_row" in options:
            self._t = int(options["start_row"])
        else:
            self._t = int(self.np_random.integers(self.start, max(self.start, last_start) + 1))
        self._steps = 0
        self._w = np.full(self.n_assets, 1.0 / self.n_assets)
        self._last_target = self._w.copy()
        return self._obs(), {"row": self._t}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        target = project_action(action, self.scale)
        turnover = float(np.abs(target - self._w).sum())
        cost = turnover * self.cost
        row = self.reward_row()
        if row >= self.end:
            raise RuntimeError("episode ran past its window (bug)")
        r = self.returns[row]
        gross = float(target @ r)
        net = (1.0 + gross) * (1.0 - cost) - 1.0
        reward = float(np.log1p(max(net, -0.999999)))
        # drift to the close of the reward row
        grown = target * (1.0 + r)
        self._w = grown / grown.sum() if grown.sum() > 0 else np.full(self.n_assets, 1.0 / self.n_assets)
        self._last_target = target
        self._t += 1
        self._steps += 1
        truncated = self._steps >= self.episode_length or self.reward_row() >= self.end
        info = {"weights": target, "portfolio_return": net, "gross_return": gross,
                "turnover": turnover, "cost": cost, "reward_row": row}
        return self._obs(), reward, False, truncated, info
