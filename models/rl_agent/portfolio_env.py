"""Legacy entry point for the portfolio RL environment — now backed by the corrected v2 env.

v1's ``PortfolioEnv`` had look-ahead (the reward was the same-day return that
was also in the observation), mixed NSE and NYSE calendars, clipped raw
features and an action projection that pinned weights near 1/N. This module
keeps the v1 import path and constructor shape for ``train_agent.py`` and the
dashboard, but all logic now lives in ``pfo.rl.env.PortfolioEnvV2``.

Data layout of one env instance (fractions of the universe's history):
    [0, train_frac)               training rows (scaler is fitted here only)
    [start_frac, end_frac)        rows this env may sample episodes from
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from loguru import logger

from pfo.data.market import load_v1_universe
from pfo.features.basic import FEATURE_NAMES, WARMUP_DAYS, FoldScaler, compute_features, to_tensor
from pfo.rl.env import PortfolioEnvV2, project_action

EPISODE_LENGTH = 252
DEFAULT_UNIVERSE = "sp500"


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Kept for backward compatibility with v1 callers."""
    z = np.asarray(x, dtype=float) / temperature
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class PortfolioEnv(PortfolioEnvV2):
    """Corrected environment on one universe's own exchange calendar."""

    def __init__(
        self,
        universe: str = DEFAULT_UNIVERSE,
        episode_length: int = EPISODE_LENGTH,
        train_frac: float = 0.70,
        start_frac: float = 0.0,
        end_frac: float | None = None,
        action_scale: float = 5.0,
        cost_bps: float = 10.0,
        execution_lag: int = 1,
        seed: int | None = None,
        tickers=None,  # accepted for v1 compatibility; the universe defines the tickers
    ):
        market = load_v1_universe(universe)
        feats = compute_features(market.prices, market.benchmark)
        x = to_tensor(feats)[WARMUP_DAYS:]
        r = market.returns.to_numpy()[WARMUP_DAYS:]
        self.dates = market.dates[WARMUP_DAYS:]
        self.tickers = market.tickers
        self.universe = universe
        T = len(self.dates)
        train_end = int(T * train_frac)
        self.scaler = FoldScaler().fit(x, range(0, train_end))
        end_frac = train_frac if end_frac is None else end_frac
        start, end = int(T * start_frac), int(T * end_frac)
        super().__init__(self.scaler.transform(x), r, start=start, end=end, episode_length=episode_length,
                         cost_bps=cost_bps, action_scale=action_scale, execution_lag=execution_lag, seed=seed)
        self._returns_log: list[float] = []
        logger.info(f"PortfolioEnv[{universe}]: {self.n_assets} assets, rows [{start}, {end}) of {T} "
                    f"({self.dates[start].date()} -> {self.dates[end - 1].date()}), scaler fit on [0, {train_end})")

    # -- compatibility helpers used by v1 scripts/tests ----------------------------
    def reset(self, *, seed=None, options=None):
        self._returns_log = []
        obs, info = super().reset(seed=seed, options=options)
        info["start_date"] = str(self.dates[self._t].date())
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        self._returns_log.append(info["portfolio_return"])
        info["portfolio_value"] = float(np.prod(1.0 + np.asarray(self._returns_log)))
        info["step"] = self._steps
        return obs, reward, term, trunc, info

    @property
    def _current_step(self) -> int:
        return self._steps

    @property
    def _weights(self) -> np.ndarray:
        return self._last_target

    @property
    def episode_sharpe(self) -> float:
        r = np.asarray(self._returns_log)
        return float(r.mean() / r.std() * np.sqrt(252)) if len(r) > 1 and r.std() > 1e-12 else 0.0

    def serving_spec(self) -> dict:
        """Everything serving needs to rebuild this env's observation."""
        return {
            "universe": self.universe,
            "tickers": list(self.tickers),
            "feature_names": list(FEATURE_NAMES),
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_std": self.scaler.std_.tolist(),
            "action_scale": self.scale,
            "execution_lag": self.lag,
            "timing_contract": "obs(t) uses rows <= t; reward uses return row t + lag + 1",
        }


__all__ = ["PortfolioEnv", "softmax", "project_action"]
