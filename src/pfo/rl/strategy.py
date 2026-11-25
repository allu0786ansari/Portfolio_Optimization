"""RL allocator re-trained inside every walk-forward fold (fixes v1 defect #3)."""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from pfo.features.basic import FoldScaler
from pfo.rl.env import PortfolioEnvV2, project_action
from pfo.strategies.base import History, Strategy, TrainData
from pfo.timing import DEFAULT_EXECUTION_LAG


class PPOAllocator(Strategy):
    name = "rl_ppo"

    def __init__(
        self,
        timesteps: int = 30_000,
        action_scale: float = 5.0,
        cost_bps: float = 10.0,
        episode_length: int = 126,
        execution_lag: int = DEFAULT_EXECUTION_LAG,
        seed: int = 0,
        ppo_kwargs: dict | None = None,
    ) -> None:
        self.timesteps = timesteps
        self.action_scale = action_scale
        self.cost_bps = cost_bps
        self.episode_length = episode_length
        self.lag = execution_lag
        self.seed = seed
        self.ppo_kwargs = {"n_steps": 512, "batch_size": 128, "learning_rate": 3e-4,
                           "gamma": 0.99, "ent_coef": 0.0, **(ppo_kwargs or {})}
        self.scaler = FoldScaler()
        self.model: Any = None

    def fit(self, data: TrainData) -> None:
        from stable_baselines3 import PPO  # heavy import kept local

        if data.features is None:
            raise ValueError("PPOAllocator needs features")
        T = len(data.returns)
        self.scaler.fit(data.features, range(0, T))  # fitted on this fold's training rows only
        feats = self.scaler.transform(data.features)
        # The first rows of a window can have NaN returns (first difference); they are zeroed.
        env = PortfolioEnvV2(feats, data.returns, start=0, end=T, episode_length=self.episode_length,
                             cost_bps=self.cost_bps, action_scale=self.action_scale,
                             execution_lag=self.lag, seed=self.seed)
        self.model = PPO("MlpPolicy", env, seed=self.seed, verbose=0, device="cpu", **self.ppo_kwargs)
        logger.info(f"{self.name}: training {self.timesteps} steps on {T} rows "
                    f"({data.dates[0].date()} -> {data.dates[-1].date()})")
        self.model.learn(total_timesteps=self.timesteps, progress_bar=False)

    def decide(self, hist: History) -> np.ndarray:
        if self.model is None or hist.features_now is None:
            raise RuntimeError("PPOAllocator.decide called before fit")
        x = self.scaler.transform(hist.features_now[None, ...])[0]
        obs = np.concatenate([x.ravel(), hist.prev_weights]).astype(np.float32)
        action, _ = self.model.predict(obs, deterministic=True)  # SB3 clips to the action space
        return project_action(action, self.action_scale)
