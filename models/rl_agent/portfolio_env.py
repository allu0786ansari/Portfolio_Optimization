"""Custom Gymnasium environment for portfolio allocation.

Implements the OpenAI Gym interface:
    reset()  -> (observation, info)
    step()   -> (observation, reward, terminated, truncated, info)
    render() -> None
    close()  -> None

Design decisions (asked in interviews):
- Action space: Box(-1, 1, N) — agent outputs raw logits, softmax projects
  onto the simplex. Weights always sum to 1, all non-negative.
- State space: flattened (N assets x 11 features) + current weights = N*12
- Reward: per-step log return minus transaction cost penalty
- Episode length: 252 steps (1 trading year)
- Random start: each reset() samples a random start date from the
  training window — prevents memorisation of specific market paths
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from loguru import logger

from data.config import ALL_TICKERS
from models.rl_agent.data_loader import load_aligned_features, build_return_matrix
from models.rl_agent.reward import step_reward
from models.forecasting.dataset import FEATURE_COLS


EPISODE_LENGTH   = 252          # trading days per episode (1 year)
N_FEATURES       = len(FEATURE_COLS)   # features per asset (10)
SOFTMAX_TEMP     = 1.0          # temperature for softmax projection


def softmax(x: np.ndarray, temperature: float = SOFTMAX_TEMP) -> np.ndarray:
    """Numerically stable softmax — projects action vector onto simplex."""
    x = x / temperature
    x = x - np.max(x)   # stability: subtract max before exp
    e = np.exp(x)
    return e / e.sum()


class PortfolioEnv(gym.Env):
    """Portfolio allocation environment.

    The agent learns to allocate capital across N assets to maximise
    risk-adjusted returns while controlling transaction costs.

    Attributes:
        n_assets      : number of assets in the universe
        observation_space : Box(N * (n_features + 1),)
        action_space      : Box(-1, 1, N) — softmaxed to weights
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        tickers: list[str] = ALL_TICKERS,
        episode_length: int = EPISODE_LENGTH,
        train_frac: float = 0.7,
        seed: int | None = None,
    ):
        super().__init__()

        # Load and align feature data
        self.features_dict, self.tickers, self.dates = load_aligned_features(tickers)
        self.n_assets        = len(self.tickers)
        self.episode_length  = episode_length
        self.return_matrix   = build_return_matrix(self.features_dict, self.tickers)

        # Feature matrix: shape (T, N, F)
        self.feature_matrix = np.stack(
            [self.features_dict[t][FEATURE_COLS].values for t in self.tickers],
            axis=1,
        ).astype(np.float32)   # (T, N, F)

        # Training window boundary (no leakage into val/test)
        self.train_end_idx = int(len(self.dates) * train_frac) - episode_length - 1
        if self.train_end_idx < episode_length:
            raise ValueError("Not enough training data for episode length")

        # Gymnasium spaces
        obs_dim = self.n_assets * (N_FEATURES + 1)   # features + current weights
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.n_assets,), dtype=np.float32,
        )

        # Episode state
        self._start_idx    : int        = 0
        self._current_step : int        = 0
        self._weights      : np.ndarray = np.ones(self.n_assets) / self.n_assets
        self._portfolio_value : float   = 1.0
        self._episode_returns : list[float] = []

        # RNG
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        logger.info(
            f"PortfolioEnv ready: {self.n_assets} assets, "
            f"obs={obs_dim}, action={self.n_assets}, "
            f"episode_length={episode_length}"
        )

    # ── Gym interface ───────────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Random start index within training window
        self._start_idx    = int(self.np_random.integers(
            low=0, high=max(1, self.train_end_idx)
        ))
        self._current_step = 0
        self._weights      = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self._portfolio_value = 1.0
        self._episode_returns = []

        obs = self._get_observation()
        return obs, {"start_date": str(self.dates[self._start_idx])}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one trading day.

        1. Project action (raw logits) -> portfolio weights via softmax
        2. Compute portfolio return for this day
        3. Compute reward: step return - transaction cost
        4. Update state
        5. Return (obs, reward, terminated, truncated, info)
        """
        # Project action onto simplex
        new_weights = softmax(action.astype(np.float32))

        # Current market step
        t = self._start_idx + self._current_step
        asset_returns = self.return_matrix[t]   # shape (N,)

        # Portfolio log return = weighted sum of asset returns
        portfolio_return = float(np.dot(new_weights, asset_returns))
        self._episode_returns.append(portfolio_return)

        # Update portfolio value (compound)
        self._portfolio_value *= np.exp(portfolio_return)

        # Reward
        reward, reward_info = step_reward(
            portfolio_return = portfolio_return,
            weights_prev     = self._weights,
            weights_curr     = new_weights,
        )

        # Advance state
        self._weights      = new_weights
        self._current_step += 1

        terminated = False
        truncated  = self._current_step >= self.episode_length

        obs  = self._get_observation()
        info = {
            "weights":          new_weights,
            "portfolio_return":  portfolio_return,
            "portfolio_value":  self._portfolio_value,
            "step":             self._current_step,
            **reward_info,
        }
        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        t = self._start_idx + self._current_step
        date_str = str(self.dates[t]) if t < len(self.dates) else "N/A"
        top3 = sorted(
            zip(self.tickers, self._weights),
            key=lambda x: x[1], reverse=True
        )[:3]
        top3_str = ", ".join(f"{t}:{w:.2f}" for t, w in top3)
        print(
            f"Step {self._current_step:3d} | {date_str[:10]} | "
            f"Value={self._portfolio_value:.4f} | Top3: {top3_str}"
        )

    def close(self) -> None:
        pass

    # ── Internal helpers ────────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        """Build flat observation vector: [asset_features | current_weights]."""
        t = self._start_idx + self._current_step
        t = min(t, len(self.dates) - 1)   # clamp to valid range

        # Feature slice: (N, F) -> flatten to (N*F,)
        features = self.feature_matrix[t]             # (N, F)
        features_flat = features.flatten()            # (N*F,)

        # Clip extreme values (outliers in returns / vol)
        features_flat = np.clip(features_flat, -10.0, 10.0)

        obs = np.concatenate([features_flat, self._weights]).astype(np.float32)
        return obs

    @property
    def episode_sharpe(self) -> float:
        """Sharpe ratio of current/last episode (for logging)."""
        if len(self._episode_returns) < 2:
            return 0.0
        r = np.array(self._episode_returns)
        std = r.std()
        return float(r.mean() / std * np.sqrt(252)) if std > 1e-8 else 0.0

    @property
    def episode_sortino(self) -> float:
        """Sortino ratio of current/last episode (for logging)."""
        if len(self._episode_returns) < 2:
            return 0.0
        r = np.array(self._episode_returns)
        downside = r[r < 0]
        if len(downside) == 0:
            return float(r.mean() * np.sqrt(252) / 1e-8)
        downside_std = np.sqrt(np.mean(downside ** 2))
        return float(r.mean() / downside_std * np.sqrt(252)) if downside_std > 1e-8 else 0.0