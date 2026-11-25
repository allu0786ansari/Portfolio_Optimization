"""Strategy interface with structurally enforced information sets.

The walk-forward runner hands each strategy:
* ``fit(TrainData)`` — rows of the training window only;
* ``decide(History)`` — rows ``<= t`` only (the decision day).

Because strategies never receive arrays that extend beyond ``t``, look-ahead
is impossible by construction rather than by convention.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrainData:
    returns: np.ndarray  # (T_train, N) simple returns of the training rows
    features: np.ndarray | None  # (T_train, N, F) raw (unscaled) features, or None
    dates: pd.DatetimeIndex
    tickers: tuple[str, ...]


@dataclass(frozen=True)
class History:
    t: int  # absolute row index of the decision day
    returns: np.ndarray  # rows <= t
    features_now: np.ndarray | None  # (N, F) raw features at row t
    prev_weights: np.ndarray  # target weights of the previous decision
    date: pd.Timestamp

    def __post_init__(self) -> None:
        if len(self.returns) != self.t + 1:
            raise ValueError("History.returns must contain exactly rows 0..t")


class Strategy(ABC):
    name: str = "strategy"

    def fit(self, data: TrainData) -> None:  # noqa: B027 - optional hook
        """Optional: estimate parameters on the training window."""

    @abstractmethod
    def decide(self, hist: History) -> np.ndarray:
        """Return long-only target weights (sum <= 1) for the decision day."""


def normalize_long_only(w: np.ndarray) -> np.ndarray:
    w = np.clip(np.nan_to_num(np.asarray(w, dtype=float), nan=0.0), 0.0, None)
    s = w.sum()
    return w / s if s > 0 else np.full_like(w, 1.0 / len(w))
