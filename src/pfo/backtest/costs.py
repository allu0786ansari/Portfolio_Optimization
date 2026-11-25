"""Transaction cost models (Phase 0: linear; Phase 1 adds spread + square-root impact)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearCostModel:
    """Cost = ``bps`` basis points per unit of one-way turnover.

    ``turnover = sum(|w_target - w_drifted|)``, so fully replacing a portfolio
    has turnover 2.0 and costs ``2 * bps``. v1's RL reward used 10 bps.
    """

    bps: float = 10.0

    def cost(self, w_from: np.ndarray, w_to: np.ndarray) -> tuple[float, float]:
        turnover = float(np.abs(np.asarray(w_to) - np.asarray(w_from)).sum())
        return turnover, turnover * self.bps / 1e4
