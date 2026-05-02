"""Compute equal-weight and benchmark baseline metrics for comparison."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from loguru import logger

from models.rl_agent.data_loader import load_aligned_features, build_return_matrix
from models.rl_agent.evaluate_agent import compute_metrics
from models.classical.markowitz import MarkowitzOptimiser
from data.config import ALL_TICKERS


def equal_weight_metrics(
    return_matrix: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> dict:
    """Simulate equal-weight (1/N) portfolio over given window."""
    n = return_matrix.shape[1]
    weights = np.ones(n) / n
    returns = [float(np.dot(weights, return_matrix[t]))
               for t in range(start_idx, end_idx)]
    m = compute_metrics(returns)
    m["strategy"] = "Equal-Weight (1/N)"
    return m


def markowitz_metrics(
    return_matrix: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> dict:
    """Simulate rolling Markowitz optimiser over given window."""
    opt = MarkowitzOptimiser(lookback_days=252)
    port_returns, _ = opt.simulate(return_matrix, start_idx, end_idx)
    m = compute_metrics(port_returns.tolist())
    m["strategy"] = "Markowitz (MVO)"
    return m


def compute_all_baselines(
    tickers: list[str] = ALL_TICKERS,
    val_frac_start: float = 0.70,
    val_frac_end: float   = 0.85,
) -> list[dict]:
    """Compute all baseline metrics on the validation window."""
    features_dict, valid_tickers, dates = load_aligned_features(tickers)
    return_matrix = build_return_matrix(features_dict, valid_tickers)

    T = len(dates)
    start_idx = int(T * val_frac_start)
    end_idx   = int(T * val_frac_end)

    logger.info(f"Baseline eval window: {dates[start_idx].date()} to {dates[end_idx-1].date()}")

    results = []
    results.append(equal_weight_metrics(return_matrix, start_idx, end_idx))
    results.append(markowitz_metrics(return_matrix, start_idx, end_idx))
    return results