"""Prediction logic: converts tickers to observations, runs model, returns weights.

This module is intentionally separate from main.py so it can be
unit-tested without starting the HTTP server.
"""
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from loguru import logger

from data.config import ALL_TICKERS
from models.rl_agent.data_loader import load_aligned_features
from models.rl_agent.portfolio_env import softmax
from models.forecasting.dataset import FEATURE_COLS
from serving.model_loader import registry


# Cache aligned feature data — loaded once at startup
_features_dict = None
_tickers_list  = None
_feature_matrix = None


def _ensure_data_loaded() -> None:
    global _features_dict, _tickers_list, _feature_matrix
    if _features_dict is None:
        logger.info("Loading feature data for predictor...")
        _features_dict, _tickers_list, _ = load_aligned_features(ALL_TICKERS)
        _feature_matrix = np.stack(
            [_features_dict[t][FEATURE_COLS].values for t in _tickers_list],
            axis=1,
        ).astype(np.float32)
        logger.info(f"Predictor data loaded: {len(_tickers_list)} tickers")


def predict_weights(
    requested_tickers: list[str],
) -> dict:
    """Run champion model and return portfolio weights for requested tickers.

    Args:
        requested_tickers: list of ticker symbols to allocate across

    Returns:
        dict with weights, metadata, and latency
    """
    t_start = time.perf_counter()
    _ensure_data_loaded()

    # Filter to tickers we have feature data for
    available = set(_tickers_list)
    valid_tickers = [t for t in requested_tickers if t in available]
    missing       = [t for t in requested_tickers if t not in available]

    if missing:
        logger.warning(f"Tickers not in feature store (skipped): {missing}")

    if len(valid_tickers) < 2:
        raise ValueError(
            f"Need at least 2 valid tickers. Got: {valid_tickers}. "
            f"Missing from feature store: {missing}"
        )

    # Build observation using the latest available features
    # Get indices of valid tickers in the full feature matrix
    ticker_indices = [_tickers_list.index(t) for t in valid_tickers]
    n_valid = len(valid_tickers)

    # Latest timestep features for requested tickers: (N_valid, F)
    latest_t         = _feature_matrix.shape[0] - 1
    features_subset  = _feature_matrix[latest_t][ticker_indices]   # (N_valid, F)
    features_flat    = np.clip(features_subset.flatten(), -10.0, 10.0)

    # Equal weight as initial portfolio weight signal
    current_weights  = np.ones(n_valid, dtype=np.float32) / n_valid

    obs = np.concatenate([features_flat, current_weights])

    # Pad or trim obs to match model's expected input size
    # Model was trained on all tickers — obs dim = N_all * 11
    n_all    = len(_tickers_list)
    obs_dim  = n_all * (len(FEATURE_COLS) + 1)

    if len(obs) < obs_dim:
        obs = np.pad(obs, (0, obs_dim - len(obs)))
    elif len(obs) > obs_dim:
        obs = obs[:obs_dim]

    # Inference
    action  = registry.predict(obs)

    # Project only the relevant slice of action to valid tickers
    action_subset = action[ticker_indices] if len(action) >= max(ticker_indices) + 1                     else action[:n_valid]
    weights = softmax(action_subset.astype(np.float32))

    latency_ms = (time.perf_counter() - t_start) * 1000

    result = {
        "weights":       dict(zip(valid_tickers, weights.tolist())),
        "weights_sum":   float(weights.sum()),
        "model_version": registry.version,
        "algo":          registry.algo,
        "latency_ms":    round(latency_ms, 2),
        "n_assets":      n_valid,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "skipped_tickers": missing,
    }

    logger.info(
        f"predict | tickers={len(valid_tickers)} | "
        f"latency={latency_ms:.1f}ms | version={registry.version}"
    )
    return result