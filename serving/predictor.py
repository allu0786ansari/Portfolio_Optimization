"""Prediction logic: loads SB3 model from disk, returns portfolio weights."""
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from loguru import logger

from data.config import ALL_TICKERS
from models.forecasting.dataset import FEATURE_COLS
from models.rl_agent.data_loader import load_aligned_features
from models.rl_agent.portfolio_env import softmax
from serving.model_loader import registry

SAVED_MODELS_DIR = Path("models/rl_agent/saved_models")

# Module-level cache — loaded once at first request
_features_dict  = None
_tickers_list   = None
_feature_matrix = None
_sb3_model      = None
_sb3_algo       = None


def _load_features() -> None:
    global _features_dict, _tickers_list, _feature_matrix
    if _features_dict is not None:
        return
    logger.info("Loading feature data...")
    _features_dict, _tickers_list, _ = load_aligned_features(ALL_TICKERS)
    _feature_matrix = np.stack(
        [_features_dict[t][FEATURE_COLS].values for t in _tickers_list],
        axis=1,
    ).astype(np.float32)
    logger.info(f"Features loaded: {len(_tickers_list)} tickers")


def _load_model() -> None:
    """Load SB3 model from local zip file. Much faster than MLflow download."""
    global _sb3_model, _sb3_algo
    if _sb3_model is not None:
        return

    from stable_baselines3 import PPO, SAC

    # Prefer the champion algo — check registry
    algo = registry.algo.lower() if registry.algo else "ppo"
    zip_path = SAVED_MODELS_DIR / f"{algo}_agent.zip"

    # Fallback: try both
    if not zip_path.exists():
        for fallback in ("ppo", "sac"):
            fp = SAVED_MODELS_DIR / f"{fallback}_agent.zip"
            if fp.exists():
                zip_path = fp
                algo = fallback
                break

    if not zip_path.exists():
        raise FileNotFoundError(
            f"No saved model found in {SAVED_MODELS_DIR}. "
            "Run: python save_models.py"
        )

    ModelClass = PPO if algo == "ppo" else SAC
    _sb3_model = ModelClass.load(str(zip_path))
    _sb3_algo  = algo.upper()
    logger.info(f"Loaded {_sb3_algo} model from {zip_path}")


def predict_weights(requested_tickers: list[str]) -> dict:
    """Run champion model and return portfolio weights."""
    t_start = time.perf_counter()

    _load_features()
    _load_model()

    # Filter to known tickers
    available     = set(_tickers_list)
    valid_tickers = [t for t in requested_tickers if t in available]
    missing       = [t for t in requested_tickers if t not in available]

    if missing:
        logger.warning(f"Tickers not in feature store (skipped): {missing}")

    if len(valid_tickers) < 2:
        raise ValueError(
            f"Need at least 2 valid tickers. Got: {valid_tickers}. "
            f"Missing from feature store: {missing}"
        )

    ticker_indices  = [_tickers_list.index(t) for t in valid_tickers]
    n_valid         = len(valid_tickers)
    latest_t        = _feature_matrix.shape[0] - 1

    # Build observation for full asset universe (model expects obs_dim=440)
    features_flat   = np.clip(
        _feature_matrix[latest_t].flatten(), -10.0, 10.0
    )
    current_weights = np.ones(len(_tickers_list), dtype=np.float32) / len(_tickers_list)
    obs             = np.concatenate([features_flat, current_weights])

    # SB3 predict — works identically for PPO and SAC
    action, _ = _sb3_model.predict(obs, deterministic=True)  # shape (40,)

    # Extract weights for requested tickers only
    action_subset = action[ticker_indices]
    weights       = softmax(action_subset.astype(np.float32))

    latency_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        f"predict | tickers={n_valid} | "
        f"latency={latency_ms:.1f}ms | algo={_sb3_algo}"
    )

    return {
        "weights":         dict(zip(valid_tickers, weights.tolist())),
        "weights_sum":     float(weights.sum()),
        "model_version":   registry.version,
        "algo":            _sb3_algo or registry.algo,
        "latency_ms":      round(latency_ms, 2),
        "n_assets":        n_valid,
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "skipped_tickers": missing,
    }
