"""Prediction logic with ONE model path (Phase 0 fix for v1 defect #8).

Model source, in order:
1. the MLflow champion (policy + ``serving_spec.json``), hot-reloaded by ``registry``;
2. a local fallback ``models/rl_agent/saved_models/<algo>_agent.zip`` **only if** a
   matching ``<algo>_agent.spec.json`` exists (written by ``save_models.py``).

The v1 zip files have no spec: they were trained in the leaky v1 environment and
are deliberately not served.

The observation is rebuilt exactly as in training: causal features for the
model's universe, standardized with the training-fold scaler stored in the
spec, plus the previous weights (equal weights at inference time).
"""

import json
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from loguru import logger

from pfo.data.market import load_v1_universe
from pfo.features.basic import FEATURE_NAMES, compute_features, to_tensor
from pfo.rl.env import project_action
from serving.model_loader import registry

SAVED_MODELS_DIR = Path("models/rl_agent/saved_models")
_lock = threading.Lock()
_feature_cache: dict[str, tuple[pd.DatetimeIndex, np.ndarray, list[str]]] = {}
_local: dict | None = None


def _local_model() -> dict | None:
    global _local
    if _local is not None:
        return _local
    for algo in ("ppo", "sac"):
        zip_path, spec_path = SAVED_MODELS_DIR / f"{algo}_agent.zip", SAVED_MODELS_DIR / f"{algo}_agent.spec.json"
        if zip_path.exists() and spec_path.exists():
            from stable_baselines3 import PPO, SAC

            model = (PPO if algo == "ppo" else SAC).load(str(zip_path), device="cpu")
            _local = {"model": model, "spec": json.loads(spec_path.read_text(encoding="utf-8")), "algo": algo.upper(),
                      "version": f"local-{algo}"}
            logger.info(f"Local fallback model loaded from {zip_path}")
            return _local
    return None


def serving_source() -> dict | None:
    """Describe what would serve a request right now (None = nothing can)."""
    if registry.is_loaded:
        return {"kind": "mlflow-registry", "version": registry.version, "algo": registry.algo}
    local = _local_model()
    if local:
        return {"kind": "local-zip", "version": local["version"], "algo": local["algo"]}
    return None


def _features(universe: str) -> tuple[pd.DatetimeIndex, np.ndarray, list[str]]:
    with _lock:
        if universe not in _feature_cache:
            market = load_v1_universe(universe)
            x = to_tensor(compute_features(market.prices, market.benchmark))
            _feature_cache[universe] = (market.dates, x, market.tickers)
        return _feature_cache[universe]


def predict_weights(requested_tickers: list[str], as_of: str | None = None) -> dict:
    t0 = time.perf_counter()
    if registry.is_loaded:
        spec, predict, version, algo = registry.spec, registry.predict, registry.version, registry.algo
    else:
        local = _local_model()
        if local is None:
            raise RuntimeError("no model available")
        spec, version, algo = local["spec"], local["version"], local["algo"]
        predict = lambda obs: local["model"].predict(obs, deterministic=True)[0]  # noqa: E731

    dates, x, tickers = _features(spec["universe"])
    if list(tickers) != list(spec["tickers"]):
        raise ValueError("feature universe differs from the model's training universe; retrain the model")
    if as_of:
        row = int(dates.searchsorted(pd.Timestamp(as_of), side="right") - 1)
        if row < 0:
            raise ValueError(f"no data on or before {as_of}")
    else:
        row = len(dates) - 1
    mean, std = np.asarray(spec["scaler_mean"]), np.asarray(spec["scaler_std"])
    z = np.clip(np.nan_to_num((x[row] - mean) / std), -5, 5)
    n = len(tickers)
    obs = np.concatenate([z.ravel(), np.full(n, 1.0 / n)]).astype(np.float32)
    w_full = project_action(np.asarray(predict(obs)), float(spec["action_scale"]))

    valid = [t for t in requested_tickers if t in tickers]
    missing = [t for t in requested_tickers if t not in tickers]
    if len(valid) < 2:
        raise ValueError(f"Need at least 2 tickers from the model universe {spec['universe']}; got {valid}")
    sub = np.array([w_full[tickers.index(t)] for t in valid])
    sub = sub / sub.sum()
    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "weights": dict(zip(valid, sub.round(6).tolist(), strict=True)),
        "weights_sum": float(sub.sum()),
        "model_version": version,
        "algo": algo,
        "latency_ms": round(latency_ms, 2),
        "n_assets": len(valid),
        "timestamp": datetime.now(UTC).isoformat(),
        "as_of_date": str(dates[row].date()),
        "skipped_tickers": missing,
        "feature_names": list(FEATURE_NAMES),
    }
