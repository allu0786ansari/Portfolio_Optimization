"""Model registry loader with hot-reload capability.

The champion model is loaded once at startup.
A background thread polls the MLflow registry every
POLL_INTERVAL_SECONDS and hot-reloads if a new champion
is promoted — zero downtime, zero server restart needed.

This is the MLOps pattern that Week 9 (Airflow retraining)
will trigger automatically.
"""

import socket
socket.setdefaulttimeout(10)   # 10 second timeout for all network calls

import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
import mlflow.pytorch
from loguru import logger

MLFLOW_URI            = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME            = "PortfolioAgent"
POLL_INTERVAL_SECONDS = 300   # check for new champion every 5 minutes


class ModelRegistry:
    """Thread-safe champion model holder with background polling."""

    def __init__(self) -> None:
        self._policy      = None
        self._version     : str  = "unknown"
        self._algo        : str  = "unknown"
        self._lock        = threading.RLock()
        self._loaded      : bool = False
        self._poll_thread : threading.Thread | None = None
        # NOTE: mlflow.set_tracking_uri is NOT called here.
        # It is called lazily inside _load_champion() and _poll_loop()
        # so that importing this module does not trigger any network calls.
        # This is required for test_api.py to import cleanly in CI.

    # ── Public interface ────────────────────────────────────────────

    def load(self) -> None:
        """Load champion model synchronously. Called at server startup."""
        self._load_champion()
        self._start_polling()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def version(self) -> str:
        with self._lock:
            return self._version

    @property
    def algo(self) -> str:
        with self._lock:
            return self._algo

    def predict(self, obs_array) -> object:
        """Run inference. Thread-safe. Handles both PPO and SAC policies."""
        with self._lock:
            if self._policy is None:
                raise RuntimeError("Model not loaded yet")
            import torch
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs_array).unsqueeze(0)

                if hasattr(self._policy, "actor"):
                    # SAC policy
                    action = self._policy.actor(obs_tensor).squeeze(0).numpy()
                elif hasattr(self._policy, "_predict"):
                    # PPO ActorCriticPolicy
                    action = self._policy._predict(obs_tensor, deterministic=True)
                    action = action.squeeze(0).numpy()
                else:
                    # Fallback: manual forward pass
                    features = self._policy.extract_features(obs_tensor)
                    latent   = self._policy.mlp_extractor(features)[0]
                    action   = self._policy.action_net(latent).squeeze(0).numpy()
            return action

    # ── Internal ────────────────────────────────────────────────────

    def _load_champion(self) -> None:
        """Load the champion model from MLflow registry."""
        mlflow.set_tracking_uri(MLFLOW_URI)   # lazy — only called here
        try:
            client = mlflow.tracking.MlflowClient()
            mv     = client.get_model_version_by_alias(MODEL_NAME, "champion")
            run    = client.get_run(mv.run_id)
            algo   = run.data.params.get("algo", "unknown").upper()
            uri    = f"models:/{MODEL_NAME}@champion"

            logger.info(f"Loading champion: {MODEL_NAME} v{mv.version} ({algo})")
            policy = mlflow.pytorch.load_model(uri)

            with self._lock:
                self._policy  = policy
                self._version = f"champion-v{mv.version}"
                self._algo    = algo
                self._loaded  = True

            logger.info(f"Champion loaded: {self._version} ({self._algo})")

        except Exception as e:
            logger.error(f"Failed to load champion model: {e}")
            self._loaded = False

    def _poll_loop(self) -> None:
        """Background thread: poll MLflow for a new champion every N seconds."""
        while True:
            time.sleep(POLL_INTERVAL_SECONDS)
            mlflow.set_tracking_uri(MLFLOW_URI)   # lazy — only called here
            try:
                client  = mlflow.tracking.MlflowClient()
                mv      = client.get_model_version_by_alias(MODEL_NAME, "champion")
                new_ver = f"champion-v{mv.version}"

                if new_ver != self._version:
                    logger.info(f"New champion detected: {new_ver} — hot-reloading...")
                    self._load_champion()
                    logger.info(f"Hot-reload complete: {self._version}")

            except Exception as e:
                logger.warning(f"Model poll failed: {e}")

    def _start_polling(self) -> None:
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="model-poll"
        )
        self._poll_thread.start()
        logger.info(f"Model polling started (every {POLL_INTERVAL_SECONDS}s)")


# Singleton — shared across all FastAPI request handlers
# Importing this module does NOT trigger any network calls.
# Call registry.load() explicitly at server startup (inside lifespan).
registry = ModelRegistry()