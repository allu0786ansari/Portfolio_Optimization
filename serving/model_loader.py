"""Champion model holder with hot-reload (Phase 0 fixes).

* The champion policy *and* its ``serving_spec.json`` (universe, tickers,
  feature scaler, action scale) are loaded together from the MLflow run, so
  serving builds exactly the observation the policy was trained on.
* ``predict`` uses SB3's public ``policy.predict`` (which clips actions to the
  action space); v1 called the private ``_predict`` (unclipped) for PPO and a
  stochastic actor forward pass for SAC.
* A background thread polls the registry and hot-reloads a new champion.
  Because ``/predict`` reads from this registry (not from a zip on disk, as in
  v1), a reload really changes the served model.
"""

import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

socket.setdefaulttimeout(10)
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
import mlflow.pytorch
from loguru import logger

MODEL_NAME = "PortfolioAgent"
POLL_INTERVAL_SECONDS = int(os.getenv("MODEL_POLL_SECONDS", "300"))


def _tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


class ModelRegistry:
    """Thread-safe champion holder. Importing this module makes no network calls."""

    def __init__(self) -> None:
        self._policy = None
        self._spec: dict | None = None
        self._version = "unknown"
        self._algo = "unknown"
        self._lock = threading.RLock()
        self._poll_thread: threading.Thread | None = None

    # -- public ---------------------------------------------------------------
    def load(self) -> None:
        self._load_champion()
        self._start_polling()

    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self._policy is not None and self._spec is not None

    @property
    def version(self) -> str:
        with self._lock:
            return self._version

    @property
    def algo(self) -> str:
        with self._lock:
            return self._algo

    @property
    def spec(self) -> dict | None:
        with self._lock:
            return self._spec

    def predict(self, obs_array):
        with self._lock:
            if self._policy is None:
                raise RuntimeError("Model not loaded yet")
            action, _ = self._policy.predict(obs_array, deterministic=True)
            return action

    # -- internal -----------------------------------------------------------------
    def _load_champion(self) -> None:
        mlflow.set_tracking_uri(_tracking_uri())
        try:
            client = mlflow.tracking.MlflowClient()
            mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
            run = client.get_run(mv.run_id)
            algo = run.data.params.get("algo", "unknown").upper()
            spec_path = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="serving_spec.json")
            spec = json.loads(Path(spec_path).read_text())
            policy = mlflow.pytorch.load_model(f"models:/{MODEL_NAME}@champion")
            policy.set_training_mode(False)
            with self._lock:
                self._policy, self._spec = policy, spec
                self._version, self._algo = f"champion-v{mv.version}", algo
            logger.info(f"Champion loaded: {self._version} ({algo}, universe={spec.get('universe')})")
        except Exception as e:
            logger.error(f"Failed to load champion model: {e}")

    def _poll_loop(self) -> None:
        while True:
            time.sleep(POLL_INTERVAL_SECONDS)
            try:
                mlflow.set_tracking_uri(_tracking_uri())
                mv = mlflow.tracking.MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion")
                if f"champion-v{mv.version}" != self.version:
                    logger.info(f"New champion v{mv.version} detected - hot-reloading")
                    self._load_champion()
            except Exception as e:
                logger.warning(f"Model poll failed: {e}")

    def _start_polling(self) -> None:
        if self._poll_thread is None:
            self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True, name="model-poll")
            self._poll_thread.start()


registry = ModelRegistry()
