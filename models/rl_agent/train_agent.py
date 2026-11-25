"""Train PPO or SAC agent on the PortfolioEnv.

Usage:
    python -m models.rl_agent.train_agent --algo ppo
    python -m models.rl_agent.train_agent --algo sac

Training progress is logged to MLflow. The best model is registered
in the MLflow Model Registry with alias 'champion' if it beats the
current champion's validation Sharpe ratio by 5%.

Phase 0 fixes: corrected (leakage-free) environment on one universe; validation
uses rows 70-85% only (v1 "validated" on the full history, mostly training data);
the version registered by *this* run is promoted (v1 promoted the highest version
number); the env's serving spec is logged so serving can rebuild observations;
MLflow is reached via MLFLOW_TRACKING_URI.
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import mlflow
import mlflow.pytorch
import numpy as np
from loguru import logger
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from models.rl_agent.evaluate_agent import evaluate_agent
from models.rl_agent.portfolio_env import PortfolioEnv

MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
UNIVERSE        = os.getenv("TRAIN_UNIVERSE", "sp500")
TRAIN_FRAC, VAL_END_FRAC = 0.70, 0.85
EXPERIMENT_NAME = "rl_agent_training"
MODEL_NAME      = "PortfolioAgent"

# Training budgets
TIMESTEPS = {
    "ppo": 100_000,
    "sac": 150_000,
}

# Hyperparameters
PPO_PARAMS = {
    "learning_rate":  3e-4,
    "n_steps":        2048,
    "batch_size":     64,
    "n_epochs":       10,
    "gamma":          0.99,
    "gae_lambda":     0.95,
    "clip_range":     0.2,
    "ent_coef":       0.01,
    "verbose":        1,
}

SAC_PARAMS = {
    "learning_rate":  3e-4,
    "buffer_size":    100_000,
    "learning_starts":5_000,
    "batch_size":     256,
    "tau":            0.005,
    "gamma":          0.99,
    "train_freq":     1,
    "ent_coef":       "auto",
    "verbose":        1,
}

CHAMPION_IMPROVEMENT_THRESHOLD = 0.05   # new model must beat champion by 5%


class MLflowCallback(BaseCallback):
    """Logs SB3 training metrics to MLflow every N steps."""

    def __init__(self, log_freq: int = 10_000):
        super().__init__()
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                ep_rewards = [ep["r"] for ep in self.model.ep_info_buffer]
                ep_lengths = [ep["l"] for ep in self.model.ep_info_buffer]
                mlflow.log_metric("mean_ep_reward", np.mean(ep_rewards), step=self.num_timesteps)
                mlflow.log_metric("mean_ep_length", np.mean(ep_lengths), step=self.num_timesteps)
        return True


def get_current_champion_sharpe(client) -> float | None:
    """Validation Sharpe of the current champion, or None if there is no champion."""
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
        run = client.get_run(mv.run_id)
        return float(run.data.metrics.get("val_sharpe", float("-inf")))
    except Exception:
        return None


def train(algo: str = "ppo", universe: str = UNIVERSE, timesteps: int | None = None) -> dict:
    algo = algo.lower()
    if algo not in ("ppo", "sac"):
        raise ValueError(f"Unknown algo: {algo}. Choose ppo or sac.")
    timesteps = timesteps or TIMESTEPS[algo]

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info(f"Initialising corrected PortfolioEnv[{universe}] for {algo.upper()} training...")
    base_train_env = PortfolioEnv(universe=universe, train_frac=TRAIN_FRAC, seed=42)
    train_env = Monitor(base_train_env)
    # Validation rows only (70%-85%); the scaler is still the training-rows scaler.
    eval_env = PortfolioEnv(universe=universe, train_frac=TRAIN_FRAC, start_frac=TRAIN_FRAC,
                            end_frac=VAL_END_FRAC, episode_length=63, seed=99)

    with mlflow.start_run(run_name=f"{algo.upper()}_{universe}") as run:
        params = PPO_PARAMS if algo == "ppo" else SAC_PARAMS
        mlflow.log_params({"algo": algo, "timesteps": timesteps, "universe": universe,
                           "train_frac": TRAIN_FRAC, "val_end_frac": VAL_END_FRAC,
                           "env_version": "v2-phase0", "execution_lag": base_train_env.lag,
                           "action_scale": base_train_env.scale})
        mlflow.log_params({f"hp_{k}": v for k, v in params.items() if k != "verbose"})

        ModelClass = PPO if algo == "ppo" else SAC
        model = ModelClass("MlpPolicy", train_env, **params, seed=42, device="cpu")
        logger.info(f"Training {algo.upper()} for {timesteps:,} timesteps...")
        model.learn(total_timesteps=timesteps, callback=MLflowCallback(log_freq=10_000), progress_bar=False)

        logger.info("Evaluating on the validation window (rows 70%-85%)...")
        metrics = evaluate_agent(model, eval_env, n_episodes=20, seed=200)
        mlflow.log_metrics({"val_sharpe": metrics["mean_sharpe"], "val_sortino": metrics["mean_sortino"],
                            "val_cagr": metrics["mean_cagr"], "val_max_dd": metrics["mean_max_dd"],
                            "val_mean_reward": metrics["mean_reward"]})

        spec = base_train_env.serving_spec()
        with tempfile.TemporaryDirectory() as tmp:
            spec_file = os.path.join(tmp, "serving_spec.json")
            with open(spec_file, "w", encoding="utf-8") as fh:
                json.dump(spec, fh, indent=2)
            mlflow.log_artifact(spec_file)

        # "pickle" keeps the full SB3 policy object (serving calls policy.predict); MLflow 3's
        # default "pt2" format is a traced graph that would drop .predict.
        model_info = mlflow.pytorch.log_model(model.policy, name="policy", registered_model_name=MODEL_NAME,
                                              serialization_format="pickle")
        version = getattr(model_info, "registered_model_version", None)

        client = mlflow.tracking.MlflowClient()
        champ = get_current_champion_sharpe(client)
        new_sharpe = metrics["mean_sharpe"]
        threshold = None if champ is None else champ + CHAMPION_IMPROVEMENT_THRESHOLD * abs(champ)
        promoted = version is not None and (threshold is None or new_sharpe > threshold)
        if promoted:
            client.set_registered_model_alias(MODEL_NAME, "champion", str(version))
            logger.info(f"NEW CHAMPION: {algo.upper()} v{version} (val Sharpe {new_sharpe:.3f}, previous {champ})")
        else:
            logger.info(f"Not promoted: val Sharpe {new_sharpe:.3f} vs champion {champ} (+5% rule)")
        mlflow.log_param("is_champion", promoted)

    train_env.close()
    eval_env.close()
    return {"run_id": run.info.run_id, "version": version, "val_sharpe": new_sharpe, "promoted": promoted}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="ppo", choices=["ppo", "sac"])
    parser.add_argument("--universe", default=UNIVERSE, choices=["nifty", "sp500"])
    parser.add_argument("--timesteps", type=int, default=None)
    args = parser.parse_args()
    print(train(args.algo, args.universe, args.timesteps))