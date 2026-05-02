"""Train PPO or SAC agent on the PortfolioEnv.

Usage:
    python -m models.rl_agent.train_agent --algo ppo
    python -m models.rl_agent.train_agent --algo sac

Training progress is logged to MLflow. The best model is registered
in the MLflow Model Registry with alias 'champion' if it beats the
current champion's validation Sharpe ratio by 5%.
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import mlflow
import mlflow.pytorch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from loguru import logger

from models.rl_agent.portfolio_env import PortfolioEnv
from models.rl_agent.evaluate_agent import evaluate_agent


MLFLOW_URI      = "http://localhost:5000"
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


def get_current_champion_sharpe(client) -> float:
    """Fetch validation Sharpe of current champion from MLflow registry."""
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
        run = client.get_run(mv.run_id)
        return float(run.data.metrics.get("val_sharpe", 0.0))
    except Exception:
        return 0.0   # no champion yet


def train(algo: str = "ppo") -> None:
    algo = algo.lower()
    assert algo in ("ppo", "sac"), f"Unknown algo: {algo}. Choose ppo or sac."

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info(f"Initialising PortfolioEnv for {algo.upper()} training...")
    train_env = Monitor(PortfolioEnv(seed=42))
    eval_env  = PortfolioEnv(seed=99, train_frac=1.0)   # full data for eval

    with mlflow.start_run(run_name=f"{algo.upper()}_run") as run:
        # Log hyperparameters
        params = PPO_PARAMS if algo == "ppo" else SAC_PARAMS
        mlflow.log_param("algo",       algo)
        mlflow.log_param("timesteps",  TIMESTEPS[algo])
        mlflow.log_params({f"hp_{k}": v for k, v in params.items() if k != "verbose"})

        # Build model
        ModelClass = PPO if algo == "ppo" else SAC
        model = ModelClass(
            "MlpPolicy",
            train_env,
            **{k: v for k, v in params.items()},
            seed=42,
        )

        logger.info(f"Training {algo.upper()} for {TIMESTEPS[algo]:,} timesteps...")
        model.learn(
            total_timesteps = TIMESTEPS[algo],
            callback        = MLflowCallback(log_freq=10_000),
            progress_bar    = True,
        )

        # Evaluate on validation episodes
        logger.info("Evaluating on validation window...")
        metrics = evaluate_agent(model, eval_env, n_episodes=20, seed=200)

        mlflow.log_metric("val_sharpe",    metrics["mean_sharpe"])
        mlflow.log_metric("val_sortino",   metrics["mean_sortino"])
        mlflow.log_metric("val_cagr",      metrics["mean_cagr"])
        mlflow.log_metric("val_max_dd",    metrics["mean_max_dd"])
        mlflow.log_metric("val_mean_reward", metrics["mean_reward"])

        logger.info(
            f"{algo.upper()} results — "
            f"Sharpe={metrics['mean_sharpe']:.3f} | "
            f"Sortino={metrics['mean_sortino']:.3f} | "
            f"CAGR={metrics['mean_cagr']:.1%} | "
            f"MaxDD={metrics['mean_max_dd']:.1%}"
        )

        # Log model artifact
        mlflow.pytorch.log_model(
            model.policy,
            name="policy",
            registered_model_name=MODEL_NAME,
        )

        # Champion-challenger: promote if beats current champion by 5%
        client = mlflow.tracking.MlflowClient()
        current_champion_sharpe = get_current_champion_sharpe(client)
        new_sharpe               = metrics["mean_sharpe"]
        threshold                = current_champion_sharpe * (1 + CHAMPION_IMPROVEMENT_THRESHOLD)

        if new_sharpe > threshold or current_champion_sharpe == 0.0:
            # Find this run's model version and promote
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            latest = sorted(versions, key=lambda v: int(v.version))[-1]
            client.set_registered_model_alias(MODEL_NAME, "champion", latest.version)
            mlflow.log_param("is_champion", True)
            logger.info(
                f"NEW CHAMPION: {algo.upper()} v{latest.version} "
                f"(Sharpe {new_sharpe:.3f} > {current_champion_sharpe:.3f})"
            )
        else:
            mlflow.log_param("is_champion", False)
            logger.info(
                f"Not promoted: {new_sharpe:.3f} did not beat "
                f"champion {current_champion_sharpe:.3f} by 5%"
            )

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="ppo", choices=["ppo", "sac"])
    args = parser.parse_args()
    train(args.algo)