"""Saves trained PPO/SAC models as SB3 zip files for fast loading.
Run once after training: python save_models.py
"""
import sys
from pathlib import Path

sys.path.insert(0, ".")

import mlflow
from loguru import logger

MLFLOW_URI = "http://localhost:5000"
SAVE_DIR   = Path("models/rl_agent/saved_models")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(MLFLOW_URI)
client = mlflow.tracking.MlflowClient()

try:
    from stable_baselines3 import PPO, SAC

    # Find all rl_agent_training runs
    exp = client.get_experiment_by_name("rl_agent_training")
    if exp is None:
        print("No rl_agent_training experiment found. Train agents first.")
        raise SystemExit(1)

    runs = client.search_runs(exp.experiment_id)
    saved = []

    for run in runs:
        algo = run.data.params.get("algo", "").lower()
        if algo not in ("ppo", "sac"):
            continue

        save_path = SAVE_DIR / f"{algo}_agent"
        if save_path.with_suffix(".zip").exists():
            logger.info(f"Already saved: {save_path}.zip")
            saved.append(algo)
            continue

        try:
            # Download the policy artifact
            artifact_path = mlflow.artifacts.download_artifacts(
                run_id=run.info.run_id,
                artifact_path="policy",
            )
            logger.info(f"Downloaded artifact for {algo.upper()}: {artifact_path}")

            # Load and resave as SB3 zip using the environment
            from models.rl_agent.portfolio_env import PortfolioEnv
            env = PortfolioEnv(seed=0)

            ModelClass = PPO if algo == "ppo" else SAC

            # Load policy weights into a fresh model
            policy_model = mlflow.pytorch.load_model(
                f"runs:/{run.info.run_id}/policy"
            )

            # Create fresh SB3 model and copy policy state
            fresh_model = ModelClass("MlpPolicy", env, verbose=0)
            fresh_model.policy.load_state_dict(policy_model.state_dict())
            fresh_model.save(str(save_path))
            env.close()

            logger.info(f"Saved: {save_path}.zip")
            saved.append(algo)

        except Exception as e:
            logger.error(f"Failed to save {algo.upper()}: {e}")

    if saved:
        print(f"\nSaved models: {saved}")
        print(f"Location: {SAVE_DIR.resolve()}")
        print("\nRestart FastAPI server:")
        print("  uvicorn serving.main:app --host 0.0.0.0 --port 8000 --reload")
    else:
        print("No models were saved successfully.")
        print("Try retraining: python -m models.rl_agent.train_agent --algo ppo")

except Exception as e:
    logger.error(f"Error: {e}")
    import traceback
    traceback.print_exc()