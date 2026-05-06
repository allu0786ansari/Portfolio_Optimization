"""Optuna hyperparameter tuning for the LSTM forecaster.

Runs N trials, each training a single LSTM on RELIANCE.NS as a proxy ticker.
Best hyperparameters are logged to MLflow and printed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import warnings  # noqa: E402, I001

warnings.filterwarnings("ignore")

import mlflow  # noqa: E402, I001
import optuna  # noqa: E402, I001
import torch  # noqa: E402, I001
from loguru import logger  # noqa: E402, I001

from models.forecasting.train_forecaster import MLFLOW_TRACKING_URI, train_ticker  # noqa: E402, I001

PROXY_TICKER = "RELIANCE.NS"   # tune on one liquid stock as proxy
N_TRIALS     = 20


def objective(trial: optuna.Trial) -> float:
    """Optuna objective — minimise validation RMSE."""
    hp = {
        "seq_len":     trial.suggest_categorical("seq_len",    [20, 30, 60]),
        "hidden_size": trial.suggest_categorical("hidden_size",[32, 64, 128]),
        "num_layers":  trial.suggest_int("num_layers",          1, 3),
        "dropout":     trial.suggest_float("dropout",           0.1, 0.5),
        "fc_hidden":   trial.suggest_categorical("fc_hidden",  [16, 32, 64]),
        "lr":          trial.suggest_float("lr",                1e-4, 1e-2, log=True),
        "batch_size":  trial.suggest_categorical("batch_size", [16, 32, 64]),
        "max_epochs":  50,
        "patience":    7,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = train_ticker(PROXY_TICKER, hp, device)
    if result is None:
        return float("inf")
    return result["val_rmse"]


def run_tuning() -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("lstm_hparam_tuning")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_params
    best["max_epochs"] = 50
    best["patience"]   = 7

    logger.info(f"Best hyperparameters: {best}")
    logger.info(f"Best val RMSE: {study.best_value:.6f}")

    with mlflow.start_run(run_name="best_hparams"):
        mlflow.log_params(best)
        mlflow.log_metric("best_val_rmse", study.best_value)

    return best


if __name__ == "__main__":
    logger.info(f"=== Optuna tuning — {N_TRIALS} trials on {PROXY_TICKER} ===")
    best_hp = run_tuning()
    print("\nBest hyperparameters found:")
    for k, v in best_hp.items():
        print(f"  {k}: {v}")
    print("\nNow re-run train_forecaster.py with these hyperparameters.")