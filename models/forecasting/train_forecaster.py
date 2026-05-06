"""Training loop for the LSTM return forecaster with MLflow logging."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from data.config import ALL_TICKERS, PROCESSED_DIR
from models.forecasting.dataset import FEATURE_COLS, make_dataloaders
from models.forecasting.lstm_model import ReturnLSTM

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME     = "lstm_forecaster"

# Default hyperparameters (overridden by Optuna in tune.py)
DEFAULT_HP = {
    "seq_len":     30,
    "hidden_size": 64,
    "num_layers":  2,
    "dropout":     0.2,
    "fc_hidden":   32,
    "lr":          1e-3,
    "batch_size":  32,
    "max_epochs":  50,
    "patience":    7,       # early stopping patience
}


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def train_one_epoch(
    model: ReturnLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


def evaluate(
    model: ReturnLSTM,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Returns (loss, rmse, directional_accuracy)."""
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            total_loss  += loss.item() * len(X_batch)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    preds_arr   = np.array(all_preds)
    targets_arr = np.array(all_targets)
    rmse    = float(np.sqrt(np.mean((targets_arr - preds_arr) ** 2)))
    dir_acc = directional_accuracy(targets_arr, preds_arr)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, rmse, dir_acc


def train_ticker(
    ticker: str,
    hp: dict = DEFAULT_HP,
    device: torch.device = torch.device("cpu"),
) -> dict | None:
    """Train LSTM for one ticker and return metrics dict."""
    path = PROCESSED_DIR / f"features_{ticker.replace('/', '_')}.parquet"
    if not path.exists():
        logger.warning(f"No feature file for {ticker}, skipping")
        return None

    df = pd.read_parquet(path)
    if len(df) < hp["seq_len"] + 50:
        logger.warning(f"{ticker}: not enough rows, skipping")
        return None

    train_loader, val_loader, test_loader, scaler = make_dataloaders(
        df,
        seq_len     = hp["seq_len"],
        batch_size  = hp["batch_size"],
        train_frac  = 0.7,
        val_frac    = 0.15,
    )

    model = ReturnLSTM(
        input_size  = len(FEATURE_COLS),
        hidden_size = hp["hidden_size"],
        num_layers  = hp["num_layers"],
        dropout     = hp["dropout"],
        fc_hidden   = hp["fc_hidden"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    best_val_loss  = float("inf")
    best_state     = None
    patience_count = 0

    for epoch in range(hp["max_epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_rmse, val_dir_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= hp["patience"]:
            logger.debug(f"{ticker} early stop at epoch {epoch+1}")
            break

    # Reload best weights and evaluate on test set
    if best_state:
        model.load_state_dict(best_state)
    _, test_rmse, test_dir_acc = evaluate(model, test_loader, criterion, device)
    _, val_rmse_best, val_dir_acc_best = evaluate(model, val_loader, criterion, device)

    return {
        "ticker":            ticker,
        "val_rmse":          val_rmse_best,
        "val_directional_acc": val_dir_acc_best,
        "test_rmse":         test_rmse,
        "test_directional_acc": test_dir_acc,
        "n_params":          model.count_parameters(),
        "model":             model,
        "scaler":            scaler,
    }


def train_all_tickers(hp: dict = DEFAULT_HP) -> pd.DataFrame:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    results = []
    for ticker in ALL_TICKERS:
        logger.info(f"Training LSTM: {ticker}")
        with mlflow.start_run(run_name=ticker):
            # Log hyperparameters
            mlflow.log_param("ticker",      ticker)
            mlflow.log_params({k: v for k, v in hp.items() if k != "model"})

            result = train_ticker(ticker, hp, device)
            if result is None:
                continue

            # Log metrics
            mlflow.log_metric("val_rmse",              result["val_rmse"])
            mlflow.log_metric("val_directional_acc",   result["val_directional_acc"])
            mlflow.log_metric("test_rmse",             result["test_rmse"])
            mlflow.log_metric("test_directional_acc",  result["test_directional_acc"])
            mlflow.log_metric("n_params",              result["n_params"])

            # Log model artifact
            mlflow.pytorch.log_model(result["model"], artifact_path="lstm_model")

            results.append({k: v for k, v in result.items() if k not in ("model", "scaler")})

    df = pd.DataFrame(results)
    logger.info("=== LSTM training summary ===")
    logger.info(f"Avg val directional accuracy:  {df['val_directional_acc'].mean():.3f}")
    logger.info(f"Avg test directional accuracy: {df['test_directional_acc'].mean():.3f}")
    return df


if __name__ == "__main__":
    logger.info("=== Starting LSTM training ===")
    results = train_all_tickers()
    print(results[["ticker","val_rmse","val_directional_acc","test_directional_acc"]].to_string(index=False))