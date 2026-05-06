"""ARIMA baseline model — trained per ticker, results logged to MLflow."""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
warnings.filterwarnings("ignore")

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from statsmodels.tsa.arima.model import ARIMA

from data.config import ALL_TICKERS, PROCESSED_DIR

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "arima_baseline"

# ARIMA(p,d,q) order — (1,0,1) is a standard starting point for returns
ARIMA_ORDER = (1, 0, 1)
TRAIN_FRAC  = 0.7
VAL_FRAC    = 0.15


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions where the sign matches the actual return."""
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def run_arima_for_ticker(ticker: str) -> dict | None:
    """Fit ARIMA on train split, evaluate on val split, return metrics dict."""
    path = PROCESSED_DIR / f"features_{ticker.replace('/', '_')}.parquet"
    if not path.exists():
        logger.warning(f"No feature file for {ticker}")
        return None

    df    = pd.read_parquet(path)
    n     = len(df)
    t_end = int(n * TRAIN_FRAC)
    v_end = int(n * (TRAIN_FRAC + VAL_FRAC))

    train_returns = df["log_return"].iloc[:t_end].values
    val_returns   = df["log_return"].iloc[t_end:v_end].values

    try:
        model  = ARIMA(train_returns, order=ARIMA_ORDER)
        fitted = model.fit()

        # One-step-ahead forecasts on validation set
        preds = []
        history = list(train_returns)
        for actual in val_returns:
            fc_model = ARIMA(history, order=ARIMA_ORDER)
            fc_fit   = fc_model.fit()
            preds.append(fc_fit.forecast(steps=1)[0])
            history.append(actual)

        preds = np.array(preds)
        rmse  = float(np.sqrt(np.mean((val_returns - preds) ** 2)))
        mae   = float(np.mean(np.abs(val_returns - preds)))
        dir_acc = directional_accuracy(val_returns, preds)

        return {
            "ticker": ticker,
            "val_rmse": rmse,
            "val_mae": mae,
            "val_directional_acc": dir_acc,
            "aic": float(fitted.aic),
            "n_train": t_end,
            "n_val": v_end - t_end,
        }

    except Exception as e:
        logger.error(f"ARIMA failed for {ticker}: {e}")
        return None


def run_all_arima() -> pd.DataFrame:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    results = []
    for ticker in ALL_TICKERS:
        logger.info(f"ARIMA: {ticker}")
        with mlflow.start_run(run_name=ticker):
            metrics = run_arima_for_ticker(ticker)
            if metrics:
                mlflow.log_param("ticker",      metrics["ticker"])
                mlflow.log_param("arima_order", str(ARIMA_ORDER))
                mlflow.log_param("n_train",     metrics["n_train"])
                mlflow.log_metric("val_rmse",           metrics["val_rmse"])
                mlflow.log_metric("val_mae",            metrics["val_mae"])
                mlflow.log_metric("val_directional_acc",metrics["val_directional_acc"])
                mlflow.log_metric("aic",                metrics["aic"])
                results.append(metrics)

    df = pd.DataFrame(results)
    avg_dir_acc = df["val_directional_acc"].mean()
    logger.info(f"ARIMA baseline — avg directional accuracy: {avg_dir_acc:.3f}")
    return df


if __name__ == "__main__":
    logger.info("=== Running ARIMA baseline ===")
    results = run_all_arima()
    print(results[["ticker","val_rmse","val_directional_acc"]].to_string(index=False))
    logger.info("=== ARIMA baseline complete ===")