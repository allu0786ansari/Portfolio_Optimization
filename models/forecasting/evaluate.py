"""Compare LSTM vs ARIMA — prints a side-by-side summary table."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import mlflow
from loguru import logger

MLFLOW_TRACKING_URI = "http://localhost:5000"


def fetch_experiment_results(experiment_name: str) -> pd.DataFrame:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        logger.warning(f"Experiment '{experiment_name}' not found in MLflow")
        return pd.DataFrame()

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.val_directional_acc DESC"],
    )

    rows = []
    for r in runs:
        rows.append({
            "ticker":      r.data.params.get("ticker", r.info.run_name),
            "val_rmse":    r.data.metrics.get("val_rmse", None),
            "val_dir_acc": r.data.metrics.get("val_directional_acc", None),
        })
    return pd.DataFrame(rows)


def compare_models() -> None:
    lstm  = fetch_experiment_results("lstm_forecaster")
    arima = fetch_experiment_results("arima_baseline")

    if lstm.empty or arima.empty:
        logger.error("Run arima_baseline.py and train_forecaster.py first")
        return

    lstm  = lstm.rename(columns={"val_rmse": "lstm_rmse",  "val_dir_acc": "lstm_dir_acc"})
    arima = arima.rename(columns={"val_rmse": "arima_rmse", "val_dir_acc": "arima_dir_acc"})

    merged = lstm.merge(arima, on="ticker", how="inner")
    merged["lstm_wins"] = merged["lstm_dir_acc"] > merged["arima_dir_acc"]
    merged = merged.sort_values("lstm_dir_acc", ascending=False)

    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_rows", 50)
    pd.set_option("display.width", 100)

    print()
    print("=" * 65)
    print("  LSTM vs ARIMA -- Directional Accuracy (validation set)")
    print("=" * 65)
    print(
        merged[["ticker", "lstm_dir_acc", "arima_dir_acc", "lstm_wins"]]
        .to_string(index=False)
    )
    print()

    n_wins  = merged["lstm_wins"].sum()
    n_total = len(merged)
    avg_lstm  = merged["lstm_dir_acc"].mean()
    avg_arima = merged["arima_dir_acc"].mean()

    print("=" * 65)
    print(f"  LSTM beats ARIMA : {n_wins}/{n_total} tickers ({100 * n_wins / n_total:.0f}%)")
    print(f"  Avg LSTM  dir acc: {avg_lstm:.4f}")
    print(f"  Avg ARIMA dir acc: {avg_arima:.4f}")
    print(f"  LSTM improvement : +{(avg_lstm - avg_arima):.4f}")
    print("=" * 65)

    if avg_lstm > avg_arima:
        print("  RESULT: LSTM outperforms ARIMA baseline -- Week 2 complete!")
    else:
        print("  NOTE: LSTM underperforms on this run -- consider tuning.")
    print()


if __name__ == "__main__":
    compare_models()