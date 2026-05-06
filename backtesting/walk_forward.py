"""Walk-forward backtesting engine.

Methodology:
  - Train window : 504 days (2 years)
  - Test window  : 126 days (6 months)
  - Step size    : 21  days (1 month forward)

For each window:
  1. Load the champion RL agent from MLflow
  2. Run the agent deterministically on the test window
  3. Simultaneously compute Markowitz and equal-weight on same window
  4. Record all metrics per window

Final output:
  - Aggregated metrics across all windows
  - JSON file for report generation
  - Console summary table
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
import mlflow.pytorch
import numpy as np
from loguru import logger

from backtesting.metrics import (
    compute_all_metrics,
    drawdown_series,
    equity_curve,
    rolling_sharpe,
    ttest_excess_returns,
)
from data.config import ALL_TICKERS
from models.classical.markowitz import MarkowitzOptimiser
from models.rl_agent.data_loader import build_return_matrix, load_aligned_features
from models.rl_agent.portfolio_env import softmax

MLFLOW_URI   = "http://localhost:5000"
MODEL_NAME   = "PortfolioAgent"
TRAIN_DAYS   = 504    # 2 years
TEST_DAYS    = 126    # 6 months
STEP_DAYS    = 21     # 1 month
OUTPUT_PATH  = Path("backtesting/backtest_results.json")


def load_champion_model():
    """Load champion RL agent from MLflow registry."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    client  = mlflow.tracking.MlflowClient()
    try:
        mv      = client.get_model_version_by_alias(MODEL_NAME, "champion")
        run     = client.get_run(mv.run_id)
        algo    = run.data.params.get("algo", "ppo").lower()
        uri     = f"models:/{MODEL_NAME}@champion"
        policy  = mlflow.pytorch.load_model(uri)
        logger.info(f"Loaded champion: {algo.upper()} v{mv.version}")
        return policy, algo, mv.version
    except Exception as e:
        logger.error(f"Could not load champion from MLflow: {e}")
        raise


def get_action_from_policy(policy, obs_tensor) -> np.ndarray:
    """Get action from policy — handles both PPO and SAC."""
    import torch

    with torch.no_grad():
        if hasattr(policy, "actor"):
            # SAC
            action = policy.actor(obs_tensor).squeeze(0).numpy()
        else:
            # PPO
            action_tensor = policy._predict(obs_tensor, deterministic=True)
            action = action_tensor.squeeze(0).numpy()

    return action


def run_rl_on_window(
    policy,
    feature_matrix: np.ndarray,
    return_matrix: np.ndarray,
    start_idx: int,
    end_idx: int,
    n_assets: int,
) -> np.ndarray:
    """Run RL policy deterministically on a test window."""
    import torch

    weights = np.ones(n_assets, dtype=np.float32) / n_assets
    returns = []

    for t in range(start_idx, end_idx):
        features_flat = feature_matrix[t].flatten().astype(np.float32)
        features_flat = np.clip(features_flat, -10.0, 10.0)
        obs = np.concatenate([features_flat, weights])
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

        action = get_action_from_policy(policy, obs_tensor)

        weights  = softmax(action)
        port_ret = float(np.dot(weights, return_matrix[t]))
        returns.append(port_ret)

    return np.array(returns)


def run_walk_forward() -> dict:
    mlflow.set_tracking_uri(MLFLOW_URI)

    # Load data
    features_dict, tickers, dates = load_aligned_features(ALL_TICKERS)
    return_matrix = build_return_matrix(features_dict, tickers)
    n_assets      = len(tickers)

    from models.forecasting.dataset import FEATURE_COLS
    feature_matrix = np.stack(
        [features_dict[t][FEATURE_COLS].values for t in tickers], axis=1
    ).astype(np.float32)

    T = len(dates)
    logger.info(f"Total days: {T} | Tickers: {n_assets}")

    policy, algo, version = load_champion_model()

    windows = []
    idx = TRAIN_DAYS
    while idx + TEST_DAYS <= T:
        windows.append((idx - TRAIN_DAYS, idx, idx, idx + TEST_DAYS))
        idx += STEP_DAYS

    logger.info(f"Running {len(windows)} walk-forward windows...")

    markowitz = MarkowitzOptimiser(lookback_days=TRAIN_DAYS)

    all_rl_returns  = []
    all_mvo_returns = []
    all_ew_returns  = []
    all_dates       = []   # ✅ FIX

    window_results = []

    for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
        logger.info(
            f"Window {i+1:2d}/{len(windows)} | "
            f"train: {dates[train_start].date()} -> {dates[train_end-1].date()} | "
            f"test:  {dates[test_start].date()} -> {dates[test_end-1].date()}"
        )

        rl_ret = run_rl_on_window(
            policy, feature_matrix, return_matrix,
            test_start, test_end, n_assets,
        )

        train_returns_window = return_matrix[train_start:train_end]
        mvo_weights = markowitz.compute_weights(train_returns_window)
        mvo_ret = np.array([
            float(np.dot(mvo_weights, return_matrix[t]))
            for t in range(test_start, test_end)
        ])

        ew_weights = np.ones(n_assets) / n_assets
        ew_ret = np.array([
            float(np.dot(ew_weights, return_matrix[t]))
            for t in range(test_start, test_end)
        ])

        # ✅ FIX: track correct dates
        all_dates.extend([str(dates[t].date()) for t in range(test_start, test_end)])

        all_rl_returns.extend(rl_ret.tolist())
        all_mvo_returns.extend(mvo_ret.tolist())
        all_ew_returns.extend(ew_ret.tolist())

        window_results.append({
            "window":      i + 1,
            "test_start":  str(dates[test_start].date()),
            "test_end":    str(dates[test_end-1].date()),
            "rl_sharpe":   float(compute_all_metrics(rl_ret)["sharpe"]),
            "mvo_sharpe":  float(compute_all_metrics(mvo_ret)["sharpe"]),
            "ew_sharpe":   float(compute_all_metrics(ew_ret)["sharpe"]),
        })

    rl_arr  = np.array(all_rl_returns)
    mvo_arr = np.array(all_mvo_returns)
    ew_arr  = np.array(all_ew_returns)

    rl_metrics  = compute_all_metrics(rl_arr,  label=f"RL-{algo.upper()}")
    mvo_metrics = compute_all_metrics(mvo_arr, label="Markowitz (MVO)")
    ew_metrics  = compute_all_metrics(ew_arr,  label="Equal-Weight")

    t_stat, p_value = ttest_excess_returns(rl_arr, ew_arr)
    rl_metrics["t_stat_vs_ew"]  = t_stat
    rl_metrics["p_value_vs_ew"] = p_value

    results = {
        "algo":     algo,
        "version":  str(version),
        "n_windows": len(windows),
        "n_days":   len(rl_arr),
        "strategies": {
            "rl":           rl_metrics,
            "markowitz":    mvo_metrics,
            "equal_weight": ew_metrics,
        },
        "windows": window_results,
        "equity_curves": {
            "rl":           equity_curve(rl_arr).tolist(),
            "markowitz":    equity_curve(mvo_arr).tolist(),
            "equal_weight": equity_curve(ew_arr).tolist(),
        },
        "rolling_sharpe": {
            "rl":           rolling_sharpe(rl_arr).tolist(),
            "equal_weight": rolling_sharpe(ew_arr).tolist(),
        },
        "drawdowns": {
            "rl":           drawdown_series(rl_arr).tolist(),
            "equal_weight": drawdown_series(ew_arr).tolist(),
        },
        "dates": all_dates,   # ✅ FIXED
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    logger.info(f"Results saved to {OUTPUT_PATH}")

    _print_summary(rl_metrics, mvo_metrics, ew_metrics)
    return results


def _print_summary(rl: dict, mvo: dict, ew: dict) -> None:
    print()
    print("=" * 68)
    print("  FULL WALK-FORWARD BACKTEST RESULTS")
    print("=" * 68)
    print(f"  {'Metric':<22} {rl['label']:>12} {mvo['label']:>14} {ew['label']:>13}")
    print("-" * 68)
    rows = [
        ("Sharpe Ratio",  "sharpe",      "{:.3f}"),
        ("Sortino Ratio", "sortino",     "{:.3f}"),
        ("Calmar Ratio",  "calmar",      "{:.3f}"),
        ("CAGR",          "cagr",        "{:.1%}"),
        ("Max Drawdown",  "max_drawdown","{:.1%}"),
        ("VaR 95%",       "var_95",      "{:.2%}"),
        ("CVaR 95%",      "cvar_95",     "{:.2%}"),
    ]
    for label, key, fmt in rows:
        print(
            f"  {label:<22}"
            f" {fmt.format(rl.get(key, 0)):>12}"
            f" {fmt.format(mvo.get(key, 0)):>14}"
            f" {fmt.format(ew.get(key, 0)):>13}"
        )
    print("-" * 68)
    t   = rl.get("t_stat_vs_ew", 0)
    p   = rl.get("p_value_vs_ew", 1)
    sig = "SIGNIFICANT (p<0.05)" if p < 0.05 else "not significant"
    print(f"  {'t-stat vs EW':<22} {t:>12.3f}")
    print(f"  {'p-value':<22} {p:>12.4f}  <- {sig}")
    print("=" * 68)
    print()


if __name__ == "__main__":
    logger.info("=== Starting walk-forward backtest ===")
    run_walk_forward()
    logger.info("=== Backtest complete ===")