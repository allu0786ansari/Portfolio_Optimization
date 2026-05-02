"""Compare all strategies: RL agents vs Markowitz vs Equal-Weight.

Fetches RL agent metrics from MLflow, runs classical baselines,
and prints a formatted comparison table.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import mlflow
from loguru import logger

from models.rl_agent.baselines import compute_all_baselines


MLFLOW_URI = "http://localhost:5000"


def fetch_rl_results() -> list[dict]:
    """Pull best run per algorithm from MLflow."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()

    exp = client.get_experiment_by_name("rl_agent_training")
    if exp is None:
        logger.warning("No rl_agent_training experiment found")
        return []

    runs = client.search_runs(
        exp.experiment_id,
        order_by=["metrics.val_sharpe DESC"],
    )

    # Best run per algorithm
    best: dict[str, dict] = {}
    for r in runs:
        algo = r.data.params.get("algo", "unknown").upper()
        if algo not in best:
            best[algo] = {
                "strategy": f"{algo} Agent",
                "sharpe":   r.data.metrics.get("val_sharpe",  0.0),
                "sortino":  r.data.metrics.get("val_sortino", 0.0),
                "cagr":     r.data.metrics.get("val_cagr",    0.0),
                "max_dd":   r.data.metrics.get("val_max_dd",  0.0),
                "is_champion": r.data.params.get("is_champion", "False"),
            }

    return list(best.values())


def print_comparison_table(rows: list[dict]) -> None:
    rows_sorted = sorted(rows, key=lambda x: x.get("sharpe", 0), reverse=True)

    print()
    print("=" * 68)
    print("  Strategy Comparison -- Validation Window")
    print("=" * 68)
    print(f"  {'Strategy':<22} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} {'MaxDD':>8}")
    print("-" * 68)

    for r in rows_sorted:
        champion_tag = " *" if str(r.get("is_champion","")) == "True" else "  "
        print(
            f"  {r['strategy']:<22}{champion_tag}"
            f" {r.get('sharpe',0):>6.3f}"
            f" {r.get('sortino',0):>8.3f}"
            f" {r.get('cagr',0):>6.1%}"
            f" {r.get('max_dd',0):>7.1%}"
        )

    print("=" * 68)

    rl_rows = [r for r in rows_sorted if "Agent" in r["strategy"]]
    baseline_rows = [r for r in rows_sorted if "Agent" not in r["strategy"]]

    if rl_rows and baseline_rows:
        best_rl     = max(rl_rows,      key=lambda x: x.get("sharpe", 0))
        best_base   = max(baseline_rows, key=lambda x: x.get("sharpe", 0))
        diff        = best_rl["sharpe"] - best_base["sharpe"]
        beats       = diff > 0
        print(f"  Best RL:    {best_rl['strategy']} (Sharpe={best_rl['sharpe']:.3f})")
        print(f"  Best Base:  {best_base['strategy']} (Sharpe={best_base['sharpe']:.3f})")
        print(f"  RL vs Base: {'BEATS' if beats else 'UNDERPERFORMS'} baseline by {abs(diff):.3f} Sharpe")
        print(f"  (* = MLflow champion)")
    print()


if __name__ == "__main__":
    logger.info("Fetching RL results from MLflow...")
    rl_results = fetch_rl_results()

    logger.info("Computing classical baselines...")
    baselines = compute_all_baselines()

    all_results = rl_results + baselines
    print_comparison_table(all_results)