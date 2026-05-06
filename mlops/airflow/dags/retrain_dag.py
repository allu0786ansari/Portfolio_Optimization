"""Airflow DAG: nightly portfolio agent retraining.

Schedule: 18:00 IST Mon-Fri (12:30 UTC)
Tasks: fetch_new_data -> engineer_features -> train_models
       -> evaluate_champion -> notify_result
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

DEFAULT_ARGS = {
    "owner": "ai-engineer",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}
IMPROVEMENT_THRESHOLD = 0.05


def task_fetch_data(**context):
    import sys; sys.path.insert(0, ".")
    import time

    from loguru import logger

    from data.config import ALL_TICKERS, RAW_DIR
    from data.ingestion.fetch_data import fetch_all_tickers, fetch_benchmarks
    cutoff = time.time() - 86400
    for f in RAW_DIR.glob("*.parquet"):
        if f.stat().st_mtime < cutoff:
            f.unlink()
    fetch_all_tickers(ALL_TICKERS)
    fetch_benchmarks()
    logger.info("DAG: Data fetch complete")


def task_engineer_features(**context):
    import sys; sys.path.insert(0, ".")
    from loguru import logger

    from data.config import PROCESSED_DIR
    from data.ingestion.feature_engineering import engineer_all_features
    for f in PROCESSED_DIR.glob("features_*.parquet"):
        f.unlink()
    engineer_all_features()
    logger.info("DAG: Feature engineering complete")


def task_train_models(**context):
    import sys; sys.path.insert(0, ".")
    from loguru import logger

    from models.rl_agent.train_agent import train
    logger.info("DAG: Training PPO...")
    train("ppo")
    logger.info("DAG: Training SAC...")
    train("sac")


def task_evaluate_champion(**context):
    import sys; sys.path.insert(0, ".")
    import mlflow
    from loguru import logger
    mlflow.set_tracking_uri("http://localhost:5000")
    client = mlflow.tracking.MlflowClient()
    try:
        mv = client.get_model_version_by_alias("PortfolioAgent", "champion")
        run = client.get_run(mv.run_id)
        champ_sh = float(run.data.metrics.get("val_sharpe", 0.0))
    except Exception:
        champ_sh = 0.0
    exp = client.get_experiment_by_name("rl_agent_training")
    runs = client.search_runs(exp.experiment_id,
        order_by=["metrics.val_sharpe DESC"], max_results=5)
    best_run, best_sh = None, 0.0
    for r in runs:
        sh = float(r.data.metrics.get("val_sharpe", 0.0))
        if sh > best_sh:
            best_sh, best_run = sh, r
    promoted = False
    if best_run and best_sh > champ_sh * (1 + IMPROVEMENT_THRESHOLD):
        versions = client.search_model_versions("name='PortfolioAgent'")
        latest = sorted(versions, key=lambda v: int(v.version))[-1]
        client.set_registered_model_alias("PortfolioAgent", "champion", latest.version)
        promoted = True
        logger.info(f"NEW CHAMPION v{latest.version} Sharpe {best_sh:.3f}")
    else:
        logger.info(f"No promotion: {best_sh:.3f} vs threshold {champ_sh*1.05:.3f}")
    context["ti"].xcom_push(key="promoted", value=promoted)
    context["ti"].xcom_push(key="new_sharpe", value=best_sh)
    context["ti"].xcom_push(key="champ_sharpe", value=champ_sh)


def task_notify_result(**context):
    from loguru import logger
    ti = context["ti"]
    promoted = ti.xcom_pull(key="promoted", task_ids="evaluate_champion")
    new_sh = ti.xcom_pull(key="new_sharpe", task_ids="evaluate_champion")
    champ_sh = ti.xcom_pull(key="champ_sharpe", task_ids="evaluate_champion")
    logger.info("=" * 50)
    logger.info(f"DAG COMPLETE — {'PROMOTED' if promoted else 'NO CHANGE'}")
    logger.info(f"  New Sharpe:    {new_sh:.3f}")
    logger.info(f"  Champ Sharpe:  {champ_sh:.3f}")
    logger.info("=" * 50)


with DAG(
    dag_id="retrain_portfolio_agent",
    default_args=DEFAULT_ARGS,
    description="Nightly retraining for portfolio RL agent",
    schedule="30 12 * * 1-5",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["portfolio", "mlops"],
    max_active_runs=1,
) as dag:
    fetch    = PythonOperator(task_id="fetch_new_data",     python_callable=task_fetch_data)
    engineer = PythonOperator(task_id="engineer_features",  python_callable=task_engineer_features)
    train    = PythonOperator(task_id="train_models",       python_callable=task_train_models, execution_timeout=timedelta(hours=2))
    evaluate = PythonOperator(task_id="evaluate_champion",  python_callable=task_evaluate_champion)
    notify   = PythonOperator(task_id="notify_result",      python_callable=task_notify_result)
    fetch >> engineer >> train >> evaluate >> notify