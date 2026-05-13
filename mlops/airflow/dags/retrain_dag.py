"""Airflow DAG: nightly portfolio retraining via HTTP calls to FastAPI.

This DAG is intentionally lightweight — it contains NO ML code.
All training happens in the FastAPI training service.
Airflow is a pure orchestrator that fires HTTP requests and
monitors the results.

Architecture:
  Airflow → POST http://api:8000/retrain/* → FastAPI ML Service
                                           → MLflow Registry
                                           → Model hot-reloaded
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

import requests
from airflow import DAG
from airflow.operators.python import PythonOperator

# ── Config ────────────────────────────────────────────────────────
TRAINING_SERVICE_URL = os.getenv("TRAINING_SERVICE_URL", "http://api:8000")
API_KEY              = os.getenv("TRAINING_SERVICE_API_KEY",
                                  "portfolio-secret-key-change-in-production")
HEADERS              = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
TIMEOUT_SHORT        = 60      # seconds — for data/feature tasks
TIMEOUT_TRAIN        = 7200    # seconds — 2 hours for training tasks

DEFAULT_ARGS = {
    "owner":            "ai-engineer",
    "depends_on_past":  False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}

IMPROVEMENT_THRESHOLD = 0.05


# ── Helper ────────────────────────────────────────────────────────

def call_training_service(
    endpoint: str,
    params:   dict | None = None,
    timeout:  int = TIMEOUT_SHORT,
) -> dict:
    """POST to the FastAPI training service and return the JSON response."""
    url = f"{TRAINING_SERVICE_URL}/retrain/{endpoint}"
    from loguru import logger
    logger.info(f"Calling training service: POST {url}")

    try:
        response = requests.post(url, headers=HEADERS, params=params, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Response from {endpoint}: {result}")
        return result
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Training service timed out on {endpoint}")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Training service error on {endpoint}: {e} — {response.text}")
    except Exception as e:
        raise RuntimeError(f"Failed to call {endpoint}: {e}")


# ── Task functions ────────────────────────────────────────────────

def task_fetch_data(**context):
    result = call_training_service("fetch-data", timeout=TIMEOUT_SHORT)
    context["ti"].xcom_push(key="fetch_result", value=result)


def task_engineer_features(**context):
    result = call_training_service("engineer-features", timeout=TIMEOUT_SHORT)
    context["ti"].xcom_push(key="engineer_result", value=result)


def task_train_ppo(**context):
    result = call_training_service(
        "train", params={"algo": "ppo"}, timeout=TIMEOUT_TRAIN
    )
    context["ti"].xcom_push(key="ppo_result", value=result)


def task_train_sac(**context):
    result = call_training_service(
        "train", params={"algo": "sac"}, timeout=TIMEOUT_TRAIN
    )
    context["ti"].xcom_push(key="sac_result", value=result)


def task_evaluate_champion(**context):
    result = call_training_service("evaluate-champion", timeout=TIMEOUT_SHORT)
    context["ti"].xcom_push(key="evaluate_result", value=result)


def task_notify_result(**context):
    from loguru import logger
    ti = context["ti"]
    evaluate_result = ti.xcom_pull(
        key="evaluate_result", task_ids="evaluate_champion"
    ) or {}

    promoted   = evaluate_result.get("details", {}).get("promoted",     False)
    new_sharpe = evaluate_result.get("details", {}).get("new_sharpe",   0.0)
    champ_sh   = evaluate_result.get("details", {}).get("champ_sharpe", 0.0)

    logger.info("=" * 55)
    logger.info(f"DAG COMPLETE — {'CHAMPION PROMOTED' if promoted else 'NO CHANGE'}")
    logger.info(f"  New model Sharpe:     {new_sharpe:.3f}")
    logger.info(f"  Previous champion:    {champ_sh:.3f}")
    logger.info(f"  Champion updated:     {promoted}")
    logger.info(f"  Full result:          {evaluate_result}")
    logger.info("=" * 55)


# ── DAG definition ────────────────────────────────────────────────

with DAG(
    dag_id          = "retrain_portfolio_agent",
    default_args    = DEFAULT_ARGS,
    description     = "Nightly retraining — Airflow calls FastAPI training service",
    schedule        = "30 12 * * 1-5",   # 18:00 IST Mon-Fri
    start_date      = datetime(2024, 1, 1),
    catchup         = False,
    tags            = ["portfolio", "mlops", "retraining"],
    max_active_runs = 1,
) as dag:

    fetch = PythonOperator(
        task_id         = "fetch_new_data",
        python_callable = task_fetch_data,
    )
    engineer = PythonOperator(
        task_id         = "engineer_features",
        python_callable = task_engineer_features,
    )
    train_ppo = PythonOperator(
        task_id           = "train_ppo",
        python_callable   = task_train_ppo,
        execution_timeout = timedelta(hours=2),
    )
    train_sac = PythonOperator(
        task_id           = "train_sac",
        python_callable   = task_train_sac,
        execution_timeout = timedelta(hours=2),
    )
    evaluate = PythonOperator(
        task_id         = "evaluate_champion",
        python_callable = task_evaluate_champion,
    )
    notify = PythonOperator(
        task_id         = "notify_result",
        python_callable = task_notify_result,
    )

    # PPO and SAC train in parallel after feature engineering
    fetch >> engineer >> [train_ppo, train_sac] >> evaluate >> notify