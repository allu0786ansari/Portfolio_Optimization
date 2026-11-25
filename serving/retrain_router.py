"""FastAPI router exposing retraining endpoints (called by the Airflow DAG).

Endpoints (all under the ``/retrain`` prefix, which the DAG expects):
  POST /retrain/fetch-data          -> refresh market data
  POST /retrain/engineer-features   -> recompute features
  POST /retrain/train?algo=ppo|sac  -> train an agent and log it to MLflow
  POST /retrain/evaluate-champion   -> champion/challenger promotion
  GET  /retrain/status              -> is a job running? last result

Phase 0 fixes (v1 defect #7):
* the router is mounted with ``prefix="/retrain"`` (v1 had none, so every DAG call 404'd);
* handlers are plain ``def`` functions, so FastAPI runs them in its thread pool and
  long jobs no longer freeze the event loop (health checks keep answering);
* MLflow is reached through ``MLFLOW_TRACKING_URI`` instead of ``localhost``;
* ``evaluate-champion`` promotes the model version of the best run (v1 promoted
  whatever version number was highest) and errors are returned as HTTP 500.

Phase 6 moves training out of the API process entirely (ADR-03).
"""

import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pydantic import BaseModel

from serving.auth import verify_api_key

MODEL_NAME = "PortfolioAgent"
IMPROVEMENT_THRESHOLD = 0.05

retrain_router = APIRouter(prefix="/retrain", tags=["Retraining"], dependencies=[Depends(verify_api_key)])

_job_lock = threading.Lock()
_state: dict = {"active": None, "last_result": {}}


class RetrainResponse(BaseModel):
    status: str
    message: str
    details: dict = {}


def _acquire(job: str) -> None:
    if not _job_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail=f"Job already in progress: {_state['active']}")
    _state["active"] = job


def _release() -> None:
    _state["active"] = None
    _job_lock.release()


@retrain_router.get("/status")
def retrain_status() -> dict:
    return {"status": "busy" if _state["active"] else "idle", "active_job": _state["active"],
            "last_result": _state["last_result"]}


@retrain_router.post("/fetch-data", response_model=RetrainResponse)
def fetch_data() -> RetrainResponse:
    _acquire("fetch-data")
    try:
        from data.config import ALL_TICKERS, RAW_DIR
        from data.ingestion.fetch_data import fetch_all_tickers, fetch_benchmarks

        cutoff = time.time() - 86400
        removed = 0
        for f in RAW_DIR.glob("*.parquet"):
            if f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        fetched = fetch_all_tickers(ALL_TICKERS)
        fetch_benchmarks()
        return RetrainResponse(status="success", message="Market data refreshed",
                               details={"stale_files_removed": removed, "tickers": len(fetched)})
    except Exception as e:  # surfaced to Airflow as a task failure
        logger.exception("fetch-data failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        _release()


@retrain_router.post("/engineer-features", response_model=RetrainResponse)
def engineer_features() -> RetrainResponse:
    _acquire("engineer-features")
    try:
        from data.ingestion.feature_engineering import engineer_all_features

        results = engineer_all_features()  # overwrites files in place; no destructive pre-delete
        return RetrainResponse(status="success", message=f"Features engineered for {len(results)} tickers",
                               details={"tickers_processed": len(results)})
    except Exception as e:
        logger.exception("engineer-features failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        _release()


@retrain_router.post("/train", response_model=RetrainResponse)
def train_agent(algo: str = "ppo") -> RetrainResponse:
    if algo not in ("ppo", "sac"):
        raise HTTPException(status_code=422, detail="algo must be 'ppo' or 'sac'")
    _acquire(f"train-{algo}")
    try:
        from models.rl_agent.train_agent import train

        summary = train(algo) or {}
        _state["last_result"] = {"job": f"train-{algo}", **summary}
        return RetrainResponse(status="success", message=f"{algo.upper()} agent trained and logged to MLflow",
                               details={"algo": algo, **summary})
    except Exception as e:
        logger.exception("train failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        _release()


@retrain_router.post("/evaluate-champion", response_model=RetrainResponse)
def evaluate_champion() -> RetrainResponse:
    try:
        import mlflow

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        client = mlflow.tracking.MlflowClient()
        try:
            mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
            champ_sh = float(client.get_run(mv.run_id).data.metrics.get("val_sharpe", float("-inf")))
        except Exception:
            mv, champ_sh = None, float("-inf")

        exp = client.get_experiment_by_name("rl_agent_training")
        if exp is None:
            return RetrainResponse(status="skipped", message="No rl_agent_training experiment found")
        runs = client.search_runs([exp.experiment_id], order_by=["metrics.val_sharpe DESC"], max_results=1)
        if not runs:
            return RetrainResponse(status="skipped", message="No training runs found")
        best = runs[0]
        best_sh = float(best.data.metrics.get("val_sharpe", float("-inf")))
        versions = client.search_model_versions(f"name='{MODEL_NAME}' and run_id='{best.info.run_id}'")

        # Improvement must be >= 5% of |champion| (works for negative Sharpe too).
        threshold = champ_sh + IMPROVEMENT_THRESHOLD * abs(champ_sh) if champ_sh != float("-inf") else float("-inf")
        promoted = False
        if versions and best_sh > threshold and (mv is None or versions[0].version != mv.version):
            client.set_registered_model_alias(MODEL_NAME, "champion", versions[0].version)
            promoted = True
            logger.info(f"NEW CHAMPION v{versions[0].version} (val Sharpe {best_sh:.3f} > {champ_sh:.3f})")
        result = {"promoted": promoted, "new_sharpe": best_sh,
                  "champ_sharpe": None if champ_sh == float("-inf") else champ_sh,
                  "best_run_id": best.info.run_id,
                  "version": versions[0].version if versions else None}
        _state["last_result"] = {"job": "evaluate-champion", **result}
        return RetrainResponse(status="success",
                               message="New champion promoted" if promoted else "Current champion retained",
                               details=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("evaluate-champion failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
