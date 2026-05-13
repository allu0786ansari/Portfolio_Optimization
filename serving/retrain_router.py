"""FastAPI router exposing retraining endpoints.

Airflow DAG calls these endpoints via HTTP — keeping Airflow
lightweight (no ML dependencies needed in Airflow container).

Endpoints:
  POST /retrain/fetch-data          → refresh market data
  POST /retrain/engineer-features   → recompute features
  POST /retrain/train               → train PPO or SAC agent
  POST /retrain/evaluate-champion   → champion-challenger promotion
  GET  /retrain/status              → check if training is running
"""
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pydantic import BaseModel

from serving.schemas import HealthResponse

# Reuse the same API key auth from main.py
import os
from fastapi import Security
from fastapi.security import APIKeyHeader

API_KEY        = os.getenv("API_KEY", "portfolio-secret-key-change-in-production")
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_key(key: str = Security(API_KEY_HEADER)) -> str:
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return key

retrain_router = APIRouter()

# ── Training state ────────────────────────────────────────────────
_training_lock   = threading.Lock()
_training_active = False
_last_result     : dict = {}


def _is_training() -> bool:
    with _training_lock:
        return _training_active


def _set_training(state: bool) -> None:
    global _training_active
    with _training_lock:
        _training_active = state


# ── Response models ───────────────────────────────────────────────

class RetrainResponse(BaseModel):
    status:  str
    message: str
    details: dict = {}


# ── Endpoints ─────────────────────────────────────────────────────

@retrain_router.get("/status")
async def retrain_status(_: str = Depends(verify_key)):
    """Check if a training job is currently running."""
    return {
        "status":              "busy" if _is_training() else "idle",
        "training_in_progress": _is_training(),
        "last_result":         _last_result,
    }


@retrain_router.post("/fetch-data")
async def fetch_data(_: str = Depends(verify_key)):
    """Pull latest market data and refresh raw Parquet files."""
    if _is_training():
        raise HTTPException(status_code=409, detail="Training already in progress")
    try:
        logger.info("Retrain: fetching fresh market data...")
        from data.ingestion.fetch_data import fetch_all_tickers, fetch_benchmarks
        from data.config import ALL_TICKERS, RAW_DIR
        import time

        # Remove stale files (older than 1 day)
        cutoff = time.time() - 86400
        removed = 0
        for f in RAW_DIR.glob("*.parquet"):
            if f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1

        fetch_all_tickers(ALL_TICKERS)
        fetch_benchmarks()
        logger.info(f"Retrain: data fetch complete (removed {removed} stale files)")
        return RetrainResponse(
            status="success",
            message="Market data refreshed",
            details={"stale_files_removed": removed},
        )
    except Exception as e:
        logger.error(f"Retrain fetch-data failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@retrain_router.post("/engineer-features")
async def engineer_features(_: str = Depends(verify_key)):
    """Recompute all features from fresh raw data."""
    if _is_training():
        raise HTTPException(status_code=409, detail="Training already in progress")
    try:
        logger.info("Retrain: engineering features...")
        from data.ingestion.feature_engineering import engineer_all_features
        from data.config import PROCESSED_DIR

        # Clear stale processed files
        removed = sum(1 for f in PROCESSED_DIR.glob("features_*.parquet")
                      if f.unlink() or True)
        results = engineer_all_features()
        logger.info(f"Retrain: features engineered for {len(results)} tickers")
        return RetrainResponse(
            status="success",
            message=f"Features engineered for {len(results)} tickers",
            details={"tickers_processed": len(results)},
        )
    except Exception as e:
        logger.error(f"Retrain engineer-features failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@retrain_router.post("/train")
async def train_agent(algo: str = "ppo", _: str = Depends(verify_key)):
    """Train a PPO or SAC agent and log to MLflow."""
    if algo not in ("ppo", "sac"):
        raise HTTPException(status_code=422, detail="algo must be 'ppo' or 'sac'")
    if _is_training():
        raise HTTPException(status_code=409, detail="Training already in progress")

    _set_training(True)
    try:
        logger.info(f"Retrain: training {algo.upper()} agent...")
        from models.rl_agent.train_agent import train
        train(algo)
        logger.info(f"Retrain: {algo.upper()} training complete")
        return RetrainResponse(
            status="success",
            message=f"{algo.upper()} agent trained and logged to MLflow",
            details={"algo": algo},
        )
    except Exception as e:
        logger.error(f"Retrain train failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _set_training(False)


@retrain_router.post("/evaluate-champion")
async def evaluate_champion(_: str = Depends(verify_key)):
    """Compare latest runs vs current champion. Promote if better by 5%."""
    global _last_result
    try:
        logger.info("Retrain: evaluating champion...")
        import mlflow

        IMPROVEMENT_THRESHOLD = 0.05
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        )
        client = mlflow.tracking.MlflowClient()

        # Get current champion Sharpe
        try:
            mv      = client.get_model_version_by_alias("PortfolioAgent", "champion")
            run     = client.get_run(mv.run_id)
            champ_sh = float(run.data.metrics.get("val_sharpe", 0.0))
        except Exception:
            champ_sh = 0.0

        # Find best new run
        exp  = client.get_experiment_by_name("rl_agent_training")
        if exp is None:
            return RetrainResponse(
                status="skipped",
                message="No rl_agent_training experiment found",
            )

        runs = client.search_runs(
            exp.experiment_id,
            order_by=["metrics.val_sharpe DESC"],
            max_results=5,
        )

        best_sh, best_run = 0.0, None
        for r in runs:
            sh = float(r.data.metrics.get("val_sharpe", 0.0))
            if sh > best_sh:
                best_sh, best_run = sh, r

        promoted = False
        if best_run and best_sh > champ_sh * (1 + IMPROVEMENT_THRESHOLD):
            versions = client.search_model_versions("name='PortfolioAgent'")
            latest   = sorted(versions, key=lambda v: int(v.version))[-1]
            client.set_registered_model_alias("PortfolioAgent", "champion", latest.version)
            promoted = True
            logger.info(f"NEW CHAMPION v{latest.version} Sharpe {best_sh:.3f}")

        _last_result = {
            "promoted":     promoted,
            "new_sharpe":   best_sh,
            "champ_sharpe": champ_sh,
        }

        return RetrainResponse(
            status="success",
            message="New champion promoted" if promoted else "Current champion retained",
            details=_last_result,
        )
    except Exception as e:
        logger.error(f"Retrain evaluate-champion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))