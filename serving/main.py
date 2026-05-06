"""FastAPI application — portfolio weight prediction service.

Endpoints:
  GET  /health   — liveness probe (always 200 if process is alive)
  GET  /ready    — readiness probe (200 only when model is loaded)
  POST /predict  — portfolio weight prediction

Authentication:
  All non-health endpoints require X-API-Key header.
  Key is read from .env API_KEY variable.

Usage:
  uvicorn serving.main:app --host 0.0.0.0 --port 8000 --reload
"""
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from loguru import logger

from serving.metrics import load_sharpe_from_backtest, metrics_endpoint
from serving.model_loader import registry
from serving.predictor import predict_weights
from serving.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
)

load_dotenv()

API_KEY        = os.getenv("API_KEY", "portfolio-secret-key-change-in-production")
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


# ── Lifespan: load model at startup ─────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Portfolio Optimization API...")
    registry.load()
    load_sharpe_from_backtest()
    logger.info("API ready.")
    yield
    logger.info("Shutting down API.")


# ── App ──────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Portfolio Optimization API",
    description = (
        "Serves portfolio weight predictions from a trained RL agent. "
        "The champion model is loaded from MLflow Model Registry and "
        "hot-reloaded automatically when a new champion is promoted."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["GET", "POST"],
    allow_headers  = ["*"],
)


# ── Middleware: request logging ──────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0       = time.perf_counter()
    response = await call_next(request)
    latency  = (time.perf_counter() - t0) * 1000
    logger.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} "
        f"latency={latency:.1f}ms"
    )
    return response


# ── Auth dependency ──────────────────────────────────────────────

async def verify_api_key(key: str = Security(API_KEY_HEADER)) -> str:
    if key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Pass X-API-Key header.",
        )
    return key


# ── Endpoints ────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Ops"],
    summary="Liveness probe — confirms the process is alive",
)
async def health():
    return HealthResponse()


@app.get("/metrics", tags=["Ops"], include_in_schema=False)
async def metrics():
    return metrics_endpoint()


@app.get(
    "/ready",
    response_model=ReadyResponse,
    tags=["Ops"],
    summary="Readiness probe — confirms model is loaded and ready for inference",
)
async def ready():
    if registry.is_loaded:
        return ReadyResponse(
            ready         = True,
            model_loaded  = True,
            model_version = registry.version,
            algo          = registry.algo,
            message       = "Model loaded and ready",
        )
    return JSONResponse(
        status_code = 503,
        content     = ReadyResponse(
            ready        = False,
            model_loaded = False,
            message      = "Model not yet loaded",
        ).model_dump(),
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Prediction"],
    summary="Return portfolio weights for a list of tickers",
    dependencies=[Depends(verify_api_key)],
)
async def predict(body: PredictRequest):
    if not registry.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Try /ready.")

    try:
        result = predict_weights(body.tickers)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error")

    return PredictResponse(
        weights       = result["weights"],
        weights_sum   = result["weights_sum"],
        model_version = result["model_version"],
        algo          = result["algo"],
        latency_ms    = result["latency_ms"],
        n_assets      = result["n_assets"],
        timestamp     = result["timestamp"],
    )