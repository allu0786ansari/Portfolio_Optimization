"""FastAPI application — portfolio weight prediction service (v1 API, Phase 0 fixes).

Endpoints:
  GET  /health    liveness probe (always 200 if the process is alive)
  GET  /ready     readiness probe (200 only when a model can serve)
  GET  /metrics   Prometheus metrics
  POST /predict   portfolio weights (X-API-Key required)
  /retrain/*      retraining endpoints used by Airflow (X-API-Key required)

Phase 0 fixes: request metrics are actually recorded; the API key has no
public default; ``/predict`` honours the ``date`` field; predictions come from
one model path (see ``serving/predictor.py``).

Usage:
  uvicorn serving.main:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from serving.auth import verify_api_key
from serving.metrics import metrics_endpoint, record_request
from serving.model_loader import registry
from serving.predictor import predict_weights, serving_source
from serving.retrain_router import retrain_router
from serving.schemas import HealthResponse, PredictRequest, PredictResponse, ReadyResponse

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Portfolio Optimization API...")
    if not os.getenv("API_KEY"):
        logger.error("API_KEY is not set: protected endpoints will return 503 until it is configured.")

    def load_in_background() -> None:
        try:
            registry.load()
        except Exception as e:  # non-fatal: /ready reports the state
            logger.warning(f"Model load failed at startup: {e}")

    threading.Thread(target=load_in_background, daemon=True).start()
    yield
    logger.info("Shutting down API.")


app = FastAPI(
    title="Portfolio Optimization API",
    description="Serves portfolio weights from the champion RL agent in the MLflow registry.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:8501").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(retrain_router)


@app.middleware("http")
async def log_and_measure(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - t0) * 1000
    path = request.url.path
    if path == "/predict":  # counter + latency histogram (v1 never recorded either)
        record_request(latency_ms, success=response.status_code < 400)
    logger.info(f"{request.method} {path} status={response.status_code} latency={latency_ms:.1f}ms")
    return response


@app.get("/health", response_model=HealthResponse, tags=["Ops"])
async def health():
    return HealthResponse()


@app.get("/metrics", tags=["Ops"], include_in_schema=False)
async def metrics():
    return metrics_endpoint()


@app.get("/ready", response_model=ReadyResponse, tags=["Ops"])
async def ready():
    source = serving_source()
    if source is not None:
        return ReadyResponse(ready=True, model_loaded=True, model_version=source["version"],
                             algo=source["algo"], message=f"Serving from {source['kind']}")
    return JSONResponse(status_code=503, content=ReadyResponse(
        ready=False, model_loaded=False, message="No model available yet").model_dump())


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"],
          dependencies=[Depends(verify_api_key)])
def predict(body: PredictRequest):
    # Plain ``def``: inference runs in the thread pool, not on the event loop.
    if serving_source() is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Try /ready.")
    try:
        result = predict_weights(body.tickers, as_of=body.date)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail="Internal prediction error") from e
    return PredictResponse(**{k: result[k] for k in PredictResponse.model_fields if k in result})
