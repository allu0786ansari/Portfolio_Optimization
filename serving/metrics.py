"""Prometheus metrics exported from FastAPI.

Metrics:
  portfolio_requests_total     - counter
  portfolio_request_latency_ms - histogram
  portfolio_return_daily       - gauge
  portfolio_rolling_sharpe_30d - gauge (drift signal)
"""
import json
from pathlib import Path
from collections import deque
import numpy as np
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST,
)

REQUEST_COUNT = Counter(
    "portfolio_requests_total",
    "Total prediction requests",
    ["status"],
)
REQUEST_LATENCY = Histogram(
    "portfolio_request_latency_ms",
    "Request latency in milliseconds",
    buckets=[10, 25, 50, 100, 200, 500, 1000],
)
PORTFOLIO_RETURN = Gauge(
    "portfolio_return_daily",
    "Most recent daily portfolio return",
)
ROLLING_SHARPE = Gauge(
    "portfolio_rolling_sharpe_30d",
    "Rolling 30-day annualised Sharpe ratio",
)

_return_buffer: deque = deque(maxlen=30)


def record_request(latency_ms: float, success: bool = True) -> None:
    REQUEST_COUNT.labels(status="success" if success else "error").inc()
    REQUEST_LATENCY.observe(latency_ms)


def record_portfolio_return(daily_return: float) -> None:
    _return_buffer.append(daily_return)
    PORTFOLIO_RETURN.set(daily_return)
    _update_rolling_sharpe()


def update_rolling_sharpe(sharpe: float) -> None:
    ROLLING_SHARPE.set(sharpe)


def _update_rolling_sharpe() -> None:
    if len(_return_buffer) < 5:
        return
    r = np.array(list(_return_buffer))
    std = r.std()
    if std > 1e-10:
        ROLLING_SHARPE.set(float(r.mean() / std * np.sqrt(252)))


def load_sharpe_from_backtest() -> None:
    try:
        path = Path("backtesting/backtest_results.json")
        if path.exists():
            results = json.loads(path.read_text())
            sharpe = results.get("strategies", {}).get("rl", {}).get("sharpe", 0.0)
            ROLLING_SHARPE.set(sharpe)
    except Exception:
        pass


def metrics_endpoint() -> PlainTextResponse:
    return PlainTextResponse(
        generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )