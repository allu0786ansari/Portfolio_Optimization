"""Pydantic request and response schemas for the portfolio API.

All inputs are validated automatically by FastAPI before
any ML code runs. Invalid requests return HTTP 422 with
a clear error message describing exactly what's wrong.
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Request body for POST /predict."""

    tickers: list[str] = Field(
        ...,
        min_length=2,
        max_length=50,
        description="List of asset tickers to allocate across. Min 2, max 50.",
        examples=[["RELIANCE.NS", "TCS.NS", "AAPL", "MSFT"]],
    )
    date: Optional[str] = Field(
        default=None,
        description="Reference date for feature lookup (YYYY-MM-DD). Defaults to latest available.",
        examples=["2024-12-31"],
    )

    @field_validator("tickers")
    @classmethod
    def tickers_non_empty_strings(cls, v: list[str]) -> list[str]:
        for t in v:
            if not t.strip():
                raise ValueError("Ticker symbols cannot be empty strings")
        return [t.strip().upper() for t in v]


class PredictResponse(BaseModel):
    """Response body for POST /predict."""

    weights:        dict[str, float] = Field(..., description="Portfolio weights per ticker. Sum = 1.0")
    weights_sum:    float            = Field(..., description="Sum of all weights. Should be 1.0")
    model_version:  str              = Field(..., description="MLflow model version identifier")
    algo:           str              = Field(..., description="Algorithm used: PPO or SAC")
    latency_ms:     float            = Field(..., description="End-to-end prediction latency in milliseconds")
    n_assets:       int              = Field(..., description="Number of assets in the allocation")
    timestamp:      str              = Field(..., description="UTC timestamp of prediction")


class HealthResponse(BaseModel):
    """Response for GET /health."""
    status:  str = "ok"
    version: str = "1.0.0"


class ReadyResponse(BaseModel):
    """Response for GET /ready."""
    ready:          bool
    model_loaded:   bool
    model_version:  Optional[str] = None
    algo:           Optional[str] = None
    message:        str